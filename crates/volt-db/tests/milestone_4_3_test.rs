//! Integration tests for Milestone 4.3: T2 + GC + WAL + Consolidation.
//!
//! Verifies the acceptance criteria from `roadmap/PHASE-4.md`:
//!
//! 1. GC decay pipeline: Full → Compressed → Gist → Tombstone
//! 2. Immortal frames: pinned + high-gamma survive GC
//! 3. WAL crash recovery: frames recovered after unclean shutdown
//! 4. WAL corrupt tail: corrupt entries skipped on replay
//! 5. Concurrent access: 10 readers + 1 writer, no panics
//! 6. T2 sorted runs: flush, compact, retrieve from mmap'd runs
//! 7. Consolidation: similar frames → wisdom frame
//! 8. Bloom filter: prevents unnecessary sorted run reads
//! 9. Frame entry roundtrip: all decay levels serialize/deserialize correctly

use std::path::PathBuf;

use volt_core::{SlotData, SlotRole, TensorFrame, SLOT_DIM};
use volt_db::compressed::{compress, to_gist_frame, to_tombstone, DecayLevel, FrameEntry};
use volt_db::gc::{GcConfig, GcEngine};
use volt_db::wal::{WalEntry, WalManager, WalOp};
use volt_db::{
    BloomFilter, ConcurrentVoltStore, Tier2Store, VoltStore, VoltStoreConfig,
};
use volt_db::tier2::T2Config;

/// Helper: create a frame with R₀ data and a specific certainty.
fn make_frame(gamma: f32, value: f32) -> TensorFrame {
    let mut frame = TensorFrame::new();
    let mut slot = SlotData::new(SlotRole::Agent);
    slot.write_resolution(0, [value; SLOT_DIM]);
    frame.write_slot(0, slot).unwrap();
    frame.frame_meta.global_certainty = gamma;
    frame
}

/// Helper: temp directory scoped to test name + PID.
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("volt_m43_test")
        .join(name)
        .join(format!("{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------------------
// Test 1: GC decay pipeline
// ---------------------------------------------------------------------------

#[test]
fn gc_decay_pipeline() {
    // Run on a thread with large stack (TensorFrame is 64KB, serde uses
    // recursive descent that can overflow on Windows' 1MB default).
    std::thread::Builder::new()
        .name("gc_decay_pipeline".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let dir = temp_dir("gc_decay");
            let config = VoltStoreConfig {
                data_dir: dir.clone(),
                t1_overflow_threshold: 2000, // high threshold to keep frames in T1
                t2_config: T2Config {
                    data_dir: dir.join("t2"),
                    ..T2Config::default()
                },
                gc_config: GcConfig {
                    // Aggressive thresholds for testing
                    threshold_full_to_compressed: 0.99, // almost everything gets compressed
                    threshold_compressed_to_gist: 0.5,
                    threshold_gist_to_tombstone: 0.2,
                    ..GcConfig::default()
                },
                ..VoltStoreConfig::default()
            };

            let mut store = VoltStore::open(config).unwrap();

            // Store > 64 frames (T0_CAPACITY) so some overflow to T1
            // Use low gamma and old timestamps → should be demoted by GC
            for _ in 0..80 {
                let mut frame = make_frame(0.1, 0.5); // low gamma
                frame.frame_meta.created_at = 1_000_000; // old timestamp
                store.store(frame).unwrap();
            }

            // Run GC with a "now" far in the future (makes frames very old)
            let far_future = 100_000_000_000_000u64; // ~100 seconds in micros
            let result = store.run_gc_at(far_future).unwrap();

            // With aggressive thresholds and old+low-gamma frames, some should be demoted
            let total_demoted = result.frames_compressed
                + result.frames_gisted
                + result.frames_tombstoned;
            assert!(
                total_demoted > 0,
                "expected some frames to be demoted, got result: {result:?}"
            );

            let _ = std::fs::remove_dir_all(&dir);
        })
        .unwrap()
        .join()
        .unwrap();
}

// ---------------------------------------------------------------------------
// Test 2: Immortal frames (pinned + high gamma)
// ---------------------------------------------------------------------------

#[test]
fn immortal_frames_survive_gc() {
    std::thread::Builder::new()
        .name("immortal_frames".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let dir = temp_dir("immortal");
            let config = VoltStoreConfig {
                data_dir: dir.clone(),
                t1_overflow_threshold: 2000,
                t2_config: T2Config {
                    data_dir: dir.join("t2"),
                    ..T2Config::default()
                },
                gc_config: GcConfig {
                    threshold_full_to_compressed: 0.99,
                    threshold_compressed_to_gist: 0.5,
                    threshold_gist_to_tombstone: 0.2,
                    ..GcConfig::default()
                },
                ..VoltStoreConfig::default()
            };

            let mut store = VoltStore::open(config).unwrap();

            // Store a pinned frame (low gamma, old)
            let mut pinned_frame = make_frame(0.1, 0.3);
            pinned_frame.frame_meta.created_at = 1_000_000;
            let pinned_id = store.store(pinned_frame).unwrap();
            store.pin_frame(pinned_id);

            // Store a high-gamma frame (gamma >= 1.0 → immortal)
            let mut high_gamma = make_frame(1.0, 0.4);
            high_gamma.frame_meta.created_at = 1_000_000;
            let high_gamma_id = store.store(high_gamma).unwrap();

            // Run GC with far future
            let far_future = 100_000_000_000_000u64;
            let _result = store.run_gc_at(far_future).unwrap();

            // Both frames should still be accessible in T0/T1
            assert!(
                store.get_by_id(pinned_id).is_some(),
                "pinned frame should survive GC"
            );
            assert!(
                store.get_by_id(high_gamma_id).is_some(),
                "high-gamma frame should survive GC"
            );

            let _ = std::fs::remove_dir_all(&dir);
        })
        .unwrap()
        .join()
        .unwrap();
}

// ---------------------------------------------------------------------------
// Test 3: WAL crash recovery
// ---------------------------------------------------------------------------

#[test]
fn wal_crash_recovery() {
    std::thread::Builder::new()
        .name("wal_recovery".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let dir = temp_dir("wal_recovery");
            let wal_dir = dir.join("wal");

            // Phase 1: write WAL entries directly (simulating a crash before T1 save)
            {
                let mut wal = WalManager::open(&wal_dir).unwrap();
                for i in 1..=5u64 {
                    let mut frame = make_frame(0.5, 0.1 * i as f32);
                    frame.frame_meta.frame_id = i;
                    frame.frame_meta.strand_id = 0;
                    let entry_data =
                        FrameEntry::Full(Box::new(frame)).to_bytes().unwrap();
                    wal.log_entry(WalEntry {
                        frame_id: i,
                        strand_id: 0,
                        op: WalOp::Store,
                        payload: entry_data,
                    })
                    .unwrap();
                }
                wal.sync_all().unwrap();
            }

            // Phase 2: open VoltStore — WAL should be replayed
            let config = VoltStoreConfig {
                data_dir: dir.clone(),
                t1_overflow_threshold: 2000,
                t2_config: T2Config {
                    data_dir: dir.join("t2"),
                    ..T2Config::default()
                },
                ..VoltStoreConfig::default()
            };

            let store = VoltStore::open(config).unwrap();

            // All 5 frames should be recovered
            for i in 1..=5u64 {
                assert!(
                    store.get_by_id(i).is_some(),
                    "frame {i} should be recovered from WAL"
                );
            }

            let _ = std::fs::remove_dir_all(&dir);
        })
        .unwrap()
        .join()
        .unwrap();
}

// ---------------------------------------------------------------------------
// Test 4: WAL corrupt tail skipped
// ---------------------------------------------------------------------------

#[test]
fn wal_corrupt_tail_recovery() {
    std::thread::Builder::new()
        .name("wal_corrupt".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let dir = temp_dir("wal_corrupt");
            let wal_dir = dir.join("wal");

            // Write 3 valid WAL entries
            {
                let mut wal = WalManager::open(&wal_dir).unwrap();
                for i in 1..=3u64 {
                    let mut frame = make_frame(0.5, 0.5);
                    frame.frame_meta.frame_id = i;
                    frame.frame_meta.strand_id = 0;
                    let entry_data =
                        FrameEntry::Full(Box::new(frame)).to_bytes().unwrap();
                    wal.log_entry(WalEntry {
                        frame_id: i,
                        strand_id: 0,
                        op: WalOp::Store,
                        payload: entry_data,
                    })
                    .unwrap();
                }
                wal.sync_all().unwrap();
            }

            // Append garbage to simulate partial write during crash
            let wal_path = wal_dir.join("strand_0.wal");
            {
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new()
                    .append(true)
                    .open(&wal_path)
                    .unwrap();
                file.write_all(b"GARBAGE_PARTIAL_WRITE_CORRUPT_DATA")
                    .unwrap();
            }

            // Open store — should recover the 3 valid entries, skip garbage
            let config = VoltStoreConfig {
                data_dir: dir.clone(),
                t1_overflow_threshold: 2000,
                t2_config: T2Config {
                    data_dir: dir.join("t2"),
                    ..T2Config::default()
                },
                ..VoltStoreConfig::default()
            };

            let store = VoltStore::open(config).unwrap();

            // All 3 valid frames should be recovered
            for i in 1..=3u64 {
                assert!(
                    store.get_by_id(i).is_some(),
                    "frame {i} should be recovered despite corrupt tail"
                );
            }

            let _ = std::fs::remove_dir_all(&dir);
        })
        .unwrap()
        .join()
        .unwrap();
}

// ---------------------------------------------------------------------------
// Test 5: Concurrent access — 10 readers + 1 writer
// ---------------------------------------------------------------------------

#[test]
fn concurrent_ten_readers_one_writer() {
    let store = VoltStore::new();
    let concurrent = ConcurrentVoltStore::new(store);

    // Pre-populate with a frame
    {
        let mut guard = concurrent.write().unwrap();
        guard.store(make_frame(0.5, 0.5)).unwrap();
    }

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(11)); // 10 readers + 1 writer
    let mut handles = Vec::new();

    // 10 reader threads
    for _ in 0..10 {
        let store = concurrent.clone();
        let barrier = barrier.clone();
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            for _ in 0..100 {
                let guard = store.read().unwrap();
                let _ = guard.get_by_id(1);
                let _ = guard.total_frame_count();
                let _ = guard.active_strand();
            }
        }));
    }

    // 1 writer thread
    {
        let store = concurrent.clone();
        let barrier = barrier.clone();
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            for _ in 0..50 {
                let mut guard = store.write().unwrap();
                guard.store(TensorFrame::new()).unwrap();
            }
        }));
    }

    // Wait for all threads — no panics, no deadlocks
    for h in handles {
        h.join().expect("thread panicked during concurrent access");
    }

    // Verify final state is consistent
    let guard = concurrent.read().unwrap();
    assert!(
        guard.total_frame_count() > 1,
        "should have multiple frames after concurrent writes"
    );
}

// ---------------------------------------------------------------------------
// Test 6: T2 sorted runs — flush, compact, retrieve
// ---------------------------------------------------------------------------

#[test]
fn t2_sorted_runs_roundtrip() {
    let dir = temp_dir("t2_sorted");
    let config = T2Config {
        data_dir: dir.clone(),
        memtable_flush_threshold: 1024, // very small, forces frequent flushes
        ..T2Config::default()
    };

    let mut t2 = Tier2Store::open(config).unwrap();

    // Store compressed frames
    for i in 0..20u64 {
        let mut frame = TensorFrame::new();
        frame.frame_meta.frame_id = i;
        frame.frame_meta.strand_id = 0;
        frame.frame_meta.global_certainty = 0.5;
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.1 * (i as f32 + 1.0); SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();
        let compressed = compress(&frame);
        t2.insert(FrameEntry::Compressed(compressed)).unwrap();
    }

    // Force flush + compact
    t2.maybe_flush_and_compact().unwrap();

    // All entries should be retrievable
    for i in 0..20u64 {
        let entry = t2.get(i);
        assert!(entry.is_some(), "frame {i} should be in T2");
        let entry = entry.unwrap();
        assert_eq!(entry.frame_id(), i);
    }

    // Non-existent ID returns None
    assert!(t2.get(9999).is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Test 7: Consolidation — similar frames → wisdom frame
// ---------------------------------------------------------------------------

#[test]
fn consolidation_creates_wisdom_frames() {
    let mut store = VoltStore::new();

    // Store > 64 similar frames so they overflow from T0 to T1
    // (all with the same R0 direction → cosine distance ≈ 0)
    for _ in 0..80 {
        let frame = make_frame(0.5, 0.5);
        store.store(frame).unwrap();
    }

    // Attempt consolidation — with many identical frames in T1, should find a cluster
    let result = store.consolidate_strand(0).unwrap();

    // With default min_cluster_size=5 and many identical frames in T1, should find clusters
    assert!(
        result.clusters_found > 0,
        "expected at least one cluster from identical frames in T1"
    );
    assert!(
        !result.wisdom_frames.is_empty(),
        "expected wisdom frames to be created"
    );

    // Wisdom frames should have high gamma
    for wisdom in &result.wisdom_frames {
        assert!(
            wisdom.frame_meta.global_certainty >= 0.9,
            "wisdom frame should have high gamma, got {}",
            wisdom.frame_meta.global_certainty
        );
        assert!(wisdom.frame_meta.verified, "wisdom frames should be verified");
    }
}

// ---------------------------------------------------------------------------
// Test 8: Bloom filter effectiveness
// ---------------------------------------------------------------------------

#[test]
fn bloom_filter_effectiveness() {
    let mut bloom = BloomFilter::new(10_000, 0.01);

    // Insert 1000 keys
    for k in 0..1000u64 {
        bloom.insert(k);
    }

    // No false negatives
    for k in 0..1000u64 {
        assert!(bloom.may_contain(k), "key {k} should be found (no false negatives)");
    }

    // False positive rate should be reasonable
    let mut false_positives = 0;
    let test_count = 10_000u64;
    for k in 1000..(1000 + test_count) {
        if bloom.may_contain(k) {
            false_positives += 1;
        }
    }
    let fp_rate = false_positives as f64 / test_count as f64;
    assert!(
        fp_rate < 0.05,
        "false positive rate {fp_rate:.4} should be < 5%"
    );

    // Serialization roundtrip
    let bytes = bloom.to_bytes();
    let restored = BloomFilter::from_bytes(&bytes).unwrap();
    for k in 0..1000u64 {
        assert!(
            restored.may_contain(k),
            "key {k} should survive serialization roundtrip"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 9: FrameEntry roundtrip — all decay levels
// ---------------------------------------------------------------------------

#[test]
fn frame_entry_roundtrip_all_levels() {
    // Run on a thread with 8MB stack — serde_json + TensorFrame can overflow.
    std::thread::Builder::new()
        .name("entry_roundtrip".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            // Full
            let mut frame = make_frame(0.7, 0.3);
            frame.frame_meta.frame_id = 100;
            frame.frame_meta.strand_id = 5;
            let full_entry = FrameEntry::Full(Box::new(frame));
            let bytes = full_entry.to_bytes().unwrap();
            let restored = FrameEntry::from_bytes(&bytes).unwrap();
            assert_eq!(restored.decay_level(), DecayLevel::Full);
            assert_eq!(restored.frame_id(), 100);
            assert_eq!(restored.strand_id(), 5);

            // Compressed
            let mut frame2 = make_frame(0.6, 0.4);
            frame2.frame_meta.frame_id = 200;
            frame2.frame_meta.strand_id = 3;
            let compressed = compress(&frame2);
            let comp_entry = FrameEntry::Compressed(compressed.clone());
            let bytes = comp_entry.to_bytes().unwrap();
            let restored = FrameEntry::from_bytes(&bytes).unwrap();
            assert_eq!(restored.decay_level(), DecayLevel::Compressed);
            assert_eq!(restored.frame_id(), 200);

            // Gist
            let gist = to_gist_frame(&compressed, [0.42; SLOT_DIM]);
            let gist_entry = FrameEntry::Gist(gist);
            let bytes = gist_entry.to_bytes().unwrap();
            let restored = FrameEntry::from_bytes(&bytes).unwrap();
            assert_eq!(restored.decay_level(), DecayLevel::Gist);
            assert_eq!(restored.frame_id(), 200);

            // Tombstone
            let ts = to_tombstone(300, 1, 999_999, Some(42));
            let ts_entry = FrameEntry::Tombstone(ts);
            let bytes = ts_entry.to_bytes().unwrap();
            let restored = FrameEntry::from_bytes(&bytes).unwrap();
            assert_eq!(restored.decay_level(), DecayLevel::Tombstoned);
            assert_eq!(restored.frame_id(), 300);
        })
        .unwrap()
        .join()
        .unwrap();
}

// ---------------------------------------------------------------------------
// Test 10: Pin/unpin integration
// ---------------------------------------------------------------------------

#[test]
fn pin_unpin_integration() {
    let mut store = VoltStore::new();

    let id1 = store.store(make_frame(0.3, 0.5)).unwrap();
    let id2 = store.store(make_frame(0.3, 0.5)).unwrap();

    // Pin id1
    store.pin_frame(id1);
    assert!(store.is_frame_pinned(id1));
    assert!(!store.is_frame_pinned(id2));

    // Unpin id1
    store.unpin_frame(id1);
    assert!(!store.is_frame_pinned(id1));
}

// ---------------------------------------------------------------------------
// Test 11: Memory-only vs disk-backed
// ---------------------------------------------------------------------------

#[test]
fn memory_only_vs_disk_backed() {
    // Memory-only
    let store = VoltStore::new();
    assert!(!store.is_disk_backed());
    assert_eq!(store.t2_len(), 0);

    // Disk-backed
    std::thread::Builder::new()
        .name("disk_backed".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let dir = temp_dir("mem_vs_disk");
            let config = VoltStoreConfig {
                data_dir: dir.clone(),
                t2_config: T2Config {
                    data_dir: dir.join("t2"),
                    ..T2Config::default()
                },
                ..VoltStoreConfig::default()
            };
            let store = VoltStore::open(config).unwrap();
            assert!(store.is_disk_backed());
            let _ = std::fs::remove_dir_all(&dir);
        })
        .unwrap()
        .join()
        .unwrap();
}

// ---------------------------------------------------------------------------
// Test 12: Total entry count across tiers
// ---------------------------------------------------------------------------

#[test]
fn total_entry_count_across_tiers() {
    std::thread::Builder::new()
        .name("total_entries".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let dir = temp_dir("total_entries");
            let config = VoltStoreConfig {
                data_dir: dir.clone(),
                t1_overflow_threshold: 5, // very low → frames quickly go to T2
                t2_config: T2Config {
                    data_dir: dir.join("t2"),
                    ..T2Config::default()
                },
                ..VoltStoreConfig::default()
            };

            let mut store = VoltStore::open(config).unwrap();

            // Store 100 frames — some will overflow to T2
            for _ in 0..100 {
                store.store(make_frame(0.5, 0.5)).unwrap();
            }

            // Total entries should account for T0 + T1 + T2
            let total = store.total_entry_count();
            assert!(
                total >= 100,
                "total_entry_count should be >= 100, got {total}"
            );

            let _ = std::fs::remove_dir_all(&dir);
        })
        .unwrap()
        .join()
        .unwrap();
}

// ---------------------------------------------------------------------------
// Test 13: GC retention scoring
// ---------------------------------------------------------------------------

#[test]
fn gc_retention_scoring() {
    use volt_db::gc::FrameGcMeta;

    let engine = GcEngine::with_defaults();
    let now = 100_000_000u64; // 100 seconds in micros

    // Fresh frame with high gamma → high score
    let fresh_high = FrameGcMeta {
        frame_id: 1,
        strand_id: 0,
        created_at: now - 1_000_000, // 1 second ago
        global_certainty: 0.9,
        current_level: DecayLevel::Full,
        reference_count: 0,
        is_pinned: false,
        is_wisdom: false,
    };
    let score_fresh_high = engine.retention_score(&fresh_high, now);
    assert!(
        score_fresh_high > 0.5,
        "fresh high-gamma frame should have high score, got {score_fresh_high}"
    );

    // Old frame with low gamma → low score
    let old_low = FrameGcMeta {
        frame_id: 2,
        strand_id: 0,
        created_at: 1_000_000, // very old
        global_certainty: 0.05,
        current_level: DecayLevel::Full,
        reference_count: 0,
        is_pinned: false,
        is_wisdom: false,
    };
    let score_old_low = engine.retention_score(&old_low, now);
    assert!(
        score_old_low < score_fresh_high,
        "old low-gamma frame should score lower: {score_old_low} vs {score_fresh_high}"
    );
}
