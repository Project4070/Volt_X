//! Integration tests for Milestone 4.1: VoltDB T0 + T1.
//!
//! Tests cover all milestone acceptance criteria:
//! - 100 frames stored → all retrievable by ID
//! - T0 fills at 64 → oldest moves to T1 → still retrievable
//! - Switch strand → new queries go to new strand → old strand intact
//! - Restart server → T1 persists (serialize/deserialize)
//! - Retrieval by ID performance (< 0.1ms)

use volt_core::{SlotData, SlotRole, TensorFrame, SLOT_DIM};
use volt_db::tier0::T0_CAPACITY;
use volt_db::VoltStore;

/// Creates a frame with distinguishable content for testing.
fn make_test_frame() -> TensorFrame {
    let mut frame = TensorFrame::new();
    let mut slot = SlotData::new(SlotRole::Agent);
    slot.write_resolution(0, [0.42; SLOT_DIM]);
    frame.write_slot(0, slot).unwrap();
    frame
}

/// Milestone 4.1 acceptance: "Ask 100 questions → all 100 frames stored → retrievable by ID"
#[test]
fn hundred_frames_all_retrievable() {
    let mut store = VoltStore::new();
    let mut ids = Vec::with_capacity(100);

    for _ in 0..100 {
        let id = store.store(make_test_frame()).unwrap();
        ids.push(id);
    }

    assert_eq!(store.total_frame_count(), 100);

    // Every single frame must be retrievable by its ID
    for &id in &ids {
        let frame = store
            .get_by_id(id)
            .unwrap_or_else(|| panic!("frame {id} not found"));
        assert_eq!(frame.frame_meta.frame_id, id);
    }
}

/// Milestone 4.1 acceptance: "T0 fills at 64 → oldest frame moves to T1 → still retrievable from T1"
#[test]
fn t0_eviction_to_t1() {
    let mut store = VoltStore::new();

    // Fill T0 exactly
    for _ in 0..T0_CAPACITY {
        store.store(make_test_frame()).unwrap();
    }
    assert_eq!(store.t0_len(), T0_CAPACITY);
    assert_eq!(store.t1_len(), 0);

    // 65th frame should evict frame_id=1 to T1
    let id_65 = store.store(make_test_frame()).unwrap();
    assert_eq!(store.t0_len(), T0_CAPACITY);
    assert_eq!(store.t1_len(), 1);

    // The evicted frame (id=1) should still be in T1
    let evicted = store.get_by_id(1).expect("evicted frame not found in T1");
    assert_eq!(evicted.frame_meta.frame_id, 1);

    // The new frame should be in T0
    let newest = store.get_by_id(id_65).expect("newest frame not found in T0");
    assert_eq!(newest.frame_meta.frame_id, id_65);
}

/// Milestone 4.1 acceptance: "Switch strand → new queries go to new strand → old strand intact"
#[test]
fn strand_switching_preserves_isolation() {
    let mut store = VoltStore::new();

    // Store 5 frames in strand 1
    store.switch_strand(1).unwrap();
    let mut strand1_ids = Vec::new();
    for _ in 0..5 {
        strand1_ids.push(store.store(make_test_frame()).unwrap());
    }

    // Switch to strand 2, store 3 frames
    store.switch_strand(2).unwrap();
    let mut strand2_ids = Vec::new();
    for _ in 0..3 {
        strand2_ids.push(store.store(make_test_frame()).unwrap());
    }

    // Verify strand 1 frames are intact
    let strand1_frames = store.get_by_strand(1);
    assert_eq!(strand1_frames.len(), 5);
    for &id in &strand1_ids {
        let frame = store.get_by_id(id).unwrap();
        assert_eq!(frame.frame_meta.strand_id, 1);
    }

    // Verify strand 2 frames are correct
    let strand2_frames = store.get_by_strand(2);
    assert_eq!(strand2_frames.len(), 3);
    for &id in &strand2_ids {
        let frame = store.get_by_id(id).unwrap();
        assert_eq!(frame.frame_meta.strand_id, 2);
    }

    // Switch back to strand 1, add more
    store.switch_strand(1).unwrap();
    store.store(make_test_frame()).unwrap();
    assert_eq!(store.get_by_strand(1).len(), 6);
    assert_eq!(store.get_by_strand(2).len(), 3);
}

/// Milestone 4.1 acceptance: "Restart server → T1 persists"
#[test]
fn persistence_across_restart() {
    let dir = std::env::temp_dir().join("volt_db_integration_persist");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("persist_test.json");

    // Phase 1: Create store, add frames, save
    let stored_ids;
    {
        let mut store = VoltStore::new();
        store.switch_strand(1).unwrap();

        // Store enough to force evictions into T1
        let mut ids = Vec::new();
        for _ in 0..(T0_CAPACITY + 20) {
            ids.push(store.store(make_test_frame()).unwrap());
        }
        stored_ids = ids;

        assert!(store.t1_len() > 0, "some frames should be in T1");
        store.save(&path).unwrap();
    }

    // Phase 2: Load from disk (simulates restart)
    {
        let loaded = VoltStore::load(&path).unwrap();

        // T0 is fresh (empty) after restart
        assert_eq!(loaded.t0_len(), 0);

        // T1 frames should be intact
        assert!(loaded.t1_len() > 0);

        // All T1 frames should be retrievable
        // (T0 frames are lost on restart — that's by design)
        let t1_frame_count = loaded.t1_len();
        for &id in &stored_ids[..t1_frame_count] {
            assert!(
                loaded.get_by_id(id).is_some(),
                "frame {id} not found after restart"
            );
        }

        // New frames should get IDs that don't conflict with T1
        let mut loaded = loaded;
        let t1_max_id = stored_ids[..t1_frame_count]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let new_id = loaded.store(make_test_frame()).unwrap();
        assert!(
            new_id > t1_max_id,
            "new ID {new_id} should be > max T1 ID {t1_max_id}"
        );
    }

    // Clean up
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

/// Milestone 4.1 acceptance: "Retrieval by ID: < 0.1ms"
#[test]
fn retrieval_performance() {
    let mut store = VoltStore::new();

    // Store 1000 frames (well beyond T0 capacity)
    let mut ids = Vec::with_capacity(1000);
    for _ in 0..1000 {
        ids.push(store.store(make_test_frame()).unwrap());
    }

    // Measure retrieval time for various IDs
    let start = std::time::Instant::now();
    let iterations = 1000;
    for &id in ids.iter().take(iterations) {
        let _ = store.get_by_id(id);
    }
    let elapsed = start.elapsed();
    let per_lookup = elapsed / iterations as u32;

    // Should be well under 0.1ms (100μs) per lookup
    assert!(
        per_lookup.as_micros() < 100,
        "retrieval took {per_lookup:?} per lookup, expected < 100μs"
    );
}

/// Test strand management operations: create, list, switch.
#[test]
fn strand_management() {
    let mut store = VoltStore::new();

    // Default strand 0 exists
    assert!(store.list_strands().contains(&0));

    // Create new strands
    store.create_strand(10).unwrap();
    store.create_strand(20).unwrap();
    store.create_strand(30).unwrap();

    let strands = store.list_strands();
    assert_eq!(strands, vec![0, 10, 20, 30]);

    // Duplicate creation should fail
    assert!(store.create_strand(10).is_err());

    // Switch strand (including to one that doesn't exist yet)
    store.switch_strand(50).unwrap();
    assert_eq!(store.active_strand(), 50);
    assert!(store.list_strands().contains(&50));
}

/// Test that recent() returns frames in newest-first order from T0.
#[test]
fn recent_frames_ordering() {
    let mut store = VoltStore::new();

    for _ in 0..20 {
        store.store(make_test_frame()).unwrap();
    }

    let recent = store.recent(5);
    assert_eq!(recent.len(), 5);

    // Should be in descending order of frame_id (newest first)
    for window in recent.windows(2) {
        assert!(
            window[0].frame_meta.frame_id > window[1].frame_meta.frame_id,
            "recent frames should be newest-first"
        );
    }
}

/// Test that frames carry correct strand_id after eviction to T1.
#[test]
fn evicted_frames_preserve_strand_id() {
    let mut store = VoltStore::new();

    // Use strand 42 for all frames
    store.switch_strand(42).unwrap();
    for _ in 0..(T0_CAPACITY + 10) {
        store.store(make_test_frame()).unwrap();
    }

    // Frames evicted to T1 should still have strand_id = 42
    let t1_frames = store.t1().get_by_strand(42);
    assert_eq!(t1_frames.len(), 10);
    for frame in t1_frames {
        assert_eq!(frame.frame_meta.strand_id, 42);
    }
}

/// Test multi-strand scenario with many frames across different strands.
#[test]
fn multi_strand_heavy() {
    let mut store = VoltStore::new();

    // Create 5 strands, store 30 frames each = 150 total
    for strand in 1..=5 {
        store.switch_strand(strand).unwrap();
        for _ in 0..30 {
            store.store(make_test_frame()).unwrap();
        }
    }

    assert_eq!(store.total_frame_count(), 150);

    // Each strand should have 30 frames (across T0 and T1)
    for strand in 1..=5u64 {
        let frames = store.get_by_strand(strand);
        assert_eq!(
            frames.len(),
            30,
            "strand {strand} should have 30 frames, got {}",
            frames.len()
        );
    }
}
