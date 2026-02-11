//! Integration tests for Milestone 4.2: HNSW Indexing + Ghost Bleed.
//!
//! Verifies the acceptance criteria from `roadmap/PHASE-4.md`:
//!
//! 1. Semantic retrieval finds relevant frames from the right strand
//! 2. HNSW finds old frames → ghost buffer populated → frame IDs valid
//! 3. Ghost buffer refreshes when topic changes
//! 4. Temporal range queries work correctly
//! 5. HNSW query performance at scale

use volt_core::{SlotData, SlotRole, TensorFrame, SLOT_DIM};
use volt_db::VoltStore;

/// Creates a frame with R₀ data pointing in a specific "direction".
///
/// Direction is encoded by setting `vector[dim] = 1.0` (plus small noise
/// to avoid perfect symmetry) to simulate different topics.
fn make_directional_frame(direction_dim: usize) -> TensorFrame {
    let mut frame = TensorFrame::new();
    let mut slot = SlotData::new(SlotRole::Agent);

    let mut vector = [0.01f32; SLOT_DIM]; // small baseline
    vector[direction_dim % SLOT_DIM] = 1.0;

    slot.write_resolution(0, vector);
    frame.write_slot(0, slot).unwrap();
    frame
}

/// Creates a frame with R₀ data pointing in a specific direction with a timestamp.
fn make_timestamped_frame(direction_dim: usize, created_at: u64) -> TensorFrame {
    let mut frame = make_directional_frame(direction_dim);
    frame.frame_meta.created_at = created_at;
    frame
}

// === Acceptance Test 1 ===
// "Ask about Rust → get back relevant frames from Coding strand (not Cooking strand)"

#[test]
fn semantic_retrieval_strand_isolation() {
    let mut store = VoltStore::new();

    // Strand 1: "Coding" — frames pointing in dim 0
    store.switch_strand(1).unwrap();
    for _ in 0..20 {
        store.store(make_directional_frame(0)).unwrap();
    }

    // Strand 2: "Cooking" — frames pointing in dim 128
    store.switch_strand(2).unwrap();
    for _ in 0..20 {
        store.store(make_directional_frame(128)).unwrap();
    }

    // Query with a "coding" direction (dim 0) — should find strand 1 frames
    let mut coding_query = [0.01f32; SLOT_DIM];
    coding_query[0] = 1.0;

    let results = store.query_similar(&coding_query, 10);
    assert!(
        !results.is_empty(),
        "should find similar frames for coding query"
    );
    // Top results should be from strand 1 (coding), not strand 2 (cooking)
    let coding_count = results.iter().filter(|r| r.strand_id == 1).count();
    let cooking_count = results.iter().filter(|r| r.strand_id == 2).count();
    assert!(
        coding_count > cooking_count,
        "coding strand should dominate results: coding={coding_count} vs cooking={cooking_count}"
    );

    // Per-strand query: strand 1 should have results
    let strand_1_results = store.query_similar_in_strand(1, &coding_query, 10);
    assert!(
        !strand_1_results.is_empty(),
        "strand 1 should have coding results"
    );
    assert!(
        strand_1_results.iter().all(|r| r.strand_id == 1),
        "all per-strand results should be from strand 1"
    );

    // Per-strand query: strand 2 with coding query should still return results
    // (cosine distance may be non-zero due to the baseline noise)
    let strand_2_results = store.query_similar_in_strand(2, &coding_query, 10);
    // But their distances should be larger than strand 1 results
    if !strand_2_results.is_empty() && !strand_1_results.is_empty() {
        let best_strand_1 = strand_1_results[0].distance;
        let best_strand_2 = strand_2_results[0].distance;
        assert!(
            best_strand_1 <= best_strand_2,
            "coding query should be closer to coding strand: d1={best_strand_1} vs d2={best_strand_2}"
        );
    }
}

// === Acceptance Test 2 ===
// "HNSW finds old frame → ghost appears → full frame loads on page fault"

#[test]
fn ghost_buffer_populated_from_hnsw() {
    let mut store = VoltStore::new();

    // Store 100 frames about topic A (dim 0)
    for _ in 0..100 {
        store.store(make_directional_frame(0)).unwrap();
    }

    // Ghost buffer should have been populated from the HNSW query
    assert!(
        !store.ghost_buffer().is_empty(),
        "ghost buffer should be populated after storing frames"
    );

    // All ghost entries should have valid frame_ids that exist in the store
    for entry in store.ghost_buffer().entries() {
        assert!(
            store.get_by_id(entry.frame_id).is_some(),
            "ghost entry frame_id {} should point to an existing frame",
            entry.frame_id
        );
    }

    // Ghost gist vectors should be 256 dims
    let gists = store.ghost_gists();
    for g in &gists {
        assert_eq!(g.len(), SLOT_DIM);
        // Each gist should have finite values
        assert!(
            g.iter().all(|x| x.is_finite()),
            "ghost gist should have finite values"
        );
    }
}

// === Acceptance Test 3 ===
// "Ghost buffer refreshes when topic changes"

#[test]
fn ghost_buffer_refreshes_on_topic_change() {
    let mut store = VoltStore::new();

    // Store frames about topic A (dim 0)
    for _ in 0..30 {
        store.store(make_directional_frame(0)).unwrap();
    }

    // Record ghost buffer state after topic A
    let ghost_ids_a: Vec<u64> = store
        .ghost_buffer()
        .entries()
        .iter()
        .map(|e| e.frame_id)
        .collect();
    assert!(!ghost_ids_a.is_empty(), "should have ghosts after topic A");

    // Store frames about topic B (dim 200) — very different direction
    for _ in 0..30 {
        store.store(make_directional_frame(200)).unwrap();
    }

    // Ghost buffer should have been refreshed with topic B content
    let ghost_ids_b: Vec<u64> = store
        .ghost_buffer()
        .entries()
        .iter()
        .map(|e| e.frame_id)
        .collect();
    assert!(!ghost_ids_b.is_empty(), "should have ghosts after topic B");

    // The ghost buffer should have changed (different frame IDs)
    // With 30 frames about topic B, the most recent query is topic B,
    // so the top ghost results should shift toward topic B frames.
    // Frame IDs 31-60 are topic B, 1-30 are topic A.
    let topic_b_ghosts = ghost_ids_b.iter().filter(|&&id| id > 30).count();
    assert!(
        topic_b_ghosts > 0,
        "ghost buffer should contain topic B frames after topic change"
    );
}

// === Acceptance Test 4 ===
// Temporal range queries

#[test]
fn temporal_range_query() {
    let mut store = VoltStore::new();

    // Store frames with specific timestamps
    // Week 1: timestamps 1_000_000 to 7_000_000
    for day in 1..=7 {
        let ts = day * 1_000_000;
        store
            .store(make_timestamped_frame(0, ts))
            .unwrap();
    }

    // Week 2: timestamps 8_000_000 to 14_000_000
    for day in 8..=14 {
        let ts = day * 1_000_000;
        store
            .store(make_timestamped_frame(0, ts))
            .unwrap();
    }

    // Query week 1 only
    let week1 = store.query_time_range(1_000_000, 7_000_000);
    assert_eq!(week1.len(), 7, "should find 7 frames in week 1");

    // Query week 2 only
    let week2 = store.query_time_range(8_000_000, 14_000_000);
    assert_eq!(week2.len(), 7, "should find 7 frames in week 2");

    // Query entire range
    let all = store.query_time_range(0, u64::MAX);
    assert_eq!(all.len(), 14, "should find all 14 frames");

    // Query empty range
    let none = store.query_time_range(100_000_000, 200_000_000);
    assert!(none.is_empty(), "should find no frames in future range");
}

// === Acceptance Test 5 ===
// HNSW query performance

#[test]
fn hnsw_query_performance() {
    let mut store = VoltStore::new();

    // Insert 1,000 frames with varied directions
    for i in 0..1_000 {
        store
            .store(make_directional_frame(i % SLOT_DIM))
            .unwrap();
    }

    assert_eq!(store.hnsw_entries(), 1_000);

    // Measure query time
    let query = [0.1f32; SLOT_DIM];
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let results = store.query_similar(&query, 10);
        assert!(!results.is_empty());
    }
    let elapsed = start.elapsed();
    let per_query = elapsed / 100;

    // Target: < 500μs per query in release, < 2ms in debug (unoptimized).
    // The ARCHITECTURE.md spec is for 65K entries in release mode;
    // 1K entries in debug mode should be well within 2ms.
    let limit_us = if cfg!(debug_assertions) { 2_000 } else { 500 };
    assert!(
        per_query.as_micros() < limit_us,
        "HNSW query should be < {}μs, got {}μs",
        limit_us,
        per_query.as_micros()
    );
}

// === Additional Tests ===

#[test]
fn load_rebuilds_indices() {
    let dir = std::env::temp_dir().join("volt_db_test_42_rebuild");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rebuild_test.json");

    let mut store = VoltStore::new();

    // Store enough frames to have some in T1
    for i in 0..80 {
        store
            .store(make_directional_frame(i % SLOT_DIM))
            .unwrap();
    }

    let hnsw_before = store.hnsw_entries();
    let temporal_before = store.temporal_entries();
    assert!(hnsw_before > 0);
    assert!(temporal_before > 0);

    // Save and reload
    store.save(&path).unwrap();
    let loaded = VoltStore::load(&path).unwrap();

    // Indices should be rebuilt from T1 frames
    // T1 has fewer frames than total (T0 is ephemeral)
    assert!(loaded.hnsw_entries() > 0, "HNSW should be rebuilt on load");
    assert!(
        loaded.temporal_entries() > 0,
        "temporal should be rebuilt on load"
    );
    // HNSW entries should equal T1 frame count (only T1 is saved/loaded)
    assert_eq!(loaded.hnsw_entries(), loaded.t1_len());

    // Semantic search should work on the reloaded store
    let query = [0.1f32; SLOT_DIM];
    let results = loaded.query_similar(&query, 10);
    assert!(!results.is_empty(), "search should work after load");

    // Clean up
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn frames_without_r0_are_gracefully_skipped() {
    let mut store = VoltStore::new();

    // Store a frame with no R₀ data
    store.store(TensorFrame::new()).unwrap();

    // Store a frame with R₀ data
    store.store(make_directional_frame(0)).unwrap();

    // Only one frame should be indexed
    assert_eq!(store.hnsw_entries(), 1);
    assert_eq!(store.temporal_entries(), 1);
    // But both frames should be stored
    assert_eq!(store.total_frame_count(), 2);
}
