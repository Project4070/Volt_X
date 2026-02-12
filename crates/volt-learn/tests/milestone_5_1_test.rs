//! Integration tests for Milestone 5.1: Learning Event Logging.
//!
//! Verifies the acceptance criteria from `roadmap/PHASE-5.md`:
//!
//! 1. After 100 queries, learning buffer has 100 events
//! 2. Statistics reflect actual usage patterns
//! 3. Events survive restart (persisted alongside strand in VoltDB)

use volt_core::meta::DiscourseType;
use volt_core::MAX_SLOTS;
use volt_learn::{EventLogger, LearningEvent, LoggerConfig};

fn make_event(
    frame_id: u64,
    strand_id: u64,
    query_type: DiscourseType,
    gamma: f32,
    iterations: u32,
    ghost_activations: usize,
) -> LearningEvent {
    LearningEvent {
        frame_id,
        strand_id,
        query_type,
        gamma_scores: [gamma; MAX_SLOTS],
        convergence_iterations: iterations,
        ghost_activations,
        timestamp: frame_id * 1_000_000,
    }
}

/// **Acceptance criterion 1**: After 100 queries, learning buffer has 100 events.
#[test]
fn after_100_queries_buffer_has_100_events() {
    let mut logger = EventLogger::new();
    for i in 0..100u64 {
        logger.log(make_event(i + 1, 0, DiscourseType::Query, 0.8, 10, 3));
    }
    assert_eq!(logger.event_count(), 100);
}

/// **Acceptance criterion 2**: Statistics reflect actual usage patterns
/// (more coding queries -> coding strand dominates).
#[test]
fn statistics_reflect_usage_patterns() {
    let mut logger = EventLogger::new();

    // 80 coding queries (Command type) to strand 1 — high gamma, fast convergence
    for i in 0..80u64 {
        logger.log(make_event(i + 1, 1, DiscourseType::Command, 0.9, 8, 2));
    }

    // 20 general queries to strand 2 — lower gamma, slower convergence
    for i in 80..100u64 {
        logger.log(make_event(i + 1, 2, DiscourseType::Query, 0.5, 15, 5));
    }

    let stats = logger.all_strand_stats();
    let s1 = &stats[&1];
    let s2 = &stats[&2];

    // Strand 1 (coding) dominates in count
    assert_eq!(s1.query_count, 80);
    assert_eq!(s2.query_count, 20);

    // Coding strand has higher gamma (more certain)
    assert!(
        s1.average_gamma > s2.average_gamma,
        "coding strand should have higher gamma: {} vs {}",
        s1.average_gamma,
        s2.average_gamma
    );

    // Coding strand converges faster (fewer iterations)
    assert!(
        s1.average_iterations < s2.average_iterations,
        "coding strand should converge faster: {} vs {}",
        s1.average_iterations,
        s2.average_iterations
    );

    // Topic distribution: strand 1 is all Command
    assert_eq!(s1.topic_distribution.command, 80);
    assert_eq!(s1.topic_distribution.query, 0);
    assert_eq!(
        s1.topic_distribution.dominant(),
        Some(DiscourseType::Command)
    );

    // Topic distribution: strand 2 is all Query
    assert_eq!(s2.topic_distribution.query, 20);
    assert_eq!(s2.topic_distribution.command, 0);
    assert_eq!(
        s2.topic_distribution.dominant(),
        Some(DiscourseType::Query)
    );
}

/// **Acceptance criterion 3**: Events survive restart (persisted alongside strand).
#[test]
fn events_survive_restart() {
    let dir = std::env::temp_dir().join(format!(
        "volt_learn_m51_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("learning_events.json");

    // Phase 1: Log 50 events across 3 strands and save
    {
        let mut logger = EventLogger::new();
        for i in 0..50u64 {
            logger.log(make_event(
                i + 1,
                i % 3,
                DiscourseType::Query,
                0.7,
                12,
                1,
            ));
        }
        logger.save(&path).expect("save should succeed");
    }

    // Phase 2: Load into a fresh logger and verify everything is intact
    {
        let logger = EventLogger::load(&path).expect("load should succeed");

        // All 50 events survived
        assert_eq!(logger.event_count(), 50);

        // Content is correct
        let events = logger.events();
        assert_eq!(events[0].frame_id, 1);
        assert_eq!(events[49].frame_id, 50);
        assert_eq!(events[0].strand_id, 0);
        assert_eq!(events[1].strand_id, 1);
        assert_eq!(events[2].strand_id, 2);

        // Statistics are recomputable from loaded events
        let stats = logger.all_strand_stats();
        assert!(stats.contains_key(&0));
        assert!(stats.contains_key(&1));
        assert!(stats.contains_key(&2));

        // Strand 0 has ceil(50/3) = 17 events, strand 1 has 17, strand 2 has 16
        assert_eq!(stats[&0].query_count, 17);
        assert_eq!(stats[&1].query_count, 17);
        assert_eq!(stats[&2].query_count, 16);
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

/// Buffer capacity enforcement: events beyond capacity cause FIFO eviction.
#[test]
fn buffer_capacity_is_enforced() {
    let config = LoggerConfig {
        buffer_capacity: 50,
    };
    let mut logger = EventLogger::with_config(config);

    for i in 0..100u64 {
        logger.log(make_event(i + 1, 0, DiscourseType::Query, 0.8, 10, 0));
    }

    // Only the last 50 remain
    assert_eq!(logger.event_count(), 50);
    assert_eq!(logger.events()[0].frame_id, 51);
    assert_eq!(logger.events()[49].frame_id, 100);
}

/// Per-strand filtering works correctly.
#[test]
fn events_for_strand_returns_correct_subset() {
    let mut logger = EventLogger::new();
    logger.log(make_event(1, 10, DiscourseType::Query, 0.5, 10, 0));
    logger.log(make_event(2, 20, DiscourseType::Command, 0.9, 5, 0));
    logger.log(make_event(3, 10, DiscourseType::Statement, 0.7, 8, 0));

    let strand_10 = logger.events_for_strand(10);
    assert_eq!(strand_10.len(), 2);
    assert_eq!(strand_10[0].frame_id, 1);
    assert_eq!(strand_10[1].frame_id, 3);

    let strand_20 = logger.events_for_strand(20);
    assert_eq!(strand_20.len(), 1);
    assert_eq!(strand_20[0].frame_id, 2);

    let strand_99 = logger.events_for_strand(99);
    assert!(strand_99.is_empty());
}
