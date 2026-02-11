//! Integration tests for volt-safety (Milestone 3.3).
//!
//! Tests the full safety layer: axioms, monitor, scorer, veto, and
//! integrated pipeline wrapping.

use volt_core::{SlotData, SlotRole, TensorFrame, SLOT_DIM};
use volt_hard::default_pipeline;
use volt_hard::math_engine::MathEngine;
use volt_hard::strand::HardStrand;
use volt_safety::axiom::{default_axioms, Severity};
use volt_safety::layer::SafetyLayer;
use volt_safety::monitor::TransitionMonitor;
use volt_safety::scorer::ViolationLevel;

/// Stack size for tests that allocate TensorFrames (each ~65KB).
/// Integration tests need extra stack because they exercise the full
/// safety-wrapped pipeline (router + strands + certainty + proof).
const TEST_STACK: usize = 8 * 1024 * 1024;

// ===========================================================================
// Milestone 3.3 acceptance tests (from PHASE-3.md)
// ===========================================================================

/// Normal query -> safety layer passes through, no interference.
#[test]
fn milestone_normal_query_passes_through() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let mut layer = SafetyLayer::new(default_pipeline());

            let mut frame = TensorFrame::new();
            let mut slot = SlotData::new(SlotRole::Agent);
            slot.write_resolution(0, [0.1; SLOT_DIM]);
            frame.write_slot(0, slot).unwrap();
            frame.meta[0].certainty = 0.8;

            let result = layer.process(&frame).unwrap();
            assert!(!result.vetoed, "Normal query should pass through safety layer");
            assert!(result.proof.is_some(), "Proof chain should be present");
            assert!(
                result.veto_log.is_none(),
                "No veto log for normal queries"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Query touching K1 (harm) -> violation detected -> Omega Veto fires
/// -> safe default response.
#[test]
fn milestone_k1_harm_triggers_omega_veto() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let axioms = default_axioms();
            let k1_vector = axioms[0].vector;
            let mut layer = SafetyLayer::new(default_pipeline());

            let mut frame = TensorFrame::new();
            let mut slot = SlotData::new(SlotRole::Predicate);
            slot.write_resolution(0, k1_vector);
            frame.write_slot(1, slot).unwrap();
            frame.meta[1].certainty = 0.9;

            let result = layer.process(&frame).unwrap();

            assert!(result.vetoed, "K1 violation should trigger Omega Veto");
            assert!(
                result.frame.is_empty(),
                "Safe default should be an empty frame"
            );
            assert!(
                !result.frame.frame_meta.verified,
                "Safe default should not be verified"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Omega Veto logs include full frame state at time of trigger.
#[test]
fn milestone_veto_log_includes_frame_state() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let axioms = default_axioms();
            let k1_vector = axioms[0].vector;
            let mut layer = SafetyLayer::new(default_pipeline());

            // Build a multi-slot frame with a K1-violating slot
            let mut frame = TensorFrame::new();

            let mut s0 = SlotData::new(SlotRole::Agent);
            s0.write_resolution(0, [0.2; SLOT_DIM]);
            frame.write_slot(0, s0).unwrap();
            frame.meta[0].certainty = 0.7;

            let mut s1 = SlotData::new(SlotRole::Predicate);
            s1.write_resolution(0, k1_vector);
            frame.write_slot(1, s1).unwrap();
            frame.meta[1].certainty = 0.95;

            let result = layer.process(&frame).unwrap();
            assert!(result.vetoed);

            // Veto log must include full frame state
            let log = result.veto_log.as_ref().unwrap();
            assert_eq!(
                log.trigger_frame.active_slot_count(),
                2,
                "Trigger frame should preserve all original slots"
            );
            assert!(
                log.aggregate_score > 0.7,
                "Aggregate score should exceed K1 threshold"
            );
            assert!(
                !log.violation_details.is_empty(),
                "Violation details should be present"
            );
            assert!(
                log.violation_details
                    .iter()
                    .any(|d| d.contains("K1_harm")),
                "Violation details should mention K1_harm"
            );

            // Veto count should be recorded
            assert_eq!(layer.veto_count(), 1);
            let logs = layer.veto_logs();
            assert_eq!(logs.len(), 1);
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Safety layer adds < 1ms latency to normal queries.
#[test]
fn milestone_safety_latency_under_1ms() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let mut layer = SafetyLayer::new(default_pipeline());

            let mut frame = TensorFrame::new();
            let mut slot = SlotData::new(SlotRole::Agent);
            slot.write_resolution(0, [0.1; SLOT_DIM]);
            frame.write_slot(0, slot).unwrap();
            frame.meta[0].certainty = 0.8;

            // Warm up
            let _ = layer.process(&frame);

            // Measure safety overhead by comparing with/without safety
            let start_safe = std::time::Instant::now();
            for _ in 0..10 {
                let _ = layer.process(&frame);
            }
            let safe_elapsed = start_safe.elapsed();

            let pipeline = default_pipeline();
            let start_raw = std::time::Instant::now();
            for _ in 0..10 {
                let _ = pipeline.process(&frame);
            }
            let raw_elapsed = start_raw.elapsed();

            let overhead_per_call = if safe_elapsed > raw_elapsed {
                (safe_elapsed - raw_elapsed) / 10
            } else {
                std::time::Duration::ZERO
            };

            assert!(
                overhead_per_call < std::time::Duration::from_millis(1),
                "Safety overhead per call: {:?}, expected < 1ms",
                overhead_per_call
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Cannot bypass safety by crafting special frame structures (adversarial testing).
#[test]
fn milestone_adversarial_cannot_bypass_safety() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let axioms = default_axioms();
            let mut layer = SafetyLayer::new(default_pipeline());

            // Strategy 1: Put violating content in different slots
            for slot_idx in 0..16_usize {
                let mut frame = TensorFrame::new();
                let mut slot = SlotData::new(SlotRole::Free(slot_idx as u8));
                slot.write_resolution(0, axioms[0].vector); // K1 harm
                frame.write_slot(slot_idx, slot).unwrap();

                let result = layer.process(&frame).unwrap();
                assert!(
                    result.vetoed,
                    "K1 violation in slot S{} should be detected",
                    slot_idx
                );
            }

            // Strategy 2: Slightly perturbed axiom vector should still trigger
            let mut perturbed = axioms[0].vector;
            // Add small noise to first 10 dims
            for val in perturbed.iter_mut().take(10) {
                *val += 0.01;
            }
            // Re-normalize
            let norm: f32 = perturbed.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut perturbed {
                *x /= norm;
            }

            let mut frame = TensorFrame::new();
            let mut slot = SlotData::new(SlotRole::Predicate);
            slot.write_resolution(0, perturbed);
            frame.write_slot(1, slot).unwrap();

            let result = layer.process(&frame).unwrap();
            assert!(
                result.vetoed,
                "Slightly perturbed K1 vector should still trigger veto"
            );

            // Strategy 3: Multiple safe slots shouldn't mask a violating one
            let mut frame = TensorFrame::new();
            for i in 0..8_usize {
                let mut safe_slot = SlotData::new(SlotRole::Free(i as u8));
                safe_slot.write_resolution(0, [0.05; SLOT_DIM]);
                frame.write_slot(i, safe_slot).unwrap();
                frame.meta[i].certainty = 1.0;
            }
            // Sneak violating content into slot 8
            let mut bad_slot = SlotData::new(SlotRole::Result);
            bad_slot.write_resolution(0, axioms[0].vector);
            frame.write_slot(8, bad_slot).unwrap();

            let result = layer.process(&frame).unwrap();
            assert!(
                result.vetoed,
                "Violating slot hidden among safe slots should be detected"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

// ===========================================================================
// Additional integration tests
// ===========================================================================

/// Math queries pass through safety with correct results.
#[test]
fn math_query_passes_safety_with_correct_result() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let mut layer = SafetyLayer::new(default_pipeline());

            let engine = MathEngine::new();
            let cap = *engine.capability_vector();

            let mut frame = TensorFrame::new();
            let mut pred = SlotData::new(SlotRole::Predicate);
            pred.write_resolution(0, cap);
            frame.write_slot(1, pred).unwrap();
            frame.meta[1].certainty = 0.8;

            let mut inst = SlotData::new(SlotRole::Instrument);
            let mut data = [0.0_f32; SLOT_DIM];
            data[0] = 3.0; // MUL
            data[1] = 847.0;
            data[2] = 392.0;
            inst.write_resolution(0, data);
            frame.write_slot(6, inst).unwrap();
            frame.meta[6].certainty = 0.9;

            let result = layer.process(&frame).unwrap();

            assert!(!result.vetoed);
            let r = result.frame.read_slot(8).unwrap();
            let vals = r.resolutions[0].unwrap();
            assert!(
                (vals[0] - 332_024.0).abs() < 0.01,
                "Expected 332024, got {}",
                vals[0]
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

/// All five axioms independently trigger vetoes.
#[test]
fn all_five_axioms_independently_trigger_veto() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let axioms = default_axioms();

            for (i, axiom) in axioms.iter().enumerate() {
                let mut layer = SafetyLayer::new(default_pipeline());

                let mut frame = TensorFrame::new();
                let mut slot = SlotData::new(SlotRole::Predicate);
                slot.write_resolution(0, axiom.vector);
                frame.write_slot(1, slot).unwrap();

                let result = layer.process(&frame).unwrap();

                if axiom.severity == Severity::Halt {
                    assert!(
                        result.vetoed,
                        "Axiom {} ({}) should trigger halt, but didn't",
                        i,
                        axiom.name
                    );
                }
                // K4 (Warning) doesn't trigger veto, just logs
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

/// K4 (autonomy, Warning) does NOT trigger veto but is scored.
#[test]
fn k4_warning_does_not_trigger_veto() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let axioms = default_axioms();
            let k4_vector = axioms[3].vector;
            let mut layer = SafetyLayer::new(default_pipeline());

            let mut frame = TensorFrame::new();
            let mut slot = SlotData::new(SlotRole::Predicate);
            slot.write_resolution(0, k4_vector);
            frame.write_slot(1, slot).unwrap();

            let result = layer.process(&frame).unwrap();
            assert!(
                !result.vetoed,
                "K4 (Warning) should not trigger veto"
            );
            // But the pre-check score should be non-zero
            assert!(
                result.pre_check_score > 0.0,
                "K4 violation should produce non-zero pre-check score"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

/// The `check()` method works without processing.
#[test]
fn check_without_processing() {
    let axioms = default_axioms();
    let k1_vector = axioms[0].vector;
    let layer = SafetyLayer::new(default_pipeline());

    let mut frame = TensorFrame::new();
    let mut slot = SlotData::new(SlotRole::Predicate);
    slot.write_resolution(0, k1_vector);
    frame.write_slot(1, slot).unwrap();

    let scoring = layer.check(&frame);
    assert_eq!(scoring.level, ViolationLevel::Halt);
    assert!(scoring.aggregate_score > 0.7);
}

/// `safe_process()` top-level convenience function works.
#[test]
fn safe_process_convenience_function() {
    let mut frame = TensorFrame::new();
    let mut slot = SlotData::new(SlotRole::Agent);
    slot.write_resolution(0, [0.1; SLOT_DIM]);
    frame.write_slot(0, slot).unwrap();
    frame.meta[0].certainty = 0.8;

    let result = volt_safety::safe_process(&frame).unwrap();
    assert_eq!(result.active_slot_count(), 1);
}

/// `safe_process()` returns SafetyViolation on K1.
#[test]
fn safe_process_blocks_k1() {
    let axioms = default_axioms();
    let k1_vector = axioms[0].vector;

    let mut frame = TensorFrame::new();
    let mut slot = SlotData::new(SlotRole::Predicate);
    slot.write_resolution(0, k1_vector);
    frame.write_slot(1, slot).unwrap();

    let result = volt_safety::safe_process(&frame);
    assert!(result.is_err());
}

/// TransitionMonitor correctly checks frame transitions.
#[test]
fn transition_monitor_detects_newly_unsafe_frame() {
    let axioms = default_axioms();
    let k1_vector = axioms[0].vector;
    let monitor = TransitionMonitor::new(axioms);

    let prev = TensorFrame::new(); // safe
    let mut next = TensorFrame::new();
    let mut slot = SlotData::new(SlotRole::Predicate);
    slot.write_resolution(0, k1_vector);
    next.write_slot(1, slot).unwrap();

    let result = monitor.check_transition(&prev, &next);
    assert!(!result.is_safe());
    assert!(result.requires_halt());
}
