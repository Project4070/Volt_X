//! Integration tests for volt-hard Milestone 3.1.
//!
//! Tests the full Soft Core -> Intent Router -> MathEngine -> result pipeline.

use volt_core::{SlotData, SlotRole, TensorFrame, SLOT_DIM};
use volt_hard::math_engine::MathEngine;
use volt_hard::router::IntentRouter;
use volt_hard::strand::HardStrand;

/// Helper: build a math frame with capability tagging and operation data.
fn build_math_frame(op: f32, left: f32, right: f32) -> TensorFrame {
    let engine = MathEngine::new();
    let cap = *engine.capability_vector();

    let mut frame = TensorFrame::new();

    // S1 (Predicate): tag with math capability vector for routing
    let mut pred = SlotData::new(SlotRole::Predicate);
    pred.write_resolution(0, cap);
    frame.write_slot(1, pred).unwrap();
    frame.meta[1].certainty = 0.8;

    // S6 (Instrument): math operation
    let mut instrument = SlotData::new(SlotRole::Instrument);
    let mut data = [0.0_f32; SLOT_DIM];
    data[0] = op;
    data[1] = left;
    data[2] = right;
    instrument.write_resolution(0, data);
    frame.write_slot(6, instrument).unwrap();
    frame.meta[6].certainty = 0.9;

    frame
}

/// Helper: build a non-math frame ("Tell me about cats").
fn build_non_math_frame() -> TensorFrame {
    let mut frame = TensorFrame::new();

    // S0 (Agent): user
    let mut agent = SlotData::new(SlotRole::Agent);
    let mut v = [0.0_f32; SLOT_DIM];
    // Deterministic pseudo-random vector (not math-related)
    for i in 0..SLOT_DIM {
        let mut h = 0xCAFE_BABE_u64.wrapping_mul(0xd2b7_4407_b1ce_6e93);
        h = h.wrapping_add(i as u64);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        v[i] = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    agent.write_resolution(0, v);
    frame.write_slot(0, agent).unwrap();
    frame.meta[0].certainty = 0.75;

    // S1 (Predicate): "tell me about"
    let mut pred = SlotData::new(SlotRole::Predicate);
    let mut v2 = [0.0_f32; SLOT_DIM];
    for i in 0..SLOT_DIM {
        let mut h = 0xDEAD_BEEF_u64.wrapping_mul(0xd2b7_4407_b1ce_6e93);
        h = h.wrapping_add(i as u64);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        v2[i] = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
    }
    let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v2 {
        *x /= norm2;
    }
    pred.write_resolution(0, v2);
    frame.write_slot(1, pred).unwrap();
    frame.meta[1].certainty = 0.8;

    frame
}

// ================================================================
// Milestone 3.1 test cases (from PHASE-3.md)
// ================================================================

#[test]
fn milestone_847_x_392_exact_answer() {
    // "What is 847 x 392?" -> MathEngine activates -> exact answer 331,824 -> gamma = 1.0
    let router = volt_hard::default_router();
    let frame = build_math_frame(3.0, 847.0, 392.0); // OP_MUL

    let result = router.route(&frame).unwrap();

    // MathEngine should activate
    assert!(
        result.decisions.iter().any(|d| d.activated),
        "MathEngine should activate for multiplication"
    );

    // Result should be exact
    let r = result.frame.read_slot(8).unwrap();
    let vals = r.resolutions[0].unwrap();
    assert!(
        (vals[0] - 332_024.0).abs() < 1.0,
        "847 * 392 should equal 332024, got {}",
        vals[0]
    );

    // Gamma should be 1.0 for the result slot
    assert_eq!(result.frame.meta[8].certainty, 1.0);
}

#[test]
fn milestone_non_math_passes_through() {
    // "Tell me about cats" -> no Hard Strand activates -> passes through Soft Core only
    let router = volt_hard::default_router();
    let frame = build_non_math_frame();
    let original_count = frame.active_slot_count();

    let result = router.route(&frame).unwrap();

    // No strand should activate
    let activated = result.decisions.iter().any(|d| d.activated);
    assert!(
        !activated,
        "Non-math query should not activate any Hard Strand"
    );

    // Frame should pass through unchanged
    assert_eq!(result.frame.active_slot_count(), original_count);
}

#[test]
fn milestone_router_accuracy_100_cases() {
    // Router correctly distinguishes math queries from non-math queries
    // (>95% accuracy on 100 test cases)
    let router = volt_hard::default_router();
    let engine = MathEngine::new();
    let math_cap = *engine.capability_vector();

    let mut correct = 0;
    let total = 100;

    for i in 0..total {
        let is_math = i % 2 == 0; // 50 math, 50 non-math

        let mut frame = TensorFrame::new();

        if is_math {
            // Math frame: tag with capability vector
            let mut pred = SlotData::new(SlotRole::Predicate);
            pred.write_resolution(0, math_cap);
            frame.write_slot(1, pred).unwrap();
            frame.meta[1].certainty = 0.8;

            let mut inst = SlotData::new(SlotRole::Instrument);
            let mut data = [0.0_f32; SLOT_DIM];
            data[0] = 1.0; // ADD
            data[1] = i as f32;
            data[2] = (i + 1) as f32;
            inst.write_resolution(0, data);
            frame.write_slot(6, inst).unwrap();
            frame.meta[6].certainty = 0.9;
        } else {
            // Non-math frame: random vector
            let mut agent = SlotData::new(SlotRole::Agent);
            let mut v = [0.0_f32; SLOT_DIM];
            for j in 0..SLOT_DIM {
                let seed = (i as u64 * 1000 + j as u64).wrapping_mul(0xBEEF_CAFE);
                let mut h = seed.wrapping_mul(0xd2b7_4407_b1ce_6e93);
                h ^= h >> 33;
                h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
                h ^= h >> 33;
                v[j] = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
            }
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            agent.write_resolution(0, v);
            frame.write_slot(0, agent).unwrap();
            frame.meta[0].certainty = 0.7;
        }

        let result = router.route(&frame).unwrap();
        let activated = result.decisions.iter().any(|d| d.activated);

        if is_math == activated {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / total as f64;
    assert!(
        accuracy > 0.95,
        "Router accuracy should be >95%, got {:.1}% ({correct}/{total})",
        accuracy * 100.0
    );
}

#[test]
fn milestone_math_engine_under_1ms() {
    // MathEngine returns in < 1ms
    let router = volt_hard::default_router();
    let frame = build_math_frame(3.0, 847.0, 392.0);

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = router.route(&frame).unwrap();
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / 100;

    assert!(
        per_call.as_micros() < 1000,
        "Full route + MathEngine should return in < 1ms, got {:?}",
        per_call
    );
}

// ================================================================
// End-to-end integration tests
// ================================================================

#[test]
fn end_to_end_addition() {
    let router = volt_hard::default_router();
    let frame = build_math_frame(1.0, 123.0, 456.0);
    let result = router.route(&frame).unwrap();

    let r = result.frame.read_slot(8).unwrap();
    assert!((r.resolutions[0].unwrap()[0] - 579.0).abs() < 0.01);
}

#[test]
fn end_to_end_division() {
    let router = volt_hard::default_router();
    let frame = build_math_frame(4.0, 100.0, 8.0);
    let result = router.route(&frame).unwrap();

    let r = result.frame.read_slot(8).unwrap();
    assert!((r.resolutions[0].unwrap()[0] - 12.5).abs() < 0.01);
}

#[test]
fn end_to_end_frame_metadata_updated() {
    let router = volt_hard::default_router();
    let frame = build_math_frame(1.0, 1.0, 2.0);
    let result = router.route(&frame).unwrap();

    // Frame should be marked as verified
    assert!(result.frame.frame_meta.verified);

    // Proof length should be at least 1
    assert!(result.frame.frame_meta.proof_length >= 1);

    // Result slot source should be HardCore
    assert_eq!(
        result.frame.meta[8].source,
        volt_core::slot::SlotSource::HardCore
    );
}

#[test]
fn end_to_end_division_by_zero_returns_error() {
    let router = volt_hard::default_router();
    let frame = build_math_frame(4.0, 100.0, 0.0);
    let result = router.route(&frame);

    assert!(result.is_err(), "Division by zero should return error");
}

#[test]
fn end_to_end_global_certainty_min_rule() {
    let router = volt_hard::default_router();
    let frame = build_math_frame(1.0, 10.0, 20.0);
    // Frame has slots with certainty 0.8 (predicate) and 0.9 (instrument)
    // Result slot will have 1.0
    // Global certainty should be min(0.8, 0.9, 1.0) = 0.8

    let result = router.route(&frame).unwrap();
    assert!(
        (result.frame.frame_meta.global_certainty - 0.8).abs() < 0.01,
        "Global certainty should be min of all slots = 0.8, got {}",
        result.frame.frame_meta.global_certainty
    );
}

#[test]
fn default_router_convenience() {
    let router = volt_hard::default_router();
    assert_eq!(router.strand_count(), 1);
}

#[test]
fn custom_strand_implementation() {
    // Verify the HardStrand trait is implementable by external code
    struct NullStrand {
        cap: [f32; SLOT_DIM],
    }

    impl NullStrand {
        fn new() -> Self {
            let mut cap = [0.0_f32; SLOT_DIM];
            cap[0] = 1.0; // simple unit vector
            Self { cap }
        }
    }

    impl HardStrand for NullStrand {
        fn name(&self) -> &str {
            "null"
        }
        fn capability_vector(&self) -> &[f32; SLOT_DIM] {
            &self.cap
        }
        fn threshold(&self) -> f32 {
            0.9
        }
        fn process(
            &self,
            frame: &TensorFrame,
        ) -> Result<volt_hard::strand::StrandResult, volt_core::VoltError> {
            Ok(volt_hard::strand::StrandResult {
                frame: frame.clone(),
                activated: false,
                description: "null: no-op".to_string(),
            })
        }
    }

    let mut router = IntentRouter::new();
    router.register(Box::new(NullStrand::new()));
    router.register(Box::new(MathEngine::new()));
    assert_eq!(router.strand_count(), 2);

    // Route a math frame — MathEngine should win over NullStrand
    let frame = build_math_frame(1.0, 1.0, 1.0);
    let result = router.route(&frame).unwrap();
    let math_activated = result
        .decisions
        .iter()
        .any(|d| d.strand_name == "math_engine" && d.activated);
    assert!(math_activated);
}
