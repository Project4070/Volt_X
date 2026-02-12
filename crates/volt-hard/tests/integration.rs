//! Integration tests for volt-hard Milestone 3.1 & 3.2.
//!
//! Tests the full Hard Core pipeline: Intent Router -> Strand Execution ->
//! CertaintyEngine -> ProofConstructor.

use volt_core::{SlotData, SlotRole, TensorFrame, SLOT_DIM};
use volt_hard::math_engine::MathEngine;
use volt_hard::router::IntentRouter;
use volt_hard::strand::HardStrand;

/// Stack size for tests that allocate multiple TensorFrames.
const TEST_STACK: usize = 4 * 1024 * 1024;

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

/// Build a normalized pseudo-random vector from a seed.
fn seeded_vector(seed: u64) -> [f32; SLOT_DIM] {
    let mut v = [0.0_f32; SLOT_DIM];
    for (i, val) in v.iter_mut().enumerate() {
        let mut h = seed.wrapping_mul(0xd2b7_4407_b1ce_6e93);
        h = h.wrapping_add(i as u64);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        *val = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

// ================================================================
// Milestone 3.1 test cases (from PHASE-3.md)
// ================================================================

#[test]
fn milestone_847_x_392_exact_answer() {
    // TensorFrame + catch_unwind overhead requires bigger stack on Windows.
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            // "What is 847 x 392?" -> MathEngine activates -> exact answer 332,024 -> gamma = 1.0
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
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn milestone_non_math_passes_through() {
    // TensorFrame + catch_unwind overhead requires bigger stack on Windows.
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
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
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn milestone_router_accuracy_100_cases() {
    // TensorFrame + catch_unwind overhead requires bigger stack on Windows.
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
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
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn milestone_math_engine_under_1ms() {
    // TensorFrame + catch_unwind overhead requires bigger stack on Windows.
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
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
        })
        .unwrap()
        .join()
        .unwrap();
}

// ================================================================
// End-to-end integration tests (Milestone 3.1)
// ================================================================

#[test]
fn end_to_end_addition() {
    // TensorFrame + catch_unwind overhead requires bigger stack on Windows.
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let router = volt_hard::default_router();
            let frame = build_math_frame(1.0, 123.0, 456.0);
            let result = router.route(&frame).unwrap();

            let r = result.frame.read_slot(8).unwrap();
            assert!((r.resolutions[0].unwrap()[0] - 579.0).abs() < 0.01);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn end_to_end_division() {
    // TensorFrame + catch_unwind overhead requires bigger stack on Windows.
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let router = volt_hard::default_router();
            let frame = build_math_frame(4.0, 100.0, 8.0);
            let result = router.route(&frame).unwrap();

            let r = result.frame.read_slot(8).unwrap();
            assert!((r.resolutions[0].unwrap()[0] - 12.5).abs() < 0.01);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn end_to_end_frame_metadata_updated() {
    // TensorFrame + catch_unwind overhead requires bigger stack on Windows.
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
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
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn end_to_end_division_by_zero_returns_error() {
    // TensorFrame + catch_unwind overhead requires bigger stack on Windows.
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let router = volt_hard::default_router();
            let frame = build_math_frame(4.0, 100.0, 0.0);
            let result = router.route(&frame);

            assert!(result.is_err(), "Division by zero should return error");
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn end_to_end_global_certainty_min_rule() {
    // TensorFrame + catch_unwind overhead requires bigger stack on Windows.
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
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
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn default_router_convenience() {
    let router = volt_hard::default_router();
    // Milestone 3.2: at least MathEngine + HDCAlgebra
    assert!(
        router.strand_count() >= 2,
        "default_router should have >= 2 strands, got {}",
        router.strand_count()
    );
}

#[test]
fn custom_strand_implementation() {
    // TensorFrame + catch_unwind overhead requires bigger stack on Windows.
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
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
        })
        .unwrap()
        .join()
        .unwrap();
}

// ================================================================
// Milestone 3.2 test cases (from PHASE-3.md)
// ================================================================

#[test]
fn milestone_certainty_engine_min_rule() {
    // CertaintyEngine: frame with gamma=[1.0, 0.8, 0.6] -> global gamma = 0.6
    use volt_hard::certainty_engine::CertaintyEngine;

    let engine = CertaintyEngine::new();
    let mut frame = TensorFrame::new();

    let mut s0 = SlotData::new(SlotRole::Agent);
    s0.write_resolution(0, [0.5; SLOT_DIM]);
    frame.write_slot(0, s0).unwrap();
    frame.meta[0].certainty = 1.0;

    let mut s1 = SlotData::new(SlotRole::Predicate);
    s1.write_resolution(0, [0.3; SLOT_DIM]);
    frame.write_slot(1, s1).unwrap();
    frame.meta[1].certainty = 0.8;

    let mut s2 = SlotData::new(SlotRole::Patient);
    s2.write_resolution(0, [0.7; SLOT_DIM]);
    frame.write_slot(2, s2).unwrap();
    frame.meta[2].certainty = 0.6;

    let result = engine.propagate(&mut frame);

    assert!(
        (result.global_certainty - 0.6).abs() < 0.01,
        "CertaintyEngine: gamma=[1.0, 0.8, 0.6] should give global=0.6, got {}",
        result.global_certainty
    );
    assert_eq!(frame.frame_meta.global_certainty, 0.6);
}

#[test]
fn milestone_proof_chain_has_steps() {
    // ProofConstructor: after processing, proof chain has >= 2 steps,
    // each with source and gamma
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let pipeline = volt_hard::default_pipeline();
            let frame = build_math_frame(1.0, 10.0, 20.0); // ADD

            let result = pipeline.process(&frame).unwrap();

            // Should have >= 2 steps
            assert!(
                result.proof.len() >= 2,
                "Proof chain should have >= 2 steps, got {}",
                result.proof.len()
            );

            // Each step should have source (strand_name) and gamma
            for step in &result.proof.steps {
                assert!(
                    !step.strand_name.is_empty(),
                    "Each proof step must have a source (strand_name)"
                );
                assert!(
                    step.gamma_after >= 0.0 && step.gamma_after <= 1.0,
                    "Each proof step must have valid gamma, got {}",
                    step.gamma_after
                );
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn milestone_code_runner_computes_2_plus_2() {
    // CodeRunner: print(2+2) -> output "4" in sandboxed environment
    #[cfg(feature = "sandbox")]
    {
        use volt_hard::code_runner::CodeRunner;

        let runner = CodeRunner::new().unwrap();

        // WAT module: computes 2+2=4, writes '4' (ASCII 52) to memory
        let wat = r#"(module
            (memory (export "memory") 1)
            (func (export "run") (result i32)
                (i32.store (i32.const 0) (i32.const 1))
                (i32.store8 (i32.const 4) (i32.const 52))
                (i32.add (i32.const 2) (i32.const 2))
            )
        )"#;

        let mut frame = TensorFrame::new();
        let mut instrument = SlotData::new(SlotRole::Instrument);

        let mut r0 = [0.0_f32; SLOT_DIM];
        r0[0] = 10.0; // OP_CODE_RUN
        instrument.write_resolution(0, r0);

        // Encode WAT bytes
        let bytes = wat.as_bytes();
        for (res_idx, chunk_start) in [0usize, SLOT_DIM, SLOT_DIM * 2].iter().enumerate() {
            let mut data = [0.0_f32; SLOT_DIM];
            for i in 0..SLOT_DIM {
                let byte_idx = chunk_start + i;
                if byte_idx < bytes.len() {
                    data[i] = bytes[byte_idx] as f32;
                }
            }
            instrument.write_resolution(res_idx + 1, data);
        }

        frame.write_slot(6, instrument).unwrap();
        frame.meta[6].certainty = 1.0;

        let result = runner.process(&frame).unwrap();
        assert!(result.activated, "CodeRunner should activate for code request");

        let r = result.frame.read_slot(8).unwrap();
        let vals = r.resolutions[0].unwrap();

        // Return value should be 4
        assert!(
            (vals[0] - 4.0).abs() < 0.01,
            "CodeRunner: 2+2 should return 4, got {}",
            vals[0]
        );

        // Stdout should contain '4' (ASCII 52)
        let stdout = r.resolutions[2].unwrap();
        assert!(
            (stdout[0] - 52.0).abs() < 0.01,
            "CodeRunner: stdout should contain ASCII '4' (52), got {}",
            stdout[0]
        );
    }
}

#[test]
fn milestone_code_runner_blocks_malicious() {
    // Malicious code (file access, network) -> blocked
    #[cfg(feature = "sandbox")]
    {
        use volt_hard::code_runner::CodeRunner;

        let runner = CodeRunner::new().unwrap();

        // WAT module trying WASI import (fd_write for file access)
        let wat = r#"(module
            (import "wasi_snapshot_preview1" "fd_write"
                (func $fd_write (param i32 i32 i32 i32) (result i32)))
            (func (export "run") (result i32) (i32.const 0))
        )"#;

        let mut frame = TensorFrame::new();
        let mut instrument = SlotData::new(SlotRole::Instrument);

        let mut r0 = [0.0_f32; SLOT_DIM];
        r0[0] = 10.0;
        instrument.write_resolution(0, r0);

        let bytes = wat.as_bytes();
        for (res_idx, chunk_start) in [0usize, SLOT_DIM, SLOT_DIM * 2].iter().enumerate() {
            let mut data = [0.0_f32; SLOT_DIM];
            for i in 0..SLOT_DIM {
                let byte_idx = chunk_start + i;
                if byte_idx < bytes.len() {
                    data[i] = bytes[byte_idx] as f32;
                }
            }
            instrument.write_resolution(res_idx + 1, data);
        }

        frame.write_slot(6, instrument).unwrap();
        frame.meta[6].certainty = 1.0;

        let result = runner.process(&frame);
        assert!(
            result.is_err(),
            "CodeRunner: WASI imports should be blocked in sandbox"
        );
    }
}

#[test]
fn milestone_hdc_algebra_bind_via_pipeline() {
    // HDCAlgebra: bind(S0, S2) via pipeline matches direct volt_bus::bind
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            use volt_hard::hdc_algebra::HDCAlgebra;

            let algebra = HDCAlgebra::new();
            let cap = *algebra.capability_vector();

            let vec_a = seeded_vector(0xABCD);
            let vec_b = seeded_vector(0xEF01);

            let mut frame = TensorFrame::new();

            // S0: source vector A
            let mut s0 = SlotData::new(SlotRole::Agent);
            s0.write_resolution(0, vec_a);
            frame.write_slot(0, s0).unwrap();
            frame.meta[0].certainty = 0.9;

            // S1: tag with HDCAlgebra capability for routing
            let mut pred = SlotData::new(SlotRole::Predicate);
            pred.write_resolution(0, cap);
            frame.write_slot(1, pred).unwrap();
            frame.meta[1].certainty = 0.85;

            // S2: source vector B
            let mut s2 = SlotData::new(SlotRole::Patient);
            s2.write_resolution(0, vec_b);
            frame.write_slot(2, s2).unwrap();
            frame.meta[2].certainty = 0.88;

            // S6 (Instrument): bind(S0, S2)
            let mut inst = SlotData::new(SlotRole::Instrument);
            let mut data = [0.0_f32; SLOT_DIM];
            data[0] = 11.0; // OP_HDC_BIND
            data[1] = 0.0; // slot A = S0
            data[2] = 2.0; // slot B = S2
            inst.write_resolution(0, data);
            frame.write_slot(6, inst).unwrap();
            frame.meta[6].certainty = 1.0;

            // Process through pipeline
            let pipeline = volt_hard::default_pipeline();
            let result = pipeline.process(&frame).unwrap();

            // Verify result matches direct volt_bus::bind
            let expected = volt_bus::bind(&vec_a, &vec_b).unwrap();
            let actual = result.frame.read_slot(8).unwrap();
            let actual_vec = actual.resolutions[0].unwrap();

            let sim = volt_bus::similarity(&expected, &actual_vec);
            assert!(
                sim > 0.99,
                "HDCAlgebra bind via pipeline should match volt_bus::bind, sim = {sim}"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn pipeline_end_to_end_math_with_proof() {
    // Full pipeline on 847*392: verify result + proof + certainty
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let pipeline = volt_hard::default_pipeline();
            let frame = build_math_frame(3.0, 847.0, 392.0); // MUL

            let result = pipeline.process(&frame).unwrap();

            // Verify exact result
            let r = result.frame.read_slot(8).unwrap();
            let vals = r.resolutions[0].unwrap();
            assert!(
                (vals[0] - 332_024.0).abs() < 1.0,
                "Pipeline: 847 * 392 should equal 332024, got {}",
                vals[0]
            );

            // Verify proof chain
            assert!(
                result.proof.len() >= 2,
                "Pipeline: proof chain should have >= 2 steps, got {}",
                result.proof.len()
            );
            assert!(
                result.proof.activated_count >= 1,
                "Pipeline: at least 1 strand should activate"
            );

            // Verify certainty propagation
            assert!(
                (result.frame.frame_meta.global_certainty - 0.8).abs() < 0.01,
                "Pipeline: global certainty should be 0.8 (min), got {}",
                result.frame.frame_meta.global_certainty
            );
            assert!(
                (result.proof.final_gamma - 0.8).abs() < 0.01,
                "Pipeline: proof final_gamma should be 0.8, got {}",
                result.proof.final_gamma
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn pipeline_non_activation_still_records_proof() {
    // Non-math frame through pipeline: no strand activates but
    // certainty propagation still recorded in proof
    std::thread::Builder::new()
        .stack_size(TEST_STACK)
        .spawn(|| {
            let pipeline = volt_hard::default_pipeline();
            let frame = build_non_math_frame();

            let result = pipeline.process(&frame).unwrap();

            // Proof should have at least 1 step (certainty propagation)
            assert!(
                !result.proof.is_empty(),
                "Even non-activated frames should have proof steps"
            );

            // Last step should be certainty engine
            let last = result.proof.steps.last().unwrap();
            assert_eq!(last.strand_name, "certainty_engine");
        })
        .unwrap()
        .join()
        .unwrap();
}
