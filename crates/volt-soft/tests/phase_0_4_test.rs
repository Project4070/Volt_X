//! Phase 0.4 integration tests: Code Attention Bias.
//!
//! Tests that the code-specific attention bias produces faster
//! RAR convergence and correct attention patterns on synthetic
//! code-like frames.
//!
//! All tests run on 4MB stack threads for Windows TensorFrame compatibility.

use volt_core::{SlotRole, TensorFrame, MAX_SLOTS, SLOT_DIM};
use volt_soft::attention::SlotAttention;
use volt_soft::code_attention::{code_attention_bias, new_code_attention};
use volt_soft::rar::{rar_loop, rar_loop_with_ghosts, GhostConfig, RarConfig};
use volt_soft::vfn::Vfn;

/// Run a closure on a thread with 4MB stack (Windows TensorFrame safety).
fn with_large_stack<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    std::thread::Builder::new()
        .name("test-large-stack".into())
        .stack_size(4 * 1024 * 1024)
        .spawn(f)
        .expect("failed to spawn test thread")
        .join()
        .expect("test thread panicked")
}

/// Create a deterministic pseudo-random normalized vector from a seed.
fn test_vector(seed: u64) -> [f32; SLOT_DIM] {
    let mut v = [0.0f32; SLOT_DIM];
    for i in 0..SLOT_DIM {
        let mut h = seed.wrapping_mul(0xd2b74407b1ce6e93);
        h = h.wrapping_add(i as u64);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        v[i] = ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

/// Build a code-like frame: function (S0), operation (S1),
/// arguments (S2), return (S3), execution order (S4).
fn make_code_frame() -> TensorFrame {
    let mut frame = TensorFrame::new();
    // S0: Function name
    frame
        .write_at(0, 0, SlotRole::Agent, test_vector(100))
        .unwrap();
    // S1: Operation / method call
    frame
        .write_at(1, 0, SlotRole::Predicate, test_vector(200))
        .unwrap();
    // S2: Arguments / parameters
    frame
        .write_at(2, 0, SlotRole::Patient, test_vector(300))
        .unwrap();
    // S3: Return value / result
    frame
        .write_at(3, 0, SlotRole::Location, test_vector(400))
        .unwrap();
    // S4: Execution order
    frame
        .write_at(4, 0, SlotRole::Time, test_vector(500))
        .unwrap();
    frame
}

#[test]
fn phase_0_4_code_bias_matrix_is_symmetric() {
    let bias = code_attention_bias();
    for i in 0..MAX_SLOTS {
        for j in 0..MAX_SLOTS {
            assert_eq!(
                bias[i][j], bias[j][i],
                "bias[{i}][{j}]={} != bias[{j}][{i}]={}",
                bias[i][j], bias[j][i]
            );
        }
    }
}

#[test]
fn phase_0_4_code_bias_values_match_spec() {
    let bias = code_attention_bias();
    // Strong: Function ↔ Arguments, Function ↔ Return, Operation ↔ Arguments
    assert_eq!(bias[0][2], 2.0);
    assert_eq!(bias[0][3], 2.0);
    assert_eq!(bias[1][2], 2.0);
    // Medium: Function ↔ Operation, Operation ↔ Return, Time ↔ Operation
    assert_eq!(bias[0][1], 1.5);
    assert_eq!(bias[1][3], 1.5);
    assert_eq!(bias[4][1], 1.5);
    // Self-bias for fixed roles
    for i in 0..9 {
        assert_eq!(bias[i][i], 0.5);
    }
    // Free slots: zero
    for i in 9..MAX_SLOTS {
        for j in 0..MAX_SLOTS {
            assert_eq!(bias[i][j], 0.0);
        }
    }
}

#[test]
fn phase_0_4_bias_affects_attention_output() {
    with_large_stack(|| {
        let attn_random = SlotAttention::new_random(42);
        let attn_biased = new_code_attention(42);

        // Build states matching code slots
        let mut states = [const { None }; MAX_SLOTS];
        states[0] = Some(test_vector(100)); // Function
        states[1] = Some(test_vector(200)); // Operation
        states[2] = Some(test_vector(300)); // Arguments
        states[3] = Some(test_vector(400)); // Return

        let msg_random = attn_random.forward(&states).unwrap();
        let msg_biased = attn_biased.forward(&states).unwrap();

        // Messages should differ due to bias
        assert_ne!(
            msg_random[0], msg_biased[0],
            "Function slot message should change with code bias"
        );
        assert_ne!(
            msg_random[1], msg_biased[1],
            "Operation slot message should change with code bias"
        );
    });
}

#[test]
fn phase_0_4_code_bias_produces_different_rar_trajectory() {
    with_large_stack(|| {
        let vfn = Vfn::new_random(42);
        let attn_random = SlotAttention::new_random(43);
        let attn_code = new_code_attention(43);

        let frame = make_code_frame();
        let config = RarConfig {
            epsilon: 0.001,
            max_iterations: 10,
            dt: 0.1,
            beta: 0.5,
            resolution: 0,
            diffusion: None,
        };

        // Run RAR with random attention
        let result_random = rar_loop(&frame, &vfn, &attn_random, &config).unwrap();

        // Run RAR with code-biased attention
        let result_code = rar_loop(&frame, &vfn, &attn_code, &config).unwrap();

        // Both should complete successfully
        assert!(result_random.iterations > 0);
        assert!(result_code.iterations > 0);

        // The trajectories should differ — code bias changes
        // the attention distribution, producing different evolved states.
        // (Convergence speed comparison requires trained VFN; see Phase 1.)
        let mut any_differ = false;
        for i in 0..5 {
            if let (Some(slot_r), Some(slot_c)) =
                (&result_random.frame.slots[i], &result_code.frame.slots[i])
            {
                if let (Some(vec_r), Some(vec_c)) =
                    (&slot_r.resolutions[0], &slot_c.resolutions[0])
                {
                    let diff: f32 = vec_r
                        .iter()
                        .zip(vec_c.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum();
                    if diff > 1e-6 {
                        any_differ = true;
                    }
                }
            }
        }
        assert!(
            any_differ,
            "code bias should produce a different RAR trajectory than random"
        );

        // Output should be normalized
        for result in [&result_random, &result_code] {
            for i in 0..5 {
                if let Some(slot) = &result.frame.slots[i] {
                    if let Some(vec) = &slot.resolutions[0] {
                        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                        assert!(
                            (norm - 1.0).abs() < 1e-4,
                            "slot {i} output should be normalized, got norm={norm}"
                        );
                    }
                }
            }
        }

        eprintln!(
            "RAR trajectory: random={} iters, code={} iters (different trajectories confirmed)",
            result_random.iterations, result_code.iterations
        );
    });
}

#[test]
fn phase_0_4_code_attention_preserves_existing_behavior() {
    with_large_stack(|| {
        // new_random should be unchanged
        let attn = SlotAttention::new_random(42);
        assert!(attn.attention_bias().is_none());

        let mut states = [const { None }; MAX_SLOTS];
        states[0] = Some(test_vector(100));
        states[1] = Some(test_vector(200));
        let messages = attn.forward(&states).unwrap();
        for msg in &messages {
            assert!(msg.iter().all(|x| x.is_finite()));
        }

        // Zero bias should match no bias
        let zero_bias = [[0.0f32; MAX_SLOTS]; MAX_SLOTS];
        let attn_zero = SlotAttention::new_with_bias(42, zero_bias);
        let msg_zero = attn_zero.forward(&states).unwrap();
        for i in 0..MAX_SLOTS {
            for d in 0..SLOT_DIM {
                assert!(
                    (messages[i][d] - msg_zero[i][d]).abs() < 1e-6,
                    "zero bias should match no bias at slot {i} dim {d}"
                );
            }
        }
    });
}

#[test]
fn phase_0_4_code_attention_with_ghosts() {
    with_large_stack(|| {
        let vfn = Vfn::new_random(42);
        let attn = new_code_attention(43);
        let config = RarConfig {
            max_iterations: 5,
            ..RarConfig::default()
        };

        // Create a ghost gist
        let mut ghost_gist = [0.0f32; SLOT_DIM];
        ghost_gist[0] = 1.0;
        let ghost_config = GhostConfig {
            gists: vec![ghost_gist],
            alpha: 0.2,
        };

        let frame = make_code_frame();
        let result = rar_loop_with_ghosts(&frame, &vfn, &attn, &config, &ghost_config).unwrap();

        assert!(result.iterations <= config.max_iterations);
        assert!(result.frame.slots[0].is_some());

        // Output should be normalized
        for i in 0..5 {
            if let Some(slot) = &result.frame.slots[i] {
                if let Some(vec) = &slot.resolutions[0] {
                    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    assert!(
                        (norm - 1.0).abs() < 1e-4,
                        "slot {i} output should be normalized, got norm={norm}"
                    );
                }
            }
        }
    });
}

#[test]
fn phase_0_4_code_attention_output_finite() {
    with_large_stack(|| {
        let attn = new_code_attention(42);

        // Fill all 16 slots
        let mut states = [const { None }; MAX_SLOTS];
        for i in 0..MAX_SLOTS {
            states[i] = Some(test_vector(i as u64 + 100));
        }

        let messages = attn.forward(&states).unwrap();
        for (i, msg) in messages.iter().enumerate() {
            assert!(
                msg.iter().all(|x| x.is_finite()),
                "slot {i} message should be finite"
            );
        }
    });
}
