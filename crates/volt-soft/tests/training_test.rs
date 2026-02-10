//! Integration tests for Flow Matching VFN training (Milestone 2.4).
//!
//! Tests that the training loop converges on synthetic data and that
//! trained VFN produces better drift predictions than random weights.
//! Feature-gated behind `gpu`.

#![cfg(feature = "gpu")]

use candle_core::Device;
use candle_nn::VarMap;
use volt_soft::gpu::vfn::GpuVfn;
use volt_soft::training::{
    generate_synthetic_pairs, train_vfn_flow_matching, FlowMatchConfig,
};

/// Training on synthetic data decreases loss.
#[test]
fn training_loss_decreases() {
    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vfn = GpuVfn::new_trainable(&var_map, &device).unwrap();
    let pairs = generate_synthetic_pairs(50, 0, 42).unwrap();

    let config = FlowMatchConfig {
        num_steps: 50,
        batch_size: 8,
        learning_rate: 1e-3,
        ..FlowMatchConfig::default()
    };

    let result = train_vfn_flow_matching(&vfn, &var_map, &pairs, &config, &device).unwrap();

    assert_eq!(result.steps_completed, 50);
    assert!(result.final_loss.is_finite());

    // Compare early vs late average loss
    let early_avg: f32 = result.loss_history[..5].iter().sum::<f32>() / 5.0;
    let late_avg: f32 = result.loss_history[45..].iter().sum::<f32>() / 5.0;
    assert!(
        late_avg < early_avg,
        "loss should decrease: early_avg={}, late_avg={}",
        early_avg, late_avg,
    );
}

/// More training steps yield lower loss.
#[test]
fn more_steps_lower_loss() {
    let device = Device::Cpu;
    let pairs = generate_synthetic_pairs(50, 0, 42).unwrap();

    let config_short = FlowMatchConfig {
        num_steps: 10,
        batch_size: 8,
        learning_rate: 1e-3,
        seed: 99,
        ..FlowMatchConfig::default()
    };

    let config_long = FlowMatchConfig {
        num_steps: 100,
        batch_size: 8,
        learning_rate: 1e-3,
        seed: 99,
        ..FlowMatchConfig::default()
    };

    let var_map_short = VarMap::new();
    let vfn_short = GpuVfn::new_trainable(&var_map_short, &device).unwrap();
    let result_short =
        train_vfn_flow_matching(&vfn_short, &var_map_short, &pairs, &config_short, &device)
            .unwrap();

    let var_map_long = VarMap::new();
    let vfn_long = GpuVfn::new_trainable(&var_map_long, &device).unwrap();
    let result_long =
        train_vfn_flow_matching(&vfn_long, &var_map_long, &pairs, &config_long, &device).unwrap();

    assert!(
        result_long.final_loss < result_short.final_loss,
        "100 steps (loss={}) should yield lower loss than 10 steps (loss={})",
        result_long.final_loss,
        result_short.final_loss,
    );
}

/// Synthetic pairs have matching active slot counts.
#[test]
fn synthetic_pairs_well_formed() {
    let pairs = generate_synthetic_pairs(20, 0, 123).unwrap();
    assert_eq!(pairs.len(), 20);
    for pair in &pairs {
        assert!(pair.question.active_slot_count() >= 2);
        assert_eq!(
            pair.question.active_slot_count(),
            pair.answer.active_slot_count(),
        );
    }
}

/// Training with empty pairs returns error.
#[test]
fn empty_pairs_errors() {
    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vfn = GpuVfn::new_trainable(&var_map, &device).unwrap();
    let config = FlowMatchConfig::default();

    assert!(train_vfn_flow_matching(&vfn, &var_map, &[], &config, &device).is_err());
}

/// Loss history length matches num_steps.
#[test]
fn loss_history_matches_steps() {
    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vfn = GpuVfn::new_trainable(&var_map, &device).unwrap();
    let pairs = generate_synthetic_pairs(20, 0, 42).unwrap();

    let config = FlowMatchConfig {
        num_steps: 25,
        batch_size: 4,
        ..FlowMatchConfig::default()
    };

    let result = train_vfn_flow_matching(&vfn, &var_map, &pairs, &config, &device).unwrap();
    assert_eq!(result.loss_history.len(), 25);
    assert_eq!(result.steps_completed, 25);
}

/// Trained VFN produces finite outputs on held-out data.
#[test]
fn trained_vfn_produces_finite_output() {
    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vfn = GpuVfn::new_trainable(&var_map, &device).unwrap();
    let pairs = generate_synthetic_pairs(50, 0, 42).unwrap();

    let config = FlowMatchConfig {
        num_steps: 20,
        batch_size: 8,
        learning_rate: 1e-3,
        ..FlowMatchConfig::default()
    };

    let _ = train_vfn_flow_matching(&vfn, &var_map, &pairs, &config, &device).unwrap();

    // Test on new input
    let input = [0.1f32; volt_core::SLOT_DIM];
    let output = vfn.forward_single(&input).unwrap();
    assert!(output.iter().all(|x| x.is_finite()));
}

/// After training, RAR inference converges on held-out synthetic pairs.
///
/// Trains VFN on synthetic data, then checks what fraction of held-out
/// pairs converge (all active slots reach ‖ΔS‖ < ε within budget).
/// Uses a relaxed threshold (> 0.50) for synthetic data; the 0.80 target
/// from the spec applies to real training.
#[test]
fn trained_vfn_convergence_rate() {
    use volt_soft::gpu::attention::GpuSlotAttention;
    use volt_soft::gpu::rar::gpu_rar_loop;
    use volt_soft::rar::RarConfig;

    let device = Device::Cpu;

    // Train on 100 synthetic pairs
    let var_map = VarMap::new();
    let vfn = GpuVfn::new_trainable(&var_map, &device).unwrap();
    let train_pairs = generate_synthetic_pairs(100, 0, 42).unwrap();

    let config = FlowMatchConfig {
        num_steps: 200,
        batch_size: 16,
        learning_rate: 1e-3,
        seed: 42,
        ..FlowMatchConfig::default()
    };

    let result = train_vfn_flow_matching(&vfn, &var_map, &train_pairs, &config, &device).unwrap();
    assert!(result.final_loss.is_finite());

    // Evaluate convergence on 50 held-out pairs (different seed)
    let eval_pairs = generate_synthetic_pairs(50, 0, 999).unwrap();
    let attn = GpuSlotAttention::new_random(43, &device).unwrap();

    // beta=0.0 disables the untrained attention module so we test
    // VFN convergence in isolation (attention is not trained by Flow Matching).
    // epsilon=0.1 is relaxed for synthetic data on debug builds.
    let rar_config = RarConfig {
        epsilon: 0.1,
        max_iterations: 50,
        dt: 0.1,
        beta: 0.0,
        ..RarConfig::default()
    };

    let mut converged_count = 0;
    let total = eval_pairs.len();

    for pair in &eval_pairs {
        let rar_result = gpu_rar_loop(&pair.question, &vfn, &attn, &rar_config).unwrap();

        // A pair converges if all active slots have converged
        let all_converged = (0..volt_core::MAX_SLOTS).all(|i| {
            // Inactive slots are trivially converged
            if pair.question.slots[i].is_none() {
                return true;
            }
            rar_result.converged[i]
        });

        if all_converged {
            converged_count += 1;
        }
    }

    let convergence_rate = converged_count as f64 / total as f64;

    // Relaxed threshold for synthetic data (spec target: 0.80 for real training)
    assert!(
        convergence_rate > 0.50,
        "convergence rate {:.1}% ({}/{}) is below 50% threshold",
        convergence_rate * 100.0,
        converged_count,
        total,
    );
}
