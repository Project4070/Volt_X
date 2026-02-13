//! Integration tests for Phase 2 components.
//!
//! Tests the ScaledVfn, LearnedDiffusionController, and (with gpu feature)
//! the Phase 2 training pipeline.

use volt_core::{MAX_SLOTS, SLOT_DIM};
use volt_soft::learned_diffusion::{DiffusionController, DiffusionControllerConfig};
use volt_soft::scaled_vfn::{ScaledVfn, ScaledVfnConfig};

fn small_config() -> ScaledVfnConfig {
    ScaledVfnConfig {
        hidden_dim: 64,
        num_blocks: 2,
        io_dim: SLOT_DIM,
    }
}

// --- ScaledVfn Integration Tests ---

#[test]
fn scaled_vfn_forward_produces_valid_drift() {
    let config = small_config();
    let vfn = ScaledVfn::new_random(42, &config);

    // Generate various inputs and check outputs
    for seed in 0..10 {
        let mut input = [0.0f32; SLOT_DIM];
        let mut rng_val = seed as f32 * 0.1 + 0.01;
        for x in &mut input {
            *x = rng_val;
            rng_val = (rng_val * 1.1 + 0.07) % 2.0 - 1.0;
        }
        // Normalize
        let norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut input {
            *x /= norm;
        }

        let drift = vfn.forward(&input).unwrap();
        assert_eq!(drift.len(), SLOT_DIM);
        assert!(
            drift.iter().all(|x| x.is_finite()),
            "drift contains NaN/Inf for seed {seed}"
        );
    }
}

#[test]
fn scaled_vfn_different_inputs_different_drifts() {
    let config = small_config();
    let vfn = ScaledVfn::new_random(42, &config);

    let input1 = [0.1f32; SLOT_DIM];
    let input2 = [0.2f32; SLOT_DIM];

    let drift1 = vfn.forward(&input1).unwrap();
    let drift2 = vfn.forward(&input2).unwrap();

    // Drifts should differ for different inputs
    let diff: f32 = drift1
        .iter()
        .zip(drift2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "different inputs should produce different drifts"
    );
}

#[test]
fn scaled_vfn_checkpoint_roundtrip() {
    let config = small_config();
    let vfn = ScaledVfn::new_random(42, &config);

    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("svfn_integ_test.bin");

    // Save
    vfn.save(&path).unwrap();

    // Load
    let loaded = ScaledVfn::load(&path).unwrap();

    // Verify identical output
    let input = [0.42f32; SLOT_DIM];
    let out1 = vfn.forward(&input).unwrap();
    let out2 = loaded.forward(&input).unwrap();

    for (i, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
        assert_eq!(*a, *b, "dim {i}: original={a}, loaded={b}");
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn scaled_vfn_param_count_correct() {
    // Default config should be ~50M params
    let config = ScaledVfnConfig::default();
    let count = config.param_count();
    assert!(
        count > 40_000_000 && count < 60_000_000,
        "default config should have ~50M params, got {count}"
    );

    // Small config should have much fewer
    let small = small_config();
    assert!(small.param_count() < 100_000);
}

// --- LearnedDiffusionController Integration Tests ---

#[test]
fn diffusion_controller_predicts_valid_sigma() {
    let controller = DiffusionController::new_random(42);
    let config = DiffusionControllerConfig::default();

    // Test across various states and iterations
    for iter in [0, 5, 10, 25, 49] {
        for delta in [0.001, 0.01, 0.1, 1.0] {
            let state = [0.1f32; SLOT_DIM];
            let sigma = controller
                .predict_sigma(&state, iter, delta, &config)
                .unwrap();
            assert!(
                sigma >= 0.0 && sigma <= config.max_sigma,
                "sigma={sigma} out of range for iter={iter}, delta={delta}"
            );
        }
    }
}

#[test]
fn diffusion_controller_predict_all_respects_masks() {
    let controller = DiffusionController::new_random(42);
    let config = DiffusionControllerConfig::default();

    let mut states = [const { None }; MAX_SLOTS];
    states[0] = Some([0.1f32; SLOT_DIM]);
    states[3] = Some([0.2f32; SLOT_DIM]);
    states[7] = Some([0.3f32; SLOT_DIM]);

    let mut converged = [true; MAX_SLOTS];
    converged[0] = false;
    converged[3] = false;
    converged[7] = true; // Slot 7 is converged

    let deltas = [0.05; MAX_SLOTS];

    let sigmas = controller
        .predict_all(&states, &converged, &deltas, 10, &config)
        .unwrap();

    // Active non-converged slots should have sigma
    // (could be 0 due to network output, but should be valid)
    assert!(sigmas[0] >= 0.0);
    assert!(sigmas[3] >= 0.0);

    // Converged and empty slots must be 0
    assert_eq!(sigmas[7], 0.0); // converged
    assert_eq!(sigmas[1], 0.0); // empty
    assert_eq!(sigmas[15], 0.0); // empty
}

#[test]
fn diffusion_controller_checkpoint_roundtrip() {
    let controller = DiffusionController::new_random(42);

    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("dc_integ_test.bin");

    controller.save(&path).unwrap();
    let loaded = DiffusionController::load(&path).unwrap();

    let state = [0.42f32; SLOT_DIM];
    let config = DiffusionControllerConfig::default();

    let s1 = controller.predict_sigma(&state, 10, 0.1, &config).unwrap();
    let s2 = loaded.predict_sigma(&state, 10, 0.1, &config).unwrap();

    assert_eq!(s1, s2, "checkpoint roundtrip must preserve weights");

    let _ = std::fs::remove_file(&path);
}

// --- GPU-gated Phase 2 Training Integration Tests ---

#[cfg(feature = "gpu")]
mod gpu_tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;
    use volt_soft::gpu::scaled_vfn::GpuScaledVfn;
    use volt_soft::training::flow_matching::generate_synthetic_pairs;
    use volt_soft::training::phase2::{
        train_phase2, CurriculumStage, Phase2Config, Phase2Result,
    };

    fn small_phase2_config() -> Phase2Config {
        Phase2Config {
            vfn_config: small_config(),
            num_steps: 20,
            batch_size: 4,
            learning_rate: 1e-3,
            validation_interval: 10,
            validation_pairs: 5,
            warmup_steps: 3,
            ..Phase2Config::default()
        }
    }

    #[test]
    fn gpu_scaled_vfn_matches_cpu() {
        let config = small_config();
        let cpu_vfn = ScaledVfn::new_random(42, &config);
        let device = Device::Cpu;
        let gpu_vfn = GpuScaledVfn::from_cpu(&cpu_vfn, &device).unwrap();

        let input = [0.1f32; SLOT_DIM];
        let cpu_out = cpu_vfn.forward(&input).unwrap();
        let gpu_out = gpu_vfn.forward_single(&input).unwrap();

        for (i, (a, b)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "dim {i}: cpu={a}, gpu={b}"
            );
        }
    }

    #[test]
    fn phase2_training_completes() {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let pairs = generate_synthetic_pairs(50, 0, 42).unwrap();
        let val_pairs = generate_synthetic_pairs(10, 0, 99).unwrap();
        let config = small_phase2_config();

        let result = train_phase2(&var_map, &pairs, &val_pairs, &config, &device).unwrap();

        assert_eq!(result.steps_completed, 20);
        assert!(result.final_loss.is_finite());
        assert!(!result.validation_metrics.is_empty());
    }

    #[test]
    fn phase2_loss_decreases_over_training() {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let pairs = generate_synthetic_pairs(100, 0, 42).unwrap();
        let config = Phase2Config {
            num_steps: 50,
            batch_size: 8,
            learning_rate: 5e-4,
            validation_interval: 0,
            ..small_phase2_config()
        };

        let result = train_phase2(&var_map, &pairs, &[], &config, &device).unwrap();

        let early_avg: f32 = result.loss_history[..5].iter().sum::<f32>() / 5.0;
        let late_avg: f32 = result.loss_history[45..].iter().sum::<f32>() / 5.0;
        assert!(
            late_avg < early_avg,
            "loss should decrease: early={early_avg:.4}, late={late_avg:.4}"
        );
    }

    #[test]
    fn phase2_curriculum_stages_work() {
        for stage in [
            CurriculumStage::SimpleFunctions,
            CurriculumStage::LoopsAndConditionals,
            CurriculumStage::MultiFunctionPrograms,
            CurriculumStage::AlgorithmicReasoning,
        ] {
            let device = Device::Cpu;
            let var_map = VarMap::new();
            let pairs = generate_synthetic_pairs(20, 0, 42).unwrap();
            let config = Phase2Config {
                stage,
                num_steps: 5,
                batch_size: 4,
                validation_interval: 0,
                ..small_phase2_config()
            };

            let result = train_phase2(&var_map, &pairs, &[], &config, &device).unwrap();
            assert_eq!(result.stage, stage);
            assert!(result.final_loss.is_finite());
        }
    }

    #[test]
    fn phase2_validation_metrics_sensible() {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let pairs = generate_synthetic_pairs(50, 0, 42).unwrap();
        let val_pairs = generate_synthetic_pairs(10, 0, 99).unwrap();
        let config = Phase2Config {
            num_steps: 20,
            validation_interval: 10,
            ..small_phase2_config()
        };

        let result = train_phase2(&var_map, &pairs, &val_pairs, &config, &device).unwrap();

        assert_eq!(result.validation_metrics.len(), 2); // steps 10, 20
        for m in &result.validation_metrics {
            assert!(m.avg_mse_loss.is_finite());
            assert!(m.avg_cosine_similarity.is_finite());
            assert!(m.directional_accuracy >= 0.0 && m.directional_accuracy <= 1.0);
        }
    }
}
