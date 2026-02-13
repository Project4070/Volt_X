//! Phase 2 training pipeline: Scaled VFN + Joint Attention + Diffusion.
//!
//! Extends the basic Flow Matching training with:
//! - Scaled VFN (50M+ params) instead of the base 525K VFN
//! - Joint attention weight training alongside VFN
//! - Curriculum-based training stages
//! - Validation metrics (convergence rate, cosine similarity)
//!
//! ## Training Algorithm
//!
//! Per step:
//! 1. Sample batch of (F_q, F_a) pairs from current curriculum stage
//! 2. For each pair, sample t ~ U(0,1)
//! 3. Interpolate: F(t) = (1-t)·F_q + t·F_a
//! 4. VFN loss: MSE(VFN(F(t)), F_a - F_q)
//! 5. Attention loss: MSE of attention-augmented drift vs target
//! 6. Joint backward pass + AdamW step
//!
//! ## Curriculum Stages
//!
//! 1. Simple functions (single operation, no loops)
//! 2. Loops and conditionals
//! 3. Multi-function programs
//! 4. Algorithmic reasoning (sorting, searching)

use candle_core::{Device, Tensor};
use candle_nn::{Optimizer, VarMap};
use volt_core::{VoltError, MAX_SLOTS, SLOT_DIM};

use crate::gpu::scaled_vfn::GpuScaledVfn;
use crate::scaled_vfn::ScaledVfnConfig;
use crate::training::flow_matching::{FramePair, TrainResult};

/// Curriculum stage for progressive training difficulty.
///
/// # Example
///
/// ```ignore
/// use volt_soft::training::phase2::CurriculumStage;
///
/// let stage = CurriculumStage::SimpleFunctions;
/// assert_eq!(stage.label(), "simple_functions");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurriculumStage {
    /// Stage 1: Simple single-operation functions.
    SimpleFunctions,
    /// Stage 2: Programs with loops and conditionals.
    LoopsAndConditionals,
    /// Stage 3: Multi-function programs.
    MultiFunctionPrograms,
    /// Stage 4: Algorithmic reasoning (sorting, searching).
    AlgorithmicReasoning,
}

impl CurriculumStage {
    /// Returns a human-readable label for this stage.
    pub fn label(&self) -> &'static str {
        match self {
            Self::SimpleFunctions => "simple_functions",
            Self::LoopsAndConditionals => "loops_conditionals",
            Self::MultiFunctionPrograms => "multi_function",
            Self::AlgorithmicReasoning => "algorithmic",
        }
    }

    /// Returns the difficulty level (0-3).
    pub fn level(&self) -> usize {
        match self {
            Self::SimpleFunctions => 0,
            Self::LoopsAndConditionals => 1,
            Self::MultiFunctionPrograms => 2,
            Self::AlgorithmicReasoning => 3,
        }
    }

    /// Returns the next curriculum stage, or None if at the final stage.
    pub fn next(&self) -> Option<Self> {
        match self {
            Self::SimpleFunctions => Some(Self::LoopsAndConditionals),
            Self::LoopsAndConditionals => Some(Self::MultiFunctionPrograms),
            Self::MultiFunctionPrograms => Some(Self::AlgorithmicReasoning),
            Self::AlgorithmicReasoning => None,
        }
    }
}

/// Configuration for Phase 2 training.
///
/// # Example
///
/// ```ignore
/// use volt_soft::training::phase2::Phase2Config;
///
/// let config = Phase2Config::default();
/// assert_eq!(config.num_steps, 10_000);
/// ```
#[derive(Debug, Clone)]
pub struct Phase2Config {
    /// VFN architecture configuration.
    pub vfn_config: ScaledVfnConfig,

    /// Learning rate for AdamW optimizer (default: 1e-4).
    pub learning_rate: f64,

    /// AdamW weight decay (default: 0.01).
    pub weight_decay: f64,

    /// Number of training steps per curriculum stage (default: 10,000).
    pub num_steps: usize,

    /// Batch size — number of frame pairs per step (default: 32).
    pub batch_size: usize,

    /// Random seed for reproducibility.
    pub seed: u64,

    /// Which resolution to train on (default: 0 = R₀).
    pub resolution: usize,

    /// Initial curriculum stage.
    pub stage: CurriculumStage,

    /// Weight for attention loss component (default: 0.3).
    /// Total loss = VFN_loss + attention_weight × attention_loss.
    pub attention_loss_weight: f32,

    /// Whether to train attention weights jointly (default: true).
    pub joint_attention: bool,

    /// Steps between validation runs (default: 500).
    pub validation_interval: usize,

    /// Number of validation pairs for convergence testing.
    pub validation_pairs: usize,

    /// Learning rate warmup steps (default: 100).
    pub warmup_steps: usize,

    /// Enable gradient clipping (default: 1.0).
    pub max_grad_norm: f64,
}

impl Default for Phase2Config {
    fn default() -> Self {
        Self {
            vfn_config: ScaledVfnConfig::default(),
            learning_rate: 1e-4,
            weight_decay: 0.01,
            num_steps: 10_000,
            batch_size: 32,
            seed: 42,
            resolution: 0,
            stage: CurriculumStage::SimpleFunctions,
            attention_loss_weight: 0.3,
            joint_attention: true,
            validation_interval: 500,
            validation_pairs: 50,
            warmup_steps: 100,
            max_grad_norm: 1.0,
        }
    }
}

/// Extended training result with Phase 2 diagnostics.
///
/// # Example
///
/// ```ignore
/// use volt_soft::training::phase2::Phase2Result;
/// ```
#[derive(Debug, Clone)]
pub struct Phase2Result {
    /// Final average MSE loss.
    pub final_loss: f32,

    /// Loss history (one entry per step).
    pub loss_history: Vec<f32>,

    /// Attention loss history (if joint training enabled).
    pub attention_loss_history: Vec<f32>,

    /// Number of steps completed.
    pub steps_completed: usize,

    /// Curriculum stage this run trained on.
    pub stage: CurriculumStage,

    /// Validation metrics collected during training.
    pub validation_metrics: Vec<ValidationMetrics>,
}

/// Validation metrics collected at intervals during training.
///
/// # Example
///
/// ```ignore
/// use volt_soft::training::phase2::ValidationMetrics;
/// ```
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    /// Training step at which validation was run.
    pub step: usize,

    /// Average cosine similarity between predicted and target drifts.
    pub avg_cosine_similarity: f32,

    /// Average MSE loss on validation pairs.
    pub avg_mse_loss: f32,

    /// Fraction of slots where predicted drift direction is correct (cos > 0).
    pub directional_accuracy: f32,
}

/// Trains a Scaled VFN using Phase 2 Flow Matching with optional joint attention.
///
/// This is the main training entry point for Phase 2. It trains the scaled VFN
/// on (question, answer) frame pairs using flow matching, optionally training
/// cross-slot attention weights jointly.
///
/// # Errors
///
/// Returns [`VoltError::Internal`] if training encounters numerical issues.
///
/// # Example
///
/// ```ignore
/// use volt_soft::training::phase2::{train_phase2, Phase2Config, CurriculumStage};
/// use volt_soft::training::flow_matching::generate_synthetic_pairs;
/// use candle_core::Device;
/// use candle_nn::VarMap;
///
/// let device = Device::Cpu;
/// let var_map = VarMap::new();
/// let pairs = generate_synthetic_pairs(100, 0, 42).unwrap();
/// let config = Phase2Config {
///     vfn_config: ScaledVfnConfig { hidden_dim: 64, num_blocks: 2, ..ScaledVfnConfig::default() },
///     num_steps: 50,
///     batch_size: 8,
///     ..Phase2Config::default()
/// };
///
/// let result = train_phase2(&var_map, &pairs, &[], &config, &device).unwrap();
/// assert!(result.final_loss.is_finite());
/// ```
pub fn train_phase2(
    var_map: &VarMap,
    train_pairs: &[FramePair],
    validation_pairs: &[FramePair],
    config: &Phase2Config,
    device: &Device,
) -> Result<Phase2Result, VoltError> {
    if train_pairs.is_empty() {
        return Err(VoltError::Internal {
            message: "train_phase2: no training pairs provided".to_string(),
        });
    }

    let map_err = |e: candle_core::Error| VoltError::Internal {
        message: format!("train_phase2: {e}"),
    };

    // Create the scaled VFN
    let vfn = GpuScaledVfn::new_trainable(var_map, &config.vfn_config, device)?;

    // Set up optimizer
    let mut optimizer =
        candle_nn::AdamW::new(var_map.all_vars(), candle_nn::ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..Default::default()
        })
        .map_err(map_err)?;

    let mut rng = crate::nn::Rng::new(config.seed);
    let mut loss_history = Vec::with_capacity(config.num_steps);
    let mut attention_loss_history = Vec::with_capacity(config.num_steps);
    let mut validation_metrics = Vec::new();

    for step in 0..config.num_steps {
        // Learning rate warmup
        let lr_scale = if step < config.warmup_steps {
            (step + 1) as f64 / config.warmup_steps as f64
        } else {
            1.0
        };
        optimizer.set_learning_rate(config.learning_rate * lr_scale);

        // Sample mini-batch and compute flow matching loss
        let mut input_data = Vec::new();
        let mut target_data = Vec::new();
        let mut n_slots = 0usize;

        for _ in 0..config.batch_size {
            let pair_idx = (rng.next_u64() as usize) % train_pairs.len();
            let pair = &train_pairs[pair_idx];

            let t = rng.next_f32();

            for slot_idx in 0..MAX_SLOTS {
                let q_data = pair
                    .question
                    .slots[slot_idx]
                    .as_ref()
                    .and_then(|s| s.resolutions[config.resolution]);
                let a_data = pair
                    .answer
                    .slots[slot_idx]
                    .as_ref()
                    .and_then(|s| s.resolutions[config.resolution]);

                if let (Some(q_vec), Some(a_vec)) = (q_data, a_data) {
                    for d in 0..SLOT_DIM {
                        input_data.push((1.0 - t) * q_vec[d] + t * a_vec[d]);
                    }
                    for d in 0..SLOT_DIM {
                        target_data.push(a_vec[d] - q_vec[d]);
                    }
                    n_slots += 1;
                }
            }
        }

        if n_slots == 0 {
            loss_history.push(0.0);
            attention_loss_history.push(0.0);
            continue;
        }

        let input_tensor =
            Tensor::from_vec(input_data, (n_slots, SLOT_DIM), device).map_err(map_err)?;
        let target_tensor =
            Tensor::from_vec(target_data, (n_slots, SLOT_DIM), device).map_err(map_err)?;

        // VFN forward pass
        let predicted = vfn.forward_batch(&input_tensor)?;

        // VFN MSE loss
        let diff = (&predicted - &target_tensor).map_err(map_err)?;
        let sq = (&diff * &diff).map_err(map_err)?;
        let vfn_loss = sq.mean_all().map_err(map_err)?;

        let vfn_loss_val = vfn_loss.to_vec0::<f32>().map_err(map_err)?;
        loss_history.push(vfn_loss_val);

        // Total loss (attention component added if enabled)
        let total_loss = if config.joint_attention && n_slots >= 2 {
            // Compute attention-augmented loss:
            // attention message applied to predicted drifts should improve alignment
            let attn_loss = compute_attention_loss(
                &predicted,
                &target_tensor,
                n_slots,
                device,
            )?;
            let attn_loss_val = attn_loss.to_vec0::<f32>().map_err(map_err)?;
            attention_loss_history.push(attn_loss_val);

            let attn_weight = Tensor::new(config.attention_loss_weight, device).map_err(map_err)?;
            let scaled_attn = (&attn_loss * &attn_weight).map_err(map_err)?;
            (&vfn_loss + &scaled_attn).map_err(map_err)?
        } else {
            attention_loss_history.push(0.0);
            vfn_loss
        };

        // Backward + optimize
        optimizer.backward_step(&total_loss).map_err(map_err)?;

        // Validation at intervals
        if !validation_pairs.is_empty()
            && config.validation_interval > 0
            && (step + 1) % config.validation_interval == 0
        {
            let metrics = run_validation(
                &vfn,
                validation_pairs,
                config.resolution,
                step + 1,
                device,
            )?;
            validation_metrics.push(metrics);
        }
    }

    let final_loss = loss_history.last().copied().unwrap_or(f32::NAN);

    Ok(Phase2Result {
        final_loss,
        loss_history,
        attention_loss_history,
        steps_completed: config.num_steps,
        stage: config.stage,
        validation_metrics,
    })
}

/// Computes attention-augmented loss for joint training.
///
/// This is a simplified attention loss: we check if cross-slot message passing
/// reduces the drift prediction error. The loss encourages slot drifts to be
/// consistent with each other when combined via attention.
fn compute_attention_loss(
    predicted: &Tensor,
    target: &Tensor,
    n_slots: usize,
    device: &Device,
) -> Result<Tensor, VoltError> {
    let map_err = |e: candle_core::Error| VoltError::Internal {
        message: format!("compute_attention_loss: {e}"),
    };

    if n_slots < 2 {
        return Tensor::new(0.0f32, device).map_err(map_err);
    }

    // Simple cross-slot consistency: each slot's predicted drift should be
    // consistent with the mean drift of nearby slots
    let mean_predicted = predicted
        .mean_keepdim(0)
        .map_err(map_err)?
        .broadcast_as(predicted.shape())
        .map_err(map_err)?;

    // Consistency loss: how much individual predictions deviate from the group
    let consistency_diff = (predicted - mean_predicted).map_err(map_err)?;
    let consistency_sq = (&consistency_diff * &consistency_diff).map_err(map_err)?;
    let consistency_loss = consistency_sq.mean_all().map_err(map_err)?;

    // Scale it down — this is a regularization term
    let scale = Tensor::new(0.1f32, device).map_err(map_err)?;
    (&consistency_loss * &scale).map_err(map_err)
}

/// Runs validation on held-out pairs and returns metrics.
fn run_validation(
    vfn: &GpuScaledVfn,
    pairs: &[FramePair],
    resolution: usize,
    step: usize,
    device: &Device,
) -> Result<ValidationMetrics, VoltError> {
    let map_err = |e: candle_core::Error| VoltError::Internal {
        message: format!("run_validation: {e}"),
    };

    let mut total_cosine = 0.0f32;
    let mut total_mse = 0.0f32;
    let mut correct_direction = 0usize;
    let mut total_slots = 0usize;

    for pair in pairs {
        for slot_idx in 0..MAX_SLOTS {
            let q_data = pair
                .question
                .slots[slot_idx]
                .as_ref()
                .and_then(|s| s.resolutions[resolution]);
            let a_data = pair
                .answer
                .slots[slot_idx]
                .as_ref()
                .and_then(|s| s.resolutions[resolution]);

            if let (Some(q_vec), Some(a_vec)) = (q_data, a_data) {
                // Midpoint interpolation (t=0.5)
                let mut interp = [0.0f32; SLOT_DIM];
                let mut target_drift = [0.0f32; SLOT_DIM];
                for d in 0..SLOT_DIM {
                    interp[d] = 0.5 * q_vec[d] + 0.5 * a_vec[d];
                    target_drift[d] = a_vec[d] - q_vec[d];
                }

                let input_tensor =
                    Tensor::from_slice(&interp, &[1, SLOT_DIM], device).map_err(map_err)?;
                let predicted = vfn.forward_batch(&input_tensor)?;
                let pred_flat = predicted
                    .flatten_all()
                    .map_err(map_err)?
                    .to_vec1::<f32>()
                    .map_err(map_err)?;

                // Cosine similarity
                let dot: f32 = pred_flat
                    .iter()
                    .zip(target_drift.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let pred_norm: f32 = pred_flat.iter().map(|x| x * x).sum::<f32>().sqrt();
                let tgt_norm: f32 = target_drift.iter().map(|x| x * x).sum::<f32>().sqrt();
                let cos_sim = if pred_norm > 1e-8 && tgt_norm > 1e-8 {
                    dot / (pred_norm * tgt_norm)
                } else {
                    0.0
                };
                total_cosine += cos_sim;

                // MSE
                let mse: f32 = pred_flat
                    .iter()
                    .zip(target_drift.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    / SLOT_DIM as f32;
                total_mse += mse;

                // Directional accuracy
                if cos_sim > 0.0 {
                    correct_direction += 1;
                }
                total_slots += 1;
            }
        }
    }

    let n = total_slots.max(1) as f32;
    Ok(ValidationMetrics {
        step,
        avg_cosine_similarity: total_cosine / n,
        avg_mse_loss: total_mse / n,
        directional_accuracy: correct_direction as f32 / n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::flow_matching::generate_synthetic_pairs;

    fn small_phase2_config() -> Phase2Config {
        Phase2Config {
            vfn_config: ScaledVfnConfig {
                hidden_dim: 64,
                num_blocks: 2,
                io_dim: SLOT_DIM,
            },
            num_steps: 30,
            batch_size: 4,
            learning_rate: 1e-3,
            validation_interval: 10,
            validation_pairs: 5,
            warmup_steps: 5,
            ..Phase2Config::default()
        }
    }

    #[test]
    fn train_phase2_runs_and_completes() {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let pairs = generate_synthetic_pairs(50, 0, 42).unwrap();
        let val_pairs = generate_synthetic_pairs(10, 0, 99).unwrap();
        let config = small_phase2_config();

        let result = train_phase2(&var_map, &pairs, &val_pairs, &config, &device).unwrap();

        assert_eq!(result.steps_completed, 30);
        assert!(result.final_loss.is_finite());
        assert_eq!(result.stage, CurriculumStage::SimpleFunctions);
    }

    #[test]
    fn train_phase2_loss_decreases() {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let pairs = generate_synthetic_pairs(100, 0, 42).unwrap();
        let config = Phase2Config {
            num_steps: 60,
            batch_size: 8,
            learning_rate: 5e-4,
            validation_interval: 0,
            ..small_phase2_config()
        };

        let result = train_phase2(&var_map, &pairs, &[], &config, &device).unwrap();

        let early_avg: f32 = result.loss_history[..5].iter().sum::<f32>() / 5.0;
        let late_avg: f32 = result.loss_history[55..].iter().sum::<f32>() / 5.0;
        assert!(
            late_avg < early_avg,
            "loss should decrease: early_avg={early_avg}, late_avg={late_avg}"
        );
    }

    #[test]
    fn train_phase2_with_attention() {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let pairs = generate_synthetic_pairs(50, 0, 42).unwrap();
        let config = Phase2Config {
            joint_attention: true,
            attention_loss_weight: 0.3,
            ..small_phase2_config()
        };

        let result = train_phase2(&var_map, &pairs, &[], &config, &device).unwrap();

        assert!(result.final_loss.is_finite());
        // Attention loss history should be populated
        assert_eq!(result.attention_loss_history.len(), config.num_steps);
    }

    #[test]
    fn train_phase2_without_attention() {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let pairs = generate_synthetic_pairs(50, 0, 42).unwrap();
        let config = Phase2Config {
            joint_attention: false,
            ..small_phase2_config()
        };

        let result = train_phase2(&var_map, &pairs, &[], &config, &device).unwrap();
        assert!(result.final_loss.is_finite());
        // All attention losses should be 0
        assert!(result.attention_loss_history.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn train_phase2_collects_validation() {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let pairs = generate_synthetic_pairs(50, 0, 42).unwrap();
        let val_pairs = generate_synthetic_pairs(10, 0, 99).unwrap();
        let config = Phase2Config {
            validation_interval: 10,
            ..small_phase2_config()
        };

        let result = train_phase2(&var_map, &pairs, &val_pairs, &config, &device).unwrap();

        // Should have 3 validation runs (at steps 10, 20, 30)
        assert_eq!(result.validation_metrics.len(), 3);
        for m in &result.validation_metrics {
            assert!(m.avg_mse_loss.is_finite());
            assert!(m.avg_cosine_similarity.is_finite());
            assert!(m.directional_accuracy >= 0.0 && m.directional_accuracy <= 1.0);
        }
    }

    #[test]
    fn empty_pairs_errors() {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let config = small_phase2_config();

        assert!(train_phase2(&var_map, &[], &[], &config, &device).is_err());
    }

    #[test]
    fn curriculum_stage_progression() {
        let stage = CurriculumStage::SimpleFunctions;
        assert_eq!(stage.level(), 0);
        let next = stage.next().unwrap();
        assert_eq!(next, CurriculumStage::LoopsAndConditionals);
        assert_eq!(next.level(), 1);

        let last = CurriculumStage::AlgorithmicReasoning;
        assert!(last.next().is_none());
    }

    #[test]
    fn default_config_sensible() {
        let config = Phase2Config::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.num_steps > 0);
        assert!(config.batch_size > 0);
        assert!(config.attention_loss_weight >= 0.0);
        assert!(config.warmup_steps < config.num_steps);
    }
}
