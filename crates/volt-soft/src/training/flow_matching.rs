//! Flow Matching training for the Vector Field Network.
//!
//! Trains the VFN to predict drift directions between question and answer
//! TensorFrame pairs using linear interpolation paths.
//!
//! ## Algorithm
//!
//! Given a pair `(F_q, F_a)`:
//! 1. Sample `t ~ U(0, 1)`
//! 2. Interpolated state: `F(t) = (1-t) * F_q + t * F_a`
//! 3. Target drift: `F_a - F_q` (constant velocity field)
//! 4. Predicted drift: `VFN(F(t))`
//! 5. Loss: `MSE(predicted, target)`
//!
//! The VFN learns to predict the direction from any question frame toward
//! its corresponding answer frame, creating a velocity field that guides
//! RAR inference.

use candle_core::{Device, Tensor};
use candle_nn::{Optimizer, VarMap};
use volt_core::{SlotRole, TensorFrame, VoltError, MAX_SLOTS, SLOT_DIM};

use crate::gpu::vfn::GpuVfn;

/// Configuration for Flow Matching VFN training.
///
/// # Example
///
/// ```ignore
/// use volt_soft::training::FlowMatchConfig;
///
/// let config = FlowMatchConfig::default();
/// assert_eq!(config.num_steps, 1000);
/// ```
#[derive(Debug, Clone)]
pub struct FlowMatchConfig {
    /// Learning rate for AdamW optimizer (default: 1e-4).
    pub learning_rate: f64,

    /// AdamW weight decay (default: 0.01).
    pub weight_decay: f64,

    /// Number of training steps (default: 1000).
    pub num_steps: usize,

    /// Batch size — number of frame pairs per step (default: 32).
    pub batch_size: usize,

    /// Random seed for reproducibility.
    pub seed: u64,

    /// Which resolution to train on (default: 0).
    pub resolution: usize,
}

impl Default for FlowMatchConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            weight_decay: 0.01,
            num_steps: 1000,
            batch_size: 32,
            seed: 42,
            resolution: 0,
        }
    }
}

/// A (question, answer) TensorFrame training pair.
///
/// Both frames must have matching active slots at the target resolution.
///
/// # Example
///
/// ```ignore
/// use volt_soft::training::FramePair;
/// use volt_core::TensorFrame;
///
/// let pair = FramePair {
///     question: TensorFrame::new(),
///     answer: TensorFrame::new(),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct FramePair {
    /// The question/input frame.
    pub question: TensorFrame,
    /// The answer/target frame.
    pub answer: TensorFrame,
}

/// Result of a training run.
///
/// # Example
///
/// ```ignore
/// use volt_soft::training::TrainResult;
///
/// // Returned by train_vfn_flow_matching
/// ```
#[derive(Debug, Clone)]
pub struct TrainResult {
    /// Final average MSE loss.
    pub final_loss: f32,

    /// Loss history (one entry per step).
    pub loss_history: Vec<f32>,

    /// Number of steps completed.
    pub steps_completed: usize,
}

/// Trains a VFN using Flow Matching on (question, answer) frame pairs.
///
/// The VFN's weights are updated in-place via the VarMap. Returns
/// training diagnostics including loss history.
///
/// # Errors
///
/// Returns [`VoltError::Internal`] if training encounters numerical issues.
///
/// # Example
///
/// ```ignore
/// use volt_soft::training::{train_vfn_flow_matching, FlowMatchConfig, generate_synthetic_pairs};
/// use volt_soft::gpu::vfn::GpuVfn;
/// use candle_core::Device;
/// use candle_nn::VarMap;
///
/// let device = Device::Cpu;
/// let var_map = VarMap::new();
/// let vfn = GpuVfn::new_trainable(&var_map, &device).unwrap();
/// let pairs = generate_synthetic_pairs(100, 0, 42).unwrap();
/// let config = FlowMatchConfig { num_steps: 10, ..FlowMatchConfig::default() };
///
/// let result = train_vfn_flow_matching(&vfn, &var_map, &pairs, &config, &device).unwrap();
/// assert!(result.final_loss.is_finite());
/// ```
pub fn train_vfn_flow_matching(
    vfn: &GpuVfn,
    var_map: &VarMap,
    pairs: &[FramePair],
    config: &FlowMatchConfig,
    device: &Device,
) -> Result<TrainResult, VoltError> {
    if pairs.is_empty() {
        return Err(VoltError::Internal {
            message: "train_vfn_flow_matching: no training pairs provided".to_string(),
        });
    }

    let map_err = |e: candle_core::Error| VoltError::Internal {
        message: format!("train_vfn_flow_matching: {e}"),
    };

    let mut optimizer =
        candle_nn::AdamW::new(var_map.all_vars(), candle_nn::ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..Default::default()
        })
        .map_err(map_err)?;

    let mut rng = crate::nn::Rng::new(config.seed);
    let mut loss_history = Vec::with_capacity(config.num_steps);

    for _step in 0..config.num_steps {
        // Sample a mini-batch of pairs
        let mut input_data = Vec::new();
        let mut target_data = Vec::new();
        let mut n_slots = 0usize;

        for _ in 0..config.batch_size {
            let pair_idx = (rng.next_u64() as usize) % pairs.len();
            let pair = &pairs[pair_idx];

            // Sample interpolation time t ~ U(0, 1)
            let t = rng.next_f32();

            // Extract matching active slots
            for slot_idx in 0..MAX_SLOTS {
                let q_data = pair.question.slots[slot_idx]
                    .as_ref()
                    .and_then(|s| s.resolutions[config.resolution]);
                let a_data = pair.answer.slots[slot_idx]
                    .as_ref()
                    .and_then(|s| s.resolutions[config.resolution]);

                if let (Some(q_vec), Some(a_vec)) = (q_data, a_data) {
                    // Interpolated state: F(t) = (1-t)*F_q + t*F_a
                    for d in 0..SLOT_DIM {
                        input_data.push((1.0 - t) * q_vec[d] + t * a_vec[d]);
                    }
                    // Target drift: F_a - F_q (constant velocity field)
                    for d in 0..SLOT_DIM {
                        target_data.push(a_vec[d] - q_vec[d]);
                    }
                    n_slots += 1;
                }
            }
        }

        if n_slots == 0 {
            loss_history.push(0.0);
            continue;
        }

        let input_tensor =
            Tensor::from_vec(input_data, (n_slots, SLOT_DIM), device).map_err(map_err)?;
        let target_tensor =
            Tensor::from_vec(target_data, (n_slots, SLOT_DIM), device).map_err(map_err)?;

        // Forward pass
        let predicted = vfn.forward_batch(&input_tensor)?;

        // MSE loss
        let diff = (predicted - &target_tensor).map_err(map_err)?;
        let sq = (&diff * &diff).map_err(map_err)?;
        let loss = sq.mean_all().map_err(map_err)?;

        let loss_val = loss.to_vec0::<f32>().map_err(map_err)?;
        loss_history.push(loss_val);

        // Backward + optimize
        optimizer.backward_step(&loss).map_err(map_err)?;
    }

    let final_loss = loss_history.last().copied().unwrap_or(f32::NAN);

    Ok(TrainResult {
        final_loss,
        loss_history,
        steps_completed: config.num_steps,
    })
}

/// Generates synthetic (question, answer) frame pairs for testing.
///
/// Creates `count` random pairs where the answer is a deterministic
/// transformation (rotation + offset) of the question. Useful for
/// verifying that the training loop converges.
///
/// # Errors
///
/// Returns [`VoltError::Internal`] if frame construction fails.
///
/// # Example
///
/// ```ignore
/// use volt_soft::training::generate_synthetic_pairs;
///
/// let pairs = generate_synthetic_pairs(100, 0, 42).unwrap();
/// assert_eq!(pairs.len(), 100);
/// ```
pub fn generate_synthetic_pairs(
    count: usize,
    resolution: usize,
    seed: u64,
) -> Result<Vec<FramePair>, VoltError> {
    let mut rng = crate::nn::Rng::new(seed);
    let mut pairs = Vec::with_capacity(count);

    let roles = [
        SlotRole::Agent,
        SlotRole::Predicate,
        SlotRole::Patient,
        SlotRole::Location,
    ];

    for _ in 0..count {
        let mut question = TensorFrame::new();
        let mut answer = TensorFrame::new();

        // Populate 4 slots with random normalized vectors
        let n_active = 2 + ((rng.next_u64() % 3) as usize); // 2-4 slots
        for slot_idx in 0..n_active {
            // Random question vector
            let mut q_vec = [0.0f32; SLOT_DIM];
            for x in &mut q_vec {
                *x = rng.next_f32_range(-1.0, 1.0);
            }
            let norm: f32 = q_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut q_vec {
                *x /= norm;
            }

            // Answer = rotated + offset version (deterministic transform)
            let mut a_vec = [0.0f32; SLOT_DIM];
            for (d, a) in a_vec.iter_mut().enumerate() {
                // Simple transform: shift indices + add small perturbation
                let src = (d + 7) % SLOT_DIM;
                *a = q_vec[src] * 0.8 + rng.next_f32_range(-0.1, 0.1);
            }
            let norm: f32 = a_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut a_vec {
                *x /= norm;
            }

            let role = roles[slot_idx % roles.len()];
            question
                .write_at(slot_idx, resolution, role, q_vec)
                .map_err(|e| VoltError::Internal {
                    message: format!("generate_synthetic_pairs: {e}"),
                })?;
            answer
                .write_at(slot_idx, resolution, role, a_vec)
                .map_err(|e| VoltError::Internal {
                    message: format!("generate_synthetic_pairs: {e}"),
                })?;
        }

        pairs.push(FramePair { question, answer });
    }

    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_sensible() {
        let config = FlowMatchConfig::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.num_steps > 0);
        assert!(config.batch_size > 0);
    }

    #[test]
    fn generate_synthetic_pairs_creates_valid_data() {
        let pairs = generate_synthetic_pairs(10, 0, 42).unwrap();
        assert_eq!(pairs.len(), 10);
        for pair in &pairs {
            // Both question and answer should have at least 2 active slots
            assert!(pair.question.active_slot_count() >= 2);
            assert!(pair.answer.active_slot_count() >= 2);
            assert_eq!(
                pair.question.active_slot_count(),
                pair.answer.active_slot_count(),
            );
        }
    }

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

        // Loss should generally decrease (compare first 5 avg vs last 5 avg)
        let early_avg: f32 = result.loss_history[..5].iter().sum::<f32>() / 5.0;
        let late_avg: f32 = result.loss_history[45..].iter().sum::<f32>() / 5.0;
        assert!(
            late_avg < early_avg,
            "loss should decrease: early_avg={}, late_avg={}",
            early_avg,
            late_avg,
        );
    }

    #[test]
    fn empty_pairs_errors() {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vfn = GpuVfn::new_trainable(&var_map, &device).unwrap();
        let config = FlowMatchConfig::default();

        assert!(train_vfn_flow_matching(&vfn, &var_map, &[], &config, &device).is_err());
    }

    #[test]
    fn synthetic_pairs_are_deterministic() {
        let p1 = generate_synthetic_pairs(5, 0, 42).unwrap();
        let p2 = generate_synthetic_pairs(5, 0, 42).unwrap();
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a.question.active_slot_count(), b.question.active_slot_count());
        }
    }
}
