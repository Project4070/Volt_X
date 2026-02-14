//! Time-conditioned Flow Matching training for the Scaled VFN.
//!
//! Extends the base flow matching algorithm with:
//! - **Time conditioning**: `t` is passed to `ScaledVfn::forward_batch`
//! - **Epoch-based training**: shuffles data each epoch, not step-based
//! - **LR schedule**: linear warmup + cosine decay
//! - **Train/validation split**: tracks both train and valid loss
//! - **Progress logging**: per-step and per-epoch summaries to stderr
//!
//! ## Algorithm
//!
//! For each training step within an epoch:
//! 1. Sample a batch of `(F_q, F_a)` frame pairs
//! 2. Sample `t ~ U(0, 1)` for each slot
//! 3. Interpolated state: `F(t) = (1-t)·F_q + t·F_a`
//! 4. Target drift: `F_a - F_q` (constant velocity field)
//! 5. Predicted drift: `ScaledVfn(F(t), t)`
//! 6. Loss: MSE(predicted, target)
//! 7. Backprop + AdamW step with scheduled LR

use candle_core::{Device, Tensor};
use candle_nn::{Optimizer, VarMap};
use volt_core::{VoltError, MAX_SLOTS, SLOT_DIM};

use crate::scaled_vfn::ScaledVfn;
use crate::training::flow_matching::FramePair;

/// Configuration for Scaled VFN flow matching training.
///
/// # Example
///
/// ```ignore
/// use volt_soft::training::scaled_flow_matching::ScaledFlowConfig;
///
/// let config = ScaledFlowConfig::default();
/// assert_eq!(config.epochs, 10);
/// assert_eq!(config.batch_size, 32);
/// ```
#[derive(Debug, Clone)]
pub struct ScaledFlowConfig {
    /// Peak learning rate after warmup (default: 1e-4).
    pub max_lr: f64,

    /// Minimum learning rate at end of cosine decay (default: 1e-6).
    pub min_lr: f64,

    /// AdamW weight decay (default: 0.01).
    pub weight_decay: f64,

    /// Number of linear warmup steps (default: 2000).
    pub warmup_steps: usize,

    /// Number of training epochs (default: 10).
    pub epochs: usize,

    /// Frame pairs per optimizer step (default: 32).
    pub batch_size: usize,

    /// Random seed for reproducibility (default: 42).
    pub seed: u64,

    /// Which TensorFrame resolution to train on (default: 0).
    pub resolution: usize,

    /// Fraction of pairs held out for validation (default: 0.1).
    pub validation_frac: f64,

    /// Print progress every N steps (default: 50).
    pub log_interval: usize,
}

impl Default for ScaledFlowConfig {
    fn default() -> Self {
        Self {
            max_lr: 1e-4,
            min_lr: 1e-6,
            weight_decay: 0.01,
            warmup_steps: 2000,
            epochs: 10,
            batch_size: 32,
            seed: 42,
            resolution: 0,
            validation_frac: 0.1,
            log_interval: 50,
        }
    }
}

/// Per-epoch training result.
#[derive(Debug, Clone)]
pub struct EpochResult {
    /// Average training MSE loss for this epoch.
    pub train_loss: f32,
    /// Average validation MSE loss for this epoch.
    pub valid_loss: f32,
    /// Number of training steps completed in this epoch.
    pub steps: usize,
}

/// Result of a complete training run.
///
/// # Example
///
/// ```ignore
/// // Returned by train_scaled_vfn()
/// ```
#[derive(Debug, Clone)]
pub struct ScaledTrainResult {
    /// Final training loss (last epoch).
    pub final_train_loss: f32,
    /// Final validation loss (last epoch).
    pub final_valid_loss: f32,
    /// Per-epoch results.
    pub epoch_results: Vec<EpochResult>,
    /// Total training steps completed across all epochs.
    pub total_steps: usize,
}

/// Trains a ScaledVfn using time-conditioned flow matching.
///
/// The VFN's weights are updated in-place via the VarMap. Returns
/// training diagnostics including per-epoch train/valid losses.
///
/// # Errors
///
/// Returns [`VoltError::Internal`] if:
/// - No training pairs provided
/// - Numerical issues during training
///
/// # Example
///
/// ```ignore
/// use volt_soft::scaled_vfn::{ScaledVfn, ScaledVfnConfig};
/// use volt_soft::training::scaled_flow_matching::{ScaledFlowConfig, train_scaled_vfn};
/// use volt_soft::training::generate_synthetic_pairs;
/// use candle_core::Device;
/// use candle_nn::VarMap;
///
/// let device = Device::Cpu;
/// let var_map = VarMap::new();
/// let vfn_config = ScaledVfnConfig { hidden_dim: 64, num_blocks: 1, time_embed_freqs: 4 };
/// let vfn = ScaledVfn::new_trainable(&vfn_config, &var_map, &device).unwrap();
/// let pairs = generate_synthetic_pairs(100, 0, 42).unwrap();
///
/// let config = ScaledFlowConfig {
///     epochs: 2, batch_size: 8, max_lr: 1e-3, warmup_steps: 10,
///     log_interval: 100, ..ScaledFlowConfig::default()
/// };
///
/// let result = train_scaled_vfn(&vfn, &var_map, &pairs, &config, &device).unwrap();
/// assert!(result.final_train_loss.is_finite());
/// ```
pub fn train_scaled_vfn(
    vfn: &ScaledVfn,
    var_map: &VarMap,
    pairs: &[FramePair],
    config: &ScaledFlowConfig,
    device: &Device,
) -> Result<ScaledTrainResult, VoltError> {
    if pairs.is_empty() {
        return Err(VoltError::Internal {
            message: "train_scaled_vfn: no training pairs provided".to_string(),
        });
    }

    let map_err = |e: candle_core::Error| VoltError::Internal {
        message: format!("train_scaled_vfn: {e}"),
    };

    // Split into train/valid
    let valid_count = (pairs.len() as f64 * config.validation_frac).max(1.0) as usize;
    let train_count = pairs.len().saturating_sub(valid_count);
    if train_count == 0 {
        return Err(VoltError::Internal {
            message: "train_scaled_vfn: no training pairs after validation split".to_string(),
        });
    }
    let train_pairs = &pairs[..train_count];
    let valid_pairs = &pairs[train_count..];

    // Optimizer
    let mut optimizer =
        candle_nn::AdamW::new(var_map.all_vars(), candle_nn::ParamsAdamW {
            lr: config.max_lr,
            weight_decay: config.weight_decay,
            ..Default::default()
        })
        .map_err(map_err)?;

    let steps_per_epoch = train_count / config.batch_size;
    if steps_per_epoch == 0 {
        return Err(VoltError::Internal {
            message: format!(
                "train_scaled_vfn: batch_size ({}) > training pairs ({})",
                config.batch_size, train_count
            ),
        });
    }

    let total_steps = steps_per_epoch * config.epochs;
    let mut global_step = 0usize;
    let mut rng_state = config.seed;
    let mut epoch_results = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        // Shuffle training indices
        let mut indices: Vec<usize> = (0..train_count).collect();
        shuffle(&mut indices, &mut rng_state);

        let mut epoch_loss_sum = 0.0f64;
        let mut epoch_step_count = 0usize;

        for step in 0..steps_per_epoch {
            global_step += 1;

            // LR schedule
            let lr = compute_lr(
                global_step,
                config.warmup_steps,
                total_steps,
                config.max_lr,
                config.min_lr,
            );
            optimizer.set_learning_rate(lr);

            // Get batch indices
            let batch_start = step * config.batch_size;
            let batch_indices = &indices[batch_start..batch_start + config.batch_size];

            // Build slot-level tensors
            let (input_data, target_data, time_data, n_slots) =
                build_flow_batch(train_pairs, batch_indices, config.resolution, &mut rng_state);

            if n_slots == 0 {
                continue;
            }

            let input_tensor =
                Tensor::from_vec(input_data, (n_slots, SLOT_DIM), device).map_err(map_err)?;
            let target_tensor =
                Tensor::from_vec(target_data, (n_slots, SLOT_DIM), device).map_err(map_err)?;
            let time_tensor =
                Tensor::from_vec(time_data, n_slots, device).map_err(map_err)?;

            // Forward: ScaledVfn with time conditioning
            let predicted = vfn.forward_batch(&input_tensor, &time_tensor)?;

            // MSE loss
            let diff = (predicted - &target_tensor).map_err(map_err)?;
            let sq = (&diff * &diff).map_err(map_err)?;
            let loss = sq.mean_all().map_err(map_err)?;

            let loss_val = loss.to_vec0::<f32>().map_err(map_err)?;

            // Backward + step
            optimizer.backward_step(&loss).map_err(map_err)?;

            if loss_val.is_finite() {
                epoch_loss_sum += loss_val as f64;
                epoch_step_count += 1;
            }

            // Progress logging
            if global_step.is_multiple_of(config.log_interval) || step == 0 {
                eprintln!(
                    "[Epoch {}/{}] Step {}/{} | Loss: {:.4} | LR: {:.2e} | {} slots",
                    epoch + 1,
                    config.epochs,
                    step + 1,
                    steps_per_epoch,
                    loss_val,
                    lr,
                    n_slots,
                );
            }
        }

        // Epoch train loss
        let avg_train_loss = if epoch_step_count > 0 {
            (epoch_loss_sum / epoch_step_count as f64) as f32
        } else {
            f32::NAN
        };

        // Validation loss
        let avg_valid_loss = compute_validation_loss(
            vfn,
            valid_pairs,
            config.batch_size,
            config.resolution,
            device,
            &mut rng_state,
        );

        eprintln!(
            "[Epoch {}/{}] Complete | Train Loss: {:.4} | Valid Loss: {:.4}",
            epoch + 1,
            config.epochs,
            avg_train_loss,
            avg_valid_loss,
        );

        epoch_results.push(EpochResult {
            train_loss: avg_train_loss,
            valid_loss: avg_valid_loss,
            steps: epoch_step_count,
        });
    }

    let final_train = epoch_results
        .last()
        .map(|r| r.train_loss)
        .unwrap_or(f32::NAN);
    let final_valid = epoch_results
        .last()
        .map(|r| r.valid_loss)
        .unwrap_or(f32::NAN);

    Ok(ScaledTrainResult {
        final_train_loss: final_train,
        final_valid_loss: final_valid,
        epoch_results,
        total_steps: global_step,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Builds slot-level tensors for a flow matching batch.
///
/// Returns `(input_data, target_data, time_data, n_slots)` flattened as
/// `Vec<f32>` for tensor construction.
fn build_flow_batch(
    pairs: &[FramePair],
    batch_indices: &[usize],
    resolution: usize,
    rng_state: &mut u64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    let mut input_data = Vec::new();
    let mut target_data = Vec::new();
    let mut time_data = Vec::new();
    let mut n_slots = 0usize;

    for &idx in batch_indices {
        let pair = &pairs[idx];

        // Sample interpolation time t ~ U(0, 1)
        let t = next_f32(rng_state);

        // Extract matching active slots at the target resolution
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
                // Interpolated state: x(t) = (1-t)*q + t*a
                for d in 0..SLOT_DIM {
                    input_data.push((1.0 - t) * q_vec[d] + t * a_vec[d]);
                }
                // Target drift: a - q
                for d in 0..SLOT_DIM {
                    target_data.push(a_vec[d] - q_vec[d]);
                }
                // Each slot gets the same t for this pair
                time_data.push(t);
                n_slots += 1;
            }
        }
    }

    (input_data, target_data, time_data, n_slots)
}

/// Compute average validation MSE loss (no gradient).
fn compute_validation_loss(
    vfn: &ScaledVfn,
    valid_pairs: &[FramePair],
    batch_size: usize,
    resolution: usize,
    device: &Device,
    rng_state: &mut u64,
) -> f32 {
    if valid_pairs.is_empty() {
        return f32::NAN;
    }

    let map_err = |e: candle_core::Error| VoltError::Internal {
        message: format!("validation: {e}"),
    };

    let indices: Vec<usize> = (0..valid_pairs.len()).collect();
    let steps = (valid_pairs.len() / batch_size).clamp(1, 50);
    let mut total_loss = 0.0f64;
    let mut count = 0;

    for step in 0..steps {
        let batch_start = step * batch_size;
        let batch_end = (batch_start + batch_size).min(valid_pairs.len());
        let batch_indices = &indices[batch_start..batch_end];

        let (input_data, target_data, time_data, n_slots) =
            build_flow_batch(valid_pairs, batch_indices, resolution, rng_state);

        if n_slots == 0 {
            continue;
        }

        let Ok(input_tensor) =
            Tensor::from_vec(input_data, (n_slots, SLOT_DIM), device).map_err(map_err)
        else {
            continue;
        };
        let Ok(target_tensor) =
            Tensor::from_vec(target_data, (n_slots, SLOT_DIM), device).map_err(map_err)
        else {
            continue;
        };
        let Ok(time_tensor) =
            Tensor::from_vec(time_data, n_slots, device).map_err(map_err)
        else {
            continue;
        };

        let Ok(predicted) = vfn.forward_batch(&input_tensor, &time_tensor) else {
            continue;
        };

        let loss_val = (|| -> Result<f32, candle_core::Error> {
            let diff = (predicted - &target_tensor)?;
            let sq = (&diff * &diff)?;
            let loss = sq.mean_all()?;
            loss.to_vec0::<f32>()
        })();

        if let Ok(val) = loss_val
            && val.is_finite()
        {
            total_loss += val as f64;
            count += 1;
        }
    }

    if count > 0 {
        (total_loss / count as f64) as f32
    } else {
        f32::NAN
    }
}

/// Cosine learning rate schedule with linear warmup.
fn compute_lr(
    step: usize,
    warmup_steps: usize,
    total_steps: usize,
    max_lr: f64,
    min_lr: f64,
) -> f64 {
    if step <= warmup_steps {
        max_lr * (step as f64 / warmup_steps.max(1) as f64)
    } else {
        let progress =
            (step - warmup_steps) as f64 / (total_steps - warmup_steps).max(1) as f64;
        min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

/// Simple Fisher-Yates shuffle using a basic PRNG.
fn shuffle(data: &mut [usize], rng_state: &mut u64) {
    for i in (1..data.len()).rev() {
        *rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let j = (*rng_state >> 33) as usize % (i + 1);
        data.swap(i, j);
    }
}

/// Generate a random f32 in [0, 1) from the PRNG state.
fn next_f32(rng_state: &mut u64) -> f32 {
    *rng_state = rng_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1);
    (*rng_state >> 40) as f32 / (1u64 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::generate_synthetic_pairs;
    use crate::scaled_vfn::ScaledVfnConfig;

    fn small_config() -> (ScaledVfnConfig, ScaledFlowConfig) {
        let vfn_config = ScaledVfnConfig {
            hidden_dim: 64,
            num_blocks: 1,
            time_embed_freqs: 4,
        };
        let flow_config = ScaledFlowConfig {
            epochs: 3,
            batch_size: 8,
            max_lr: 1e-3,
            min_lr: 1e-5,
            warmup_steps: 5,
            log_interval: 1000, // Suppress output in tests
            validation_frac: 0.2,
            ..ScaledFlowConfig::default()
        };
        (vfn_config, flow_config)
    }

    #[test]
    fn default_config_sensible() {
        let config = ScaledFlowConfig::default();
        assert!(config.max_lr > 0.0);
        assert!(config.epochs > 0);
        assert!(config.batch_size > 0);
        assert!(config.validation_frac > 0.0 && config.validation_frac < 1.0);
    }

    #[test]
    fn training_loss_decreases() {
        let (vfn_config, flow_config) = small_config();
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vfn =
            ScaledVfn::new_trainable(&vfn_config, &var_map, &device).unwrap();
        let pairs = generate_synthetic_pairs(50, 0, 42).unwrap();

        let result =
            train_scaled_vfn(&vfn, &var_map, &pairs, &flow_config, &device).unwrap();

        assert_eq!(result.epoch_results.len(), 3);
        assert!(result.final_train_loss.is_finite());
        assert!(result.total_steps > 0);

        // Loss should generally decrease
        let first = result.epoch_results[0].train_loss;
        let last = result.epoch_results[2].train_loss;
        assert!(
            last < first,
            "loss should decrease: first={first}, last={last}"
        );
    }

    #[test]
    fn validation_loss_computed() {
        let (vfn_config, flow_config) = small_config();
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vfn =
            ScaledVfn::new_trainable(&vfn_config, &var_map, &device).unwrap();
        let pairs = generate_synthetic_pairs(50, 0, 42).unwrap();

        let result =
            train_scaled_vfn(&vfn, &var_map, &pairs, &flow_config, &device).unwrap();

        for epoch in &result.epoch_results {
            assert!(epoch.valid_loss.is_finite(), "valid loss should be finite");
        }
    }

    #[test]
    fn empty_pairs_errors() {
        let (vfn_config, flow_config) = small_config();
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vfn =
            ScaledVfn::new_trainable(&vfn_config, &var_map, &device).unwrap();

        let result = train_scaled_vfn(&vfn, &var_map, &[], &flow_config, &device);
        assert!(result.is_err());
    }

    #[test]
    fn lr_schedule_correct() {
        let max_lr = 1e-4;
        let min_lr = 1e-6;

        // Step 0: lr = 0
        assert!((compute_lr(0, 100, 1000, max_lr, min_lr) - 0.0).abs() < 1e-10);

        // Step 50 (middle of warmup): lr ≈ 0.5 * max_lr
        let mid_warmup = compute_lr(50, 100, 1000, max_lr, min_lr);
        assert!((mid_warmup - 5e-5).abs() < 1e-8);

        // Step 100 (end of warmup): lr = max_lr
        let end_warmup = compute_lr(100, 100, 1000, max_lr, min_lr);
        assert!((end_warmup - max_lr).abs() < 1e-10);

        // Step 1000 (end): lr ≈ min_lr
        let end = compute_lr(1000, 100, 1000, max_lr, min_lr);
        assert!((end - min_lr).abs() < 1e-8);
    }
}
