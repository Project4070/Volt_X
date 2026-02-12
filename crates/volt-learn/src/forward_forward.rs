//! Forward-Forward training for the VFN.
//!
//! Trains the Vector Field Network layer-by-layer using Hinton's
//! Forward-Forward algorithm. Each layer learns independently to
//! assign high "goodness" (sum of squared activations) to positive
//! data and low goodness to negative data.
//!
//! ## Algorithm
//!
//! For each layer L (0, 1, 2):
//! 1. Forward pass through layer L only
//! 2. Compute goodness = Σ(activation²)
//! 3. For positive samples: increase goodness toward threshold
//! 4. For negative samples: decrease goodness toward threshold
//! 5. Use L's output as input to L+1 (detached — no cross-layer gradient)
//!
//! ## Positive vs Negative Data
//!
//! - **Positive**: slot embeddings from high-gamma verified frames
//! - **Negative**: slot embeddings from low-gamma frames + random corruptions

use volt_core::{VoltError, MAX_SLOTS, SLOT_DIM};
use volt_db::VoltStore;
use volt_soft::vfn::Vfn;

use crate::event::LearningEvent;

/// A labeled sample for Forward-Forward training.
///
/// # Example
///
/// ```
/// use volt_learn::forward_forward::FfSample;
/// use volt_core::SLOT_DIM;
///
/// let sample = FfSample {
///     embedding: [0.1; SLOT_DIM],
///     is_positive: true,
/// };
/// assert!(sample.is_positive);
/// ```
#[derive(Debug, Clone)]
pub struct FfSample {
    /// The 256-dim slot embedding (R₀ vector from a frame slot).
    pub embedding: [f32; SLOT_DIM],
    /// `true` for high-gamma verified data, `false` for low-gamma/corrupted.
    pub is_positive: bool,
}

/// Configuration for Forward-Forward training.
///
/// # Example
///
/// ```
/// use volt_learn::forward_forward::FfConfig;
///
/// let config = FfConfig::default();
/// assert_eq!(config.num_epochs, 5);
/// assert!(config.learning_rate > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct FfConfig {
    /// Learning rate for weight updates. Default: 0.001.
    pub learning_rate: f32,
    /// Goodness threshold — positive samples should exceed this,
    /// negative samples should stay below. Default: 2.0.
    pub goodness_threshold: f32,
    /// Number of training epochs per layer. Default: 5.
    pub num_epochs: usize,
    /// Standard deviation of Gaussian noise for corrupting negative
    /// samples. Default: 0.3.
    pub corruption_noise: f32,
    /// Gamma threshold above which a frame is considered positive.
    /// Default: 0.7.
    pub positive_gamma_threshold: f32,
    /// Gamma threshold below which a frame is considered negative.
    /// Default: 0.3.
    pub negative_gamma_threshold: f32,
    /// Random seed for corruption noise generation. Default: 42.
    pub seed: u64,
}

impl Default for FfConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            goodness_threshold: 2.0,
            num_epochs: 5,
            corruption_noise: 0.3,
            positive_gamma_threshold: 0.7,
            negative_gamma_threshold: 0.3,
            seed: 42,
        }
    }
}

/// Result of a Forward-Forward training run.
///
/// # Example
///
/// ```
/// use volt_learn::forward_forward::FfResult;
///
/// let result = FfResult {
///     layers_updated: 3,
///     positive_goodness_before: vec![1.0, 1.5, 1.2],
///     positive_goodness_after: vec![2.5, 2.8, 2.1],
///     negative_goodness_before: vec![2.5, 2.2, 2.0],
///     negative_goodness_after: vec![1.5, 1.2, 1.0],
/// };
/// assert_eq!(result.layers_updated, 3);
/// ```
#[derive(Debug, Clone)]
pub struct FfResult {
    /// Number of VFN layers that were updated.
    pub layers_updated: usize,
    /// Average positive-sample goodness per layer before training.
    pub positive_goodness_before: Vec<f32>,
    /// Average positive-sample goodness per layer after training.
    pub positive_goodness_after: Vec<f32>,
    /// Average negative-sample goodness per layer before training.
    pub negative_goodness_before: Vec<f32>,
    /// Average negative-sample goodness per layer after training.
    pub negative_goodness_after: Vec<f32>,
}

/// Simple PRNG for deterministic noise generation (splitmix64).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Returns approximate Gaussian noise via Box-Muller using
    /// two uniform samples.
    fn next_gaussian(&mut self, stddev: f32) -> f32 {
        let u1 = (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32);
        let u2 = (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32);
        // Clamp u1 away from zero to avoid log(0)
        let u1 = u1.max(1e-10);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        z * stddev
    }
}

/// Collects Forward-Forward training samples from learning events.
///
/// Extracts R₀ slot embeddings from frames referenced by the events.
/// High-gamma frames become positive samples; low-gamma frames become
/// negative samples. Additional negative samples are created by
/// corrupting positive embeddings with Gaussian noise.
///
/// # Errors
///
/// Returns [`VoltError::LearnError`] if no samples could be collected.
///
/// # Example
///
/// ```
/// use volt_learn::forward_forward::{collect_ff_samples, FfConfig};
///
/// // With empty events, returns an error
/// let config = FfConfig::default();
/// let store = volt_db::VoltStore::new();
/// let result = collect_ff_samples(&[], &store, &config);
/// assert!(result.is_err());
/// ```
pub fn collect_ff_samples(
    events: &[LearningEvent],
    store: &VoltStore,
    config: &FfConfig,
) -> Result<Vec<FfSample>, VoltError> {
    if events.is_empty() {
        return Err(VoltError::LearnError {
            message: "collect_ff_samples: no events provided".to_string(),
        });
    }

    let mut samples = Vec::new();
    let mut rng = Rng::new(config.seed);

    for event in events {
        // Compute average gamma for this event (non-zero slots only)
        let active_gammas: Vec<f32> = event
            .gamma_scores
            .iter()
            .copied()
            .filter(|&g| g > 0.0)
            .collect();

        if active_gammas.is_empty() {
            continue;
        }

        let avg_gamma: f32 =
            active_gammas.iter().sum::<f32>() / active_gammas.len() as f32;

        // Try to get the frame from the store
        let frame = match store.get_by_id(event.frame_id) {
            Some(f) => f,
            None => continue,
        };

        let is_positive = avg_gamma >= config.positive_gamma_threshold;
        let is_negative = avg_gamma <= config.negative_gamma_threshold;

        if !is_positive && !is_negative {
            // Ambiguous gamma — skip
            continue;
        }

        // Extract R₀ embeddings from active slots
        for slot_idx in 0..MAX_SLOTS {
            if let Some(slot) = &frame.slots[slot_idx]
                && let Some(r0) = &slot.resolutions[0]
            {
                samples.push(FfSample {
                    embedding: *r0,
                    is_positive,
                });

                // For positive samples, also create a corrupted negative
                if is_positive {
                    let mut corrupted = *r0;
                    for v in &mut corrupted {
                        *v += rng.next_gaussian(config.corruption_noise);
                    }
                    // L2-normalize the corrupted vector
                    let norm: f32 =
                        corrupted.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 1e-10 {
                        for v in &mut corrupted {
                            *v /= norm;
                        }
                    }
                    samples.push(FfSample {
                        embedding: corrupted,
                        is_positive: false,
                    });
                }
            }
        }
    }

    if samples.is_empty() {
        return Err(VoltError::LearnError {
            message: "collect_ff_samples: no valid samples extracted".to_string(),
        });
    }

    Ok(samples)
}

/// Computes goodness: sum of squared activations.
fn goodness(activations: &[f32]) -> f32 {
    activations.iter().map(|a| a * a).sum()
}

/// Computes average goodness for a set of samples through a VFN layer.
fn avg_goodness_for_samples(
    vfn: &Vfn,
    layer_idx: usize,
    samples: &[&FfSample],
    prev_layers: &[usize],
) -> Result<f32, VoltError> {
    if samples.is_empty() {
        return Ok(0.0);
    }
    let mut total = 0.0;
    for sample in samples {
        let mut input: Vec<f32> = sample.embedding.to_vec();
        // Forward through all previous layers to get input for this layer
        for &prev in prev_layers {
            input = vfn.forward_layer(prev, &input)?;
        }
        let activations = vfn.forward_layer(layer_idx, &input)?;
        total += goodness(&activations);
    }
    Ok(total / samples.len() as f32)
}

/// Trains the VFN using the Forward-Forward algorithm.
///
/// Updates each layer independently. For each layer:
/// - Positive samples: pushes goodness above the threshold
/// - Negative samples: pushes goodness below the threshold
///
/// No backpropagation — gradients never flow between layers.
/// VRAM usage is approximately 1x inference (one layer active at a time).
///
/// # Errors
///
/// Returns [`VoltError::LearnError`] if no samples are provided or
/// training encounters numerical issues.
///
/// # Example
///
/// ```
/// use volt_learn::forward_forward::{train_ff, FfSample, FfConfig};
/// use volt_soft::vfn::Vfn;
/// use volt_core::SLOT_DIM;
///
/// let mut vfn = Vfn::new_random(42);
/// let pos = FfSample { embedding: [0.1; SLOT_DIM], is_positive: true };
/// let neg = FfSample { embedding: [0.5; SLOT_DIM], is_positive: false };
/// let config = FfConfig { num_epochs: 2, ..FfConfig::default() };
/// let result = train_ff(&mut vfn, &[pos, neg], &config).unwrap();
/// assert_eq!(result.layers_updated, 3);
/// ```
pub fn train_ff(
    vfn: &mut Vfn,
    samples: &[FfSample],
    config: &FfConfig,
) -> Result<FfResult, VoltError> {
    if samples.is_empty() {
        return Err(VoltError::LearnError {
            message: "train_ff: no samples provided".to_string(),
        });
    }

    let positive_samples: Vec<&FfSample> =
        samples.iter().filter(|s| s.is_positive).collect();
    let negative_samples: Vec<&FfSample> =
        samples.iter().filter(|s| !s.is_positive).collect();

    let n_layers = vfn.layer_count();
    let mut pos_goodness_before = Vec::with_capacity(n_layers);
    let mut pos_goodness_after = Vec::with_capacity(n_layers);
    let mut neg_goodness_before = Vec::with_capacity(n_layers);
    let mut neg_goodness_after = Vec::with_capacity(n_layers);

    for layer_idx in 0..n_layers {
        let prev_layers: Vec<usize> = (0..layer_idx).collect();

        // Measure goodness before training
        let pos_before = avg_goodness_for_samples(
            vfn,
            layer_idx,
            &positive_samples,
            &prev_layers,
        )?;
        let neg_before = avg_goodness_for_samples(
            vfn,
            layer_idx,
            &negative_samples,
            &prev_layers,
        )?;
        pos_goodness_before.push(pos_before);
        neg_goodness_before.push(neg_before);

        let (in_dim, out_dim) = vfn.layer_shape(layer_idx)?;

        // Train this layer for num_epochs
        for _epoch in 0..config.num_epochs {
            for sample in samples {
                // Forward through previous layers (detached)
                let mut input: Vec<f32> = sample.embedding.to_vec();
                for &prev in &prev_layers {
                    input = vfn.forward_layer(prev, &input)?;
                }

                // Forward through current layer
                let activations = vfn.forward_layer(layer_idx, &input)?;
                let g = goodness(&activations);

                let needs_update = if sample.is_positive {
                    g < config.goodness_threshold
                } else {
                    g > config.goodness_threshold
                };

                if !needs_update {
                    continue;
                }

                // Compute gradient of goodness w.r.t. weights.
                // goodness = Σ a_i²
                // d(goodness)/d(w_ij) = 2 * a_i * d(a_i)/d(w_ij)
                //
                // For a linear layer (pre-ReLU): a_i = Σ_j w_ij * x_j + b_i
                // d(a_i)/d(w_ij) = x_j (if ReLU active, i.e. a_i > 0)
                //
                // So d(goodness)/d(w_ij) = 2 * a_i * x_j * relu_mask_i
                // And d(goodness)/d(b_i) = 2 * a_i * relu_mask_i
                //
                // For the output layer (no ReLU): relu_mask is always 1.

                let mut weight_deltas = vec![0.0f32; in_dim * out_dim];
                let mut bias_deltas = vec![0.0f32; out_dim];

                let sign = if sample.is_positive { 1.0 } else { -1.0 };

                for i in 0..out_dim {
                    // ReLU mask: for hidden layers, only update if activation > 0
                    let relu_active =
                        layer_idx >= 2 || activations[i] > 0.0;
                    if !relu_active {
                        continue;
                    }

                    let grad_factor = sign * 2.0 * activations[i];

                    // Weight gradient
                    let row_start = i * in_dim;
                    for j in 0..in_dim {
                        weight_deltas[row_start + j] = grad_factor * input[j];
                    }

                    // Bias gradient
                    bias_deltas[i] = grad_factor;
                }

                vfn.update_layer(
                    layer_idx,
                    &weight_deltas,
                    &bias_deltas,
                    config.learning_rate,
                )?;
            }
        }

        // Measure goodness after training
        let pos_after = avg_goodness_for_samples(
            vfn,
            layer_idx,
            &positive_samples,
            &prev_layers,
        )?;
        let neg_after = avg_goodness_for_samples(
            vfn,
            layer_idx,
            &negative_samples,
            &prev_layers,
        )?;
        pos_goodness_after.push(pos_after);
        neg_goodness_after.push(neg_after);
    }

    Ok(FfResult {
        layers_updated: n_layers,
        positive_goodness_before: pos_goodness_before,
        positive_goodness_after: pos_goodness_after,
        negative_goodness_before: neg_goodness_before,
        negative_goodness_after: neg_goodness_after,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::meta::DiscourseType;

    fn make_positive_sample(val: f32) -> FfSample {
        let mut embedding = [val; SLOT_DIM];
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in &mut embedding {
            *v /= norm;
        }
        FfSample {
            embedding,
            is_positive: true,
        }
    }

    fn make_negative_sample(val: f32) -> FfSample {
        let mut embedding = [val; SLOT_DIM];
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in &mut embedding {
            *v /= norm;
        }
        FfSample {
            embedding,
            is_positive: false,
        }
    }

    #[test]
    fn default_config_sensible() {
        let config = FfConfig::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.goodness_threshold > 0.0);
        assert!(config.num_epochs > 0);
        assert!(config.positive_gamma_threshold > config.negative_gamma_threshold);
    }

    #[test]
    fn train_ff_empty_samples_errors() {
        let mut vfn = Vfn::new_random(42);
        let config = FfConfig::default();
        assert!(train_ff(&mut vfn, &[], &config).is_err());
    }

    #[test]
    fn train_ff_updates_all_layers() {
        let mut vfn = Vfn::new_random(42);
        let samples = vec![
            make_positive_sample(0.1),
            make_negative_sample(0.5),
        ];
        let config = FfConfig {
            num_epochs: 2,
            ..FfConfig::default()
        };
        let result = train_ff(&mut vfn, &samples, &config).unwrap();
        assert_eq!(result.layers_updated, 3);
        assert_eq!(result.positive_goodness_before.len(), 3);
        assert_eq!(result.positive_goodness_after.len(), 3);
    }

    #[test]
    fn train_ff_positive_goodness_increases() {
        let mut vfn = Vfn::new_random(42);
        let mut samples = Vec::new();
        for i in 0..20 {
            samples.push(make_positive_sample(0.1 + i as f32 * 0.01));
        }
        for i in 0..20 {
            samples.push(make_negative_sample(0.5 + i as f32 * 0.01));
        }
        let config = FfConfig {
            num_epochs: 10,
            learning_rate: 0.01,
            ..FfConfig::default()
        };
        let result = train_ff(&mut vfn, &samples, &config).unwrap();

        // For at least the first hidden layer, positive goodness should increase
        assert!(
            result.positive_goodness_after[0]
                >= result.positive_goodness_before[0],
            "positive goodness should increase for layer 0: before={}, after={}",
            result.positive_goodness_before[0],
            result.positive_goodness_after[0],
        );
    }

    #[test]
    fn collect_ff_samples_empty_events_errors() {
        let store = VoltStore::new();
        let config = FfConfig::default();
        assert!(collect_ff_samples(&[], &store, &config).is_err());
    }

    #[test]
    fn collect_ff_samples_no_matching_frames_errors() {
        let store = VoltStore::new();
        let config = FfConfig::default();
        let events = vec![LearningEvent {
            frame_id: 999, // Doesn't exist in store
            strand_id: 0,
            query_type: DiscourseType::Query,
            gamma_scores: [0.9; MAX_SLOTS],
            convergence_iterations: 5,
            ghost_activations: 0,
            timestamp: 1,
        }];
        assert!(collect_ff_samples(&events, &store, &config).is_err());
    }

    #[test]
    fn goodness_computation_correct() {
        let activations = vec![1.0, 2.0, 3.0];
        assert!((goodness(&activations) - 14.0).abs() < 1e-6);
    }

    #[test]
    fn goodness_zero_for_zero_activations() {
        let activations = vec![0.0; 512];
        assert!(goodness(&activations).abs() < 1e-10);
    }

    #[test]
    fn rng_gaussian_distribution() {
        let mut rng = Rng::new(42);
        let mut sum = 0.0;
        let n = 10_000;
        for _ in 0..n {
            sum += rng.next_gaussian(1.0);
        }
        let mean = sum / n as f32;
        // Mean should be close to 0.0 for large N
        assert!(
            mean.abs() < 0.1,
            "Gaussian mean should be ~0, got {mean}"
        );
    }

    #[test]
    fn ff_sample_creation() {
        let pos = make_positive_sample(0.1);
        assert!(pos.is_positive);
        let neg = make_negative_sample(0.5);
        assert!(!neg.is_positive);
    }
}
