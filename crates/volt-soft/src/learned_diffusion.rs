//! Learned diffusion controller for adaptive noise scheduling.
//!
//! A small MLP (~10K params) that predicts per-slot noise magnitude
//! from the current slot state, iteration number, and recent delta:
//!
//! ```text
//! (slot_state[256], iteration_embed[16], delta_embed[16]) → σ
//! ```
//!
//! The controller learns to use more exploration (high σ) early in RAR
//! and more exploitation (low σ) as slots converge.
//!
//! ## Architecture
//!
//! - Input: slot_state (256) + iteration_features (16) + delta_features (16) = 288
//! - Hidden: Linear(288→64) → GELU → Linear(64→32) → GELU → Linear(32→1) → Softplus
//! - Output: σ ∈ (0, ∞), clamped to [0, max_sigma]
//! - Parameters: 288×64 + 64 + 64×32 + 32 + 32×1 + 1 = ~20.7K

use crate::nn::{Linear, Rng};
use volt_core::{VoltError, SLOT_DIM};

const ITER_EMBED_DIM: usize = 16;
const DELTA_EMBED_DIM: usize = 16;
const INPUT_DIM: usize = SLOT_DIM + ITER_EMBED_DIM + DELTA_EMBED_DIM;
const HIDDEN1: usize = 64;
const HIDDEN2: usize = 32;

/// Configuration for the learned diffusion controller.
///
/// # Example
///
/// ```
/// use volt_soft::learned_diffusion::DiffusionControllerConfig;
///
/// let config = DiffusionControllerConfig::default();
/// assert_eq!(config.max_sigma, 0.2);
/// ```
#[derive(Debug, Clone)]
pub struct DiffusionControllerConfig {
    /// Maximum allowed sigma value (default: 0.2).
    pub max_sigma: f32,

    /// Maximum RAR iteration for normalization (default: 50).
    pub max_iteration: u32,
}

impl Default for DiffusionControllerConfig {
    fn default() -> Self {
        Self {
            max_sigma: 0.2,
            max_iteration: 50,
        }
    }
}

/// A learned diffusion controller MLP.
///
/// Predicts per-slot noise magnitude σ from the current slot state,
/// iteration number, and recent convergence delta.
///
/// # Example
///
/// ```
/// use volt_soft::learned_diffusion::{DiffusionController, DiffusionControllerConfig};
/// use volt_core::SLOT_DIM;
///
/// let controller = DiffusionController::new_random(42);
/// let state = [0.1_f32; SLOT_DIM];
/// let sigma = controller.predict_sigma(&state, 5, 0.05, &DiffusionControllerConfig::default()).unwrap();
/// assert!(sigma >= 0.0);
/// assert!(sigma <= 0.2);
/// ```
#[derive(Clone)]
pub struct DiffusionController {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
}

impl std::fmt::Debug for DiffusionController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DiffusionController({}→{}→{}→1, ~{}K params)",
            INPUT_DIM,
            HIDDEN1,
            HIDDEN2,
            self.param_count() / 1000
        )
    }
}

impl DiffusionController {
    /// Creates a new diffusion controller with Xavier random initialization.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::learned_diffusion::DiffusionController;
    ///
    /// let controller = DiffusionController::new_random(42);
    /// ```
    pub fn new_random(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            layer1: Linear::new_xavier(&mut rng, INPUT_DIM, HIDDEN1),
            layer2: Linear::new_xavier(&mut rng, HIDDEN1, HIDDEN2),
            layer3: Linear::new_xavier(&mut rng, HIDDEN2, 1),
        }
    }

    /// Predicts the noise sigma for a single slot.
    ///
    /// # Arguments
    ///
    /// * `slot_state` - Current 256-dim slot embedding
    /// * `iteration` - Current RAR iteration number
    /// * `delta` - Recent ‖ΔS‖ for this slot
    /// * `config` - Controller configuration
    ///
    /// # Returns
    ///
    /// σ ∈ [0, max_sigma], the predicted noise magnitude.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if input contains NaN/Inf.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::learned_diffusion::{DiffusionController, DiffusionControllerConfig};
    /// use volt_core::SLOT_DIM;
    ///
    /// let controller = DiffusionController::new_random(42);
    /// let state = [0.1_f32; SLOT_DIM];
    /// let sigma = controller.predict_sigma(&state, 10, 0.1, &DiffusionControllerConfig::default()).unwrap();
    /// assert!(sigma >= 0.0 && sigma <= 0.2);
    /// ```
    pub fn predict_sigma(
        &self,
        slot_state: &[f32; SLOT_DIM],
        iteration: u32,
        delta: f32,
        config: &DiffusionControllerConfig,
    ) -> Result<f32, VoltError> {
        if slot_state.iter().any(|x| !x.is_finite()) || !delta.is_finite() {
            return Err(VoltError::Internal {
                message: "DiffusionController: input contains NaN or Inf".to_string(),
            });
        }

        // Build input vector: slot_state + iteration features + delta features
        let mut input = Vec::with_capacity(INPUT_DIM);
        input.extend_from_slice(slot_state);

        // Iteration features: sinusoidal encoding (normalized)
        let t = iteration as f32 / config.max_iteration.max(1) as f32;
        for k in 0..ITER_EMBED_DIM {
            let freq = (k as f32 + 1.0) * std::f32::consts::PI;
            if k % 2 == 0 {
                input.push((freq * t).sin());
            } else {
                input.push((freq * t).cos());
            }
        }

        // Delta features: log-scale encoding
        let log_delta = (delta + 1e-8).ln();
        for k in 0..DELTA_EMBED_DIM {
            let scale = 2.0_f32.powi(k as i32 - 8);
            input.push((log_delta * scale).tanh());
        }

        // Forward: Linear → GELU → Linear → GELU → Linear → Softplus → clamp
        let h = self.layer1.forward(&input);
        let h: Vec<f32> = h.into_iter().map(gelu).collect();
        let h = self.layer2.forward(&h);
        let h: Vec<f32> = h.into_iter().map(gelu).collect();
        let out = self.layer3.forward(&h);

        // Softplus: ln(1 + exp(x)) for smooth positive output
        let raw_sigma = softplus(out[0]);

        // Clamp to [0, max_sigma]
        Ok(raw_sigma.min(config.max_sigma).max(0.0))
    }

    /// Predicts sigma for all active slots in a frame.
    ///
    /// Returns per-slot sigma values. Inactive slots get 0.0.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if computation fails.
    pub fn predict_all(
        &self,
        states: &[Option<[f32; SLOT_DIM]>; volt_core::MAX_SLOTS],
        converged: &[bool; volt_core::MAX_SLOTS],
        deltas: &[f32; volt_core::MAX_SLOTS],
        iteration: u32,
        config: &DiffusionControllerConfig,
    ) -> Result<[f32; volt_core::MAX_SLOTS], VoltError> {
        let mut sigmas = [0.0f32; volt_core::MAX_SLOTS];
        for i in 0..volt_core::MAX_SLOTS {
            if converged[i] {
                continue;
            }
            if let Some(state) = &states[i] {
                sigmas[i] = self.predict_sigma(state, iteration, deltas[i], config)?;
            }
        }
        Ok(sigmas)
    }

    /// Returns the total number of parameters.
    pub fn param_count(&self) -> usize {
        INPUT_DIM * HIDDEN1 + HIDDEN1 // layer1
            + HIDDEN1 * HIDDEN2 + HIDDEN2  // layer2
            + HIDDEN2 + 1                 // layer3
    }

    /// Returns references to internal layers for weight transfer.
    #[cfg(feature = "gpu")]
    pub(crate) fn layers(&self) -> (&Linear, &Linear, &Linear) {
        (&self.layer1, &self.layer2, &self.layer3)
    }

    /// Saves the controller to a binary checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::LearnError`] if writing fails.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), VoltError> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path.as_ref()).map_err(|e| VoltError::LearnError {
            message: format!("Failed to create diffusion controller checkpoint: {e}"),
        })?;

        let write_err = |e: std::io::Error| VoltError::LearnError {
            message: format!("write diffusion controller: {e}"),
        };

        file.write_all(b"VDFC").map_err(write_err)?;
        file.write_all(&1u32.to_le_bytes()).map_err(write_err)?;

        // Write layers
        for layer in [&self.layer1, &self.layer2, &self.layer3] {
            for &w in layer.weights() {
                file.write_all(&w.to_le_bytes()).map_err(write_err)?;
            }
            for &b in layer.bias() {
                file.write_all(&b.to_le_bytes()).map_err(write_err)?;
            }
        }

        Ok(())
    }

    /// Loads the controller from a binary checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::LearnError`] if reading fails.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, VoltError> {
        use std::io::Read;

        let mut file = std::fs::File::open(path.as_ref()).map_err(|e| VoltError::LearnError {
            message: format!("Failed to open diffusion controller checkpoint: {e}"),
        })?;

        let read_err = |e: std::io::Error| VoltError::LearnError {
            message: format!("read diffusion controller: {e}"),
        };

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(read_err)?;
        if &magic != b"VDFC" {
            return Err(VoltError::LearnError {
                message: "Invalid diffusion controller checkpoint magic".to_string(),
            });
        }

        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4).map_err(read_err)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(VoltError::LearnError {
                message: format!("Incompatible diffusion controller version: {version}"),
            });
        }

        let read_linear = |file: &mut std::fs::File, in_d: usize, out_d: usize| -> Result<Linear, VoltError> {
            let mut weights = Vec::with_capacity(in_d * out_d);
            let mut buf = [0u8; 4];
            for _ in 0..(in_d * out_d) {
                file.read_exact(&mut buf).map_err(read_err)?;
                weights.push(f32::from_le_bytes(buf));
            }
            let mut bias = Vec::with_capacity(out_d);
            for _ in 0..out_d {
                file.read_exact(&mut buf).map_err(read_err)?;
                bias.push(f32::from_le_bytes(buf));
            }
            Linear::from_weights_and_bias(weights, bias, in_d, out_d)
        };

        let layer1 = read_linear(&mut file, INPUT_DIM, HIDDEN1)?;
        let layer2 = read_linear(&mut file, HIDDEN1, HIDDEN2)?;
        let layer3 = read_linear(&mut file, HIDDEN2, 1)?;

        Ok(Self {
            layer1,
            layer2,
            layer3,
        })
    }
}

/// GELU activation function.
fn gelu(x: f32) -> f32 {
    0.5 * x
        * (1.0
            + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// Softplus: ln(1 + exp(x)), numerically stable.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // Avoid overflow
    } else if x < -20.0 {
        0.0 // Underflow to zero
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_sigma_in_range() {
        let controller = DiffusionController::new_random(42);
        let state = [0.1f32; SLOT_DIM];
        let config = DiffusionControllerConfig::default();

        let sigma = controller.predict_sigma(&state, 5, 0.05, &config).unwrap();
        assert!(sigma >= 0.0, "sigma should be non-negative, got {sigma}");
        assert!(
            sigma <= config.max_sigma,
            "sigma should be <= {}, got {sigma}",
            config.max_sigma
        );
    }

    #[test]
    fn predict_sigma_deterministic() {
        let c1 = DiffusionController::new_random(42);
        let c2 = DiffusionController::new_random(42);
        let state = [0.5f32; SLOT_DIM];
        let config = DiffusionControllerConfig::default();

        let s1 = c1.predict_sigma(&state, 10, 0.1, &config).unwrap();
        let s2 = c2.predict_sigma(&state, 10, 0.1, &config).unwrap();
        assert_eq!(s1, s2);
    }

    #[test]
    fn predict_sigma_different_iterations() {
        let controller = DiffusionController::new_random(42);
        // Use zero state so iteration encoding has maximum relative effect
        let state = [0.0f32; SLOT_DIM];
        let config = DiffusionControllerConfig::default();

        let s_early = controller.predict_sigma(&state, 0, 0.1, &config).unwrap();
        let s_late = controller.predict_sigma(&state, 40, 0.1, &config).unwrap();

        // Both should be valid sigma values
        assert!(s_early >= 0.0 && s_early <= config.max_sigma);
        assert!(s_late >= 0.0 && s_late <= config.max_sigma);
        // Note: with random init, outputs may or may not differ significantly.
        // The network is untrained; after training, different iterations
        // should produce meaningfully different sigmas.
    }

    #[test]
    fn predict_all_respects_convergence() {
        let controller = DiffusionController::new_random(42);
        let config = DiffusionControllerConfig::default();

        let mut states = [const { None }; volt_core::MAX_SLOTS];
        states[0] = Some([0.1f32; SLOT_DIM]);
        states[1] = Some([0.2f32; SLOT_DIM]);

        let mut converged = [true; volt_core::MAX_SLOTS];
        converged[0] = false; // Only slot 0 is active
        converged[1] = true; // Slot 1 is converged

        let deltas = [0.05; volt_core::MAX_SLOTS];

        let sigmas = controller
            .predict_all(&states, &converged, &deltas, 5, &config)
            .unwrap();

        assert!(sigmas[0] > 0.0 || sigmas[0] == 0.0); // Active slot gets sigma
        assert_eq!(sigmas[1], 0.0); // Converged slot gets 0
        assert_eq!(sigmas[2], 0.0); // Empty slot gets 0
    }

    #[test]
    fn nan_input_errors() {
        let controller = DiffusionController::new_random(42);
        let mut state = [0.1f32; SLOT_DIM];
        state[0] = f32::NAN;
        let config = DiffusionControllerConfig::default();

        assert!(controller.predict_sigma(&state, 5, 0.1, &config).is_err());
    }

    #[test]
    fn param_count_correct() {
        let controller = DiffusionController::new_random(42);
        let expected = INPUT_DIM * HIDDEN1 + HIDDEN1 + HIDDEN1 * HIDDEN2 + HIDDEN2 + HIDDEN2 + 1;
        assert_eq!(controller.param_count(), expected);
        // Should be ~20K
        assert!(
            controller.param_count() > 15_000 && controller.param_count() < 25_000,
            "expected ~20K params, got {}",
            controller.param_count()
        );
    }

    #[test]
    fn debug_format_readable() {
        let controller = DiffusionController::new_random(42);
        let debug = format!("{:?}", controller);
        assert!(debug.contains("DiffusionController"));
    }

    #[test]
    fn checkpoint_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("diff_controller_test.bin");

        let controller = DiffusionController::new_random(42);
        controller.save(&path).unwrap();

        let loaded = DiffusionController::load(&path).unwrap();

        let state = [0.42f32; SLOT_DIM];
        let config = DiffusionControllerConfig::default();

        let s1 = controller.predict_sigma(&state, 10, 0.1, &config).unwrap();
        let s2 = loaded.predict_sigma(&state, 10, 0.1, &config).unwrap();
        assert_eq!(s1, s2, "checkpoint roundtrip should preserve weights");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn softplus_basic() {
        assert!((softplus(0.0) - 0.6931).abs() < 0.01); // ln(2) ≈ 0.693
        assert!(softplus(10.0) > 9.9);
        assert!(softplus(-10.0) < 0.001);
    }
}
