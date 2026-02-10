//! Diffusion noise injection for the RAR Refine phase.
//!
//! Adds per-slot adaptive Gaussian noise to the RAR state update,
//! enabling exploration of the energy landscape. The SDE becomes:
//!
//! ```text
//! S_i(t+1) = normalize(S_i(t) + dt*(drift_i + β·msg_i) + σ_i·noise_i)
//! ```
//!
//! where `σ_i` is an adaptive per-slot standard deviation and `noise_i`
//! is sampled from N(0, I).
//!
//! ## Adaptive Sigma
//!
//! - Converging slots (delta shrinking) get `σ *= decay` (less exploration)
//! - Stuck slots (delta not shrinking) get `σ *= growth` (more exploration)
//! - Converged/frozen slots get `σ = 0`
//!
//! ## Default Behavior
//!
//! With default configuration (`sigma = [0.0; MAX_SLOTS]`), no noise is
//! injected, preserving exact Milestone 2.3 behavior.

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use volt_core::{VoltError, MAX_SLOTS, SLOT_DIM};

/// Per-slot diffusion noise configuration.
///
/// Controls the stochastic component of the RAR state update.
/// Each slot has an independent sigma controlling noise magnitude.
///
/// # Example
///
/// ```
/// use volt_soft::diffusion::DiffusionConfig;
/// use volt_core::MAX_SLOTS;
///
/// // Default: no noise (backward compatible)
/// let config = DiffusionConfig::default();
/// assert!(config.sigma.iter().all(|&s| s == 0.0));
///
/// // Creative mode: moderate noise on all slots
/// let mut creative = DiffusionConfig::default();
/// for s in &mut creative.sigma {
///     *s = 0.05;
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionConfig {
    /// Per-slot noise standard deviation. `sigma[i] = 0.0` means no noise.
    pub sigma: [f32; MAX_SLOTS],

    /// Global noise scale multiplier (default 1.0).
    /// Final noise magnitude = `sigma[i] * noise_scale`.
    pub noise_scale: f32,

    /// RNG seed for reproducible noise generation.
    pub seed: u64,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            sigma: [0.0; MAX_SLOTS],
            noise_scale: 1.0,
            seed: 0,
        }
    }
}

impl DiffusionConfig {
    /// Returns true if all sigma values are zero (no noise will be injected).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::diffusion::DiffusionConfig;
    ///
    /// assert!(DiffusionConfig::default().is_silent());
    /// ```
    pub fn is_silent(&self) -> bool {
        self.sigma.iter().all(|&s| s == 0.0) || self.noise_scale == 0.0
    }

    /// Creates a uniform diffusion config where all active slots share
    /// the same sigma value.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::diffusion::DiffusionConfig;
    ///
    /// let config = DiffusionConfig::uniform(0.05, 42);
    /// assert!(config.sigma.iter().all(|&s| s == 0.05));
    /// ```
    pub fn uniform(sigma: f32, seed: u64) -> Self {
        Self {
            sigma: [sigma; MAX_SLOTS],
            noise_scale: 1.0,
            seed,
        }
    }
}

/// Generates one round of per-slot Gaussian noise vectors.
///
/// Returns noise vectors for each slot. Slots with `sigma = 0` or
/// where `active_mask[i]` is false receive `None` (no noise).
/// Active slots with non-zero sigma receive `N(0, sigma_i * noise_scale)` noise.
///
/// # Errors
///
/// Returns [`VoltError::Internal`] if sigma values produce invalid distributions.
///
/// # Example
///
/// ```
/// use volt_soft::diffusion::{DiffusionConfig, generate_noise};
/// use volt_core::MAX_SLOTS;
///
/// let config = DiffusionConfig::uniform(0.05, 42);
/// let active = [true; MAX_SLOTS];
/// let noise = generate_noise(&config, &active, 0).unwrap();
/// assert!(noise[0].is_some()); // active slot with sigma > 0
/// ```
pub fn generate_noise(
    config: &DiffusionConfig,
    active_mask: &[bool; MAX_SLOTS],
    iteration: u32,
) -> Result<[Option<[f32; SLOT_DIM]>; MAX_SLOTS], VoltError> {
    let mut result: [Option<[f32; SLOT_DIM]>; MAX_SLOTS] = [const { None }; MAX_SLOTS];

    if config.is_silent() {
        return Ok(result);
    }

    // Seed varies per iteration for different noise each step
    let iter_seed = config.seed.wrapping_add(iteration as u64).wrapping_mul(0x9e3779b97f4a7c15);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(iter_seed);

    for i in 0..MAX_SLOTS {
        if !active_mask[i] || config.sigma[i] == 0.0 {
            continue;
        }

        let effective_sigma = config.sigma[i] * config.noise_scale;
        if effective_sigma <= 0.0 {
            continue;
        }

        let dist = Normal::new(0.0f32, effective_sigma).map_err(|e| VoltError::Internal {
            message: format!("diffusion: invalid sigma for slot {i}: {e}"),
        })?;

        let mut noise = [0.0f32; SLOT_DIM];
        for x in &mut noise {
            *x = dist.sample(&mut rng);
        }
        result[i] = Some(noise);
    }

    Ok(result)
}

/// Adapts per-slot sigma based on convergence progress.
///
/// - Converging slots (delta < epsilon) get `sigma *= decay`
/// - Non-converging slots get `sigma *= growth`
/// - Already-converged (frozen) slots get `sigma = 0`
///
/// This creates an exploration/exploitation tradeoff: slots far from
/// convergence explore more, while near-converged slots settle down.
///
/// # Example
///
/// ```
/// use volt_soft::diffusion::{DiffusionConfig, adapt_sigma};
/// use volt_core::MAX_SLOTS;
///
/// let mut config = DiffusionConfig::uniform(0.1, 42);
/// let deltas = [0.001; MAX_SLOTS]; // all slots nearly converged
/// let converged = [false; MAX_SLOTS];
/// adapt_sigma(&mut config, &deltas, &converged, 0.01, 0.95, 1.05);
/// // sigmas should have decayed
/// assert!(config.sigma[0] < 0.1);
/// ```
pub fn adapt_sigma(
    config: &mut DiffusionConfig,
    deltas: &[f32; MAX_SLOTS],
    converged: &[bool; MAX_SLOTS],
    epsilon: f32,
    decay: f32,
    growth: f32,
) {
    for i in 0..MAX_SLOTS {
        if converged[i] {
            config.sigma[i] = 0.0;
        } else if deltas[i] < epsilon {
            config.sigma[i] *= decay;
        } else {
            config.sigma[i] *= growth;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_silent() {
        let config = DiffusionConfig::default();
        assert!(config.is_silent());
    }

    #[test]
    fn uniform_creates_consistent_sigma() {
        let config = DiffusionConfig::uniform(0.05, 42);
        assert!(config.sigma.iter().all(|&s| (s - 0.05).abs() < 1e-10));
        assert!(!config.is_silent());
    }

    #[test]
    fn silent_config_produces_no_noise() {
        let config = DiffusionConfig::default();
        let active = [true; MAX_SLOTS];
        let noise = generate_noise(&config, &active, 0).unwrap();
        for n in &noise {
            assert!(n.is_none());
        }
    }

    #[test]
    fn inactive_slots_get_no_noise() {
        let config = DiffusionConfig::uniform(0.1, 42);
        let mut active = [false; MAX_SLOTS];
        active[3] = true;
        let noise = generate_noise(&config, &active, 0).unwrap();
        for (i, n) in noise.iter().enumerate() {
            if i == 3 {
                assert!(n.is_some(), "active slot should get noise");
            } else {
                assert!(n.is_none(), "inactive slot {} should get no noise", i);
            }
        }
    }

    #[test]
    fn noise_is_finite() {
        let config = DiffusionConfig::uniform(0.1, 42);
        let active = [true; MAX_SLOTS];
        let noise = generate_noise(&config, &active, 0).unwrap();
        for (i, n) in noise.iter().enumerate() {
            if let Some(v) = n {
                assert!(
                    v.iter().all(|x| x.is_finite()),
                    "slot {} has non-finite noise",
                    i,
                );
            }
        }
    }

    #[test]
    fn noise_is_deterministic_same_seed() {
        let config = DiffusionConfig::uniform(0.1, 42);
        let active = [true; MAX_SLOTS];
        let n1 = generate_noise(&config, &active, 0).unwrap();
        let n2 = generate_noise(&config, &active, 0).unwrap();
        assert_eq!(n1, n2);
    }

    #[test]
    fn noise_differs_across_iterations() {
        let config = DiffusionConfig::uniform(0.1, 42);
        let active = [true; MAX_SLOTS];
        let n1 = generate_noise(&config, &active, 0).unwrap();
        let n2 = generate_noise(&config, &active, 1).unwrap();
        // At least one slot's noise should differ
        let differs = n1.iter().zip(n2.iter()).any(|(a, b)| a != b);
        assert!(differs, "noise should differ between iterations");
    }

    #[test]
    fn noise_differs_across_seeds() {
        let c1 = DiffusionConfig::uniform(0.1, 42);
        let c2 = DiffusionConfig::uniform(0.1, 99);
        let active = [true; MAX_SLOTS];
        let n1 = generate_noise(&c1, &active, 0).unwrap();
        let n2 = generate_noise(&c2, &active, 0).unwrap();
        let differs = n1.iter().zip(n2.iter()).any(|(a, b)| a != b);
        assert!(differs, "different seeds should produce different noise");
    }

    #[test]
    fn adapt_sigma_decays_converging() {
        let mut config = DiffusionConfig::uniform(0.1, 42);
        let deltas = [0.001; MAX_SLOTS];
        let converged = [false; MAX_SLOTS];
        adapt_sigma(&mut config, &deltas, &converged, 0.01, 0.95, 1.05);
        for &s in &config.sigma {
            assert!((s - 0.095).abs() < 1e-6, "expected 0.095, got {}", s);
        }
    }

    #[test]
    fn adapt_sigma_grows_stuck() {
        let mut config = DiffusionConfig::uniform(0.1, 42);
        let deltas = [0.5; MAX_SLOTS]; // well above epsilon
        let converged = [false; MAX_SLOTS];
        adapt_sigma(&mut config, &deltas, &converged, 0.01, 0.95, 1.05);
        for &s in &config.sigma {
            assert!((s - 0.105).abs() < 1e-6, "expected 0.105, got {}", s);
        }
    }

    #[test]
    fn adapt_sigma_zeros_converged() {
        let mut config = DiffusionConfig::uniform(0.1, 42);
        let deltas = [0.0; MAX_SLOTS];
        let converged = [true; MAX_SLOTS];
        adapt_sigma(&mut config, &deltas, &converged, 0.01, 0.95, 1.05);
        for &s in &config.sigma {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn zero_noise_scale_is_silent() {
        let mut config = DiffusionConfig::uniform(0.1, 42);
        config.noise_scale = 0.0;
        assert!(config.is_silent());
    }
}
