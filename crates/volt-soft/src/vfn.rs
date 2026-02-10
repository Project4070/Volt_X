//! Vector Field Network (VFN) — the slot-local neural network.
//!
//! The VFN is a simple MLP that takes a 256-dim slot embedding and produces
//! a 256-dim drift vector. In the RAR loop, the drift guides each slot's
//! evolution toward a semantic attractor.
//!
//! ## Architecture (Milestone 2.3)
//!
//! Four layer sizes (256 → 512 → 512 → 256) with ReLU activations
//! between hidden layers:
//! - Linear(256 → 512) + ReLU
//! - Linear(512 → 512) + ReLU
//! - Linear(512 → 256), no activation
//!
//! Weights are randomly initialized (Xavier/Glorot). Training comes in
//! Milestone 2.4 (Flow Matching on GPU).

use crate::nn::{Linear, Rng};
use volt_core::{VoltError, SLOT_DIM};

/// Hidden dimension for the VFN's intermediate layers.
const HIDDEN_DIM: usize = 512;

/// A Vector Field Network: slot-local MLP for RAR inference.
///
/// Takes a 256-dim slot embedding and produces a 256-dim drift vector.
/// The drift represents the direction the slot should evolve during
/// one RAR iteration.
///
/// # Example
///
/// ```
/// use volt_soft::vfn::Vfn;
/// use volt_core::SLOT_DIM;
///
/// let vfn = Vfn::new_random(42);
/// let input = [0.1_f32; SLOT_DIM];
/// let drift = vfn.forward(&input).unwrap();
/// assert!(drift.iter().all(|x| x.is_finite()));
/// ```
#[derive(Clone)]
pub struct Vfn {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
}

impl std::fmt::Debug for Vfn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Vfn({}→{}→{}→{})",
            self.layer1.in_dim(),
            self.layer1.out_dim(),
            self.layer2.out_dim(),
            self.layer3.out_dim()
        )
    }
}

impl Vfn {
    /// Creates a new VFN with Xavier/Glorot random weight initialization.
    ///
    /// The seed determines the initial weights. Same seed produces identical
    /// weights (deterministic).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::vfn::Vfn;
    ///
    /// let vfn = Vfn::new_random(42);
    /// ```
    pub fn new_random(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            layer1: Linear::new_xavier(&mut rng, SLOT_DIM, HIDDEN_DIM),
            layer2: Linear::new_xavier(&mut rng, HIDDEN_DIM, HIDDEN_DIM),
            layer3: Linear::new_xavier(&mut rng, HIDDEN_DIM, SLOT_DIM),
        }
    }

    /// Computes the drift vector for a single slot embedding.
    ///
    /// Passes the input through 3 linear layers with ReLU activations
    /// on the first two. Returns a 256-dim drift vector.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if the input or output contains NaN or Inf.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::vfn::Vfn;
    /// use volt_core::SLOT_DIM;
    ///
    /// let vfn = Vfn::new_random(42);
    /// let input = [0.1_f32; SLOT_DIM];
    /// let drift = vfn.forward(&input).unwrap();
    /// assert!(drift.iter().all(|x| x.is_finite()));
    /// ```
    pub fn forward(&self, input: &[f32; SLOT_DIM]) -> Result<[f32; SLOT_DIM], VoltError> {
        // Validate input
        if input.iter().any(|x| !x.is_finite()) {
            return Err(VoltError::Internal {
                message: "VFN forward: input contains NaN or Inf".to_string(),
            });
        }

        // Layer 1: 256 → 512, ReLU
        let h1 = self.layer1.forward(input);
        let h1: Vec<f32> = h1.into_iter().map(|x| x.max(0.0)).collect();

        // Layer 2: 512 → 512, ReLU
        let h2 = self.layer2.forward(&h1);
        let h2: Vec<f32> = h2.into_iter().map(|x| x.max(0.0)).collect();

        // Layer 3: 512 → 256, no activation
        let out = self.layer3.forward(&h2);

        // Convert to fixed-size array
        let mut result = [0.0f32; SLOT_DIM];
        result.copy_from_slice(&out);

        // Validate output
        if result.iter().any(|x| !x.is_finite()) {
            return Err(VoltError::Internal {
                message: "VFN forward: output contains NaN or Inf".to_string(),
            });
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_produces_correct_size() {
        let vfn = Vfn::new_random(42);
        let input = [0.1; SLOT_DIM];
        let output = vfn.forward(&input).unwrap();
        // output is [f32; SLOT_DIM], so size is guaranteed by type system
        assert_eq!(output.len(), SLOT_DIM);
    }

    #[test]
    fn forward_output_is_finite() {
        let vfn = Vfn::new_random(42);
        // Test with various inputs
        for seed in 0..20 {
            let mut input = [0.0f32; SLOT_DIM];
            let mut rng = Rng::new(seed);
            for x in &mut input {
                *x = rng.next_f32_range(-1.0, 1.0);
            }
            let output = vfn.forward(&input).unwrap();
            assert!(
                output.iter().all(|x| x.is_finite()),
                "VFN output contains NaN/Inf for seed {}",
                seed
            );
        }
    }

    #[test]
    fn forward_deterministic() {
        let vfn1 = Vfn::new_random(42);
        let vfn2 = Vfn::new_random(42);
        let input = [0.5; SLOT_DIM];
        let out1 = vfn1.forward(&input).unwrap();
        let out2 = vfn2.forward(&input).unwrap();
        assert_eq!(out1, out2);
    }

    #[test]
    fn different_seeds_different_output() {
        let vfn1 = Vfn::new_random(42);
        let vfn2 = Vfn::new_random(43);
        let input = [0.5; SLOT_DIM];
        let out1 = vfn1.forward(&input).unwrap();
        let out2 = vfn2.forward(&input).unwrap();
        assert_ne!(out1, out2);
    }

    #[test]
    fn nan_input_errors() {
        let vfn = Vfn::new_random(42);
        let mut input = [0.1; SLOT_DIM];
        input[0] = f32::NAN;
        assert!(vfn.forward(&input).is_err());
    }

    #[test]
    fn inf_input_errors() {
        let vfn = Vfn::new_random(42);
        let mut input = [0.1; SLOT_DIM];
        input[0] = f32::INFINITY;
        assert!(vfn.forward(&input).is_err());
    }

    #[test]
    fn zero_input_produces_zero_output() {
        let vfn = Vfn::new_random(42);
        let input = [0.0; SLOT_DIM];
        let output = vfn.forward(&input).unwrap();
        // With zero bias and zero input, ReLU of zero is zero,
        // so all layers produce zero
        for x in &output {
            assert!(x.abs() < 1e-10, "expected zero output, got {}", x);
        }
    }

    #[test]
    fn debug_format_readable() {
        let vfn = Vfn::new_random(42);
        let debug = format!("{:?}", vfn);
        assert!(debug.contains("Vfn(256→512→512→256)"));
    }
}
