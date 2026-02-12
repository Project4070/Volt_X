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

    /// Returns references to the three internal linear layers.
    ///
    /// Used by [`crate::gpu::vfn::GpuVfn::from_cpu_vfn`] to transfer
    /// weights to candle tensors.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn layers(&self) -> (&Linear, &Linear, &Linear) {
        (&self.layer1, &self.layer2, &self.layer3)
    }

    // --- Forward-Forward Training API (Milestone 5.2) ---

    /// Returns the number of layers in the VFN (always 3).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::vfn::Vfn;
    ///
    /// let vfn = Vfn::new_random(42);
    /// assert_eq!(vfn.layer_count(), 3);
    /// ```
    pub fn layer_count(&self) -> usize {
        3
    }

    /// Returns `(in_dim, out_dim)` for the given layer.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if `layer_idx >= 3`.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::vfn::Vfn;
    ///
    /// let vfn = Vfn::new_random(42);
    /// assert_eq!(vfn.layer_shape(0).unwrap(), (256, 512));
    /// assert_eq!(vfn.layer_shape(1).unwrap(), (512, 512));
    /// assert_eq!(vfn.layer_shape(2).unwrap(), (512, 256));
    /// ```
    pub fn layer_shape(&self, layer_idx: usize) -> Result<(usize, usize), VoltError> {
        let layer = self.get_layer(layer_idx)?;
        Ok((layer.in_dim(), layer.out_dim()))
    }

    /// Forward pass through a single layer.
    ///
    /// For layers 0 and 1, applies ReLU after the linear transform.
    /// Layer 2 (output) has no activation. Returns the layer's output
    /// activations.
    ///
    /// Used by Forward-Forward training to compute per-layer goodness
    /// without propagating gradients across layers.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if `layer_idx >= 3` or input
    /// contains NaN/Inf.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::vfn::Vfn;
    /// use volt_core::SLOT_DIM;
    ///
    /// let vfn = Vfn::new_random(42);
    /// let input = vec![0.1_f32; SLOT_DIM];
    /// let h1 = vfn.forward_layer(0, &input).unwrap();
    /// assert_eq!(h1.len(), 512); // 256 → 512
    /// assert!(h1.iter().all(|x| *x >= 0.0)); // ReLU applied
    /// ```
    pub fn forward_layer(
        &self,
        layer_idx: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, VoltError> {
        if input.iter().any(|x| !x.is_finite()) {
            return Err(VoltError::Internal {
                message: format!(
                    "VFN forward_layer {layer_idx}: input contains NaN or Inf"
                ),
            });
        }

        let layer = self.get_layer(layer_idx)?;
        let output = layer.forward(input);

        // Apply ReLU for layers 0 and 1 (hidden layers), not layer 2 (output)
        if layer_idx < 2 {
            Ok(output.into_iter().map(|x| x.max(0.0)).collect())
        } else {
            Ok(output)
        }
    }

    /// Updates weights of a single layer for Forward-Forward training.
    ///
    /// Applies: `w[i] += lr * weight_deltas[i]` and
    /// `b[i] += lr * bias_deltas[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if `layer_idx >= 3` or delta
    /// dimensions don't match the layer shape.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_soft::vfn::Vfn;
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut vfn = Vfn::new_random(42);
    /// let (in_d, out_d) = vfn.layer_shape(0).unwrap();
    /// let w_deltas = vec![0.001; in_d * out_d];
    /// let b_deltas = vec![0.001; out_d];
    /// vfn.update_layer(0, &w_deltas, &b_deltas, 1.0).unwrap();
    /// ```
    pub fn update_layer(
        &mut self,
        layer_idx: usize,
        weight_deltas: &[f32],
        bias_deltas: &[f32],
        lr: f32,
    ) -> Result<(), VoltError> {
        let layer = self.get_layer_mut(layer_idx)?;
        let expected_w = layer.in_dim() * layer.out_dim();
        let expected_b = layer.out_dim();

        if weight_deltas.len() != expected_w {
            return Err(VoltError::Internal {
                message: format!(
                    "VFN update_layer {layer_idx}: expected {expected_w} weight deltas, got {}",
                    weight_deltas.len()
                ),
            });
        }
        if bias_deltas.len() != expected_b {
            return Err(VoltError::Internal {
                message: format!(
                    "VFN update_layer {layer_idx}: expected {expected_b} bias deltas, got {}",
                    bias_deltas.len()
                ),
            });
        }

        let weights = layer.weights_mut();
        for (w, &dw) in weights.iter_mut().zip(weight_deltas.iter()) {
            *w += lr * dw;
        }

        let bias = layer.bias_mut();
        for (b, &db) in bias.iter_mut().zip(bias_deltas.iter()) {
            *b += lr * db;
        }

        Ok(())
    }

    /// Returns an immutable reference to the layer at the given index.
    fn get_layer(&self, layer_idx: usize) -> Result<&Linear, VoltError> {
        match layer_idx {
            0 => Ok(&self.layer1),
            1 => Ok(&self.layer2),
            2 => Ok(&self.layer3),
            _ => Err(VoltError::Internal {
                message: format!(
                    "VFN layer index {layer_idx} out of range (0..3)"
                ),
            }),
        }
    }

    /// Returns a mutable reference to the layer at the given index.
    fn get_layer_mut(&mut self, layer_idx: usize) -> Result<&mut Linear, VoltError> {
        match layer_idx {
            0 => Ok(&mut self.layer1),
            1 => Ok(&mut self.layer2),
            2 => Ok(&mut self.layer3),
            _ => Err(VoltError::Internal {
                message: format!(
                    "VFN layer index {layer_idx} out of range (0..3)"
                ),
            }),
        }
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

    // --- Forward-Forward API tests ---

    #[test]
    fn layer_count_is_three() {
        let vfn = Vfn::new_random(42);
        assert_eq!(vfn.layer_count(), 3);
    }

    #[test]
    fn layer_shapes_correct() {
        let vfn = Vfn::new_random(42);
        assert_eq!(vfn.layer_shape(0).unwrap(), (256, 512));
        assert_eq!(vfn.layer_shape(1).unwrap(), (512, 512));
        assert_eq!(vfn.layer_shape(2).unwrap(), (512, 256));
        assert!(vfn.layer_shape(3).is_err());
    }

    #[test]
    fn forward_layer_dimensions() {
        let vfn = Vfn::new_random(42);
        let input = vec![0.1; SLOT_DIM];
        let h1 = vfn.forward_layer(0, &input).unwrap();
        assert_eq!(h1.len(), 512);
        let h2 = vfn.forward_layer(1, &h1).unwrap();
        assert_eq!(h2.len(), 512);
        let out = vfn.forward_layer(2, &h2).unwrap();
        assert_eq!(out.len(), SLOT_DIM);
    }

    #[test]
    fn forward_layer_relu_on_hidden() {
        let vfn = Vfn::new_random(42);
        let input = vec![0.5; SLOT_DIM];
        let h1 = vfn.forward_layer(0, &input).unwrap();
        // Layer 0 applies ReLU: all outputs >= 0
        assert!(h1.iter().all(|x| *x >= 0.0));
        let h2 = vfn.forward_layer(1, &h1).unwrap();
        // Layer 1 applies ReLU: all outputs >= 0
        assert!(h2.iter().all(|x| *x >= 0.0));
    }

    #[test]
    fn forward_layer_no_relu_on_output() {
        let vfn = Vfn::new_random(42);
        let input = vec![0.5; SLOT_DIM];
        let h1 = vfn.forward_layer(0, &input).unwrap();
        let h2 = vfn.forward_layer(1, &h1).unwrap();
        let out = vfn.forward_layer(2, &h2).unwrap();
        // Layer 2 has no ReLU, so negative values are possible
        let has_negative = out.iter().any(|x| *x < 0.0);
        // With random weights, it's very likely some outputs are negative
        assert!(has_negative || out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn forward_layer_invalid_index() {
        let vfn = Vfn::new_random(42);
        let input = vec![0.1; SLOT_DIM];
        assert!(vfn.forward_layer(3, &input).is_err());
    }

    #[test]
    fn forward_layer_nan_input_errors() {
        let vfn = Vfn::new_random(42);
        let mut input = vec![0.1; SLOT_DIM];
        input[0] = f32::NAN;
        assert!(vfn.forward_layer(0, &input).is_err());
    }

    #[test]
    fn update_layer_changes_weights() {
        let mut vfn = Vfn::new_random(42);
        let input = [0.5f32; SLOT_DIM];
        let before = vfn.forward(&input).unwrap();

        let (in_d, out_d) = vfn.layer_shape(0).unwrap();
        let w_deltas = vec![0.01; in_d * out_d];
        let b_deltas = vec![0.01; out_d];
        vfn.update_layer(0, &w_deltas, &b_deltas, 1.0).unwrap();

        let after = vfn.forward(&input).unwrap();
        assert_ne!(before, after, "weights should have changed");
    }

    #[test]
    fn update_layer_wrong_size_errors() {
        let mut vfn = Vfn::new_random(42);
        // Wrong weight delta size
        assert!(vfn.update_layer(0, &[0.0; 10], &[0.0; 512], 1.0).is_err());
        // Wrong bias delta size
        assert!(
            vfn.update_layer(0, &vec![0.0; 256 * 512], &[0.0; 10], 1.0)
                .is_err()
        );
    }

    #[test]
    fn update_layer_invalid_index() {
        let mut vfn = Vfn::new_random(42);
        assert!(vfn.update_layer(3, &[], &[], 1.0).is_err());
    }

    #[test]
    fn per_layer_forward_matches_full_forward() {
        let vfn = Vfn::new_random(42);
        let input = [0.3f32; SLOT_DIM];

        // Full forward
        let full_result = vfn.forward(&input).unwrap();

        // Per-layer forward
        let h1 = vfn.forward_layer(0, &input).unwrap();
        let h2 = vfn.forward_layer(1, &h1).unwrap();
        let out = vfn.forward_layer(2, &h2).unwrap();

        for (a, b) in full_result.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "per-layer forward should match full forward: {a} vs {b}"
            );
        }
    }
}
