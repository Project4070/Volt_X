//! Internal neural network primitives for the Soft Core.
//!
//! Provides a deterministic PRNG and linear layer used by the VFN
//! and attention modules. These are internal building blocks, not
//! part of the public API.

/// Deterministic PRNG based on splitmix64.
///
/// Used for reproducible weight initialization. Not cryptographically secure.
#[derive(Clone)]
pub(crate) struct Rng(u64);

impl Rng {
    /// Creates a new PRNG with the given seed.
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Returns the next pseudo-random u64.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Returns a uniform f32 in [0, 1).
    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }

    /// Returns a uniform f32 in [lo, hi).
    pub(crate) fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

/// A single linear (fully connected) layer: y = Wx + b.
///
/// Weights are stored row-major: `weights[i * in_dim + j]` is the weight
/// from input j to output i.
#[derive(Clone)]
pub(crate) struct Linear {
    weights: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Linear {
    /// Creates a new linear layer with Xavier/Glorot uniform initialization.
    ///
    /// Weights ~ Uniform(-limit, limit) where limit = sqrt(6 / (in + out)).
    /// Biases initialized to zero.
    pub(crate) fn new_xavier(rng: &mut Rng, in_dim: usize, out_dim: usize) -> Self {
        let limit = (6.0 / (in_dim + out_dim) as f32).sqrt();
        let weights: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| rng.next_f32_range(-limit, limit))
            .collect();
        let bias = vec![0.0; out_dim];
        Self {
            weights,
            bias,
            in_dim,
            out_dim,
        }
    }

    /// Forward pass: y = Wx + b.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `input.len() == self.in_dim`.
    pub(crate) fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.in_dim);
        let mut output = self.bias.clone();
        for (i, out_val) in output.iter_mut().enumerate() {
            let row_start = i * self.in_dim;
            let mut sum = *out_val;
            for (j, &inp_val) in input.iter().enumerate() {
                sum += self.weights[row_start + j] * inp_val;
            }
            *out_val = sum;
        }
        output
    }

    /// Returns the input dimension.
    pub(crate) fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Returns the output dimension.
    pub(crate) fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Returns a reference to the weight matrix (row-major, `out_dim × in_dim`).
    ///
    /// Used by GPU modules to transfer weights to candle tensors.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Returns a reference to the bias vector (`out_dim` elements).
    ///
    /// Used by GPU modules to transfer weights to candle tensors.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Returns a mutable reference to the weight matrix.
    ///
    /// Used by Forward-Forward training to apply layer-local weight updates.
    pub(crate) fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    /// Returns a mutable reference to the bias vector.
    ///
    /// Used by Forward-Forward training to apply layer-local bias updates.
    pub(crate) fn bias_mut(&mut self) -> &mut [f32] {
        &mut self.bias
    }

    /// Creates a Linear layer from existing weights and biases.
    ///
    /// Used by checkpoint loading (Phase 0.1). Validates that weight
    /// dimensions match in_dim × out_dim and bias matches out_dim.
    pub(crate) fn from_weights_and_bias(
        weights: Vec<f32>,
        bias: Vec<f32>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self, volt_core::VoltError> {
        if weights.len() != in_dim * out_dim {
            return Err(volt_core::VoltError::LearnError {
                message: format!(
                    "Linear::from_weights_and_bias: expected {} weights, got {}",
                    in_dim * out_dim,
                    weights.len()
                ),
            });
        }
        if bias.len() != out_dim {
            return Err(volt_core::VoltError::LearnError {
                message: format!(
                    "Linear::from_weights_and_bias: expected {} biases, got {}",
                    out_dim,
                    bias.len()
                ),
            });
        }
        Ok(Self {
            weights,
            bias,
            in_dim,
            out_dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_deterministic() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn rng_different_seeds_different_output() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(43);
        // At least one of 10 values should differ
        let differs = (0..10).any(|_| r1.next_u64() != r2.next_u64());
        assert!(differs);
    }

    #[test]
    fn rng_f32_in_range() {
        let mut rng = Rng::new(42);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "next_f32 out of range: {}", v);
        }
    }

    #[test]
    fn rng_f32_range_in_bounds() {
        let mut rng = Rng::new(42);
        for _ in 0..1000 {
            let v = rng.next_f32_range(-0.5, 0.5);
            assert!(
                (-0.5..0.5).contains(&v),
                "next_f32_range out of range: {}",
                v
            );
        }
    }

    #[test]
    fn linear_output_correct_size() {
        let mut rng = Rng::new(42);
        let layer = Linear::new_xavier(&mut rng, 8, 4);
        let input = vec![1.0; 8];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn linear_zero_input_gives_bias() {
        let mut rng = Rng::new(42);
        let layer = Linear::new_xavier(&mut rng, 8, 4);
        let input = vec![0.0; 8];
        let output = layer.forward(&input);
        // With zero bias, output should be zero
        for v in &output {
            assert!(v.abs() < 1e-10, "zero input should give zero output with zero bias");
        }
    }

    #[test]
    fn linear_xavier_init_bounded() {
        let mut rng = Rng::new(42);
        let in_dim = 256;
        let out_dim = 512;
        let layer = Linear::new_xavier(&mut rng, in_dim, out_dim);
        let limit = (6.0 / (in_dim + out_dim) as f32).sqrt();
        for &w in &layer.weights {
            assert!(
                w.abs() <= limit + 1e-6,
                "Xavier weight {} exceeds limit {}",
                w,
                limit
            );
        }
    }

    #[test]
    fn linear_deterministic() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        let l1 = Linear::new_xavier(&mut rng1, 8, 4);
        let l2 = Linear::new_xavier(&mut rng2, 8, 4);
        let input = vec![0.5; 8];
        assert_eq!(l1.forward(&input), l2.forward(&input));
    }
}
