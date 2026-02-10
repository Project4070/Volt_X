//! Frame Projection Head: maps LLM hidden states to TensorFrame slots.
//!
//! Architecture: configurable MLP (default 1024→4096→4096→4096, GELU) with
//! two heads:
//! - **Role Classifier** (mlp_dim→16, softmax): P(role | token)
//! - **Embedding Projector** (mlp_dim→256): per-token slot embeddings
//!
//! Slot vectors are computed as weighted averages per role, then
//! L2-normalized. The projection head is trained in Python
//! (`tools/train_translator.py`) and exported as safetensors for
//! Rust inference.

use std::path::Path;

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Linear, VarBuilder, VarMap};
use volt_core::{VoltError, SLOT_DIM};

use super::roles::NUM_ROLE_CLASSES;

/// Default MLP hidden dimension (Qwen3-0.6B hidden_dim=1024 → 4× expansion).
pub const DEFAULT_MLP_DIM: usize = 4096;

/// Configuration for the [`FrameProjectionHead`].
///
/// # Example
///
/// ```
/// use volt_translate::llm::projection::ProjectionConfig;
///
/// let config = ProjectionConfig::default();
/// assert_eq!(config.hidden_dim, 1024);
/// assert_eq!(config.mlp_dim, 4096);
/// ```
#[derive(Debug, Clone)]
pub struct ProjectionConfig {
    /// Input hidden dimension from the LLM backbone (1024 for Qwen3-0.6B).
    pub hidden_dim: usize,
    /// MLP hidden dimension (default [`DEFAULT_MLP_DIM`]).
    pub mlp_dim: usize,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 1024,
            mlp_dim: DEFAULT_MLP_DIM,
        }
    }
}

/// Frame Projection Head: maps LLM hidden states to role probabilities
/// and slot embeddings.
///
/// # Architecture
///
/// ```text
/// hidden_states [seq_len, hidden_dim]
///   → MLP Layer 0: Linear(hidden_dim, mlp_dim) + GELU
///   → MLP Layer 1: Linear(mlp_dim, mlp_dim) + GELU
///   → MLP Layer 2: Linear(mlp_dim, mlp_dim) + GELU
///   ├→ Role Head:  Linear(mlp_dim, 16)  + softmax → role_probs
///   └→ Embed Head: Linear(mlp_dim, 256)           → token_embeds
/// ```
///
/// Total parameters (default config): ~39M.
///
/// # Example
///
/// ```
/// use volt_translate::llm::projection::{FrameProjectionHead, ProjectionConfig};
/// use candle_core::Device;
///
/// let config = ProjectionConfig { hidden_dim: 64, mlp_dim: 128 };
/// let head = FrameProjectionHead::new_random(&config, &Device::Cpu).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct FrameProjectionHead {
    mlp_0: Linear,
    mlp_1: Linear,
    mlp_2: Linear,
    role_head: Linear,
    embed_head: Linear,
}

impl FrameProjectionHead {
    /// Loads a trained projection head from safetensors weights.
    ///
    /// Expects weight keys under the `proj.` prefix:
    /// - `proj.mlp.{0,1,2}.{weight,bias}`
    /// - `proj.role_head.{weight,bias}`
    /// - `proj.embed_head.{weight,bias}`
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::TranslateError`] if the file cannot be read
    /// or weight shapes do not match the config.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_translate::llm::projection::{FrameProjectionHead, ProjectionConfig};
    /// use candle_core::Device;
    /// use std::path::Path;
    ///
    /// let head = FrameProjectionHead::load(
    ///     Path::new("projection.safetensors"),
    ///     &ProjectionConfig::default(),
    ///     &Device::Cpu,
    /// ).unwrap();
    /// ```
    pub fn load(
        weights_path: &Path,
        config: &ProjectionConfig,
        device: &Device,
    ) -> Result<Self, VoltError> {
        // SAFETY: The safetensors file is only read during initialization
        // and not modified while the memory map is alive.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)
                .map_err(|e| VoltError::TranslateError {
                    message: format!(
                        "failed to load projection weights from {}: {e}",
                        weights_path.display()
                    ),
                })?
        };
        Self::from_var_builder(vb.pp("proj"), config)
    }

    /// Creates a projection head with random (Kaiming-uniform) weights.
    ///
    /// Used for testing without real trained weights.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_translate::llm::projection::{FrameProjectionHead, ProjectionConfig};
    /// use candle_core::Device;
    ///
    /// let config = ProjectionConfig { hidden_dim: 64, mlp_dim: 128 };
    /// let head = FrameProjectionHead::new_random(&config, &Device::Cpu).unwrap();
    /// ```
    pub fn new_random(config: &ProjectionConfig, device: &Device) -> Result<Self, VoltError> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        Self::from_var_builder(vb, config)
    }

    /// Builds the projection head layers from a [`VarBuilder`].
    fn from_var_builder(vb: VarBuilder<'_>, config: &ProjectionConfig) -> Result<Self, VoltError> {
        let mlp = vb.pp("mlp");
        let mlp_0 = candle_nn::linear(config.hidden_dim, config.mlp_dim, mlp.pp("0"))
            .map_err(candle_to_volt)?;
        let mlp_1 = candle_nn::linear(config.mlp_dim, config.mlp_dim, mlp.pp("1"))
            .map_err(candle_to_volt)?;
        let mlp_2 = candle_nn::linear(config.mlp_dim, config.mlp_dim, mlp.pp("2"))
            .map_err(candle_to_volt)?;
        let role_head =
            candle_nn::linear(config.mlp_dim, NUM_ROLE_CLASSES, vb.pp("role_head"))
                .map_err(candle_to_volt)?;
        let embed_head =
            candle_nn::linear(config.mlp_dim, SLOT_DIM, vb.pp("embed_head"))
                .map_err(candle_to_volt)?;

        Ok(Self {
            mlp_0,
            mlp_1,
            mlp_2,
            role_head,
            embed_head,
        })
    }

    /// Runs the forward pass through the MLP and both heads.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` — LLM hidden states, shape `[seq_len, hidden_dim]`
    ///
    /// # Returns
    ///
    /// `(role_probs, token_embeds)`:
    /// - `role_probs`: `[seq_len, 16]` — P(role | token) after softmax
    /// - `token_embeds`: `[seq_len, 256]` — per-token slot embeddings
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::TranslateError`] on shape mismatch or
    /// computation error.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_translate::llm::projection::{FrameProjectionHead, ProjectionConfig};
    /// use candle_core::{Device, Tensor};
    ///
    /// let config = ProjectionConfig { hidden_dim: 64, mlp_dim: 128 };
    /// let head = FrameProjectionHead::new_random(&config, &Device::Cpu).unwrap();
    /// let input = Tensor::randn(0f32, 1.0, (5, 64), &Device::Cpu).unwrap();
    /// let (role_probs, token_embeds) = head.forward(&input).unwrap();
    /// assert_eq!(role_probs.dims(), &[5, 16]);
    /// assert_eq!(token_embeds.dims(), &[5, 256]);
    /// ```
    pub fn forward(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor), VoltError> {
        // MLP: 3 layers with GELU activation
        let x = self.mlp_0.forward(hidden_states).map_err(candle_to_volt)?;
        let x = x.gelu().map_err(candle_to_volt)?;
        let x = self.mlp_1.forward(&x).map_err(candle_to_volt)?;
        let x = x.gelu().map_err(candle_to_volt)?;
        let x = self.mlp_2.forward(&x).map_err(candle_to_volt)?;
        let x = x.gelu().map_err(candle_to_volt)?;

        // Role classifier: softmax over 16 roles
        let role_logits = self.role_head.forward(&x).map_err(candle_to_volt)?;
        let role_probs =
            candle_nn::ops::softmax(&role_logits, D::Minus1).map_err(candle_to_volt)?;

        // Embedding projector: raw 256-dim embeddings
        let token_embeds = self.embed_head.forward(&x).map_err(candle_to_volt)?;

        Ok((role_probs, token_embeds))
    }
}

/// Aggregates per-token role probabilities and embeddings into slot vectors.
///
/// For each role whose total probability mass exceeds `threshold`,
/// computes the weighted average of token embeddings, L2-normalizes
/// the result, and returns `(slot_index, slot_vector, confidence)`.
///
/// # Arguments
///
/// * `role_probs` — `[seq_len, 16]` from [`FrameProjectionHead::forward`]
/// * `token_embeds` — `[seq_len, 256]` from [`FrameProjectionHead::forward`]
/// * `threshold` — minimum total role probability to activate a slot
///
/// # Returns
///
/// Vec of `(slot_index, L2-normalized slot vector, total_weight)`.
/// Empty vec if no role exceeds the threshold.
///
/// # Example
///
/// ```
/// use volt_translate::llm::projection::aggregate_to_slots;
/// use candle_core::{Device, Tensor, DType};
///
/// let role_probs = Tensor::new(
///     &[[0.9f32, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0,
///        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
///     &Device::Cpu,
/// ).unwrap();
/// let token_embeds = Tensor::ones((1, 256), DType::F32, &Device::Cpu).unwrap();
///
/// let slots = aggregate_to_slots(&role_probs, &token_embeds, 0.1).unwrap();
/// assert_eq!(slots.len(), 1);
/// assert_eq!(slots[0].0, 0); // slot 0 (Agent)
/// ```
pub fn aggregate_to_slots(
    role_probs: &Tensor,
    token_embeds: &Tensor,
    threshold: f32,
) -> Result<Vec<(usize, [f32; SLOT_DIM], f32)>, VoltError> {
    let probs = role_probs.to_vec2::<f32>().map_err(candle_to_volt)?;
    let embeds = token_embeds.to_vec2::<f32>().map_err(candle_to_volt)?;

    let seq_len = probs.len();
    let mut result = Vec::new();

    for r in 0..NUM_ROLE_CLASSES {
        let mut total_weight = 0.0_f32;
        let mut weighted_sum = [0.0_f32; SLOT_DIM];

        for t in 0..seq_len {
            let w = probs[t][r];
            total_weight += w;
            for (ws, &e) in weighted_sum.iter_mut().zip(embeds[t].iter()) {
                *ws += w * e;
            }
        }

        if total_weight > threshold {
            // Weighted average
            for v in &mut weighted_sum {
                *v /= total_weight;
            }
            // L2 normalize
            let norm: f32 = weighted_sum.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for v in &mut weighted_sum {
                    *v /= norm;
                }
            }
            result.push((r, weighted_sum, total_weight));
        }
    }

    Ok(result)
}

/// Converts a candle error into a [`VoltError::TranslateError`].
pub(crate) fn candle_to_volt(e: candle_core::Error) -> VoltError {
    VoltError::TranslateError {
        message: format!("candle error: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ProjectionConfig {
        ProjectionConfig {
            hidden_dim: 64,
            mlp_dim: 128,
        }
    }

    #[test]
    fn config_default() {
        let config = ProjectionConfig::default();
        assert_eq!(config.hidden_dim, 1024);
        assert_eq!(config.mlp_dim, DEFAULT_MLP_DIM);
    }

    #[test]
    fn forward_output_shapes() {
        let config = test_config();
        let head = FrameProjectionHead::new_random(&config, &Device::Cpu).unwrap();
        let input = Tensor::randn(0f32, 1.0, (10, 64), &Device::Cpu).unwrap();
        let (role_probs, token_embeds) = head.forward(&input).unwrap();

        assert_eq!(role_probs.dims(), &[10, NUM_ROLE_CLASSES]);
        assert_eq!(token_embeds.dims(), &[10, SLOT_DIM]);
    }

    #[test]
    fn role_probs_sum_to_one() {
        let config = test_config();
        let head = FrameProjectionHead::new_random(&config, &Device::Cpu).unwrap();
        let input = Tensor::randn(0f32, 1.0, (5, 64), &Device::Cpu).unwrap();
        let (role_probs, _) = head.forward(&input).unwrap();

        let probs = role_probs.to_vec2::<f32>().unwrap();
        for row in &probs {
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "softmax sum = {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn aggregate_basic() {
        let probs = Tensor::new(
            &[
                [
                    0.9_f32, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                ],
                [
                    0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0,
                ],
            ],
            &Device::Cpu,
        )
        .unwrap();
        let embeds = Tensor::ones((2, SLOT_DIM), DType::F32, &Device::Cpu).unwrap();

        let slots = aggregate_to_slots(&probs, &embeds, 0.5).unwrap();
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].0, 0); // Agent
        assert_eq!(slots[1].0, 1); // Predicate
    }

    #[test]
    fn aggregate_l2_normalized() {
        let probs = Tensor::new(
            &[[
                1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0,
            ]],
            &Device::Cpu,
        )
        .unwrap();
        let embeds = Tensor::randn(0f32, 1.0, (1, SLOT_DIM), &Device::Cpu).unwrap();

        let slots = aggregate_to_slots(&probs, &embeds, 0.1).unwrap();
        assert_eq!(slots.len(), 1);

        let norm: f32 = slots[0].1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "slot vector norm = {norm}, expected 1.0"
        );
    }

    #[test]
    fn aggregate_high_threshold_empty() {
        let probs = Tensor::new(&[[0.05_f32; NUM_ROLE_CLASSES]], &Device::Cpu).unwrap();
        let embeds = Tensor::ones((1, SLOT_DIM), DType::F32, &Device::Cpu).unwrap();

        let slots = aggregate_to_slots(&probs, &embeds, 1.0).unwrap();
        assert!(slots.is_empty());
    }

    #[test]
    fn aggregate_confidence_equals_total_weight() {
        let probs = Tensor::new(
            &[
                [
                    0.7_f32, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0,
                ],
                [
                    0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0,
                ],
            ],
            &Device::Cpu,
        )
        .unwrap();
        let embeds = Tensor::ones((2, SLOT_DIM), DType::F32, &Device::Cpu).unwrap();

        let slots = aggregate_to_slots(&probs, &embeds, 0.1).unwrap();
        // Role 0: 0.7 + 0.6 = 1.3
        assert!((slots[0].2 - 1.3).abs() < 1e-5);
        // Role 1: 0.3 + 0.4 = 0.7
        assert!((slots[1].2 - 0.7).abs() < 1e-5);
    }
}
