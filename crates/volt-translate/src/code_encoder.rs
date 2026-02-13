//! CNN-based code encoder for Phase 1 translator training.
//!
//! Architecture: BPE tokens → Embedding → 3×Conv1D → Role Head + Embed Head
//! → aggregate by role → 16×256 TensorFrame.
//!
//! Total parameters: ~5.1M (mostly in the embedding layer).
//!
//! # Example
//!
//! ```ignore
//! use volt_translate::code_encoder::{CodeEncoder, CodeEncoderConfig};
//! use candle_core::Device;
//!
//! let config = CodeEncoderConfig::default();
//! let encoder = CodeEncoder::new_random(&config, &Device::Cpu).unwrap();
//! ```

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{
    conv1d, embedding, linear, Conv1d, Conv1dConfig, Embedding, Linear, VarBuilder, VarMap,
};
use volt_core::slot::SlotSource;
use volt_core::{SlotRole, TensorFrame, VoltError, MAX_SLOTS, SLOT_DIM};

/// Configuration for the [`CodeEncoder`].
///
/// # Example
///
/// ```
/// use volt_translate::code_encoder::CodeEncoderConfig;
///
/// let config = CodeEncoderConfig::default();
/// assert_eq!(config.vocab_size, 32768);
/// assert_eq!(config.embed_dim, 128);
/// assert_eq!(config.hidden_dim, 256);
/// assert_eq!(config.max_seq_len, 512);
/// ```
#[derive(Debug, Clone)]
pub struct CodeEncoderConfig {
    /// BPE vocabulary size (default: 32768).
    pub vocab_size: usize,
    /// Token embedding dimension (default: 128).
    pub embed_dim: usize,
    /// Conv hidden dimension / output dimension (default: 256 = SLOT_DIM).
    pub hidden_dim: usize,
    /// Maximum input sequence length in BPE tokens (default: 512).
    pub max_seq_len: usize,
    /// Minimum role probability to create a slot (default: 0.05).
    pub role_threshold: f32,
}

impl Default for CodeEncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32768,
            embed_dim: 128,
            hidden_dim: SLOT_DIM,
            max_seq_len: 512,
            role_threshold: 0.05,
        }
    }
}

/// Slot role mapping for code (matches Phase 0.4 code_attention bias).
const CODE_SLOT_ROLES: [SlotRole; MAX_SLOTS] = [
    SlotRole::Agent,      // S0: Function name / class
    SlotRole::Predicate,  // S1: Operation / method call
    SlotRole::Patient,    // S2: Arguments / parameters
    SlotRole::Location,   // S3: Return value / result
    SlotRole::Time,       // S4: Execution order
    SlotRole::Manner,     // S5: Algorithm pattern
    SlotRole::Instrument, // S6: Control flow 1 (if/else)
    SlotRole::Cause,      // S7: Control flow 2 (try/catch)
    SlotRole::Result,     // S8: Control flow 3 (match/switch)
    SlotRole::Free(0),    // S9-S15: Overflow
    SlotRole::Free(1),
    SlotRole::Free(2),
    SlotRole::Free(3),
    SlotRole::Free(4),
    SlotRole::Free(5),
    SlotRole::Free(6),
];

/// CNN-based code encoder.
///
/// Encodes BPE token sequences into TensorFrame slot embeddings using
/// a 3-layer CNN with role classification and embedding projection.
///
/// # Architecture
///
/// ```text
/// BPE tokens [batch, seq_len]
///   → Embedding(vocab_size, embed_dim)    [batch, seq_len, embed_dim]
///   → Conv1D(embed_dim, hidden, k=3, p=1) + GELU
///   → Conv1D(hidden, hidden, k=5, p=2)    + GELU
///   → Conv1D(hidden, hidden, k=7, p=3)    + GELU
///   → [batch, seq_len, hidden_dim]
///   → Role Head: Linear(hidden, 16) → softmax
///   → Embed Head: Linear(hidden, SLOT_DIM)
/// ```
pub struct CodeEncoder {
    embed: Embedding,
    conv1: Conv1d,
    conv2: Conv1d,
    conv3: Conv1d,
    role_head: Linear,
    embed_head: Linear,
    var_map: VarMap,
    config: CodeEncoderConfig,
    device: Device,
}

impl std::fmt::Debug for CodeEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CodeEncoder(vocab={}, embed={}, hidden={}, device={:?})",
            self.config.vocab_size, self.config.embed_dim, self.config.hidden_dim, self.device
        )
    }
}

/// Output of the encoder's forward pass (differentiable).
///
/// Contains the intermediate tensors needed for training losses.
pub struct EncoderOutput {
    /// Per-token features from the CNN backbone. Shape: `[batch, seq_len, hidden_dim]`.
    pub features: Tensor,
    /// Per-token role probabilities (softmax). Shape: `[batch, seq_len, 16]`.
    pub role_probs: Tensor,
    /// Per-token slot embeddings. Shape: `[batch, seq_len, SLOT_DIM]`.
    pub token_embeds: Tensor,
    /// Mean-pooled summary vector (L2-normalized). Shape: `[batch, SLOT_DIM]`.
    pub summary: Tensor,
}

impl CodeEncoder {
    /// Creates a new encoder with random initialization.
    ///
    /// Uses candle's default initialization for all parameters.
    /// The returned encoder holds a [`VarMap`] for training.
    ///
    /// # Errors
    ///
    /// Returns error if tensor allocation fails on the device.
    pub fn new_random(config: &CodeEncoderConfig, device: &Device) -> Result<Self, VoltError> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);

        let embed = embedding(config.vocab_size, config.embed_dim, vb.pp("embed"))
            .map_err(candle_err)?;

        let conv1 = conv1d(
            config.embed_dim,
            config.hidden_dim,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )
        .map_err(candle_err)?;

        let conv2 = conv1d(
            config.hidden_dim,
            config.hidden_dim,
            5,
            Conv1dConfig {
                padding: 2,
                ..Default::default()
            },
            vb.pp("conv2"),
        )
        .map_err(candle_err)?;

        let conv3 = conv1d(
            config.hidden_dim,
            config.hidden_dim,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv3"),
        )
        .map_err(candle_err)?;

        let role_head =
            linear(config.hidden_dim, MAX_SLOTS, vb.pp("role_head")).map_err(candle_err)?;
        let embed_head =
            linear(config.hidden_dim, SLOT_DIM, vb.pp("embed_head")).map_err(candle_err)?;

        Ok(Self {
            embed,
            conv1,
            conv2,
            conv3,
            role_head,
            embed_head,
            var_map,
            config: config.clone(),
            device: device.clone(),
        })
    }

    /// Loads encoder weights from a safetensors file.
    ///
    /// Creates the model structure first, then overwrites weights
    /// with saved values.
    ///
    /// # Errors
    ///
    /// Returns error if the file doesn't exist or weights are incompatible.
    pub fn load(
        config: &CodeEncoderConfig,
        path: &std::path::Path,
        device: &Device,
    ) -> Result<Self, VoltError> {
        let mut encoder = Self::new_random(config, device)?;
        encoder.var_map.load(path).map_err(|e| VoltError::StorageError {
            message: format!("failed to load encoder weights from {}: {e}", path.display()),
        })?;
        Ok(encoder)
    }

    /// Saves encoder weights to a safetensors file.
    ///
    /// # Errors
    ///
    /// Returns error if the file cannot be written.
    pub fn save(&self, path: &std::path::Path) -> Result<(), VoltError> {
        self.var_map.save(path).map_err(|e| VoltError::StorageError {
            message: format!("failed to save encoder weights to {}: {e}", path.display()),
        })
    }

    /// Returns a reference to the VarMap for optimizer construction.
    pub fn var_map(&self) -> &VarMap {
        &self.var_map
    }

    /// Returns the encoder configuration.
    pub fn config(&self) -> &CodeEncoderConfig {
        &self.config
    }

    /// Returns the device this encoder is on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Forward pass returning all intermediate tensors.
    ///
    /// # Arguments
    ///
    /// * `token_ids` — BPE token IDs, shape `[batch, seq_len]` (u32).
    ///
    /// # Returns
    ///
    /// [`EncoderOutput`] with features, role_probs, token_embeds, and summary.
    pub fn forward(&self, token_ids: &Tensor) -> Result<EncoderOutput, VoltError> {
        // Embedding: [batch, seq_len] → [batch, seq_len, embed_dim]
        let emb = self.embed.forward(token_ids).map_err(candle_err)?;

        // Transpose for Conv1d: [batch, embed_dim, seq_len]
        let x = emb.transpose(1, 2).map_err(candle_err)?;

        // 3 conv layers with GELU activation
        let x = self.conv1.forward(&x).map_err(candle_err)?;
        let x = x.gelu_erf().map_err(candle_err)?;
        let x = self.conv2.forward(&x).map_err(candle_err)?;
        let x = x.gelu_erf().map_err(candle_err)?;
        let x = self.conv3.forward(&x).map_err(candle_err)?;
        let x = x.gelu_erf().map_err(candle_err)?;

        // Transpose back: [batch, seq_len, hidden_dim]
        let features = x.transpose(1, 2).map_err(candle_err)?;

        // Role head: [batch, seq_len, 16]
        let role_logits = self.role_head.forward(&features).map_err(candle_err)?;
        let role_probs =
            candle_nn::ops::softmax(&role_logits, D::Minus1).map_err(candle_err)?;

        // Embed head: [batch, seq_len, SLOT_DIM]
        let token_embeds = self.embed_head.forward(&features).map_err(candle_err)?;

        // Summary: mean pool across seq_len → [batch, SLOT_DIM]
        let summary = token_embeds.mean(1).map_err(candle_err)?;

        // L2 normalize summary
        let summary = l2_normalize(&summary)?;

        Ok(EncoderOutput {
            features,
            role_probs,
            token_embeds,
            summary,
        })
    }

    /// Encode text to a TensorFrame using role-based aggregation.
    ///
    /// This is the inference-time encoding path. For training, use
    /// [`forward`](Self::forward) instead.
    ///
    /// # Arguments
    ///
    /// * `token_ids` — BPE token IDs for a single sample, shape `[1, seq_len]` (u32).
    ///
    /// # Returns
    ///
    /// A TensorFrame with slots filled based on role classification.
    pub fn encode_to_frame(&self, token_ids: &Tensor) -> Result<TensorFrame, VoltError> {
        let output = self.forward(token_ids)?;

        // Extract from batch dim 0
        let role_probs = output.role_probs.squeeze(0).map_err(candle_err)?; // [seq_len, 16]
        let token_embeds = output.token_embeds.squeeze(0).map_err(candle_err)?; // [seq_len, SLOT_DIM]

        let role_probs_vec: Vec<Vec<f32>> = tensor_to_2d(&role_probs)?;
        let embeds_vec: Vec<Vec<f32>> = tensor_to_2d(&token_embeds)?;

        let seq_len = role_probs_vec.len();
        let mut frame = TensorFrame::new();

        // Aggregate embeddings per role (weighted average)
        for (role_idx, &role) in CODE_SLOT_ROLES.iter().enumerate() {
            let mut weighted_sum = [0.0f32; SLOT_DIM];
            let mut total_weight = 0.0f32;

            for t in 0..seq_len {
                let w = role_probs_vec[t][role_idx];
                total_weight += w;
                for (d, ws) in weighted_sum.iter_mut().enumerate() {
                    *ws += w * embeds_vec[t][d];
                }
            }

            if total_weight > self.config.role_threshold {
                // Compute weighted average
                for ws in &mut weighted_sum {
                    *ws /= total_weight;
                }

                // L2 normalize
                let norm: f32 = weighted_sum.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    for x in &mut weighted_sum {
                        *x /= norm;
                    }
                }

                frame
                    .write_at(role_idx, 0, role, weighted_sum)
                    .map_err(|e| VoltError::FrameError {
                        message: format!("failed to write slot {role_idx}: {e}"),
                    })?;

                frame.meta[role_idx].certainty = total_weight.min(1.0);
                frame.meta[role_idx].source = SlotSource::Translator;
                frame.meta[role_idx].needs_verify = true;
            }
        }

        Ok(frame)
    }

    /// Count total trainable parameters.
    pub fn param_count(&self) -> usize {
        self.var_map
            .all_vars()
            .iter()
            .map(|v| v.elem_count())
            .sum()
    }
}

/// L2-normalize along the last dimension.
fn l2_normalize(x: &Tensor) -> Result<Tensor, VoltError> {
    let norm = x
        .sqr()
        .map_err(candle_err)?
        .sum(D::Minus1)
        .map_err(candle_err)?
        .sqrt()
        .map_err(candle_err)?
        .clamp(1e-8, f64::INFINITY)
        .map_err(candle_err)?
        .unsqueeze(D::Minus1)
        .map_err(candle_err)?;
    x.broadcast_div(&norm).map_err(candle_err)
}

/// Convert a 2D tensor to Vec<Vec<f32>>.
fn tensor_to_2d(t: &Tensor) -> Result<Vec<Vec<f32>>, VoltError> {
    let t = t.to_device(&Device::Cpu).map_err(candle_err)?;
    let rows = t.dims()[0];
    let cols = t.dims()[1];
    let data = t.flatten_all().map_err(candle_err)?;
    let data: Vec<f32> = data.to_vec1().map_err(candle_err)?;
    let mut result = Vec::with_capacity(rows);
    for r in 0..rows {
        result.push(data[r * cols..(r + 1) * cols].to_vec());
    }
    Ok(result)
}

/// Convert candle errors to VoltError.
fn candle_err(e: candle_core::Error) -> VoltError {
    VoltError::Internal {
        message: format!("candle error: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_config_defaults() {
        let config = CodeEncoderConfig::default();
        assert_eq!(config.vocab_size, 32768);
        assert_eq!(config.embed_dim, 128);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.max_seq_len, 512);
    }

    #[test]
    fn encoder_creates_on_cpu() {
        let config = CodeEncoderConfig::default();
        let encoder = CodeEncoder::new_random(&config, &Device::Cpu).unwrap();
        assert!(encoder.param_count() > 4_000_000); // > 4M params (embedding alone)
        assert!(encoder.param_count() < 8_000_000); // < 8M total
    }

    #[test]
    fn encoder_forward_shapes() {
        let config = CodeEncoderConfig::default();
        let encoder = CodeEncoder::new_random(&config, &Device::Cpu).unwrap();

        // Batch of 2, seq_len 10
        let token_ids = Tensor::zeros((2, 10), DType::U32, &Device::Cpu).unwrap();
        let output = encoder.forward(&token_ids).unwrap();

        assert_eq!(output.features.dims(), &[2, 10, 256]);
        assert_eq!(output.role_probs.dims(), &[2, 10, 16]);
        assert_eq!(output.token_embeds.dims(), &[2, 10, 256]);
        assert_eq!(output.summary.dims(), &[2, 256]);
    }

    #[test]
    fn encoder_summary_is_normalized() {
        let config = CodeEncoderConfig::default();
        let encoder = CodeEncoder::new_random(&config, &Device::Cpu).unwrap();

        let token_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &Device::Cpu).unwrap();
        let output = encoder.forward(&token_ids).unwrap();

        let summary: Vec<f32> = output.summary.squeeze(0).unwrap().to_vec1().unwrap();
        let norm: f32 = summary.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "summary should be L2-normalized, got norm={norm}"
        );
    }

    #[test]
    fn encoder_role_probs_sum_to_one() {
        let config = CodeEncoderConfig::default();
        let encoder = CodeEncoder::new_random(&config, &Device::Cpu).unwrap();

        let token_ids = Tensor::new(&[[1u32, 2, 3]], &Device::Cpu).unwrap();
        let output = encoder.forward(&token_ids).unwrap();

        // role_probs should sum to ~1.0 across roles for each token
        let sums = output
            .role_probs
            .sum(D::Minus1)
            .unwrap()
            .squeeze(0)
            .unwrap();
        let sums: Vec<f32> = sums.to_vec1().unwrap();
        for (t, sum) in sums.iter().enumerate() {
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "token {t} role_probs sum={sum}, expected ~1.0"
            );
        }
    }

    #[test]
    fn encode_to_frame_produces_valid_frame() {
        let config = CodeEncoderConfig::default();
        let encoder = CodeEncoder::new_random(&config, &Device::Cpu).unwrap();

        let token_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &Device::Cpu).unwrap();
        let frame = encoder.encode_to_frame(&token_ids).unwrap();

        // Should have some active slots
        assert!(
            frame.active_slot_count() > 0,
            "frame should have at least one active slot"
        );
    }
}
