//! Autoregressive code decoder for Phase 1 translator training.
//!
//! Architecture: causal self-attention + cross-attention to encoder slots + MLP.
//! Uses teacher forcing during training, autoregressive generation at inference.
//!
//! # Example
//!
//! ```ignore
//! use volt_translate::code_decoder::{CodeDecoder, CodeDecoderConfig};
//! use candle_core::Device;
//!
//! let config = CodeDecoderConfig::default();
//! let decoder = CodeDecoder::new_random(&config, &Device::Cpu).unwrap();
//! ```

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder, VarMap};
use volt_core::{VoltError, SLOT_DIM};

/// Model dimension (matches SLOT_DIM for cross-attention compatibility).
const MODEL_DIM: usize = SLOT_DIM; // 256

/// RMS layer normalization using only basic tensor ops (CUDA-compatible).
///
/// Candle's built-in `LayerNorm` lacks a CUDA kernel, so we use RMSNorm
/// which achieves similar training stability with only elementwise ops.
///
/// Formula: `y = x / rms(x) * weight`, where `rms(x) = sqrt(mean(x^2) + eps)`.
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, vb: VarBuilder<'_>) -> Result<Self, candle_core::Error> {
        let weight = vb.get_with_hints(dim, "weight", candle_nn::Init::Const(1.0))?;
        Ok(Self { weight, eps: 1e-6 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        // rms = sqrt(mean(x^2) + eps)
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = (variance + self.eps)?.sqrt()?;
        let normalized = x.broadcast_div(&rms)?;
        normalized.broadcast_mul(&self.weight)
    }
}

/// Configuration for the [`CodeDecoder`].
///
/// # Example
///
/// ```
/// use volt_translate::code_decoder::CodeDecoderConfig;
///
/// let config = CodeDecoderConfig::default();
/// assert_eq!(config.max_seq_len, 256);
/// assert_eq!(config.vocab_size, 32768);
/// ```
#[derive(Debug, Clone)]
pub struct CodeDecoderConfig {
    /// Maximum output sequence length (default: 256).
    pub max_seq_len: usize,
    /// BPE vocabulary size — must match encoder (default: 32768).
    pub vocab_size: usize,
    /// Hidden dimension for the feed-forward MLP (default: 1024).
    pub hidden_dim: usize,
    /// Token embedding dimension (default: 128).
    pub embed_dim: usize,
}

impl Default for CodeDecoderConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 256,
            vocab_size: 32768,
            hidden_dim: 1024,
            embed_dim: 128,
        }
    }
}

/// Autoregressive code decoder with causal self-attention and cross-attention.
///
/// During training, uses teacher forcing: the model receives ground-truth
/// tokens (shifted right by 1) and predicts the next token at each position.
/// During inference, generates tokens one at a time autoregressively.
///
/// # Architecture
///
/// ```text
/// input_ids [batch, seq_len]  (shifted-right target tokens, 0=BOS)
///   → token_embed lookup → [batch, seq_len, embed_dim=128]
///   → input_proj           → [batch, seq_len, model_dim=256]
///   + positional encoding
///
///   Pre-RMSNorm Causal Self-Attention:
///     RMSNorm → Q,K,V projections → causal mask → softmax → attend
///     + residual
///
///   Pre-RMSNorm Cross-Attention to encoder slots:
///     RMSNorm → Q(decoder) × K,V(slots) → softmax → attend
///     + residual
///
///   Pre-RMSNorm Feed-Forward:
///     RMSNorm → Linear(256, 1024) + GELU → Linear(1024, 256)
///     + residual
///
///   → output_proj(model_dim → embed_dim)
///   → logits = output @ token_embed.T  → [batch, seq_len, vocab_size]
/// ```
pub struct CodeDecoder {
    // Input: token embedding (tied with output) + projection to model_dim
    token_embed: Tensor,
    input_proj: Linear,
    pos_encoding: Tensor,

    // Causal self-attention (pre-norm)
    self_norm: RmsNorm,
    self_q: Linear,
    self_k: Linear,
    self_v: Linear,

    // Cross-attention to encoder slots (pre-norm)
    cross_norm: RmsNorm,
    cross_q: Linear,
    cross_k: Linear,
    cross_v: Linear,

    // Feed-forward network (pre-norm)
    ffn_norm: RmsNorm,
    ffn1: Linear,
    ffn2: Linear,

    // Output projection
    output_proj: Linear,

    var_map: VarMap,
    config: CodeDecoderConfig,
    device: Device,
}

impl std::fmt::Debug for CodeDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CodeDecoder(vocab={}, hidden={}, max_len={}, device={:?})",
            self.config.vocab_size, self.config.hidden_dim, self.config.max_seq_len, self.device
        )
    }
}

impl CodeDecoder {
    /// Creates a new decoder with random initialization.
    ///
    /// # Errors
    ///
    /// Returns error if tensor allocation fails.
    pub fn new_random(config: &CodeDecoderConfig, device: &Device) -> Result<Self, VoltError> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);

        // Shared token embedding: [vocab, embed_dim]
        let token_embed = vb
            .get((config.vocab_size, config.embed_dim), "token_embed")
            .map_err(candle_err)?;

        // Project from embed_dim to model_dim
        let input_proj =
            linear(config.embed_dim, MODEL_DIM, vb.pp("input_proj")).map_err(candle_err)?;

        // Learnable positional encoding: [max_seq_len, MODEL_DIM]
        let pos_encoding = vb
            .get((config.max_seq_len, MODEL_DIM), "pos_encoding")
            .map_err(candle_err)?;

        // Causal self-attention
        let self_norm = RmsNorm::new(MODEL_DIM, vb.pp("self_norm")).map_err(candle_err)?;
        let self_q = linear(MODEL_DIM, MODEL_DIM, vb.pp("self_q")).map_err(candle_err)?;
        let self_k = linear(MODEL_DIM, MODEL_DIM, vb.pp("self_k")).map_err(candle_err)?;
        let self_v = linear(MODEL_DIM, MODEL_DIM, vb.pp("self_v")).map_err(candle_err)?;

        // Cross-attention
        let cross_norm = RmsNorm::new(MODEL_DIM, vb.pp("cross_norm")).map_err(candle_err)?;
        let cross_q = linear(MODEL_DIM, MODEL_DIM, vb.pp("cross_q")).map_err(candle_err)?;
        let cross_k = linear(SLOT_DIM, MODEL_DIM, vb.pp("cross_k")).map_err(candle_err)?;
        let cross_v = linear(SLOT_DIM, MODEL_DIM, vb.pp("cross_v")).map_err(candle_err)?;

        // Feed-forward with expansion (MODEL_DIM → hidden_dim → MODEL_DIM)
        let ffn_norm = RmsNorm::new(MODEL_DIM, vb.pp("ffn_norm")).map_err(candle_err)?;
        let ffn1 =
            linear(MODEL_DIM, config.hidden_dim, vb.pp("ffn1")).map_err(candle_err)?;
        let ffn2 =
            linear(config.hidden_dim, MODEL_DIM, vb.pp("ffn2")).map_err(candle_err)?;

        // Output: model_dim → embed_dim (for tied logits)
        let output_proj =
            linear(MODEL_DIM, config.embed_dim, vb.pp("output_proj")).map_err(candle_err)?;

        Ok(Self {
            token_embed,
            input_proj,
            pos_encoding,
            self_norm,
            self_q,
            self_k,
            self_v,
            cross_norm,
            cross_q,
            cross_k,
            cross_v,
            ffn_norm,
            ffn1,
            ffn2,
            output_proj,
            var_map,
            config: config.clone(),
            device: device.clone(),
        })
    }

    /// Loads decoder weights from a safetensors file.
    pub fn load(
        config: &CodeDecoderConfig,
        path: &std::path::Path,
        device: &Device,
    ) -> Result<Self, VoltError> {
        let mut decoder = Self::new_random(config, device)?;
        decoder.var_map.load(path).map_err(|e| VoltError::StorageError {
            message: format!("failed to load decoder weights from {}: {e}", path.display()),
        })?;
        Ok(decoder)
    }

    /// Saves decoder weights to a safetensors file.
    pub fn save(&self, path: &std::path::Path) -> Result<(), VoltError> {
        self.var_map.save(path).map_err(|e| VoltError::StorageError {
            message: format!("failed to save decoder weights to {}: {e}", path.display()),
        })
    }

    /// Returns a reference to the VarMap for optimizer construction.
    pub fn var_map(&self) -> &VarMap {
        &self.var_map
    }

    /// Returns the decoder configuration.
    pub fn config(&self) -> &CodeDecoderConfig {
        &self.config
    }

    /// Forward pass with teacher forcing: context vectors + input tokens → logits.
    ///
    /// # Arguments
    ///
    /// * `context` — Encoder context vectors, shape `[batch, n_ctx, dim]`.
    ///   During training this is per-token CNN features `[batch, seq_len, 256]`.
    ///   During inference this can be TensorFrame slots `[batch, 16, 256]`.
    /// * `input_ids` — Shifted-right target tokens (0=BOS prefix), shape `[batch, seq_len]` (u32).
    ///
    /// # Returns
    ///
    /// Token logits, shape `[batch, seq_len, vocab_size]`.
    pub fn forward(
        &self,
        context: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor, VoltError> {
        let batch_size = input_ids.dims()[0];
        let seq_len = input_ids.dims()[1].min(self.config.max_seq_len);

        // === Input embedding + position ===

        // Token embedding lookup: [batch, seq_len] → [batch, seq_len, embed_dim]
        let embeds = embed_lookup(&self.token_embed, input_ids)?;

        // Project to model_dim: [batch, seq_len, MODEL_DIM]
        let x = self.input_proj.forward(&embeds).map_err(candle_err)?;

        // Add positional encoding
        let pos = self
            .pos_encoding
            .narrow(0, 0, seq_len)
            .map_err(candle_err)?
            .unsqueeze(0)
            .map_err(candle_err)?;
        let mut x = x.broadcast_add(&pos).map_err(candle_err)?;

        // === Causal self-attention (pre-norm) ===
        let x_normed = self.self_norm.forward(&x).map_err(candle_err)?;

        let q = self.self_q.forward(&x_normed).map_err(candle_err)?;
        let k = self.self_k.forward(&x_normed).map_err(candle_err)?;
        let v = self.self_v.forward(&x_normed).map_err(candle_err)?;

        let inv_scale = 1.0 / (MODEL_DIM as f64).sqrt();
        let k_t = k.transpose(1, 2).map_err(candle_err)?; // [batch, MODEL_DIM, seq_len]
        let attn = q.matmul(&k_t).map_err(candle_err)?; // [batch, seq_len, seq_len]
        let attn = (attn * inv_scale).map_err(candle_err)?;

        // Apply causal mask (prevent attending to future positions)
        let mask = causal_mask(seq_len, &self.device)?; // [1, seq_len, seq_len]
        let attn = attn.broadcast_add(&mask).map_err(candle_err)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1).map_err(candle_err)?;

        let self_attn_out = attn.matmul(&v).map_err(candle_err)?; // [batch, seq_len, MODEL_DIM]
        x = x.broadcast_add(&self_attn_out).map_err(candle_err)?; // residual

        // === Cross-attention to encoder context (pre-norm) ===
        let x_normed = self.cross_norm.forward(&x).map_err(candle_err)?;

        let cq = self.cross_q.forward(&x_normed).map_err(candle_err)?; // [batch, seq_len, MODEL_DIM]
        let ck = self.cross_k.forward(context).map_err(candle_err)?; // [batch, n_ctx, MODEL_DIM]
        let cv = self.cross_v.forward(context).map_err(candle_err)?; // [batch, n_ctx, MODEL_DIM]

        let ck_t = ck.transpose(1, 2).map_err(candle_err)?; // [batch, MODEL_DIM, n_ctx]
        let cross_attn = cq.matmul(&ck_t).map_err(candle_err)?; // [batch, seq_len, n_ctx]
        let cross_attn = (cross_attn * inv_scale).map_err(candle_err)?;
        let cross_attn = candle_nn::ops::softmax(&cross_attn, D::Minus1).map_err(candle_err)?;

        let cross_out = cross_attn.matmul(&cv).map_err(candle_err)?; // [batch, seq_len, MODEL_DIM]
        x = x.broadcast_add(&cross_out).map_err(candle_err)?; // residual

        // === Feed-forward network (pre-norm) ===
        let x_normed = self.ffn_norm.forward(&x).map_err(candle_err)?;

        let h = self.ffn1.forward(&x_normed).map_err(candle_err)?; // [batch, seq_len, hidden_dim]
        let h = h.gelu_erf().map_err(candle_err)?;
        let h = self.ffn2.forward(&h).map_err(candle_err)?; // [batch, seq_len, MODEL_DIM]
        x = x.broadcast_add(&h).map_err(candle_err)?; // residual

        // === Output logits ===

        // Project to embed dim: [batch, seq_len, embed_dim]
        let out = self.output_proj.forward(&x).map_err(candle_err)?;

        // Tied embedding logits: out @ token_embed.T → [batch, seq_len, vocab]
        let embed_t = self.token_embed.t().map_err(candle_err)?;
        let embed_t = embed_t
            .unsqueeze(0)
            .map_err(candle_err)?
            .expand((batch_size, self.config.embed_dim, self.config.vocab_size))
            .map_err(candle_err)?;
        let logits = out.matmul(&embed_t).map_err(candle_err)?;

        Ok(logits)
    }

    /// Generate token IDs via autoregressive greedy decoding.
    ///
    /// # Arguments
    ///
    /// * `context` — Encoder context vectors, shape `[1, n_ctx, dim]`.
    ///   Can be per-token features or TensorFrame slot vectors.
    /// * `max_len` — Maximum number of tokens to generate.
    ///
    /// # Returns
    ///
    /// Token IDs as a Vec<u32>.
    pub fn generate(
        &self,
        context: &Tensor,
        max_len: usize,
    ) -> Result<Vec<u32>, VoltError> {
        let max_len = max_len.min(self.config.max_seq_len);
        let mut generated: Vec<u32> = vec![0]; // BOS token

        for _ in 0..max_len {
            // Build input tensor: [1, current_len]
            let current_len = generated.len();
            let input = Tensor::from_vec(generated.clone(), (1, current_len), &self.device)
                .map_err(candle_err)?;

            // Forward: get logits for all positions
            let logits = self.forward(context, &input)?;

            // Take logits at the last position → next token
            let last_pos = generated.len() - 1;
            let last_logits = logits
                .narrow(1, last_pos, 1)
                .map_err(candle_err)?
                .squeeze(1)
                .map_err(candle_err)?; // [1, vocab]

            let next_id: u32 = last_logits
                .argmax(D::Minus1)
                .map_err(candle_err)?
                .squeeze(0)
                .map_err(candle_err)?
                .to_scalar()
                .map_err(candle_err)?;

            generated.push(next_id);

            if generated.len() > max_len {
                break;
            }
        }

        // Remove BOS token (position 0)
        Ok(generated[1..].to_vec())
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

/// Embedding lookup: index into embedding table.
///
/// `embed_table`: `[vocab, dim]`, `ids`: `[batch, seq_len]` (u32).
/// Returns: `[batch, seq_len, dim]`.
fn embed_lookup(embed_table: &Tensor, ids: &Tensor) -> Result<Tensor, VoltError> {
    let batch = ids.dims()[0];
    let seq_len = ids.dims()[1];
    let dim = embed_table.dims()[1];

    let flat_ids = ids.reshape(batch * seq_len).map_err(candle_err)?;
    let embeds = embed_table
        .index_select(&flat_ids, 0)
        .map_err(candle_err)?; // [batch*seq, dim]
    embeds.reshape((batch, seq_len, dim)).map_err(candle_err)
}

/// Create a causal attention mask (lower-triangular, -inf for future positions).
///
/// Returns shape `[1, seq_len, seq_len]` for batch broadcasting.
fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor, VoltError> {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (1, seq_len, seq_len), device).map_err(candle_err)
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
    fn decoder_config_defaults() {
        let config = CodeDecoderConfig::default();
        assert_eq!(config.max_seq_len, 256);
        assert_eq!(config.vocab_size, 32768);
        assert_eq!(config.hidden_dim, 1024);
    }

    #[test]
    fn decoder_creates_on_cpu() {
        let config = CodeDecoderConfig::default();
        let decoder = CodeDecoder::new_random(&config, &Device::Cpu).unwrap();
        // Token embed (32K×128) + projections + attention + MLP + pos
        assert!(decoder.param_count() > 4_000_000);
    }

    #[test]
    fn decoder_forward_shapes() {
        let config = CodeDecoderConfig::default();
        let decoder = CodeDecoder::new_random(&config, &Device::Cpu).unwrap();

        // Context: per-token features [batch=2, n_ctx=30, dim=256]
        let context =
            Tensor::zeros((2, 30, SLOT_DIM), DType::F32, &Device::Cpu).unwrap();

        // Input tokens (shifted right, 0=BOS): [batch=2, seq=50]
        let input_ids = Tensor::zeros((2, 50), DType::U32, &Device::Cpu).unwrap();

        let logits = decoder.forward(&context, &input_ids).unwrap();
        assert_eq!(logits.dims(), &[2, 50, 32768]);
    }

    #[test]
    fn decoder_generate_returns_token_ids() {
        let config = CodeDecoderConfig {
            max_seq_len: 20,
            ..CodeDecoderConfig::default()
        };
        let decoder = CodeDecoder::new_random(&config, &Device::Cpu).unwrap();

        // Context: [1, n_ctx=10, dim=256]
        let context =
            Tensor::zeros((1, 10, SLOT_DIM), DType::F32, &Device::Cpu).unwrap();

        let ids = decoder.generate(&context, 10).unwrap();
        assert!(ids.len() <= 10, "got {} ids", ids.len());
        for &id in &ids {
            assert!(
                (id as usize) < config.vocab_size,
                "token id {id} out of range"
            );
        }
    }

    #[test]
    fn causal_mask_blocks_future() {
        let mask = causal_mask(4, &Device::Cpu).unwrap();
        let mask_data: Vec<Vec<f32>> = {
            let m = mask.squeeze(0).unwrap();
            let rows = m.dims()[0];
            let cols = m.dims()[1];
            let flat: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
            (0..rows)
                .map(|r| flat[r * cols..(r + 1) * cols].to_vec())
                .collect()
        };

        // Position 0 can only attend to position 0
        assert_eq!(mask_data[0][0], 0.0);
        assert!(mask_data[0][1].is_infinite() && mask_data[0][1] < 0.0);

        // Position 3 can attend to all positions 0..=3
        assert_eq!(mask_data[3][0], 0.0);
        assert_eq!(mask_data[3][1], 0.0);
        assert_eq!(mask_data[3][2], 0.0);
        assert_eq!(mask_data[3][3], 0.0);
    }

    #[test]
    fn embed_lookup_shapes() {
        let embed_table =
            Tensor::randn(0.0f32, 1.0, (100, 64), &Device::Cpu).unwrap();
        let ids = Tensor::new(&[[1u32, 5, 10], [2, 3, 4]], &Device::Cpu).unwrap();
        let result = embed_lookup(&embed_table, &ids).unwrap();
        assert_eq!(result.dims(), &[2, 3, 64]);
    }

    #[test]
    fn rms_norm_normalizes() {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &Device::Cpu);
        let norm = RmsNorm::new(4, vb.pp("test_norm")).unwrap();

        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();

        // Output should have similar magnitude to input (weight=1)
        assert_eq!(y.dims(), &[1, 4]);
        let vals: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.738
        // normalized ≈ [0.365, 0.730, 1.095, 1.461]
        assert!((vals[0] - 0.365).abs() < 0.01);
    }
}
