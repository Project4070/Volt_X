//! Scaled Vector Field Network for Phase 2+ training.
//!
//! A deep residual MLP (~51M params) with sinusoidal time conditioning,
//! replacing the 3-layer MLP (525K params) for code-domain flow matching.
//!
//! ## Architecture
//!
//! ```text
//! Input: [SLOT_DIM=256], Time: t ∈ [0,1]
//! → Time Embedding: sinusoidal(64) → Linear(128, hidden) + GELU
//! → Entry: Linear(256, hidden) + GELU
//! → Add time embedding
//! → N × ResidualBlock { RMSNorm → Linear(H,H) + GELU → Linear(H,H) → residual }
//! → RMSNorm → Linear(hidden, 256)
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use volt_soft::scaled_vfn::{ScaledVfn, ScaledVfnConfig};
//! use candle_core::Device;
//! use candle_nn::VarMap;
//!
//! let config = ScaledVfnConfig::default();
//! let var_map = VarMap::new();
//! let device = Device::Cpu;
//! let vfn = ScaledVfn::new_trainable(&config, &var_map, &device).unwrap();
//! assert!(vfn.param_count() > 50_000_000);
//! ```

use std::path::Path;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
use volt_core::{VoltError, SLOT_DIM};

// ---------------------------------------------------------------------------
// RMSNorm (custom, CUDA-compatible — same pattern as code_decoder.rs)
// ---------------------------------------------------------------------------

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
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = (variance + self.eps)?.sqrt()?;
        let normalized = x.broadcast_div(&rms)?;
        normalized.broadcast_mul(&self.weight)
    }
}

// ---------------------------------------------------------------------------
// Residual Block
// ---------------------------------------------------------------------------

struct ResidualBlock {
    norm: RmsNorm,
    linear1: Linear,
    linear2: Linear,
}

impl ResidualBlock {
    fn new(dim: usize, vb: VarBuilder<'_>) -> Result<Self, candle_core::Error> {
        let norm = RmsNorm::new(dim, vb.pp("norm"))?;
        let linear1 = linear(dim, dim, vb.pp("fc1"))?;
        let linear2 = linear(dim, dim, vb.pp("fc2"))?;
        Ok(Self {
            norm,
            linear1,
            linear2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let h = self.norm.forward(x)?;
        let h = self.linear1.forward(&h)?.gelu_erf()?;
        let h = self.linear2.forward(&h)?;
        x + h
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the [`ScaledVfn`].
///
/// # Example
///
/// ```ignore
/// use volt_soft::scaled_vfn::ScaledVfnConfig;
///
/// let config = ScaledVfnConfig::default();
/// assert_eq!(config.hidden_dim, 2048);
/// assert_eq!(config.num_blocks, 6);
/// ```
#[derive(Debug, Clone)]
pub struct ScaledVfnConfig {
    /// Width of residual blocks (default: 2048).
    pub hidden_dim: usize,

    /// Number of residual blocks (default: 6).
    pub num_blocks: usize,

    /// Number of sinusoidal frequency pairs for time embedding (default: 64).
    /// Produces `2 * time_embed_freqs` raw dimensions before projection.
    pub time_embed_freqs: usize,
}

impl Default for ScaledVfnConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 2048,
            num_blocks: 6,
            time_embed_freqs: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// ScaledVfn
// ---------------------------------------------------------------------------

/// Scaled Vector Field Network (~51M params at default config).
///
/// Deep residual MLP with sinusoidal time conditioning for flow matching.
/// Operates on individual slot vectors (`[N, 256]`) and predicts drift
/// vectors conditioned on interpolation time `t`.
///
/// # Example
///
/// ```ignore
/// use volt_soft::scaled_vfn::{ScaledVfn, ScaledVfnConfig};
/// use candle_core::{Device, Tensor};
/// use candle_nn::VarMap;
///
/// let config = ScaledVfnConfig::default();
/// let var_map = VarMap::new();
/// let device = Device::Cpu;
/// let vfn = ScaledVfn::new_trainable(&config, &var_map, &device).unwrap();
///
/// let input = Tensor::zeros((4, 256), candle_core::DType::F32, &device).unwrap();
/// let time = Tensor::zeros(4, candle_core::DType::F32, &device).unwrap();
/// let drift = vfn.forward_batch(&input, &time).unwrap();
/// assert_eq!(drift.dims(), &[4, 256]);
/// ```
pub struct ScaledVfn {
    // Time embedding
    time_proj: Linear, // 2*freqs → hidden_dim

    // Entry projection
    entry: Linear, // SLOT_DIM → hidden_dim

    // Residual blocks
    blocks: Vec<ResidualBlock>,

    // Final normalization + exit
    final_norm: RmsNorm,
    exit: Linear, // hidden_dim → SLOT_DIM

    // Metadata
    config: ScaledVfnConfig,
    var_map: VarMap,
    device: Device,
}

impl std::fmt::Debug for ScaledVfn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ScaledVfn(blocks={}, hidden={}, params={}, device={:?})",
            self.config.num_blocks,
            self.config.hidden_dim,
            self.param_count(),
            self.device
        )
    }
}

impl ScaledVfn {
    /// Creates a new ScaledVfn with trainable parameters backed by a VarMap.
    ///
    /// Used for flow matching training — weights are tracked for autograd.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if parameter creation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use volt_soft::scaled_vfn::{ScaledVfn, ScaledVfnConfig};
    /// use candle_core::Device;
    /// use candle_nn::VarMap;
    ///
    /// let var_map = VarMap::new();
    /// let vfn = ScaledVfn::new_trainable(
    ///     &ScaledVfnConfig::default(),
    ///     &var_map,
    ///     &Device::Cpu,
    /// ).unwrap();
    /// ```
    pub fn new_trainable(
        config: &ScaledVfnConfig,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self, VoltError> {
        let map_err =
            |e: candle_core::Error| VoltError::Internal {
                message: format!("ScaledVfn::new_trainable: {e}"),
            };

        let vb = VarBuilder::from_varmap(var_map, DType::F32, device);

        let time_raw_dim = config.time_embed_freqs * 2;
        let time_proj =
            linear(time_raw_dim, config.hidden_dim, vb.pp("time_proj")).map_err(map_err)?;

        let entry = linear(SLOT_DIM, config.hidden_dim, vb.pp("entry")).map_err(map_err)?;

        let mut blocks = Vec::with_capacity(config.num_blocks);
        for i in 0..config.num_blocks {
            let block =
                ResidualBlock::new(config.hidden_dim, vb.pp(format!("block_{i}")))
                    .map_err(map_err)?;
            blocks.push(block);
        }

        let final_norm =
            RmsNorm::new(config.hidden_dim, vb.pp("final_norm")).map_err(map_err)?;
        let exit = linear(config.hidden_dim, SLOT_DIM, vb.pp("exit")).map_err(map_err)?;

        Ok(Self {
            time_proj,
            entry,
            blocks,
            final_norm,
            exit,
            config: config.clone(),
            var_map: var_map.clone(),
            device: device.clone(),
        })
    }

    /// Loads a ScaledVfn from a safetensors checkpoint.
    ///
    /// Creates the model structure then loads weights from disk.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if loading fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use volt_soft::scaled_vfn::{ScaledVfn, ScaledVfnConfig};
    /// use candle_core::Device;
    ///
    /// let vfn = ScaledVfn::load(
    ///     &ScaledVfnConfig::default(),
    ///     "checkpoints/scaled_vfn.safetensors",
    ///     &Device::Cpu,
    /// ).unwrap();
    /// ```
    pub fn load<P: AsRef<Path>>(
        config: &ScaledVfnConfig,
        path: P,
        device: &Device,
    ) -> Result<Self, VoltError> {
        let mut var_map = VarMap::new();
        let mut vfn = Self::new_trainable(config, &var_map, device)?;

        var_map.load(path.as_ref()).map_err(|e| VoltError::StorageError {
            message: format!(
                "failed to load ScaledVfn from {}: {e}",
                path.as_ref().display()
            ),
        })?;

        vfn.var_map = var_map;
        Ok(vfn)
    }

    /// Saves model weights to a safetensors file.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if saving fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), VoltError> {
        self.var_map
            .save(path.as_ref())
            .map_err(|e| VoltError::StorageError {
                message: format!(
                    "failed to save ScaledVfn to {}: {e}",
                    path.as_ref().display()
                ),
            })
    }

    /// Returns the VarMap backing this model's parameters.
    pub fn var_map(&self) -> &VarMap {
        &self.var_map
    }

    /// Returns the device this VFN operates on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the model configuration.
    pub fn config(&self) -> &ScaledVfnConfig {
        &self.config
    }

    /// Counts total trainable parameters.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use volt_soft::scaled_vfn::{ScaledVfn, ScaledVfnConfig};
    /// use candle_core::Device;
    /// use candle_nn::VarMap;
    ///
    /// let var_map = VarMap::new();
    /// let vfn = ScaledVfn::new_trainable(
    ///     &ScaledVfnConfig::default(),
    ///     &var_map,
    ///     &Device::Cpu,
    /// ).unwrap();
    /// assert!(vfn.param_count() > 50_000_000);
    /// ```
    pub fn param_count(&self) -> usize {
        self.var_map
            .all_vars()
            .iter()
            .map(|v| v.elem_count())
            .sum()
    }

    /// Batched forward pass: process N slot vectors with time conditioning.
    ///
    /// # Arguments
    ///
    /// * `input` — slot vectors, shape `[N, SLOT_DIM]`
    /// * `time` — interpolation times, shape `[N]`, values in `[0, 1]`
    ///
    /// # Returns
    ///
    /// Predicted drift vectors, shape `[N, SLOT_DIM]`.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if tensor operations fail.
    pub fn forward_batch(
        &self,
        input: &Tensor,
        time: &Tensor,
    ) -> Result<Tensor, VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("ScaledVfn::forward_batch: {e}"),
        };

        // Sinusoidal time embedding: [N] → [N, 2*freqs]
        let t_embed = self.sinusoidal_embedding(time).map_err(map_err)?;

        // Project time embedding: [N, 2*freqs] → [N, hidden]
        let t_proj = self.time_proj.forward(&t_embed).map_err(map_err)?;
        let t_proj = t_proj.gelu_erf().map_err(map_err)?;

        // Entry projection: [N, SLOT_DIM] → [N, hidden]
        let mut h = self.entry.forward(input).map_err(map_err)?;
        h = h.gelu_erf().map_err(map_err)?;

        // Add time conditioning
        h = (h + t_proj).map_err(map_err)?;

        // Residual blocks
        for block in &self.blocks {
            h = block.forward(&h).map_err(map_err)?;
        }

        // Final norm + exit projection
        h = self.final_norm.forward(&h).map_err(map_err)?;
        self.exit.forward(&h).map_err(map_err)
    }

    /// Single-slot forward pass (convenience wrapper).
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if tensor operations fail.
    pub fn forward_single(
        &self,
        input: &[f32; SLOT_DIM],
        time: f32,
    ) -> Result<[f32; SLOT_DIM], VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("ScaledVfn::forward_single: {e}"),
        };

        let input_t = Tensor::from_slice(input.as_slice(), &[1, SLOT_DIM], &self.device)
            .map_err(map_err)?;
        let time_t =
            Tensor::from_slice(&[time], &[1], &self.device).map_err(map_err)?;

        let output = self.forward_batch(&input_t, &time_t)?;
        let flat = output.flatten_all().map_err(map_err)?;
        let data = flat.to_vec1::<f32>().map_err(map_err)?;

        let mut result = [0.0f32; SLOT_DIM];
        result.copy_from_slice(&data[..SLOT_DIM]);
        Ok(result)
    }

    /// Computes sinusoidal time embeddings.
    ///
    /// `time`: shape `[N]` with values in `[0, 1]`.
    /// Returns: shape `[N, 2*freqs]` with interleaved sin/cos.
    fn sinusoidal_embedding(
        &self,
        time: &Tensor,
    ) -> Result<Tensor, candle_core::Error> {
        let n_freqs = self.config.time_embed_freqs;

        // Log-spaced frequencies from 1.0 to 10000.0
        let freqs: Vec<f32> = (0..n_freqs)
            .map(|i| {
                let frac = i as f32 / (n_freqs - 1).max(1) as f32;
                (1.0f32.ln() + frac * (10000.0f32.ln() - 1.0f32.ln())).exp()
            })
            .collect();

        let freq_tensor =
            Tensor::from_vec(freqs, (1, n_freqs), time.device())?;

        // time: [N] → [N, 1]
        let t = time.unsqueeze(D::Minus1)?;

        // angles: [N, n_freqs] = t * freqs
        let angles = t.broadcast_mul(&freq_tensor)?;

        // sin and cos: each [N, n_freqs]
        let sin_vals = angles.sin()?;
        let cos_vals = angles.cos()?;

        // Interleave: [N, 2*n_freqs]
        Tensor::cat(&[&sin_vals, &cos_vals], D::Minus1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_sensible() {
        let config = ScaledVfnConfig::default();
        assert_eq!(config.hidden_dim, 2048);
        assert_eq!(config.num_blocks, 6);
        assert_eq!(config.time_embed_freqs, 64);
    }

    #[test]
    fn new_trainable_creates_valid_vfn() {
        let config = ScaledVfnConfig::default();
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vfn = ScaledVfn::new_trainable(&config, &var_map, &device).unwrap();
        assert!(format!("{:?}", vfn).contains("ScaledVfn"));
    }

    #[test]
    fn param_count_exceeds_50m() {
        let config = ScaledVfnConfig::default();
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vfn = ScaledVfn::new_trainable(&config, &var_map, &device).unwrap();
        let count = vfn.param_count();
        assert!(
            count > 50_000_000,
            "expected >50M params, got {count}"
        );
        assert!(
            count < 60_000_000,
            "expected <60M params, got {count}"
        );
    }

    #[test]
    fn forward_batch_correct_shape() {
        let config = ScaledVfnConfig {
            hidden_dim: 128,
            num_blocks: 2,
            time_embed_freqs: 8,
        };
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vfn = ScaledVfn::new_trainable(&config, &var_map, &device).unwrap();

        let input = Tensor::zeros((4, SLOT_DIM), DType::F32, &device).unwrap();
        let time = Tensor::from_vec(vec![0.0f32, 0.25, 0.5, 1.0], 4, &device).unwrap();

        let output = vfn.forward_batch(&input, &time).unwrap();
        assert_eq!(output.dims(), &[4, SLOT_DIM]);
    }

    #[test]
    fn forward_single_correct_size() {
        let config = ScaledVfnConfig {
            hidden_dim: 128,
            num_blocks: 2,
            time_embed_freqs: 8,
        };
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vfn = ScaledVfn::new_trainable(&config, &var_map, &device).unwrap();

        let input = [0.1f32; SLOT_DIM];
        let output = vfn.forward_single(&input, 0.5).unwrap();
        assert_eq!(output.len(), SLOT_DIM);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn forward_produces_finite_output() {
        let config = ScaledVfnConfig {
            hidden_dim: 128,
            num_blocks: 2,
            time_embed_freqs: 8,
        };
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vfn = ScaledVfn::new_trainable(&config, &var_map, &device).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (8, SLOT_DIM), &device).unwrap();
        let time = Tensor::from_vec(
            (0..8).map(|i| i as f32 / 7.0).collect::<Vec<_>>(),
            8,
            &device,
        )
        .unwrap();

        let output = vfn.forward_batch(&input, &time).unwrap();
        let data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn save_load_roundtrip() {
        let config = ScaledVfnConfig {
            hidden_dim: 64,
            num_blocks: 1,
            time_embed_freqs: 4,
        };
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vfn = ScaledVfn::new_trainable(&config, &var_map, &device).unwrap();

        let input = [0.5f32; SLOT_DIM];
        let out_before = vfn.forward_single(&input, 0.3).unwrap();

        let temp = std::env::temp_dir().join("scaled_vfn_test.safetensors");
        vfn.save(&temp).unwrap();

        let loaded = ScaledVfn::load(&config, &temp, &device).unwrap();
        let out_after = loaded.forward_single(&input, 0.3).unwrap();

        for (i, (a, b)) in out_before.iter().zip(out_after.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "dim {i}: before={a}, after={b}"
            );
        }

        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn small_config_param_count() {
        let config = ScaledVfnConfig {
            hidden_dim: 64,
            num_blocks: 1,
            time_embed_freqs: 4,
        };
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vfn = ScaledVfn::new_trainable(&config, &var_map, &device).unwrap();

        // time_proj: 8*64 + 64 = 576
        // entry: 256*64 + 64 = 16,448
        // block: norm(64) + fc1(64*64+64) + fc2(64*64+64) = 64 + 4160 + 4160 = 8384
        // final_norm: 64
        // exit: 64*256 + 256 = 16,640
        // Total ≈ 576 + 16448 + 8384 + 64 + 16640 = 42,112
        let count = vfn.param_count();
        assert!(
            count > 40_000 && count < 50_000,
            "small config param count unexpected: {count}"
        );
    }
}
