//! GPU-accelerated Scaled VFN using candle.
//!
//! Mirrors the CPU [`crate::scaled_vfn::ScaledVfn`] architecture but uses
//! candle tensors for batched GPU execution and autograd support during
//! Phase 2 Flow Matching training.
//!
//! Architecture: Input(256) → Linear(256→H) → GELU → [ResBlock] × N → LayerNorm → Linear(H→256)

use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
use volt_core::{VoltError, SLOT_DIM};

use crate::scaled_vfn::ScaledVfnConfig;

/// GPU-accelerated Scaled Vector Field Network.
///
/// Uses candle tensors for batched forward passes and autograd.
/// Supports both inference (from CPU weights) and training (trainable VarMap).
///
/// # Example
///
/// ```ignore
/// use volt_soft::gpu::scaled_vfn::GpuScaledVfn;
/// use volt_soft::scaled_vfn::ScaledVfnConfig;
/// use candle_core::Device;
/// use candle_nn::VarMap;
///
/// let device = Device::Cpu;
/// let var_map = VarMap::new();
/// let config = ScaledVfnConfig { hidden_dim: 64, num_blocks: 2, ..ScaledVfnConfig::default() };
/// let vfn = GpuScaledVfn::new_trainable(&var_map, &config, &device).unwrap();
/// ```
pub struct GpuScaledVfn {
    proj_up: Linear,
    blocks: Vec<GpuResBlock>,
    final_norm: GpuLayerNorm,
    proj_down: Linear,
    config: ScaledVfnConfig,
    device: Device,
}

struct GpuResBlock {
    norm: GpuLayerNorm,
    linear1: Linear,
    linear2: Linear,
}

struct GpuLayerNorm {
    gamma: Tensor,
    beta: Tensor,
    eps: f64,
}

impl GpuLayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        // input: [batch, dim]
        let mean = input.mean_keepdim(candle_core::D::Minus1)?;
        let centered = input.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let std = (var + self.eps)?.sqrt()?;
        let normed = centered.broadcast_div(&std)?;
        normed.broadcast_mul(&self.gamma)?.broadcast_add(&self.beta)
    }
}

impl std::fmt::Debug for GpuScaledVfn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuScaledVfn({}→{}×{}→{}, {:.1}M params, device={:?})",
            self.config.io_dim,
            self.config.hidden_dim,
            self.config.num_blocks,
            self.config.io_dim,
            self.config.param_count() as f64 / 1_000_000.0,
            self.device
        )
    }
}

impl GpuScaledVfn {
    /// Creates a trainable GPU ScaledVfn backed by a VarMap.
    ///
    /// All parameters are tracked for autograd — used for Flow Matching training.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if parameter creation fails.
    pub fn new_trainable(
        var_map: &VarMap,
        config: &ScaledVfnConfig,
        device: &Device,
    ) -> Result<Self, VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("GpuScaledVfn new_trainable: {e}"),
        };

        let vb = VarBuilder::from_varmap(var_map, DType::F32, device);

        let proj_up = linear(config.io_dim, config.hidden_dim, vb.pp("svfn.proj_up"))
            .map_err(map_err)?;

        let mut blocks = Vec::with_capacity(config.num_blocks);
        for i in 0..config.num_blocks {
            let prefix = format!("svfn.block_{i}");
            let norm = Self::make_trainable_norm(
                config.hidden_dim,
                &vb.pp(format!("{prefix}.norm")),
            )?;
            let l1 = linear(
                config.hidden_dim,
                config.hidden_dim,
                vb.pp(format!("{prefix}.l1")),
            )
            .map_err(map_err)?;
            let l2 = linear(
                config.hidden_dim,
                config.hidden_dim,
                vb.pp(format!("{prefix}.l2")),
            )
            .map_err(map_err)?;
            blocks.push(GpuResBlock {
                norm,
                linear1: l1,
                linear2: l2,
            });
        }

        let final_norm =
            Self::make_trainable_norm(config.hidden_dim, &vb.pp("svfn.final_norm"))?;
        let proj_down = linear(config.hidden_dim, config.io_dim, vb.pp("svfn.proj_down"))
            .map_err(map_err)?;

        Ok(Self {
            proj_up,
            blocks,
            final_norm,
            proj_down,
            config: config.clone(),
            device: device.clone(),
        })
    }

    /// Creates a GPU ScaledVfn from CPU weights (non-trainable).
    ///
    /// Transfers all weight matrices to the specified device.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if weight transfer fails.
    pub fn from_cpu(
        cpu_vfn: &crate::scaled_vfn::ScaledVfn,
        device: &Device,
    ) -> Result<Self, VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("GpuScaledVfn from_cpu: {e}"),
        };

        let config = cpu_vfn.config().clone();
        let all_weights = cpu_vfn.all_weights();
        let all_norms = cpu_vfn.all_norms();

        // First entry is proj_up
        let (_, (out_dim, in_dim), w_data, b_data) = &all_weights[0];
        let proj_up = make_linear(w_data, b_data, *out_dim, *in_dim, device).map_err(map_err)?;

        // Block weights: 2 entries per block (l1, l2) starting at index 1
        let mut blocks = Vec::with_capacity(config.num_blocks);
        for i in 0..config.num_blocks {
            let idx = 1 + i * 2;
            let (_, (h, _), w1, b1) = &all_weights[idx];
            let (_, (_, _), w2, b2) = &all_weights[idx + 1];
            let l1 = make_linear(w1, b1, *h, *h, device).map_err(map_err)?;
            let l2 = make_linear(w2, b2, *h, *h, device).map_err(map_err)?;

            let (gamma, beta) = &all_norms[i];
            let norm = make_layer_norm(gamma, beta, device).map_err(map_err)?;

            blocks.push(GpuResBlock {
                norm,
                linear1: l1,
                linear2: l2,
            });
        }

        // Final norm is the last entry in all_norms
        let (gamma, beta) = &all_norms[config.num_blocks];
        let final_norm = make_layer_norm(gamma, beta, device).map_err(map_err)?;

        // Last entry in all_weights is proj_down
        let last = all_weights.len() - 1;
        let (_, (out_dim, in_dim), w_data, b_data) = &all_weights[last];
        let proj_down = make_linear(w_data, b_data, *out_dim, *in_dim, device).map_err(map_err)?;

        Ok(Self {
            proj_up,
            blocks,
            final_norm,
            proj_down,
            config,
            device: device.clone(),
        })
    }

    /// Batched forward pass through the scaled VFN.
    ///
    /// Input shape: `[N, SLOT_DIM]` where N is the batch of slot embeddings.
    /// Output shape: `[N, SLOT_DIM]` — drift vectors.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if tensor operations fail.
    pub fn forward_batch(&self, input: &Tensor) -> Result<Tensor, VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("GpuScaledVfn forward_batch: {e}"),
        };

        // Project up + GELU
        let mut h = self.proj_up.forward(input).map_err(map_err)?;
        h = h.gelu_erf().map_err(map_err)?;

        // Residual blocks
        for block in &self.blocks {
            let normed = block.norm.forward(&h).map_err(map_err)?;
            let ff = block.linear1.forward(&normed).map_err(map_err)?;
            let ff = ff.gelu_erf().map_err(map_err)?;
            let ff = block.linear2.forward(&ff).map_err(map_err)?;
            h = (h + ff).map_err(map_err)?;
        }

        // Final norm + project down
        h = self.final_norm.forward(&h).map_err(map_err)?;
        self.proj_down.forward(&h).map_err(map_err)
    }

    /// Single-slot forward pass (convenience wrapper).
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if tensor operations fail.
    pub fn forward_single(
        &self,
        input: &[f32; SLOT_DIM],
    ) -> Result<[f32; SLOT_DIM], VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("GpuScaledVfn forward_single: {e}"),
        };

        let tensor =
            Tensor::from_slice(input.as_slice(), &[1, SLOT_DIM], &self.device).map_err(map_err)?;
        let output = self.forward_batch(&tensor)?;
        let flat = output.flatten_all().map_err(map_err)?;
        let data = flat.to_vec1::<f32>().map_err(map_err)?;

        let mut result = [0.0f32; SLOT_DIM];
        result.copy_from_slice(&data[..SLOT_DIM]);
        Ok(result)
    }

    /// Returns the device this VFN operates on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the config for this VFN.
    pub fn config(&self) -> &ScaledVfnConfig {
        &self.config
    }

    fn make_trainable_norm(
        dim: usize,
        vb: &VarBuilder,
    ) -> Result<GpuLayerNorm, VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("GpuScaledVfn make_trainable_norm: {e}"),
        };

        let gamma = vb
            .get_with_hints(dim, "gamma", candle_nn::Init::Const(1.0))
            .map_err(map_err)?;
        let beta = vb
            .get_with_hints(dim, "beta", candle_nn::Init::Const(0.0))
            .map_err(map_err)?;

        Ok(GpuLayerNorm {
            gamma,
            beta,
            eps: 1e-5,
        })
    }
}

fn make_linear(
    weights: &[f32],
    bias: &[f32],
    out_dim: usize,
    in_dim: usize,
    device: &Device,
) -> Result<Linear, candle_core::Error> {
    let w = Tensor::from_slice(weights, (out_dim, in_dim), device)?;
    let b = Tensor::from_slice(bias, (out_dim,), device)?;
    Ok(Linear::new(w, Some(b)))
}

fn make_layer_norm(
    gamma: &[f32],
    beta: &[f32],
    device: &Device,
) -> Result<GpuLayerNorm, candle_core::Error> {
    let dim = gamma.len();
    let g = Tensor::from_slice(gamma, (1, dim), device)?;
    let b = Tensor::from_slice(beta, (1, dim), device)?;
    Ok(GpuLayerNorm {
        gamma: g,
        beta: b,
        eps: 1e-5,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> ScaledVfnConfig {
        ScaledVfnConfig {
            hidden_dim: 64,
            num_blocks: 2,
            io_dim: SLOT_DIM,
        }
    }

    #[test]
    fn new_trainable_creates_valid_vfn() {
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let config = small_config();
        let vfn = GpuScaledVfn::new_trainable(&var_map, &config, &device).unwrap();
        let input = [0.1f32; SLOT_DIM];
        let output = vfn.forward_single(&input).unwrap();
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn forward_batch_correct_shape() {
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let config = small_config();
        let vfn = GpuScaledVfn::new_trainable(&var_map, &config, &device).unwrap();

        let batch: Vec<f32> = (0..4 * SLOT_DIM).map(|i| (i as f32) * 0.001).collect();
        let input = Tensor::from_vec(batch, (4, SLOT_DIM), &device).unwrap();

        let output = vfn.forward_batch(&input).unwrap();
        assert_eq!(output.dims(), &[4, SLOT_DIM]);
    }

    #[test]
    fn from_cpu_matches_cpu_output() {
        let config = small_config();
        let cpu_vfn = crate::scaled_vfn::ScaledVfn::new_random(42, &config);
        let device = Device::Cpu;
        let gpu_vfn = GpuScaledVfn::from_cpu(&cpu_vfn, &device).unwrap();

        let input = [0.1f32; SLOT_DIM];
        let cpu_out = cpu_vfn.forward(&input).unwrap();
        let gpu_out = gpu_vfn.forward_single(&input).unwrap();

        for (i, (a, b)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "dim {} mismatch: cpu={}, gpu={}",
                i, a, b,
            );
        }
    }

    #[test]
    fn debug_format_readable() {
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let config = small_config();
        let vfn = GpuScaledVfn::new_trainable(&var_map, &config, &device).unwrap();
        let debug = format!("{:?}", vfn);
        assert!(debug.contains("GpuScaledVfn"));
    }

    #[test]
    fn output_is_finite() {
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let config = small_config();
        let vfn = GpuScaledVfn::new_trainable(&var_map, &config, &device).unwrap();

        let mut rng = crate::nn::Rng::new(42);
        let batch: Vec<f32> = (0..8 * SLOT_DIM)
            .map(|_| rng.next_f32_range(-1.0, 1.0))
            .collect();
        let input = Tensor::from_vec(batch, (8, SLOT_DIM), &device).unwrap();

        let output = vfn.forward_batch(&input).unwrap();
        let flat = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(flat.iter().all(|x| x.is_finite()));
    }
}
