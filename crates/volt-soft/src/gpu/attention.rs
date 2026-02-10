//! GPU-accelerated cross-slot attention using candle.
//!
//! Computes Q/K/V projections and scaled dot-product attention using
//! batched matrix multiplications — the Attend phase runs as 3 matmuls
//! + 1 softmax instead of O(S²) scalar loops.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};
use volt_core::{VoltError, SLOT_DIM};

/// GPU-accelerated cross-slot attention.
///
/// Mirrors the CPU [`crate::attention::SlotAttention`] but uses batched
/// matrix operations for efficient GPU execution.
///
/// # Example
///
/// ```ignore
/// use volt_soft::gpu::attention::GpuSlotAttention;
/// use candle_core::Device;
///
/// let attn = GpuSlotAttention::new_random(43, &Device::Cpu).unwrap();
/// ```
pub struct GpuSlotAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    scale: f64,
    device: Device,
}

impl std::fmt::Debug for GpuSlotAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuSlotAttention(dim={}, scale={:.4}, device={:?})",
            SLOT_DIM, self.scale, self.device
        )
    }
}

impl GpuSlotAttention {
    /// Creates a new attention module with Xavier random initialization.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if tensor creation fails.
    pub fn new_random(seed: u64, device: &Device) -> Result<Self, VoltError> {
        let mut rng = crate::nn::Rng::new(seed);

        let wq = Self::init_projection(&mut rng, device)?;
        let wk = Self::init_projection(&mut rng, device)?;
        let wv = Self::init_projection(&mut rng, device)?;

        Ok(Self {
            wq,
            wk,
            wv,
            scale: 1.0 / (SLOT_DIM as f64).sqrt(),
            device: device.clone(),
        })
    }

    /// Creates a GPU attention module by loading weights from a CPU one.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if weight transfer fails.
    pub fn from_cpu_attention(
        cpu_attn: &crate::attention::SlotAttention,
        device: &Device,
    ) -> Result<Self, VoltError> {
        let (q, k, v) = cpu_attn.projections();

        let wq = Self::cpu_to_candle_linear(q, device).map_err(|e| VoltError::Internal {
            message: format!("wq transfer: {e}"),
        })?;
        let wk = Self::cpu_to_candle_linear(k, device).map_err(|e| VoltError::Internal {
            message: format!("wk transfer: {e}"),
        })?;
        let wv = Self::cpu_to_candle_linear(v, device).map_err(|e| VoltError::Internal {
            message: format!("wv transfer: {e}"),
        })?;

        Ok(Self {
            wq,
            wk,
            wv,
            scale: 1.0 / (SLOT_DIM as f64).sqrt(),
            device: device.clone(),
        })
    }

    /// Batched attention forward pass.
    ///
    /// Input: `states` tensor of shape `[N, SLOT_DIM]` (N active slots).
    /// Output: `messages` tensor of shape `[N, SLOT_DIM]`.
    ///
    /// Computes:
    /// ```text
    /// Q = states @ Wq    [N, D]
    /// K = states @ Wk    [N, D]
    /// V = states @ Wv    [N, D]
    /// scores = Q @ K^T / sqrt(D)  [N, N]
    /// weights = softmax(scores)    [N, N]
    /// messages = weights @ V       [N, D]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if tensor operations fail.
    pub fn forward_batch(&self, states: &Tensor) -> Result<Tensor, VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("GpuSlotAttention forward_batch: {e}"),
        };

        // Q, K, V projections
        let q = self.wq.forward(states).map_err(map_err)?;
        let k = self.wk.forward(states).map_err(map_err)?;
        let v = self.wv.forward(states).map_err(map_err)?;

        // Scaled dot-product attention: Q @ K^T / sqrt(d)
        let k_t = k.t().map_err(map_err)?;
        let scores = q.matmul(&k_t).map_err(map_err)?;
        let scores = (scores * self.scale).map_err(map_err)?;

        // Softmax over the key dimension (last dim)
        let weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1).map_err(map_err)?;

        // Weighted value aggregation: weights @ V
        weights.matmul(&v).map_err(map_err)
    }

    /// Returns the device this module operates on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    // -- Internal helpers --

    fn init_projection(rng: &mut crate::nn::Rng, device: &Device) -> Result<Linear, VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("GpuSlotAttention init_projection: {e}"),
        };

        let limit = (6.0 / (SLOT_DIM + SLOT_DIM) as f32).sqrt();
        let weights: Vec<f32> = (0..SLOT_DIM * SLOT_DIM)
            .map(|_| rng.next_f32_range(-limit, limit))
            .collect();

        let w = Tensor::from_vec(weights, (SLOT_DIM, SLOT_DIM), device).map_err(map_err)?;
        let b = Tensor::zeros(SLOT_DIM, DType::F32, device).map_err(map_err)?;

        Ok(Linear::new(w, Some(b)))
    }

    fn cpu_to_candle_linear(
        cpu_layer: &crate::nn::Linear,
        device: &Device,
    ) -> Result<Linear, VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("cpu_to_candle_linear: {e}"),
        };

        let in_dim = cpu_layer.in_dim();
        let out_dim = cpu_layer.out_dim();

        let w = Tensor::from_slice(cpu_layer.weights(), (out_dim, in_dim), device)
            .map_err(map_err)?;
        let b = Tensor::from_slice(cpu_layer.bias(), (out_dim,), device).map_err(map_err)?;

        Ok(Linear::new(w, Some(b)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::MAX_SLOTS;

    #[test]
    fn new_random_creates_valid_attention() {
        let device = Device::Cpu;
        let attn = GpuSlotAttention::new_random(43, &device).unwrap();
        assert!(format!("{:?}", attn).contains("GpuSlotAttention"));
    }

    #[test]
    fn forward_batch_correct_shape() {
        let device = Device::Cpu;
        let attn = GpuSlotAttention::new_random(43, &device).unwrap();

        let batch: Vec<f32> = (0..4 * SLOT_DIM).map(|i| (i as f32) * 0.001).collect();
        let states = Tensor::from_vec(batch, (4, SLOT_DIM), &device).unwrap();

        let messages = attn.forward_batch(&states).unwrap();
        assert_eq!(messages.dims(), &[4, SLOT_DIM]);
    }

    #[test]
    fn forward_batch_output_is_finite() {
        let device = Device::Cpu;
        let attn = GpuSlotAttention::new_random(43, &device).unwrap();

        let mut rng = crate::nn::Rng::new(100);
        let batch: Vec<f32> = (0..8 * SLOT_DIM)
            .map(|_| rng.next_f32_range(-1.0, 1.0))
            .collect();
        let states = Tensor::from_vec(batch, (8, SLOT_DIM), &device).unwrap();

        let messages = attn.forward_batch(&states).unwrap();
        let flat = messages.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(flat.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn from_cpu_attention_matches() {
        let cpu_attn = crate::attention::SlotAttention::new_random(43);
        let device = Device::Cpu;
        let gpu_attn = GpuSlotAttention::from_cpu_attention(&cpu_attn, &device).unwrap();

        // Build matching states for both
        let mut rng = crate::nn::Rng::new(200);
        let mut cpu_states = [const { None }; MAX_SLOTS];
        let mut flat_data = Vec::with_capacity(4 * SLOT_DIM);
        let active_indices = [0usize, 1, 2, 3];

        for &idx in &active_indices {
            let mut v = [0.0f32; SLOT_DIM];
            for x in &mut v {
                *x = rng.next_f32_range(-1.0, 1.0);
            }
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            cpu_states[idx] = Some(v);
            flat_data.extend_from_slice(&v);
        }

        let cpu_msgs = cpu_attn.forward(&cpu_states).unwrap();
        let gpu_tensor = Tensor::from_vec(flat_data, (4, SLOT_DIM), &device).unwrap();
        let gpu_msgs = gpu_attn.forward_batch(&gpu_tensor).unwrap();
        let gpu_flat = gpu_msgs.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for (slot_idx, &orig_idx) in active_indices.iter().enumerate() {
            for d in 0..SLOT_DIM {
                let cpu_val = cpu_msgs[orig_idx][d];
                let gpu_val = gpu_flat[slot_idx * SLOT_DIM + d];
                assert!(
                    (cpu_val - gpu_val).abs() < 1e-4,
                    "slot {} dim {} mismatch: cpu={}, gpu={}",
                    orig_idx,
                    d,
                    cpu_val,
                    gpu_val,
                );
            }
        }
    }

    #[test]
    fn single_slot_self_attention() {
        let device = Device::Cpu;
        let attn = GpuSlotAttention::new_random(43, &device).unwrap();

        let mut v = [0.0f32; SLOT_DIM];
        v[0] = 1.0;
        let states = Tensor::from_slice(&v, (1, SLOT_DIM), &device).unwrap();

        let messages = attn.forward_batch(&states).unwrap();
        let flat = messages.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(flat.iter().any(|&x| x != 0.0), "self-attention should produce non-zero");
    }
}
