//! GPU-accelerated Vector Field Network using candle.
//!
//! Same architecture as the CPU [`crate::vfn::Vfn`] (256→512→512→256 MLP
//! with ReLU) but uses candle tensors for batched GPU execution and
//! autograd support during training.

use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
use volt_core::{VoltError, SLOT_DIM};

/// Hidden dimension for the VFN's intermediate layers.
const HIDDEN_DIM: usize = 512;

/// GPU-accelerated Vector Field Network.
///
/// Mirrors the CPU [`crate::vfn::Vfn`] architecture but operates on
/// candle tensors for batched forward passes and autograd.
///
/// # Example
///
/// ```ignore
/// use volt_soft::gpu::vfn::GpuVfn;
/// use candle_core::Device;
///
/// let device = Device::Cpu;
/// let vfn = GpuVfn::new_random(42, &device).unwrap();
/// ```
pub struct GpuVfn {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
    device: Device,
}

impl std::fmt::Debug for GpuVfn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuVfn({}→{}→{}→{}, device={:?})",
            SLOT_DIM, HIDDEN_DIM, HIDDEN_DIM, SLOT_DIM, self.device
        )
    }
}

impl GpuVfn {
    /// Creates a new GPU VFN with Xavier/Glorot random initialization.
    ///
    /// Uses candle's built-in initialization on the specified device.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if tensor creation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use volt_soft::gpu::vfn::GpuVfn;
    /// use candle_core::Device;
    ///
    /// let vfn = GpuVfn::new_random(42, &Device::Cpu).unwrap();
    /// ```
    pub fn new_random(seed: u64, device: &Device) -> Result<Self, VoltError> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);

        // Use splitmix64-style seeded initialization for reproducibility
        let mut rng = crate::nn::Rng::new(seed);

        let l1 = Self::init_layer(&mut rng, SLOT_DIM, HIDDEN_DIM, vb.pp("l1"))?;
        let l2 = Self::init_layer(&mut rng, HIDDEN_DIM, HIDDEN_DIM, vb.pp("l2"))?;
        let l3 = Self::init_layer(&mut rng, HIDDEN_DIM, SLOT_DIM, vb.pp("l3"))?;

        Ok(Self {
            layer1: l1,
            layer2: l2,
            layer3: l3,
            device: device.clone(),
        })
    }

    /// Creates a GPU VFN by loading weights from a CPU VFN.
    ///
    /// Transfers the CPU weight matrices to candle tensors on the target
    /// device. Useful for testing GPU/CPU equivalence.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if weight transfer fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use volt_soft::vfn::Vfn;
    /// use volt_soft::gpu::vfn::GpuVfn;
    /// use candle_core::Device;
    ///
    /// let cpu_vfn = Vfn::new_random(42);
    /// let gpu_vfn = GpuVfn::from_cpu_vfn(&cpu_vfn, &Device::Cpu).unwrap();
    /// ```
    pub fn from_cpu_vfn(
        cpu_vfn: &crate::vfn::Vfn,
        device: &Device,
    ) -> Result<Self, VoltError> {
        let (l1, l2, l3) = cpu_vfn.layers();

        let layer1 = Self::cpu_linear_to_candle(l1, device)?;
        let layer2 = Self::cpu_linear_to_candle(l2, device)?;
        let layer3 = Self::cpu_linear_to_candle(l3, device)?;

        Ok(Self {
            layer1,
            layer2,
            layer3,
            device: device.clone(),
        })
    }

    /// Creates a GPU VFN with trainable parameters backed by a VarMap.
    ///
    /// Used for Flow Matching training — weights are tracked for autograd.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if parameter creation fails.
    pub fn new_trainable(var_map: &VarMap, device: &Device) -> Result<Self, VoltError> {
        let vb = VarBuilder::from_varmap(var_map, DType::F32, device);

        let layer1 = linear(SLOT_DIM, HIDDEN_DIM, vb.pp("vfn.l1")).map_err(|e| {
            VoltError::Internal {
                message: format!("GpuVfn trainable layer1: {e}"),
            }
        })?;
        let layer2 = linear(HIDDEN_DIM, HIDDEN_DIM, vb.pp("vfn.l2")).map_err(|e| {
            VoltError::Internal {
                message: format!("GpuVfn trainable layer2: {e}"),
            }
        })?;
        let layer3 = linear(HIDDEN_DIM, SLOT_DIM, vb.pp("vfn.l3")).map_err(|e| {
            VoltError::Internal {
                message: format!("GpuVfn trainable layer3: {e}"),
            }
        })?;

        Ok(Self {
            layer1,
            layer2,
            layer3,
            device: device.clone(),
        })
    }

    /// Batched forward pass: process multiple slots in parallel.
    ///
    /// Input shape: `[N, SLOT_DIM]` where N is the number of active slots.
    /// Output shape: `[N, SLOT_DIM]` — drift vectors for each slot.
    ///
    /// Architecture: Linear(256→512) + ReLU → Linear(512→512) + ReLU → Linear(512→256).
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if tensor operations fail.
    pub fn forward_batch(&self, input: &Tensor) -> Result<Tensor, VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("GpuVfn forward_batch: {e}"),
        };

        // Layer 1: Linear + ReLU
        let h1 = self.layer1.forward(input).map_err(map_err)?;
        let h1 = h1.relu().map_err(map_err)?;

        // Layer 2: Linear + ReLU
        let h2 = self.layer2.forward(&h1).map_err(map_err)?;
        let h2 = h2.relu().map_err(map_err)?;

        // Layer 3: Linear (no activation)
        self.layer3.forward(&h2).map_err(map_err)
    }

    /// Single-slot forward pass (convenience wrapper).
    ///
    /// Wraps a single `[SLOT_DIM]` input into a `[1, SLOT_DIM]` batch,
    /// runs forward, and extracts the result.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::Internal`] if tensor operations fail.
    pub fn forward_single(
        &self,
        input: &[f32; SLOT_DIM],
    ) -> Result<[f32; SLOT_DIM], VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("GpuVfn forward_single: {e}"),
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

    // -- Internal helpers --

    /// Initializes a candle Linear from our custom Rng with Xavier init.
    fn init_layer(
        rng: &mut crate::nn::Rng,
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
    ) -> Result<Linear, VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("GpuVfn init_layer: {e}"),
        };

        let limit = (6.0 / (in_dim + out_dim) as f32).sqrt();
        let weights: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| rng.next_f32_range(-limit, limit))
            .collect();

        let w = Tensor::from_vec(weights, (out_dim, in_dim), vb.device()).map_err(map_err)?;
        let b = Tensor::zeros(out_dim, DType::F32, vb.device()).map_err(map_err)?;

        Ok(Linear::new(w, Some(b)))
    }

    /// Converts a CPU `nn::Linear` to a candle `Linear` on the given device.
    fn cpu_linear_to_candle(
        cpu_layer: &crate::nn::Linear,
        device: &Device,
    ) -> Result<Linear, VoltError> {
        let map_err = |e: candle_core::Error| VoltError::Internal {
            message: format!("GpuVfn cpu_linear_to_candle: {e}"),
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

    #[test]
    fn new_random_creates_valid_vfn() {
        let device = Device::Cpu;
        let vfn = GpuVfn::new_random(42, &device).unwrap();
        assert!(format!("{:?}", vfn).contains("GpuVfn"));
    }

    #[test]
    fn forward_single_produces_correct_size() {
        let device = Device::Cpu;
        let vfn = GpuVfn::new_random(42, &device).unwrap();
        let input = [0.1f32; SLOT_DIM];
        let output = vfn.forward_single(&input).unwrap();
        assert_eq!(output.len(), SLOT_DIM);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn forward_batch_processes_multiple_slots() {
        let device = Device::Cpu;
        let vfn = GpuVfn::new_random(42, &device).unwrap();

        let batch: Vec<f32> = (0..4 * SLOT_DIM).map(|i| (i as f32) * 0.001).collect();
        let input = Tensor::from_vec(batch, (4, SLOT_DIM), &device).unwrap();

        let output = vfn.forward_batch(&input).unwrap();
        let shape = output.dims();
        assert_eq!(shape, &[4, SLOT_DIM]);
    }

    #[test]
    fn from_cpu_vfn_matches_cpu_output() {
        let cpu_vfn = crate::vfn::Vfn::new_random(42);
        let device = Device::Cpu;
        let gpu_vfn = GpuVfn::from_cpu_vfn(&cpu_vfn, &device).unwrap();

        let input = [0.1f32; SLOT_DIM];
        let cpu_out = cpu_vfn.forward(&input).unwrap();
        let gpu_out = gpu_vfn.forward_single(&input).unwrap();

        // Should match within float32 precision
        for (i, (a, b)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "slot dim {} mismatch: cpu={}, gpu={}",
                i,
                a,
                b,
            );
        }
    }

    #[test]
    fn deterministic_same_seed() {
        let device = Device::Cpu;
        let vfn1 = GpuVfn::new_random(42, &device).unwrap();
        let vfn2 = GpuVfn::new_random(42, &device).unwrap();

        let input = [0.5f32; SLOT_DIM];
        let out1 = vfn1.forward_single(&input).unwrap();
        let out2 = vfn2.forward_single(&input).unwrap();
        assert_eq!(out1, out2);
    }

    #[test]
    fn zero_input_produces_zero_output() {
        let device = Device::Cpu;
        let vfn = GpuVfn::new_random(42, &device).unwrap();
        let input = [0.0f32; SLOT_DIM];
        let output = vfn.forward_single(&input).unwrap();
        for &x in &output {
            assert!(x.abs() < 1e-6, "expected ~zero, got {}", x);
        }
    }

    #[test]
    fn new_trainable_creates_valid_vfn() {
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vfn = GpuVfn::new_trainable(&var_map, &device).unwrap();
        let input = [0.1f32; SLOT_DIM];
        let output = vfn.forward_single(&input).unwrap();
        assert!(output.iter().all(|x| x.is_finite()));
    }
}
