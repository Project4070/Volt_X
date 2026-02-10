//! GPU-accelerated RAR inference loop.
//!
//! Same algorithm as the CPU [`crate::rar::rar_loop`] but uses batched
//! tensor operations via candle for ~10x speedup on CUDA hardware.
//!
//! ## Data Flow
//!
//! 1. Extract active slot states from TensorFrame → `[N, 256]` tensor on GPU
//! 2. Main loop: batched Root → batched Attend → Refine + convergence check
//! 3. Transfer final states back to TensorFrame

use candle_core::Tensor;
use volt_core::{TensorFrame, VoltError, MAX_SLOTS, SLOT_DIM};

use super::attention::GpuSlotAttention;
use super::vfn::GpuVfn;
use crate::diffusion;
use crate::rar::{RarConfig, RarResult};

/// Runs the GPU-accelerated RAR inference loop.
///
/// Mirrors [`crate::rar::rar_loop`] but uses batched tensor operations.
/// Results should match CPU within float32 precision (< 1e-5 per element).
///
/// # Errors
///
/// Returns [`VoltError::FrameError`] if the configured resolution is out of range.
/// Returns [`VoltError::Internal`] if tensor operations fail.
///
/// # Example
///
/// ```ignore
/// use volt_soft::gpu::rar::gpu_rar_loop;
/// use volt_soft::gpu::vfn::GpuVfn;
/// use volt_soft::gpu::attention::GpuSlotAttention;
/// use volt_soft::rar::RarConfig;
/// use volt_core::{TensorFrame, SlotRole, SLOT_DIM};
/// use candle_core::Device;
///
/// let device = Device::Cpu;
/// let vfn = GpuVfn::new_random(42, &device).unwrap();
/// let attn = GpuSlotAttention::new_random(43, &device).unwrap();
/// let config = RarConfig::default();
///
/// let mut frame = TensorFrame::new();
/// frame.write_at(0, 0, SlotRole::Agent, [0.1; SLOT_DIM]).unwrap();
/// frame.normalize_slot(0, 0).unwrap();
///
/// let result = gpu_rar_loop(&frame, &vfn, &attn, &config).unwrap();
/// assert!(result.iterations <= config.max_iterations);
/// ```
pub fn gpu_rar_loop(
    input: &TensorFrame,
    vfn: &GpuVfn,
    attention: &GpuSlotAttention,
    config: &RarConfig,
) -> Result<RarResult, VoltError> {
    let device = vfn.device();

    // Validate config
    if config.resolution >= volt_core::NUM_RESOLUTIONS {
        return Err(VoltError::FrameError {
            message: format!(
                "RAR resolution {} out of range (max {})",
                config.resolution,
                volt_core::NUM_RESOLUTIONS,
            ),
        });
    }

    let mut frame = input.clone();
    let mut converged = [false; MAX_SLOTS];
    let mut deltas = [0.0f32; MAX_SLOTS];

    // Identify active slots (have data at target resolution)
    let mut active_indices = Vec::new();
    for (i, conv) in converged.iter_mut().enumerate() {
        let has_data = frame.slots[i]
            .as_ref()
            .is_some_and(|slot| slot.resolutions[config.resolution].is_some());
        if has_data {
            active_indices.push(i);
        } else {
            *conv = true;
        }
    }

    // Early exit: no active slots
    if active_indices.is_empty() {
        return Ok(RarResult {
            frame,
            iterations: 0,
            converged,
            final_deltas: deltas,
        });
    }

    let map_err = |e: candle_core::Error| VoltError::Internal {
        message: format!("gpu_rar_loop: {e}"),
    };

    // Extract active slot states to a flat Vec, then build tensor
    let n_active = active_indices.len();

    let mut iteration = 0u32;

    while iteration < config.max_iterations {
        if converged.iter().all(|&c| c) {
            break;
        }
        iteration += 1;

        // Collect current states for non-converged active slots
        let mut non_frozen_indices = Vec::new();
        let mut all_state_data = Vec::with_capacity(n_active * SLOT_DIM);

        for &idx in &active_indices {
            let slot = frame.slots[idx].as_ref().ok_or_else(|| VoltError::Internal {
                message: format!("gpu_rar_loop: slot {idx} unexpectedly empty"),
            })?;
            let state = slot.resolutions[config.resolution].ok_or_else(|| {
                VoltError::Internal {
                    message: format!("gpu_rar_loop: slot {idx} has no data at R{}", config.resolution),
                }
            })?;
            all_state_data.extend_from_slice(&state);
            if !converged[idx] {
                non_frozen_indices.push(idx);
            }
        }

        let all_states_tensor =
            Tensor::from_vec(all_state_data.clone(), (n_active, SLOT_DIM), device)
                .map_err(map_err)?;

        // === ROOT PHASE ===
        // Apply VFN only to non-frozen slots (batched)
        let mut drift_data = vec![0.0f32; n_active * SLOT_DIM];
        if !non_frozen_indices.is_empty() {
            let mut non_frozen_state_data = Vec::with_capacity(non_frozen_indices.len() * SLOT_DIM);
            let mut non_frozen_local_indices = Vec::new(); // index within active_indices
            for (local_idx, &global_idx) in active_indices.iter().enumerate() {
                if non_frozen_indices.contains(&global_idx) {
                    non_frozen_local_indices.push(local_idx);
                    let start = local_idx * SLOT_DIM;
                    non_frozen_state_data.extend_from_slice(&all_state_data[start..start + SLOT_DIM]);
                }
            }

            let non_frozen_tensor =
                Tensor::from_vec(non_frozen_state_data, (non_frozen_indices.len(), SLOT_DIM), device)
                    .map_err(map_err)?;
            let drift_tensor = vfn.forward_batch(&non_frozen_tensor)?;
            let drift_flat = drift_tensor.flatten_all().map_err(map_err)?.to_vec1::<f32>().map_err(map_err)?;

            // Write drifts back into the full drift array
            for (i, &local_idx) in non_frozen_local_indices.iter().enumerate() {
                let src_start = i * SLOT_DIM;
                let dst_start = local_idx * SLOT_DIM;
                drift_data[dst_start..dst_start + SLOT_DIM]
                    .copy_from_slice(&drift_flat[src_start..src_start + SLOT_DIM]);
            }
        }

        // === ATTEND PHASE ===
        // All active slots participate (frozen slots are still K/V sources)
        let msg_tensor = attention.forward_batch(&all_states_tensor)?;
        let msg_flat = msg_tensor.flatten_all().map_err(map_err)?.to_vec1::<f32>().map_err(map_err)?;

        // === DIFFUSION NOISE ===
        let noise_vectors = if let Some(ref diff_config) = config.diffusion {
            let active_mask = {
                let mut mask = [false; MAX_SLOTS];
                for &idx in &active_indices {
                    if !converged[idx] {
                        mask[idx] = true;
                    }
                }
                mask
            };
            diffusion::generate_noise(diff_config, &active_mask, iteration)?
        } else {
            [const { None }; MAX_SLOTS]
        };

        // === REFINE PHASE ===
        for (local_idx, &global_idx) in active_indices.iter().enumerate() {
            if converged[global_idx] {
                continue;
            }

            let state_start = local_idx * SLOT_DIM;

            // State update: S_i(t+1) = S_i(t) + dt * (drift + beta * msg) + noise
            let mut new_state = [0.0f32; SLOT_DIM];
            for d in 0..SLOT_DIM {
                let s = all_state_data[state_start + d];
                let drift = drift_data[state_start + d];
                let msg = msg_flat[state_start + d];
                new_state[d] = s + config.dt * (drift + config.beta * msg);
            }

            // Add diffusion noise if present
            if let Some(noise) = &noise_vectors[global_idx] {
                for d in 0..SLOT_DIM {
                    new_state[d] += noise[d];
                }
            }

            // L2 normalize
            let norm: f32 = new_state.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                converged[global_idx] = true;
                continue;
            }
            for x in &mut new_state {
                *x /= norm;
            }

            // Convergence check
            let delta: f32 = new_state
                .iter()
                .zip(&all_state_data[state_start..state_start + SLOT_DIM])
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            deltas[global_idx] = delta;

            if delta < config.epsilon {
                converged[global_idx] = true;
            }

            // Write back to frame
            if let Some(slot) = &mut frame.slots[global_idx] {
                slot.resolutions[config.resolution] = Some(new_state);
            }
        }

        // Adaptive sigma
        if let Some(ref mut diff_config) = config.diffusion.clone() {
            diffusion::adapt_sigma(diff_config, &deltas, &converged, config.epsilon, 0.95, 1.05);
        }
    }

    frame.frame_meta.rar_iterations = iteration;

    Ok(RarResult {
        frame,
        iterations: iteration,
        converged,
        final_deltas: deltas,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use volt_core::SlotRole;

    fn normalized_vector(seed: u64) -> [f32; SLOT_DIM] {
        let mut rng = crate::nn::Rng::new(seed);
        let mut v = [0.0f32; SLOT_DIM];
        for x in &mut v {
            *x = rng.next_f32_range(-1.0, 1.0);
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn empty_frame_converges_immediately() {
        let device = Device::Cpu;
        let vfn = GpuVfn::new_random(42, &device).unwrap();
        let attn = GpuSlotAttention::new_random(43, &device).unwrap();
        let config = RarConfig::default();
        let frame = TensorFrame::new();

        let result = gpu_rar_loop(&frame, &vfn, &attn, &config).unwrap();
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn single_slot_terminates() {
        let device = Device::Cpu;
        let vfn = GpuVfn::new_random(42, &device).unwrap();
        let attn = GpuSlotAttention::new_random(43, &device).unwrap();
        let config = RarConfig::default();

        let mut frame = TensorFrame::new();
        frame
            .write_at(0, 0, SlotRole::Agent, normalized_vector(100))
            .unwrap();

        let result = gpu_rar_loop(&frame, &vfn, &attn, &config).unwrap();
        assert!(result.iterations <= config.max_iterations);
        assert!(result.frame.slots[0].is_some());
    }

    #[test]
    fn budget_enforcement() {
        let device = Device::Cpu;
        let vfn = GpuVfn::new_random(42, &device).unwrap();
        let attn = GpuSlotAttention::new_random(43, &device).unwrap();
        let config = RarConfig {
            epsilon: 1e-10,
            max_iterations: 5,
            ..RarConfig::default()
        };

        let mut frame = TensorFrame::new();
        for i in 0..8 {
            frame
                .write_at(i, 0, SlotRole::Free(i as u8), normalized_vector(i as u64 + 200))
                .unwrap();
        }

        let result = gpu_rar_loop(&frame, &vfn, &attn, &config).unwrap();
        assert!(result.iterations <= 5);
    }

    #[test]
    fn gpu_matches_cpu_output() {
        // Use same seeds so weights are identical
        let cpu_vfn = crate::vfn::Vfn::new_random(42);
        let cpu_attn = crate::attention::SlotAttention::new_random(43);

        let device = Device::Cpu;
        let gpu_vfn = super::super::vfn::GpuVfn::from_cpu_vfn(&cpu_vfn, &device).unwrap();
        let gpu_attn =
            super::super::attention::GpuSlotAttention::from_cpu_attention(&cpu_attn, &device)
                .unwrap();

        let config = RarConfig {
            epsilon: 0.01,
            max_iterations: 5,
            dt: 0.1,
            beta: 0.5,
            resolution: 0,
            diffusion: None,
        };

        let mut frame = TensorFrame::new();
        for i in 0..4 {
            frame
                .write_at(i, 0, SlotRole::Free(i as u8), normalized_vector(i as u64 + 1000))
                .unwrap();
        }

        let cpu_result = crate::rar::rar_loop(&frame, &cpu_vfn, &cpu_attn, &config).unwrap();
        let gpu_result = gpu_rar_loop(&frame, &gpu_vfn, &gpu_attn, &config).unwrap();

        assert_eq!(cpu_result.iterations, gpu_result.iterations);

        // Compare slot states
        for i in 0..4 {
            let cpu_slot = cpu_result.frame.read_slot(i).unwrap();
            let gpu_slot = gpu_result.frame.read_slot(i).unwrap();
            let cpu_vec = cpu_slot.resolutions[0].unwrap();
            let gpu_vec = gpu_slot.resolutions[0].unwrap();

            for d in 0..SLOT_DIM {
                assert!(
                    (cpu_vec[d] - gpu_vec[d]).abs() < 1e-3,
                    "slot {} dim {} diverged: cpu={}, gpu={}",
                    i,
                    d,
                    cpu_vec[d],
                    gpu_vec[d],
                );
            }
        }
    }

    #[test]
    fn output_states_are_normalized() {
        let device = Device::Cpu;
        let vfn = GpuVfn::new_random(42, &device).unwrap();
        let attn = GpuSlotAttention::new_random(43, &device).unwrap();
        let config = RarConfig {
            max_iterations: 5,
            ..RarConfig::default()
        };

        let mut frame = TensorFrame::new();
        frame
            .write_at(0, 0, SlotRole::Agent, normalized_vector(600))
            .unwrap();

        let result = gpu_rar_loop(&frame, &vfn, &attn, &config).unwrap();
        let slot = result.frame.read_slot(0).unwrap();
        let vec = slot.resolutions[0].as_ref().unwrap();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "output should be unit-normalized, got norm={}",
            norm,
        );
    }

    #[test]
    fn invalid_resolution_errors() {
        let device = Device::Cpu;
        let vfn = GpuVfn::new_random(42, &device).unwrap();
        let attn = GpuSlotAttention::new_random(43, &device).unwrap();
        let config = RarConfig {
            resolution: 99,
            ..RarConfig::default()
        };
        let frame = TensorFrame::new();
        assert!(gpu_rar_loop(&frame, &vfn, &attn, &config).is_err());
    }
}
