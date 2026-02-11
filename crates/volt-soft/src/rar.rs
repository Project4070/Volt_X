//! Root-Attend-Refine (RAR) inference loop.
//!
//! The RAR loop is the core inference algorithm of Volt X's Soft Core.
//! Each iteration has three phases:
//!
//! 1. **Root** — Apply VFN to each active, non-frozen slot independently.
//!    Converged slots are skipped (progressive freezing).
//! 2. **Attend** — Compute 16×16 cross-slot attention. All active slots
//!    participate as keys/values, but only non-frozen slots receive messages.
//! 3. **Refine** — Update: `S_i(t+1) = S_i(t) + dt × (drift_i + β·msg_i)`,
//!    then L2-normalize. Check per-slot convergence: `‖ΔS‖ < ε`.
//!
//! The loop terminates when all slots converge OR the iteration budget
//! is exhausted.

use crate::attention::SlotAttention;
use crate::diffusion::{self, DiffusionConfig};
use crate::ghost_attention::{self, GhostAttentionConfig};
use crate::vfn::Vfn;
use volt_core::{TensorFrame, VoltError, MAX_SLOTS, NUM_RESOLUTIONS, SLOT_DIM};

/// Configuration for the RAR inference loop.
///
/// # Example
///
/// ```
/// use volt_soft::rar::RarConfig;
///
/// let config = RarConfig::default();
/// assert_eq!(config.max_iterations, 50);
/// assert_eq!(config.resolution, 0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RarConfig {
    /// Per-slot convergence threshold: slot converges when ‖ΔS‖ < epsilon.
    pub epsilon: f32,

    /// Maximum number of RAR iterations (budget cap).
    pub max_iterations: u32,

    /// Step size for state updates: `S += dt × (drift + β·msg)`.
    pub dt: f32,

    /// Weight for attention messages: `β` in the update rule.
    pub beta: f32,

    /// Which resolution level to operate on (0=R₀ discourse, 1=R₁ proposition, etc.).
    pub resolution: usize,

    /// Optional diffusion noise injection configuration.
    /// `None` disables noise (backward compatible with Milestone 2.3).
    pub diffusion: Option<DiffusionConfig>,
}

impl Default for RarConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.001,
            max_iterations: 50,
            dt: 0.1,
            beta: 0.5,
            resolution: 0,
            diffusion: None,
        }
    }
}

/// Result of a RAR inference loop execution.
///
/// Contains the evolved frame plus diagnostic information about
/// convergence and iteration count.
///
/// # Example
///
/// ```
/// use volt_soft::rar::RarResult;
/// use volt_core::MAX_SLOTS;
///
/// // RarResult is returned by rar_loop
/// ```
#[derive(Debug, Clone)]
pub struct RarResult {
    /// The output TensorFrame with evolved slot embeddings.
    pub frame: TensorFrame,

    /// Number of RAR iterations actually performed.
    pub iterations: u32,

    /// Per-slot convergence status after the loop.
    /// `true` = converged (‖ΔS‖ < ε), `false` = did not converge.
    pub converged: [bool; MAX_SLOTS],

    /// Per-slot final delta (‖S(t) - S(t-1)‖) at the last iteration.
    /// 0.0 for empty/inactive slots.
    pub final_deltas: [f32; MAX_SLOTS],
}

/// Runs the Root-Attend-Refine inference loop on a TensorFrame.
///
/// Takes an input frame and iteratively evolves the slot embeddings at
/// the configured resolution using the VFN (drift) and cross-slot
/// attention (messages). Slots that converge are progressively frozen
/// and skip the Root phase in subsequent iterations.
///
/// # Errors
///
/// Returns [`VoltError::FrameError`] if the configured resolution is out of range.
/// Returns [`VoltError::Internal`] if numerical issues occur during inference.
///
/// # Example
///
/// ```no_run
/// use volt_soft::rar::{rar_loop, RarConfig};
/// use volt_soft::vfn::Vfn;
/// use volt_soft::attention::SlotAttention;
/// use volt_core::{TensorFrame, SlotRole, SLOT_DIM};
///
/// let vfn = Vfn::new_random(42);
/// let attn = SlotAttention::new_random(43);
/// let config = RarConfig::default();
///
/// let mut frame = TensorFrame::new();
/// frame.write_at(0, 0, SlotRole::Agent, [0.1; SLOT_DIM]).unwrap();
/// frame.normalize_slot(0, 0).unwrap();
///
/// let result = rar_loop(&frame, &vfn, &attn, &config).unwrap();
/// assert!(result.iterations <= config.max_iterations);
/// ```
pub fn rar_loop(
    input: &TensorFrame,
    vfn: &Vfn,
    attention: &SlotAttention,
    config: &RarConfig,
) -> Result<RarResult, VoltError> {
    // Validate config
    if config.resolution >= NUM_RESOLUTIONS {
        return Err(VoltError::FrameError {
            message: format!(
                "RAR resolution {} out of range (max {})",
                config.resolution, NUM_RESOLUTIONS
            ),
        });
    }

    let mut frame = input.clone();
    let mut converged = [false; MAX_SLOTS];
    let mut deltas = [0.0f32; MAX_SLOTS];

    // Mark slots as active or trivially converged
    for (i, conv) in converged.iter_mut().enumerate() {
        let has_data = frame
            .slots[i]
            .as_ref()
            .is_some_and(|slot| slot.resolutions[config.resolution].is_some());
        if !has_data {
            *conv = true;
        }
    }

    // Early exit: no active slots
    if converged.iter().all(|&c| c) {
        return Ok(RarResult {
            frame,
            iterations: 0,
            converged,
            final_deltas: deltas,
        });
    }

    let mut iteration = 0;

    while iteration < config.max_iterations {
        // Check if all slots converged
        if converged.iter().all(|&c| c) {
            break;
        }
        iteration += 1;

        // Snapshot current slot states at target resolution
        let mut states: [Option<[f32; SLOT_DIM]>; MAX_SLOTS] = [const { None }; MAX_SLOTS];
        for (i, state) in states.iter_mut().enumerate() {
            if let Some(slot) = &frame.slots[i] {
                *state = slot.resolutions[config.resolution];
            }
        }

        // === ROOT PHASE ===
        // Apply VFN to each active, non-converged slot
        let mut drifts: [Option<[f32; SLOT_DIM]>; MAX_SLOTS] = [const { None }; MAX_SLOTS];
        for i in 0..MAX_SLOTS {
            if converged[i] {
                continue; // Progressive freezing: skip converged slots
            }
            if let Some(state) = &states[i] {
                drifts[i] = Some(vfn.forward(state)?);
            }
        }

        // === ATTEND PHASE ===
        // Cross-slot attention: all active slots participate as K/V,
        // but messages are only used for non-converged slots.
        let messages = attention.forward(&states)?;

        // === DIFFUSION NOISE ===
        // Use Option<Box<...>> to avoid 16KB stack allocation when diffusion is off
        let noise_vectors = if let Some(ref diff_config) = config.diffusion {
            let active_mask = {
                let mut mask = [false; MAX_SLOTS];
                for (i, conv) in converged.iter().enumerate() {
                    if !conv {
                        mask[i] = states[i].is_some();
                    }
                }
                mask
            };
            Some(Box::new(diffusion::generate_noise(
                diff_config,
                &active_mask,
                iteration,
            )?))
        } else {
            None
        };

        // === REFINE PHASE ===
        for i in 0..MAX_SLOTS {
            if converged[i] {
                continue;
            }
            if let (Some(state), Some(drift)) = (&states[i], &drifts[i]) {
                let msg = &messages[i];

                // State update: S_i(t+1) = S_i(t) + dt × (drift + β·msg) + noise
                let mut new_state = [0.0f32; SLOT_DIM];
                for d in 0..SLOT_DIM {
                    new_state[d] = state[d] + config.dt * (drift[d] + config.beta * msg[d]);
                }

                // Add diffusion noise if present
                if let Some(ref noise_arr) = noise_vectors
                    && let Some(noise) = &noise_arr[i]
                {
                    for d in 0..SLOT_DIM {
                        new_state[d] += noise[d];
                    }
                }

                // L2 normalize to unit hypersphere
                let norm: f32 = new_state.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 {
                    // Degenerate state — leave unchanged and mark converged
                    converged[i] = true;
                    continue;
                }
                for x in &mut new_state {
                    *x /= norm;
                }

                // Convergence check: ‖S(t+1) − S(t)‖
                let delta: f32 = new_state
                    .iter()
                    .zip(state.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                deltas[i] = delta;

                if delta < config.epsilon {
                    converged[i] = true;
                }

                // Write new state back to frame
                if let Some(slot) = &mut frame.slots[i] {
                    slot.resolutions[config.resolution] = Some(new_state);
                }
            }
        }
    }

    // Update frame metadata with iteration count
    frame.frame_meta.rar_iterations = iteration;

    Ok(RarResult {
        frame,
        iterations: iteration,
        converged,
        final_deltas: deltas,
    })
}

/// Configuration for ghost frame cross-attention in the RAR loop.
///
/// Provides the ghost gist vectors and the alpha blending weight.
///
/// # Example
///
/// ```
/// use volt_soft::rar::GhostConfig;
/// use volt_core::SLOT_DIM;
///
/// let config = GhostConfig {
///     gists: vec![[0.1; SLOT_DIM]],
///     alpha: 0.1,
/// };
/// assert_eq!(config.gists.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct GhostConfig {
    /// Ghost gist vectors from the bleed buffer.
    /// Each is a 256-dim R₀ gist.
    pub gists: Vec<[f32; SLOT_DIM]>,

    /// Weight for ghost attention blending (0.0–1.0).
    /// Default: 0.1 (subtle memory influence).
    pub alpha: f32,
}

/// Runs the RAR loop with ghost frame cross-attention.
///
/// Identical to [`rar_loop`] except the Attend phase uses
/// [`ghost_attention::forward_with_ghosts`] to include ghost gists
/// as additional Key/Value sources in cross-slot attention.
///
/// Ghost gists are blended with the standard slot attention
/// messages via the `ghost_config.alpha` weight.
///
/// # Errors
///
/// Returns [`VoltError::FrameError`] if the configured resolution is out of range.
/// Returns [`VoltError::Internal`] if numerical issues occur during inference.
///
/// # Example
///
/// ```no_run
/// use volt_soft::rar::{rar_loop_with_ghosts, RarConfig, GhostConfig};
/// use volt_soft::vfn::Vfn;
/// use volt_soft::attention::SlotAttention;
/// use volt_core::{TensorFrame, SlotRole, SLOT_DIM};
///
/// let vfn = Vfn::new_random(42);
/// let attn = SlotAttention::new_random(43);
/// let config = RarConfig::default();
/// let ghost_config = GhostConfig { gists: vec![], alpha: 0.1 };
///
/// let mut frame = TensorFrame::new();
/// frame.write_at(0, 0, SlotRole::Agent, [0.1; SLOT_DIM]).unwrap();
/// frame.normalize_slot(0, 0).unwrap();
///
/// let result = rar_loop_with_ghosts(&frame, &vfn, &attn, &config, &ghost_config).unwrap();
/// assert!(result.iterations <= config.max_iterations);
/// ```
pub fn rar_loop_with_ghosts(
    input: &TensorFrame,
    vfn: &Vfn,
    attention: &SlotAttention,
    config: &RarConfig,
    ghost_config: &GhostConfig,
) -> Result<RarResult, VoltError> {
    // Validate config
    if config.resolution >= NUM_RESOLUTIONS {
        return Err(VoltError::FrameError {
            message: format!(
                "RAR resolution {} out of range (max {})",
                config.resolution, NUM_RESOLUTIONS
            ),
        });
    }

    let ghost_attn_config = GhostAttentionConfig {
        alpha: ghost_config.alpha,
    };

    let mut frame = input.clone();
    let mut converged = [false; MAX_SLOTS];
    let mut deltas = [0.0f32; MAX_SLOTS];

    // Mark slots as active or trivially converged
    for (i, conv) in converged.iter_mut().enumerate() {
        let has_data = frame.slots[i]
            .as_ref()
            .is_some_and(|slot| slot.resolutions[config.resolution].is_some());
        if !has_data {
            *conv = true;
        }
    }

    // Early exit: no active slots
    if converged.iter().all(|&c| c) {
        return Ok(RarResult {
            frame,
            iterations: 0,
            converged,
            final_deltas: deltas,
        });
    }

    let mut iteration = 0;

    while iteration < config.max_iterations {
        // Check if all slots converged
        if converged.iter().all(|&c| c) {
            break;
        }
        iteration += 1;

        // Snapshot current slot states at target resolution
        let mut states: [Option<[f32; SLOT_DIM]>; MAX_SLOTS] = [const { None }; MAX_SLOTS];
        for (i, state) in states.iter_mut().enumerate() {
            if let Some(slot) = &frame.slots[i] {
                *state = slot.resolutions[config.resolution];
            }
        }

        // === ROOT PHASE ===
        let mut drifts: [Option<[f32; SLOT_DIM]>; MAX_SLOTS] = [const { None }; MAX_SLOTS];
        for i in 0..MAX_SLOTS {
            if converged[i] {
                continue;
            }
            if let Some(state) = &states[i] {
                drifts[i] = Some(vfn.forward(state)?);
            }
        }

        // === ATTEND PHASE (with ghost frames) ===
        let messages =
            ghost_attention::forward_with_ghosts(attention, &states, &ghost_config.gists, &ghost_attn_config)?;

        // === DIFFUSION NOISE ===
        let noise_vectors = if let Some(ref diff_config) = config.diffusion {
            let active_mask = {
                let mut mask = [false; MAX_SLOTS];
                for (i, conv) in converged.iter().enumerate() {
                    if !conv {
                        mask[i] = states[i].is_some();
                    }
                }
                mask
            };
            Some(Box::new(diffusion::generate_noise(
                diff_config,
                &active_mask,
                iteration,
            )?))
        } else {
            None
        };

        // === REFINE PHASE ===
        for i in 0..MAX_SLOTS {
            if converged[i] {
                continue;
            }
            if let (Some(state), Some(drift)) = (&states[i], &drifts[i]) {
                let msg = &messages[i];

                let mut new_state = [0.0f32; SLOT_DIM];
                for d in 0..SLOT_DIM {
                    new_state[d] = state[d] + config.dt * (drift[d] + config.beta * msg[d]);
                }

                if let Some(ref noise_arr) = noise_vectors
                    && let Some(noise) = &noise_arr[i]
                {
                    for d in 0..SLOT_DIM {
                        new_state[d] += noise[d];
                    }
                }

                let norm: f32 = new_state.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 {
                    converged[i] = true;
                    continue;
                }
                for x in &mut new_state {
                    *x /= norm;
                }

                let delta: f32 = new_state
                    .iter()
                    .zip(state.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                deltas[i] = delta;

                if delta < config.epsilon {
                    converged[i] = true;
                }

                if let Some(slot) = &mut frame.slots[i] {
                    slot.resolutions[config.resolution] = Some(new_state);
                }
            }
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
    use volt_core::SlotRole;

    fn make_vfn() -> Vfn {
        Vfn::new_random(42)
    }

    fn make_attention() -> SlotAttention {
        SlotAttention::new_random(43)
    }

    fn normalized_vector(seed: u64) -> [f32; SLOT_DIM] {
        use crate::nn::Rng;
        let mut rng = Rng::new(seed);
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
        let vfn = make_vfn();
        let attn = make_attention();
        let config = RarConfig::default();
        let frame = TensorFrame::new();

        let result = rar_loop(&frame, &vfn, &attn, &config).unwrap();
        assert_eq!(result.iterations, 0);
        assert!(result.converged.iter().all(|&c| c));
    }

    #[test]
    fn single_slot_runs_and_terminates() {
        let vfn = make_vfn();
        let attn = make_attention();
        let config = RarConfig::default();

        let mut frame = TensorFrame::new();
        frame
            .write_at(0, 0, SlotRole::Agent, normalized_vector(100))
            .unwrap();

        let result = rar_loop(&frame, &vfn, &attn, &config).unwrap();
        assert!(result.iterations <= config.max_iterations);
        // The frame should still have slot 0 active
        assert!(result.frame.slots[0].is_some());
    }

    #[test]
    fn budget_enforcement() {
        let vfn = make_vfn();
        let attn = make_attention();
        let config = RarConfig {
            epsilon: 1e-10, // Very tight — unlikely to converge
            max_iterations: 5,
            ..RarConfig::default()
        };

        let mut frame = TensorFrame::new();
        for i in 0..8 {
            frame
                .write_at(i, 0, SlotRole::Free(i as u8), normalized_vector(i as u64 + 200))
                .unwrap();
        }

        let result = rar_loop(&frame, &vfn, &attn, &config).unwrap();
        assert!(
            result.iterations <= 5,
            "should respect budget cap, got {} iterations",
            result.iterations
        );
    }

    #[test]
    fn invalid_resolution_errors() {
        let vfn = make_vfn();
        let attn = make_attention();
        let config = RarConfig {
            resolution: 99,
            ..RarConfig::default()
        };
        let frame = TensorFrame::new();

        assert!(rar_loop(&frame, &vfn, &attn, &config).is_err());
    }

    #[test]
    fn converged_slots_freeze() {
        let vfn = make_vfn();
        let attn = make_attention();
        // Use large epsilon so slot converges after 1 iteration
        let config = RarConfig {
            epsilon: 100.0, // Everything converges immediately
            max_iterations: 10,
            ..RarConfig::default()
        };

        let initial_vec = normalized_vector(300);
        let mut frame = TensorFrame::new();
        frame
            .write_at(0, 0, SlotRole::Agent, initial_vec)
            .unwrap();

        let result = rar_loop(&frame, &vfn, &attn, &config).unwrap();

        // Should converge in 1 iteration with such large epsilon
        assert_eq!(result.iterations, 1, "should converge after 1 iteration");
        assert!(result.converged[0]);
    }

    #[test]
    fn frame_metadata_tracks_iterations() {
        let vfn = make_vfn();
        let attn = make_attention();
        let config = RarConfig {
            max_iterations: 3,
            epsilon: 1e-10,
            ..RarConfig::default()
        };

        let mut frame = TensorFrame::new();
        frame
            .write_at(0, 0, SlotRole::Agent, normalized_vector(400))
            .unwrap();

        let result = rar_loop(&frame, &vfn, &attn, &config).unwrap();
        assert_eq!(result.frame.frame_meta.rar_iterations, result.iterations);
    }

    #[test]
    fn default_config_sensible() {
        let config = RarConfig::default();
        assert!(config.epsilon > 0.0);
        assert!(config.max_iterations > 0);
        assert!(config.dt > 0.0);
        assert!(config.beta >= 0.0);
        assert!(config.resolution < NUM_RESOLUTIONS);
    }

    #[test]
    fn slots_without_target_resolution_are_skipped() {
        let vfn = make_vfn();
        let attn = make_attention();
        let config = RarConfig {
            resolution: 1, // R1
            ..RarConfig::default()
        };

        // Write data at R0 only — R1 is empty
        let mut frame = TensorFrame::new();
        frame
            .write_at(0, 0, SlotRole::Agent, normalized_vector(500))
            .unwrap();

        let result = rar_loop(&frame, &vfn, &attn, &config).unwrap();
        // No active slots at R1, so should converge immediately
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn output_states_are_normalized() {
        let vfn = make_vfn();
        let attn = make_attention();
        let config = RarConfig {
            max_iterations: 5,
            ..RarConfig::default()
        };

        let mut frame = TensorFrame::new();
        frame
            .write_at(0, 0, SlotRole::Agent, normalized_vector(600))
            .unwrap();

        let result = rar_loop(&frame, &vfn, &attn, &config).unwrap();
        let slot = result.frame.read_slot(0).unwrap();
        let vec = slot.resolutions[0].as_ref().unwrap();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "output should be unit-normalized, got norm={}",
            norm
        );
    }

    // --- Ghost-aware RAR loop tests ---

    #[test]
    fn ghost_rar_empty_ghosts_matches_rar() {
        let vfn = make_vfn();
        let attn = make_attention();
        let config = RarConfig {
            max_iterations: 3,
            ..RarConfig::default()
        };
        let ghost_config = GhostConfig {
            gists: vec![],
            alpha: 0.1,
        };

        let mut frame = TensorFrame::new();
        frame
            .write_at(0, 0, SlotRole::Agent, normalized_vector(700))
            .unwrap();

        let normal = rar_loop(&frame, &vfn, &attn, &config).unwrap();
        let with_ghosts =
            rar_loop_with_ghosts(&frame, &vfn, &attn, &config, &ghost_config).unwrap();

        assert_eq!(normal.iterations, with_ghosts.iterations);
        assert_eq!(normal.converged, with_ghosts.converged);
    }

    #[test]
    fn ghost_rar_runs_and_terminates() {
        let vfn = make_vfn();
        let attn = make_attention();
        let config = RarConfig {
            max_iterations: 5,
            ..RarConfig::default()
        };

        // Create some ghost gists
        let mut ghost_gist = [0.0f32; SLOT_DIM];
        ghost_gist[0] = 1.0;
        let ghost_config = GhostConfig {
            gists: vec![ghost_gist],
            alpha: 0.2,
        };

        let mut frame = TensorFrame::new();
        frame
            .write_at(0, 0, SlotRole::Agent, normalized_vector(800))
            .unwrap();

        let result =
            rar_loop_with_ghosts(&frame, &vfn, &attn, &config, &ghost_config).unwrap();
        assert!(result.iterations <= config.max_iterations);
        assert!(result.frame.slots[0].is_some());
    }

    #[test]
    fn ghost_rar_output_normalized() {
        let vfn = make_vfn();
        let attn = make_attention();
        let config = RarConfig {
            max_iterations: 3,
            ..RarConfig::default()
        };

        let mut ghost_gist = [0.0f32; SLOT_DIM];
        ghost_gist[42] = 1.0;
        let ghost_config = GhostConfig {
            gists: vec![ghost_gist],
            alpha: 0.3,
        };

        let mut frame = TensorFrame::new();
        frame
            .write_at(0, 0, SlotRole::Agent, normalized_vector(900))
            .unwrap();

        let result =
            rar_loop_with_ghosts(&frame, &vfn, &attn, &config, &ghost_config).unwrap();
        let slot = result.frame.read_slot(0).unwrap();
        let vec = slot.resolutions[0].as_ref().unwrap();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "ghost RAR output should be unit-normalized, got norm={}",
            norm
        );
    }
}
