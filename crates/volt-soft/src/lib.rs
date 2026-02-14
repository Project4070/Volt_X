//! # volt-soft
//!
//! The GPU Soft Core — the "right brain" of Volt X.
//!
//! Implements the Root-Attend-Refine (RAR) inference loop:
//! - **Root**: Slot-local VFN forward passes (parallel per-slot)
//! - **Attend**: Cross-slot attention O(S²) + ghost frame attention
//! - **Refine**: State update, manifold projection, convergence check
//!
//! ## Key Components
//!
//! - [`vfn::Vfn`]: Vector Field Network — slot-local MLP (256→512→512→256)
//! - [`attention::SlotAttention`]: Cross-slot attention (Q/K/V + softmax)
//! - [`rar::rar_loop`]: The RAR inference loop orchestrator
//! - [`rar::RarConfig`] — Configuration (epsilon, dt, beta, budget)
//! - [`rar::RarResult`] — Output frame + convergence diagnostics
//!
//! ## Architecture Rules
//!
//! - All GPU code lives here — no GPU code in other crates.
//! - Feature-gated: `cargo test --features gpu` for GPU tests.
//! - Depends on `volt-core` and `volt-bus`.
//!
//! ## Milestones
//!
//! - **2.3** (complete): CPU-only RAR with randomly initialized weights
//! - **2.4** (current): GPU port via candle, diffusion noise, Flow Matching training
//!
//! ## GPU Support
//!
//! GPU-accelerated RAR is available behind the `gpu` feature:
//! ```bash
//! cargo test -p volt-soft --features gpu
//! ```

pub use volt_core;

// Internal shared primitives
mod nn;

// Public modules — always compiled (CPU path)
pub mod attention;
pub mod code_attention;
pub mod diffusion;
pub mod ghost_attention;
pub mod rar;
pub mod vfn;

// GPU modules — compiled only with `gpu` feature
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod scaled_vfn;
#[cfg(feature = "gpu")]
pub mod training;

use volt_core::{TensorFrame, VoltError, SLOT_DIM};

use crate::attention::SlotAttention;
use crate::rar::{rar_loop, rar_loop_with_ghosts, GhostConfig, RarConfig, RarResult};
use crate::vfn::Vfn;

/// Phase 1 stub: copies input frame to output unchanged.
///
/// Deprecated: use [`process_rar`] instead for real inference.
///
/// # Example
///
/// ```
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
/// use volt_soft::process_stub;
///
/// let mut frame = TensorFrame::new();
/// frame.write_at(0, 0, SlotRole::Agent, [1.0; SLOT_DIM]).unwrap();
///
/// let result = process_stub(&frame).unwrap();
/// assert_eq!(result.active_slot_count(), 1);
/// ```
#[deprecated(note = "use process_rar() for real RAR inference")]
pub fn process_stub(frame: &TensorFrame) -> Result<TensorFrame, VoltError> {
    Ok(frame.clone())
}

/// Process a frame through the RAR inference loop with default parameters.
///
/// Creates randomly-initialized VFN and SlotAttention models with fixed
/// seeds (42, 43) and runs the RAR loop with default configuration.
/// Returns the full [`RarResult`] including iteration count and
/// convergence diagnostics.
///
/// Spawns a thread with adequate stack (4 MB) because TensorFrame
/// copies in the RAR loop require more stack than the default.
///
/// # Example
///
/// ```
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
/// use volt_soft::process_rar;
///
/// let mut frame = TensorFrame::new();
/// frame.write_at(0, 0, SlotRole::Agent, [0.1; SLOT_DIM]).unwrap();
/// frame.normalize_slot(0, 0).unwrap();
///
/// let result = process_rar(&frame).unwrap();
/// assert!(result.iterations >= 1);
/// ```
pub fn process_rar(frame: &TensorFrame) -> Result<RarResult, VoltError> {
    let frame = Box::new(frame.clone());
    std::thread::Builder::new()
        .stack_size(4 * 1024 * 1024)
        .spawn(move || {
            let vfn = Vfn::new_random(42);
            let attention = SlotAttention::new_random(43);
            let config = RarConfig::default();
            rar_loop(&frame, &vfn, &attention, &config)
        })
        .map_err(|e| VoltError::FrameError {
            message: format!("failed to spawn RAR thread: {e}"),
        })?
        .join()
        .map_err(|_| VoltError::FrameError {
            message: "RAR thread panicked".to_string(),
        })?
}

/// Process a frame through the RAR loop with ghost frame cross-attention.
///
/// Like [`process_rar`] but includes ghost gist vectors from the Bleed
/// Buffer as additional Key/Value sources in the Attend phase. Ghost
/// frames provide subtle memory influence controlled by `alpha`.
///
/// Spawns a thread with adequate stack (4 MB) because TensorFrame
/// copies in the RAR loop require more stack than the default.
///
/// # Example
///
/// ```
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
/// use volt_soft::process_rar_with_ghosts;
///
/// let mut frame = TensorFrame::new();
/// let mut slot = SlotData::new(SlotRole::Agent);
/// slot.write_resolution(0, [0.1; SLOT_DIM]);
/// frame.write_slot(0, slot).unwrap();
/// frame.normalize_slot(0, 0).unwrap();
///
/// let result = process_rar_with_ghosts(&frame, &[], 0.1).unwrap();
/// assert!(result.iterations >= 1);
/// ```
pub fn process_rar_with_ghosts(
    frame: &TensorFrame,
    ghost_gists: &[[f32; SLOT_DIM]],
    alpha: f32,
) -> Result<RarResult, VoltError> {
    let frame = Box::new(frame.clone());
    let gists = ghost_gists.to_vec();
    std::thread::Builder::new()
        .stack_size(4 * 1024 * 1024)
        .spawn(move || {
            let vfn = Vfn::new_random(42);
            let attention = SlotAttention::new_random(43);
            let config = RarConfig::default();
            let ghost_config = GhostConfig { gists, alpha };
            rar_loop_with_ghosts(&frame, &vfn, &attention, &config, &ghost_config)
        })
        .map_err(|e| VoltError::FrameError {
            message: format!("failed to spawn RAR thread: {e}"),
        })?
        .join()
        .map_err(|_| VoltError::FrameError {
            message: "RAR thread panicked".to_string(),
        })?
}

#[cfg(test)]
mod tests {
    #[allow(deprecated)]
    use super::*;
    use volt_core::{SlotData, SlotRole, SLOT_DIM};

    #[test]
    fn process_stub_returns_clone_of_input() {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.42; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();
        frame.meta[0].certainty = 0.9;

        let result = process_stub(&frame).unwrap();

        assert_eq!(result.active_slot_count(), frame.active_slot_count());
        assert_eq!(result.meta[0].certainty, frame.meta[0].certainty);

        let orig = frame.read_slot(0).unwrap();
        let copy = result.read_slot(0).unwrap();
        assert_eq!(orig.role, copy.role);
        assert_eq!(orig.resolutions[0], copy.resolutions[0]);
    }

    #[test]
    fn process_stub_empty_frame() {
        let frame = TensorFrame::new();
        let result = process_stub(&frame).unwrap();
        assert_eq!(result.active_slot_count(), 0);
    }

    #[test]
    fn process_rar_runs_inference() {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.1; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();
        frame.meta[0].certainty = 0.9;
        frame.normalize_slot(0, 0).unwrap();

        let result = process_rar(&frame).unwrap();
        assert!(result.iterations >= 1);
        assert_eq!(result.frame.active_slot_count(), 1);
    }

    #[test]
    fn process_rar_empty_frame() {
        let frame = TensorFrame::new();
        let result = process_rar(&frame).unwrap();
        assert_eq!(result.frame.active_slot_count(), 0);
    }

    #[test]
    fn process_rar_with_ghosts_runs_inference() {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.1; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();
        frame.meta[0].certainty = 0.9;
        frame.normalize_slot(0, 0).unwrap();

        let mut ghost = [0.0f32; SLOT_DIM];
        ghost[0] = 1.0;

        let result = process_rar_with_ghosts(&frame, &[ghost], 0.1).unwrap();
        assert!(result.iterations >= 1);
        assert_eq!(result.frame.active_slot_count(), 1);
    }

    #[test]
    fn process_rar_with_ghosts_empty_ghosts() {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.1; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();
        frame.normalize_slot(0, 0).unwrap();

        let result = process_rar_with_ghosts(&frame, &[], 0.1).unwrap();
        assert!(result.iterations >= 1);
    }
}
