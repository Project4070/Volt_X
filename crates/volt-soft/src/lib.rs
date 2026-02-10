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
pub mod diffusion;
pub mod rar;
pub mod vfn;

// GPU modules — compiled only with `gpu` feature
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod training;

use volt_core::{TensorFrame, VoltError};

/// Phase 1 stub: copies input frame to output unchanged.
///
/// In later phases, use [`rar::rar_loop`] instead for real inference.
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
pub fn process_stub(frame: &TensorFrame) -> Result<TensorFrame, VoltError> {
    // MILESTONE: 2.4 — Replace with GPU RAR inference loop
    Ok(frame.clone())
}


#[cfg(test)]
mod tests {
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
}
