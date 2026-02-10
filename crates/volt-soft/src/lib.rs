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
//! - Vector Field Network (VFN): 500M–2B params, operates on frame slots
//! - Diffusion controller: Per-slot noise for creative exploration
//! - ODE Solver: RK4/DOPRI5 adaptive stepping
//! - Manifold Projector: Keeps slot vectors on semantic manifold
//! - Bleed Buffer: Ghost R₀ gists from long-term memory
//! - Retention Cache: Warm-start from recently-converged attractors
//!
//! ## Architecture Rules
//!
//! - All GPU code lives here — no GPU code in other crates.
//! - Feature-gated: `cargo test --features gpu` for GPU tests.
//! - Depends on `volt-core` and `volt-bus`.

pub use volt_core;

use volt_core::{TensorFrame, VoltError};

/// Phase 1 stub: copies input frame to output unchanged.
///
/// In later phases, this will become the full RAR inference loop
/// (Root-Attend-Refine) with GPU acceleration.
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
    // MILESTONE: 3.1 — Replace with RAR inference loop
    Ok(frame.clone())
}

// MILESTONE: 3.1 — RAR inference loop (CPU simulation first)
// TODO: Implement Root phase (slot-local forward pass)
// TODO: Implement Attend phase (cross-slot attention)
// TODO: Implement Refine phase (state update + convergence)
// TODO: Implement per-slot convergence detection
// TODO: Implement adaptive computation budget

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
