//! # volt-hard
//!
//! The CPU Hard Core — the "left brain" of Volt X.
//!
//! Implements deterministic, exact computation in pure Rust:
//! - **Intent Router**: Routes frame slots to Hard Strands by vector similarity
//! - **Hard Strands**: Pluggable Rust modules (MathEngine, CodeRunner, APIDispatch)
//! - **HDC Algebra**: FFT-based bind/unbind for verification
//! - **Certainty Engine**: Per-slot γ computation using min-rule propagation
//! - **Proof Constructor**: Traceable reasoning chains
//! - **Causal Simulation**: do(X=x) counterfactual reasoning
//!
//! ## Architecture Rules
//!
//! - Pure CPU, no GPU code.
//! - No network code (network goes in `volt-ledger` or `volt-server`).
//! - Depends on `volt-core` and `volt-bus`.
//! - Hard Strands are hot-pluggable via `impl HardStrand` trait.

pub use volt_core;

use volt_core::{TensorFrame, VoltError};

/// Phase 1 stub: passes frame through unchanged.
///
/// In later phases, this will route slots to Hard Strands and
/// compute per-slot certainty via the Certainty Engine.
///
/// # Example
///
/// ```
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
/// use volt_hard::verify_stub;
///
/// let mut frame = TensorFrame::new();
/// frame.write_at(0, 0, SlotRole::Agent, [1.0; SLOT_DIM]).unwrap();
///
/// let result = verify_stub(&frame).unwrap();
/// assert_eq!(result.active_slot_count(), 1);
/// ```
pub fn verify_stub(frame: &TensorFrame) -> Result<TensorFrame, VoltError> {
    // MILESTONE: 3.2 — Replace with Intent Router + Certainty Engine
    Ok(frame.clone())
}

// MILESTONE: 3.2 — Hard Core basics
// TODO: Define HardStrand trait
// TODO: Implement Intent Router (cosine similarity routing)
// TODO: Implement MathEngine Hard Strand
// TODO: Implement Certainty Engine (min-rule γ propagation)
// TODO: Implement Proof Chain Constructor

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::{SlotData, SlotRole, SLOT_DIM};

    #[test]
    fn verify_stub_returns_clone_of_input() {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Predicate);
        slot.write_resolution(0, [0.7; SLOT_DIM]);
        frame.write_slot(1, slot).unwrap();
        frame.meta[1].certainty = 0.85;

        let result = verify_stub(&frame).unwrap();

        assert_eq!(result.active_slot_count(), frame.active_slot_count());
        assert_eq!(result.meta[1].certainty, frame.meta[1].certainty);

        let orig = frame.read_slot(1).unwrap();
        let copy = result.read_slot(1).unwrap();
        assert_eq!(orig.role, copy.role);
        assert_eq!(orig.resolutions[0], copy.resolutions[0]);
    }

    #[test]
    fn verify_stub_empty_frame() {
        let frame = TensorFrame::new();
        let result = verify_stub(&frame).unwrap();
        assert_eq!(result.active_slot_count(), 0);
    }
}
