//! # volt-hard
//!
//! The CPU Hard Core — the "left brain" of Volt X.
//!
//! Implements deterministic, exact computation in pure Rust:
//! - **[`strand::HardStrand`]**: Pluggable trait for CPU-side tools
//! - **[`router::IntentRouter`]**: Routes frame slots to Hard Strands by cosine similarity
//! - **[`math_engine::MathEngine`]**: Exact arithmetic, algebra, basic calculus
//!
//! ## Architecture Rules
//!
//! - Pure CPU, no GPU code.
//! - No network code (network goes in `volt-ledger` or `volt-server`).
//! - Depends on `volt-core` and `volt-bus`.
//! - Hard Strands are hot-pluggable via `impl HardStrand` trait.
//!
//! ## Milestone 3.1: Intent Router + MathEngine
//!
//! The Intent Router receives a TensorFrame from the Soft Core, computes
//! cosine similarity against registered Hard Strand capability vectors,
//! and routes to the best match. The MathEngine handles exact arithmetic.
//!
//! ## Usage
//!
//! ```
//! use volt_hard::router::IntentRouter;
//! use volt_hard::math_engine::MathEngine;
//! use volt_hard::strand::HardStrand;
//! use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
//!
//! // TensorFrame is large (~65KB), so spawn a thread with adequate stack.
//! std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
//!     let mut router = IntentRouter::new();
//!     let engine = MathEngine::new();
//!     let math_cap = *engine.capability_vector();
//!     router.register(Box::new(engine));
//!
//!     let mut frame = TensorFrame::new();
//!     let mut pred = SlotData::new(SlotRole::Predicate);
//!     pred.write_resolution(0, math_cap);
//!     frame.write_slot(1, pred).unwrap();
//!     frame.meta[1].certainty = 0.8;
//!
//!     let mut inst = SlotData::new(SlotRole::Instrument);
//!     let mut data = [0.0_f32; SLOT_DIM];
//!     data[0] = 3.0; // MUL
//!     data[1] = 6.0;
//!     data[2] = 7.0;
//!     inst.write_resolution(0, data);
//!     frame.write_slot(6, inst).unwrap();
//!     frame.meta[6].certainty = 0.9;
//!
//!     let result = router.route(&frame).unwrap();
//!     let r = result.frame.read_slot(8).unwrap();
//!     assert!((r.resolutions[0].unwrap()[0] - 42.0).abs() < 0.01);
//! }).unwrap().join().unwrap();
//! ```

pub use volt_core;

pub mod math_engine;
pub mod router;
pub mod strand;

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

/// Create a default Hard Core pipeline with the MathEngine registered.
///
/// Convenience function that returns an [`IntentRouter`] pre-configured
/// with the standard set of Hard Strands for Milestone 3.1.
///
/// # Example
///
/// ```
/// use volt_hard::default_router;
///
/// let router = default_router();
/// assert_eq!(router.strand_count(), 1);
/// ```
pub fn default_router() -> router::IntentRouter {
    let mut router = router::IntentRouter::new();
    router.register(Box::new(math_engine::MathEngine::new()));
    router
}

// MILESTONE: 3.2 — More Hard Strands
// TODO: Implement CodeRunner Hard Strand (wasmtime sandbox)
// TODO: Implement HDCAlgebra Hard Strand (bind/unbind as callable strand)
// TODO: Implement CertaintyEngine (min-rule γ propagation across frame)
// TODO: Implement ProofConstructor (full proof chain recording)

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

    #[test]
    fn default_router_has_math_engine() {
        let router = default_router();
        assert_eq!(router.strand_count(), 1);
    }
}
