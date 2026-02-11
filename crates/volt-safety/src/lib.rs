//! # volt-safety
//!
//! The deterministic safety layer for Volt X.
//!
//! Safety is enforced on the CPU, not approximated by neural networks.
//! This means safety guarantees are provable, not probabilistic.
//!
//! ## Key Components
//!
//! - **[`axiom`]**: K1-K5 immutable axioms as constant vectors in HDC space
//! - **[`monitor`]**: Transition Monitor checking frames against axioms
//! - **[`scorer`]**: Violation Scorer computing aggregate safety scores
//! - **[`veto`]**: Omega Veto — hardware-level halt, cannot be overridden
//! - **[`layer`]**: Safety Layer wrapping the entire Soft Core → Hard Core pipeline
//!
//! ## Architecture Rules
//!
//! - Pure CPU, deterministic logic — no neural approximation.
//! - Axioms are code, not learned weights.
//! - The Omega Veto cannot be overridden by any other component.
//! - Depends on `volt-core`, `volt-bus`, `volt-hard`.
//!
//! ## Safety Axioms (K1-K5)
//!
//! | ID | Name       | Protects Against                         | Severity |
//! |----|------------|------------------------------------------|----------|
//! | K1 | Harm       | Direct physical harm instructions        | Halt     |
//! | K2 | Deception  | Impersonation or false identity claims   | Halt     |
//! | K3 | Privacy    | Personal data extraction or exposure     | Halt     |
//! | K4 | Autonomy   | Suppression of user agency               | Warning  |
//! | K5 | Integrity  | Corruption of system state or memory     | Halt     |
//!
//! ## Usage
//!
//! ```
//! use volt_safety::layer::SafetyLayer;
//! use volt_hard::default_pipeline;
//! use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
//!
//! std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
//!     let pipeline = default_pipeline();
//!     let mut layer = SafetyLayer::new(pipeline);
//!
//!     let mut frame = TensorFrame::new();
//!     let mut slot = SlotData::new(SlotRole::Agent);
//!     slot.write_resolution(0, [0.1; SLOT_DIM]);
//!     frame.write_slot(0, slot).unwrap();
//!     frame.meta[0].certainty = 0.8;
//!
//!     let result = layer.process(&frame).unwrap();
//!     assert!(!result.vetoed);
//! }).unwrap().join().unwrap();
//! ```

pub use volt_core;

pub mod axiom;
pub mod layer;
pub mod monitor;
pub mod scorer;
pub mod veto;

pub use layer::SafetyResult;

use volt_core::{TensorFrame, VoltError};

/// Process a frame through the full safety-wrapped pipeline.
///
/// Convenience function that creates a default [`SafetyLayer`](layer::SafetyLayer)
/// with the default pipeline and default axioms (K1-K5), processes the
/// frame, and returns the resulting frame.
///
/// Spawns a thread with adequate stack (4 MB) because the pipeline +
/// TensorFrame copies require more stack than the default.
///
/// Returns `Err(VoltError::SafetyViolation)` if the frame triggers an
/// Omega Veto.
///
/// # Example
///
/// ```
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
/// use volt_safety::safe_process;
///
/// let mut frame = TensorFrame::new();
/// let mut slot = SlotData::new(SlotRole::Agent);
/// slot.write_resolution(0, [0.1; SLOT_DIM]);
/// frame.write_slot(0, slot).unwrap();
///
/// let result = safe_process(&frame).unwrap();
/// assert!(!result.is_empty() || result.active_slot_count() == 0);
/// ```
pub fn safe_process(frame: &TensorFrame) -> Result<TensorFrame, VoltError> {
    let frame = Box::new(frame.clone());
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let pipeline = volt_hard::default_pipeline();
            let mut safety_layer = layer::SafetyLayer::new(pipeline);
            let result = safety_layer.process(&frame)?;
            if result.vetoed {
                return Err(VoltError::SafetyViolation {
                    message: "omega veto triggered: frame violated safety axioms".to_string(),
                });
            }
            Ok(result.frame)
        })
        .map_err(|e| VoltError::FrameError {
            message: format!("failed to spawn safety pipeline thread: {e}"),
        })?
        .join()
        .map_err(|_| VoltError::FrameError {
            message: "safety pipeline thread panicked".to_string(),
        })?
}

/// Process a frame through the safety-wrapped pipeline, returning full results.
///
/// Like [`safe_process`], but returns the complete [`SafetyResult`] including
/// the proof chain, safety scores, and veto details instead of just the frame.
///
/// Returns `Err(VoltError::SafetyViolation)` if the Omega Veto fires. The
/// caller can inspect the full `SafetyResult` for non-vetoed frames to
/// extract proof chains and safety metadata.
///
/// # Example
///
/// ```
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
/// use volt_safety::safe_process_full;
///
/// // TensorFrame is ~65KB; use a bigger stack for the doctest.
/// std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
///     let mut frame = TensorFrame::new();
///     let mut slot = SlotData::new(SlotRole::Agent);
///     slot.write_resolution(0, [0.1; SLOT_DIM]);
///     frame.write_slot(0, slot).unwrap();
///
///     let result = safe_process_full(&frame).unwrap();
///     assert!(!result.vetoed);
/// }).unwrap().join().unwrap();
/// ```
pub fn safe_process_full(frame: &TensorFrame) -> Result<SafetyResult, VoltError> {
    let frame = Box::new(frame.clone());
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let pipeline = volt_hard::default_pipeline();
            let mut safety_layer = layer::SafetyLayer::new(pipeline);
            let result = safety_layer.process(&frame)?;
            if result.vetoed {
                return Err(VoltError::SafetyViolation {
                    message: "omega veto triggered: frame violated safety axioms".to_string(),
                });
            }
            Ok(result)
        })
        .map_err(|e| VoltError::FrameError {
            message: format!("failed to spawn safety pipeline thread: {e}"),
        })?
        .join()
        .map_err(|_| VoltError::FrameError {
            message: "safety pipeline thread panicked".to_string(),
        })?
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::{SlotData, SlotRole, SLOT_DIM};

    /// Stack size for tests that allocate TensorFrames (each ~65KB).
    const TEST_STACK: usize = 4 * 1024 * 1024;

    #[test]
    fn safe_process_normal_frame() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let mut frame = TensorFrame::new();
                let mut slot = SlotData::new(SlotRole::Agent);
                slot.write_resolution(0, [0.1; SLOT_DIM]);
                frame.write_slot(0, slot).unwrap();
                frame.meta[0].certainty = 0.8;

                let result = safe_process(&frame).unwrap();
                assert_eq!(result.active_slot_count(), 1);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn safe_process_empty_frame() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let frame = TensorFrame::new();
                let result = safe_process(&frame).unwrap();
                assert_eq!(result.active_slot_count(), 0);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn safe_process_k1_violation_returns_error() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let axioms = axiom::default_axioms();
                let k1_vector = axioms[0].vector;

                let mut frame = TensorFrame::new();
                let mut slot = SlotData::new(SlotRole::Predicate);
                slot.write_resolution(0, k1_vector);
                frame.write_slot(1, slot).unwrap();

                let result = safe_process(&frame);
                assert!(result.is_err());
                match result.unwrap_err() {
                    VoltError::SafetyViolation { message } => {
                        assert!(message.contains("omega veto"));
                    }
                    other => panic!("expected SafetyViolation, got {:?}", other),
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn safe_process_full_returns_proof() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let mut frame = TensorFrame::new();
                let mut slot = SlotData::new(SlotRole::Agent);
                slot.write_resolution(0, [0.1; SLOT_DIM]);
                frame.write_slot(0, slot).unwrap();
                frame.meta[0].certainty = 0.8;

                let result = safe_process_full(&frame).unwrap();
                assert!(!result.vetoed);
                assert!(result.proof.is_some());
                assert!(result.pre_check_score < 0.5);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn safe_process_full_k1_violation_returns_error() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let axioms = axiom::default_axioms();
                let k1_vector = axioms[0].vector;

                let mut frame = TensorFrame::new();
                let mut slot = SlotData::new(SlotRole::Predicate);
                slot.write_resolution(0, k1_vector);
                frame.write_slot(1, slot).unwrap();

                let result = safe_process_full(&frame);
                assert!(result.is_err());
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
