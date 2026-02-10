//! CertaintyEngine — min-rule gamma propagation across frame slots.
//!
//! The CertaintyEngine computes frame-level global certainty as the
//! minimum of all active slot gammas. This implements the **min-rule**:
//! the overall frame certainty is bounded by the least-certain component.
//!
//! The CertaintyEngine is **not** a [`HardStrand`](crate::strand::HardStrand).
//! It is pipeline infrastructure that runs on every frame after strand
//! execution, unconditionally.
//!
//! # Example
//!
//! ```
//! use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
//! use volt_hard::certainty_engine::CertaintyEngine;
//!
//! let engine = CertaintyEngine::new();
//! let mut frame = TensorFrame::new();
//!
//! let mut s0 = SlotData::new(SlotRole::Agent);
//! s0.write_resolution(0, [0.5; SLOT_DIM]);
//! frame.write_slot(0, s0).unwrap();
//! frame.meta[0].certainty = 0.8;
//!
//! let mut s1 = SlotData::new(SlotRole::Predicate);
//! s1.write_resolution(0, [0.3; SLOT_DIM]);
//! frame.write_slot(1, s1).unwrap();
//! frame.meta[1].certainty = 0.6;
//!
//! let result = engine.propagate(&mut frame);
//! assert!((result.global_certainty - 0.6).abs() < 0.01);
//! assert_eq!(frame.frame_meta.global_certainty, 0.6);
//! ```

use volt_core::{TensorFrame, MAX_SLOTS};

/// The result of certainty propagation across a frame.
///
/// # Example
///
/// ```
/// use volt_hard::certainty_engine::CertaintyResult;
///
/// let result = CertaintyResult {
///     global_certainty: 0.6,
///     weakest_slot: Some(2),
///     slot_gammas: vec![(0, 1.0), (1, 0.8), (2, 0.6)],
/// };
/// assert_eq!(result.weakest_slot, Some(2));
/// ```
#[derive(Debug, Clone)]
pub struct CertaintyResult {
    /// The computed global certainty (min of all active slot gammas).
    pub global_certainty: f32,

    /// The slot index with the lowest certainty, or `None` if no slots are active.
    pub weakest_slot: Option<usize>,

    /// Per-slot gamma values for all active slots: `(slot_index, gamma)`.
    pub slot_gammas: Vec<(usize, f32)>,
}

/// Min-rule certainty propagation engine.
///
/// Computes the frame-level global certainty as the minimum of all
/// active slot gammas. This ensures the overall frame certainty is
/// bounded by the least-certain component.
///
/// # Example
///
/// ```
/// use volt_hard::certainty_engine::CertaintyEngine;
///
/// let engine = CertaintyEngine::new();
/// assert_eq!(engine.name(), "certainty_engine");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CertaintyEngine;

impl CertaintyEngine {
    /// Creates a new CertaintyEngine.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::certainty_engine::CertaintyEngine;
    ///
    /// let engine = CertaintyEngine::new();
    /// ```
    pub fn new() -> Self {
        Self
    }

    /// Returns the name of this engine.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::certainty_engine::CertaintyEngine;
    ///
    /// let engine = CertaintyEngine::new();
    /// assert_eq!(engine.name(), "certainty_engine");
    /// ```
    pub fn name(&self) -> &str {
        "certainty_engine"
    }

    /// Propagate certainty across a frame using the min-rule.
    ///
    /// Computes global certainty as `min(all active slot gammas)` and
    /// writes the result to `frame.frame_meta.global_certainty`.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
    /// use volt_hard::certainty_engine::CertaintyEngine;
    ///
    /// let engine = CertaintyEngine::new();
    /// let mut frame = TensorFrame::new();
    ///
    /// let mut slot = SlotData::new(SlotRole::Agent);
    /// slot.write_resolution(0, [0.5; SLOT_DIM]);
    /// frame.write_slot(0, slot).unwrap();
    /// frame.meta[0].certainty = 0.75;
    ///
    /// let result = engine.propagate(&mut frame);
    /// assert!((result.global_certainty - 0.75).abs() < 0.01);
    /// ```
    pub fn propagate(&self, frame: &mut TensorFrame) -> CertaintyResult {
        let result = self.compute(frame);
        frame.frame_meta.global_certainty = result.global_certainty;
        result
    }

    /// Compute certainty without modifying the frame (read-only query).
    ///
    /// Returns the same result as [`propagate`](Self::propagate) but does
    /// not write to `frame.frame_meta.global_certainty`.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
    /// use volt_hard::certainty_engine::CertaintyEngine;
    ///
    /// let engine = CertaintyEngine::new();
    /// let mut frame = TensorFrame::new();
    ///
    /// let mut slot = SlotData::new(SlotRole::Agent);
    /// slot.write_resolution(0, [0.5; SLOT_DIM]);
    /// frame.write_slot(0, slot).unwrap();
    /// frame.meta[0].certainty = 0.9;
    ///
    /// let result = engine.compute(&frame);
    /// assert!((result.global_certainty - 0.9).abs() < 0.01);
    /// // frame.frame_meta.global_certainty is NOT modified
    /// assert!((frame.frame_meta.global_certainty - 0.0).abs() < 0.01);
    /// ```
    pub fn compute(&self, frame: &TensorFrame) -> CertaintyResult {
        let mut min_gamma = f32::MAX;
        let mut weakest_slot = None;
        let mut slot_gammas = Vec::new();

        for i in 0..MAX_SLOTS {
            if frame.slots[i].is_some() {
                let gamma = frame.meta[i].certainty;
                slot_gammas.push((i, gamma));
                if gamma < min_gamma {
                    min_gamma = gamma;
                    weakest_slot = Some(i);
                }
            }
        }

        let global = if min_gamma < f32::MAX {
            min_gamma
        } else {
            0.0
        };

        CertaintyResult {
            global_certainty: global,
            weakest_slot,
            slot_gammas,
        }
    }
}

impl Default for CertaintyEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::{SlotData, SlotRole, SLOT_DIM};

    #[test]
    fn certainty_engine_name() {
        let engine = CertaintyEngine::new();
        assert_eq!(engine.name(), "certainty_engine");
    }

    #[test]
    fn certainty_engine_empty_frame() {
        let engine = CertaintyEngine::new();
        let mut frame = TensorFrame::new();
        let result = engine.propagate(&mut frame);

        assert!((result.global_certainty - 0.0).abs() < 0.01);
        assert_eq!(result.weakest_slot, None);
        assert!(result.slot_gammas.is_empty());
        assert!((frame.frame_meta.global_certainty - 0.0).abs() < 0.01);
    }

    #[test]
    fn certainty_engine_single_slot() {
        let engine = CertaintyEngine::new();
        let mut frame = TensorFrame::new();

        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.5; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();
        frame.meta[0].certainty = 0.8;

        let result = engine.propagate(&mut frame);

        assert!((result.global_certainty - 0.8).abs() < 0.01);
        assert_eq!(result.weakest_slot, Some(0));
        assert_eq!(result.slot_gammas.len(), 1);
        assert_eq!(result.slot_gammas[0], (0, 0.8));
    }

    #[test]
    fn milestone_certainty_min_rule_three_slots() {
        // Milestone 3.2 test: frame with gamma=[1.0, 0.8, 0.6] -> global gamma = 0.6
        let engine = CertaintyEngine::new();
        let mut frame = TensorFrame::new();

        let mut s0 = SlotData::new(SlotRole::Agent);
        s0.write_resolution(0, [0.5; SLOT_DIM]);
        frame.write_slot(0, s0).unwrap();
        frame.meta[0].certainty = 1.0;

        let mut s1 = SlotData::new(SlotRole::Predicate);
        s1.write_resolution(0, [0.3; SLOT_DIM]);
        frame.write_slot(1, s1).unwrap();
        frame.meta[1].certainty = 0.8;

        let mut s2 = SlotData::new(SlotRole::Patient);
        s2.write_resolution(0, [0.7; SLOT_DIM]);
        frame.write_slot(2, s2).unwrap();
        frame.meta[2].certainty = 0.6;

        let result = engine.propagate(&mut frame);

        assert!(
            (result.global_certainty - 0.6).abs() < 0.01,
            "global certainty should be 0.6 (min-rule), got {}",
            result.global_certainty
        );
        assert_eq!(result.weakest_slot, Some(2));
        assert_eq!(result.slot_gammas.len(), 3);
        assert!((frame.frame_meta.global_certainty - 0.6).abs() < 0.01);
    }

    #[test]
    fn certainty_engine_all_certain() {
        let engine = CertaintyEngine::new();
        let mut frame = TensorFrame::new();

        for i in 0..3 {
            let mut slot = SlotData::new(SlotRole::Agent);
            slot.write_resolution(0, [0.5; SLOT_DIM]);
            frame.write_slot(i, slot).unwrap();
            frame.meta[i].certainty = 1.0;
        }

        let result = engine.propagate(&mut frame);

        assert!((result.global_certainty - 1.0).abs() < 0.01);
    }

    #[test]
    fn certainty_engine_compute_does_not_mutate() {
        let engine = CertaintyEngine::new();
        let mut frame = TensorFrame::new();

        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.5; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();
        frame.meta[0].certainty = 0.7;
        frame.frame_meta.global_certainty = 0.0;

        let result = engine.compute(&frame);

        assert!((result.global_certainty - 0.7).abs() < 0.01);
        // Frame should NOT be mutated
        assert!(
            (frame.frame_meta.global_certainty - 0.0).abs() < 0.01,
            "compute() should not mutate frame"
        );
    }

    #[test]
    fn certainty_engine_propagate_updates_frame() {
        let engine = CertaintyEngine::new();
        let mut frame = TensorFrame::new();

        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.5; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();
        frame.meta[0].certainty = 0.55;
        frame.frame_meta.global_certainty = 0.0;

        let result = engine.propagate(&mut frame);

        assert!((result.global_certainty - 0.55).abs() < 0.01);
        assert!(
            (frame.frame_meta.global_certainty - 0.55).abs() < 0.01,
            "propagate() should update frame.frame_meta.global_certainty"
        );
    }

    #[test]
    fn certainty_engine_default_trait() {
        let engine = CertaintyEngine::default();
        assert_eq!(engine.name(), "certainty_engine");
    }
}
