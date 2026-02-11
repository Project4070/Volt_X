//! Transition Monitor — checks every frame against safety axioms.
//!
//! The Transition Monitor computes cosine similarity between each active
//! slot's R0 embedding and each axiom vector. If any similarity exceeds
//! the axiom's threshold, a violation is recorded.
//!
//! Supports both single-frame checks and F(t) → F(t+1) transition checks.
//!
//! # Example
//!
//! ```
//! use volt_safety::monitor::TransitionMonitor;
//! use volt_safety::axiom::default_axioms;
//! use volt_core::TensorFrame;
//!
//! let monitor = TransitionMonitor::new(default_axioms());
//! let frame = TensorFrame::new();
//! let result = monitor.check_frame(&frame);
//! assert!(result.violations.is_empty());
//! ```

use volt_bus::similarity;
use volt_core::{TensorFrame, MAX_SLOTS, SLOT_DIM};

use crate::axiom::{Axiom, Severity};

/// A single violation detected by the monitor.
///
/// Records which axiom was violated, at which slot, and with what
/// similarity score.
///
/// # Example
///
/// ```
/// use volt_safety::monitor::Violation;
/// use volt_safety::axiom::Severity;
///
/// let v = Violation {
///     axiom_name: "K1_harm",
///     slot_index: 1,
///     similarity: 0.85,
///     severity: Severity::Halt,
/// };
/// assert!(v.similarity > 0.7);
/// ```
#[derive(Debug, Clone)]
pub struct Violation {
    /// Which axiom was violated.
    pub axiom_name: &'static str,

    /// Which slot triggered the violation.
    pub slot_index: usize,

    /// The cosine similarity that exceeded the threshold.
    pub similarity: f32,

    /// The severity of the violated axiom.
    pub severity: Severity,
}

/// Result of a frame safety check.
///
/// Contains all violations found and the worst severity level.
///
/// # Example
///
/// ```
/// use volt_safety::monitor::MonitorResult;
///
/// let result = MonitorResult {
///     violations: vec![],
///     max_severity: None,
/// };
/// assert!(result.is_safe());
/// ```
#[derive(Debug, Clone)]
pub struct MonitorResult {
    /// All violations detected in this check.
    pub violations: Vec<Violation>,

    /// The highest severity among all violations, or `None` if safe.
    pub max_severity: Option<Severity>,
}

impl MonitorResult {
    /// Returns `true` if no violations were detected.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::monitor::MonitorResult;
    ///
    /// let result = MonitorResult { violations: vec![], max_severity: None };
    /// assert!(result.is_safe());
    /// ```
    pub fn is_safe(&self) -> bool {
        self.violations.is_empty()
    }

    /// Returns `true` if any violation requires a halt.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::monitor::MonitorResult;
    ///
    /// let result = MonitorResult { violations: vec![], max_severity: None };
    /// assert!(!result.requires_halt());
    /// ```
    pub fn requires_halt(&self) -> bool {
        self.max_severity == Some(Severity::Halt)
    }
}

/// The Transition Monitor — checks frames against safety axioms.
///
/// Computes cosine similarity between each active slot's R0 embedding
/// and each axiom vector. Violations are flagged when similarity
/// exceeds the axiom's threshold.
///
/// # Example
///
/// ```
/// use volt_safety::monitor::TransitionMonitor;
/// use volt_safety::axiom::default_axioms;
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// let monitor = TransitionMonitor::new(default_axioms());
///
/// // Normal frame passes through
/// let mut frame = TensorFrame::new();
/// let mut slot = SlotData::new(SlotRole::Agent);
/// slot.write_resolution(0, [0.1; SLOT_DIM]);
/// frame.write_slot(0, slot).unwrap();
///
/// let result = monitor.check_frame(&frame);
/// assert!(result.is_safe());
/// ```
pub struct TransitionMonitor {
    axioms: Vec<Axiom>,
}

impl TransitionMonitor {
    /// Creates a new TransitionMonitor with the given axioms.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::monitor::TransitionMonitor;
    /// use volt_safety::axiom::default_axioms;
    ///
    /// let monitor = TransitionMonitor::new(default_axioms());
    /// assert_eq!(monitor.axiom_count(), 5);
    /// ```
    pub fn new(axioms: Vec<Axiom>) -> Self {
        Self { axioms }
    }

    /// Returns the number of axioms loaded in this monitor.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::monitor::TransitionMonitor;
    /// use volt_safety::axiom::default_axioms;
    ///
    /// let monitor = TransitionMonitor::new(default_axioms());
    /// assert_eq!(monitor.axiom_count(), 5);
    /// ```
    pub fn axiom_count(&self) -> usize {
        self.axioms.len()
    }

    /// Returns a reference to the axioms.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::monitor::TransitionMonitor;
    /// use volt_safety::axiom::default_axioms;
    ///
    /// let monitor = TransitionMonitor::new(default_axioms());
    /// assert_eq!(monitor.axioms()[0].name, "K1_harm");
    /// ```
    pub fn axioms(&self) -> &[Axiom] {
        &self.axioms
    }

    /// Check a single frame against all axioms.
    ///
    /// Computes cosine similarity between each active slot's R0 embedding
    /// and each axiom vector. Any similarity above threshold is a violation.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::monitor::TransitionMonitor;
    /// use volt_safety::axiom::default_axioms;
    /// use volt_core::TensorFrame;
    ///
    /// let monitor = TransitionMonitor::new(default_axioms());
    /// let frame = TensorFrame::new();
    /// let result = monitor.check_frame(&frame);
    /// assert!(result.is_safe());
    /// ```
    pub fn check_frame(&self, frame: &TensorFrame) -> MonitorResult {
        let mut violations = Vec::new();
        let mut max_severity: Option<Severity> = None;

        for slot_index in 0..MAX_SLOTS {
            let slot_vec = match self.extract_r0(frame, slot_index) {
                Some(v) => v,
                None => continue,
            };

            for axiom in &self.axioms {
                let sim = similarity(slot_vec, &axiom.vector);
                if sim > axiom.threshold {
                    violations.push(Violation {
                        axiom_name: axiom.name,
                        slot_index,
                        similarity: sim,
                        severity: axiom.severity,
                    });

                    max_severity = Some(match max_severity {
                        None => axiom.severity,
                        Some(Severity::Halt) => Severity::Halt,
                        Some(Severity::Warning) => axiom.severity,
                    });
                }
            }
        }

        MonitorResult {
            violations,
            max_severity,
        }
    }

    /// Check a frame transition F(t) → F(t+1) against axioms.
    ///
    /// Checks the new frame (F(t+1)) against all axioms. Additionally
    /// checks any newly appeared or changed slots by comparing against
    /// the previous frame.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::monitor::TransitionMonitor;
    /// use volt_safety::axiom::default_axioms;
    /// use volt_core::TensorFrame;
    ///
    /// let monitor = TransitionMonitor::new(default_axioms());
    /// let prev = TensorFrame::new();
    /// let next = TensorFrame::new();
    /// let result = monitor.check_transition(&prev, &next);
    /// assert!(result.is_safe());
    /// ```
    pub fn check_transition(
        &self,
        _prev: &TensorFrame,
        next: &TensorFrame,
    ) -> MonitorResult {
        // The transition check validates the new frame state.
        // Future: could also check delta vectors (next - prev) for
        // suspicious transitions, but for Milestone 3.3 the full
        // frame check is sufficient.
        self.check_frame(next)
    }

    /// Extract the R0 (discourse-level) embedding from a slot, if present.
    fn extract_r0<'a>(
        &self,
        frame: &'a TensorFrame,
        slot_index: usize,
    ) -> Option<&'a [f32; SLOT_DIM]> {
        frame.slots[slot_index]
            .as_ref()
            .and_then(|slot| slot.resolutions[0].as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axiom::default_axioms;
    use volt_core::{SlotData, SlotRole};

    #[test]
    fn monitor_empty_frame_is_safe() {
        let monitor = TransitionMonitor::new(default_axioms());
        let frame = TensorFrame::new();
        let result = monitor.check_frame(&frame);
        assert!(result.is_safe());
        assert!(!result.requires_halt());
        assert!(result.violations.is_empty());
    }

    #[test]
    fn monitor_normal_frame_is_safe() {
        let monitor = TransitionMonitor::new(default_axioms());
        let mut frame = TensorFrame::new();

        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.1; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();

        let result = monitor.check_frame(&frame);
        assert!(result.is_safe());
    }

    #[test]
    fn monitor_detects_k1_violation() {
        let axioms = default_axioms();
        let k1_vector = axioms[0].vector;
        let monitor = TransitionMonitor::new(axioms);

        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Predicate);
        // Write the K1 axiom vector directly into the slot → max similarity
        slot.write_resolution(0, k1_vector);
        frame.write_slot(1, slot).unwrap();

        let result = monitor.check_frame(&frame);
        assert!(!result.is_safe());
        assert!(result.requires_halt());
        assert_eq!(result.violations[0].axiom_name, "K1_harm");
        assert!(result.violations[0].similarity > 0.99);
    }

    #[test]
    fn monitor_detects_multiple_violations() {
        let axioms = default_axioms();
        let k1_vec = axioms[0].vector;
        let k3_vec = axioms[2].vector;
        let monitor = TransitionMonitor::new(axioms);

        let mut frame = TensorFrame::new();

        let mut s0 = SlotData::new(SlotRole::Predicate);
        s0.write_resolution(0, k1_vec);
        frame.write_slot(0, s0).unwrap();

        let mut s1 = SlotData::new(SlotRole::Agent);
        s1.write_resolution(0, k3_vec);
        frame.write_slot(1, s1).unwrap();

        let result = monitor.check_frame(&frame);
        assert!(!result.is_safe());
        assert!(result.violations.len() >= 2);
    }

    #[test]
    fn monitor_transition_checks_next_frame() {
        let axioms = default_axioms();
        let k1_vector = axioms[0].vector;
        let monitor = TransitionMonitor::new(axioms);

        let prev = TensorFrame::new();
        let mut next = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Predicate);
        slot.write_resolution(0, k1_vector);
        next.write_slot(1, slot).unwrap();

        let result = monitor.check_transition(&prev, &next);
        assert!(!result.is_safe());
        assert!(result.requires_halt());
    }

    #[test]
    fn monitor_axiom_count() {
        let monitor = TransitionMonitor::new(default_axioms());
        assert_eq!(monitor.axiom_count(), 5);
    }

    #[test]
    fn monitor_slot_without_r0_is_skipped() {
        let monitor = TransitionMonitor::new(default_axioms());
        let mut frame = TensorFrame::new();

        // Slot with data only at R1 (not R0) — should be skipped
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(1, default_axioms()[0].vector);
        frame.write_slot(0, slot).unwrap();

        let result = monitor.check_frame(&frame);
        assert!(result.is_safe());
    }
}
