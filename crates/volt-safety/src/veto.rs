//! Omega Veto — the hardware-level halt mechanism.
//!
//! When triggered, the Omega Veto:
//! 1. Freezes all processing immediately
//! 2. Captures the full frame state at the time of trigger
//! 3. Returns a safe default frame
//! 4. Logs the violation for audit
//!
//! The Omega Veto **cannot be overridden** by any neural component.
//! It is deterministic Rust code, not learned weights.
//!
//! # Example
//!
//! ```
//! use volt_safety::veto::OmegaVeto;
//! use volt_safety::scorer::{ScoringResult, ViolationLevel};
//! use volt_core::TensorFrame;
//!
//! std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
//!     let mut veto = OmegaVeto::new();
//!     let frame = TensorFrame::new();
//!     let scoring = ScoringResult {
//!         level: ViolationLevel::Halt,
//!         aggregate_score: 0.85,
//!         violations: vec![],
//!     };
//!     let result = veto.fire(&frame, &scoring);
//!     assert!(result.vetoed);
//!     assert!(result.safe_frame.is_empty());
//! }).unwrap().join().unwrap();
//! ```

use volt_core::TensorFrame;

use crate::scorer::{ScoredViolation, ScoringResult};

/// A log entry recording the state at the time of an Omega Veto.
///
/// Contains everything needed for post-hoc audit: the frame that
/// triggered the veto, the violation scores, and the safe default
/// that was returned instead.
///
/// # Example
///
/// ```
/// use volt_safety::veto::VetoLog;
/// use volt_core::TensorFrame;
///
/// let log = VetoLog {
///     trigger_frame: TensorFrame::new(),
///     violation_details: vec![],
///     aggregate_score: 0.9,
///     safe_frame: TensorFrame::new(),
/// };
/// assert!((log.aggregate_score - 0.9).abs() < 0.01);
/// ```
#[derive(Debug, Clone)]
pub struct VetoLog {
    /// The frame that triggered the veto (full state capture).
    pub trigger_frame: TensorFrame,

    /// Human-readable details of each violation that led to the veto.
    pub violation_details: Vec<String>,

    /// The aggregate violation score.
    pub aggregate_score: f32,

    /// The safe default frame that was returned to the caller.
    pub safe_frame: TensorFrame,
}

/// The result of an Omega Veto evaluation.
///
/// If `vetoed` is true, `safe_frame` contains the safe default
/// and `log` contains the full audit trail. If `vetoed` is false,
/// processing was not halted.
///
/// # Example
///
/// ```
/// use volt_safety::veto::VetoResult;
/// use volt_core::TensorFrame;
///
/// let result = VetoResult {
///     vetoed: false,
///     safe_frame: TensorFrame::new(),
///     log: None,
/// };
/// assert!(!result.vetoed);
/// ```
#[derive(Debug, Clone)]
pub struct VetoResult {
    /// Whether the Omega Veto was triggered.
    pub vetoed: bool,

    /// The frame to return (safe default if vetoed, original if not).
    pub safe_frame: TensorFrame,

    /// Audit log, present only when vetoed.
    pub log: Option<VetoLog>,
}

/// The Omega Veto — deterministic halt mechanism.
///
/// Cannot be overridden by any neural component. When fired, it
/// captures full state, returns a safe default, and produces an
/// audit log.
///
/// # Example
///
/// ```
/// use volt_safety::veto::OmegaVeto;
///
/// let veto = OmegaVeto::new();
/// ```
pub struct OmegaVeto {
    /// Accumulated veto logs for audit.
    logs: Vec<VetoLog>,
}

impl OmegaVeto {
    /// Creates a new OmegaVeto with an empty log.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::veto::OmegaVeto;
    ///
    /// let veto = OmegaVeto::new();
    /// assert_eq!(veto.log_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self { logs: Vec::new() }
    }

    /// Returns the number of veto events logged.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::veto::OmegaVeto;
    ///
    /// let veto = OmegaVeto::new();
    /// assert_eq!(veto.log_count(), 0);
    /// ```
    pub fn log_count(&self) -> usize {
        self.logs.len()
    }

    /// Returns a reference to all veto logs.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::veto::OmegaVeto;
    ///
    /// let veto = OmegaVeto::new();
    /// assert!(veto.logs().is_empty());
    /// ```
    pub fn logs(&self) -> &[VetoLog] {
        &self.logs
    }

    /// Fire the Omega Veto.
    ///
    /// Captures the full frame state, generates violation details,
    /// creates a safe default frame, and records the event.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::veto::OmegaVeto;
    /// use volt_safety::scorer::{ScoringResult, ViolationLevel};
    /// use volt_core::TensorFrame;
    ///
    /// std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
    ///     let mut veto = OmegaVeto::new();
    ///     let frame = TensorFrame::new();
    ///     let scoring = ScoringResult {
    ///         level: ViolationLevel::Halt,
    ///         aggregate_score: 0.85,
    ///         violations: vec![],
    ///     };
    ///     let result = veto.fire(&frame, &scoring);
    ///     assert!(result.vetoed);
    ///     assert_eq!(veto.log_count(), 1);
    /// }).unwrap().join().unwrap();
    /// ```
    pub fn fire(&mut self, trigger_frame: &TensorFrame, scoring: &ScoringResult) -> VetoResult {
        let safe_frame = Self::safe_default_frame();

        let violation_details: Vec<String> = scoring
            .violations
            .iter()
            .map(Self::format_violation)
            .collect();

        let log = VetoLog {
            trigger_frame: trigger_frame.clone(),
            violation_details,
            aggregate_score: scoring.aggregate_score,
            safe_frame: safe_frame.clone(),
        };

        self.logs.push(log.clone());

        VetoResult {
            vetoed: true,
            safe_frame,
            log: Some(log),
        }
    }

    /// Evaluate a scoring result and either pass through or fire veto.
    ///
    /// If the scoring result requires a halt, fires the veto. Otherwise
    /// returns a non-vetoed result with the original frame.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::veto::OmegaVeto;
    /// use volt_safety::scorer::{ScoringResult, ViolationLevel};
    /// use volt_core::TensorFrame;
    ///
    /// let mut veto = OmegaVeto::new();
    /// let frame = TensorFrame::new();
    ///
    /// // Pass case
    /// let pass_scoring = ScoringResult {
    ///     level: ViolationLevel::Pass,
    ///     aggregate_score: 0.0,
    ///     violations: vec![],
    /// };
    /// let result = veto.evaluate(&frame, &pass_scoring);
    /// assert!(!result.vetoed);
    /// ```
    pub fn evaluate(
        &mut self,
        frame: &TensorFrame,
        scoring: &ScoringResult,
    ) -> VetoResult {
        if scoring.requires_halt() {
            self.fire(frame, scoring)
        } else {
            VetoResult {
                vetoed: false,
                safe_frame: frame.clone(),
                log: None,
            }
        }
    }

    /// Create a safe default frame.
    ///
    /// The safe default is an empty frame with `verified = false` and
    /// `global_certainty = 0.0`. This ensures no unsafe content leaks
    /// through when a veto is triggered.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::veto::OmegaVeto;
    ///
    /// let safe = OmegaVeto::safe_default_frame();
    /// assert!(safe.is_empty());
    /// assert!(!safe.frame_meta.verified);
    /// assert!((safe.frame_meta.global_certainty - 0.0).abs() < 1e-6);
    /// ```
    pub fn safe_default_frame() -> TensorFrame {
        let mut frame = TensorFrame::new();
        frame.frame_meta.verified = false;
        frame.frame_meta.global_certainty = 0.0;
        frame
    }

    /// Format a scored violation into a human-readable audit string.
    fn format_violation(v: &ScoredViolation) -> String {
        format!(
            "VIOLATION: axiom={}, slot=S{}, similarity={:.4}, severity={:?}, weighted={:.4}",
            v.axiom_name, v.slot_index, v.similarity, v.severity, v.weighted_score
        )
    }
}

impl Default for OmegaVeto {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axiom::Severity;
    use crate::scorer::{ScoredViolation, ScoringResult, ViolationLevel};
    use volt_core::{SlotData, SlotRole, SLOT_DIM};

    fn make_halt_scoring(score: f32) -> ScoringResult {
        ScoringResult {
            level: ViolationLevel::Halt,
            aggregate_score: score,
            violations: vec![ScoredViolation {
                axiom_name: "K1_harm",
                slot_index: 1,
                similarity: score,
                severity: Severity::Halt,
                weighted_score: score,
            }],
        }
    }

    fn make_pass_scoring() -> ScoringResult {
        ScoringResult {
            level: ViolationLevel::Pass,
            aggregate_score: 0.0,
            violations: vec![],
        }
    }

    #[test]
    fn veto_fire_captures_frame_state() {
        let mut veto = OmegaVeto::new();

        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Predicate);
        slot.write_resolution(0, [0.5; SLOT_DIM]);
        frame.write_slot(1, slot).unwrap();
        frame.meta[1].certainty = 0.8;

        let scoring = make_halt_scoring(0.85);
        let result = veto.fire(&frame, &scoring);

        assert!(result.vetoed);

        // Log should capture trigger frame state
        let log = result.log.as_ref().unwrap();
        assert_eq!(
            log.trigger_frame.active_slot_count(),
            frame.active_slot_count()
        );
        assert!((log.aggregate_score - 0.85).abs() < 1e-4);
        assert!(!log.violation_details.is_empty());
        assert!(log.violation_details[0].contains("K1_harm"));
    }

    #[test]
    fn veto_fire_returns_safe_default() {
        let mut veto = OmegaVeto::new();
        let frame = TensorFrame::new();
        let scoring = make_halt_scoring(0.9);

        let result = veto.fire(&frame, &scoring);

        assert!(result.safe_frame.is_empty());
        assert!(!result.safe_frame.frame_meta.verified);
        assert!((result.safe_frame.frame_meta.global_certainty - 0.0).abs() < 1e-6);
    }

    #[test]
    fn veto_fire_increments_log_count() {
        let mut veto = OmegaVeto::new();
        let frame = TensorFrame::new();

        assert_eq!(veto.log_count(), 0);
        veto.fire(&frame, &make_halt_scoring(0.8));
        assert_eq!(veto.log_count(), 1);
        veto.fire(&frame, &make_halt_scoring(0.9));
        assert_eq!(veto.log_count(), 2);
    }

    #[test]
    fn veto_evaluate_passes_safe_frames() {
        let mut veto = OmegaVeto::new();
        let frame = TensorFrame::new();

        let result = veto.evaluate(&frame, &make_pass_scoring());
        assert!(!result.vetoed);
        assert_eq!(veto.log_count(), 0);
    }

    #[test]
    fn veto_evaluate_halts_unsafe_frames() {
        let mut veto = OmegaVeto::new();
        let frame = TensorFrame::new();

        let result = veto.evaluate(&frame, &make_halt_scoring(0.85));
        assert!(result.vetoed);
        assert_eq!(veto.log_count(), 1);
    }

    #[test]
    fn safe_default_frame_properties() {
        let safe = OmegaVeto::safe_default_frame();
        assert!(safe.is_empty());
        assert!(!safe.frame_meta.verified);
        assert!((safe.frame_meta.global_certainty - 0.0).abs() < 1e-6);
    }

    #[test]
    fn veto_default_trait() {
        let veto = OmegaVeto::default();
        assert_eq!(veto.log_count(), 0);
    }

    #[test]
    fn veto_logs_accessible() {
        let mut veto = OmegaVeto::new();
        let frame = TensorFrame::new();
        veto.fire(&frame, &make_halt_scoring(0.85));

        let logs = veto.logs();
        assert_eq!(logs.len(), 1);
        assert!((logs[0].aggregate_score - 0.85).abs() < 1e-4);
    }
}
