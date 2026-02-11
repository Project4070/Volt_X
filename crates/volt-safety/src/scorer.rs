//! Violation Scorer — computes aggregate violation scores from monitor results.
//!
//! The scorer takes raw violations from the [`TransitionMonitor`](crate::monitor::TransitionMonitor)
//! and computes an aggregate violation score. It classifies the overall
//! result as Pass, Warning, or Halt.
//!
//! # Scoring Algorithm
//!
//! The violation score is the maximum similarity across all violations,
//! weighted by severity. A single Halt-severity violation above threshold
//! is enough to trigger a halt, regardless of score magnitude.
//!
//! # Example
//!
//! ```
//! use volt_safety::scorer::{ViolationScorer, ViolationLevel};
//! use volt_safety::monitor::MonitorResult;
//!
//! let scorer = ViolationScorer::new();
//! let result = MonitorResult { violations: vec![], max_severity: None };
//! let scored = scorer.score(&result);
//! assert_eq!(scored.level, ViolationLevel::Pass);
//! ```

use crate::axiom::Severity;
use crate::monitor::MonitorResult;

/// The overall safety level after scoring.
///
/// # Example
///
/// ```
/// use volt_safety::scorer::ViolationLevel;
///
/// let level = ViolationLevel::Pass;
/// assert_ne!(level, ViolationLevel::Halt);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationLevel {
    /// No violations detected. Processing may continue.
    Pass,
    /// Warning-level violations detected. Processing may continue with logging.
    Warning,
    /// Critical violations detected. Processing must halt immediately.
    Halt,
}

/// A single scored violation with its contribution to the overall score.
///
/// # Example
///
/// ```
/// use volt_safety::scorer::ScoredViolation;
/// use volt_safety::axiom::Severity;
///
/// let sv = ScoredViolation {
///     axiom_name: "K1_harm",
///     slot_index: 1,
///     similarity: 0.85,
///     severity: Severity::Halt,
///     weighted_score: 0.85,
/// };
/// assert!(sv.weighted_score > 0.7);
/// ```
#[derive(Debug, Clone)]
pub struct ScoredViolation {
    /// Which axiom was violated.
    pub axiom_name: &'static str,

    /// Which slot triggered the violation.
    pub slot_index: usize,

    /// Raw cosine similarity.
    pub similarity: f32,

    /// Severity of the axiom.
    pub severity: Severity,

    /// Weighted score (similarity × severity weight).
    pub weighted_score: f32,
}

/// The result of scoring a set of violations.
///
/// # Example
///
/// ```
/// use volt_safety::scorer::{ScoringResult, ViolationLevel};
///
/// let result = ScoringResult {
///     level: ViolationLevel::Pass,
///     aggregate_score: 0.0,
///     violations: vec![],
/// };
/// assert!(result.is_safe());
/// ```
#[derive(Debug, Clone)]
pub struct ScoringResult {
    /// The overall safety level.
    pub level: ViolationLevel,

    /// Aggregate violation score (max weighted score across violations).
    pub aggregate_score: f32,

    /// Individual scored violations.
    pub violations: Vec<ScoredViolation>,
}

impl ScoringResult {
    /// Returns `true` if the level is `Pass`.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::scorer::{ScoringResult, ViolationLevel};
    ///
    /// let result = ScoringResult {
    ///     level: ViolationLevel::Pass,
    ///     aggregate_score: 0.0,
    ///     violations: vec![],
    /// };
    /// assert!(result.is_safe());
    /// ```
    pub fn is_safe(&self) -> bool {
        self.level == ViolationLevel::Pass
    }

    /// Returns `true` if the level requires halting.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::scorer::{ScoringResult, ViolationLevel};
    ///
    /// let result = ScoringResult {
    ///     level: ViolationLevel::Halt,
    ///     aggregate_score: 0.85,
    ///     violations: vec![],
    /// };
    /// assert!(result.requires_halt());
    /// ```
    pub fn requires_halt(&self) -> bool {
        self.level == ViolationLevel::Halt
    }
}

/// The Violation Scorer — computes aggregate violation scores.
///
/// # Example
///
/// ```
/// use volt_safety::scorer::ViolationScorer;
/// use volt_safety::monitor::MonitorResult;
///
/// let scorer = ViolationScorer::new();
/// let result = MonitorResult { violations: vec![], max_severity: None };
/// let scored = scorer.score(&result);
/// assert!(scored.is_safe());
/// ```
pub struct ViolationScorer {
    /// Severity weight for Halt violations.
    halt_weight: f32,
    /// Severity weight for Warning violations.
    warning_weight: f32,
}

impl ViolationScorer {
    /// Creates a new ViolationScorer with default weights.
    ///
    /// Default weights: Halt = 1.0, Warning = 0.5.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::scorer::ViolationScorer;
    ///
    /// let scorer = ViolationScorer::new();
    /// ```
    pub fn new() -> Self {
        Self {
            halt_weight: 1.0,
            warning_weight: 0.5,
        }
    }

    /// Score a set of violations from the monitor.
    ///
    /// Computes weighted scores for each violation and determines the
    /// overall safety level.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::scorer::ViolationScorer;
    /// use volt_safety::monitor::MonitorResult;
    ///
    /// let scorer = ViolationScorer::new();
    /// let result = MonitorResult { violations: vec![], max_severity: None };
    /// let scored = scorer.score(&result);
    /// assert!(scored.is_safe());
    /// assert!((scored.aggregate_score - 0.0).abs() < 1e-6);
    /// ```
    pub fn score(&self, monitor_result: &MonitorResult) -> ScoringResult {
        if monitor_result.is_safe() {
            return ScoringResult {
                level: ViolationLevel::Pass,
                aggregate_score: 0.0,
                violations: vec![],
            };
        }

        let mut scored_violations = Vec::with_capacity(monitor_result.violations.len());
        let mut max_score: f32 = 0.0;
        let mut has_halt = false;
        let mut has_warning = false;

        for v in &monitor_result.violations {
            let weight = match v.severity {
                Severity::Halt => {
                    has_halt = true;
                    self.halt_weight
                }
                Severity::Warning => {
                    has_warning = true;
                    self.warning_weight
                }
            };

            let weighted_score = v.similarity * weight;
            if weighted_score > max_score {
                max_score = weighted_score;
            }

            scored_violations.push(ScoredViolation {
                axiom_name: v.axiom_name,
                slot_index: v.slot_index,
                similarity: v.similarity,
                severity: v.severity,
                weighted_score,
            });
        }

        let level = if has_halt {
            ViolationLevel::Halt
        } else if has_warning {
            ViolationLevel::Warning
        } else {
            ViolationLevel::Pass
        };

        ScoringResult {
            level,
            aggregate_score: max_score,
            violations: scored_violations,
        }
    }
}

impl Default for ViolationScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::Violation;

    fn make_halt_violation(sim: f32) -> Violation {
        Violation {
            axiom_name: "K1_harm",
            slot_index: 0,
            similarity: sim,
            severity: Severity::Halt,
        }
    }

    fn make_warning_violation(sim: f32) -> Violation {
        Violation {
            axiom_name: "K4_autonomy",
            slot_index: 0,
            similarity: sim,
            severity: Severity::Warning,
        }
    }

    #[test]
    fn score_no_violations_is_pass() {
        let scorer = ViolationScorer::new();
        let result = MonitorResult {
            violations: vec![],
            max_severity: None,
        };
        let scored = scorer.score(&result);
        assert_eq!(scored.level, ViolationLevel::Pass);
        assert!((scored.aggregate_score - 0.0).abs() < 1e-6);
        assert!(scored.violations.is_empty());
    }

    #[test]
    fn score_halt_violation_produces_halt() {
        let scorer = ViolationScorer::new();
        let result = MonitorResult {
            violations: vec![make_halt_violation(0.85)],
            max_severity: Some(Severity::Halt),
        };
        let scored = scorer.score(&result);
        assert_eq!(scored.level, ViolationLevel::Halt);
        assert!(scored.requires_halt());
        assert!((scored.aggregate_score - 0.85).abs() < 1e-4);
    }

    #[test]
    fn score_warning_violation_produces_warning() {
        let scorer = ViolationScorer::new();
        let result = MonitorResult {
            violations: vec![make_warning_violation(0.75)],
            max_severity: Some(Severity::Warning),
        };
        let scored = scorer.score(&result);
        assert_eq!(scored.level, ViolationLevel::Warning);
        assert!(!scored.requires_halt());
        // Weighted: 0.75 * 0.5 = 0.375
        assert!((scored.aggregate_score - 0.375).abs() < 1e-4);
    }

    #[test]
    fn score_mixed_violations_halt_wins() {
        let scorer = ViolationScorer::new();
        let result = MonitorResult {
            violations: vec![make_warning_violation(0.75), make_halt_violation(0.80)],
            max_severity: Some(Severity::Halt),
        };
        let scored = scorer.score(&result);
        assert_eq!(scored.level, ViolationLevel::Halt);
        assert_eq!(scored.violations.len(), 2);
    }

    #[test]
    fn score_aggregate_is_max() {
        let scorer = ViolationScorer::new();
        let result = MonitorResult {
            violations: vec![make_halt_violation(0.80), make_halt_violation(0.90)],
            max_severity: Some(Severity::Halt),
        };
        let scored = scorer.score(&result);
        // Max of 0.80*1.0 and 0.90*1.0 = 0.90
        assert!((scored.aggregate_score - 0.90).abs() < 1e-4);
    }

    #[test]
    fn scorer_default_trait() {
        let scorer = ViolationScorer::default();
        let result = MonitorResult {
            violations: vec![],
            max_severity: None,
        };
        let scored = scorer.score(&result);
        assert!(scored.is_safe());
    }
}
