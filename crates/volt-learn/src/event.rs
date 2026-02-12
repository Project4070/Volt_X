//! Learning event data structure.
//!
//! A [`LearningEvent`] captures diagnostic information from a single
//! inference run. Events are accumulated in an
//! [`EventBuffer`](crate::buffer::EventBuffer) and used to compute
//! per-strand learning statistics.

use serde::{Deserialize, Serialize};
use volt_core::meta::DiscourseType;
use volt_core::MAX_SLOTS;

/// A single learning event captured from one inference run.
///
/// Contains enough information to track per-strand usage patterns,
/// convergence behavior, and memory influence for the learning pipeline.
///
/// # Example
///
/// ```
/// use volt_learn::LearningEvent;
/// use volt_core::meta::DiscourseType;
/// use volt_core::MAX_SLOTS;
///
/// let event = LearningEvent {
///     frame_id: 42,
///     strand_id: 1,
///     query_type: DiscourseType::Query,
///     gamma_scores: [0.0; MAX_SLOTS],
///     convergence_iterations: 10,
///     ghost_activations: 5,
///     timestamp: 1_000_000,
/// };
/// assert_eq!(event.frame_id, 42);
/// assert_eq!(event.convergence_iterations, 10);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEvent {
    /// The unique ID of the frame produced by this inference.
    pub frame_id: u64,

    /// The strand this frame was stored to.
    pub strand_id: u64,

    /// The discourse type of the query (Query, Command, Creative, etc.).
    pub query_type: DiscourseType,

    /// Per-slot certainty (gamma) scores. `0.0` for inactive slots.
    pub gamma_scores: [f32; MAX_SLOTS],

    /// Number of RAR iterations used to reach convergence (or budget).
    pub convergence_iterations: u32,

    /// Number of ghost gists that influenced the RAR Attend phase.
    pub ghost_activations: usize,

    /// Timestamp in microseconds since epoch.
    pub timestamp: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event() -> LearningEvent {
        LearningEvent {
            frame_id: 1,
            strand_id: 0,
            query_type: DiscourseType::Query,
            gamma_scores: [0.8; MAX_SLOTS],
            convergence_iterations: 10,
            ghost_activations: 3,
            timestamp: 1_000_000,
        }
    }

    #[test]
    fn event_fields_are_accessible() {
        let event = make_event();
        assert_eq!(event.frame_id, 1);
        assert_eq!(event.strand_id, 0);
        assert_eq!(event.query_type, DiscourseType::Query);
        assert_eq!(event.convergence_iterations, 10);
        assert_eq!(event.ghost_activations, 3);
        assert_eq!(event.timestamp, 1_000_000);
        assert!((event.gamma_scores[0] - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn event_is_serializable() {
        let event = make_event();
        let json = serde_json::to_string(&event).expect("serialize");
        let roundtrip: LearningEvent =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(roundtrip.frame_id, event.frame_id);
        assert_eq!(roundtrip.strand_id, event.strand_id);
        assert_eq!(roundtrip.convergence_iterations, event.convergence_iterations);
        assert_eq!(roundtrip.ghost_activations, event.ghost_activations);
        assert_eq!(roundtrip.timestamp, event.timestamp);
        for i in 0..MAX_SLOTS {
            assert!(
                (roundtrip.gamma_scores[i] - event.gamma_scores[i]).abs() < f32::EPSILON
            );
        }
    }

    #[test]
    fn event_clone_is_independent() {
        let event = make_event();
        let mut cloned = event.clone();
        cloned.frame_id = 999;
        cloned.gamma_scores[0] = 0.1;
        assert_eq!(event.frame_id, 1);
        assert!((event.gamma_scores[0] - 0.8).abs() < f32::EPSILON);
    }
}
