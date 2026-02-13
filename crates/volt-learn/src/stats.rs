//! Per-strand learning statistics computed from event buffers.
//!
//! Statistics are computed on demand from the current buffer contents.
//! No incremental counters are maintained — this keeps the code simple
//! and avoids synchronization issues between counters and the buffer.

use std::collections::HashMap;

use crate::event::LearningEvent;
use volt_core::meta::DiscourseType;
use volt_core::MAX_SLOTS;

/// Aggregated statistics for a single strand.
///
/// Computed from all learning events in the buffer that belong
/// to the strand.
///
/// # Example
///
/// ```
/// use volt_learn::StrandStatistics;
/// use volt_learn::TopicDistribution;
///
/// let stats = StrandStatistics {
///     strand_id: 1,
///     query_count: 42,
///     average_gamma: 0.75,
///     average_iterations: 12.5,
///     topic_distribution: TopicDistribution::default(),
/// };
/// assert_eq!(stats.query_count, 42);
/// ```
#[derive(Debug, Clone)]
pub struct StrandStatistics {
    /// The strand ID these statistics apply to.
    pub strand_id: u64,

    /// Total number of learning events (queries) for this strand.
    pub query_count: usize,

    /// Average global gamma across all events. Computed as the mean
    /// of per-event mean active-slot gammas.
    pub average_gamma: f32,

    /// Average number of RAR iterations per query.
    pub average_iterations: f32,

    /// Distribution of discourse types across events.
    pub topic_distribution: TopicDistribution,
}

/// Distribution of discourse types across learning events.
///
/// Each field is a count of events with that discourse type.
///
/// # Example
///
/// ```
/// use volt_learn::TopicDistribution;
///
/// let dist = TopicDistribution::default();
/// assert_eq!(dist.query, 0);
/// assert_eq!(dist.total(), 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct TopicDistribution {
    /// Count of `DiscourseType::Query` events.
    pub query: usize,
    /// Count of `DiscourseType::Statement` events.
    pub statement: usize,
    /// Count of `DiscourseType::Command` events.
    pub command: usize,
    /// Count of `DiscourseType::Response` events.
    pub response: usize,
    /// Count of `DiscourseType::Creative` events.
    pub creative: usize,
    /// Count of `DiscourseType::Unknown` events.
    pub unknown: usize,
}

impl TopicDistribution {
    /// Returns the total number of events across all discourse types.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::TopicDistribution;
    ///
    /// let mut dist = TopicDistribution::default();
    /// dist.query = 10;
    /// dist.command = 5;
    /// assert_eq!(dist.total(), 15);
    /// ```
    pub fn total(&self) -> usize {
        self.query + self.statement + self.command + self.response + self.creative + self.unknown
    }

    /// Returns normalized proportions (0.0–1.0) for each discourse type.
    ///
    /// Returns an empty map if there are no events.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::TopicDistribution;
    ///
    /// let mut dist = TopicDistribution::default();
    /// dist.query = 3;
    /// dist.command = 1;
    /// let props = dist.proportions();
    /// assert!((props["Query"] - 0.75).abs() < 0.01);
    /// assert!((props["Command"] - 0.25).abs() < 0.01);
    /// ```
    pub fn proportions(&self) -> HashMap<String, f32> {
        let total = self.total();
        if total == 0 {
            return HashMap::new();
        }
        let t = total as f32;
        let mut map = HashMap::new();
        map.insert("Query".to_string(), self.query as f32 / t);
        map.insert("Statement".to_string(), self.statement as f32 / t);
        map.insert("Command".to_string(), self.command as f32 / t);
        map.insert("Response".to_string(), self.response as f32 / t);
        map.insert("Creative".to_string(), self.creative as f32 / t);
        map.insert("Unknown".to_string(), self.unknown as f32 / t);
        map
    }

    /// Returns the dominant discourse type, or `None` if empty.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::TopicDistribution;
    /// use volt_core::meta::DiscourseType;
    ///
    /// let mut dist = TopicDistribution::default();
    /// dist.command = 10;
    /// dist.query = 3;
    /// assert_eq!(dist.dominant(), Some(DiscourseType::Command));
    /// ```
    pub fn dominant(&self) -> Option<DiscourseType> {
        if self.total() == 0 {
            return None;
        }
        let pairs = [
            (self.query, DiscourseType::Query),
            (self.statement, DiscourseType::Statement),
            (self.command, DiscourseType::Command),
            (self.response, DiscourseType::Response),
            (self.creative, DiscourseType::Creative),
            (self.unknown, DiscourseType::Unknown),
        ];
        pairs.into_iter().max_by_key(|(count, _)| *count).map(|(_, dt)| dt)
    }

    /// Increments the count for the given discourse type.
    fn increment(&mut self, dt: &DiscourseType) {
        match dt {
            DiscourseType::Query => self.query += 1,
            DiscourseType::Statement => self.statement += 1,
            DiscourseType::Command => self.command += 1,
            DiscourseType::Response => self.response += 1,
            DiscourseType::Creative => self.creative += 1,
            DiscourseType::Unknown => self.unknown += 1,
        }
    }
}

/// Computes the mean gamma for a single event.
///
/// The event gamma is the mean of all non-zero `gamma_scores` entries.
/// Returns `0.0` if all slots are inactive.
fn event_gamma(event: &LearningEvent) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for &g in &event.gamma_scores[..MAX_SLOTS] {
        if g > 0.0 {
            sum += g;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

/// Computes aggregated statistics for a single strand from a slice of events.
///
/// Only events whose `strand_id` matches are included.
///
/// # Example
///
/// ```
/// use volt_learn::LearningEvent;
/// use volt_learn::stats::compute_strand_stats;
/// use volt_core::meta::DiscourseType;
/// use volt_core::MAX_SLOTS;
///
/// let events = vec![
///     LearningEvent {
///         frame_id: 1, strand_id: 5,
///         query_type: DiscourseType::Query,
///         gamma_scores: [0.9; MAX_SLOTS],
///         convergence_iterations: 8,
///         ghost_activations: 2,
///         timestamp: 1000,
///     },
/// ];
/// let stats = compute_strand_stats(&events, 5);
/// assert_eq!(stats.query_count, 1);
/// ```
pub fn compute_strand_stats(events: &[LearningEvent], strand_id: u64) -> StrandStatistics {
    let strand_events: Vec<&LearningEvent> = events
        .iter()
        .filter(|e| e.strand_id == strand_id)
        .collect();

    let query_count = strand_events.len();

    if query_count == 0 {
        return StrandStatistics {
            strand_id,
            query_count: 0,
            average_gamma: 0.0,
            average_iterations: 0.0,
            topic_distribution: TopicDistribution::default(),
        };
    }

    let gamma_sum: f32 = strand_events.iter().map(|e| event_gamma(e)).sum();
    let iter_sum: u64 = strand_events
        .iter()
        .map(|e| e.convergence_iterations as u64)
        .sum();

    let mut topic_distribution = TopicDistribution::default();
    for event in &strand_events {
        topic_distribution.increment(&event.query_type);
    }

    StrandStatistics {
        strand_id,
        query_count,
        average_gamma: gamma_sum / query_count as f32,
        average_iterations: iter_sum as f32 / query_count as f32,
        topic_distribution,
    }
}

/// Computes statistics for all strands present in the events.
///
/// Returns a map from strand ID to [`StrandStatistics`].
///
/// # Example
///
/// ```
/// use volt_learn::LearningEvent;
/// use volt_learn::stats::compute_all_strand_stats;
/// use volt_core::meta::DiscourseType;
/// use volt_core::MAX_SLOTS;
///
/// let events = vec![
///     LearningEvent {
///         frame_id: 1, strand_id: 1,
///         query_type: DiscourseType::Query,
///         gamma_scores: [0.9; MAX_SLOTS],
///         convergence_iterations: 8,
///         ghost_activations: 2,
///         timestamp: 1000,
///     },
///     LearningEvent {
///         frame_id: 2, strand_id: 2,
///         query_type: DiscourseType::Command,
///         gamma_scores: [0.7; MAX_SLOTS],
///         convergence_iterations: 12,
///         ghost_activations: 1,
///         timestamp: 2000,
///     },
/// ];
/// let all_stats = compute_all_strand_stats(&events);
/// assert_eq!(all_stats.len(), 2);
/// assert!(all_stats.contains_key(&1));
/// assert!(all_stats.contains_key(&2));
/// ```
pub fn compute_all_strand_stats(events: &[LearningEvent]) -> HashMap<u64, StrandStatistics> {
    let mut strand_ids: Vec<u64> = events.iter().map(|e| e.strand_id).collect();
    strand_ids.sort_unstable();
    strand_ids.dedup();

    strand_ids
        .into_iter()
        .map(|sid| (sid, compute_strand_stats(events, sid)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event_with(
        strand_id: u64,
        query_type: DiscourseType,
        gamma: f32,
        iterations: u32,
    ) -> LearningEvent {
        LearningEvent {
            frame_id: 0,
            strand_id,
            query_type,
            gamma_scores: [gamma; MAX_SLOTS],
            convergence_iterations: iterations,
            ghost_activations: 0,
            timestamp: 0,
        }
    }

    #[test]
    fn empty_events_yields_zero_stats() {
        let stats = compute_strand_stats(&[], 1);
        assert_eq!(stats.query_count, 0);
        assert_eq!(stats.average_gamma, 0.0);
        assert_eq!(stats.average_iterations, 0.0);
    }

    #[test]
    fn single_event_stats_correct() {
        let events = vec![make_event_with(1, DiscourseType::Query, 0.9, 10)];
        let stats = compute_strand_stats(&events, 1);
        assert_eq!(stats.query_count, 1);
        assert!((stats.average_gamma - 0.9).abs() < 0.01);
        assert!((stats.average_iterations - 10.0).abs() < 0.01);
        assert_eq!(stats.topic_distribution.query, 1);
    }

    #[test]
    fn multi_event_average_gamma() {
        let events = vec![
            make_event_with(1, DiscourseType::Query, 0.6, 10),
            make_event_with(1, DiscourseType::Query, 0.8, 20),
        ];
        let stats = compute_strand_stats(&events, 1);
        assert_eq!(stats.query_count, 2);
        // mean gamma = (0.6 + 0.8) / 2 = 0.7
        assert!((stats.average_gamma - 0.7).abs() < 0.01);
        // mean iterations = (10 + 20) / 2 = 15
        assert!((stats.average_iterations - 15.0).abs() < 0.01);
    }

    #[test]
    fn topic_distribution_reflects_usage() {
        let events = vec![
            make_event_with(1, DiscourseType::Command, 0.9, 5),
            make_event_with(1, DiscourseType::Command, 0.9, 5),
            make_event_with(1, DiscourseType::Command, 0.9, 5),
            make_event_with(1, DiscourseType::Query, 0.5, 15),
        ];
        let stats = compute_strand_stats(&events, 1);
        assert_eq!(stats.topic_distribution.command, 3);
        assert_eq!(stats.topic_distribution.query, 1);
        assert_eq!(stats.topic_distribution.dominant(), Some(DiscourseType::Command));
    }

    #[test]
    fn multiple_strands_independent() {
        let events = vec![
            make_event_with(1, DiscourseType::Query, 0.9, 5),
            make_event_with(2, DiscourseType::Command, 0.5, 20),
            make_event_with(1, DiscourseType::Query, 0.8, 8),
        ];
        let all = compute_all_strand_stats(&events);
        assert_eq!(all.len(), 2);
        assert_eq!(all[&1].query_count, 2);
        assert_eq!(all[&2].query_count, 1);
    }

    #[test]
    fn proportions_sum_to_one() {
        let dist = TopicDistribution {
            query: 3,
            command: 5,
            creative: 2,
            ..Default::default()
        };
        let props = dist.proportions();
        let sum: f32 = props.values().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn proportions_empty_returns_empty_map() {
        let dist = TopicDistribution::default();
        assert!(dist.proportions().is_empty());
    }

    #[test]
    fn dominant_empty_returns_none() {
        let dist = TopicDistribution::default();
        assert_eq!(dist.dominant(), None);
    }

    #[test]
    fn event_gamma_with_inactive_slots() {
        let mut event = make_event_with(1, DiscourseType::Query, 0.0, 10);
        // Only two active slots
        event.gamma_scores[0] = 0.8;
        event.gamma_scores[1] = 0.6;
        let gamma = event_gamma(&event);
        assert!((gamma - 0.7).abs() < 0.01);
    }

    #[test]
    fn event_gamma_all_inactive() {
        let event = make_event_with(1, DiscourseType::Query, 0.0, 10);
        assert_eq!(event_gamma(&event), 0.0);
    }
}
