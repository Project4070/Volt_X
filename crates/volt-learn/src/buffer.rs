//! Bounded event buffer for accumulating learning events.
//!
//! The [`EventBuffer`] holds learning events in memory until they
//! are flushed to disk or consumed by statistics computation.
//! When the buffer reaches its configured capacity, the oldest
//! events are dropped to make room.

use crate::event::LearningEvent;

/// Default maximum number of events in the buffer.
pub const DEFAULT_BUFFER_CAPACITY: usize = 10_000;

/// A bounded buffer that accumulates [`LearningEvent`]s.
///
/// Events are stored in insertion order. When the buffer reaches
/// its maximum capacity, the oldest event is removed to make room
/// (FIFO eviction). This ensures bounded memory usage while
/// preserving the most recent events.
///
/// # Example
///
/// ```
/// use volt_learn::EventBuffer;
/// use volt_learn::LearningEvent;
/// use volt_core::meta::DiscourseType;
/// use volt_core::MAX_SLOTS;
///
/// let mut buffer = EventBuffer::new();
/// let event = LearningEvent {
///     frame_id: 1,
///     strand_id: 0,
///     query_type: DiscourseType::Query,
///     gamma_scores: [0.8; MAX_SLOTS],
///     convergence_iterations: 5,
///     ghost_activations: 3,
///     timestamp: 1_000_000,
/// };
/// buffer.push(event);
/// assert_eq!(buffer.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct EventBuffer {
    events: Vec<LearningEvent>,
    capacity: usize,
}

impl Default for EventBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBuffer {
    /// Creates a new buffer with the default capacity (10,000).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::EventBuffer;
    ///
    /// let buffer = EventBuffer::new();
    /// assert!(buffer.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            capacity: DEFAULT_BUFFER_CAPACITY,
        }
    }

    /// Creates a new buffer with a custom capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::EventBuffer;
    ///
    /// let buffer = EventBuffer::with_capacity(500);
    /// assert!(buffer.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity.min(1024)),
            capacity,
        }
    }

    /// Pushes a learning event into the buffer.
    ///
    /// If the buffer is at capacity, the oldest event is removed first.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::{EventBuffer, LearningEvent};
    /// use volt_core::meta::DiscourseType;
    /// use volt_core::MAX_SLOTS;
    ///
    /// let mut buffer = EventBuffer::with_capacity(2);
    /// for i in 0..3 {
    ///     buffer.push(LearningEvent {
    ///         frame_id: i,
    ///         strand_id: 0,
    ///         query_type: DiscourseType::Query,
    ///         gamma_scores: [0.0; MAX_SLOTS],
    ///         convergence_iterations: 1,
    ///         ghost_activations: 0,
    ///         timestamp: 0,
    ///     });
    /// }
    /// // Oldest event (frame_id=0) was evicted
    /// assert_eq!(buffer.len(), 2);
    /// assert_eq!(buffer.events()[0].frame_id, 1);
    /// ```
    pub fn push(&mut self, event: LearningEvent) {
        if self.events.len() >= self.capacity {
            self.events.remove(0);
        }
        self.events.push(event);
    }

    /// Returns the number of events in the buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::EventBuffer;
    ///
    /// let buffer = EventBuffer::new();
    /// assert_eq!(buffer.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if the buffer contains no events.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::EventBuffer;
    ///
    /// let buffer = EventBuffer::new();
    /// assert!(buffer.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns a read-only slice of all events in the buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::EventBuffer;
    ///
    /// let buffer = EventBuffer::new();
    /// assert!(buffer.events().is_empty());
    /// ```
    pub fn events(&self) -> &[LearningEvent] {
        &self.events
    }

    /// Drains all events from the buffer, returning them as a `Vec`.
    ///
    /// The buffer is empty after this call.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::{EventBuffer, LearningEvent};
    /// use volt_core::meta::DiscourseType;
    /// use volt_core::MAX_SLOTS;
    ///
    /// let mut buffer = EventBuffer::new();
    /// buffer.push(LearningEvent {
    ///     frame_id: 1,
    ///     strand_id: 0,
    ///     query_type: DiscourseType::Query,
    ///     gamma_scores: [0.0; MAX_SLOTS],
    ///     convergence_iterations: 1,
    ///     ghost_activations: 0,
    ///     timestamp: 0,
    /// });
    /// let events = buffer.drain();
    /// assert_eq!(events.len(), 1);
    /// assert!(buffer.is_empty());
    /// ```
    pub fn drain(&mut self) -> Vec<LearningEvent> {
        std::mem::take(&mut self.events)
    }

    /// Returns references to events belonging to the given strand.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::{EventBuffer, LearningEvent};
    /// use volt_core::meta::DiscourseType;
    /// use volt_core::MAX_SLOTS;
    ///
    /// let mut buffer = EventBuffer::new();
    /// buffer.push(LearningEvent {
    ///     frame_id: 1, strand_id: 10,
    ///     query_type: DiscourseType::Query,
    ///     gamma_scores: [0.0; MAX_SLOTS],
    ///     convergence_iterations: 1,
    ///     ghost_activations: 0, timestamp: 0,
    /// });
    /// buffer.push(LearningEvent {
    ///     frame_id: 2, strand_id: 20,
    ///     query_type: DiscourseType::Query,
    ///     gamma_scores: [0.0; MAX_SLOTS],
    ///     convergence_iterations: 1,
    ///     ghost_activations: 0, timestamp: 0,
    /// });
    /// assert_eq!(buffer.events_for_strand(10).len(), 1);
    /// assert_eq!(buffer.events_for_strand(99).len(), 0);
    /// ```
    pub fn events_for_strand(&self, strand_id: u64) -> Vec<&LearningEvent> {
        self.events
            .iter()
            .filter(|e| e.strand_id == strand_id)
            .collect()
    }

    /// Removes all events from the buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::EventBuffer;
    ///
    /// let mut buffer = EventBuffer::new();
    /// buffer.clear();
    /// assert!(buffer.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::meta::DiscourseType;
    use volt_core::MAX_SLOTS;

    fn make_event(frame_id: u64, strand_id: u64) -> LearningEvent {
        LearningEvent {
            frame_id,
            strand_id,
            query_type: DiscourseType::Query,
            gamma_scores: [0.5; MAX_SLOTS],
            convergence_iterations: 10,
            ghost_activations: 2,
            timestamp: frame_id * 1000,
        }
    }

    #[test]
    fn empty_buffer_has_zero_len() {
        let buffer = EventBuffer::new();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn push_increments_len() {
        let mut buffer = EventBuffer::new();
        buffer.push(make_event(1, 0));
        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());
        buffer.push(make_event(2, 0));
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn push_100_events_has_100() {
        let mut buffer = EventBuffer::new();
        for i in 0..100 {
            buffer.push(make_event(i + 1, 0));
        }
        assert_eq!(buffer.len(), 100);
    }

    #[test]
    fn capacity_enforcement_drops_oldest() {
        let mut buffer = EventBuffer::with_capacity(3);
        for i in 0..5 {
            buffer.push(make_event(i + 1, 0));
        }
        assert_eq!(buffer.len(), 3);
        // Oldest two (frame_id 1, 2) were dropped
        assert_eq!(buffer.events()[0].frame_id, 3);
        assert_eq!(buffer.events()[1].frame_id, 4);
        assert_eq!(buffer.events()[2].frame_id, 5);
    }

    #[test]
    fn drain_empties_buffer() {
        let mut buffer = EventBuffer::new();
        buffer.push(make_event(1, 0));
        buffer.push(make_event(2, 0));
        let drained = buffer.drain();
        assert_eq!(drained.len(), 2);
        assert!(buffer.is_empty());
    }

    #[test]
    fn events_for_strand_filters_correctly() {
        let mut buffer = EventBuffer::new();
        buffer.push(make_event(1, 10));
        buffer.push(make_event(2, 20));
        buffer.push(make_event(3, 10));
        buffer.push(make_event(4, 30));

        let strand_10 = buffer.events_for_strand(10);
        assert_eq!(strand_10.len(), 2);
        assert_eq!(strand_10[0].frame_id, 1);
        assert_eq!(strand_10[1].frame_id, 3);

        assert_eq!(buffer.events_for_strand(20).len(), 1);
        assert_eq!(buffer.events_for_strand(99).len(), 0);
    }

    #[test]
    fn clear_removes_all() {
        let mut buffer = EventBuffer::new();
        buffer.push(make_event(1, 0));
        buffer.push(make_event(2, 0));
        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn default_is_same_as_new() {
        let buffer = EventBuffer::default();
        assert!(buffer.is_empty());
    }
}
