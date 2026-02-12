//! Event logger — the primary API for learning event management.
//!
//! [`EventLogger`] wraps an [`EventBuffer`](crate::buffer::EventBuffer)
//! with persistence (save/load) and on-demand statistics computation.

use std::collections::HashMap;
use std::path::Path;

use crate::buffer::{EventBuffer, DEFAULT_BUFFER_CAPACITY};
use crate::event::LearningEvent;
use crate::stats::{compute_all_strand_stats, compute_strand_stats, StrandStatistics};
use volt_core::VoltError;

/// Configuration for the event logger.
///
/// # Example
///
/// ```
/// use volt_learn::LoggerConfig;
///
/// let config = LoggerConfig::default();
/// assert_eq!(config.buffer_capacity, 10_000);
/// ```
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    /// Maximum events in the buffer before FIFO eviction. Default: 10,000.
    pub buffer_capacity: usize,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: DEFAULT_BUFFER_CAPACITY,
        }
    }
}

/// The event logger — accumulates learning events, computes
/// statistics, and persists events across restarts.
///
/// # Example
///
/// ```
/// use volt_learn::{EventLogger, LearningEvent};
/// use volt_core::meta::DiscourseType;
/// use volt_core::MAX_SLOTS;
///
/// let mut logger = EventLogger::new();
/// logger.log(LearningEvent {
///     frame_id: 1,
///     strand_id: 0,
///     query_type: DiscourseType::Query,
///     gamma_scores: [0.8; MAX_SLOTS],
///     convergence_iterations: 5,
///     ghost_activations: 3,
///     timestamp: 1_000_000,
/// });
/// assert_eq!(logger.event_count(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct EventLogger {
    buffer: EventBuffer,
}

impl Default for EventLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl EventLogger {
    /// Creates a new logger with default configuration (capacity 10,000).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::EventLogger;
    ///
    /// let logger = EventLogger::new();
    /// assert_eq!(logger.event_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            buffer: EventBuffer::new(),
        }
    }

    /// Creates a new logger with custom configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::{EventLogger, LoggerConfig};
    ///
    /// let config = LoggerConfig { buffer_capacity: 500 };
    /// let logger = EventLogger::with_config(config);
    /// assert_eq!(logger.event_count(), 0);
    /// ```
    pub fn with_config(config: LoggerConfig) -> Self {
        Self {
            buffer: EventBuffer::with_capacity(config.buffer_capacity),
        }
    }

    /// Logs a learning event into the buffer.
    ///
    /// If the buffer is at capacity, the oldest event is evicted.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::{EventLogger, LearningEvent};
    /// use volt_core::meta::DiscourseType;
    /// use volt_core::MAX_SLOTS;
    ///
    /// let mut logger = EventLogger::new();
    /// logger.log(LearningEvent {
    ///     frame_id: 1,
    ///     strand_id: 0,
    ///     query_type: DiscourseType::Query,
    ///     gamma_scores: [0.0; MAX_SLOTS],
    ///     convergence_iterations: 1,
    ///     ghost_activations: 0,
    ///     timestamp: 0,
    /// });
    /// assert_eq!(logger.event_count(), 1);
    /// ```
    pub fn log(&mut self, event: LearningEvent) {
        self.buffer.push(event);
    }

    /// Returns the number of events currently in the buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::EventLogger;
    ///
    /// let logger = EventLogger::new();
    /// assert_eq!(logger.event_count(), 0);
    /// ```
    pub fn event_count(&self) -> usize {
        self.buffer.len()
    }

    /// Returns a read-only slice of all events in the buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::EventLogger;
    ///
    /// let logger = EventLogger::new();
    /// assert!(logger.events().is_empty());
    /// ```
    pub fn events(&self) -> &[LearningEvent] {
        self.buffer.events()
    }

    /// Returns references to events belonging to the given strand.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::EventLogger;
    ///
    /// let logger = EventLogger::new();
    /// assert!(logger.events_for_strand(1).is_empty());
    /// ```
    pub fn events_for_strand(&self, strand_id: u64) -> Vec<&LearningEvent> {
        self.buffer.events_for_strand(strand_id)
    }

    /// Computes statistics for a single strand from the current buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::{EventLogger, LearningEvent};
    /// use volt_core::meta::DiscourseType;
    /// use volt_core::MAX_SLOTS;
    ///
    /// let mut logger = EventLogger::new();
    /// logger.log(LearningEvent {
    ///     frame_id: 1, strand_id: 5,
    ///     query_type: DiscourseType::Query,
    ///     gamma_scores: [0.9; MAX_SLOTS],
    ///     convergence_iterations: 8,
    ///     ghost_activations: 2,
    ///     timestamp: 1000,
    /// });
    /// let stats = logger.strand_stats(5);
    /// assert_eq!(stats.query_count, 1);
    /// ```
    pub fn strand_stats(&self, strand_id: u64) -> StrandStatistics {
        compute_strand_stats(self.buffer.events(), strand_id)
    }

    /// Computes statistics for all strands present in the buffer.
    ///
    /// Returns a map from strand ID to [`StrandStatistics`].
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::EventLogger;
    ///
    /// let logger = EventLogger::new();
    /// assert!(logger.all_strand_stats().is_empty());
    /// ```
    pub fn all_strand_stats(&self) -> HashMap<u64, StrandStatistics> {
        compute_all_strand_stats(self.buffer.events())
    }

    /// Drains all events from the buffer, returning them as a `Vec`.
    ///
    /// The logger is empty after this call.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::EventLogger;
    ///
    /// let mut logger = EventLogger::new();
    /// let events = logger.drain();
    /// assert!(events.is_empty());
    /// ```
    pub fn drain(&mut self) -> Vec<LearningEvent> {
        self.buffer.drain()
    }

    /// Removes all events from the buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_learn::EventLogger;
    ///
    /// let mut logger = EventLogger::new();
    /// logger.clear();
    /// assert_eq!(logger.event_count(), 0);
    /// ```
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Serializes the event buffer to a JSON file on disk.
    ///
    /// Runs on a dedicated thread with 8 MB stack for Windows
    /// compatibility (matching the `StrandStore::save` pattern).
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::LearnError`] if serialization or file I/O fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::EventLogger;
    /// use std::path::Path;
    ///
    /// let logger = EventLogger::new();
    /// logger.save(Path::new("learning_events.json")).unwrap();
    /// ```
    pub fn save(&self, path: &Path) -> Result<(), VoltError> {
        let events: Vec<LearningEvent> = self.buffer.events().to_vec();
        let path = path.to_owned();
        std::thread::Builder::new()
            .name("learn-save".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let file =
                    std::fs::File::create(&path).map_err(|e| VoltError::LearnError {
                        message: format!(
                            "failed to create learning events file {}: {e}",
                            path.display()
                        ),
                    })?;
                let writer = std::io::BufWriter::new(file);
                serde_json::to_writer(writer, &events).map_err(|e| VoltError::LearnError {
                    message: format!("failed to serialize learning events: {e}"),
                })
            })
            .map_err(|e| VoltError::LearnError {
                message: format!("failed to spawn save thread: {e}"),
            })?
            .join()
            .map_err(|_| VoltError::LearnError {
                message: "save thread panicked".to_string(),
            })?
    }

    /// Loads a logger from a JSON file on disk.
    ///
    /// Deserialization runs on a dedicated thread with 8 MB stack for
    /// Windows compatibility (matching the `StrandStore::load` pattern).
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::LearnError`] if file I/O or deserialization fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_learn::EventLogger;
    /// use std::path::Path;
    ///
    /// let logger = EventLogger::load(Path::new("learning_events.json")).unwrap();
    /// ```
    pub fn load(path: &Path) -> Result<Self, VoltError> {
        let path = path.to_owned();
        let events: Vec<LearningEvent> = std::thread::Builder::new()
            .name("learn-load".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let file =
                    std::fs::File::open(&path).map_err(|e| VoltError::LearnError {
                        message: format!(
                            "failed to open learning events file {}: {e}",
                            path.display()
                        ),
                    })?;
                let reader = std::io::BufReader::new(file);
                serde_json::from_reader(reader).map_err(|e| VoltError::LearnError {
                    message: format!("failed to deserialize learning events: {e}"),
                })
            })
            .map_err(|e| VoltError::LearnError {
                message: format!("failed to spawn load thread: {e}"),
            })?
            .join()
            .map_err(|_| VoltError::LearnError {
                message: "load thread panicked".to_string(),
            })??;

        let mut logger = Self::new();
        for event in events {
            logger.buffer.push(event);
        }
        Ok(logger)
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
            gamma_scores: [0.8; MAX_SLOTS],
            convergence_iterations: 10,
            ghost_activations: 2,
            timestamp: frame_id * 1000,
        }
    }

    #[test]
    fn new_logger_is_empty() {
        let logger = EventLogger::new();
        assert_eq!(logger.event_count(), 0);
        assert!(logger.events().is_empty());
    }

    #[test]
    fn log_adds_event() {
        let mut logger = EventLogger::new();
        logger.log(make_event(1, 0));
        assert_eq!(logger.event_count(), 1);
        assert_eq!(logger.events()[0].frame_id, 1);
    }

    #[test]
    fn strand_stats_computed_correctly() {
        let mut logger = EventLogger::new();
        logger.log(make_event(1, 5));
        logger.log(make_event(2, 5));
        logger.log(make_event(3, 10));

        let stats_5 = logger.strand_stats(5);
        assert_eq!(stats_5.query_count, 2);

        let stats_10 = logger.strand_stats(10);
        assert_eq!(stats_10.query_count, 1);

        let all = logger.all_strand_stats();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn drain_empties_logger() {
        let mut logger = EventLogger::new();
        logger.log(make_event(1, 0));
        logger.log(make_event(2, 0));
        let events = logger.drain();
        assert_eq!(events.len(), 2);
        assert_eq!(logger.event_count(), 0);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "volt_learn_logger_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("learning_events.json");

        // Save
        {
            let mut logger = EventLogger::new();
            for i in 0..25u64 {
                logger.log(make_event(i + 1, i % 3));
            }
            logger.save(&path).expect("save");
        }

        // Load
        {
            let logger = EventLogger::load(&path).expect("load");
            assert_eq!(logger.event_count(), 25);
            assert_eq!(logger.events()[0].frame_id, 1);
            assert_eq!(logger.events()[24].frame_id, 25);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_is_same_as_new() {
        let logger = EventLogger::default();
        assert_eq!(logger.event_count(), 0);
    }

    #[test]
    fn with_config_sets_capacity() {
        let config = LoggerConfig { buffer_capacity: 5 };
        let mut logger = EventLogger::with_config(config);
        for i in 0..10u64 {
            logger.log(make_event(i + 1, 0));
        }
        // Only last 5 should remain
        assert_eq!(logger.event_count(), 5);
        assert_eq!(logger.events()[0].frame_id, 6);
    }
}
