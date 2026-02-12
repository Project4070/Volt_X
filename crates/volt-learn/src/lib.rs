//! # volt-learn
//!
//! Continual learning engine for Volt X.
//!
//! ## Milestone 5.1: Learning Event Logging
//!
//! - [`LearningEvent`] — diagnostic data from a single inference run
//! - [`EventBuffer`] — bounded accumulator for learning events
//! - [`EventLogger`] — main API: logging, statistics, persistence
//! - [`StrandStatistics`] — per-strand aggregated usage data
//!
//! ## Three Timescales of Learning
//!
//! - **Instant Learning** (ms–min): Strand vector updates in RAM, no GPU needed
//! - **Sleep Consolidation** (hours): Forward-Forward weight updates during idle
//! - **Developmental Growth** (days–months): Strand graduation + module hot-plug
//!
//! ## Architecture Rules
//!
//! - Depends on `volt-core`, `volt-bus`, `volt-db`, `volt-soft`.
//! - Forward-Forward training uses same VRAM budget as inference.
//! - No backpropagation — layer-local updates only.
//! - No async code — pure synchronous logic.

pub mod event;
pub mod buffer;
pub mod stats;
pub mod logger;

pub use event::LearningEvent;
pub use buffer::{EventBuffer, DEFAULT_BUFFER_CAPACITY};
pub use logger::{EventLogger, LoggerConfig};
pub use stats::{StrandStatistics, TopicDistribution};

pub use volt_core;

// TODO(5.2): Implement sleep consolidation scheduler
// TODO(5.2): Implement Forward-Forward training loop
// TODO(5.2): Implement strand graduation
// TODO(5.3): Implement RLVF joint alignment
