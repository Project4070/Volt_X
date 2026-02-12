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
//! ## Milestone 5.2: Sleep Consolidation
//!
//! - [`forward_forward`] — Forward-Forward VFN training (layer-local, no backprop)
//! - [`distillation`] — Frame distillation (clusters → wisdom frames)
//! - [`graduation`] — Strand graduation (novel topics → new strands)
//! - [`sleep`] — Sleep scheduler (idle detection, orchestration)
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

// Milestone 5.1: Learning Event Logging
pub mod event;
pub mod buffer;
pub mod stats;
pub mod logger;

// Milestone 5.2: Sleep Consolidation
pub mod forward_forward;
pub mod distillation;
pub mod graduation;
pub mod sleep;

// 5.1 re-exports
pub use event::LearningEvent;
pub use buffer::{EventBuffer, DEFAULT_BUFFER_CAPACITY};
pub use logger::{EventLogger, LoggerConfig};
pub use stats::{StrandStatistics, TopicDistribution};

// 5.2 re-exports
pub use forward_forward::{FfSample, FfConfig, FfResult, collect_ff_samples, train_ff};
pub use distillation::{DistillationConfig, DistillationResult, distill_all_strands, distill_strand};
pub use graduation::{GraduationConfig, GraduationResult, check_graduation};
pub use sleep::{SleepConfig, SleepScheduler, SleepHandle, SleepCycleResult};

pub use volt_core;

// TODO(5.3): Implement RLVF joint alignment
