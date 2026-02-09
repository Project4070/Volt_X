//! # volt-learn
//!
//! Continual learning engine for Volt X.
//!
//! ## Three Timescales of Learning
//!
//! - **Instant Learning** (ms–min): Strand vector updates in RAM, no GPU needed
//! - **Sleep Consolidation** (hours): Forward-Forward weight updates during idle
//! - **Developmental Growth** (days–months): Strand graduation + module hot-plug
//!
//! ## Key Components
//!
//! - Learning event accumulator (tracks what to consolidate)
//! - Forward-Forward training loop (layer-local, low VRAM)
//! - Replay buffer generation from strands
//! - Strand clustering and graduation
//!
//! ## Architecture Rules
//!
//! - Depends on `volt-core`, `volt-bus`, `volt-db`, `volt-soft`.
//! - Forward-Forward training uses same VRAM budget as inference.
//! - No backpropagation — layer-local updates only.

pub use volt_core;

// MILESTONE: 6.1 — Instant learning (strand updates)
// TODO: Implement learning event logging
// TODO: Implement strand vector update mechanism
// TODO: Implement sleep consolidation scheduler
// TODO: Implement Forward-Forward training loop (Phase 2+)
