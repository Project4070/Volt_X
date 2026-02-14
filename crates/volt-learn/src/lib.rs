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
//! ## Milestone 5.3: RLVF Joint Alignment
//!
//! - [`eval_dataset`] — 1000 evaluation (question, answer) pairs
//! - [`reward`] — Reward computation from correctness + gamma calibration
//! - [`calibration`] — Expected Calibration Error (ECE) metric
//! - [`self_play`] — Logic puzzle generation and grading
//! - [`rlvf`] — REINFORCE with baseline training loop
//!
//! ## Phase 0: Code Training (Before B200)
//!
//! - [`code_dataset`] — Unified dataset loader for code problems (HumanEval, MBPP, APPS)
//! - [`stack_corpus`] — Streaming JSONL reader for The Stack dataset
//! - [`kmeans`] — Mini-batch k-means with k-means++ initialization
//! - [`codebook_init`] — Codebook initialization pipeline (encode → cluster → save)
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

// Phase 0: Code Training
pub mod code_dataset;
pub mod stack_corpus;
pub mod kmeans;
pub mod codebook_init;

// Phase 1: Translator Training
pub mod codesearchnet;
pub mod role_labels;
#[cfg(feature = "code-training")]
pub mod contrastive;

// Phase 2: VFN Training
#[cfg(feature = "vfn-training")]
pub mod code_pairs;

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

// Milestone 5.3: RLVF Joint Alignment
pub mod eval_dataset;
pub mod reward;
pub mod calibration;
pub mod self_play;
pub mod rlvf;

// 5.3 re-exports
pub use eval_dataset::{EvalCategory, EvalPair, generate_eval_dataset};
pub use reward::{RewardConfig, RewardOutcome, compute_reward};
pub use calibration::{CalibrationBin, CalibrationResult, compute_calibration};
pub use self_play::{PuzzleType, LogicPuzzle, PuzzleResult, generate_puzzles, grade_puzzle};
pub use rlvf::{RlvfConfig, RlvfResult, train_rlvf};

// Phase 0 re-exports
pub use code_dataset::{CodeProblem, CodeDataset};
pub use stack_corpus::{StackEntry, StackCorpusReader};
pub use kmeans::{KMeansConfig, KMeansResult, mini_batch_kmeans};
pub use codebook_init::{CodebookInitConfig, CodebookInitResult, init_codebook_from_corpus};

// Phase 1 re-exports
pub use codesearchnet::{CsnRecord, CsnDataset};
pub use role_labels::label_code_tokens;
