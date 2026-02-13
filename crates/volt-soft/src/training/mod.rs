//! VFN training via Flow Matching.
//!
//! Trains the Vector Field Network to predict drift directions between
//! question and answer TensorFrame pairs using linear interpolation paths.
//!
//! ## Algorithm
//!
//! For each training step:
//! 1. Sample a batch of (F_q, F_a) frame pairs
//! 2. Sample t ~ Uniform(0, 1) for each pair
//! 3. Compute interpolated state: `F(t) = (1-t)·F_q + t·F_a`
//! 4. Target drift: `F_a - F_q` (constant velocity field)
//! 5. Predicted drift: `VFN(F(t))`
//! 6. Loss: MSE(predicted, target)
//! 7. Backprop + AdamW step
//!
//! ## Phase 2 Training
//!
//! The [`phase2`] module extends basic flow matching with:
//! - Scaled VFN (50M+ params) with deep residual architecture
//! - Joint attention weight training
//! - Curriculum-based progressive training
//! - Validation metrics (cosine similarity, directional accuracy)

pub mod flow_matching;
pub mod phase2;

pub use flow_matching::{
    generate_synthetic_pairs, train_vfn_flow_matching, FlowMatchConfig, FramePair, TrainResult,
};

pub use phase2::{
    train_phase2, CurriculumStage, Phase2Config, Phase2Result, ValidationMetrics,
};
