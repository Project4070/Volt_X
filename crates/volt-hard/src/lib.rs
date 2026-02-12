//! # volt-hard
//!
//! The CPU Hard Core — the "left brain" of Volt X.
//!
//! Implements deterministic, exact computation in pure Rust:
//! - **[`strand::HardStrand`]**: Pluggable trait for CPU-side tools
//! - **[`router::IntentRouter`]**: Routes frame slots to Hard Strands by cosine similarity
//! - **[`math_engine::MathEngine`]**: Exact arithmetic, algebra, basic calculus
//! - **[`hdc_algebra::HDCAlgebra`]**: Compositional reasoning via HDC operations
//! - **[`certainty_engine::CertaintyEngine`]**: Min-rule gamma propagation
//! - **[`proof_constructor::ProofConstructor`]**: Proof chain recording
//! - **[`pipeline::HardCorePipeline`]**: Integrated processing pipeline
//!
//! ## Architecture Rules
//!
//! - Pure CPU, no GPU code.
//! - No network code (network goes in `volt-ledger` or `volt-server`).
//! - Depends on `volt-core` and `volt-bus`.
//! - Hard Strands are hot-pluggable via `impl HardStrand` trait.
//!
//! ## Milestone 3.1: Intent Router + MathEngine
//!
//! The Intent Router receives a TensorFrame from the Soft Core, computes
//! cosine similarity against registered Hard Strand capability vectors,
//! and routes to the best match. The MathEngine handles exact arithmetic.
//!
//! ## Milestone 3.2: More Hard Strands + Pipeline
//!
//! Added HDCAlgebra for compositional reasoning, CodeRunner for sandboxed
//! WASM execution, CertaintyEngine for min-rule gamma propagation,
//! ProofConstructor for proof chain recording, and HardCorePipeline to
//! integrate them all.
//!
//! ## Usage
//!
//! ```
//! use volt_hard::pipeline::HardCorePipeline;
//! use volt_hard::math_engine::MathEngine;
//! use volt_hard::strand::HardStrand;
//! use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
//!
//! // TensorFrame is large (~65KB), so spawn a thread with adequate stack.
//! std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
//!     let pipeline = volt_hard::default_pipeline();
//!
//!     let engine = MathEngine::new();
//!     let math_cap = *engine.capability_vector();
//!
//!     let mut frame = TensorFrame::new();
//!     let mut pred = SlotData::new(SlotRole::Predicate);
//!     pred.write_resolution(0, math_cap);
//!     frame.write_slot(1, pred).unwrap();
//!     frame.meta[1].certainty = 0.8;
//!
//!     let mut inst = SlotData::new(SlotRole::Instrument);
//!     let mut data = [0.0_f32; SLOT_DIM];
//!     data[0] = 3.0; // MUL
//!     data[1] = 6.0;
//!     data[2] = 7.0;
//!     inst.write_resolution(0, data);
//!     frame.write_slot(6, inst).unwrap();
//!     frame.meta[6].certainty = 0.9;
//!
//!     let result = pipeline.process(&frame).unwrap();
//!     let r = result.frame.read_slot(8).unwrap();
//!     assert!((r.resolutions[0].unwrap()[0] - 42.0).abs() < 0.01);
//!     assert!(result.proof.len() >= 2);
//! }).unwrap().join().unwrap();
//! ```

pub use volt_core;

pub mod certainty_engine;
#[cfg(feature = "sandbox")]
pub mod code_runner;
pub mod hdc_algebra;
pub mod math_engine;
pub mod pipeline;
pub mod proof_constructor;
pub mod router;
pub mod strand;
#[cfg(feature = "weather")]
pub mod weather_strand;

use volt_core::{TensorFrame, VoltError};

/// Process a frame through the full Hard Core pipeline.
///
/// Routes to the best-matching strand, propagates certainty via
/// the min-rule, and builds a proof chain. This replaces the old
/// `verify_stub()` passthrough.
///
/// Internally spawns a thread with adequate stack (4 MB) because
/// the pipeline + TensorFrame copies require more stack than the
/// default thread size on some platforms.
///
/// # Example
///
/// ```
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
/// use volt_hard::verify_stub;
///
/// // TensorFrame is ~65KB; use a bigger stack for the doctest.
/// std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
///     let mut frame = TensorFrame::new();
///     frame.write_at(0, 0, SlotRole::Agent, [1.0; SLOT_DIM]).unwrap();
///
///     let result = verify_stub(&frame).unwrap();
///     assert_eq!(result.active_slot_count(), 1);
/// }).unwrap().join().unwrap();
/// ```
pub fn verify_stub(frame: &TensorFrame) -> Result<TensorFrame, VoltError> {
    let frame = Box::new(frame.clone());
    std::thread::Builder::new()
        .stack_size(4 * 1024 * 1024)
        .spawn(move || {
            let pipeline = default_pipeline();
            let result = pipeline.process(&frame)?;
            Ok(result.frame)
        })
        .map_err(|e| VoltError::FrameError {
            message: format!("failed to spawn pipeline thread: {e}"),
        })?
        .join()
        .map_err(|_| VoltError::FrameError {
            message: "pipeline thread panicked".to_string(),
        })?
}

/// Create a default Intent Router with all standard Hard Strands.
///
/// Registers MathEngine, HDCAlgebra, and (if the `sandbox` feature is
/// enabled) CodeRunner.
///
/// # Example
///
/// ```
/// use volt_hard::default_router;
///
/// let router = default_router();
/// assert!(router.strand_count() >= 2);
/// ```
pub fn default_router() -> router::IntentRouter {
    let mut router = router::IntentRouter::new();
    router.register(Box::new(math_engine::MathEngine::new()));
    router.register(Box::new(hdc_algebra::HDCAlgebra::new()));

    #[cfg(feature = "sandbox")]
    if let Ok(runner) = code_runner::CodeRunner::new() {
        router.register(Box::new(runner));
    }

    #[cfg(feature = "weather")]
    router.register(Box::new(weather_strand::WeatherStrand::new()));

    router
}

/// Create a default Hard Core pipeline with all standard strands.
///
/// Returns a [`HardCorePipeline`](pipeline::HardCorePipeline) pre-configured
/// with the standard router (MathEngine + HDCAlgebra + CodeRunner).
///
/// # Example
///
/// ```
/// use volt_hard::default_pipeline;
///
/// let pipeline = default_pipeline();
/// assert!(pipeline.strand_count() >= 2);
/// ```
pub fn default_pipeline() -> pipeline::HardCorePipeline {
    pipeline::HardCorePipeline::new(default_router())
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::{SlotData, SlotRole, SLOT_DIM};

    #[test]
    fn verify_stub_returns_frame() {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Predicate);
        slot.write_resolution(0, [0.7; SLOT_DIM]);
        frame.write_slot(1, slot).unwrap();
        frame.meta[1].certainty = 0.85;

        let result = verify_stub(&frame).unwrap();

        assert_eq!(result.active_slot_count(), frame.active_slot_count());
    }

    #[test]
    fn verify_stub_empty_frame() {
        let frame = TensorFrame::new();
        let result = verify_stub(&frame).unwrap();
        assert_eq!(result.active_slot_count(), 0);
    }

    #[test]
    fn default_router_has_strands() {
        let router = default_router();
        // At least MathEngine + HDCAlgebra
        assert!(
            router.strand_count() >= 2,
            "default_router should have >= 2 strands, got {}",
            router.strand_count()
        );
    }

    #[test]
    fn default_pipeline_has_strands() {
        let pipeline = default_pipeline();
        assert!(
            pipeline.strand_count() >= 2,
            "default_pipeline should have >= 2 strands, got {}",
            pipeline.strand_count()
        );
    }
}
