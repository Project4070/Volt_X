//! HardCorePipeline — integrated Router + CertaintyEngine + ProofConstructor.
//!
//! The pipeline is the main entry point for Hard Core processing. It:
//! 1. Routes the frame to the best-matching strand via [`IntentRouter`]
//! 2. Records routing decisions in a [`ProofConstructor`]
//! 3. Propagates certainty via [`CertaintyEngine`] (min-rule)
//! 4. Returns the processed frame with a complete [`ProofChain`]
//!
//! # Example
//!
//! ```
//! use volt_hard::pipeline::HardCorePipeline;
//! use volt_hard::router::IntentRouter;
//! use volt_hard::math_engine::MathEngine;
//! use volt_hard::strand::HardStrand;
//! use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
//!
//! std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
//!     let mut router = IntentRouter::new();
//!     router.register(Box::new(MathEngine::new()));
//!     let pipeline = HardCorePipeline::new(router);
//!
//!     let mut frame = TensorFrame::new();
//!     // Set up a math frame...
//!     let engine = MathEngine::new();
//!     let cap = *engine.capability_vector();
//!
//!     let mut pred = SlotData::new(SlotRole::Predicate);
//!     pred.write_resolution(0, cap);
//!     frame.write_slot(1, pred).unwrap();
//!     frame.meta[1].certainty = 0.8;
//!
//!     let mut inst = SlotData::new(SlotRole::Instrument);
//!     let mut data = [0.0_f32; SLOT_DIM];
//!     data[0] = 1.0; // ADD
//!     data[1] = 10.0;
//!     data[2] = 20.0;
//!     inst.write_resolution(0, data);
//!     frame.write_slot(6, inst).unwrap();
//!     frame.meta[6].certainty = 0.9;
//!
//!     let result = pipeline.process(&frame).unwrap();
//!     assert!(result.proof.len() >= 2);
//! }).unwrap().join().unwrap();
//! ```

use volt_core::{TensorFrame, VoltError};

use crate::certainty_engine::CertaintyEngine;
use crate::proof_constructor::{ProofChain, ProofConstructor};
use crate::router::IntentRouter;

/// The result of the full Hard Core pipeline.
///
/// Contains the processed frame and the proof chain recording
/// all processing steps.
///
/// # Example
///
/// ```
/// use volt_hard::pipeline::PipelineResult;
/// use volt_hard::proof_constructor::ProofChain;
/// use volt_core::TensorFrame;
///
/// let result = PipelineResult {
///     frame: TensorFrame::new(),
///     proof: ProofChain {
///         steps: vec![],
///         final_gamma: 0.0,
///         activated_count: 0,
///     },
/// };
/// assert!(result.proof.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The processed frame after all strand execution and certainty propagation.
    pub frame: TensorFrame,

    /// The proof chain recording all processing steps.
    pub proof: ProofChain,
}

/// The Hard Core Pipeline: Router -> Strand Execution -> CertaintyEngine -> ProofChain.
///
/// This is the main entry point for Hard Core processing. It integrates
/// the [`IntentRouter`], [`CertaintyEngine`], and [`ProofConstructor`]
/// into a single processing pipeline.
///
/// # Example
///
/// ```
/// use volt_hard::pipeline::HardCorePipeline;
/// use volt_hard::router::IntentRouter;
///
/// let router = IntentRouter::new();
/// let pipeline = HardCorePipeline::new(router);
/// assert_eq!(pipeline.strand_count(), 0);
/// ```
pub struct HardCorePipeline {
    router: IntentRouter,
    certainty_engine: CertaintyEngine,
}

impl HardCorePipeline {
    /// Creates a new pipeline wrapping the given router.
    ///
    /// The pipeline automatically creates a [`CertaintyEngine`] for
    /// min-rule gamma propagation.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::pipeline::HardCorePipeline;
    /// use volt_hard::router::IntentRouter;
    /// use volt_hard::math_engine::MathEngine;
    ///
    /// let mut router = IntentRouter::new();
    /// router.register(Box::new(MathEngine::new()));
    /// let pipeline = HardCorePipeline::new(router);
    /// assert_eq!(pipeline.strand_count(), 1);
    /// ```
    pub fn new(router: IntentRouter) -> Self {
        Self {
            router,
            certainty_engine: CertaintyEngine::new(),
        }
    }

    /// Process a frame through the full Hard Core pipeline.
    ///
    /// 1. Route frame to best-matching strand via [`IntentRouter`]
    /// 2. Record routing decisions in proof chain
    /// 3. Propagate certainty via [`CertaintyEngine`] (min-rule)
    /// 4. Record certainty propagation in proof chain
    /// 5. Return processed frame with complete proof chain
    ///
    /// # Errors
    ///
    /// Returns `Err(VoltError)` if strand execution fails.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::pipeline::HardCorePipeline;
    /// use volt_hard::router::IntentRouter;
    /// use volt_core::TensorFrame;
    ///
    /// std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
    ///     let router = IntentRouter::new();
    ///     let pipeline = HardCorePipeline::new(router);
    ///     let frame = TensorFrame::new();
    ///
    ///     let result = pipeline.process(&frame).unwrap();
    ///     assert!(result.proof.is_empty() || result.proof.len() >= 1);
    /// }).unwrap().join().unwrap();
    /// ```
    pub fn process(&self, frame: &TensorFrame) -> Result<PipelineResult, VoltError> {
        let mut proof = ProofConstructor::new();

        // Step 1 & 2: Route and execute strand
        let router_result = self.router.route(frame)?;

        // Step 3: Record routing decisions in proof
        for decision in &router_result.decisions {
            let description = if decision.activated {
                format!(
                    "routed to {} (sim={:.4}, slot=S{})",
                    decision.strand_name, decision.similarity, decision.slot_index
                )
            } else {
                format!(
                    "{} below threshold (sim={:.4}, slot=S{})",
                    decision.strand_name, decision.similarity, decision.slot_index
                )
            };

            proof.record_from_decision(
                decision,
                &description,
                router_result.frame.frame_meta.global_certainty,
            );
        }

        // Step 4: Propagate certainty via min-rule
        let mut result_frame = router_result.frame;
        let certainty_result = self.certainty_engine.propagate(&mut result_frame);

        // Step 5: Record certainty propagation in proof
        proof.record_certainty_propagation(certainty_result.global_certainty);

        // Update proof_length to count activated steps
        let activated_count = proof
            .steps()
            .iter()
            .filter(|s| s.activated)
            .count();
        result_frame.frame_meta.proof_length = activated_count as u32;

        let chain = proof.build(certainty_result.global_certainty);

        Ok(PipelineResult {
            frame: result_frame,
            proof: chain,
        })
    }

    /// Returns a reference to the inner router.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::pipeline::HardCorePipeline;
    /// use volt_hard::router::IntentRouter;
    ///
    /// let router = IntentRouter::new();
    /// let pipeline = HardCorePipeline::new(router);
    /// assert_eq!(pipeline.router().strand_count(), 0);
    /// ```
    pub fn router(&self) -> &IntentRouter {
        &self.router
    }

    /// Returns a mutable reference to the inner router.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::pipeline::HardCorePipeline;
    /// use volt_hard::router::IntentRouter;
    /// use volt_hard::math_engine::MathEngine;
    ///
    /// let router = IntentRouter::new();
    /// let mut pipeline = HardCorePipeline::new(router);
    /// pipeline.router_mut().register(Box::new(MathEngine::new()));
    /// assert_eq!(pipeline.strand_count(), 1);
    /// ```
    pub fn router_mut(&mut self) -> &mut IntentRouter {
        &mut self.router
    }

    /// Returns the number of registered strands in the inner router.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::pipeline::HardCorePipeline;
    /// use volt_hard::router::IntentRouter;
    ///
    /// let router = IntentRouter::new();
    /// let pipeline = HardCorePipeline::new(router);
    /// assert_eq!(pipeline.strand_count(), 0);
    /// ```
    pub fn strand_count(&self) -> usize {
        self.router.strand_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_engine::MathEngine;
    use crate::strand::HardStrand;
    use volt_core::{SlotData, SlotRole, SLOT_DIM};

    /// Stack size for tests that allocate TensorFrames (each ~65KB).
    const TEST_STACK: usize = 4 * 1024 * 1024;

    fn make_pipeline() -> HardCorePipeline {
        let mut router = IntentRouter::new();
        router.register(Box::new(MathEngine::new()));
        HardCorePipeline::new(router)
    }

    fn make_math_frame(op: f32, left: f32, right: f32) -> TensorFrame {
        let engine = MathEngine::new();
        let cap = *engine.capability_vector();

        let mut frame = TensorFrame::new();

        let mut pred = SlotData::new(SlotRole::Predicate);
        pred.write_resolution(0, cap);
        frame.write_slot(1, pred).unwrap();
        frame.meta[1].certainty = 0.8;

        let mut instrument = SlotData::new(SlotRole::Instrument);
        let mut data = [0.0_f32; SLOT_DIM];
        data[0] = op;
        data[1] = left;
        data[2] = right;
        instrument.write_resolution(0, data);
        frame.write_slot(6, instrument).unwrap();
        frame.meta[6].certainty = 0.9;

        frame
    }

    #[test]
    fn pipeline_empty_frame_passthrough() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let pipeline = make_pipeline();
                let frame = TensorFrame::new();

                let result = pipeline.process(&frame).unwrap();

                assert_eq!(result.frame.active_slot_count(), 0);
                // Only certainty propagation step (no routing decisions for empty frame)
                assert_eq!(result.proof.len(), 1);
                assert_eq!(result.proof.steps[0].strand_name, "certainty_engine");
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn pipeline_math_produces_proof_chain() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let pipeline = make_pipeline();
                let frame = make_math_frame(1.0, 10.0, 20.0); // ADD

                let result = pipeline.process(&frame).unwrap();

                // Should have >= 2 steps: routing decision + certainty propagation
                assert!(
                    result.proof.len() >= 2,
                    "Proof chain should have >= 2 steps, got {}",
                    result.proof.len()
                );

                // Each step should have source (strand_name) and gamma
                for step in &result.proof.steps {
                    assert!(!step.strand_name.is_empty(), "Each step must have a source");
                    assert!(
                        step.gamma_after >= 0.0 && step.gamma_after <= 1.0,
                        "Each step must have valid gamma, got {}",
                        step.gamma_after
                    );
                }

                // First step should be routing to math_engine
                assert_eq!(result.proof.steps[0].strand_name, "math_engine");
                assert!(result.proof.steps[0].activated);

                // Last step should be certainty propagation
                let last = result.proof.steps.last().unwrap();
                assert_eq!(last.strand_name, "certainty_engine");
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn pipeline_certainty_matches_engine() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let pipeline = make_pipeline();
                let frame = make_math_frame(1.0, 10.0, 20.0); // ADD

                let result = pipeline.process(&frame).unwrap();

                // Global certainty should be min(0.8, 0.9, 1.0) = 0.8
                assert!(
                    (result.frame.frame_meta.global_certainty - 0.8).abs() < 0.01,
                    "pipeline certainty should be 0.8, got {}",
                    result.frame.frame_meta.global_certainty
                );
                assert!(
                    (result.proof.final_gamma - 0.8).abs() < 0.01,
                    "proof final_gamma should be 0.8, got {}",
                    result.proof.final_gamma
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn pipeline_strand_count() {
        let pipeline = make_pipeline();
        assert_eq!(pipeline.strand_count(), 1);
    }

    #[test]
    fn pipeline_router_accessors() {
        let mut pipeline = make_pipeline();
        assert_eq!(pipeline.router().strand_count(), 1);
        pipeline
            .router_mut()
            .register(Box::new(MathEngine::new()));
        assert_eq!(pipeline.strand_count(), 2);
    }

    #[test]
    fn pipeline_non_math_frame_still_has_certainty_step() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let pipeline = make_pipeline();

                let mut frame = TensorFrame::new();
                let mut agent = SlotData::new(SlotRole::Agent);
                agent.write_resolution(0, [0.1; SLOT_DIM]);
                frame.write_slot(0, agent).unwrap();
                frame.meta[0].certainty = 0.7;

                let result = pipeline.process(&frame).unwrap();

                // Should have routing decision (non-activated) + certainty propagation
                assert!(result.proof.len() >= 1);
                // Last step should be certainty engine
                let last = result.proof.steps.last().unwrap();
                assert_eq!(last.strand_name, "certainty_engine");
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn pipeline_proof_length_in_frame_meta() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let pipeline = make_pipeline();
                let frame = make_math_frame(1.0, 5.0, 3.0); // ADD

                let result = pipeline.process(&frame).unwrap();

                // proof_length should reflect activated steps
                assert!(
                    result.frame.frame_meta.proof_length >= 2,
                    "proof_length should be >= 2 (strand + certainty), got {}",
                    result.frame.frame_meta.proof_length
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
