//! Safety Layer — wraps the entire Soft Core → Hard Core pipeline.
//!
//! The SafetyLayer is the top-level integration point for Milestone 3.3.
//! It sits between frame input and the Hard Core pipeline, checking every
//! frame against the axiomatic invariants (K1-K5) before and after
//! processing.
//!
//! ## Processing Flow
//!
//! 1. **Pre-check**: Monitor incoming frame against axioms
//! 2. **Score**: Compute violation scores
//! 3. **Evaluate**: If Halt → Omega Veto → return safe default
//! 4. **Pipeline**: Pass to Hard Core pipeline for processing
//! 5. **Post-check**: Monitor result frame against axioms
//! 6. **Evaluate**: If Halt → Omega Veto → return safe default
//! 7. **Return**: Return processed frame with safety metadata
//!
//! # Example
//!
//! ```
//! use volt_safety::layer::SafetyLayer;
//! use volt_hard::pipeline::HardCorePipeline;
//! use volt_hard::default_pipeline;
//! use volt_core::TensorFrame;
//!
//! std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
//!     let pipeline = default_pipeline();
//!     let mut layer = SafetyLayer::new(pipeline);
//!
//!     let frame = TensorFrame::new();
//!     let result = layer.process(&frame).unwrap();
//!     assert!(!result.vetoed);
//! }).unwrap().join().unwrap();
//! ```

use volt_core::{TensorFrame, VoltError};
use volt_hard::pipeline::HardCorePipeline;

use crate::axiom::default_axioms;
use crate::monitor::TransitionMonitor;
use crate::scorer::{ScoringResult, ViolationScorer};
use crate::veto::{OmegaVeto, VetoLog};

/// The result of processing a frame through the safety-wrapped pipeline.
///
/// # Example
///
/// ```
/// use volt_safety::layer::SafetyResult;
/// use volt_hard::proof_constructor::ProofChain;
/// use volt_core::TensorFrame;
///
/// let result = SafetyResult {
///     frame: TensorFrame::new(),
///     proof: None,
///     vetoed: false,
///     veto_log: None,
///     pre_check_score: 0.0,
///     post_check_score: 0.0,
/// };
/// assert!(!result.vetoed);
/// ```
#[derive(Debug, Clone)]
pub struct SafetyResult {
    /// The output frame (safe default if vetoed, processed if not).
    pub frame: TensorFrame,

    /// The proof chain from the Hard Core pipeline (None if vetoed pre-check).
    pub proof: Option<volt_hard::proof_constructor::ProofChain>,

    /// Whether the Omega Veto was triggered.
    pub vetoed: bool,

    /// Audit log if the veto was triggered.
    pub veto_log: Option<VetoLog>,

    /// The aggregate violation score from the pre-check.
    pub pre_check_score: f32,

    /// The aggregate violation score from the post-check.
    pub post_check_score: f32,
}

/// The Safety Layer — wraps the full Soft Core → Hard Core pipeline.
///
/// Every frame passes through safety checks before and after the Hard
/// Core pipeline. Violations trigger the Omega Veto which cannot be
/// overridden by any neural component.
///
/// # Example
///
/// ```
/// use volt_safety::layer::SafetyLayer;
/// use volt_hard::default_pipeline;
/// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
///
/// std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
///     let pipeline = default_pipeline();
///     let mut layer = SafetyLayer::new(pipeline);
///
///     let mut frame = TensorFrame::new();
///     let mut slot = SlotData::new(SlotRole::Agent);
///     slot.write_resolution(0, [0.1; SLOT_DIM]);
///     frame.write_slot(0, slot).unwrap();
///     frame.meta[0].certainty = 0.8;
///
///     let result = layer.process(&frame).unwrap();
///     assert!(!result.vetoed);
/// }).unwrap().join().unwrap();
/// ```
pub struct SafetyLayer {
    pipeline: HardCorePipeline,
    monitor: TransitionMonitor,
    scorer: ViolationScorer,
    veto: OmegaVeto,
}

impl SafetyLayer {
    /// Creates a new SafetyLayer wrapping the given pipeline.
    ///
    /// Uses the default axioms (K1-K5) for safety checking.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::layer::SafetyLayer;
    /// use volt_hard::default_pipeline;
    ///
    /// let layer = SafetyLayer::new(default_pipeline());
    /// assert_eq!(layer.axiom_count(), 5);
    /// ```
    pub fn new(pipeline: HardCorePipeline) -> Self {
        Self {
            pipeline,
            monitor: TransitionMonitor::new(default_axioms()),
            scorer: ViolationScorer::new(),
            veto: OmegaVeto::new(),
        }
    }

    /// Creates a SafetyLayer with custom axioms.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::layer::SafetyLayer;
    /// use volt_safety::axiom::default_axioms;
    /// use volt_hard::default_pipeline;
    ///
    /// let layer = SafetyLayer::with_axioms(default_pipeline(), default_axioms());
    /// assert_eq!(layer.axiom_count(), 5);
    /// ```
    pub fn with_axioms(
        pipeline: HardCorePipeline,
        axioms: Vec<crate::axiom::Axiom>,
    ) -> Self {
        Self {
            pipeline,
            monitor: TransitionMonitor::new(axioms),
            scorer: ViolationScorer::new(),
            veto: OmegaVeto::new(),
        }
    }

    /// Returns the number of axioms loaded.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::layer::SafetyLayer;
    /// use volt_hard::default_pipeline;
    ///
    /// let layer = SafetyLayer::new(default_pipeline());
    /// assert_eq!(layer.axiom_count(), 5);
    /// ```
    pub fn axiom_count(&self) -> usize {
        self.monitor.axiom_count()
    }

    /// Returns the number of veto events that have occurred.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::layer::SafetyLayer;
    /// use volt_hard::default_pipeline;
    ///
    /// let layer = SafetyLayer::new(default_pipeline());
    /// assert_eq!(layer.veto_count(), 0);
    /// ```
    pub fn veto_count(&self) -> usize {
        self.veto.log_count()
    }

    /// Returns a reference to all veto logs for audit.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::layer::SafetyLayer;
    /// use volt_hard::default_pipeline;
    ///
    /// let layer = SafetyLayer::new(default_pipeline());
    /// assert!(layer.veto_logs().is_empty());
    /// ```
    pub fn veto_logs(&self) -> &[VetoLog] {
        self.veto.logs()
    }

    /// Returns a reference to the inner pipeline.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::layer::SafetyLayer;
    /// use volt_hard::default_pipeline;
    ///
    /// let layer = SafetyLayer::new(default_pipeline());
    /// assert!(layer.pipeline().strand_count() >= 2);
    /// ```
    pub fn pipeline(&self) -> &HardCorePipeline {
        &self.pipeline
    }

    /// Process a frame through the safety-wrapped pipeline.
    ///
    /// 1. Pre-check: Monitor frame → Score → Evaluate
    /// 2. If Halt → Omega Veto → return safe default
    /// 3. Pipeline: Route → Strand → Certainty → Proof
    /// 4. Post-check: Monitor result → Score → Evaluate
    /// 5. If Halt → Omega Veto → return safe default
    /// 6. Return processed frame with safety metadata
    ///
    /// # Errors
    ///
    /// Returns `Err(VoltError::SafetyViolation)` if a safety violation
    /// causes a veto. The error message includes violation details.
    /// Returns other `VoltError` variants if the pipeline fails.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::layer::SafetyLayer;
    /// use volt_hard::default_pipeline;
    /// use volt_core::TensorFrame;
    ///
    /// std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
    ///     let mut layer = SafetyLayer::new(default_pipeline());
    ///     let frame = TensorFrame::new();
    ///     let result = layer.process(&frame).unwrap();
    ///     assert!(!result.vetoed);
    /// }).unwrap().join().unwrap();
    /// ```
    pub fn process(&mut self, frame: &TensorFrame) -> Result<SafetyResult, VoltError> {
        // Step 1: Pre-check
        let pre_monitor = self.monitor.check_frame(frame);
        let pre_scoring = self.scorer.score(&pre_monitor);
        let pre_score = pre_scoring.aggregate_score;

        // Step 2: Evaluate pre-check
        if pre_scoring.requires_halt() {
            let veto_result = self.veto.fire(frame, &pre_scoring);
            return Ok(SafetyResult {
                frame: veto_result.safe_frame,
                proof: None,
                vetoed: true,
                veto_log: veto_result.log,
                pre_check_score: pre_score,
                post_check_score: 0.0,
            });
        }

        // Step 3: Process through pipeline
        let pipeline_result = self.pipeline.process(frame)?;

        // Step 4: Post-check
        let post_monitor = self.monitor.check_frame(&pipeline_result.frame);
        let post_scoring = self.scorer.score(&post_monitor);
        let post_score = post_scoring.aggregate_score;

        // Step 5: Evaluate post-check
        if post_scoring.requires_halt() {
            let veto_result = self.veto.fire(&pipeline_result.frame, &post_scoring);
            return Ok(SafetyResult {
                frame: veto_result.safe_frame,
                proof: Some(pipeline_result.proof),
                vetoed: true,
                veto_log: veto_result.log,
                pre_check_score: pre_score,
                post_check_score: post_score,
            });
        }

        // Step 6: Return safe result
        Ok(SafetyResult {
            frame: pipeline_result.frame,
            proof: Some(pipeline_result.proof),
            vetoed: false,
            veto_log: None,
            pre_check_score: pre_score,
            post_check_score: post_score,
        })
    }

    /// Check a frame without processing it through the pipeline.
    ///
    /// Useful for pre-screening frames before committing to full
    /// pipeline processing.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_safety::layer::SafetyLayer;
    /// use volt_hard::default_pipeline;
    /// use volt_core::TensorFrame;
    ///
    /// let layer = SafetyLayer::new(default_pipeline());
    /// let frame = TensorFrame::new();
    /// let scoring = layer.check(&frame);
    /// assert!(scoring.is_safe());
    /// ```
    pub fn check(&self, frame: &TensorFrame) -> ScoringResult {
        let monitor_result = self.monitor.check_frame(frame);
        self.scorer.score(&monitor_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axiom::default_axioms;
    use volt_core::{SlotData, SlotRole, SLOT_DIM};
    use volt_hard::default_pipeline;
    use volt_hard::math_engine::MathEngine;
    use volt_hard::strand::HardStrand;

    /// Stack size for tests with TensorFrames.
    /// Safety layer tests need extra stack because they wrap the full pipeline
    /// (router + strands + certainty + proof) with pre/post safety checks.
    const TEST_STACK: usize = 8 * 1024 * 1024;

    fn make_layer() -> SafetyLayer {
        SafetyLayer::new(default_pipeline())
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
    fn safety_layer_empty_frame_passes() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let mut layer = make_layer();
                let frame = TensorFrame::new();
                let result = layer.process(&frame).unwrap();

                assert!(!result.vetoed);
                assert!(result.veto_log.is_none());
                assert!((result.pre_check_score - 0.0).abs() < 1e-6);
                assert!((result.post_check_score - 0.0).abs() < 1e-6);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn safety_layer_normal_query_passes() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let mut layer = make_layer();

                let mut frame = TensorFrame::new();
                let mut slot = SlotData::new(SlotRole::Agent);
                slot.write_resolution(0, [0.1; SLOT_DIM]);
                frame.write_slot(0, slot).unwrap();
                frame.meta[0].certainty = 0.8;

                let result = layer.process(&frame).unwrap();
                assert!(!result.vetoed);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn safety_layer_math_query_passes() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let mut layer = make_layer();
                let frame = make_math_frame(1.0, 10.0, 20.0); // ADD

                let result = layer.process(&frame).unwrap();
                assert!(!result.vetoed);
                assert!(result.proof.is_some());
                assert!(result.proof.unwrap().len() >= 2);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn safety_layer_k1_violation_vetoed() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let axioms = default_axioms();
                let k1_vector = axioms[0].vector;
                let mut layer = make_layer();

                let mut frame = TensorFrame::new();
                let mut slot = SlotData::new(SlotRole::Predicate);
                slot.write_resolution(0, k1_vector);
                frame.write_slot(1, slot).unwrap();
                frame.meta[1].certainty = 0.9;

                let result = layer.process(&frame).unwrap();

                assert!(result.vetoed);
                assert!(result.frame.is_empty());
                assert!(!result.frame.frame_meta.verified);
                assert!(result.veto_log.is_some());

                let log = result.veto_log.unwrap();
                assert!(log.aggregate_score > 0.7);
                assert!(!log.violation_details.is_empty());
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn safety_layer_veto_log_includes_frame_state() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let axioms = default_axioms();
                let k1_vector = axioms[0].vector;
                let mut layer = make_layer();

                let mut frame = TensorFrame::new();
                let mut s0 = SlotData::new(SlotRole::Agent);
                s0.write_resolution(0, [0.1; SLOT_DIM]);
                frame.write_slot(0, s0).unwrap();
                frame.meta[0].certainty = 0.5;

                let mut s1 = SlotData::new(SlotRole::Predicate);
                s1.write_resolution(0, k1_vector);
                frame.write_slot(1, s1).unwrap();
                frame.meta[1].certainty = 0.9;

                let result = layer.process(&frame).unwrap();
                assert!(result.vetoed);

                let log = result.veto_log.unwrap();
                // Trigger frame should have the original slots
                assert_eq!(log.trigger_frame.active_slot_count(), 2);
                // Violation details should mention K1
                assert!(log.violation_details.iter().any(|d| d.contains("K1_harm")));
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn safety_layer_veto_count_increments() {
        std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                let axioms = default_axioms();
                let k1_vector = axioms[0].vector;
                let mut layer = make_layer();

                assert_eq!(layer.veto_count(), 0);

                let mut frame = TensorFrame::new();
                let mut slot = SlotData::new(SlotRole::Predicate);
                slot.write_resolution(0, k1_vector);
                frame.write_slot(1, slot).unwrap();

                layer.process(&frame).unwrap();
                assert_eq!(layer.veto_count(), 1);

                layer.process(&frame).unwrap();
                assert_eq!(layer.veto_count(), 2);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn safety_layer_check_without_processing() {
        let layer = make_layer();
        let frame = TensorFrame::new();
        let scoring = layer.check(&frame);
        assert!(scoring.is_safe());
    }

    #[test]
    fn safety_layer_axiom_count() {
        let layer = make_layer();
        assert_eq!(layer.axiom_count(), 5);
    }

    #[test]
    fn safety_layer_pipeline_accessible() {
        let layer = make_layer();
        assert!(layer.pipeline().strand_count() >= 2);
    }
}
