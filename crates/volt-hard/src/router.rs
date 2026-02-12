//! Intent Router — routes TensorFrame slots to Hard Strands by cosine similarity.
//!
//! The Intent Router is the "dispatcher" of the CPU Hard Core. It receives a
//! frame from the Soft Core, computes cosine similarity between frame slots
//! and each registered Hard Strand's capability vector, and routes to the
//! best match above threshold.
//!
//! ## Routing Algorithm
//!
//! 1. For each registered strand, compute similarity between the strand's
//!    capability vector and every active slot at R0 (discourse resolution).
//! 2. Find the (strand, slot) pair with the highest similarity.
//! 3. If the best similarity exceeds the strand's threshold, activate it.
//! 4. The strand processes the frame and returns the result.
//! 5. Record the routing decision in the proof chain.
//!
//! If no strand exceeds threshold, the frame passes through unchanged.

use std::panic::{catch_unwind, AssertUnwindSafe};

use volt_bus::similarity;
use volt_core::{TensorFrame, VoltError, MAX_SLOTS, SLOT_DIM};

use crate::strand::HardStrand;

/// A routing decision made by the Intent Router.
///
/// Records which strand was selected, which slot triggered it,
/// and the cosine similarity score.
///
/// # Example
///
/// ```
/// use volt_hard::router::RoutingDecision;
///
/// let decision = RoutingDecision {
///     strand_name: "math_engine".to_string(),
///     slot_index: 6,
///     similarity: 0.85,
///     activated: true,
/// };
/// assert!(decision.activated);
/// ```
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Name of the strand that was selected (or "none").
    pub strand_name: String,

    /// The slot index that best matched the strand's capability.
    pub slot_index: usize,

    /// The cosine similarity score between slot and capability vector.
    pub similarity: f32,

    /// Whether the strand was actually activated (similarity >= threshold).
    pub activated: bool,
}

/// The result of the full Hard Core pipeline (router + strand execution).
///
/// # Example
///
/// ```
/// use volt_hard::router::{RouterResult, RoutingDecision};
/// use volt_core::TensorFrame;
///
/// let result = RouterResult {
///     frame: TensorFrame::new(),
///     decisions: vec![],
/// };
/// assert!(result.decisions.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct RouterResult {
    /// The frame after all strand processing.
    pub frame: TensorFrame,

    /// The routing decisions made (one per registered strand evaluated).
    pub decisions: Vec<RoutingDecision>,
}

/// The Intent Router dispatches frames to Hard Strands by vector similarity.
///
/// # Example
///
/// ```
/// use volt_hard::router::IntentRouter;
/// use volt_hard::math_engine::MathEngine;
/// use volt_core::TensorFrame;
///
/// std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
///     let mut router = IntentRouter::new();
///     router.register(Box::new(MathEngine::new()));
///     let frame = TensorFrame::new();
///     let result = router.route(&frame).unwrap();
///     assert!(result.decisions.is_empty() || !result.decisions[0].activated);
/// }).unwrap().join().unwrap();
/// ```
pub struct IntentRouter {
    strands: Vec<Box<dyn HardStrand>>,
}

impl IntentRouter {
    /// Creates an empty Intent Router with no registered strands.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::router::IntentRouter;
    ///
    /// let router = IntentRouter::new();
    /// assert_eq!(router.strand_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            strands: Vec::new(),
        }
    }

    /// Register a Hard Strand with this router.
    ///
    /// Strands are evaluated in order of their best similarity score,
    /// not registration order.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::router::IntentRouter;
    /// use volt_hard::math_engine::MathEngine;
    ///
    /// let mut router = IntentRouter::new();
    /// router.register(Box::new(MathEngine::new()));
    /// assert_eq!(router.strand_count(), 1);
    /// ```
    pub fn register(&mut self, strand: Box<dyn HardStrand>) {
        self.strands.push(strand);
    }

    /// Returns the number of registered strands.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::router::IntentRouter;
    ///
    /// let router = IntentRouter::new();
    /// assert_eq!(router.strand_count(), 0);
    /// ```
    pub fn strand_count(&self) -> usize {
        self.strands.len()
    }

    /// Unregister a strand by name.
    ///
    /// Returns `true` if a strand with the given name was found and removed,
    /// `false` if no strand with that name was registered.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::router::IntentRouter;
    /// use volt_hard::math_engine::MathEngine;
    ///
    /// let mut router = IntentRouter::new();
    /// router.register(Box::new(MathEngine::new()));
    /// assert_eq!(router.strand_count(), 1);
    /// assert!(router.unregister("math_engine"));
    /// assert_eq!(router.strand_count(), 0);
    /// assert!(!router.unregister("nonexistent"));
    /// ```
    pub fn unregister(&mut self, name: &str) -> bool {
        let before = self.strands.len();
        self.strands.retain(|s| s.name() != name);
        self.strands.len() < before
    }

    /// List the names of all registered strands.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::router::IntentRouter;
    /// use volt_hard::math_engine::MathEngine;
    ///
    /// let mut router = IntentRouter::new();
    /// router.register(Box::new(MathEngine::new()));
    /// let names = router.strand_names();
    /// assert_eq!(names, vec!["math_engine"]);
    /// ```
    pub fn strand_names(&self) -> Vec<&str> {
        self.strands.iter().map(|s| s.name()).collect()
    }

    /// Route a frame through the Hard Core pipeline.
    ///
    /// Computes cosine similarity between each strand's capability vector
    /// and all active frame slots at R0. Routes to the best-matching strand
    /// above threshold.
    ///
    /// If no strand activates, the frame passes through unchanged.
    ///
    /// # Errors
    ///
    /// Returns `Err(VoltError)` if a strand's `process()` fails.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::router::IntentRouter;
    /// use volt_hard::math_engine::MathEngine;
    /// use volt_hard::strand::HardStrand;
    /// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
    ///
    /// std::thread::Builder::new().stack_size(4 * 1024 * 1024).spawn(|| {
    ///     let mut router = IntentRouter::new();
    ///     let engine = MathEngine::new();
    ///     let cap = *engine.capability_vector();
    ///     router.register(Box::new(engine));
    ///
    ///     let mut frame = TensorFrame::new();
    ///     let mut pred = SlotData::new(SlotRole::Predicate);
    ///     pred.write_resolution(0, cap);
    ///     frame.write_slot(1, pred).unwrap();
    ///     frame.meta[1].certainty = 0.8;
    ///
    ///     let mut inst = SlotData::new(SlotRole::Instrument);
    ///     let mut data = [0.0_f32; SLOT_DIM];
    ///     data[0] = 1.0; // ADD
    ///     data[1] = 10.0;
    ///     data[2] = 20.0;
    ///     inst.write_resolution(0, data);
    ///     frame.write_slot(6, inst).unwrap();
    ///     frame.meta[6].certainty = 0.9;
    ///
    ///     let result = router.route(&frame).unwrap();
    ///     assert!(!result.decisions.is_empty());
    /// }).unwrap().join().unwrap();
    /// ```
    pub fn route(&self, frame: &TensorFrame) -> Result<RouterResult, VoltError> {
        if self.strands.is_empty() {
            return Ok(RouterResult {
                frame: frame.clone(),
                decisions: vec![],
            });
        }

        // Collect active slot R0 vectors
        let slot_vectors: Vec<(usize, &[f32; SLOT_DIM])> = (0..MAX_SLOTS)
            .filter_map(|i| {
                frame.slots[i]
                    .as_ref()
                    .and_then(|slot| slot.resolutions[0].as_ref().map(|v| (i, v)))
            })
            .collect();

        if slot_vectors.is_empty() {
            return Ok(RouterResult {
                frame: frame.clone(),
                decisions: vec![],
            });
        }

        // Find the best (strand_index, slot_index, similarity) across all strands
        let mut best_strand_idx: Option<usize> = None;
        let mut best_slot_idx: usize = 0;
        let mut best_sim: f32 = f32::NEG_INFINITY;

        for (strand_idx, strand) in self.strands.iter().enumerate() {
            let cap = strand.capability_vector();
            for &(slot_idx, slot_vec) in &slot_vectors {
                let sim = similarity(cap, slot_vec);
                tracing::debug!(
                    "Strand '{}' vs slot {}: sim={:.4}, cap=[{:.4},{:.4},{:.4}...], slot=[{:.4},{:.4},{:.4}...]",
                    strand.name(), slot_idx, sim,
                    cap[0], cap[1], cap[2],
                    slot_vec[0], slot_vec[1], slot_vec[2]
                );
                if sim > best_sim {
                    best_sim = sim;
                    best_strand_idx = Some(strand_idx);
                    best_slot_idx = slot_idx;
                }
            }
        }

        // Check if the best match exceeds the strand's threshold
        let mut decisions = Vec::new();
        let result_frame;

        if let Some(strand_idx) = best_strand_idx {
            let strand = &self.strands[strand_idx];
            let threshold = strand.threshold();

            if best_sim >= threshold {
                // Activate the strand with panic safety.
                // If a buggy module panics, we catch it, log an error,
                // and pass the frame through unchanged rather than
                // crashing the entire server.
                let strand_name = strand.name().to_string();
                let catch_result = catch_unwind(AssertUnwindSafe(|| strand.process(frame)));

                match catch_result {
                    Ok(Ok(strand_result)) => {
                        decisions.push(RoutingDecision {
                            strand_name,
                            slot_index: best_slot_idx,
                            similarity: best_sim,
                            activated: strand_result.activated,
                        });
                        result_frame = strand_result.frame;
                    }
                    Ok(Err(e)) => {
                        // Strand returned an error — propagate it.
                        return Err(e);
                    }
                    Err(panic_payload) => {
                        // Strand panicked — log and continue with frame unchanged.
                        let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                            (*s).to_string()
                        } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "unknown panic".to_string()
                        };
                        tracing::error!(
                            strand = %strand_name,
                            panic = %panic_msg,
                            "Hard strand panicked during processing — skipping"
                        );
                        decisions.push(RoutingDecision {
                            strand_name,
                            slot_index: best_slot_idx,
                            similarity: best_sim,
                            activated: false,
                        });
                        result_frame = frame.clone();
                    }
                }
            } else {
                // Below threshold — pass through
                decisions.push(RoutingDecision {
                    strand_name: strand.name().to_string(),
                    slot_index: best_slot_idx,
                    similarity: best_sim,
                    activated: false,
                });

                result_frame = frame.clone();
            }
        } else {
            result_frame = frame.clone();
        }

        Ok(RouterResult {
            frame: result_frame,
            decisions,
        })
    }
}

impl Default for IntentRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_engine::MathEngine;
    use crate::strand::HardStrand;
    use volt_core::{SlotData, SlotRole};

    #[test]
    fn router_new_has_no_strands() {
        let router = IntentRouter::new();
        assert_eq!(router.strand_count(), 0);
    }

    #[test]
    fn router_register_increments_count() {
        let mut router = IntentRouter::new();
        router.register(Box::new(MathEngine::new()));
        assert_eq!(router.strand_count(), 1);
    }

    #[test]
    fn router_empty_frame_passthrough() {
        let mut router = IntentRouter::new();
        router.register(Box::new(MathEngine::new()));

        let frame = TensorFrame::new();
        let result = router.route(&frame).unwrap();
        assert_eq!(result.frame.active_slot_count(), 0);
        assert!(result.decisions.is_empty());
    }

    #[test]
    fn router_no_strands_passthrough() {
        let router = IntentRouter::new();
        let frame = TensorFrame::new();
        let result = router.route(&frame).unwrap();
        assert_eq!(result.frame.active_slot_count(), 0);
    }

    #[test]
    fn router_activates_math_engine_on_capability_match() {
        // TensorFrame is ~65KB; catch_unwind on Windows SEH needs extra stack.
        std::thread::Builder::new()
            .stack_size(4 * 1024 * 1024)
            .spawn(|| {
                let mut router = IntentRouter::new();
                let engine = MathEngine::new();
                let cap = *engine.capability_vector();
                router.register(Box::new(engine));

                // Build frame with math capability vector in predicate slot
                // and operation data in instrument slot
                let mut frame = TensorFrame::new();

                // Tag predicate with math capability vector (high similarity)
                let mut pred = SlotData::new(SlotRole::Predicate);
                pred.write_resolution(0, cap);
                frame.write_slot(1, pred).unwrap();
                frame.meta[1].certainty = 0.8;

                // Put actual math operation in instrument slot
                let mut instrument = SlotData::new(SlotRole::Instrument);
                let mut data = [0.0_f32; SLOT_DIM];
                data[0] = 3.0; // MUL
                data[1] = 847.0;
                data[2] = 392.0;
                instrument.write_resolution(0, data);
                frame.write_slot(6, instrument).unwrap();
                frame.meta[6].certainty = 0.9;

                let result = router.route(&frame).unwrap();

                assert_eq!(result.decisions.len(), 1);
                assert!(result.decisions[0].activated);
                assert_eq!(result.decisions[0].strand_name, "math_engine");
                assert!(result.decisions[0].similarity > 0.9);

                // Verify the math was computed
                let r = result.frame.read_slot(8).unwrap();
                assert!(
                    (r.resolutions[0].unwrap()[0] - 332_024.0).abs() < 1.0,
                    "Should compute 847 * 392 = 332024, got {}",
                    r.resolutions[0].unwrap()[0]
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn router_non_math_frame_does_not_activate() {
        std::thread::Builder::new()
            .stack_size(4 * 1024 * 1024)
            .spawn(|| {
                let mut router = IntentRouter::new();
                router.register(Box::new(MathEngine::new()));

                // Build a frame with random (non-math) content
                let mut frame = TensorFrame::new();
                let mut agent = SlotData::new(SlotRole::Agent);
                // Use a vector that's very different from math capability
                let mut v = [0.0_f32; SLOT_DIM];
                // "Tell me about cats" — orthogonal to math capability
                for i in 0..SLOT_DIM {
                    v[i] = if i % 2 == 0 { 0.1 } else { -0.1 };
                }
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm;
                }
                agent.write_resolution(0, v);
                frame.write_slot(0, agent).unwrap();
                frame.meta[0].certainty = 0.8;

                let result = router.route(&frame).unwrap();

                // Either no decision, or decision with activated=false
                let activated = result.decisions.iter().any(|d| d.activated);
                assert!(!activated, "Non-math frame should not activate math engine");
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn router_preserves_frame_on_no_activation() {
        let mut router = IntentRouter::new();
        router.register(Box::new(MathEngine::new()));

        let mut frame = TensorFrame::new();
        let mut agent = SlotData::new(SlotRole::Agent);
        agent.write_resolution(0, [0.1; SLOT_DIM]);
        frame.write_slot(0, agent).unwrap();
        frame.meta[0].certainty = 0.7;

        let result = router.route(&frame).unwrap();

        assert_eq!(result.frame.active_slot_count(), frame.active_slot_count());
    }
}
