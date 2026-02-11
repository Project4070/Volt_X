//! ProofConstructor — records Hard Strand processing steps into proof chains.
//!
//! The ProofConstructor is **not** a [`HardStrand`](crate::strand::HardStrand).
//! It is pipeline infrastructure that accumulates a proof chain as strands
//! process a frame. Each step records which strand executed, what it did,
//! the routing similarity score, and the certainty (gamma) after that step.
//!
//! # Example
//!
//! ```
//! use volt_hard::proof_constructor::ProofConstructor;
//!
//! let mut proof = ProofConstructor::new();
//! proof.record_step("math_engine", "847 * 392 = 332024", 0.95, 1.0, true);
//! proof.record_certainty_propagation(0.8);
//!
//! let chain = proof.build(0.8);
//! assert!(chain.len() >= 2);
//! assert_eq!(chain.activated_count, 2);
//! ```

use serde::{Deserialize, Serialize};

use crate::router::RoutingDecision;

/// A single step in a proof chain.
///
/// Records what a strand did during frame processing, including
/// the routing similarity that triggered it and the certainty after.
///
/// # Example
///
/// ```
/// use volt_hard::proof_constructor::ProofStep;
///
/// let step = ProofStep {
///     strand_name: "math_engine".to_string(),
///     description: "100 + 200 = 300".to_string(),
///     similarity: 0.92,
///     gamma_after: 1.0,
///     activated: true,
/// };
/// assert!(step.activated);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStep {
    /// Name of the strand that was evaluated.
    pub strand_name: String,

    /// Human-readable description of what the strand did.
    pub description: String,

    /// The cosine similarity that triggered routing to this strand.
    pub similarity: f32,

    /// The frame certainty (gamma) after this step completed.
    pub gamma_after: f32,

    /// Whether the strand actually activated and performed computation.
    pub activated: bool,
}

/// A complete proof chain recording all Hard Strand processing for a frame.
///
/// # Example
///
/// ```
/// use volt_hard::proof_constructor::ProofChain;
///
/// let chain = ProofChain {
///     steps: vec![],
///     final_gamma: 0.0,
///     activated_count: 0,
/// };
/// assert!(chain.is_empty());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofChain {
    /// Ordered list of proof steps.
    pub steps: Vec<ProofStep>,

    /// Final global certainty after all steps.
    pub final_gamma: f32,

    /// Total number of steps where a strand actually activated.
    pub activated_count: usize,
}

impl ProofChain {
    /// Returns the number of steps in the proof chain.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::proof_constructor::ProofChain;
    ///
    /// let chain = ProofChain {
    ///     steps: vec![],
    ///     final_gamma: 0.0,
    ///     activated_count: 0,
    /// };
    /// assert_eq!(chain.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns `true` if the proof chain has no steps.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::proof_constructor::ProofChain;
    ///
    /// let chain = ProofChain {
    ///     steps: vec![],
    ///     final_gamma: 0.0,
    ///     activated_count: 0,
    /// };
    /// assert!(chain.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Constructs proof chains by recording Hard Strand processing steps.
///
/// Create a new `ProofConstructor`, record steps as strands execute,
/// then call [`build`](Self::build) to produce a finalized [`ProofChain`].
///
/// # Example
///
/// ```
/// use volt_hard::proof_constructor::ProofConstructor;
///
/// let mut proof = ProofConstructor::new();
/// proof.record_step("math_engine", "1 + 2 = 3", 0.9, 1.0, true);
/// proof.record_certainty_propagation(0.8);
///
/// let chain = proof.build(0.8);
/// assert_eq!(chain.len(), 2);
/// assert_eq!(chain.activated_count, 2);
/// ```
#[derive(Debug, Clone)]
pub struct ProofConstructor {
    steps: Vec<ProofStep>,
}

impl ProofConstructor {
    /// Creates a new empty ProofConstructor.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::proof_constructor::ProofConstructor;
    ///
    /// let proof = ProofConstructor::new();
    /// let chain = proof.build(0.0);
    /// assert!(chain.is_empty());
    /// ```
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Returns a read-only reference to the accumulated steps.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::proof_constructor::ProofConstructor;
    ///
    /// let mut proof = ProofConstructor::new();
    /// proof.record_step("s1", "thing", 0.9, 1.0, true);
    /// assert_eq!(proof.steps().len(), 1);
    /// ```
    pub fn steps(&self) -> &[ProofStep] {
        &self.steps
    }

    /// Record a proof step from a strand execution.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::proof_constructor::ProofConstructor;
    ///
    /// let mut proof = ProofConstructor::new();
    /// proof.record_step("math_engine", "2 + 3 = 5", 0.9, 1.0, true);
    ///
    /// let chain = proof.build(1.0);
    /// assert_eq!(chain.len(), 1);
    /// assert_eq!(chain.steps[0].strand_name, "math_engine");
    /// ```
    pub fn record_step(
        &mut self,
        strand_name: &str,
        description: &str,
        similarity: f32,
        gamma_after: f32,
        activated: bool,
    ) {
        self.steps.push(ProofStep {
            strand_name: strand_name.to_string(),
            description: description.to_string(),
            similarity,
            gamma_after,
            activated,
        });
    }

    /// Record a step from a [`RoutingDecision`] and strand result description.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::proof_constructor::ProofConstructor;
    /// use volt_hard::router::RoutingDecision;
    ///
    /// let mut proof = ProofConstructor::new();
    /// let decision = RoutingDecision {
    ///     strand_name: "math_engine".to_string(),
    ///     slot_index: 1,
    ///     similarity: 0.95,
    ///     activated: true,
    /// };
    /// proof.record_from_decision(&decision, "math_engine: 1 + 1 = 2", 1.0);
    ///
    /// let chain = proof.build(1.0);
    /// assert_eq!(chain.steps[0].similarity, 0.95);
    /// ```
    pub fn record_from_decision(
        &mut self,
        decision: &RoutingDecision,
        description: &str,
        gamma_after: f32,
    ) {
        self.record_step(
            &decision.strand_name,
            description,
            decision.similarity,
            gamma_after,
            decision.activated,
        );
    }

    /// Record a CertaintyEngine propagation step.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::proof_constructor::ProofConstructor;
    ///
    /// let mut proof = ProofConstructor::new();
    /// proof.record_certainty_propagation(0.6);
    ///
    /// let chain = proof.build(0.6);
    /// assert_eq!(chain.steps[0].strand_name, "certainty_engine");
    /// assert!(chain.steps[0].activated);
    /// ```
    pub fn record_certainty_propagation(&mut self, global_gamma: f32) {
        self.steps.push(ProofStep {
            strand_name: "certainty_engine".to_string(),
            description: format!("min-rule propagation: global_gamma = {global_gamma:.4}"),
            similarity: 1.0,
            gamma_after: global_gamma,
            activated: true,
        });
    }

    /// Finalize and return the proof chain.
    ///
    /// Consumes the constructor and produces a [`ProofChain`].
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::proof_constructor::ProofConstructor;
    ///
    /// let mut proof = ProofConstructor::new();
    /// proof.record_step("s1", "did thing", 0.9, 1.0, true);
    /// proof.record_step("s2", "skipped", 0.2, 1.0, false);
    ///
    /// let chain = proof.build(1.0);
    /// assert_eq!(chain.len(), 2);
    /// assert_eq!(chain.activated_count, 1);
    /// ```
    pub fn build(self, final_gamma: f32) -> ProofChain {
        let activated_count = self.steps.iter().filter(|s| s.activated).count();
        ProofChain {
            steps: self.steps,
            final_gamma,
            activated_count,
        }
    }

    /// Reset the constructor for reuse.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::proof_constructor::ProofConstructor;
    ///
    /// let mut proof = ProofConstructor::new();
    /// proof.record_step("s1", "thing", 0.9, 1.0, true);
    /// proof.reset();
    ///
    /// let chain = proof.build(0.0);
    /// assert!(chain.is_empty());
    /// ```
    pub fn reset(&mut self) {
        self.steps.clear();
    }
}

impl Default for ProofConstructor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_constructor_new_is_empty() {
        let proof = ProofConstructor::new();
        let chain = proof.build(0.0);
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert_eq!(chain.activated_count, 0);
    }

    #[test]
    fn proof_constructor_record_step() {
        let mut proof = ProofConstructor::new();
        proof.record_step("math_engine", "1 + 2 = 3", 0.92, 1.0, true);

        let chain = proof.build(1.0);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.steps[0].strand_name, "math_engine");
        assert_eq!(chain.steps[0].description, "1 + 2 = 3");
        assert!((chain.steps[0].similarity - 0.92).abs() < 0.01);
        assert!((chain.steps[0].gamma_after - 1.0).abs() < 0.01);
        assert!(chain.steps[0].activated);
    }

    #[test]
    fn proof_constructor_activated_count() {
        let mut proof = ProofConstructor::new();
        proof.record_step("s1", "activated", 0.9, 1.0, true);
        proof.record_step("s2", "skipped", 0.1, 0.5, false);
        proof.record_step("s3", "activated", 0.8, 1.0, true);

        let chain = proof.build(0.5);
        assert_eq!(chain.len(), 3);
        assert_eq!(chain.activated_count, 2);
    }

    #[test]
    fn proof_constructor_record_from_decision() {
        let mut proof = ProofConstructor::new();
        let decision = RoutingDecision {
            strand_name: "math_engine".to_string(),
            slot_index: 1,
            similarity: 0.95,
            activated: true,
        };
        proof.record_from_decision(&decision, "847 * 392 = 332024", 1.0);

        let chain = proof.build(1.0);
        assert_eq!(chain.steps[0].strand_name, "math_engine");
        assert!((chain.steps[0].similarity - 0.95).abs() < 0.01);
        assert!(chain.steps[0].activated);
    }

    #[test]
    fn proof_constructor_record_certainty_propagation() {
        let mut proof = ProofConstructor::new();
        proof.record_certainty_propagation(0.6);

        let chain = proof.build(0.6);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.steps[0].strand_name, "certainty_engine");
        assert!(chain.steps[0].activated);
        assert!((chain.steps[0].gamma_after - 0.6).abs() < 0.01);
    }

    #[test]
    fn proof_constructor_full_chain() {
        let mut proof = ProofConstructor::new();
        proof.record_step("math_engine", "computed 2+2=4", 0.92, 1.0, true);
        proof.record_certainty_propagation(0.8);

        let chain = proof.build(0.8);
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.activated_count, 2); // both activated
        assert!((chain.final_gamma - 0.8).abs() < 0.01);
    }

    #[test]
    fn proof_constructor_reset() {
        let mut proof = ProofConstructor::new();
        proof.record_step("s1", "thing", 0.9, 1.0, true);
        assert_eq!(proof.steps.len(), 1);

        proof.reset();
        let chain = proof.build(0.0);
        assert!(chain.is_empty());
    }

    #[test]
    fn proof_chain_final_gamma() {
        let mut proof = ProofConstructor::new();
        proof.record_step("s1", "thing", 0.9, 0.5, true);

        let chain = proof.build(0.42);
        assert!((chain.final_gamma - 0.42).abs() < 0.01);
    }

    #[test]
    fn proof_constructor_default_trait() {
        let proof = ProofConstructor::default();
        let chain = proof.build(0.0);
        assert!(chain.is_empty());
    }
}
