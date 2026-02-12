//! The [`HardStrand`] trait — pluggable CPU-side deterministic tools.
//!
//! Each Hard Strand registers a capability vector and processes frames
//! that the Intent Router determines are relevant to its capability.
//!
//! ## Building a Module
//!
//! To create a new Hard Strand module:
//!
//! 1. Implement `HardStrand` for your struct.
//! 2. Generate a capability vector via a seeded hash (see [`MathEngine`](crate::math_engine::MathEngine) for the pattern).
//! 3. Set a threshold (0.3 is typical).
//! 4. Register with the Intent Router via `router.register(Box::new(your_strand))`.
//! 5. Optionally implement `info()` to provide module metadata.

use volt_core::{ModuleInfo, TensorFrame, VoltError, SLOT_DIM};

/// A pluggable deterministic computation module for the CPU Hard Core.
///
/// Each Hard Strand:
/// - Declares a **capability vector** the Intent Router uses for routing.
/// - Declares a **similarity threshold** below which it declines activation.
/// - Processes a [`TensorFrame`], modifying specific slots with exact results.
/// - Returns a result with certainty gamma = 1.0 for exact computations.
///
/// # Example
///
/// ```
/// use volt_hard::strand::{HardStrand, StrandResult};
/// use volt_core::{TensorFrame, VoltError, SLOT_DIM};
///
/// struct EchoStrand;
///
/// impl HardStrand for EchoStrand {
///     fn name(&self) -> &str { "echo" }
///     fn capability_vector(&self) -> &[f32; SLOT_DIM] {
///         static V: [f32; SLOT_DIM] = [0.0; SLOT_DIM];
///         &V
///     }
///     fn threshold(&self) -> f32 { 0.5 }
///     fn process(&self, frame: &TensorFrame) -> Result<StrandResult, VoltError> {
///         Ok(StrandResult {
///             frame: frame.clone(),
///             activated: false,
///             description: "echo: no-op".to_string(),
///         })
///     }
/// }
/// ```
pub trait HardStrand: Send + Sync {
    /// Human-readable name for this strand (e.g., "math_engine").
    fn name(&self) -> &str;

    /// The 256-dimensional capability vector used for routing.
    ///
    /// The Intent Router computes cosine similarity between incoming frame
    /// slots and this vector to decide whether to activate this strand.
    fn capability_vector(&self) -> &[f32; SLOT_DIM];

    /// Minimum cosine similarity required for this strand to activate.
    ///
    /// If the best matching slot has similarity below this threshold,
    /// the strand declines activation. Typical range: 0.3–0.7.
    fn threshold(&self) -> f32;

    /// Process a frame and return the result.
    ///
    /// The strand may modify specific slots in the frame. Slots produced
    /// by exact computation should have gamma = 1.0 and source = HardCore.
    fn process(&self, frame: &TensorFrame) -> Result<StrandResult, VoltError>;

    /// Optional metadata about this module.
    ///
    /// Returns `None` by default. Community modules should override this
    /// to provide introspectable metadata for the module registry.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_hard::strand::HardStrand;
    /// use volt_hard::math_engine::MathEngine;
    ///
    /// let engine = MathEngine::new();
    /// // Built-in strands return None by default.
    /// assert!(engine.info().is_none());
    /// ```
    fn info(&self) -> Option<ModuleInfo> {
        None
    }
}

/// The result of a [`HardStrand`] processing a frame.
///
/// # Example
///
/// ```
/// use volt_hard::strand::StrandResult;
/// use volt_core::TensorFrame;
///
/// let result = StrandResult {
///     frame: TensorFrame::new(),
///     activated: false,
///     description: "no-op".to_string(),
/// };
/// assert!(!result.activated);
/// ```
#[derive(Debug, Clone)]
pub struct StrandResult {
    /// The (potentially modified) frame after strand processing.
    pub frame: TensorFrame,

    /// Whether this strand actually performed computation.
    ///
    /// `false` means the strand inspected the frame but found nothing
    /// it could handle (pass-through).
    pub activated: bool,

    /// Human-readable description of what the strand did.
    ///
    /// Used for proof chain construction and logging.
    pub description: String,
}
