//! Unified error type for the Volt X workspace.
//!
//! All crates re-export [`VoltError`] for consistent error handling.
//! Error messages must include context:
//! `"failed to load strand #{id} from T1: {inner_error}"`

/// The unified error type for all Volt X operations.
///
/// # Example
///
/// ```
/// use volt_core::VoltError;
///
/// fn example_operation() -> Result<(), VoltError> {
///     Err(VoltError::SlotOutOfRange { index: 20, max: 16 })
/// }
///
/// let err = example_operation().unwrap_err();
/// assert!(err.to_string().contains("20"));
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum VoltError {
    /// Attempted to access a slot index beyond MAX_SLOTS.
    #[error("slot index {index} out of range (max {max})")]
    SlotOutOfRange {
        /// The invalid index that was requested.
        index: usize,
        /// The maximum valid index.
        max: usize,
    },

    /// Attempted to access a resolution index beyond NUM_RESOLUTIONS.
    #[error("resolution index {index} out of range (max {max})")]
    ResolutionOutOfRange {
        /// The invalid index that was requested.
        index: usize,
        /// The maximum valid index.
        max: usize,
    },

    /// The slot at the given index is empty (None).
    #[error("slot {index} is empty")]
    EmptySlot {
        /// The index of the empty slot.
        index: usize,
    },

    /// A frame operation failed.
    #[error("frame operation failed: {message}")]
    FrameError {
        /// Description of what went wrong.
        message: String,
    },

    /// A strand-related operation failed.
    #[error("strand operation failed for strand #{strand_id}: {message}")]
    StrandError {
        /// The strand identifier.
        strand_id: u64,
        /// Description of what went wrong.
        message: String,
    },

    /// A storage operation failed.
    #[error("storage error: {message}")]
    StorageError {
        /// Description of what went wrong.
        message: String,
    },

    /// A bus/algebra operation failed.
    #[error("bus operation failed: {message}")]
    BusError {
        /// Description of what went wrong.
        message: String,
    },

    /// A translate operation failed.
    #[error("translate error: {message}")]
    TranslateError {
        /// Description of what went wrong.
        message: String,
    },

    /// A safety invariant was violated.
    #[error("safety violation: {message}")]
    SafetyViolation {
        /// Description of the violation.
        message: String,
    },

    /// A learning or event-logging operation failed.
    #[error("learn error: {message}")]
    LearnError {
        /// Description of what went wrong.
        message: String,
    },

    /// A module operation failed (load, register, or execute).
    #[error("module '{name}' error: {message}")]
    ModuleError {
        /// The module identifier.
        name: String,
        /// Description of what went wrong.
        message: String,
    },

    /// An internal error that should not happen.
    #[error("internal error: {message}")]
    Internal {
        /// Description of the internal error.
        message: String,
    },
}
