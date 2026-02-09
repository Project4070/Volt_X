//! Frame-level metadata for TensorFrames.
//!
//! [`FrameMeta`] holds information that applies to the entire frame,
//! such as which strand it belongs to, the discourse type, and
//! the global certainty score.

/// Frame-level metadata.
///
/// # Example
///
/// ```
/// use volt_core::meta::FrameMeta;
///
/// let meta = FrameMeta::default();
/// assert_eq!(meta.strand_id, 0);
/// assert_eq!(meta.global_certainty, 0.0);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
pub struct FrameMeta {
    /// Unique frame identifier.
    pub frame_id: u64,

    /// The strand this frame belongs to.
    pub strand_id: u64,

    /// Global certainty score (gamma, γ). Typically min(slot γ values).
    pub global_certainty: f32,

    /// The discourse type of this frame.
    pub discourse_type: DiscourseType,

    /// Timestamp (microseconds since epoch) when this frame was created.
    pub created_at: u64,

    /// Number of RAR iterations used to produce this frame.
    pub rar_iterations: u32,

    /// Whether this frame has been verified by the CPU Hard Core.
    pub verified: bool,

    /// Proof chain length (number of reasoning steps).
    pub proof_length: u32,
}

impl Default for FrameMeta {
    fn default() -> Self {
        Self {
            frame_id: 0,
            strand_id: 0,
            global_certainty: 0.0,
            discourse_type: DiscourseType::Unknown,
            created_at: 0,
            rar_iterations: 0,
            verified: false,
            proof_length: 0,
        }
    }
}

/// The discourse type classifies the frame's communicative purpose.
///
/// # Example
///
/// ```
/// use volt_core::meta::DiscourseType;
///
/// let dt = DiscourseType::Query;
/// assert_ne!(dt, DiscourseType::Statement);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
pub enum DiscourseType {
    /// A question from the user.
    Query,
    /// A declarative statement.
    Statement,
    /// A command or request.
    Command,
    /// A response to a query.
    Response,
    /// A creative/generative output.
    Creative,
    /// Unknown or not yet classified.
    Unknown,
}
