//! JSON request and response models for the HTTP API.

use serde::{Deserialize, Serialize};

/// Request body for `POST /api/think`.
///
/// # Example
///
/// ```
/// use volt_server::models::ThinkRequest;
///
/// let json = r#"{"text": "hello world"}"#;
/// let req: ThinkRequest = serde_json::from_str(json).unwrap();
/// assert_eq!(req.text, "hello world");
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ThinkRequest {
    /// The input text to process.
    pub text: String,
}

/// Response body for `POST /api/think`.
///
/// # Example
///
/// ```
/// use volt_server::models::{ThinkResponse, SlotState, ProofStepResponse, TimingMs};
///
/// let resp = ThinkResponse {
///     text: "cat sat mat.".into(),
///     gamma: vec![0.8, 0.8, 0.8],
///     strand_id: 0,
///     iterations: 1,
///     slot_states: vec![SlotState {
///         index: 0,
///         role: "Agent".into(),
///         word: "cat".into(),
///         certainty: 0.8,
///         source: "Translator".into(),
///         resolution_count: 1,
///     }],
///     proof_steps: vec![ProofStepResponse {
///         strand_name: "certainty_engine".into(),
///         description: "min-rule propagation".into(),
///         similarity: 1.0,
///         gamma_after: 0.8,
///         activated: true,
///     }],
///     safety_score: 0.0,
///     memory_frame_count: 1,
///     ghost_count: 0,
///     timing_ms: TimingMs { encode_ms: 0.1, decode_ms: 0.05, total_ms: 0.15 },
/// };
/// let json = serde_json::to_string(&resp).unwrap();
/// assert!(json.contains("cat sat mat"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkResponse {
    /// The decoded output text.
    pub text: String,
    /// Per-slot certainty (gamma) values for active slots.
    pub gamma: Vec<f32>,
    /// The strand ID.
    pub strand_id: u64,
    /// Number of RAR iterations performed by the Soft Core.
    pub iterations: u32,
    /// Per-slot debug state for all active slots.
    pub slot_states: Vec<SlotState>,
    /// Proof chain steps from the Hard Core pipeline.
    pub proof_steps: Vec<ProofStepResponse>,
    /// Pre-check safety score (0.0 = safe, higher = more violations).
    pub safety_score: f32,
    /// Total frames stored in memory (T0 + T1).
    pub memory_frame_count: usize,
    /// Number of ghost gists that influenced this RAR pass.
    pub ghost_count: usize,
    /// Timing breakdown in milliseconds.
    pub timing_ms: TimingMs,
}

/// A single step from the Hard Core proof chain.
///
/// # Example
///
/// ```
/// use volt_server::models::ProofStepResponse;
///
/// let step = ProofStepResponse {
///     strand_name: "math_engine".into(),
///     description: "10 + 20 = 30".into(),
///     similarity: 0.95,
///     gamma_after: 0.8,
///     activated: true,
/// };
/// let json = serde_json::to_string(&step).unwrap();
/// assert!(json.contains("math_engine"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStepResponse {
    /// Name of the strand that was evaluated.
    pub strand_name: String,
    /// Human-readable description of what the strand did.
    pub description: String,
    /// Cosine similarity that triggered routing to this strand.
    pub similarity: f32,
    /// Frame certainty (gamma) after this step completed.
    pub gamma_after: f32,
    /// Whether the strand actually activated and performed computation.
    pub activated: bool,
}

/// Debug information for a single active slot in the TensorFrame.
///
/// # Example
///
/// ```
/// use volt_server::models::SlotState;
///
/// let state = SlotState {
///     index: 0,
///     role: "Agent".into(),
///     word: "cat".into(),
///     certainty: 0.8,
///     source: "Translator".into(),
///     resolution_count: 1,
/// };
/// let json = serde_json::to_string(&state).unwrap();
/// assert!(json.contains("Agent"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotState {
    /// Slot index (0-15).
    pub index: usize,
    /// Semantic role name (e.g., "Agent", "Predicate", "Patient").
    pub role: String,
    /// The decoded word for this slot.
    pub word: String,
    /// Per-slot certainty (gamma), range 0.0 to 1.0.
    pub certainty: f32,
    /// Data source name (e.g., "Translator", "SoftCore").
    pub source: String,
    /// Number of populated resolution levels (0-4).
    pub resolution_count: u32,
}

/// Timing breakdown for a single think operation, in milliseconds.
///
/// # Example
///
/// ```
/// use volt_server::models::TimingMs;
///
/// let timing = TimingMs { encode_ms: 0.5, decode_ms: 0.3, total_ms: 0.8 };
/// let json = serde_json::to_string(&timing).unwrap();
/// assert!(json.contains("total_ms"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMs {
    /// Time spent encoding text to TensorFrame (ms).
    pub encode_ms: f64,
    /// Time spent decoding TensorFrame to text (ms).
    pub decode_ms: f64,
    /// Total end-to-end time (ms).
    pub total_ms: f64,
}

/// Error response body.
///
/// # Example
///
/// ```
/// use volt_server::models::ErrorResponse;
///
/// let err = ErrorResponse { error: "bad input".into() };
/// let json = serde_json::to_string(&err).unwrap();
/// assert!(json.contains("bad input"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error message.
    pub error: String,
}

/// Health check response.
///
/// # Example
///
/// ```
/// use volt_server::models::HealthResponse;
///
/// let h = HealthResponse { status: "ok".into(), version: "0.1.0".into() };
/// let json = serde_json::to_string(&h).unwrap();
/// assert!(json.contains("ok"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Service status.
    pub status: String,
    /// Service version.
    pub version: String,
}
