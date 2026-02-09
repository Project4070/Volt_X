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
/// use volt_server::models::ThinkResponse;
///
/// let resp = ThinkResponse {
///     text: "cat sat mat.".into(),
///     gamma: vec![0.8, 0.8, 0.8],
///     strand_id: 0,
///     iterations: 1,
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
    /// The strand ID (always 0 for stub).
    pub strand_id: u64,
    /// Number of RAR iterations (always 1 for stub).
    pub iterations: u32,
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
