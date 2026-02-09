//! Axum route handlers for the HTTP API.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use volt_core::MAX_SLOTS;
use volt_translate::Translator;

use crate::models::{ErrorResponse, HealthResponse, ThinkRequest, ThinkResponse};
use crate::state::AppState;

/// `GET /health` — health check endpoint.
///
/// Returns a JSON object with service status and version.
///
/// # Example Response
///
/// ```json
/// {"status": "ok", "version": "0.1.0"}
/// ```
pub async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// `POST /api/think` — process text through the translation pipeline.
///
/// Accepts a JSON body with a `text` field, encodes it into a
/// TensorFrame, decodes it back, and returns the result with
/// per-slot certainty values.
///
/// # Errors
///
/// - 400 Bad Request: empty text, input too large
/// - 422 Unprocessable Entity: invalid JSON (handled by Axum)
pub async fn think(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ThinkRequest>,
) -> Result<Json<ThinkResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Encode: text -> TensorFrame
    let output = state.translator.encode(&request.text).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    // Extract gamma values from active slots
    let gamma: Vec<f32> = (0..MAX_SLOTS)
        .filter(|&i| output.frame.slots[i].is_some())
        .map(|i| output.frame.meta[i].certainty)
        .collect();

    // Decode: TensorFrame -> text
    let decoded_text = state.translator.decode(&output.frame).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("decode failed: {e}"),
            }),
        )
    })?;

    Ok(Json(ThinkResponse {
        text: decoded_text,
        gamma,
        strand_id: output.frame.frame_meta.strand_id,
        iterations: 1,
    }))
}
