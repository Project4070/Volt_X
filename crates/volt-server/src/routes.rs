//! Axum route handlers for the HTTP API.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use volt_bus::similarity_frames;
use volt_core::slot::SlotSource;
use volt_core::{SlotRole, VoltError, SLOT_DIM, MAX_SLOTS};
use volt_translate::decode::format_output;
use volt_translate::Translator;

use crate::models::{
    ErrorResponse, HealthResponse, ProofStepResponse, SlotState, ThinkRequest, ThinkResponse,
    TimingMs,
};
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

/// Result of the CPU-heavy pipeline work, lightweight enough to
/// return across a thread boundary to the async handler.
struct PipelineOutput {
    /// The verified frame (boxed to keep off the caller's stack).
    frame: Box<volt_core::TensorFrame>,
    /// RAR iteration count.
    iterations: u32,
    /// Proof steps extracted from the proof chain.
    proof_steps: Vec<ProofStepResponse>,
    /// Pre-check safety score.
    safety_score: f32,
    /// Number of ghost gists that influenced RAR.
    ghost_count: usize,
}

/// `POST /api/think` — process text through the full pipeline.
///
/// Pipeline: `Encode -> Soft Core (RAR) -> Safety + Hard Core -> Bus Check -> Decode`
///
/// Accepts a JSON body with a `text` field, encodes it into a
/// TensorFrame, runs RAR inference (Soft Core), routes through the
/// safety-wrapped Hard Core pipeline, verifies frame integrity via
/// the Bus, then decodes back to text.
///
/// # Errors
///
/// - 400 Bad Request: empty text, input too large
/// - 422 Unprocessable Entity: invalid JSON (handled by Axum)
/// - 403 Forbidden: safety violation (Omega Veto triggered)
pub async fn think(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ThinkRequest>,
) -> Result<Json<ThinkResponse>, (StatusCode, Json<ErrorResponse>)> {
    let total_start = Instant::now();

    // Encode: text -> TensorFrame
    let encode_start = Instant::now();
    let output = state.translator.encode(&request.text).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;
    let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;

    // Fetch ghost gists from memory before entering the pipeline thread.
    // Read lock is cheap — many concurrent readers allowed.
    let ghost_gists: Vec<[f32; SLOT_DIM]> = state
        .memory
        .read()
        .map(|guard| guard.ghost_gists())
        .unwrap_or_default();
    let ghost_count = ghost_gists.len();

    // Run the full CPU-heavy pipeline on a thread with adequate stack.
    // TensorFrame is ~65KB and the pipeline creates multiple copies,
    // so we need more than the default async executor thread stack.
    let pipeline_frame = Box::new(output.frame.clone());
    let pipeline_output = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || -> Result<PipelineOutput, (StatusCode, String)> {
            // Soft Core: RAR inference loop with ghost frame cross-attention.
            // Ghost gists from the Bleed Buffer provide subtle memory
            // influence (alpha=0.1) during the Attend phase.
            let rar_result = volt_soft::process_rar_with_ghosts(
                &pipeline_frame,
                &ghost_gists,
                0.1,
            )
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("soft core RAR failed: {e}"),
                )
            })?;
            let iterations = rar_result.iterations;

            // Hard Core + Safety: safety-wrapped pipeline
            let safety_result =
                volt_safety::safe_process_full(&rar_result.frame).map_err(|e| match &e {
                    VoltError::SafetyViolation { .. } => {
                        (StatusCode::FORBIDDEN, format!("safety violation: {e}"))
                    }
                    _ => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("hard core pipeline failed: {e}"),
                    ),
                })?;

            // Bus integrity check
            let _bus_similarity = similarity_frames(&pipeline_frame, &safety_result.frame);

            // Extract proof steps
            let proof_steps: Vec<ProofStepResponse> = safety_result
                .proof
                .map(|chain| {
                    chain
                        .steps
                        .into_iter()
                        .map(|step| ProofStepResponse {
                            strand_name: step.strand_name,
                            description: step.description,
                            similarity: step.similarity,
                            gamma_after: step.gamma_after,
                            activated: step.activated,
                        })
                        .collect()
                })
                .unwrap_or_default();

            Ok(PipelineOutput {
                frame: Box::new(safety_result.frame),
                iterations,
                proof_steps,
                safety_score: safety_result.pre_check_score,
                ghost_count,
            })
        })
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to spawn pipeline thread: {e}"),
                }),
            )
        })?
        .join()
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "pipeline thread panicked".to_string(),
                }),
            )
        })?
        .map_err(|(status, msg)| (status, Json(ErrorResponse { error: msg })))?;

    let verified_frame = pipeline_output.frame;

    // Store verified frame to memory (T0 working memory, auto-evicts to T1).
    // This feeds the HNSW index and refreshes the Ghost Bleed Buffer
    // so future requests benefit from memory of past conversations.
    let memory_frame_count = {
        let mut guard = state.memory.write().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("memory store lock failed: {e}"),
                }),
            )
        })?;
        guard.store(*verified_frame.clone()).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("memory store failed: {e}"),
                }),
            )
        })?;
        guard.total_frame_count()
    };

    // Extract gamma values from active slots
    let gamma: Vec<f32> = (0..MAX_SLOTS)
        .filter(|&i| verified_frame.slots[i].is_some())
        .map(|i| verified_frame.meta[i].certainty)
        .collect();

    // Decode: TensorFrame -> per-slot words and full text
    let decode_start = Instant::now();
    let slot_words = state
        .translator
        .decode_slots(&verified_frame)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("decode failed: {e}"),
                }),
            )
        })?;
    let decoded_text = format_output(&slot_words);
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    // Build per-slot debug state
    let slot_states: Vec<SlotState> = slot_words
        .iter()
        .map(|(index, role, word)| {
            let res_count = verified_frame.slots[*index]
                .as_ref()
                .map(|s| s.active_resolution_count() as u32)
                .unwrap_or(0);
            SlotState {
                index: *index,
                role: format_role(role),
                word: word.clone(),
                certainty: verified_frame.meta[*index].certainty,
                source: format_source(&verified_frame.meta[*index].source),
                resolution_count: res_count,
            }
        })
        .collect();

    Ok(Json(ThinkResponse {
        text: decoded_text,
        gamma,
        strand_id: verified_frame.frame_meta.strand_id,
        iterations: pipeline_output.iterations,
        slot_states,
        proof_steps: pipeline_output.proof_steps,
        safety_score: pipeline_output.safety_score,
        memory_frame_count,
        ghost_count: pipeline_output.ghost_count,
        timing_ms: TimingMs {
            encode_ms,
            decode_ms,
            total_ms,
        },
    }))
}

/// Format a [`SlotRole`] to a human-readable string.
fn format_role(role: &SlotRole) -> String {
    match role {
        SlotRole::Agent => "Agent".to_string(),
        SlotRole::Predicate => "Predicate".to_string(),
        SlotRole::Patient => "Patient".to_string(),
        SlotRole::Location => "Location".to_string(),
        SlotRole::Time => "Time".to_string(),
        SlotRole::Manner => "Manner".to_string(),
        SlotRole::Instrument => "Instrument".to_string(),
        SlotRole::Cause => "Cause".to_string(),
        SlotRole::Result => "Result".to_string(),
        SlotRole::Free(n) => format!("Free({n})"),
    }
}

/// Format a [`SlotSource`] to a human-readable string.
fn format_source(source: &SlotSource) -> String {
    match source {
        SlotSource::Empty => "Empty".to_string(),
        SlotSource::Translator => "Translator".to_string(),
        SlotSource::SoftCore => "SoftCore".to_string(),
        SlotSource::HardCore => "HardCore".to_string(),
        SlotSource::Memory => "Memory".to_string(),
        SlotSource::Personal => "Personal".to_string(),
    }
}
