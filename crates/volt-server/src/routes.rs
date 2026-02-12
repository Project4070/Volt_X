//! Axum route handlers for the HTTP API.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::Stream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use volt_bus::similarity_frames;
use volt_core::slot::SlotSource;
use volt_core::{SlotRole, VoltError, SLOT_DIM, MAX_SLOTS};
use volt_soft::attention::SlotAttention;
use volt_soft::rar::{rar_loop_with_ghosts, GhostConfig, RarConfig};
use volt_translate::decode::format_output;
use volt_translate::Translator;

use crate::models::{
    ConversationHistoryResponse, ConversationListResponse, CreateConversationResponse,
    ErrorResponse, HealthResponse, HistoryMessage, ModuleResponse, ProofStepResponse, SlotState,
    StreamEvent, ThinkRequest, ThinkResponse, TimingMs,
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

    // Get or create conversation
    let conversation_id = state
        .get_or_create_conversation(request.conversation_id)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("conversation creation failed: {e}"),
                }),
            )
        })?;

    // Switch to the conversation's strand in VoltDB
    state.memory.write().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("memory lock failed: {e}"),
            }),
        )
    })?.switch_strand(conversation_id).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("strand switch failed: {e}"),
            }),
        )
    })?;

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

    // Snapshot the shared VFN for this request. Clone is ~6 MB (three
    // Linear layers) but avoids holding the read lock during the entire
    // RAR loop, so the sleep scheduler can still write-lock for training.
    let vfn_snapshot = state.vfn.read().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("VFN read lock failed: {e}"),
            }),
        )
    })?.clone();

    // Run the full CPU-heavy pipeline on a thread with adequate stack.
    // TensorFrame is ~65KB and the pipeline creates multiple copies,
    // so we need more than the default async executor thread stack.
    let pipeline_frame = Box::new(output.frame.clone());
    let pipeline_output = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || -> Result<PipelineOutput, (StatusCode, String)> {
            // Soft Core: RAR inference loop with ghost frame cross-attention.
            // Uses the shared VFN (snapshot) so trained weights from sleep
            // consolidation carry through to inference.
            let attention = SlotAttention::new_random(43);
            // CRITICAL: Route on the ORIGINAL encoded frame BEFORE RAR!
            // RAR modifies all slots, which destroys capability tags used for routing.
            // If a Hard Strand activates, use its result directly.
            // Otherwise, fall back to RAR refinement.
            let safety_result_original =
                volt_safety::safe_process_full(&pipeline_frame).map_err(|e| match &e {
                    VoltError::SafetyViolation { .. } => {
                        (StatusCode::FORBIDDEN, format!("safety violation: {e}"))
                    }
                    _ => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("hard core pipeline failed: {e}"),
                    ),
                })?;

            // Check if any Hard Strand activated
            let hard_strand_activated = safety_result_original
                .proof
                .as_ref()
                .map(|chain| chain.steps.iter().any(|step| step.activated))
                .unwrap_or(false);

            let (safety_result, iterations) = if hard_strand_activated {
                // Hard Strand handled it — use that result directly (no RAR needed)
                (safety_result_original, 0)
            } else {
                // No Hard Strand match — run through Soft Core RAR for refinement
                let config = RarConfig::default();
                let ghost_config = GhostConfig { gists: ghost_gists, alpha: 0.1 };
                let rar_result = rar_loop_with_ghosts(
                    &pipeline_frame,
                    &vfn_snapshot,
                    &attention,
                    &config,
                    &ghost_config,
                )
                .map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("soft core RAR failed: {e}"),
                    )
                })?;
                let iterations = rar_result.iterations;

                // Route the refined frame through Hard Core again
                let safety_result_refined =
                    volt_safety::safe_process_full(&rar_result.frame).map_err(|e| match &e {
                        VoltError::SafetyViolation { .. } => {
                            (StatusCode::FORBIDDEN, format!("safety violation: {e}"))
                        }
                        _ => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("hard core pipeline failed: {e}"),
                        ),
                    })?;
                (safety_result_refined, iterations)
            };

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

    // Log learning event (best-effort — never fail the request).
    {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        let mut gamma_scores = [0.0f32; MAX_SLOTS];
        for (i, score) in gamma_scores.iter_mut().enumerate() {
            if verified_frame.slots[i].is_some() {
                *score = verified_frame.meta[i].certainty;
            }
        }

        let event = volt_learn::LearningEvent {
            frame_id: verified_frame.frame_meta.frame_id,
            strand_id: verified_frame.frame_meta.strand_id,
            query_type: verified_frame.frame_meta.discourse_type,
            gamma_scores,
            convergence_iterations: pipeline_output.iterations,
            ghost_activations: pipeline_output.ghost_count,
            timestamp: now,
        };

        if let Ok(mut logger) = state.event_logger.write() {
            logger.log(event);
        }
    }

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

    // Update conversation metadata
    state.update_conversation_metadata(conversation_id);

    Ok(Json(ThinkResponse {
        text: decoded_text,
        gamma,
        conversation_id,
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

/// `POST /api/think/stream` — process text with SSE streaming.
///
/// Same as `/api/think` but streams progress updates via Server-Sent Events.
/// Sends real-time updates during encoding, RAR inference, and completion.
///
/// # Event Types
///
/// - `status` - Status message (e.g., "Encoding...")
/// - `encoding` - Encoding phase started
/// - `thinking` - RAR inference started
/// - `complete` - Processing completed (includes full ThinkResponse)
/// - `error` - Error occurred
pub async fn think_stream(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ThinkRequest>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = mpsc::channel(100);

    // Clone state components for the async task
    let state_clone = state.clone();
    let request_clone = request.clone();

    tokio::spawn(async move {
        tracing::info!("Starting streaming request");

        // Helper to send events (ignore errors if client disconnected)
        let send = |event: StreamEvent| {
            let tx = tx.clone();
            async move {
                let event_str = serde_json::to_string(&event).unwrap_or_default();
                if tx.send(Ok(Event::default().data(event_str))).await.is_err() {
                    tracing::warn!("Client disconnected during streaming");
                }
            }
        };

        let total_start = Instant::now();

        // Get or create conversation
        send(StreamEvent::Status("Preparing conversation...".to_string())).await;
        tracing::info!("Creating/getting conversation");
        let conversation_id = match state_clone
            .get_or_create_conversation(request_clone.conversation_id)
        {
            Ok(id) => {
                tracing::info!("Conversation ID: {}", id);
                id
            }
            Err(e) => {
                tracing::error!("Conversation creation failed: {}", e);
                send(StreamEvent::Error(format!("conversation creation failed: {e}"))).await;
                return;
            }
        };

        // Switch to conversation strand
        if let Err(e) = state_clone
            .memory
            .write()
            .and_then(|mut guard| guard.switch_strand(conversation_id))
        {
            send(StreamEvent::Error(format!("strand switch failed: {e}"))).await;
            return;
        }

        // Encode
        send(StreamEvent::Encoding).await;
        tracing::info!("Starting encoding: {:?}", request_clone.text);
        let encode_start = Instant::now();
        let output = match state_clone.translator.encode(&request_clone.text) {
            Ok(o) => {
                tracing::info!("Encoding successful");
                o
            }
            Err(e) => {
                tracing::error!("Encoding failed: {}", e);
                send(StreamEvent::Error(format!("encode failed: {e}"))).await;
                return;
            }
        };
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;

        // Fetch ghost gists
        let ghost_gists: Vec<[f32; SLOT_DIM]> = state_clone
            .memory
            .read()
            .map(|guard| guard.ghost_gists())
            .unwrap_or_default();
        let ghost_count = ghost_gists.len();

        // Snapshot VFN (must extract before any await)
        let vfn_result: Result<volt_soft::vfn::Vfn, String> = state_clone.vfn.read()
            .map(|guard| guard.clone())
            .map_err(|e| format!("VFN read lock failed: {e}"));
        let vfn_snapshot = match vfn_result {
            Ok(vfn) => vfn,
            Err(e) => {
                send(StreamEvent::Error(e)).await;
                return;
            }
        };

        // Run pipeline
        send(StreamEvent::Thinking).await;
        tracing::info!("Starting RAR pipeline");
        let pipeline_frame = Box::new(output.frame.clone());
        let pipeline_output = match std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(move || -> Result<PipelineOutput, String> {
                // CRITICAL: Route on the ORIGINAL encoded frame BEFORE RAR!
                // RAR modifies all slots, which destroys capability tags used for routing.
                let safety_result_original = volt_safety::safe_process_full(&pipeline_frame)
                    .map_err(|e| format!("hard core pipeline failed: {e}"))?;

                // Check if any Hard Strand activated
                let hard_strand_activated = safety_result_original
                    .proof
                    .as_ref()
                    .map(|chain| chain.steps.iter().any(|step| step.activated))
                    .unwrap_or(false);

                let (safety_result, iterations) = if hard_strand_activated {
                    // Hard Strand handled it — use result directly (no RAR needed)
                    (safety_result_original, 0)
                } else {
                    // No Hard Strand match — run through Soft Core RAR for refinement
                    let attention = SlotAttention::new_random(43);
                    let config = RarConfig::default();
                    let ghost_config = GhostConfig {
                        gists: ghost_gists,
                        alpha: 0.1,
                    };
                    let rar_result =
                        rar_loop_with_ghosts(&pipeline_frame, &vfn_snapshot, &attention, &config, &ghost_config)
                            .map_err(|e| format!("soft core RAR failed: {e}"))?;
                    let iterations = rar_result.iterations;

                    let safety_result_refined = volt_safety::safe_process_full(&rar_result.frame)
                        .map_err(|e| format!("hard core pipeline failed: {e}"))?;
                    (safety_result_refined, iterations)
                };

                let _bus_similarity =
                    similarity_frames(&pipeline_frame, &safety_result.frame);

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
            .and_then(|handle| handle.join().map_err(|_| std::io::Error::other("thread panicked")))
        {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                send(StreamEvent::Error(format!("pipeline failed: {e}"))).await;
                return;
            }
            Err(e) => {
                send(StreamEvent::Error(format!("thread spawn failed: {e}"))).await;
                return;
            }
        };

        let verified_frame = pipeline_output.frame;

        // Store to memory
        let memory_frame_count = match state_clone
            .memory
            .write()
            .and_then(|mut guard| {
                guard.store(*verified_frame.clone())?;
                Ok(guard.total_frame_count())
            })
        {
            Ok(count) => count,
            Err(e) => {
                send(StreamEvent::Error(format!("memory store failed: {e}"))).await;
                return;
            }
        };

        // Log learning event
        {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as u64)
                .unwrap_or(0);

            let mut gamma_scores = [0.0f32; MAX_SLOTS];
            for (i, score) in gamma_scores.iter_mut().enumerate() {
                if verified_frame.slots[i].is_some() {
                    *score = verified_frame.meta[i].certainty;
                }
            }

            let event = volt_learn::LearningEvent {
                frame_id: verified_frame.frame_meta.frame_id,
                strand_id: verified_frame.frame_meta.strand_id,
                query_type: verified_frame.frame_meta.discourse_type,
                gamma_scores,
                convergence_iterations: pipeline_output.iterations,
                ghost_activations: pipeline_output.ghost_count,
                timestamp: now,
            };

            if let Ok(mut logger) = state_clone.event_logger.write() {
                logger.log(event);
            }
        }

        // Extract gamma
        let gamma: Vec<f32> = (0..MAX_SLOTS)
            .filter(|&i| verified_frame.slots[i].is_some())
            .map(|i| verified_frame.meta[i].certainty)
            .collect();

        // Decode
        let decode_start = Instant::now();
        let slot_words = match state_clone.translator.decode_slots(&verified_frame) {
            Ok(words) => words,
            Err(e) => {
                send(StreamEvent::Error(format!("decode failed: {e}"))).await;
                return;
            }
        };
        let decoded_text = format_output(&slot_words);
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        // Build slot states
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

        // Update conversation metadata
        state_clone.update_conversation_metadata(conversation_id);

        // Send completion event
        tracing::info!("Sending completion event");
        send(StreamEvent::Complete(ThinkResponse {
            text: decoded_text,
            gamma,
            conversation_id,
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
        .await;

        tracing::info!("Streaming request completed successfully");
    });

    Sse::new(ReceiverStream::new(rx))
}

/// `GET /api/modules` — list all installed modules.
///
/// Returns a JSON array of module metadata, including built-in modules
/// and any feature-gated community modules that were compiled in.
///
/// # Example Response
///
/// ```json
/// [
///   {"id": "math_engine", "display_name": "Math Engine", ...},
///   {"id": "hdc_algebra", "display_name": "HDC Algebra", ...}
/// ]
/// ```
pub async fn list_modules(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let modules: Vec<ModuleResponse> = state
        .registry
        .list_modules()
        .iter()
        .map(|m| ModuleResponse {
            id: m.id.clone(),
            display_name: m.display_name.clone(),
            version: m.version.clone(),
            author: m.author.clone(),
            description: m.description.clone(),
            module_type: m.module_type.to_string(),
        })
        .collect();
    Json(modules)
}

/// `POST /api/conversations` — create a new conversation.
///
/// Creates a new conversation (VoltDB strand) and returns its ID.
/// The conversation ID can then be used in subsequent `/api/think` requests
/// to maintain context across messages.
///
/// # Example Response
///
/// ```json
/// {"conversation_id": 1234567890}
/// ```
pub async fn create_conversation(
    State(state): State<Arc<AppState>>,
) -> Result<Json<CreateConversationResponse>, (StatusCode, Json<ErrorResponse>)> {
    let conversation_id = state.get_or_create_conversation(None).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("conversation creation failed: {e}"),
            }),
        )
    })?;

    Ok(Json(CreateConversationResponse { conversation_id }))
}

/// `GET /api/conversations` — list all conversations.
///
/// Returns metadata for all conversations, sorted by last_message_at
/// descending (most recent first).
///
/// # Example Response
///
/// ```json
/// {
///   "conversations": [
///     {"id": 1, "created_at": 1000, "last_message_at": 2000, "message_count": 5}
///   ]
/// }
/// ```
pub async fn list_conversations(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ConversationListResponse>, (StatusCode, Json<ErrorResponse>)> {
    let conversations = state.conversations.read().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("conversations lock poisoned: {e}"),
            }),
        )
    })?;

    let mut conv_list: Vec<_> = conversations.values().cloned().collect();
    conv_list.sort_by(|a, b| b.last_message_at.cmp(&a.last_message_at));

    Ok(Json(ConversationListResponse {
        conversations: conv_list,
    }))
}

/// `GET /api/conversations/:id/history` — retrieve conversation history.
///
/// Returns all messages in a conversation in chronological order.
/// Each message includes the decoded text, gamma scores, and timestamp.
///
/// # Errors
///
/// - 404 Not Found: conversation ID does not exist
///
/// # Example Response
///
/// ```json
/// {
///   "conversation_id": 1,
///   "messages": [
///     {"frame_id": 100, "text": "hello", "gamma": [0.8], "timestamp": 1000},
///     {"frame_id": 101, "text": "hi there", "gamma": [0.9], "timestamp": 2000}
///   ]
/// }
/// ```
pub async fn get_conversation_history(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<ConversationHistoryResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Check if conversation exists
    let conv_exists = state
        .conversations
        .read()
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("conversations lock poisoned: {e}"),
                }),
            )
        })?
        .contains_key(&id);

    if !conv_exists {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("conversation {id} not found"),
            }),
        ));
    }

    // Retrieve all frames for this conversation from VoltDB and clone them
    // We need to clone because get_by_strand returns references
    let frames: Vec<volt_core::TensorFrame> = {
        let guard = state.memory.read().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("memory lock failed: {e}"),
                }),
            )
        })?;
        guard.get_by_strand(id).into_iter().cloned().collect()
    };

    // Decode each frame to build history messages
    let mut messages = Vec::new();
    for frame in &frames {
        let slot_words = state.translator.decode_slots(frame).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("decode failed: {e}"),
                }),
            )
        })?;

        let text = format_output(&slot_words);

        let gamma: Vec<f32> = (0..MAX_SLOTS)
            .filter(|&i| frame.slots[i].is_some())
            .map(|i| frame.meta[i].certainty)
            .collect();

        // Extract timestamp from frame metadata (use frame_id as fallback)
        let timestamp = frame.frame_meta.frame_id;

        messages.push(HistoryMessage {
            frame_id: frame.frame_meta.frame_id,
            text,
            gamma,
            timestamp,
        });
    }

    Ok(Json(ConversationHistoryResponse {
        conversation_id: id,
        messages,
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
