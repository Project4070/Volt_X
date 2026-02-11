//! Integration tests for the HTTP server.
//!
//! Uses Axum's tower integration for in-process testing
//! without starting a real TCP listener.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt; // for oneshot()

use volt_server::build_app;
use volt_server::models::{HealthResponse, ThinkResponse};

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let app = build_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let health: HealthResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(health.status, "ok");
    assert_eq!(health.version, "0.1.0");
}

#[tokio::test]
async fn think_basic_input() {
    let app = build_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/think")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text": "The cat sat on the mat"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let think: ThinkResponse = serde_json::from_slice(&body).unwrap();

    assert!(!think.text.is_empty());
    assert!(!think.gamma.is_empty());
    assert_eq!(think.strand_id, 0);
    assert!(think.iterations <= 50, "RAR iterations should be within budget");
    assert!(!think.slot_states.is_empty());
    assert!(think.timing_ms.total_ms > 0.0);
}

#[tokio::test]
async fn think_response_gamma_matches_slots() {
    let app = build_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/think")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text": "hello world"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let think: ThinkResponse = serde_json::from_slice(&body).unwrap();

    // "hello world" = 2 words = 2 slots = 2 gamma values
    assert_eq!(think.gamma.len(), 2);
    assert_eq!(think.slot_states.len(), 2);
    for g in &think.gamma {
        assert!(
            *g >= 0.0 && *g <= 1.0,
            "gamma should be in [0, 1], got {}",
            g
        );
    }
}

#[tokio::test]
async fn think_empty_input_returns_400() {
    let app = build_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/think")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text": ""}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn think_huge_input_returns_400() {
    let app = build_app();
    let huge_text = "a ".repeat(10_000);
    let body = format!(r#"{{"text": "{huge_text}"}}"#);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/think")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn think_invalid_json_returns_client_error() {
    let app = build_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/think")
                .header("content-type", "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(response.status().is_client_error());
}

#[tokio::test]
async fn think_missing_text_field_returns_client_error() {
    let app = build_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/think")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"wrong_field": "hello"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(response.status().is_client_error());
}

#[tokio::test]
async fn think_response_slot_states_have_correct_roles() {
    let app = build_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/think")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text": "cat sat mat"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let think: ThinkResponse = serde_json::from_slice(&body).unwrap();

    // "cat sat mat" = 3 words -> slots 0, 1, 2
    assert_eq!(think.slot_states.len(), 3);

    assert_eq!(think.slot_states[0].index, 0);
    assert_eq!(think.slot_states[0].role, "Agent");
    assert!(!think.slot_states[0].word.is_empty());
    assert!(
        think.slot_states[0].certainty >= 0.0 && think.slot_states[0].certainty <= 1.0,
        "certainty should be in [0,1], got {}",
        think.slot_states[0].certainty
    );
    assert!(think.slot_states[0].resolution_count >= 1);

    assert_eq!(think.slot_states[1].index, 1);
    assert_eq!(think.slot_states[1].role, "Predicate");

    assert_eq!(think.slot_states[2].index, 2);
    assert_eq!(think.slot_states[2].role, "Patient");
}

#[tokio::test]
async fn think_response_timing_is_consistent() {
    let app = build_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/think")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text": "timing test input"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let think: ThinkResponse = serde_json::from_slice(&body).unwrap();

    assert!(think.timing_ms.encode_ms >= 0.0);
    assert!(think.timing_ms.decode_ms >= 0.0);
    assert!(think.timing_ms.total_ms >= 0.0);
    // Total should be at least as large as encode + decode
    assert!(
        think.timing_ms.total_ms >= think.timing_ms.encode_ms + think.timing_ms.decode_ms - 0.001,
        "total_ms ({}) should be >= encode_ms ({}) + decode_ms ({})",
        think.timing_ms.total_ms,
        think.timing_ms.encode_ms,
        think.timing_ms.decode_ms,
    );
}

#[tokio::test]
async fn think_response_has_proof_chain() {
    let app = build_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/think")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text": "cat sat mat"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let think: ThinkResponse = serde_json::from_slice(&body).unwrap();

    // Should have at least a certainty_engine step
    assert!(
        !think.proof_steps.is_empty(),
        "proof_steps should not be empty"
    );

    // Every step should have valid fields
    for step in &think.proof_steps {
        assert!(!step.strand_name.is_empty());
        assert!(!step.description.is_empty());
        assert!(step.gamma_after >= 0.0 && step.gamma_after <= 1.0);
        assert!(step.similarity >= 0.0 && step.similarity <= 1.0);
    }

    // Last step should be certainty_engine
    let last = think.proof_steps.last().unwrap();
    assert_eq!(last.strand_name, "certainty_engine");
}

#[tokio::test]
async fn think_response_has_safety_score() {
    let app = build_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/think")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text": "hello world"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let think: ThinkResponse = serde_json::from_slice(&body).unwrap();

    // Normal input should have a low safety score
    assert!(
        think.safety_score >= 0.0,
        "safety_score should be >= 0, got {}",
        think.safety_score
    );
    assert!(
        think.safety_score < 0.5,
        "normal input should have low safety_score, got {}",
        think.safety_score
    );
}

#[tokio::test]
async fn concurrent_requests_do_not_crash() {
    use tokio::task::JoinSet;

    let mut tasks = JoinSet::new();

    // Reduced from 100 to 10 since each request now runs full RAR + safety pipeline
    for i in 0..10 {
        tasks.spawn(async move {
            let app = build_app();
            let text = format!("request number {i}");
            let body = format!(r#"{{"text": "{text}"}}"#);

            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/think")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK);
        });
    }

    while let Some(result) = tasks.join_next().await {
        result.expect("task should not panic");
    }
}

// --------------------------------------------------------------------------
// Memory integration tests (Phase 4 wiring)
// --------------------------------------------------------------------------

/// Helper: send a think request and parse the response.
async fn think_once(app: axum::Router, text: &str) -> ThinkResponse {
    let body = format!(r#"{{"text": "{text}"}}"#);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/think")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn think_stores_frame_to_memory() {
    let app = build_app();

    let resp = think_once(app, "cat sat mat").await;

    // After one request, exactly one frame should be stored
    assert_eq!(
        resp.memory_frame_count, 1,
        "first request should store 1 frame"
    );
}

#[tokio::test]
async fn think_first_request_has_zero_ghosts() {
    let app = build_app();

    let resp = think_once(app, "hello world").await;

    // First request has no history, so ghost count should be 0
    assert_eq!(
        resp.ghost_count, 0,
        "first request should have 0 ghost gists"
    );
}

#[tokio::test]
async fn think_memory_accumulates_across_requests() {
    let app = build_app();

    // Send 3 sequential requests, each using a clone of the router
    // (all share the same AppState and therefore the same memory).
    let resp1 = think_once(app.clone(), "the cat sat").await;
    let resp2 = think_once(app.clone(), "on the mat").await;
    let resp3 = think_once(app, "in the hat").await;

    assert_eq!(resp1.memory_frame_count, 1);
    assert_eq!(resp2.memory_frame_count, 2);
    assert_eq!(resp3.memory_frame_count, 3);
}

/// Verifies that frames produced by the full pipeline
/// (encode → RAR → safety) have extractable R₀ gists and populate
/// the ghost buffer when stored in VoltStore.
#[tokio::test]
async fn pipeline_frames_produce_valid_gists() {
    use volt_db::{VoltStore, extract_gist};
    use volt_translate::Translator;

    // Run on a thread with adequate stack (TensorFrame is ~64KB,
    // VoltStore operations stack-allocate multiple copies on Windows).
    let result = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let translator = volt_translate::StubTranslator::new();

            // Run full pipeline: encode → RAR → safety
            let output = translator.encode("cat sat mat").unwrap();
            let rar_result = volt_soft::process_rar(&output.frame).unwrap();
            let safety_result = volt_safety::safe_process_full(&rar_result.frame).unwrap();

            // Verified frame must have extractable R₀ gist
            let gist = extract_gist(&safety_result.frame).unwrap();
            assert!(
                gist.is_some(),
                "verified frame should have extractable R₀ gist; active_slots={}",
                safety_result.frame.active_slot_count(),
            );

            // VoltStore indexes gists and populates ghost buffer
            let mut store = VoltStore::new();
            for i in 0..5 {
                let out = translator.encode(&format!("word{i} sat mat")).unwrap();
                let rar = volt_soft::process_rar(&out.frame).unwrap();
                let safe = volt_safety::safe_process_full(&rar.frame).unwrap();
                store.store(safe.frame).unwrap();
            }

            assert_eq!(store.hnsw_entries(), 5);
            assert!(
                !store.ghost_gists().is_empty(),
                "after storing 5 pipeline frames, ghost buffer should be non-empty",
            );
        })
        .expect("failed to spawn test thread")
        .join();
    result.expect("test thread panicked");
}

#[tokio::test]
async fn think_ghost_buffer_populates_after_history() {
    let app = build_app();

    // Store several frames with varied text to build up HNSW history.
    // Using different texts produces diverse R₀ gists, which gives the
    // HNSW index a richer graph to search over.
    let queries = [
        "cat sat mat",
        "dog ran park",
        "bird flew sky",
        "fish swam sea",
        "cat sat mat",
        "dog ran park",
        "bird flew sky",
        "fish swam sea",
        "cat sat mat",
        "dog ran park",
    ];
    for text in &queries {
        think_once(app.clone(), text).await;
    }

    // The next request should see ghost gists from the accumulated history
    let resp = think_once(app, "cat sat mat").await;
    assert!(
        resp.ghost_count > 0,
        "after storing {} frames, ghost_count should be > 0, got {}",
        queries.len(),
        resp.ghost_count,
    );
}

#[tokio::test]
async fn think_concurrent_shared_memory() {
    use tokio::task::JoinSet;

    let app = build_app();
    let mut tasks = JoinSet::new();

    // 5 concurrent requests all sharing the same memory
    for i in 0..5 {
        let app_clone = app.clone();
        tasks.spawn(async move {
            let text = format!("concurrent request {i}");
            let resp = think_once(app_clone, &text).await;
            assert!(
                resp.memory_frame_count >= 1,
                "memory should contain at least 1 frame"
            );
        });
    }

    while let Some(result) = tasks.join_next().await {
        result.expect("concurrent memory task should not panic");
    }
}

#[tokio::test]
async fn think_response_includes_memory_fields() {
    let app = build_app();

    let resp = think_once(app, "hello world").await;

    // Verify the new fields exist and have sane values
    assert!(
        resp.memory_frame_count >= 1,
        "memory_frame_count should be >= 1 after a request"
    );
    // ghost_count is a usize, so always >= 0 — just verify it's present
    // by accessing it (deserialization would have failed if missing)
    let _ = resp.ghost_count;
}
