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
