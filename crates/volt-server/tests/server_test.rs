//! Integration tests for Milestone 1.3 HTTP server.
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
    assert_eq!(think.iterations, 1);
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
    for g in &think.gamma {
        assert!((*g - 0.8).abs() < 0.01, "gamma should be 0.8, got {}", g);
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
async fn concurrent_requests_do_not_crash() {
    // PHASE-1.md: Handle 100 concurrent requests without crash
    use tokio::task::JoinSet;

    let mut tasks = JoinSet::new();

    for i in 0..100 {
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
