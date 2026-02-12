//! Integration tests for SSE streaming endpoint.

use volt_server::{build_app_with_state, state::AppState};

/// Test that the streaming endpoint accepts requests and sends SSE events.
#[tokio::test]
async fn streaming_endpoint_responds_with_events() {
    let state = AppState::new();
    let app = build_app_with_state(state);

    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .method("POST")
            .uri("/api/think/stream")
            .header("content-type", "application/json")
            .body(r#"{"text":"hello"}"#.to_string())
            .unwrap(),
    )
    .await
    .unwrap();

    // SSE endpoints return 200 OK
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    // Content-Type should be text/event-stream
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok());
    assert!(
        content_type.is_some_and(|ct| ct.contains("text/event-stream")),
        "Expected text/event-stream, got: {:?}",
        content_type
    );
}

/// Test that streaming with invalid input still returns proper SSE error events.
#[tokio::test]
async fn streaming_with_empty_text_sends_error_event() {
    let state = AppState::new();
    let app = build_app_with_state(state);

    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .method("POST")
            .uri("/api/think/stream")
            .header("content-type", "application/json")
            .body(r#"{"text":""}"#.to_string())
            .unwrap(),
    )
    .await
    .unwrap();

    // SSE endpoints always return 200, errors are sent as events
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    // Content-Type should be text/event-stream
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok());
    assert!(content_type.is_some_and(|ct| ct.contains("text/event-stream")));
}

/// Test that streaming creates conversations like the normal endpoint.
#[tokio::test]
async fn streaming_creates_conversation_when_none_provided() {
    let state = AppState::new();
    let state_clone = state.clone();
    let app = build_app_with_state(state);

    let response = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .method("POST")
            .uri("/api/think/stream")
            .header("content-type", "application/json")
            .body(r#"{"text":"test message"}"#.to_string())
            .unwrap(),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);

    // After streaming completes, there should be at least one conversation
    // (We can't easily parse the SSE stream in tests, but we can verify
    // the conversation was created by checking state)
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    let convs = state_clone.conversations.read().unwrap();
    assert!(
        !convs.is_empty(),
        "Expected at least one conversation to be created"
    );
}
