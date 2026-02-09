//! # volt-server
//!
//! The HTTP server and orchestration layer for Volt X.
//!
//! This is the leaf crate — it imports from all other crates and
//! provides the user-facing API. No other crate may import from here.
//!
//! ## Endpoints
//!
//! - `GET /health` — health check
//! - `POST /api/think` — process text through the translation pipeline
//!
//! ## Architecture Rules
//!
//! - This is the ONLY crate that wires everything together.
//! - No other `volt-*` crate may depend on `volt-server`.
//! - Network code also lives in `volt-ledger`.

pub mod models;
pub mod routes;
pub mod state;

pub use volt_core;

use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;

use crate::state::AppState;

/// Build the Axum application router with all routes.
///
/// Returns a configured [`Router`] ready to be served. The router
/// uses shared [`AppState`] for the stub translator.
///
/// # Example
///
/// ```no_run
/// use volt_server::build_app;
///
/// #[tokio::main]
/// async fn main() {
///     let app = build_app();
///     let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
///     axum::serve(listener, app).await.unwrap();
/// }
/// ```
pub fn build_app() -> Router {
    let state: Arc<AppState> = AppState::new();

    Router::new()
        .route("/health", get(routes::health))
        .route("/api/think", post(routes::think))
        .with_state(state)
}
