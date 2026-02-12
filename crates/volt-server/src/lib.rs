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
//! - `GET /api/modules` — list installed modules
//!
//! ## Architecture Rules
//!
//! - This is the ONLY crate that wires everything together.
//! - No other `volt-*` crate may depend on `volt-server`.
//! - Network code also lives in `volt-ledger`.

pub mod models;
pub mod registry;
pub mod routes;
pub mod state;

pub use volt_core;

use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;

use crate::state::AppState;

/// Build the Axum application router with all routes.
///
/// Creates a default [`AppState`] internally. Use
/// [`build_app_with_state`] when you need to share state with
/// other components (e.g., the sleep scheduler).
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
    build_app_with_state(state)
}

/// Build the Axum router with a pre-created [`AppState`].
///
/// This allows the caller to retain `Arc` clones of the shared
/// VFN, memory store, and event logger for use by other components
/// such as the sleep consolidation scheduler.
///
/// # Example
///
/// ```no_run
/// use volt_server::build_app_with_state;
/// use volt_server::state::AppState;
///
/// #[tokio::main]
/// async fn main() {
///     let state = AppState::new();
///     let app = build_app_with_state(state);
///     let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
///     axum::serve(listener, app).await.unwrap();
/// }
/// ```
pub fn build_app_with_state(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(routes::health))
        .route("/api/think", post(routes::think))
        .route("/api/modules", get(routes::list_modules))
        .with_state(state)
}
