//! # volt-server
//!
//! The HTTP server and orchestration layer for Volt X.
//!
//! This is the leaf crate — it imports from all other crates and
//! provides the user-facing API. No other crate may import from here.
//!
//! ## Key Components
//!
//! - Axum HTTP server with REST + WebSocket endpoints
//! - Request → Translate → Soft Core → Hard Core → Decode pipeline
//! - Hot-reload development server
//! - Health checks and metrics
//!
//! ## Architecture Rules
//!
//! - This is the ONLY crate that wires everything together.
//! - No other `volt-*` crate may depend on `volt-server`.
//! - Network code also lives in `volt-ledger`.

pub use volt_core;

// MILESTONE: 8.1 — Basic HTTP server
// TODO: Implement Axum server with health endpoint
// TODO: Implement /infer endpoint (text in → frame → text out)
// TODO: Implement WebSocket streaming
// TODO: Wire full inference pipeline
