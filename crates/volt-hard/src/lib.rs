//! # volt-hard
//!
//! The CPU Hard Core — the "left brain" of Volt X.
//!
//! Implements deterministic, exact computation in pure Rust:
//! - **Intent Router**: Routes frame slots to Hard Strands by vector similarity
//! - **Hard Strands**: Pluggable Rust modules (MathEngine, CodeRunner, APIDispatch)
//! - **HDC Algebra**: FFT-based bind/unbind for verification
//! - **Certainty Engine**: Per-slot γ computation using min-rule propagation
//! - **Proof Constructor**: Traceable reasoning chains
//! - **Causal Simulation**: do(X=x) counterfactual reasoning
//!
//! ## Architecture Rules
//!
//! - Pure CPU, no GPU code.
//! - No network code (network goes in `volt-ledger` or `volt-server`).
//! - Depends on `volt-core` and `volt-bus`.
//! - Hard Strands are hot-pluggable via `impl HardStrand` trait.

pub use volt_core;

// MILESTONE: 3.2 — Hard Core basics
// TODO: Define HardStrand trait
// TODO: Implement Intent Router (cosine similarity routing)
// TODO: Implement MathEngine Hard Strand
// TODO: Implement Certainty Engine (min-rule γ propagation)
// TODO: Implement Proof Chain Constructor
