//! # volt-bus
//!
//! The LLL (Low-Level Language) algebra engine for Volt X.
//!
//! Implements Hyperdimensional Computing (HDC) operations on TensorFrames:
//! - **Bind** (⊗): FFT-based circular convolution for role-filler binding
//! - **Unbind** (⊗⁻¹): Inverse binding for content retrieval
//! - **Superpose** (+): Additive superposition of multiple bindings
//! - **Permute** (ρ): Role permutation for structural manipulation
//! - **Similarity**: Cosine similarity between slot embeddings
//!
//! ## Architecture Rules
//!
//! - No `async` code — pure synchronous algebra.
//! - Depends only on `volt-core`.
//! - All operations work on `TensorFrame` slot embeddings.

pub use volt_core;

// MILESTONE: 2.1 — LLL Vector Bus implementation
// TODO: Implement HDC bind operation using FFT
// TODO: Implement unbind (inverse bind)
// TODO: Implement superposition
// TODO: Implement permutation
// TODO: Implement cosine similarity
