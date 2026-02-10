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
//! ## HDC Background
//!
//! Hyperdimensional Computing represents concepts as high-dimensional vectors
//! (here: 256 dims). Key properties:
//! - **Binding** creates compound representations: `ROLE ⊗ FILLER`
//! - **Superposition** combines representations: `A + B + C`
//! - **Similarity** measures relatedness: `cos(A, B)`
//!
//! ## Performance Characteristics
//!
//! All operations on 256-dimensional vectors (commodity hardware):
//! - `bind`: ~8-15µs (FFT-based, target < 10µs)
//! - `unbind`: ~8-15µs (FFT-based)
//! - `superpose`: ~2-5µs (sum + normalize)
//! - `permute`: ~0.5-1µs (array rotation)
//! - `similarity`: ~0.5-1µs (dot product)
//!
//! ## Example
//!
//! ```
//! use volt_bus::{bind, unbind, similarity};
//! use volt_core::SLOT_DIM;
//!
//! // Create role and filler vectors (normalized)
//! let mut role = [0.0; SLOT_DIM];
//! role[0] = 1.0;
//! let mut filler = [0.0; SLOT_DIM];
//! filler[1] = 1.0;
//!
//! // Bind them together
//! let bound = bind(&role, &filler).unwrap();
//!
//! // Bound vector is dissimilar to inputs (binding property)
//! assert!(similarity(&bound, &role).abs() < 0.1);
//!
//! // Unbind to recover filler
//! let recovered = unbind(&bound, &role).unwrap();
//! assert!(similarity(&recovered, &filler) > 0.85);
//! ```
//!
//! ## Architecture Rules
//!
//! - No `async` code — pure synchronous algebra
//! - Depends only on `volt-core`
//! - All operations work on `TensorFrame` slot embeddings

// Re-export volt-core for convenience
pub use volt_core;

// Internal modules
mod fft;
mod ops;
mod batch;
pub mod codebook;

// Public API: Single-vector operations
pub use ops::{bind, unbind, superpose, permute, similarity};

// Public API: Batch operations on TensorFrames
pub use batch::{bind_frames, unbind_frames, similarity_frames};
