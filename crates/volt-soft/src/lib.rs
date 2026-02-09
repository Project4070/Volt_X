//! # volt-soft
//!
//! The GPU Soft Core — the "right brain" of Volt X.
//!
//! Implements the Root-Attend-Refine (RAR) inference loop:
//! - **Root**: Slot-local VFN forward passes (parallel per-slot)
//! - **Attend**: Cross-slot attention O(S²) + ghost frame attention
//! - **Refine**: State update, manifold projection, convergence check
//!
//! ## Key Components
//!
//! - Vector Field Network (VFN): 500M–2B params, operates on frame slots
//! - Diffusion controller: Per-slot noise for creative exploration
//! - ODE Solver: RK4/DOPRI5 adaptive stepping
//! - Manifold Projector: Keeps slot vectors on semantic manifold
//! - Bleed Buffer: Ghost R₀ gists from long-term memory
//! - Retention Cache: Warm-start from recently-converged attractors
//!
//! ## Architecture Rules
//!
//! - All GPU code lives here — no GPU code in other crates.
//! - Feature-gated: `cargo test --features gpu` for GPU tests.
//! - Depends on `volt-core` and `volt-bus`.

pub use volt_core;

// MILESTONE: 3.1 — RAR inference loop (CPU simulation first)
// TODO: Implement Root phase (slot-local forward pass)
// TODO: Implement Attend phase (cross-slot attention)
// TODO: Implement Refine phase (state update + convergence)
// TODO: Implement per-slot convergence detection
// TODO: Implement adaptive computation budget
