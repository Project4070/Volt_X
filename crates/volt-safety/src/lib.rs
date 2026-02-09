//! # volt-safety
//!
//! The deterministic safety layer for Volt X.
//!
//! Safety is enforced on the CPU, not approximated by neural networks.
//! This means safety guarantees are provable, not probabilistic.
//!
//! ## Key Components
//!
//! - **Axiomatic Guard**: K immutable axioms that can never be violated
//! - **Transition Monitor**: Validates every F(t) → F(t+1) state transition
//! - **Omega Veto**: Hardware-level interrupt, system HALT on critical violation
//! - **Proof Verification**: Validates proof chains from the Hard Core
//!
//! ## Architecture Rules
//!
//! - Pure CPU, deterministic logic — no neural approximation.
//! - Axioms are code, not learned weights.
//! - The Omega Veto cannot be overridden by any other component.
//! - Depends on `volt-core`, `volt-bus`, `volt-hard`.

pub use volt_core;

// MILESTONE: 5.1 — Safety layer basics
// TODO: Define Axiom trait and axiomatic guard
// TODO: Implement Transition Monitor (F(t) → F(t+1) validation)
// TODO: Implement Omega Veto (halt mechanism)
// TODO: Implement proof chain verification
