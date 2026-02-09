//! # volt-ledger
//!
//! The Intelligence Commons — Volt X's P2P knowledge sharing layer.
//!
//! Enables federated learning and knowledge exchange between Volt instances:
//! - Strand sharing (with consent and attribution)
//! - Community module distribution
//! - Federated model improvements
//!
//! ## Architecture Rules
//!
//! - Network code lives here (and in `volt-server`).
//! - All shared data is signed and attributed.
//! - Privacy-preserving: differential privacy on shared strands.
//! - Depends on `volt-core`, `volt-bus`, `volt-db`.

pub use volt_core;

// MILESTONE: 7.1 — Intelligence Commons foundation
// TODO: Define strand sharing protocol
// TODO: Implement append-only audit log
// TODO: Implement module distribution format
// TODO: Implement P2P mesh discovery
