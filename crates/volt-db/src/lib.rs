//! # volt-db
//!
//! VoltDB — the three-tier memory storage engine for Volt X.
//!
//! ## Memory Tiers
//!
//! - **T0 (Working Memory)**: 64 frames in RAM, instant access, ring buffer
//! - **T1 (Strand Storage)**: Frames in RAM organized by strand, persists to disk
//! - **T2 (Archive)**: *Not yet implemented* — compressed frames on disk
//!
//! ## Usage
//!
//! The primary entry point is [`VoltStore`], which combines T0 and T1
//! into a unified API with automatic eviction.
//!
//! ```
//! use volt_db::VoltStore;
//! use volt_core::TensorFrame;
//!
//! let mut store = VoltStore::new();
//!
//! // Store a frame (goes into T0 working memory)
//! let id = store.store(TensorFrame::new()).unwrap();
//!
//! // Retrieve by ID (searches T0, then T1)
//! let frame = store.get_by_id(id).unwrap();
//! assert_eq!(frame.frame_meta.frame_id, id);
//! ```
//!
//! ## Architecture Rules
//!
//! - Depends on `volt-core` and `volt-bus`.
//! - No network code (that's `volt-ledger`).
//! - No HNSW index yet (linear scan sufficient for thousands of frames).
//! - No T2 compression, ghost frames, or GC (Milestone 4.2 / 4.3).

pub mod tier0;
pub mod tier1;
mod store;

pub use store::VoltStore;
pub use volt_core;

// MILESTONE: 4.2 — HNSW Indexing + Ghost Bleed
// TODO: Implement HNSW index for T1 similarity search
// TODO: Implement ghost frame extraction (R₀ gists)
// TODO: Implement Bleed Engine integration with Soft Core

// MILESTONE: 4.3 — Tier 2 + GC + Consolidation
// TODO: Implement T2 archive (compressed, mmap'd, LSM-Tree)
// TODO: Implement WAL for crash recovery
// TODO: Implement GC pipeline with retention scoring
// TODO: Implement frame consolidation
