//! # volt-db
//!
//! VoltDB — the three-tier memory storage engine for Volt X.
//!
//! ## Memory Tiers
//!
//! - **T0 (Working Memory)**: 64 frames in VRAM/RAM, instant access
//! - **T1 (Strand Storage)**: Millions of frames in RAM, organized by strand
//! - **T2 (Archive)**: Compressed frames on disk, holographic retrieval
//!
//! ## Key Components
//!
//! - Strand-organized storage (HashMap per strand)
//! - HNSW index for approximate nearest neighbor queries
//! - WAL (Write-Ahead Log) for durability
//! - Ghost frame management (R₀ gists for Bleed Buffer)
//! - Eviction policies (T0→T1, T1→T2)
//!
//! ## Architecture Rules
//!
//! - Depends on `volt-core` and `volt-bus`.
//! - No network code (that's `volt-ledger`).
//! - Frame storage and retrieval must be lock-free where possible.

pub use volt_core;

// MILESTONE: 4.1 — VoltDB T0 + T1
// TODO: Implement StrandStore (HashMap<u64, Vec<TensorFrame>>)
// TODO: Implement T0 working memory (fixed-size ring buffer)
// TODO: Implement T0→T1 eviction
// TODO: Implement HNSW index for T1 similarity search
// TODO: Implement ghost frame extraction (R₀ gists)
// TODO: Implement WAL for crash recovery
