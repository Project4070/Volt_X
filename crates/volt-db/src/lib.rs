//! # volt-db
//!
//! VoltDB — the three-tier memory storage engine for Volt X.
//!
//! ## Memory Tiers
//!
//! - **T0 (Working Memory)**: 64 frames in RAM, instant access, ring buffer
//! - **T1 (Strand Storage)**: Frames in RAM organized by strand, persists to disk
//! - **T2 (Archive)**: Compressed frames on disk via LSM-Tree + mmap
//!
//! ## Indexing (Milestone 4.2)
//!
//! - **HNSW**: Per-strand approximate nearest-neighbour index over frame R₀ gists
//! - **Temporal**: B-tree index over `created_at` timestamps for range queries
//! - **Ghost Bleed**: Buffer of ~1000 R₀ gists for cross-attention in RAR
//!
//! ## Milestone 4.3: T2 + GC + WAL + Consolidation
//!
//! - **Compressed frames**: 4-tier decay (Full → Compressed → Gist → Tombstone)
//! - **LSM-Tree**: Memtable + mmap'd sorted runs + compaction
//! - **WAL**: Per-strand append-only log for crash recovery
//! - **GC**: Retention scoring with configurable decay thresholds
//! - **Consolidation**: Cluster detection + wisdom frame creation
//! - **Bloom filters**: Fast negative checks on sorted runs
//!
//! ## Usage
//!
//! The primary entry point is [`VoltStore`], which combines T0, T1, and T2
//! into a unified API with automatic eviction, indexing, and ghost bleed.
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

pub mod tier0;
pub mod tier1;
pub mod tier2;
pub mod gist;
pub mod hnsw_index;
pub mod temporal;
pub mod ghost;
pub mod compressed;
pub mod bloom;
pub mod wal;
pub mod gc;
pub mod consolidation;
mod store;

pub use store::{VoltStore, VoltStoreConfig, ConcurrentVoltStore};
pub use gist::{FrameGist, extract_gist};
pub use hnsw_index::{HnswIndex, SimilarityResult, StrandHnsw};
pub use temporal::TemporalIndex;
pub use ghost::{GhostBuffer, GhostEntry, BleedEngine, GHOST_BUFFER_CAPACITY};
pub use compressed::{
    CompressedFrame, CompressedSlot, GistFrame, Tombstone,
    DecayLevel, FrameEntry, compress, to_gist_frame, to_tombstone,
};
pub use bloom::BloomFilter;
pub use wal::{WalManager, WalEntry, WalOp};
pub use tier2::{Tier2Store, T2Config};
pub use gc::{GcEngine, GcConfig, GcResult, FrameGcMeta};
pub use consolidation::{
    ConsolidationEngine, ConsolidationConfig, ConsolidationResult, FrameCluster,
};
pub use volt_core;
