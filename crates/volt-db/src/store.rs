//! VoltStore — unified memory facade combining T0, T1, T2, and indices.
//!
//! VoltStore is the primary API for storing and retrieving TensorFrames.
//! It manages the T0 working memory (ring buffer), T1 strand storage,
//! T2 disk archive (LSM-Tree), handles automatic eviction from T0 → T1 → T2,
//! strand management, frame ID generation, persistence, HNSW semantic indexing,
//! temporal indexing, Ghost Bleed Engine, WAL crash recovery, garbage collection,
//! and frame consolidation.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use volt_core::{TensorFrame, VoltError, SLOT_DIM};

use crate::compressed::{compress, to_gist_frame, to_tombstone, DecayLevel, FrameEntry};
use crate::consolidation::{ConsolidationConfig, ConsolidationEngine, ConsolidationResult};
use crate::gc::{FrameGcMeta, GcConfig, GcEngine, GcResult};
use crate::ghost::{BleedEngine, GhostBuffer};
use crate::gist::{extract_gist, FrameGist};
use crate::hnsw_index::{HnswIndex, SimilarityResult};
use crate::temporal::TemporalIndex;
use crate::tier0::WorkingMemory;
use crate::tier1::StrandStore;
use crate::tier2::{T2Config, Tier2Store};
use crate::wal::{WalEntry, WalManager, WalOp};

/// Configuration for opening a disk-backed VoltStore.
///
/// # Example
///
/// ```no_run
/// use volt_db::VoltStoreConfig;
/// use std::path::PathBuf;
///
/// let config = VoltStoreConfig {
///     data_dir: PathBuf::from("/tmp/voltdb"),
///     ..VoltStoreConfig::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct VoltStoreConfig {
    /// Root data directory. T2 and WAL subdirectories are created beneath it.
    pub data_dir: PathBuf,
    /// Maximum frames in T1 before overflow to T2. Default: 1024.
    pub t1_overflow_threshold: usize,
    /// T2 storage configuration.
    pub t2_config: T2Config,
    /// GC configuration.
    pub gc_config: GcConfig,
    /// Consolidation configuration.
    pub consolidation_config: ConsolidationConfig,
}

impl Default for VoltStoreConfig {
    fn default() -> Self {
        let data_dir = PathBuf::from("voltdb_data");
        Self {
            t2_config: T2Config {
                data_dir: data_dir.join("t2"),
                ..T2Config::default()
            },
            data_dir,
            t1_overflow_threshold: 1024,
            gc_config: GcConfig::default(),
            consolidation_config: ConsolidationConfig::default(),
        }
    }
}

/// Unified memory facade combining T0 working memory, T1 strand storage,
/// T2 disk archive, HNSW semantic index, temporal index, Ghost Bleed Engine,
/// WAL crash recovery, GC, and frame consolidation.
///
/// # Frame Lifecycle
///
/// 1. Caller stores a frame via [`VoltStore::store`]
/// 2. Frame gets a unique `frame_id` and the active `strand_id`
/// 3. WAL logs the store operation (if disk-backed)
/// 4. Frame is placed in T0 (working memory)
/// 5. If T0 is full, the oldest frame is evicted to T1 (strand storage)
/// 6. R₀ gist is extracted and inserted into HNSW + temporal indices
/// 7. The Bleed Engine refreshes the ghost buffer with similar historical gists
/// 8. If T1 exceeds threshold, oldest frames overflow to T2 (compressed)
/// 9. GC periodically decays frames: Full → Compressed → Gist → Tombstone
///
/// # Memory-Only vs Disk-Backed
///
/// - [`VoltStore::new()`] creates a memory-only store (no T2, no WAL)
/// - [`VoltStore::open()`] creates a disk-backed store with T2 and WAL
///
/// # Example
///
/// ```
/// use volt_db::VoltStore;
/// use volt_core::TensorFrame;
///
/// let mut store = VoltStore::new();
/// store.create_strand(1).unwrap();
/// store.switch_strand(1).unwrap();
///
/// let frame = TensorFrame::new();
/// let id = store.store(frame).unwrap();
/// assert!(store.get_by_id(id).is_some());
/// ```
pub struct VoltStore {
    t0: WorkingMemory,
    t1: StrandStore,
    t2: Option<Tier2Store>,
    wal: Option<WalManager>,
    gc: GcEngine,
    consolidation: ConsolidationEngine,
    active_strand: u64,
    next_id: u64,
    hnsw: HnswIndex,
    temporal: TemporalIndex,
    bleed: BleedEngine,
    data_dir: Option<PathBuf>,
    t1_overflow_threshold: usize,
}

impl std::fmt::Debug for VoltStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoltStore")
            .field("t0_len", &self.t0.len())
            .field("t1_len", &self.t1.total_frame_count())
            .field("t2_entries", &self.t2.as_ref().map(|t| t.total_entries()))
            .field("active_strand", &self.active_strand)
            .field("next_id", &self.next_id)
            .field("hnsw_entries", &self.hnsw.total_entries())
            .field("temporal_entries", &self.temporal.len())
            .field("ghost_count", &self.bleed.buffer().len())
            .field("disk_backed", &self.data_dir.is_some())
            .finish()
    }
}

impl Default for VoltStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VoltStore {
    /// Creates a new memory-only VoltStore with empty T0 and T1.
    ///
    /// No T2 archive, no WAL. Use [`VoltStore::open`] for disk-backed mode.
    /// The default active strand is 0 (automatically created).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    ///
    /// let store = VoltStore::new();
    /// assert_eq!(store.active_strand(), 0);
    /// ```
    pub fn new() -> Self {
        let mut t1 = StrandStore::new();
        t1.create_strand(0);
        Self {
            t0: WorkingMemory::new(),
            t1,
            t2: None,
            wal: None,
            gc: GcEngine::with_defaults(),
            consolidation: ConsolidationEngine::with_defaults(),
            active_strand: 0,
            next_id: 1,
            hnsw: HnswIndex::new(),
            temporal: TemporalIndex::new(),
            bleed: BleedEngine::new(),
            data_dir: None,
            t1_overflow_threshold: 1024,
        }
    }

    /// Opens a disk-backed VoltStore with T2 archive and WAL.
    ///
    /// Creates the data directory structure, opens T2 and WAL,
    /// replays any WAL entries for crash recovery, and rebuilds
    /// HNSW and temporal indices from T1 data.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if directory creation,
    /// T2 open, WAL open, or WAL replay fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_db::{VoltStore, VoltStoreConfig};
    /// use std::path::PathBuf;
    ///
    /// let config = VoltStoreConfig {
    ///     data_dir: PathBuf::from("/tmp/voltdb"),
    ///     ..VoltStoreConfig::default()
    /// };
    /// let mut store = VoltStore::open(config).unwrap();
    /// ```
    pub fn open(config: VoltStoreConfig) -> Result<Self, VoltError> {
        std::fs::create_dir_all(&config.data_dir).map_err(|e| VoltError::StorageError {
            message: format!(
                "failed to create data directory {}: {e}",
                config.data_dir.display()
            ),
        })?;

        // Open T2
        let mut t2_config = config.t2_config.clone();
        t2_config.data_dir = config.data_dir.join("t2");
        let t2 = Tier2Store::open(t2_config)?;

        // Open WAL
        let wal_dir = config.data_dir.join("wal");
        let wal = WalManager::open(&wal_dir)?;

        // Load T1 if it exists
        let t1_path = config.data_dir.join("t1_strands.json");
        let mut t1 = if t1_path.exists() {
            // Spawn a thread with a larger stack for serde on Windows
            let t1_path_clone = t1_path.clone();
            let handle = std::thread::Builder::new()
                .stack_size(8 * 1024 * 1024)
                .spawn(move || StrandStore::load(&t1_path_clone))
                .map_err(|e| VoltError::StorageError {
                    message: format!("failed to spawn T1 load thread: {e}"),
                })?;
            handle.join().map_err(|_| VoltError::StorageError {
                message: "T1 load thread panicked".to_string(),
            })??
        } else {
            let mut t1 = StrandStore::new();
            t1.create_strand(0);
            t1
        };

        // Ensure strand 0 exists
        if !t1.has_strand(0) {
            t1.create_strand(0);
        }

        // Find max frame ID across T1 and T2
        let max_t1 = Self::find_max_frame_id(&t1);
        let max_t2 = t2
            .scan_all()
            .iter()
            .map(|e| e.frame_id())
            .max()
            .unwrap_or(0);
        let max_id = max_t1.max(max_t2);

        // Rebuild HNSW and temporal indices from T1
        let mut hnsw = HnswIndex::new();
        let mut temporal = TemporalIndex::new();
        for strand_id in t1.list_strands() {
            for frame in t1.get_by_strand(strand_id) {
                if let Some(gist) = extract_gist(frame)? {
                    hnsw.insert(&gist)?;
                    temporal.insert(gist.created_at, gist.frame_id);
                }
            }
        }

        // Replay WAL for crash recovery
        let wal_entries = wal.replay_all()?;
        let mut recovered_count = 0u64;
        for entries in wal_entries.values() {
            for entry in entries {
                if entry.op == WalOp::Store && !entry.payload.is_empty() {
                    // Try to parse as a FrameEntry and recover to T1
                    if let Ok(FrameEntry::Full(frame)) =
                        FrameEntry::from_bytes(&entry.payload)
                    {
                        let fid = frame.frame_meta.frame_id;
                        // Only recover if not already in T1
                        if t1.get_by_id(fid).is_none() {
                            if !t1.has_strand(frame.frame_meta.strand_id) {
                                t1.create_strand(frame.frame_meta.strand_id);
                            }
                            if let Some(gist) = extract_gist(&frame)? {
                                hnsw.insert(&gist)?;
                                temporal.insert(gist.created_at, gist.frame_id);
                            }
                            t1.store(*frame)?;
                            recovered_count += 1;
                        }
                    }
                }
            }
        }

        // Update max_id with recovered frames
        let final_max = if recovered_count > 0 {
            Self::find_max_frame_id(&t1).max(max_id)
        } else {
            max_id
        };

        Ok(Self {
            t0: WorkingMemory::new(),
            t1,
            t2: Some(t2),
            wal: Some(wal),
            gc: GcEngine::new(config.gc_config),
            consolidation: ConsolidationEngine::new(config.consolidation_config),
            active_strand: 0,
            next_id: final_max + 1,
            hnsw,
            temporal,
            bleed: BleedEngine::new(),
            data_dir: Some(config.data_dir),
            t1_overflow_threshold: config.t1_overflow_threshold,
        })
    }

    /// Stores a frame, assigning it a unique frame ID and the active strand ID.
    ///
    /// The frame is placed in T0. If T0 is full, the oldest frame is
    /// evicted to T1 automatically. The frame's R₀ gist (if present) is
    /// extracted and inserted into the HNSW and temporal indices, and the
    /// Bleed Engine refreshes the ghost buffer.
    ///
    /// In disk-backed mode, the frame is also WAL-logged and T1 overflow
    /// to T2 is checked.
    ///
    /// Returns the assigned frame ID.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    /// use volt_core::TensorFrame;
    ///
    /// let mut store = VoltStore::new();
    /// let id = store.store(TensorFrame::new()).unwrap();
    /// assert_eq!(id, 1);
    /// assert!(store.get_by_id(id).is_some());
    /// ```
    pub fn store(&mut self, mut frame: TensorFrame) -> Result<u64, VoltError> {
        let frame_id = self.next_id;
        self.next_id += 1;

        frame.frame_meta.frame_id = frame_id;
        frame.frame_meta.strand_id = self.active_strand;

        // WAL log if disk-backed
        // Build payload directly to avoid cloning 64KB TensorFrame on the stack
        // (Windows default thread stack is 1MB; serde + clone can overflow).
        if let Some(ref mut wal) = self.wal {
            let mut payload = vec![DecayLevel::Full.tag()];
            let json = serde_json::to_vec(&frame).map_err(|e| VoltError::StorageError {
                message: format!("failed to serialize frame for WAL: {e}"),
            })?;
            payload.extend_from_slice(&json);
            wal.log_entry(WalEntry {
                frame_id,
                strand_id: self.active_strand,
                op: WalOp::Store,
                payload,
            })?;
        }

        // Extract gist before storing (we need the frame reference)
        let gist = extract_gist(&frame)?;

        if let Some(evicted) = self.t0.store(frame) {
            self.t1.store(evicted)?;
        }

        // Update indices with gist
        if let Some(ref g) = gist {
            self.hnsw.insert(g)?;
            self.temporal.insert(g.created_at, frame_id);
            self.bleed.on_new_frame(g, &self.hnsw)?;
        }

        // T1 → T2 overflow check
        if self.t2.is_some()
            && self.t1.total_frame_count() > self.t1_overflow_threshold
        {
            self.maybe_overflow_t1_to_t2()?;
        }

        // T2 maintenance
        if let Some(ref mut t2) = self.t2 {
            t2.maybe_flush_and_compact()?;
        }

        Ok(frame_id)
    }

    /// Retrieves a frame by its `frame_id`, searching T0 first, then T1.
    ///
    /// Does **not** search T2 (compressed frames). Use [`get_entry_by_id`]
    /// to search all tiers including compressed/gist forms.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    /// use volt_core::TensorFrame;
    ///
    /// let mut store = VoltStore::new();
    /// let id = store.store(TensorFrame::new()).unwrap();
    /// assert!(store.get_by_id(id).is_some());
    /// assert!(store.get_by_id(9999).is_none());
    /// ```
    pub fn get_by_id(&self, frame_id: u64) -> Option<&TensorFrame> {
        self.t0
            .get_by_id(frame_id)
            .or_else(|| self.t1.get_by_id(frame_id))
    }

    /// Retrieves a frame entry at any decay level from T0, T1, or T2.
    ///
    /// Search order: T0 → T1 → T2 (memtable → sorted runs).
    /// Returns `Full` for T0/T1 frames, or `Compressed`/`Gist`/`Tombstone`
    /// for T2 frames.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    /// use volt_core::TensorFrame;
    /// use volt_db::compressed::DecayLevel;
    ///
    /// let mut store = VoltStore::new();
    /// let id = store.store(TensorFrame::new()).unwrap();
    /// let entry = store.get_entry_by_id(id).unwrap();
    /// assert_eq!(entry.decay_level(), DecayLevel::Full);
    /// ```
    pub fn get_entry_by_id(&self, frame_id: u64) -> Option<FrameEntry> {
        // Check T0
        if let Some(frame) = self.t0.get_by_id(frame_id) {
            return Some(FrameEntry::Full(Box::new(frame.clone())));
        }
        // Check T1
        if let Some(frame) = self.t1.get_by_id(frame_id) {
            return Some(FrameEntry::Full(Box::new(frame.clone())));
        }
        // Check T2
        if let Some(ref t2) = self.t2 {
            return t2.get(frame_id);
        }
        None
    }

    /// Returns all frames belonging to a strand, from both T0 and T1.
    ///
    /// Results are ordered: T1 frames first (oldest), then T0 frames.
    /// Does not include T2 compressed frames.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    /// use volt_core::TensorFrame;
    ///
    /// let mut store = VoltStore::new();
    /// store.store(TensorFrame::new()).unwrap();
    /// store.store(TensorFrame::new()).unwrap();
    ///
    /// let frames = store.get_by_strand(0);
    /// assert_eq!(frames.len(), 2);
    /// ```
    pub fn get_by_strand(&self, strand_id: u64) -> Vec<&TensorFrame> {
        let mut frames: Vec<&TensorFrame> = Vec::new();
        // T1 first (older frames)
        frames.extend(self.t1.get_by_strand(strand_id));
        // T0 second (newer frames)
        frames.extend(self.t0.get_by_strand(strand_id));
        frames
    }

    /// Returns the most recent `n` frames from T0, newest-first.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    /// use volt_core::TensorFrame;
    ///
    /// let mut store = VoltStore::new();
    /// for _ in 0..5 {
    ///     store.store(TensorFrame::new()).unwrap();
    /// }
    ///
    /// let recent = store.recent(3);
    /// assert_eq!(recent.len(), 3);
    /// ```
    pub fn recent(&self, n: usize) -> Vec<&TensorFrame> {
        self.t0.recent(n)
    }

    /// Creates a new strand.
    ///
    /// Returns an error if the strand already exists.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    ///
    /// let mut store = VoltStore::new();
    /// store.create_strand(42).unwrap();
    /// assert!(store.list_strands().contains(&42));
    /// ```
    pub fn create_strand(&mut self, strand_id: u64) -> Result<(), VoltError> {
        if self.t1.has_strand(strand_id) {
            return Err(VoltError::StrandError {
                strand_id,
                message: "strand already exists".to_string(),
            });
        }
        self.t1.create_strand(strand_id);
        Ok(())
    }

    /// Switches the active strand.
    ///
    /// New frames will be stored under this strand ID.
    /// The strand is created automatically if it doesn't exist.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    ///
    /// let mut store = VoltStore::new();
    /// store.switch_strand(42).unwrap();
    /// assert_eq!(store.active_strand(), 42);
    /// ```
    pub fn switch_strand(&mut self, strand_id: u64) -> Result<(), VoltError> {
        if !self.t1.has_strand(strand_id) {
            self.t1.create_strand(strand_id);
        }
        self.active_strand = strand_id;
        Ok(())
    }

    /// Returns the currently active strand ID.
    pub fn active_strand(&self) -> u64 {
        self.active_strand
    }

    /// Returns a sorted list of all strand IDs.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    ///
    /// let mut store = VoltStore::new();
    /// store.create_strand(3).unwrap();
    /// store.create_strand(1).unwrap();
    ///
    /// let strands = store.list_strands();
    /// assert!(strands.contains(&0)); // default strand
    /// assert!(strands.contains(&1));
    /// assert!(strands.contains(&3));
    /// ```
    pub fn list_strands(&self) -> Vec<u64> {
        self.t1.list_strands()
    }

    /// Returns the number of frames in T0 working memory.
    pub fn t0_len(&self) -> usize {
        self.t0.len()
    }

    /// Returns the total number of frames in T1 strand storage.
    pub fn t1_len(&self) -> usize {
        self.t1.total_frame_count()
    }

    /// Returns the total number of entries in T2 archive.
    pub fn t2_len(&self) -> usize {
        self.t2.as_ref().map(|t| t.total_entries()).unwrap_or(0)
    }

    /// Returns the total number of frames across T0 and T1.
    pub fn total_frame_count(&self) -> usize {
        self.t0.len() + self.t1.total_frame_count()
    }

    /// Returns the total number of entries across all tiers (T0 + T1 + T2).
    pub fn total_entry_count(&self) -> usize {
        self.t0.len()
            + self.t1.total_frame_count()
            + self.t2.as_ref().map(|t| t.total_entries()).unwrap_or(0)
    }

    /// Returns a reference to the T0 working memory.
    pub fn t0(&self) -> &WorkingMemory {
        &self.t0
    }

    /// Returns a reference to the T1 strand store.
    pub fn t1(&self) -> &StrandStore {
        &self.t1
    }

    // --- Semantic search (Milestone 4.2) ---

    /// Queries ALL strands for the top-k most similar frames by R₀ gist.
    ///
    /// Returns results sorted by ascending cosine distance (closest first).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    /// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
    ///
    /// let mut store = VoltStore::new();
    /// let mut frame = TensorFrame::new();
    /// let mut slot = SlotData::new(SlotRole::Agent);
    /// slot.write_resolution(0, [0.1; SLOT_DIM]);
    /// frame.write_slot(0, slot).unwrap();
    /// store.store(frame).unwrap();
    ///
    /// let results = store.query_similar(&[0.1; SLOT_DIM], 10);
    /// assert_eq!(results.len(), 1);
    /// ```
    pub fn query_similar(&self, query: &[f32; SLOT_DIM], k: usize) -> Vec<SimilarityResult> {
        self.hnsw.query_all(query, k)
    }

    /// Queries a single strand for the top-k most similar frames by R₀ gist.
    ///
    /// Returns an empty vec if the strand has no indexed frames.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    /// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
    ///
    /// let mut store = VoltStore::new();
    /// let mut frame = TensorFrame::new();
    /// let mut slot = SlotData::new(SlotRole::Agent);
    /// slot.write_resolution(0, [0.1; SLOT_DIM]);
    /// frame.write_slot(0, slot).unwrap();
    /// store.store(frame).unwrap();
    ///
    /// let results = store.query_similar_in_strand(0, &[0.1; SLOT_DIM], 10);
    /// assert_eq!(results.len(), 1);
    /// ```
    pub fn query_similar_in_strand(
        &self,
        strand_id: u64,
        query: &[f32; SLOT_DIM],
        k: usize,
    ) -> Vec<SimilarityResult> {
        self.hnsw.query_strand(strand_id, query, k)
    }

    /// Returns all frame IDs created within the time range `[start, end]` inclusive.
    ///
    /// Timestamps are in microseconds.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    /// use volt_core::TensorFrame;
    ///
    /// let store = VoltStore::new();
    /// let results = store.query_time_range(0, u64::MAX);
    /// assert!(results.is_empty());
    /// ```
    pub fn query_time_range(&self, start: u64, end: u64) -> Vec<u64> {
        self.temporal.query_range(start, end)
    }

    /// Returns a reference to the Ghost Bleed Buffer.
    ///
    /// The buffer contains R₀ gists from historical frames that are
    /// semantically similar to the most recently stored frame. These
    /// gists are intended to be passed to the Soft Core's RAR Attend
    /// phase as additional Key/Value sources.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    ///
    /// let store = VoltStore::new();
    /// assert!(store.ghost_buffer().is_empty());
    /// ```
    pub fn ghost_buffer(&self) -> &GhostBuffer {
        self.bleed.buffer()
    }

    /// Returns ghost gist vectors for consumption by the Soft Core.
    ///
    /// This is a convenience method that extracts just the `[f32; 256]`
    /// vectors from the ghost buffer, suitable for passing directly
    /// to the RAR Attend phase.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    ///
    /// let store = VoltStore::new();
    /// let gists = store.ghost_gists();
    /// assert!(gists.is_empty());
    /// ```
    pub fn ghost_gists(&self) -> Vec<[f32; SLOT_DIM]> {
        self.bleed.buffer().gist_vectors()
    }

    /// Returns the total number of entries in the HNSW index.
    pub fn hnsw_entries(&self) -> usize {
        self.hnsw.total_entries()
    }

    /// Returns the total number of entries in the temporal index.
    pub fn temporal_entries(&self) -> usize {
        self.temporal.len()
    }

    // --- GC (Milestone 4.3) ---

    /// Runs a garbage collection pass across T1 and T2.
    ///
    /// Evaluates retention scores for all frames and demotes those
    /// below the configured thresholds:
    /// - Full → Compressed: removes from T1, compresses, inserts to T2
    /// - Compressed → Gist: converts in T2
    /// - Gist → Tombstoned: updates T2
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if T2 operations fail.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    ///
    /// let mut store = VoltStore::new();
    /// let result = store.run_gc().unwrap();
    /// assert_eq!(result.frames_preserved, 0);
    /// ```
    pub fn run_gc(&mut self) -> Result<GcResult, VoltError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        self.run_gc_at(now)
    }

    /// Runs GC with a specified timestamp (useful for testing).
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if T2 operations fail.
    pub fn run_gc_at(&mut self, now: u64) -> Result<GcResult, VoltError> {
        let mut result = GcResult::default();

        // Collect metadata from T1 frames
        let mut gc_metas: Vec<FrameGcMeta> = Vec::new();
        for strand_id in self.t1.list_strands() {
            for frame in self.t1.get_by_strand(strand_id) {
                gc_metas.push(FrameGcMeta {
                    frame_id: frame.frame_meta.frame_id,
                    strand_id: frame.frame_meta.strand_id,
                    created_at: frame.frame_meta.created_at,
                    global_certainty: frame.frame_meta.global_certainty,
                    current_level: DecayLevel::Full,
                    reference_count: 0,
                    is_pinned: self.gc.is_pinned(frame.frame_meta.frame_id),
                    is_wisdom: false,
                });
            }
        }

        // Collect metadata from T2 entries
        if let Some(ref t2) = self.t2 {
            for entry in t2.scan_all() {
                let fid = entry.frame_id();
                gc_metas.push(FrameGcMeta {
                    frame_id: fid,
                    strand_id: entry.strand_id(),
                    created_at: entry.created_at(),
                    global_certainty: entry.global_certainty(),
                    current_level: entry.decay_level(),
                    reference_count: 0,
                    is_pinned: self.gc.is_pinned(fid),
                    is_wisdom: false,
                });
            }
        }

        // Evaluate
        let demotions = self.gc.evaluate(&gc_metas, now);

        // Apply demotions
        for (frame_id, target_level) in demotions {
            // Find current level
            let current_level = gc_metas
                .iter()
                .find(|m| m.frame_id == frame_id)
                .map(|m| m.current_level)
                .unwrap_or(DecayLevel::Full);

            match (current_level, target_level) {
                (DecayLevel::Full, DecayLevel::Compressed) => {
                    // Remove from T1, compress, insert to T2
                    if let Some(frame) = self.t1.remove_frame(frame_id) {
                        let compressed = compress(&frame);

                        // WAL log
                        if let Some(ref mut wal) = self.wal {
                            let payload =
                                FrameEntry::Compressed(compressed.clone()).to_bytes()?;
                            wal.log_entry(WalEntry {
                                frame_id,
                                strand_id: frame.frame_meta.strand_id,
                                op: WalOp::Compress,
                                payload,
                            })?;
                        }

                        if let Some(ref mut t2) = self.t2 {
                            t2.insert(FrameEntry::Compressed(compressed))?;
                        }

                        // Update indices
                        self.hnsw.mark_deleted(frame_id);
                        self.temporal.remove(frame_id);
                        result.frames_compressed += 1;
                    }
                }
                (DecayLevel::Full, DecayLevel::Gist | DecayLevel::Tombstoned) => {
                    // Full → skip to Gist or Tombstone (multi-step demotion)
                    if let Some(frame) = self.t1.remove_frame(frame_id) {
                        let compressed = compress(&frame);
                        let gist_vector =
                            extract_gist_vector_from_compressed(&compressed);
                        let gist_frame = to_gist_frame(&compressed, gist_vector);

                        if target_level == DecayLevel::Tombstoned {
                            let ts = to_tombstone(
                                frame_id,
                                frame.frame_meta.strand_id,
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|d| d.as_micros() as u64)
                                    .unwrap_or(0),
                                None,
                            );
                            if let Some(ref mut t2) = self.t2 {
                                t2.insert(FrameEntry::Tombstone(ts))?;
                            }
                            result.frames_tombstoned += 1;
                        } else {
                            if let Some(ref mut t2) = self.t2 {
                                t2.insert(FrameEntry::Gist(gist_frame))?;
                            }
                            result.frames_gisted += 1;
                        }

                        self.hnsw.mark_deleted(frame_id);
                        self.temporal.remove(frame_id);
                    }
                }
                (DecayLevel::Compressed, DecayLevel::Gist) => {
                    // Convert in T2
                    if let Some(ref mut t2) = self.t2
                        && let Some(FrameEntry::Compressed(ref c)) = t2.get(frame_id)
                    {
                        let gist_vector =
                            extract_gist_vector_from_compressed(c);
                        let gist_frame = to_gist_frame(c, gist_vector);
                        t2.update(FrameEntry::Gist(gist_frame))?;
                        result.frames_gisted += 1;
                    }
                }
                (DecayLevel::Compressed, DecayLevel::Tombstoned) => {
                    if let Some(ref mut t2) = self.t2
                        && let Some(entry) = t2.get(frame_id)
                    {
                        let strand_id = entry.strand_id();
                        let ts = to_tombstone(
                            frame_id,
                            strand_id,
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_micros() as u64)
                                .unwrap_or(0),
                            None,
                        );
                        t2.update(FrameEntry::Tombstone(ts))?;
                        result.frames_tombstoned += 1;
                    }
                }
                (DecayLevel::Gist, DecayLevel::Tombstoned) => {
                    if let Some(ref mut t2) = self.t2
                        && let Some(entry) = t2.get(frame_id)
                    {
                        let strand_id = entry.strand_id();
                        let ts = to_tombstone(
                            frame_id,
                            strand_id,
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_micros() as u64)
                                .unwrap_or(0),
                            None,
                        );
                        t2.update(FrameEntry::Tombstone(ts))?;
                        result.frames_tombstoned += 1;
                    }
                }
                _ => {
                    // No-op for same level or unexpected transitions
                }
            }
        }

        let total_actions =
            result.frames_compressed + result.frames_gisted + result.frames_tombstoned;
        result.frames_preserved = gc_metas.len().saturating_sub(total_actions);

        Ok(result)
    }

    /// Consolidates a strand by finding clusters and creating wisdom frames.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if wisdom frame storage fails.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    ///
    /// let mut store = VoltStore::new();
    /// let result = store.consolidate_strand(0).unwrap();
    /// assert_eq!(result.clusters_found, 0);
    /// ```
    pub fn consolidate_strand(
        &mut self,
        strand_id: u64,
    ) -> Result<ConsolidationResult, VoltError> {
        // Collect gists for this strand from T1
        let mut gists: Vec<FrameGist> = Vec::new();
        for frame in self.t1.get_by_strand(strand_id) {
            if let Some(gist) = extract_gist(frame)? {
                gists.push(gist);
            }
        }

        // Find clusters
        let clusters = self.consolidation.find_clusters(strand_id, &self.hnsw, &gists);

        if clusters.is_empty() {
            return Ok(ConsolidationResult {
                clusters_found: 0,
                wisdom_frames: Vec::new(),
                superseded_frame_ids: Vec::new(),
            });
        }

        let mut wisdom_frames = Vec::new();
        let mut superseded_ids = Vec::new();

        for cluster in &clusters {
            // Gather source frames for this cluster (clone to release borrow on self)
            let source_frames: Vec<TensorFrame> = cluster
                .member_frame_ids
                .iter()
                .filter_map(|&id| self.get_by_id(id).cloned())
                .collect();

            if source_frames.len() < self.consolidation.config().min_cluster_size {
                continue;
            }

            // Assign an ID to the wisdom frame
            let wisdom_id = self.next_id;
            self.next_id += 1;

            let source_refs: Vec<&TensorFrame> = source_frames.iter().collect();
            let wisdom = self.consolidation.create_wisdom_frame(
                cluster,
                &source_refs,
                strand_id,
                wisdom_id,
            );

            // Store the wisdom frame
            let gist = extract_gist(&wisdom)?;
            if let Some(evicted) = self.t0.store(wisdom.clone()) {
                self.t1.store(evicted)?;
            }

            if let Some(ref g) = gist {
                self.hnsw.insert(g)?;
                self.temporal.insert(g.created_at, wisdom_id);
            }

            superseded_ids.extend_from_slice(&cluster.member_frame_ids);
            wisdom_frames.push(wisdom);
        }

        Ok(ConsolidationResult {
            clusters_found: clusters.len(),
            wisdom_frames,
            superseded_frame_ids: superseded_ids,
        })
    }

    /// Pins a frame so it is never garbage collected.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    /// use volt_core::TensorFrame;
    ///
    /// let mut store = VoltStore::new();
    /// let id = store.store(TensorFrame::new()).unwrap();
    /// store.pin_frame(id);
    /// assert!(store.is_frame_pinned(id));
    /// ```
    pub fn pin_frame(&mut self, frame_id: u64) {
        self.gc.pin_frame(frame_id);
    }

    /// Unpins a frame, making it eligible for GC.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    /// use volt_core::TensorFrame;
    ///
    /// let mut store = VoltStore::new();
    /// let id = store.store(TensorFrame::new()).unwrap();
    /// store.pin_frame(id);
    /// store.unpin_frame(id);
    /// assert!(!store.is_frame_pinned(id));
    /// ```
    pub fn unpin_frame(&mut self, frame_id: u64) {
        self.gc.unpin_frame(frame_id);
    }

    /// Checks whether a frame is pinned.
    pub fn is_frame_pinned(&self, frame_id: u64) -> bool {
        self.gc.is_pinned(frame_id)
    }

    /// Returns whether the store is disk-backed (has T2 and WAL).
    pub fn is_disk_backed(&self) -> bool {
        self.data_dir.is_some()
    }

    /// Reassigns a frame from its current strand to a new strand.
    ///
    /// Removes the frame from T1, updates its `strand_id`, and stores
    /// it back. Also updates the HNSW index so the gist moves to the
    /// new strand's partition. The target strand is created if it does
    /// not exist.
    ///
    /// Only operates on T1 frames. T0 frames are ephemeral and should
    /// not be reassigned.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if the frame is not found in T1,
    /// or if re-insertion fails.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::VoltStore;
    /// use volt_core::TensorFrame;
    ///
    /// let mut store = VoltStore::new();
    /// store.create_strand(1).unwrap();
    /// store.switch_strand(1).unwrap();
    ///
    /// // Fill T0 so frames overflow to T1
    /// let mut target_id = 0;
    /// for _ in 0..65 {
    ///     target_id = store.store(TensorFrame::new()).unwrap();
    /// }
    ///
    /// store.create_strand(2).unwrap();
    /// // The first frame should now be in T1
    /// let first_t1 = store.get_by_strand(1).first()
    ///     .map(|f| f.frame_meta.frame_id).unwrap();
    /// store.reassign_frame_strand(first_t1, 2).unwrap();
    /// ```
    pub fn reassign_frame_strand(
        &mut self,
        frame_id: u64,
        new_strand_id: u64,
    ) -> Result<(), VoltError> {
        // Remove from T1
        let mut frame = self.t1.remove_frame(frame_id).ok_or_else(|| {
            VoltError::StorageError {
                message: format!(
                    "reassign_frame_strand: frame {frame_id} not found in T1"
                ),
            }
        })?;

        // Update strand_id on the frame
        frame.frame_meta.strand_id = new_strand_id;

        // Ensure target strand exists
        if !self.t1.has_strand(new_strand_id) {
            self.t1.create_strand(new_strand_id);
        }

        // Remove old gist from HNSW and re-insert with new strand
        self.hnsw.mark_deleted(frame_id);
        if let Some(gist) = extract_gist(&frame)? {
            self.hnsw.insert(&gist)?;
        }

        // Store back in T1
        self.t1.store(*frame)?;

        Ok(())
    }

    // --- Persistence ---

    /// Saves T1 strand storage to disk for persistence across restarts.
    ///
    /// T0 is intentionally not saved — it's ephemeral working memory.
    /// HNSW and temporal indices are rebuilt on load from T1 data.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if serialization or I/O fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_db::VoltStore;
    /// use std::path::Path;
    ///
    /// let store = VoltStore::new();
    /// store.save(Path::new("voltdb_t1.json")).unwrap();
    /// ```
    pub fn save(&self, path: &Path) -> Result<(), VoltError> {
        self.t1.save(path)
    }

    /// Loads T1 strand storage from disk, creating a fresh T0.
    ///
    /// Rebuilds the HNSW and temporal indices by scanning all T1 frames.
    /// The active strand is set to the default (0). If strand 0
    /// doesn't exist in the loaded data, it is created.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if I/O or deserialization fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_db::VoltStore;
    /// use std::path::Path;
    ///
    /// let store = VoltStore::load(Path::new("voltdb_t1.json")).unwrap();
    /// ```
    pub fn load(path: &Path) -> Result<Self, VoltError> {
        let mut t1 = StrandStore::load(path)?;
        if !t1.has_strand(0) {
            t1.create_strand(0);
        }

        // Find the highest frame_id in T1 to set next_id correctly
        let max_id = Self::find_max_frame_id(&t1);

        // Rebuild HNSW and temporal indices from T1 frames
        let mut hnsw = HnswIndex::new();
        let mut temporal = TemporalIndex::new();
        for strand_id in t1.list_strands() {
            for frame in t1.get_by_strand(strand_id) {
                if let Some(gist) = extract_gist(frame)? {
                    hnsw.insert(&gist)?;
                    temporal.insert(gist.created_at, gist.frame_id);
                }
            }
        }

        Ok(Self {
            t0: WorkingMemory::new(),
            t1,
            t2: None,
            wal: None,
            gc: GcEngine::with_defaults(),
            consolidation: ConsolidationEngine::with_defaults(),
            active_strand: 0,
            next_id: max_id + 1,
            hnsw,
            temporal,
            bleed: BleedEngine::new(),
            data_dir: None,
            t1_overflow_threshold: 1024,
        })
    }

    /// Scans T1 to find the highest frame_id for ID generation continuity.
    fn find_max_frame_id(t1: &StrandStore) -> u64 {
        let mut max = 0u64;
        for strand_id in t1.list_strands() {
            for frame in t1.get_by_strand(strand_id) {
                if frame.frame_meta.frame_id > max {
                    max = frame.frame_meta.frame_id;
                }
            }
        }
        max
    }

    /// Overflows the oldest T1 frames to T2 (compressed).
    fn maybe_overflow_t1_to_t2(&mut self) -> Result<(), VoltError> {
        let overflow_count = self
            .t1
            .total_frame_count()
            .saturating_sub(self.t1_overflow_threshold);

        if overflow_count == 0 {
            return Ok(());
        }

        // Get oldest frame IDs
        let oldest_ids = self.t1.oldest_frame_ids(overflow_count);

        for frame_id in oldest_ids {
            if let Some(frame) = self.t1.remove_frame(frame_id) {
                let compressed = compress(&frame);

                // WAL log if enabled
                if let Some(ref mut wal) = self.wal {
                    let payload =
                        FrameEntry::Compressed(compressed.clone()).to_bytes()?;
                    wal.log_entry(WalEntry {
                        frame_id,
                        strand_id: frame.frame_meta.strand_id,
                        op: WalOp::Compress,
                        payload,
                    })?;
                }

                if let Some(ref mut t2) = self.t2 {
                    t2.insert(FrameEntry::Compressed(compressed))?;
                }

                // Mark deleted in HNSW (frame is no longer in Full form)
                self.hnsw.mark_deleted(frame_id);
                self.temporal.remove(frame_id);
            }
        }

        Ok(())
    }
}

/// Extracts a gist vector from a CompressedFrame by averaging R₀ slots.
fn extract_gist_vector_from_compressed(
    compressed: &crate::compressed::CompressedFrame,
) -> [f32; SLOT_DIM] {
    let mut sum = [0.0f32; SLOT_DIM];
    let mut count = 0u32;

    for s in compressed.slots.iter().flatten() {
        if let Some(r0) = &s.r0 {
            for (d, &v) in sum.iter_mut().zip(r0.iter()) {
                *d += v;
            }
            count += 1;
        }
    }

    if count > 0 {
        let n = count as f32;
        for d in &mut sum {
            *d /= n;
        }
        // L2 normalize
        let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for d in &mut sum {
                *d /= norm;
            }
        }
    }

    sum
}

/// Thread-safe concurrent wrapper around VoltStore.
///
/// Wraps `VoltStore` in `Arc<RwLock<_>>` for concurrent access.
/// Provides read/write access patterns for the "10 readers + 1 writer" use case.
///
/// # Example
///
/// ```
/// use volt_db::{VoltStore, ConcurrentVoltStore};
///
/// let store = VoltStore::new();
/// let concurrent = ConcurrentVoltStore::new(store);
///
/// // Read access (many concurrent readers)
/// let guard = concurrent.read().unwrap();
/// assert_eq!(guard.active_strand(), 0);
/// ```
#[derive(Debug, Clone)]
pub struct ConcurrentVoltStore {
    inner: Arc<RwLock<VoltStore>>,
}

impl ConcurrentVoltStore {
    /// Creates a new concurrent store from an existing VoltStore.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::{VoltStore, ConcurrentVoltStore};
    ///
    /// let store = VoltStore::new();
    /// let concurrent = ConcurrentVoltStore::new(store);
    /// ```
    pub fn new(store: VoltStore) -> Self {
        Self {
            inner: Arc::new(RwLock::new(store)),
        }
    }

    /// Acquires a read lock on the store.
    ///
    /// Multiple readers can hold read locks simultaneously.
    ///
    /// # Errors
    ///
    /// Returns an error if the lock is poisoned.
    pub fn read(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, VoltStore>, VoltError> {
        self.inner.read().map_err(|e| VoltError::StorageError {
            message: format!("VoltStore read lock poisoned: {e}"),
        })
    }

    /// Acquires a write lock on the store.
    ///
    /// Only one writer can hold the lock at a time, and it
    /// excludes all readers.
    ///
    /// # Errors
    ///
    /// Returns an error if the lock is poisoned.
    pub fn write(
        &self,
    ) -> Result<std::sync::RwLockWriteGuard<'_, VoltStore>, VoltError> {
        self.inner.write().map_err(|e| VoltError::StorageError {
            message: format!("VoltStore write lock poisoned: {e}"),
        })
    }

    /// Returns a clone of the inner `Arc<RwLock<VoltStore>>`.
    ///
    /// Useful for passing to components that require raw Arc access,
    /// such as the sleep consolidation scheduler.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::{VoltStore, ConcurrentVoltStore};
    ///
    /// let concurrent = ConcurrentVoltStore::new(VoltStore::new());
    /// let arc = concurrent.inner_arc();
    /// let guard = arc.read().unwrap();
    /// assert_eq!(guard.active_strand(), 0);
    /// ```
    pub fn inner_arc(&self) -> Arc<RwLock<VoltStore>> {
        Arc::clone(&self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tier0::T0_CAPACITY;
    use volt_core::{SlotData, SlotRole, SLOT_DIM};

    fn make_frame_with_content() -> TensorFrame {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [0.5; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();
        frame
    }

    #[test]
    fn new_store_has_default_strand() {
        let store = VoltStore::new();
        assert_eq!(store.active_strand(), 0);
        assert!(store.list_strands().contains(&0));
    }

    #[test]
    fn store_assigns_unique_ids() {
        let mut store = VoltStore::new();
        let id1 = store.store(TensorFrame::new()).unwrap();
        let id2 = store.store(TensorFrame::new()).unwrap();
        assert_ne!(id1, id2);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn store_assigns_active_strand() {
        let mut store = VoltStore::new();
        store.switch_strand(42).unwrap();

        let id = store.store(TensorFrame::new()).unwrap();
        let frame = store.get_by_id(id).unwrap();
        assert_eq!(frame.frame_meta.strand_id, 42);
    }

    #[test]
    fn get_by_id_searches_t0_and_t1() {
        let mut store = VoltStore::new();

        // Store enough frames to force eviction to T1
        let mut first_id = 0;
        for i in 0..(T0_CAPACITY + 10) {
            let id = store.store(make_frame_with_content()).unwrap();
            if i == 0 {
                first_id = id;
            }
        }

        // First frame should have been evicted to T1
        assert!(store.get_by_id(first_id).is_some());
        // Last frame should be in T0
        let last_id = first_id + T0_CAPACITY as u64 + 9;
        assert!(store.get_by_id(last_id).is_some());
    }

    #[test]
    fn eviction_from_t0_to_t1() {
        let mut store = VoltStore::new();

        // Fill T0
        for _ in 0..T0_CAPACITY {
            store.store(make_frame_with_content()).unwrap();
        }
        assert_eq!(store.t0_len(), T0_CAPACITY);
        assert_eq!(store.t1_len(), 0);

        // One more triggers eviction
        store.store(make_frame_with_content()).unwrap();
        assert_eq!(store.t0_len(), T0_CAPACITY);
        assert_eq!(store.t1_len(), 1);
    }

    #[test]
    fn switch_strand_creates_if_needed() {
        let mut store = VoltStore::new();
        store.switch_strand(99).unwrap();
        assert_eq!(store.active_strand(), 99);
        assert!(store.list_strands().contains(&99));
    }

    #[test]
    fn create_strand_rejects_duplicates() {
        let mut store = VoltStore::new();
        store.create_strand(5).unwrap();
        let result = store.create_strand(5);
        assert!(result.is_err());
    }

    #[test]
    fn get_by_strand_combines_t0_and_t1() {
        let mut store = VoltStore::new();
        store.switch_strand(1).unwrap();

        // Store enough to have some in T1 and some in T0
        for _ in 0..(T0_CAPACITY + 5) {
            store.store(make_frame_with_content()).unwrap();
        }

        let frames = store.get_by_strand(1);
        assert_eq!(frames.len(), T0_CAPACITY + 5);
    }

    #[test]
    fn recent_returns_from_t0() {
        let mut store = VoltStore::new();
        for _ in 0..10 {
            store.store(make_frame_with_content()).unwrap();
        }

        let recent = store.recent(3);
        assert_eq!(recent.len(), 3);
        // Most recent should have the highest ID
        assert_eq!(recent[0].frame_meta.frame_id, 10);
        assert_eq!(recent[1].frame_meta.frame_id, 9);
        assert_eq!(recent[2].frame_meta.frame_id, 8);
    }

    #[test]
    fn multiple_strands_independent() {
        let mut store = VoltStore::new();

        store.switch_strand(1).unwrap();
        store.store(make_frame_with_content()).unwrap();
        store.store(make_frame_with_content()).unwrap();

        store.switch_strand(2).unwrap();
        store.store(make_frame_with_content()).unwrap();

        assert_eq!(store.get_by_strand(1).len(), 2);
        assert_eq!(store.get_by_strand(2).len(), 1);
    }

    #[test]
    fn total_frame_count() {
        let mut store = VoltStore::new();
        for _ in 0..(T0_CAPACITY + 10) {
            store.store(make_frame_with_content()).unwrap();
        }
        assert_eq!(store.total_frame_count(), T0_CAPACITY + 10);
    }

    #[test]
    fn save_and_load_preserves_t1() {
        let mut store = VoltStore::new();
        store.switch_strand(1).unwrap();

        // Store enough to get some into T1
        for _ in 0..(T0_CAPACITY + 5) {
            store.store(make_frame_with_content()).unwrap();
        }
        let t1_count = store.t1_len();
        assert!(t1_count > 0);

        let dir = std::env::temp_dir().join("volt_db_test_store_42");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_store.json");

        store.save(&path).unwrap();
        let loaded = VoltStore::load(&path).unwrap();

        // T1 should be preserved
        assert_eq!(loaded.t1_len(), t1_count);
        // T0 should be empty (fresh after load)
        assert_eq!(loaded.t0_len(), 0);
        // All T1 frames should be retrievable
        for id in 1..=(t1_count as u64) {
            assert!(
                loaded.get_by_id(id).is_some(),
                "frame {id} not found after load"
            );
        }

        // HNSW and temporal indices should be rebuilt
        assert_eq!(loaded.hnsw_entries(), t1_count);
        assert_eq!(loaded.temporal_entries(), t1_count);

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn load_continues_id_generation() {
        let mut store = VoltStore::new();
        for _ in 0..(T0_CAPACITY + 5) {
            store.store(make_frame_with_content()).unwrap();
        }

        let dir = std::env::temp_dir().join("volt_db_test_ids_42");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_ids.json");

        store.save(&path).unwrap();
        let mut loaded = VoltStore::load(&path).unwrap();

        // T1 has frames 1..5 (oldest evicted). After load, next_id = 6.
        // New frame should get an ID higher than any T1 frame.
        let t1_max = loaded
            .t1()
            .list_strands()
            .iter()
            .flat_map(|&s| loaded.t1().get_by_strand(s))
            .map(|f| f.frame_meta.frame_id)
            .max()
            .unwrap_or(0);
        let new_id = loaded.store(make_frame_with_content()).unwrap();
        assert!(
            new_id > t1_max,
            "new ID {new_id} should be > max T1 ID {t1_max}"
        );

        // IDs should not collide with existing T1 frames
        let collision = loaded.t1().get_by_id(new_id).is_some();
        assert!(!collision, "new ID collided with existing T1 frame");

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn default_is_same_as_new() {
        let store = VoltStore::default();
        assert_eq!(store.active_strand(), 0);
        assert!(store.list_strands().contains(&0));
    }

    // --- Milestone 4.2 specific tests ---

    #[test]
    fn store_indexes_frames_with_r0() {
        let mut store = VoltStore::new();
        store.store(make_frame_with_content()).unwrap();
        assert_eq!(store.hnsw_entries(), 1);
        assert_eq!(store.temporal_entries(), 1);
    }

    #[test]
    fn store_skips_indexing_frames_without_r0() {
        let mut store = VoltStore::new();
        // Frame with no R₀ data
        store.store(TensorFrame::new()).unwrap();
        assert_eq!(store.hnsw_entries(), 0);
        assert_eq!(store.temporal_entries(), 0);
    }

    #[test]
    fn query_similar_finds_stored_frames() {
        let mut store = VoltStore::new();
        store.store(make_frame_with_content()).unwrap();

        let results = store.query_similar(&[0.5; SLOT_DIM], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].frame_id, 1);
    }

    #[test]
    fn query_similar_in_strand_respects_isolation() {
        let mut store = VoltStore::new();

        // Store in strand 0
        store.store(make_frame_with_content()).unwrap();

        // Store in strand 1
        store.switch_strand(1).unwrap();
        store.store(make_frame_with_content()).unwrap();

        let results = store.query_similar_in_strand(0, &[0.5; SLOT_DIM], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].strand_id, 0);
    }

    #[test]
    fn ghost_buffer_populates_on_store() {
        let mut store = VoltStore::new();

        // Store some frames so the HNSW has entries
        for _ in 0..5 {
            store.store(make_frame_with_content()).unwrap();
        }

        // The ghost buffer should have been refreshed
        // (may have entries from the HNSW query)
        // With identical gists, all should match
        assert!(store.ghost_buffer().len() > 0);
    }

    #[test]
    fn ghost_gists_returns_vectors() {
        let mut store = VoltStore::new();
        for _ in 0..3 {
            store.store(make_frame_with_content()).unwrap();
        }
        let gists = store.ghost_gists();
        // Each gist should be 256 dims
        for g in &gists {
            assert_eq!(g.len(), SLOT_DIM);
        }
    }

    // --- Milestone 4.3 specific tests ---

    #[test]
    fn new_store_is_memory_only() {
        let store = VoltStore::new();
        assert!(!store.is_disk_backed());
        assert_eq!(store.t2_len(), 0);
    }

    #[test]
    fn get_entry_by_id_returns_full() {
        let mut store = VoltStore::new();
        let id = store.store(make_frame_with_content()).unwrap();
        let entry = store.get_entry_by_id(id).unwrap();
        assert_eq!(entry.decay_level(), DecayLevel::Full);
        assert_eq!(entry.frame_id(), id);
    }

    #[test]
    fn pin_unpin_frame() {
        let mut store = VoltStore::new();
        let id = store.store(TensorFrame::new()).unwrap();

        assert!(!store.is_frame_pinned(id));
        store.pin_frame(id);
        assert!(store.is_frame_pinned(id));
        store.unpin_frame(id);
        assert!(!store.is_frame_pinned(id));
    }

    #[test]
    fn gc_on_empty_store() {
        let mut store = VoltStore::new();
        let result = store.run_gc().unwrap();
        assert_eq!(result.frames_preserved, 0);
        assert_eq!(result.frames_compressed, 0);
        assert_eq!(result.frames_gisted, 0);
        assert_eq!(result.frames_tombstoned, 0);
    }

    #[test]
    fn consolidate_empty_strand() {
        let mut store = VoltStore::new();
        let result = store.consolidate_strand(0).unwrap();
        assert_eq!(result.clusters_found, 0);
        assert!(result.wisdom_frames.is_empty());
    }

    #[test]
    fn concurrent_store_read_write() {
        let store = VoltStore::new();
        let concurrent = ConcurrentVoltStore::new(store);

        // Write
        {
            let mut guard = concurrent.write().unwrap();
            guard.store(make_frame_with_content()).unwrap();
        }

        // Read
        {
            let guard = concurrent.read().unwrap();
            assert!(guard.get_by_id(1).is_some());
            assert_eq!(guard.total_frame_count(), 1);
        }
    }

    #[test]
    fn concurrent_multi_reader() {
        let store = VoltStore::new();
        let concurrent = ConcurrentVoltStore::new(store);

        // Multiple readers at the same time
        let r1 = concurrent.read().unwrap();
        let r2 = concurrent.read().unwrap();
        assert_eq!(r1.active_strand(), r2.active_strand());
        drop(r1);
        drop(r2);
    }

    #[test]
    fn disk_backed_store_roundtrip() {
        // Run on a thread with 8MB stack — TensorFrame is 64KB and
        // serde_json + WAL serialization can overflow Windows' 1MB default.
        std::thread::Builder::new()
            .name("disk_backed_test".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let dir = std::env::temp_dir()
                    .join("volt_store_disk_test")
                    .join(format!("{}", std::process::id()));
                let _ = std::fs::remove_dir_all(&dir);

                let config = VoltStoreConfig {
                    data_dir: dir.clone(),
                    t1_overflow_threshold: 1024,
                    t2_config: T2Config {
                        data_dir: dir.join("t2"),
                        ..T2Config::default()
                    },
                    ..VoltStoreConfig::default()
                };

                {
                    let mut store = VoltStore::open(config.clone()).unwrap();
                    assert!(store.is_disk_backed());

                    for _ in 0..10 {
                        store.store(make_frame_with_content()).unwrap();
                    }

                    // Save T1 for persistence
                    let t1_path = dir.join("t1_strands.json");
                    store.save(&t1_path).unwrap();
                }

                // Reopen
                {
                    let store = VoltStore::open(config).unwrap();
                    // T0 is empty after reopen, but T1 frames should be loaded
                    // (depending on what was evicted to T1 before save)
                    assert!(store.is_disk_backed());
                }

                let _ = std::fs::remove_dir_all(&dir);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn total_entry_count_includes_t0_t1() {
        let mut store = VoltStore::new();
        for _ in 0..(T0_CAPACITY + 10) {
            store.store(make_frame_with_content()).unwrap();
        }
        assert_eq!(store.total_entry_count(), T0_CAPACITY + 10);
    }
}
