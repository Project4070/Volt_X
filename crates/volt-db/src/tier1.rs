//! T1 Strand Store — strand-organized frame storage in RAM.
//!
//! T1 holds frames organized by strand ID. Frames evicted from T0
//! are stored here. T1 persists across queries in RAM and can be
//! serialized to disk for persistence across restarts.
//!
//! Frames are stored as `Box<TensorFrame>` to avoid stack overflow
//! during serialization — each TensorFrame is ~64KB.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use volt_core::{TensorFrame, VoltError};

/// T1 Strand Store — frames organized by strand ID in RAM.
///
/// Each strand is a `Vec<Box<TensorFrame>>` ordered by insertion time.
/// Frames are boxed to avoid stack overflow during serialization
/// (each TensorFrame is ~64KB).
///
/// # Example
///
/// ```
/// use volt_db::tier1::StrandStore;
/// use volt_core::TensorFrame;
///
/// let mut store = StrandStore::new();
/// store.create_strand(1);
///
/// let mut frame = TensorFrame::new();
/// frame.frame_meta.strand_id = 1;
/// frame.frame_meta.frame_id = 42;
/// store.store(frame).unwrap();
///
/// assert_eq!(store.total_frame_count(), 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrandStore {
    strands: HashMap<u64, Vec<Box<TensorFrame>>>,
}

impl Default for StrandStore {
    fn default() -> Self {
        Self::new()
    }
}

impl StrandStore {
    /// Creates a new empty strand store.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier1::StrandStore;
    ///
    /// let store = StrandStore::new();
    /// assert_eq!(store.strand_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            strands: HashMap::new(),
        }
    }

    /// Creates a strand entry if it doesn't already exist.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier1::StrandStore;
    ///
    /// let mut store = StrandStore::new();
    /// store.create_strand(1);
    /// store.create_strand(1); // idempotent
    /// assert_eq!(store.strand_count(), 1);
    /// ```
    pub fn create_strand(&mut self, strand_id: u64) {
        self.strands.entry(strand_id).or_default();
    }

    /// Stores a frame in the appropriate strand.
    ///
    /// The strand is determined by `frame.frame_meta.strand_id`.
    /// If the strand doesn't exist, it is created automatically.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier1::StrandStore;
    /// use volt_core::TensorFrame;
    ///
    /// let mut store = StrandStore::new();
    /// let mut frame = TensorFrame::new();
    /// frame.frame_meta.strand_id = 1;
    /// frame.frame_meta.frame_id = 10;
    /// store.store(frame).unwrap();
    ///
    /// assert_eq!(store.strand_count(), 1);
    /// assert_eq!(store.total_frame_count(), 1);
    /// ```
    pub fn store(&mut self, frame: TensorFrame) -> Result<(), VoltError> {
        let strand_id = frame.frame_meta.strand_id;
        self.strands
            .entry(strand_id)
            .or_default()
            .push(Box::new(frame));
        Ok(())
    }

    /// Retrieves a frame by its `frame_id` via linear scan across all strands.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier1::StrandStore;
    /// use volt_core::TensorFrame;
    ///
    /// let mut store = StrandStore::new();
    /// let mut frame = TensorFrame::new();
    /// frame.frame_meta.strand_id = 1;
    /// frame.frame_meta.frame_id = 42;
    /// store.store(frame).unwrap();
    ///
    /// assert!(store.get_by_id(42).is_some());
    /// assert!(store.get_by_id(99).is_none());
    /// ```
    pub fn get_by_id(&self, frame_id: u64) -> Option<&TensorFrame> {
        for frames in self.strands.values() {
            if let Some(frame) = frames
                .iter()
                .find(|f| f.frame_meta.frame_id == frame_id)
            {
                return Some(frame);
            }
        }
        None
    }

    /// Returns all frames in the given strand, in insertion order.
    ///
    /// Returns an empty Vec if the strand doesn't exist.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier1::StrandStore;
    /// use volt_core::TensorFrame;
    ///
    /// let mut store = StrandStore::new();
    /// let mut f = TensorFrame::new();
    /// f.frame_meta.strand_id = 5;
    /// f.frame_meta.frame_id = 1;
    /// store.store(f).unwrap();
    ///
    /// assert_eq!(store.get_by_strand(5).len(), 1);
    /// assert_eq!(store.get_by_strand(99).len(), 0);
    /// ```
    pub fn get_by_strand(&self, strand_id: u64) -> Vec<&TensorFrame> {
        self.strands
            .get(&strand_id)
            .map(|v| v.iter().map(|b| b.as_ref()).collect())
            .unwrap_or_default()
    }

    /// Returns the most recent `n` frames from a specific strand, newest-first.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier1::StrandStore;
    /// use volt_core::TensorFrame;
    ///
    /// let mut store = StrandStore::new();
    /// for i in 0..10 {
    ///     let mut f = TensorFrame::new();
    ///     f.frame_meta.strand_id = 1;
    ///     f.frame_meta.frame_id = i;
    ///     store.store(f).unwrap();
    /// }
    ///
    /// let recent = store.recent_in_strand(1, 3);
    /// assert_eq!(recent.len(), 3);
    /// assert_eq!(recent[0].frame_meta.frame_id, 9);
    /// ```
    pub fn recent_in_strand(&self, strand_id: u64, n: usize) -> Vec<&TensorFrame> {
        self.strands
            .get(&strand_id)
            .map(|frames| frames.iter().rev().take(n).map(|b| b.as_ref()).collect())
            .unwrap_or_default()
    }

    /// Returns a sorted list of all strand IDs.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier1::StrandStore;
    ///
    /// let mut store = StrandStore::new();
    /// store.create_strand(3);
    /// store.create_strand(1);
    /// store.create_strand(2);
    ///
    /// assert_eq!(store.list_strands(), vec![1, 2, 3]);
    /// ```
    pub fn list_strands(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.strands.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Returns `true` if a strand with the given ID exists.
    pub fn has_strand(&self, strand_id: u64) -> bool {
        self.strands.contains_key(&strand_id)
    }

    /// Returns the number of strands.
    pub fn strand_count(&self) -> usize {
        self.strands.len()
    }

    /// Returns the total number of frames across all strands.
    pub fn total_frame_count(&self) -> usize {
        self.strands.values().map(|v| v.len()).sum()
    }

    /// Returns the number of frames in a specific strand.
    pub fn strand_frame_count(&self, strand_id: u64) -> usize {
        self.strands.get(&strand_id).map(|v| v.len()).unwrap_or(0)
    }

    /// Removes and returns a frame by its `frame_id`.
    ///
    /// Used by the GC pipeline to demote full frames from T1 to compressed
    /// storage in T2. Returns `None` if the frame is not found.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier1::StrandStore;
    /// use volt_core::TensorFrame;
    ///
    /// let mut store = StrandStore::new();
    /// let mut frame = TensorFrame::new();
    /// frame.frame_meta.strand_id = 1;
    /// frame.frame_meta.frame_id = 42;
    /// store.store(frame).unwrap();
    ///
    /// let removed = store.remove_frame(42);
    /// assert!(removed.is_some());
    /// assert_eq!(removed.unwrap().frame_meta.frame_id, 42);
    /// assert_eq!(store.total_frame_count(), 0);
    /// ```
    pub fn remove_frame(&mut self, frame_id: u64) -> Option<Box<TensorFrame>> {
        for frames in self.strands.values_mut() {
            if let Some(pos) = frames
                .iter()
                .position(|f| f.frame_meta.frame_id == frame_id)
            {
                return Some(frames.remove(pos));
            }
        }
        None
    }

    /// Returns the oldest frames across all strands, up to `n` frames.
    ///
    /// Frames are ordered by `created_at` timestamp. Used by VoltStore
    /// to decide which T1 frames to demote to T2.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier1::StrandStore;
    /// use volt_core::TensorFrame;
    ///
    /// let mut store = StrandStore::new();
    /// for i in 0..5u64 {
    ///     let mut f = TensorFrame::new();
    ///     f.frame_meta.strand_id = 0;
    ///     f.frame_meta.frame_id = i;
    ///     f.frame_meta.created_at = i * 1000;
    ///     store.store(f).unwrap();
    /// }
    ///
    /// let oldest = store.oldest_frame_ids(3);
    /// assert_eq!(oldest.len(), 3);
    /// ```
    pub fn oldest_frame_ids(&self, n: usize) -> Vec<u64> {
        let mut all: Vec<(u64, u64)> = self
            .strands
            .values()
            .flat_map(|frames| {
                frames
                    .iter()
                    .map(|f| (f.frame_meta.created_at, f.frame_meta.frame_id))
            })
            .collect();
        all.sort_unstable();
        all.into_iter().take(n).map(|(_, id)| id).collect()
    }

    /// Serializes the strand store to a JSON file on disk.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if serialization or file I/O fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_db::tier1::StrandStore;
    /// use std::path::Path;
    ///
    /// let store = StrandStore::new();
    /// store.save(Path::new("voltdb_t1.json")).unwrap();
    /// ```
    pub fn save(&self, path: &Path) -> Result<(), VoltError> {
        // Run on a dedicated thread with 8MB stack — TensorFrame is ~64KB
        // and serde's recursive descent can overflow Windows' 1MB default stack.
        let data = self.clone();
        let path = path.to_owned();
        std::thread::Builder::new()
            .name("voltdb-save".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let file =
                    std::fs::File::create(&path).map_err(|e| VoltError::StorageError {
                        message: format!("failed to create T1 file {}: {e}", path.display()),
                    })?;
                let writer = std::io::BufWriter::new(file);
                serde_json::to_writer(writer, &data).map_err(|e| VoltError::StorageError {
                    message: format!("failed to serialize T1 strand store: {e}"),
                })
            })
            .map_err(|e| VoltError::StorageError {
                message: format!("failed to spawn save thread: {e}"),
            })?
            .join()
            .map_err(|_| VoltError::StorageError {
                message: "save thread panicked".to_string(),
            })?
    }

    /// Loads a strand store from a JSON file on disk.
    ///
    /// Deserialization runs on a dedicated thread with 8MB stack —
    /// TensorFrame is ~64KB and serde's recursive descent can overflow
    /// Windows' default 1MB stack.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if file I/O or deserialization fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use volt_db::tier1::StrandStore;
    /// use std::path::Path;
    ///
    /// let store = StrandStore::load(Path::new("voltdb_t1.json")).unwrap();
    /// ```
    pub fn load(path: &Path) -> Result<Self, VoltError> {
        let path = path.to_owned();
        std::thread::Builder::new()
            .name("voltdb-load".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let file =
                    std::fs::File::open(&path).map_err(|e| VoltError::StorageError {
                        message: format!("failed to open T1 file {}: {e}", path.display()),
                    })?;
                let reader = std::io::BufReader::new(file);
                serde_json::from_reader(reader).map_err(|e| VoltError::StorageError {
                    message: format!("failed to deserialize T1 strand store: {e}"),
                })
            })
            .map_err(|e| VoltError::StorageError {
                message: format!("failed to spawn load thread: {e}"),
            })?
            .join()
            .map_err(|_| VoltError::StorageError {
                message: "load thread panicked".to_string(),
            })?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::{SlotData, SlotRole, SLOT_DIM};

    fn make_frame(id: u64, strand: u64) -> TensorFrame {
        let mut frame = TensorFrame::new();
        frame.frame_meta.frame_id = id;
        frame.frame_meta.strand_id = strand;
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [id as f32 * 0.01; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();
        frame
    }

    #[test]
    fn new_store_is_empty() {
        let store = StrandStore::new();
        assert_eq!(store.strand_count(), 0);
        assert_eq!(store.total_frame_count(), 0);
    }

    #[test]
    fn create_strand_is_idempotent() {
        let mut store = StrandStore::new();
        store.create_strand(1);
        store.create_strand(1);
        assert_eq!(store.strand_count(), 1);
    }

    #[test]
    fn store_creates_strand_automatically() {
        let mut store = StrandStore::new();
        store.store(make_frame(1, 42)).unwrap();

        assert!(store.has_strand(42));
        assert_eq!(store.strand_count(), 1);
    }

    #[test]
    fn store_and_retrieve_by_id() {
        let mut store = StrandStore::new();
        store.store(make_frame(10, 1)).unwrap();
        store.store(make_frame(20, 2)).unwrap();

        assert!(store.get_by_id(10).is_some());
        assert!(store.get_by_id(20).is_some());
        assert!(store.get_by_id(99).is_none());
    }

    #[test]
    fn get_by_strand_returns_correct_frames() {
        let mut store = StrandStore::new();
        store.store(make_frame(1, 10)).unwrap();
        store.store(make_frame(2, 10)).unwrap();
        store.store(make_frame(3, 20)).unwrap();

        assert_eq!(store.get_by_strand(10).len(), 2);
        assert_eq!(store.get_by_strand(20).len(), 1);
        assert_eq!(store.get_by_strand(99).len(), 0);
    }

    #[test]
    fn recent_in_strand_returns_newest_first() {
        let mut store = StrandStore::new();
        for i in 0..10 {
            store.store(make_frame(i, 1)).unwrap();
        }

        let recent = store.recent_in_strand(1, 3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].frame_meta.frame_id, 9);
        assert_eq!(recent[1].frame_meta.frame_id, 8);
        assert_eq!(recent[2].frame_meta.frame_id, 7);
    }

    #[test]
    fn list_strands_sorted() {
        let mut store = StrandStore::new();
        store.create_strand(5);
        store.create_strand(1);
        store.create_strand(3);

        assert_eq!(store.list_strands(), vec![1, 3, 5]);
    }

    #[test]
    fn strand_frame_count() {
        let mut store = StrandStore::new();
        store.store(make_frame(1, 10)).unwrap();
        store.store(make_frame(2, 10)).unwrap();
        store.store(make_frame(3, 20)).unwrap();

        assert_eq!(store.strand_frame_count(10), 2);
        assert_eq!(store.strand_frame_count(20), 1);
        assert_eq!(store.strand_frame_count(99), 0);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let mut store = StrandStore::new();
        store.store(make_frame(1, 10)).unwrap();
        store.store(make_frame(2, 10)).unwrap();
        store.store(make_frame(3, 20)).unwrap();

        let dir = std::env::temp_dir().join("volt_db_test_t1");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_roundtrip.json");

        store.save(&path).unwrap();
        let loaded = StrandStore::load(&path).unwrap();

        assert_eq!(loaded.strand_count(), 2);
        assert_eq!(loaded.total_frame_count(), 3);
        assert!(loaded.get_by_id(1).is_some());
        assert!(loaded.get_by_id(2).is_some());
        assert!(loaded.get_by_id(3).is_some());
        assert_eq!(loaded.get_by_strand(10).len(), 2);
        assert_eq!(loaded.get_by_strand(20).len(), 1);

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn default_is_same_as_new() {
        let store = StrandStore::default();
        assert_eq!(store.strand_count(), 0);
        assert_eq!(store.total_frame_count(), 0);
    }
}
