//! VoltStore — unified memory facade combining T0 and T1.
//!
//! VoltStore is the primary API for storing and retrieving TensorFrames.
//! It manages the T0 working memory (ring buffer) and T1 strand storage,
//! handles automatic eviction from T0 to T1, strand management, frame ID
//! generation, and persistence.

use std::path::Path;

use volt_core::{TensorFrame, VoltError};

use crate::tier0::WorkingMemory;
use crate::tier1::StrandStore;

/// Unified memory facade combining T0 working memory and T1 strand storage.
///
/// # Frame Lifecycle
///
/// 1. Caller stores a frame via [`VoltStore::store`]
/// 2. Frame gets a unique `frame_id` and the active `strand_id`
/// 3. Frame is placed in T0 (working memory)
/// 4. If T0 is full, the oldest frame is evicted to T1 (strand storage)
/// 5. Retrieval searches T0 first, then T1
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
#[derive(Debug, Clone)]
pub struct VoltStore {
    t0: WorkingMemory,
    t1: StrandStore,
    active_strand: u64,
    next_id: u64,
}

impl Default for VoltStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VoltStore {
    /// Creates a new VoltStore with empty T0 and T1.
    ///
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
            active_strand: 0,
            next_id: 1,
        }
    }

    /// Stores a frame, assigning it a unique frame ID and the active strand ID.
    ///
    /// The frame is placed in T0. If T0 is full, the oldest frame is
    /// evicted to T1 automatically.
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

        if let Some(evicted) = self.t0.store(frame) {
            self.t1.store(evicted)?;
        }

        Ok(frame_id)
    }

    /// Retrieves a frame by its `frame_id`, searching T0 first then T1.
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

    /// Returns all frames belonging to a strand, from both T0 and T1.
    ///
    /// Results are ordered: T1 frames first (oldest), then T0 frames.
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

    /// Returns the total number of frames across T0 and T1.
    pub fn total_frame_count(&self) -> usize {
        self.t0.len() + self.t1.total_frame_count()
    }

    /// Returns a reference to the T0 working memory.
    pub fn t0(&self) -> &WorkingMemory {
        &self.t0
    }

    /// Returns a reference to the T1 strand store.
    pub fn t1(&self) -> &StrandStore {
        &self.t1
    }

    /// Saves T1 strand storage to disk for persistence across restarts.
    ///
    /// T0 is intentionally not saved — it's ephemeral working memory.
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
        Ok(Self {
            t0: WorkingMemory::new(),
            t1,
            active_strand: 0,
            next_id: max_id + 1,
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

        let dir = std::env::temp_dir().join("volt_db_test_store");
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

        let dir = std::env::temp_dir().join("volt_db_test_ids");
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
}
