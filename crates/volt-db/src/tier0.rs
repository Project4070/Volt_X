//! T0 Working Memory — fixed-size ring buffer of TensorFrames.
//!
//! T0 holds the most recent frames in RAM for instant access.
//! When capacity is reached, the oldest frame is evicted (FIFO order).
//! Evicted frames should be moved to T1 by the caller.
//!
//! # Capacity
//!
//! The default capacity is [`T0_CAPACITY`] (64 frames).

use std::collections::VecDeque;

use volt_core::TensorFrame;

/// Maximum number of frames in T0 working memory.
pub const T0_CAPACITY: usize = 64;

/// T0 Working Memory — a fixed-capacity ring buffer of TensorFrames.
///
/// Frames are stored in insertion order. When the buffer is full,
/// the oldest frame is evicted and returned to the caller so it
/// can be promoted to T1.
///
/// # Example
///
/// ```
/// use volt_db::tier0::WorkingMemory;
/// use volt_core::TensorFrame;
///
/// let mut wm = WorkingMemory::new();
/// assert!(wm.is_empty());
///
/// let frame = TensorFrame::new();
/// let evicted = wm.store(frame);
/// assert!(evicted.is_none()); // not full yet
/// assert_eq!(wm.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct WorkingMemory {
    buffer: VecDeque<TensorFrame>,
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkingMemory {
    /// Creates a new empty working memory with default capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier0::WorkingMemory;
    ///
    /// let wm = WorkingMemory::new();
    /// assert_eq!(wm.len(), 0);
    /// assert_eq!(wm.capacity(), 64);
    /// ```
    pub fn new() -> Self {
        Self {
            buffer: VecDeque::with_capacity(T0_CAPACITY),
        }
    }

    /// Stores a frame in working memory.
    ///
    /// If the buffer is full, the oldest frame is evicted and returned.
    /// The caller is responsible for moving evicted frames to T1.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier0::WorkingMemory;
    /// use volt_core::TensorFrame;
    ///
    /// let mut wm = WorkingMemory::new();
    /// let evicted = wm.store(TensorFrame::new());
    /// assert!(evicted.is_none());
    /// ```
    pub fn store(&mut self, frame: TensorFrame) -> Option<TensorFrame> {
        let evicted = if self.buffer.len() >= T0_CAPACITY {
            self.buffer.pop_front()
        } else {
            None
        };
        self.buffer.push_back(frame);
        evicted
    }

    /// Retrieves a frame by its `frame_id` via linear scan.
    ///
    /// Returns `None` if no frame with that ID exists in T0.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier0::WorkingMemory;
    /// use volt_core::TensorFrame;
    ///
    /// let mut wm = WorkingMemory::new();
    /// let mut frame = TensorFrame::new();
    /// frame.frame_meta.frame_id = 42;
    /// wm.store(frame);
    ///
    /// assert!(wm.get_by_id(42).is_some());
    /// assert!(wm.get_by_id(99).is_none());
    /// ```
    pub fn get_by_id(&self, frame_id: u64) -> Option<&TensorFrame> {
        self.buffer
            .iter()
            .find(|f| f.frame_meta.frame_id == frame_id)
    }

    /// Returns the most recent `n` frames, ordered newest-first.
    ///
    /// If `n` exceeds the number of stored frames, returns all frames.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier0::WorkingMemory;
    /// use volt_core::TensorFrame;
    ///
    /// let mut wm = WorkingMemory::new();
    /// for i in 0..5 {
    ///     let mut f = TensorFrame::new();
    ///     f.frame_meta.frame_id = i;
    ///     wm.store(f);
    /// }
    ///
    /// let recent = wm.recent(3);
    /// assert_eq!(recent.len(), 3);
    /// // Newest first
    /// assert_eq!(recent[0].frame_meta.frame_id, 4);
    /// assert_eq!(recent[1].frame_meta.frame_id, 3);
    /// assert_eq!(recent[2].frame_meta.frame_id, 2);
    /// ```
    pub fn recent(&self, n: usize) -> Vec<&TensorFrame> {
        self.buffer.iter().rev().take(n).collect()
    }

    /// Returns all frames in T0 that belong to the given strand.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier0::WorkingMemory;
    /// use volt_core::TensorFrame;
    ///
    /// let mut wm = WorkingMemory::new();
    /// let mut f1 = TensorFrame::new();
    /// f1.frame_meta.strand_id = 1;
    /// wm.store(f1);
    ///
    /// let mut f2 = TensorFrame::new();
    /// f2.frame_meta.strand_id = 2;
    /// wm.store(f2);
    ///
    /// assert_eq!(wm.get_by_strand(1).len(), 1);
    /// assert_eq!(wm.get_by_strand(2).len(), 1);
    /// assert_eq!(wm.get_by_strand(99).len(), 0);
    /// ```
    pub fn get_by_strand(&self, strand_id: u64) -> Vec<&TensorFrame> {
        self.buffer
            .iter()
            .filter(|f| f.frame_meta.strand_id == strand_id)
            .collect()
    }

    /// Returns the number of frames currently stored.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if no frames are stored.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns `true` if the buffer is at capacity.
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= T0_CAPACITY
    }

    /// Returns the maximum capacity of this working memory.
    pub fn capacity(&self) -> usize {
        T0_CAPACITY
    }

    /// Returns an iterator over all frames in insertion order (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &TensorFrame> {
        self.buffer.iter()
    }

    /// Clears all frames from working memory.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::tier0::WorkingMemory;
    /// use volt_core::TensorFrame;
    ///
    /// let mut wm = WorkingMemory::new();
    /// wm.store(TensorFrame::new());
    /// assert!(!wm.is_empty());
    ///
    /// wm.clear();
    /// assert!(wm.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.buffer.clear();
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
        // Give it some content so it's not empty
        let mut slot = SlotData::new(SlotRole::Agent);
        slot.write_resolution(0, [id as f32 * 0.01; SLOT_DIM]);
        frame.write_slot(0, slot).unwrap();
        frame
    }

    #[test]
    fn new_working_memory_is_empty() {
        let wm = WorkingMemory::new();
        assert!(wm.is_empty());
        assert_eq!(wm.len(), 0);
        assert!(!wm.is_full());
        assert_eq!(wm.capacity(), T0_CAPACITY);
    }

    #[test]
    fn store_and_retrieve_by_id() {
        let mut wm = WorkingMemory::new();
        let frame = make_frame(42, 1);
        let evicted = wm.store(frame);

        assert!(evicted.is_none());
        assert_eq!(wm.len(), 1);

        let found = wm.get_by_id(42);
        assert!(found.is_some());
        assert_eq!(found.unwrap().frame_meta.frame_id, 42);
    }

    #[test]
    fn get_by_id_returns_none_for_missing() {
        let wm = WorkingMemory::new();
        assert!(wm.get_by_id(999).is_none());
    }

    #[test]
    fn eviction_at_capacity() {
        let mut wm = WorkingMemory::new();

        // Fill to capacity
        for i in 0..T0_CAPACITY as u64 {
            let evicted = wm.store(make_frame(i, 1));
            assert!(evicted.is_none());
        }
        assert!(wm.is_full());
        assert_eq!(wm.len(), T0_CAPACITY);

        // One more should evict the oldest (id=0)
        let evicted = wm.store(make_frame(T0_CAPACITY as u64, 1));
        assert!(evicted.is_some());
        assert_eq!(evicted.unwrap().frame_meta.frame_id, 0);
        assert_eq!(wm.len(), T0_CAPACITY);

        // Frame 0 no longer in T0, but frame 1 and the new one are
        assert!(wm.get_by_id(0).is_none());
        assert!(wm.get_by_id(1).is_some());
        assert!(wm.get_by_id(T0_CAPACITY as u64).is_some());
    }

    #[test]
    fn recent_returns_newest_first() {
        let mut wm = WorkingMemory::new();
        for i in 0..10 {
            wm.store(make_frame(i, 1));
        }

        let recent = wm.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].frame_meta.frame_id, 9);
        assert_eq!(recent[1].frame_meta.frame_id, 8);
        assert_eq!(recent[2].frame_meta.frame_id, 7);
    }

    #[test]
    fn recent_clamps_to_available() {
        let mut wm = WorkingMemory::new();
        for i in 0..3 {
            wm.store(make_frame(i, 1));
        }

        let recent = wm.recent(100);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn get_by_strand_filters_correctly() {
        let mut wm = WorkingMemory::new();
        wm.store(make_frame(1, 10));
        wm.store(make_frame(2, 20));
        wm.store(make_frame(3, 10));

        let strand_10 = wm.get_by_strand(10);
        assert_eq!(strand_10.len(), 2);

        let strand_20 = wm.get_by_strand(20);
        assert_eq!(strand_20.len(), 1);

        let strand_99 = wm.get_by_strand(99);
        assert!(strand_99.is_empty());
    }

    #[test]
    fn clear_removes_all_frames() {
        let mut wm = WorkingMemory::new();
        for i in 0..10 {
            wm.store(make_frame(i, 1));
        }
        assert_eq!(wm.len(), 10);

        wm.clear();
        assert!(wm.is_empty());
        assert_eq!(wm.len(), 0);
    }

    #[test]
    fn iter_returns_insertion_order() {
        let mut wm = WorkingMemory::new();
        for i in 0..5 {
            wm.store(make_frame(i, 1));
        }

        let ids: Vec<u64> = wm.iter().map(|f| f.frame_meta.frame_id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn default_is_same_as_new() {
        let wm = WorkingMemory::default();
        assert!(wm.is_empty());
        assert_eq!(wm.capacity(), T0_CAPACITY);
    }
}
