//! B-tree temporal index for time-range queries.
//!
//! Maps `created_at` timestamps (microseconds) to frame IDs, enabling
//! efficient range queries like "all frames from last week".

use std::collections::BTreeMap;

/// Temporal index mapping `created_at` timestamps to frame IDs.
///
/// Uses a [`BTreeMap`] for O(log N) insertion and efficient range queries.
/// Multiple frames at the same timestamp are stored in a `Vec`.
///
/// # Example
///
/// ```
/// use volt_db::temporal::TemporalIndex;
///
/// let mut idx = TemporalIndex::new();
/// idx.insert(1000, 1);
/// idx.insert(2000, 2);
/// idx.insert(3000, 3);
///
/// let range = idx.query_range(1000, 2000);
/// assert_eq!(range.len(), 2);
/// assert!(range.contains(&1));
/// assert!(range.contains(&2));
/// ```
#[derive(Debug, Clone, Default)]
pub struct TemporalIndex {
    /// Maps created_at (microseconds) → list of frame_ids at that timestamp.
    index: BTreeMap<u64, Vec<u64>>,
    /// Total entry count.
    count: usize,
}

impl TemporalIndex {
    /// Creates a new empty temporal index.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::temporal::TemporalIndex;
    ///
    /// let idx = TemporalIndex::new();
    /// assert!(idx.is_empty());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a frame entry into the temporal index.
    ///
    /// # Arguments
    ///
    /// * `created_at` — timestamp in microseconds.
    /// * `frame_id` — the unique frame ID.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::temporal::TemporalIndex;
    ///
    /// let mut idx = TemporalIndex::new();
    /// idx.insert(1000, 42);
    /// assert_eq!(idx.len(), 1);
    /// ```
    pub fn insert(&mut self, created_at: u64, frame_id: u64) {
        self.index.entry(created_at).or_default().push(frame_id);
        self.count += 1;
    }

    /// Returns all frame IDs in the time range `[start, end]` inclusive.
    ///
    /// Results are ordered by ascending timestamp, then by insertion order
    /// within the same timestamp.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::temporal::TemporalIndex;
    ///
    /// let mut idx = TemporalIndex::new();
    /// idx.insert(1000, 1);
    /// idx.insert(2000, 2);
    /// idx.insert(3000, 3);
    /// idx.insert(4000, 4);
    ///
    /// let range = idx.query_range(2000, 3000);
    /// assert_eq!(range, vec![2, 3]);
    /// ```
    pub fn query_range(&self, start: u64, end: u64) -> Vec<u64> {
        self.index
            .range(start..=end)
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect()
    }

    /// Returns the N most recent frame IDs, newest first.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::temporal::TemporalIndex;
    ///
    /// let mut idx = TemporalIndex::new();
    /// idx.insert(1000, 1);
    /// idx.insert(2000, 2);
    /// idx.insert(3000, 3);
    ///
    /// let recent = idx.most_recent(2);
    /// assert_eq!(recent, vec![3, 2]);
    /// ```
    pub fn most_recent(&self, n: usize) -> Vec<u64> {
        self.index
            .iter()
            .rev()
            .flat_map(|(_, ids)| ids.iter().rev().copied())
            .take(n)
            .collect()
    }

    /// Returns the total number of indexed entries.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Removes a frame from the temporal index by its `frame_id`.
    ///
    /// Scans all timestamps to find and remove the entry. Used by GC
    /// when tombstoning a frame.
    ///
    /// Returns `true` if the frame was found and removed.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::temporal::TemporalIndex;
    ///
    /// let mut idx = TemporalIndex::new();
    /// idx.insert(1000, 42);
    /// assert_eq!(idx.len(), 1);
    ///
    /// assert!(idx.remove(42));
    /// assert_eq!(idx.len(), 0);
    /// assert!(!idx.remove(42)); // already removed
    /// ```
    pub fn remove(&mut self, frame_id: u64) -> bool {
        for ids in self.index.values_mut() {
            if let Some(pos) = ids.iter().position(|&id| id == frame_id) {
                ids.remove(pos);
                self.count -= 1;
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_index() {
        let idx = TemporalIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert!(idx.query_range(0, u64::MAX).is_empty());
        assert!(idx.most_recent(10).is_empty());
    }

    #[test]
    fn insert_and_query_range() {
        let mut idx = TemporalIndex::new();
        idx.insert(1000, 1);
        idx.insert(2000, 2);
        idx.insert(3000, 3);
        idx.insert(4000, 4);
        idx.insert(5000, 5);

        let range = idx.query_range(2000, 4000);
        assert_eq!(range, vec![2, 3, 4]);
    }

    #[test]
    fn query_range_exact_bounds() {
        let mut idx = TemporalIndex::new();
        idx.insert(1000, 1);
        idx.insert(2000, 2);

        let range = idx.query_range(1000, 1000);
        assert_eq!(range, vec![1]);
    }

    #[test]
    fn query_range_no_match() {
        let mut idx = TemporalIndex::new();
        idx.insert(1000, 1);
        idx.insert(5000, 2);

        let range = idx.query_range(2000, 4000);
        assert!(range.is_empty());
    }

    #[test]
    fn most_recent_ordering() {
        let mut idx = TemporalIndex::new();
        idx.insert(1000, 1);
        idx.insert(2000, 2);
        idx.insert(3000, 3);
        idx.insert(4000, 4);

        let recent = idx.most_recent(3);
        assert_eq!(recent, vec![4, 3, 2]);
    }

    #[test]
    fn most_recent_more_than_available() {
        let mut idx = TemporalIndex::new();
        idx.insert(1000, 1);
        idx.insert(2000, 2);

        let recent = idx.most_recent(10);
        assert_eq!(recent, vec![2, 1]);
    }

    #[test]
    fn same_timestamp_multiple_frames() {
        let mut idx = TemporalIndex::new();
        idx.insert(1000, 1);
        idx.insert(1000, 2);
        idx.insert(1000, 3);

        assert_eq!(idx.len(), 3);
        let range = idx.query_range(1000, 1000);
        assert_eq!(range.len(), 3);
        assert!(range.contains(&1));
        assert!(range.contains(&2));
        assert!(range.contains(&3));
    }

    #[test]
    fn all_frames_in_range() {
        let mut idx = TemporalIndex::new();
        for i in 0..100 {
            idx.insert(i * 1000, i);
        }
        let all = idx.query_range(0, u64::MAX);
        assert_eq!(all.len(), 100);
    }
}
