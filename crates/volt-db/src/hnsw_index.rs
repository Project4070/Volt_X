//! Per-strand HNSW index for semantic similarity search.
//!
//! Wraps [`hnsw_rs`] to provide O(log N) approximate nearest-neighbour
//! queries over frame R₀ gists. Each strand has its own HNSW index,
//! providing natural strand isolation.
//!
//! The HNSW index is rebuilt from stored gists on load (not serialized),
//! matching the pattern from [`volt_bus::codebook::Codebook`].

use std::collections::{HashMap, HashSet};

use hnsw_rs::prelude::*;
use volt_core::{VoltError, SLOT_DIM};

use crate::gist::FrameGist;

// HNSW tuning constants — same as volt-bus codebook for consistency.
const HNSW_M: usize = 24;
const HNSW_MAX_LAYER: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 200;
const HNSW_EF_SEARCH: usize = 32;

/// A single search result from the HNSW index.
///
/// # Example
///
/// ```
/// use volt_db::hnsw_index::SimilarityResult;
/// use volt_core::SLOT_DIM;
///
/// let result = SimilarityResult {
///     frame_id: 42,
///     strand_id: 1,
///     distance: 0.15,
///     gist: [0.0; SLOT_DIM],
/// };
/// assert_eq!(result.frame_id, 42);
/// ```
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// The source frame's unique ID.
    pub frame_id: u64,
    /// The strand this frame belongs to.
    pub strand_id: u64,
    /// Cosine distance from the query (0.0 = identical, 2.0 = opposite).
    pub distance: f32,
    /// The R₀ gist vector of the matched frame.
    pub gist: [f32; SLOT_DIM],
}

/// HNSW index for a single strand.
///
/// Stores gist vectors and maintains an HNSW graph for fast ANN queries.
/// The internal HNSW uses `DistCosine` for cosine distance.
pub struct StrandHnsw {
    /// HNSW index over this strand's gists.
    index: Hnsw<'static, f32, DistCosine>,
    /// Maps internal HNSW ID (usize) → frame_id (u64).
    id_map: Vec<u64>,
    /// Stored gist vectors (parallel to id_map).
    gists: Vec<[f32; SLOT_DIM]>,
    /// The strand this index covers.
    strand_id: u64,
}

impl std::fmt::Debug for StrandHnsw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StrandHnsw(strand={}, entries={})",
            self.strand_id,
            self.gists.len()
        )
    }
}

impl StrandHnsw {
    /// Creates a new empty HNSW index for a strand.
    ///
    /// # Arguments
    ///
    /// * `strand_id` — the strand this index covers.
    /// * `initial_capacity` — estimated number of entries (used for pre-allocation).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::hnsw_index::StrandHnsw;
    ///
    /// let idx = StrandHnsw::new(0, 100);
    /// assert!(idx.is_empty());
    /// assert_eq!(idx.len(), 0);
    /// ```
    pub fn new(strand_id: u64, initial_capacity: usize) -> Self {
        let capacity = initial_capacity.max(16);
        let index = Hnsw::<f32, DistCosine>::new(
            HNSW_M,
            capacity,
            HNSW_MAX_LAYER,
            HNSW_EF_CONSTRUCTION,
            DistCosine,
        );
        Self {
            index,
            id_map: Vec::with_capacity(capacity),
            gists: Vec::with_capacity(capacity),
            strand_id,
        }
    }

    /// Inserts a frame gist into this strand's HNSW index.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if the gist vector contains NaN or Inf.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::hnsw_index::StrandHnsw;
    /// use volt_db::gist::FrameGist;
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut idx = StrandHnsw::new(0, 100);
    /// let gist = FrameGist {
    ///     vector: [0.1; SLOT_DIM],
    ///     frame_id: 1,
    ///     strand_id: 0,
    ///     created_at: 0,
    /// };
    /// idx.insert(&gist).unwrap();
    /// assert_eq!(idx.len(), 1);
    /// ```
    pub fn insert(&mut self, gist: &FrameGist) -> Result<(), VoltError> {
        if gist.vector.iter().any(|x| !x.is_finite()) {
            return Err(VoltError::StorageError {
                message: format!(
                    "HNSW insert: gist for frame {} contains NaN or Inf",
                    gist.frame_id
                ),
            });
        }

        let internal_id = self.gists.len();
        self.gists.push(gist.vector);
        self.id_map.push(gist.frame_id);
        self.index.insert((&self.gists[internal_id][..], internal_id));

        Ok(())
    }

    /// Queries the top-k most similar gists to the given query vector.
    ///
    /// Returns results sorted by ascending distance (closest first).
    /// Returns an empty vec if the index is empty or k is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::hnsw_index::StrandHnsw;
    /// use volt_db::gist::FrameGist;
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut idx = StrandHnsw::new(0, 100);
    /// let gist = FrameGist {
    ///     vector: [0.1; SLOT_DIM],
    ///     frame_id: 1,
    ///     strand_id: 0,
    ///     created_at: 0,
    /// };
    /// idx.insert(&gist).unwrap();
    ///
    /// let results = idx.query(&[0.1; SLOT_DIM], 5);
    /// assert_eq!(results.len(), 1);
    /// assert_eq!(results[0].frame_id, 1);
    /// ```
    pub fn query(&self, query: &[f32; SLOT_DIM], k: usize) -> Vec<SimilarityResult> {
        if self.gists.is_empty() || k == 0 {
            return Vec::new();
        }

        let neighbours = self.index.search(query.as_slice(), k, HNSW_EF_SEARCH);

        neighbours
            .into_iter()
            .map(|n| SimilarityResult {
                frame_id: self.id_map[n.d_id],
                strand_id: self.strand_id,
                distance: n.distance,
                gist: self.gists[n.d_id],
            })
            .collect()
    }

    /// Returns the number of entries in this strand's index.
    pub fn len(&self) -> usize {
        self.gists.len()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.gists.is_empty()
    }

    /// Returns the strand ID this index covers.
    pub fn strand_id(&self) -> u64 {
        self.strand_id
    }
}

/// Collection of per-strand HNSW indices.
///
/// Manages one [`StrandHnsw`] per strand and provides both per-strand
/// and cross-strand similarity queries.
///
/// # Example
///
/// ```
/// use volt_db::hnsw_index::HnswIndex;
/// use volt_db::gist::FrameGist;
/// use volt_core::SLOT_DIM;
///
/// let mut index = HnswIndex::new();
/// let gist = FrameGist {
///     vector: [0.1; SLOT_DIM],
///     frame_id: 1,
///     strand_id: 0,
///     created_at: 0,
/// };
/// index.insert(&gist).unwrap();
/// assert_eq!(index.total_entries(), 1);
/// ```
pub struct HnswIndex {
    strands: HashMap<u64, StrandHnsw>,
    /// Frame IDs that have been soft-deleted (tombstoned by GC).
    /// Query results filter these out. Cleared on index rebuild (load).
    deleted: HashSet<u64>,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HnswIndex(strands={}, total_entries={})",
            self.strands.len(),
            self.total_entries()
        )
    }
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl HnswIndex {
    /// Creates a new empty collection of HNSW indices.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::hnsw_index::HnswIndex;
    ///
    /// let index = HnswIndex::new();
    /// assert_eq!(index.total_entries(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            strands: HashMap::new(),
            deleted: HashSet::new(),
        }
    }

    /// Inserts a gist into the appropriate strand's HNSW index.
    ///
    /// Creates the strand index automatically if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError::StorageError`] if the gist vector is invalid.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::hnsw_index::HnswIndex;
    /// use volt_db::gist::FrameGist;
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut index = HnswIndex::new();
    /// let gist = FrameGist {
    ///     vector: [0.1; SLOT_DIM],
    ///     frame_id: 1,
    ///     strand_id: 5,
    ///     created_at: 0,
    /// };
    /// index.insert(&gist).unwrap();
    /// assert_eq!(index.total_entries(), 1);
    /// ```
    pub fn insert(&mut self, gist: &FrameGist) -> Result<(), VoltError> {
        let strand_index = self
            .strands
            .entry(gist.strand_id)
            .or_insert_with(|| StrandHnsw::new(gist.strand_id, 64));
        strand_index.insert(gist)
    }

    /// Queries a single strand for the top-k most similar gists.
    ///
    /// Returns an empty vec if the strand doesn't exist or has no entries.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::hnsw_index::HnswIndex;
    /// use volt_db::gist::FrameGist;
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut index = HnswIndex::new();
    /// let gist = FrameGist {
    ///     vector: [0.1; SLOT_DIM],
    ///     frame_id: 1,
    ///     strand_id: 0,
    ///     created_at: 0,
    /// };
    /// index.insert(&gist).unwrap();
    ///
    /// let results = index.query_strand(0, &[0.1; SLOT_DIM], 5);
    /// assert_eq!(results.len(), 1);
    ///
    /// let empty = index.query_strand(999, &[0.1; SLOT_DIM], 5);
    /// assert!(empty.is_empty());
    /// ```
    pub fn query_strand(
        &self,
        strand_id: u64,
        query: &[f32; SLOT_DIM],
        k: usize,
    ) -> Vec<SimilarityResult> {
        match self.strands.get(&strand_id) {
            Some(strand_index) => {
                if self.deleted.is_empty() {
                    strand_index.query(query, k)
                } else {
                    // Over-fetch to compensate for filtered-out deleted entries
                    let fetch_k = k + self.deleted.len();
                    strand_index
                        .query(query, fetch_k)
                        .into_iter()
                        .filter(|r| !self.deleted.contains(&r.frame_id))
                        .take(k)
                        .collect()
                }
            }
            None => Vec::new(),
        }
    }

    /// Queries ALL strands and returns the combined top-k results.
    ///
    /// Collects results from every strand, sorts by ascending distance,
    /// and returns the top-k globally.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::hnsw_index::HnswIndex;
    /// use volt_db::gist::FrameGist;
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut index = HnswIndex::new();
    /// // Insert into strand 0
    /// index.insert(&FrameGist {
    ///     vector: [0.1; SLOT_DIM],
    ///     frame_id: 1,
    ///     strand_id: 0,
    ///     created_at: 0,
    /// }).unwrap();
    /// // Insert into strand 1
    /// index.insert(&FrameGist {
    ///     vector: [0.2; SLOT_DIM],
    ///     frame_id: 2,
    ///     strand_id: 1,
    ///     created_at: 0,
    /// }).unwrap();
    ///
    /// let results = index.query_all(&[0.1; SLOT_DIM], 10);
    /// assert_eq!(results.len(), 2);
    /// ```
    pub fn query_all(&self, query: &[f32; SLOT_DIM], k: usize) -> Vec<SimilarityResult> {
        if k == 0 {
            return Vec::new();
        }

        let fetch_k = if self.deleted.is_empty() {
            k
        } else {
            k + self.deleted.len()
        };

        let mut all_results: Vec<SimilarityResult> = self
            .strands
            .values()
            .flat_map(|strand_index| strand_index.query(query, fetch_k))
            .filter(|r| !self.deleted.contains(&r.frame_id))
            .collect();

        // Sort by ascending distance (closest first)
        all_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);
        all_results
    }

    /// Returns a list of strand IDs that have HNSW indices.
    pub fn indexed_strands(&self) -> Vec<u64> {
        self.strands.keys().copied().collect()
    }

    /// Returns the total number of indexed gists across all strands
    /// (excluding soft-deleted entries).
    pub fn total_entries(&self) -> usize {
        let raw: usize = self.strands.values().map(|s| s.len()).sum();
        raw.saturating_sub(self.deleted.len())
    }

    /// Marks a frame as soft-deleted in the HNSW index.
    ///
    /// Deleted frames are filtered out of query results. The actual HNSW
    /// graph entries remain but are invisible to callers. On index rebuild
    /// (during load), deleted entries are simply not re-inserted.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::hnsw_index::HnswIndex;
    /// use volt_db::gist::FrameGist;
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut index = HnswIndex::new();
    /// let gist = FrameGist {
    ///     vector: [0.1; SLOT_DIM],
    ///     frame_id: 1,
    ///     strand_id: 0,
    ///     created_at: 0,
    /// };
    /// index.insert(&gist).unwrap();
    /// assert_eq!(index.total_entries(), 1);
    ///
    /// index.mark_deleted(1);
    /// assert_eq!(index.total_entries(), 0);
    ///
    /// let results = index.query_strand(0, &[0.1; SLOT_DIM], 5);
    /// assert!(results.is_empty());
    /// ```
    pub fn mark_deleted(&mut self, frame_id: u64) {
        self.deleted.insert(frame_id);
    }

    /// Returns true if a frame ID is soft-deleted.
    pub fn is_deleted(&self, frame_id: u64) -> bool {
        self.deleted.contains(&frame_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gist::FrameGist;

    fn make_gist(frame_id: u64, strand_id: u64, value: f32) -> FrameGist {
        FrameGist {
            vector: [value; SLOT_DIM],
            frame_id,
            strand_id,
            created_at: frame_id * 1000,
        }
    }

    fn make_directional_gist(frame_id: u64, strand_id: u64, direction_dim: usize) -> FrameGist {
        let mut vector = [0.0f32; SLOT_DIM];
        vector[direction_dim % SLOT_DIM] = 1.0;
        FrameGist {
            vector,
            frame_id,
            strand_id,
            created_at: frame_id * 1000,
        }
    }

    // --- StrandHnsw tests ---

    #[test]
    fn strand_hnsw_insert_and_query() {
        let mut idx = StrandHnsw::new(0, 100);
        idx.insert(&make_gist(1, 0, 0.1)).unwrap();

        let results = idx.query(&[0.1; SLOT_DIM], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].frame_id, 1);
        assert_eq!(results[0].strand_id, 0);
    }

    #[test]
    fn strand_hnsw_empty_query() {
        let idx = StrandHnsw::new(0, 100);
        let results = idx.query(&[0.1; SLOT_DIM], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn strand_hnsw_rejects_nan() {
        let mut idx = StrandHnsw::new(0, 100);
        let mut gist = make_gist(1, 0, 0.1);
        gist.vector[0] = f32::NAN;
        assert!(idx.insert(&gist).is_err());
    }

    #[test]
    fn strand_hnsw_multiple_entries_top_k() {
        let mut idx = StrandHnsw::new(0, 100);

        // Insert 20 gists with different directions
        for i in 0..20 {
            idx.insert(&make_directional_gist(i + 1, 0, i as usize))
                .unwrap();
        }

        // Query with direction matching frame 1 (dim 0)
        let mut query = [0.0f32; SLOT_DIM];
        query[0] = 1.0;
        let results = idx.query(&query, 5);

        assert_eq!(results.len(), 5);
        // The closest result should be frame 1 (same direction)
        assert_eq!(results[0].frame_id, 1);
    }

    // --- HnswIndex tests ---

    #[test]
    fn hnsw_index_insert_creates_strand() {
        let mut index = HnswIndex::new();
        index.insert(&make_gist(1, 5, 0.1)).unwrap();
        assert!(index.indexed_strands().contains(&5));
        assert_eq!(index.total_entries(), 1);
    }

    #[test]
    fn hnsw_index_query_strand_isolation() {
        let mut index = HnswIndex::new();

        // Strand 0: "coding" direction (dim 0)
        for i in 0..10 {
            index
                .insert(&make_directional_gist(i + 1, 0, 0))
                .unwrap();
        }

        // Strand 1: "cooking" direction (dim 100)
        for i in 0..10 {
            index
                .insert(&make_directional_gist(i + 11, 1, 100))
                .unwrap();
        }

        // Query strand 0 with "coding" direction
        let mut coding_query = [0.0f32; SLOT_DIM];
        coding_query[0] = 1.0;
        let results = index.query_strand(0, &coding_query, 5);
        assert_eq!(results.len(), 5);
        // All results should be from strand 0
        assert!(results.iter().all(|r| r.strand_id == 0));
    }

    #[test]
    fn hnsw_index_query_nonexistent_strand() {
        let index = HnswIndex::new();
        let results = index.query_strand(999, &[0.1; SLOT_DIM], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn hnsw_index_query_all_cross_strand() {
        let mut index = HnswIndex::new();

        // Strand 0: close to query
        let mut close = [0.0f32; SLOT_DIM];
        close[0] = 1.0;
        index
            .insert(&FrameGist {
                vector: close,
                frame_id: 1,
                strand_id: 0,
                created_at: 0,
            })
            .unwrap();

        // Strand 1: far from query
        let mut far = [0.0f32; SLOT_DIM];
        far[128] = 1.0;
        index
            .insert(&FrameGist {
                vector: far,
                frame_id: 2,
                strand_id: 1,
                created_at: 0,
            })
            .unwrap();

        let results = index.query_all(&close, 10);
        assert_eq!(results.len(), 2);
        // Closest first (strand 0)
        assert_eq!(results[0].frame_id, 1);
        assert_eq!(results[0].strand_id, 0);
    }

    #[test]
    fn hnsw_index_query_all_empty() {
        let index = HnswIndex::new();
        let results = index.query_all(&[0.1; SLOT_DIM], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn hnsw_index_total_entries() {
        let mut index = HnswIndex::new();
        for i in 0..5 {
            index.insert(&make_gist(i + 1, 0, 0.1)).unwrap();
        }
        for i in 0..3 {
            index.insert(&make_gist(i + 6, 1, 0.2)).unwrap();
        }
        assert_eq!(index.total_entries(), 8);
    }

    #[test]
    fn hnsw_index_query_k_zero() {
        let mut index = HnswIndex::new();
        index.insert(&make_gist(1, 0, 0.1)).unwrap();
        assert!(index.query_strand(0, &[0.1; SLOT_DIM], 0).is_empty());
        assert!(index.query_all(&[0.1; SLOT_DIM], 0).is_empty());
    }
}
