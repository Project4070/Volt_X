//! Ghost Bleed Buffer and Bleed Engine.
//!
//! The Ghost Bleed Buffer holds ~1000 R₀ gists from VoltDB memory,
//! making them accessible to the Soft Core's RAR Attend phase as
//! additional Key/Value sources in cross-attention.
//!
//! The Bleed Engine refreshes the ghost buffer on each new frame by
//! querying the HNSW index for semantically similar historical gists.

use volt_core::{VoltError, SLOT_DIM};

use crate::gist::FrameGist;
use crate::hnsw_index::HnswIndex;

/// Maximum number of ghost gists in the bleed buffer.
pub const GHOST_BUFFER_CAPACITY: usize = 1000;

/// Default number of similar gists to retrieve per HNSW query.
const DEFAULT_QUERY_K: usize = 100;

/// Default cosine similarity threshold for ghost inclusion.
/// Only gists with similarity >= this value are kept.
const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.0;

/// A ghost gist entry in the bleed buffer.
///
/// Contains the R₀ vector and enough metadata for page-fault loading
/// (retrieving the full TensorFrame when a ghost strongly influences
/// the RAR loop).
///
/// # Example
///
/// ```
/// use volt_db::ghost::GhostEntry;
/// use volt_core::SLOT_DIM;
///
/// let entry = GhostEntry {
///     gist: [0.1; SLOT_DIM],
///     frame_id: 42,
///     strand_id: 1,
///     relevance: 0.85,
/// };
/// assert_eq!(entry.frame_id, 42);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GhostEntry {
    /// The R₀ gist vector (256 dims, unit normalized).
    pub gist: [f32; SLOT_DIM],
    /// The frame ID this gist came from (for page-fault full-frame loading).
    pub frame_id: u64,
    /// The strand ID this gist belongs to.
    pub strand_id: u64,
    /// Cosine similarity to the query that placed this ghost (0.0–1.0).
    pub relevance: f32,
}

/// The Ghost Bleed Buffer — holds R₀ gists for Soft Core cross-attention.
///
/// Populated by the [`BleedEngine`] and consumed by the RAR Attend phase
/// as additional Key/Value sources. The buffer is refreshed on each new
/// frame, replacing the previous contents.
///
/// # Example
///
/// ```
/// use volt_db::ghost::GhostBuffer;
///
/// let buffer = GhostBuffer::new();
/// assert!(buffer.is_empty());
/// assert_eq!(buffer.len(), 0);
/// ```
#[derive(Debug, Clone)]
pub struct GhostBuffer {
    entries: Vec<GhostEntry>,
    capacity: usize,
}

impl Default for GhostBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl GhostBuffer {
    /// Creates a new ghost buffer with the default capacity (1000).
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::ghost::GhostBuffer;
    ///
    /// let buffer = GhostBuffer::new();
    /// assert!(buffer.is_empty());
    /// ```
    pub fn new() -> Self {
        Self::with_capacity(GHOST_BUFFER_CAPACITY)
    }

    /// Creates a new ghost buffer with the given capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::ghost::GhostBuffer;
    ///
    /// let buffer = GhostBuffer::with_capacity(500);
    /// assert!(buffer.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Replaces the buffer contents with new ghost entries.
    ///
    /// Entries beyond the buffer capacity are silently dropped.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::ghost::{GhostBuffer, GhostEntry};
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut buffer = GhostBuffer::new();
    /// let entries = vec![GhostEntry {
    ///     gist: [0.1; SLOT_DIM],
    ///     frame_id: 1,
    ///     strand_id: 0,
    ///     relevance: 0.9,
    /// }];
    /// buffer.refresh(entries);
    /// assert_eq!(buffer.len(), 1);
    /// ```
    pub fn refresh(&mut self, mut entries: Vec<GhostEntry>) {
        entries.truncate(self.capacity);
        self.entries = entries;
    }

    /// Returns the current ghost entries as a slice.
    pub fn entries(&self) -> &[GhostEntry] {
        &self.entries
    }

    /// Returns just the gist vectors, suitable for passing to SlotAttention.
    ///
    /// This is the primary interface for the RAR Attend phase.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::ghost::{GhostBuffer, GhostEntry};
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut buffer = GhostBuffer::new();
    /// buffer.refresh(vec![GhostEntry {
    ///     gist: [0.5; SLOT_DIM],
    ///     frame_id: 1,
    ///     strand_id: 0,
    ///     relevance: 0.9,
    /// }]);
    /// let gists = buffer.gist_vectors();
    /// assert_eq!(gists.len(), 1);
    /// assert_eq!(gists[0], [0.5; SLOT_DIM]);
    /// ```
    pub fn gist_vectors(&self) -> Vec<[f32; SLOT_DIM]> {
        self.entries.iter().map(|e| e.gist).collect()
    }

    /// Returns the number of ghost entries in the buffer.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clears the buffer, removing all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns the buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// The Bleed Engine queries the HNSW index and refreshes the ghost buffer.
///
/// On each new frame, the engine:
/// 1. Queries the HNSW index for the most similar historical gists
/// 2. Filters results by a cosine similarity threshold
/// 3. Replaces the ghost buffer contents with the filtered results
///
/// # Example
///
/// ```
/// use volt_db::ghost::BleedEngine;
///
/// let engine = BleedEngine::new();
/// assert!(engine.buffer().is_empty());
/// ```
pub struct BleedEngine {
    buffer: GhostBuffer,
    /// Number of results to request from HNSW per query.
    query_k: usize,
    /// Cosine similarity threshold for ghost inclusion.
    similarity_threshold: f32,
}

impl std::fmt::Debug for BleedEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BleedEngine(ghosts={}, query_k={}, threshold={})",
            self.buffer.len(),
            self.query_k,
            self.similarity_threshold
        )
    }
}

impl Default for BleedEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BleedEngine {
    /// Creates a new Bleed Engine with default settings.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::ghost::BleedEngine;
    ///
    /// let engine = BleedEngine::new();
    /// assert!(engine.buffer().is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            buffer: GhostBuffer::new(),
            query_k: DEFAULT_QUERY_K,
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
        }
    }

    /// Called when a new frame is stored.
    ///
    /// Queries the HNSW index for similar gists and refreshes the
    /// ghost buffer with the results. The previous buffer contents
    /// are completely replaced.
    ///
    /// # Errors
    ///
    /// Returns [`VoltError`] if the HNSW query encounters an error.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::ghost::BleedEngine;
    /// use volt_db::hnsw_index::HnswIndex;
    /// use volt_db::gist::FrameGist;
    /// use volt_core::SLOT_DIM;
    ///
    /// let mut engine = BleedEngine::new();
    /// let mut index = HnswIndex::new();
    ///
    /// // Insert some gists into the index
    /// let gist = FrameGist {
    ///     vector: [0.1; SLOT_DIM],
    ///     frame_id: 1,
    ///     strand_id: 0,
    ///     created_at: 0,
    /// };
    /// index.insert(&gist).unwrap();
    ///
    /// // Trigger bleed on new frame
    /// let new_gist = FrameGist {
    ///     vector: [0.1; SLOT_DIM],
    ///     frame_id: 2,
    ///     strand_id: 0,
    ///     created_at: 1000,
    /// };
    /// engine.on_new_frame(&new_gist, &index).unwrap();
    /// assert!(!engine.buffer().is_empty());
    /// ```
    pub fn on_new_frame(
        &mut self,
        gist: &FrameGist,
        index: &HnswIndex,
    ) -> Result<(), VoltError> {
        let results = index.query_all(&gist.vector, self.query_k);

        let entries: Vec<GhostEntry> = results
            .into_iter()
            .filter_map(|r| {
                // DistCosine returns distance in [0, 2]: 0 = identical, 2 = opposite.
                // Convert to similarity in [-1, 1].
                let similarity = 1.0 - r.distance;
                if similarity >= self.similarity_threshold {
                    Some(GhostEntry {
                        gist: r.gist,
                        frame_id: r.frame_id,
                        strand_id: r.strand_id,
                        relevance: similarity,
                    })
                } else {
                    None
                }
            })
            .collect();

        self.buffer.refresh(entries);
        Ok(())
    }

    /// Returns a reference to the current ghost buffer.
    pub fn buffer(&self) -> &GhostBuffer {
        &self.buffer
    }

    /// Returns a mutable reference to the ghost buffer.
    pub fn buffer_mut(&mut self) -> &mut GhostBuffer {
        &mut self.buffer
    }

    /// Sets the number of HNSW results to request per query.
    pub fn set_query_k(&mut self, k: usize) {
        self.query_k = k;
    }

    /// Sets the cosine similarity threshold for ghost inclusion.
    pub fn set_similarity_threshold(&mut self, threshold: f32) {
        self.similarity_threshold = threshold;
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

    fn make_directional_gist(frame_id: u64, strand_id: u64, dim: usize) -> FrameGist {
        let mut vector = [0.0f32; SLOT_DIM];
        vector[dim % SLOT_DIM] = 1.0;
        FrameGist {
            vector,
            frame_id,
            strand_id,
            created_at: frame_id * 1000,
        }
    }

    // --- GhostBuffer tests ---

    #[test]
    fn buffer_new_is_empty() {
        let buf = GhostBuffer::new();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), GHOST_BUFFER_CAPACITY);
    }

    #[test]
    fn buffer_refresh_replaces_contents() {
        let mut buf = GhostBuffer::new();

        let entries1 = vec![GhostEntry {
            gist: [0.1; SLOT_DIM],
            frame_id: 1,
            strand_id: 0,
            relevance: 0.9,
        }];
        buf.refresh(entries1);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.entries()[0].frame_id, 1);

        let entries2 = vec![
            GhostEntry {
                gist: [0.2; SLOT_DIM],
                frame_id: 2,
                strand_id: 0,
                relevance: 0.8,
            },
            GhostEntry {
                gist: [0.3; SLOT_DIM],
                frame_id: 3,
                strand_id: 0,
                relevance: 0.7,
            },
        ];
        buf.refresh(entries2);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.entries()[0].frame_id, 2);
    }

    #[test]
    fn buffer_respects_capacity() {
        let mut buf = GhostBuffer::with_capacity(2);

        let entries: Vec<GhostEntry> = (0..5)
            .map(|i| GhostEntry {
                gist: [0.1; SLOT_DIM],
                frame_id: i,
                strand_id: 0,
                relevance: 0.5,
            })
            .collect();
        buf.refresh(entries);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn buffer_gist_vectors() {
        let mut buf = GhostBuffer::new();
        buf.refresh(vec![
            GhostEntry {
                gist: [0.1; SLOT_DIM],
                frame_id: 1,
                strand_id: 0,
                relevance: 0.9,
            },
            GhostEntry {
                gist: [0.2; SLOT_DIM],
                frame_id: 2,
                strand_id: 0,
                relevance: 0.8,
            },
        ]);

        let gists = buf.gist_vectors();
        assert_eq!(gists.len(), 2);
        assert_eq!(gists[0], [0.1; SLOT_DIM]);
        assert_eq!(gists[1], [0.2; SLOT_DIM]);
    }

    #[test]
    fn buffer_clear() {
        let mut buf = GhostBuffer::new();
        buf.refresh(vec![GhostEntry {
            gist: [0.1; SLOT_DIM],
            frame_id: 1,
            strand_id: 0,
            relevance: 0.9,
        }]);
        assert_eq!(buf.len(), 1);
        buf.clear();
        assert!(buf.is_empty());
    }

    // --- BleedEngine tests ---

    #[test]
    fn bleed_engine_empty_index() {
        let mut engine = BleedEngine::new();
        let index = HnswIndex::new();
        let gist = make_gist(1, 0, 0.1);
        engine.on_new_frame(&gist, &index).unwrap();
        assert!(engine.buffer().is_empty());
    }

    #[test]
    fn bleed_engine_populates_buffer() {
        let mut engine = BleedEngine::new();
        let mut index = HnswIndex::new();

        // Insert 10 gists
        for i in 0..10 {
            index.insert(&make_gist(i + 1, 0, 0.1)).unwrap();
        }

        // Trigger bleed with a similar gist
        let query = make_gist(11, 0, 0.1);
        engine.on_new_frame(&query, &index).unwrap();
        assert!(!engine.buffer().is_empty());
        assert!(engine.buffer().len() <= 10);
    }

    #[test]
    fn bleed_engine_refresh_replaces_buffer() {
        let mut engine = BleedEngine::new();
        let mut index = HnswIndex::new();

        // Insert topic A gists (dim 0)
        for i in 0..5 {
            index
                .insert(&make_directional_gist(i + 1, 0, 0))
                .unwrap();
        }

        // Insert topic B gists (dim 100)
        for i in 0..5 {
            index
                .insert(&make_directional_gist(i + 6, 0, 100))
                .unwrap();
        }

        // Query with topic A → buffer should have topic A gists
        engine
            .on_new_frame(&make_directional_gist(20, 0, 0), &index)
            .unwrap();
        let buf_a: Vec<u64> = engine.buffer().entries().iter().map(|e| e.frame_id).collect();

        // Query with topic B → buffer should change
        engine
            .on_new_frame(&make_directional_gist(21, 0, 100), &index)
            .unwrap();
        let buf_b: Vec<u64> = engine.buffer().entries().iter().map(|e| e.frame_id).collect();

        // Buffers should differ — topic A gists should rank higher in buf_a,
        // topic B gists higher in buf_b
        assert_ne!(buf_a, buf_b);
    }

    #[test]
    fn bleed_engine_similarity_threshold() {
        let mut engine = BleedEngine::new();
        engine.set_similarity_threshold(0.99); // Very strict

        let mut index = HnswIndex::new();

        // Insert a gist pointing in dim 0
        index
            .insert(&make_directional_gist(1, 0, 0))
            .unwrap();

        // Query with a very different direction (dim 128) — should be below threshold
        engine
            .on_new_frame(&make_directional_gist(2, 0, 128), &index)
            .unwrap();
        // With such a strict threshold, the orthogonal gist should be filtered out
        assert!(
            engine.buffer().is_empty(),
            "orthogonal gist should be filtered by strict threshold"
        );
    }
}
