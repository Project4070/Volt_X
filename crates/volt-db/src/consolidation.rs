//! Frame consolidation: cluster detection + wisdom frame creation.
//!
//! Groups semantically similar frames within a strand and creates
//! "wisdom frames" — high-certainty summary frames that capture
//! the distilled knowledge of the cluster.
//!
//! ## Algorithm
//!
//! 1. For each frame gist in a strand, query HNSW for similar neighbors
//! 2. Group frames into clusters using greedy union-find
//! 3. For each cluster above min size, average R₀ vectors into a wisdom frame
//! 4. Mark source frames as superseded by the wisdom frame

use std::collections::HashMap;

use volt_core::meta::DiscourseType;
use volt_core::slot::{SlotData, SlotRole, SlotSource};
use volt_core::{TensorFrame, SLOT_DIM};

use crate::gist::FrameGist;
use crate::hnsw_index::HnswIndex;

/// Configuration for frame consolidation.
///
/// # Example
///
/// ```
/// use volt_db::consolidation::ConsolidationConfig;
///
/// let config = ConsolidationConfig::default();
/// assert_eq!(config.min_cluster_size, 5);
/// ```
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Minimum cluster size to trigger consolidation. Default: 5.
    pub min_cluster_size: usize,
    /// Similarity threshold for grouping (cosine, 0.0-1.0). Default: 0.85.
    pub similarity_threshold: f32,
    /// Certainty (gamma) assigned to wisdom frames. Default: 0.95.
    pub wisdom_gamma: f32,
    /// Number of HNSW neighbors to query per frame. Default: 20.
    pub query_k: usize,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 5,
            similarity_threshold: 0.85,
            wisdom_gamma: 0.95,
            query_k: 20,
        }
    }
}

/// A cluster of semantically similar frames.
///
/// # Example
///
/// ```
/// use volt_db::consolidation::FrameCluster;
/// use volt_core::SLOT_DIM;
///
/// let cluster = FrameCluster {
///     member_frame_ids: vec![1, 2, 3, 4, 5],
///     centroid: [0.5; SLOT_DIM],
///     average_certainty: 0.7,
/// };
/// assert_eq!(cluster.member_frame_ids.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct FrameCluster {
    /// Frame IDs in this cluster.
    pub member_frame_ids: Vec<u64>,
    /// Centroid (average) R₀ gist vector.
    pub centroid: [f32; SLOT_DIM],
    /// Average certainty across cluster members.
    pub average_certainty: f32,
}

/// Result of a consolidation pass.
///
/// # Example
///
/// ```
/// use volt_db::consolidation::ConsolidationResult;
///
/// let result = ConsolidationResult {
///     clusters_found: 0,
///     wisdom_frames: Vec::new(),
///     superseded_frame_ids: Vec::new(),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    /// Number of clusters detected.
    pub clusters_found: usize,
    /// The wisdom frames created (one per cluster).
    pub wisdom_frames: Vec<TensorFrame>,
    /// Source frame IDs that were superseded by wisdom frames.
    pub superseded_frame_ids: Vec<u64>,
}

/// Engine for detecting clusters and creating wisdom frames.
///
/// # Example
///
/// ```
/// use volt_db::consolidation::ConsolidationEngine;
///
/// let engine = ConsolidationEngine::with_defaults();
/// ```
#[derive(Debug, Clone)]
pub struct ConsolidationEngine {
    config: ConsolidationConfig,
}

impl ConsolidationEngine {
    /// Creates a consolidation engine with the given configuration.
    pub fn new(config: ConsolidationConfig) -> Self {
        Self { config }
    }

    /// Creates a consolidation engine with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ConsolidationConfig::default())
    }

    /// Finds clusters of similar frames in a strand using HNSW.
    ///
    /// Uses greedy union-find: for each gist, query HNSW for neighbors
    /// above the similarity threshold, then merge groups.
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::consolidation::ConsolidationEngine;
    /// use volt_db::hnsw_index::HnswIndex;
    ///
    /// let engine = ConsolidationEngine::with_defaults();
    /// let hnsw = HnswIndex::new();
    ///
    /// // With empty index, no clusters are found
    /// let clusters = engine.find_clusters(0, &hnsw, &[]);
    /// assert!(clusters.is_empty());
    /// ```
    pub fn find_clusters(
        &self,
        strand_id: u64,
        hnsw: &HnswIndex,
        gists: &[FrameGist],
    ) -> Vec<FrameCluster> {
        if gists.is_empty() {
            return Vec::new();
        }

        // Union-Find for clustering
        let id_to_idx: HashMap<u64, usize> = gists
            .iter()
            .enumerate()
            .map(|(i, g)| (g.frame_id, i))
            .collect();
        let mut parent: Vec<usize> = (0..gists.len()).collect();

        // Query HNSW for each gist and union similar frames
        for gist in gists {
            let results = hnsw.query_strand(strand_id, &gist.vector, self.config.query_k);
            for result in results {
                if result.frame_id == gist.frame_id {
                    continue;
                }
                let similarity = 1.0 - result.distance;
                if similarity >= self.config.similarity_threshold
                    && let (Some(&idx_a), Some(&idx_b)) = (
                        id_to_idx.get(&gist.frame_id),
                        id_to_idx.get(&result.frame_id),
                    )
                {
                    union(&mut parent, idx_a, idx_b);
                }
            }
        }

        // Group by root
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..gists.len() {
            let root = find(&parent, i);
            groups.entry(root).or_default().push(i);
        }

        // Filter by minimum cluster size and build FrameCluster
        let mut clusters = Vec::new();
        for indices in groups.values() {
            if indices.len() < self.config.min_cluster_size {
                continue;
            }

            let mut centroid = [0.0f32; SLOT_DIM];
            let mut total_certainty = 0.0f32;
            let mut member_ids = Vec::with_capacity(indices.len());

            for &idx in indices {
                member_ids.push(gists[idx].frame_id);
                for (d, &v) in centroid.iter_mut().zip(gists[idx].vector.iter()) {
                    *d += v;
                }
                // We don't have gamma in FrameGist, so we use a default
                total_certainty += 0.5; // Will be refined when source frames are available
            }

            // Normalize centroid
            let n = indices.len() as f32;
            for d in &mut centroid {
                *d /= n;
            }
            l2_normalize(&mut centroid);

            clusters.push(FrameCluster {
                member_frame_ids: member_ids,
                centroid,
                average_certainty: total_certainty / n,
            });
        }

        clusters
    }

    /// Creates a wisdom frame by averaging R₀ vectors of source frames.
    ///
    /// The wisdom frame gets:
    /// - A single Agent slot with the averaged R₀ vector
    /// - High gamma (configurable, default 0.95)
    /// - DiscourseType::Response
    /// - verified = true
    ///
    /// # Example
    ///
    /// ```
    /// use volt_db::consolidation::{ConsolidationEngine, FrameCluster};
    /// use volt_core::{TensorFrame, SlotData, SlotRole, SLOT_DIM};
    ///
    /// let engine = ConsolidationEngine::with_defaults();
    ///
    /// let mut frames = Vec::new();
    /// for i in 0..5 {
    ///     let mut f = TensorFrame::new();
    ///     let mut slot = SlotData::new(SlotRole::Agent);
    ///     slot.write_resolution(0, [0.1 * i as f32; SLOT_DIM]);
    ///     f.write_slot(0, slot).unwrap();
    ///     frames.push(f);
    /// }
    ///
    /// let cluster = FrameCluster {
    ///     member_frame_ids: vec![1, 2, 3, 4, 5],
    ///     centroid: [0.5; SLOT_DIM],
    ///     average_certainty: 0.7,
    /// };
    ///
    /// let refs: Vec<&TensorFrame> = frames.iter().collect();
    /// let wisdom = engine.create_wisdom_frame(&cluster, &refs, 0, 100);
    /// assert_eq!(wisdom.frame_meta.frame_id, 100);
    /// assert!(wisdom.frame_meta.global_certainty >= 0.9);
    /// ```
    pub fn create_wisdom_frame(
        &self,
        _cluster: &FrameCluster,
        source_frames: &[&TensorFrame],
        strand_id: u64,
        frame_id: u64,
    ) -> TensorFrame {
        // Average R₀ vectors across all source frames, per slot
        let mut slot_sums: [Option<[f64; SLOT_DIM]>; 16] = [const { None }; 16];
        let mut slot_counts: [usize; 16] = [0; 16];
        let mut slot_roles: [Option<SlotRole>; 16] = [const { None }; 16];

        for frame in source_frames {
            for (i, slot_opt) in frame.slots.iter().enumerate() {
                if let Some(slot) = slot_opt
                    && let Some(r0) = &slot.resolutions[0]
                {
                    let sum = slot_sums[i].get_or_insert([0.0f64; SLOT_DIM]);
                    for (s, &v) in sum.iter_mut().zip(r0.iter()) {
                        *s += v as f64;
                    }
                    slot_counts[i] += 1;
                    if slot_roles[i].is_none() {
                        slot_roles[i] = Some(slot.role);
                    }
                }
            }
        }

        // Build the wisdom frame
        let mut wisdom = TensorFrame::new();
        wisdom.frame_meta.frame_id = frame_id;
        wisdom.frame_meta.strand_id = strand_id;
        wisdom.frame_meta.global_certainty = self.config.wisdom_gamma;
        wisdom.frame_meta.discourse_type = DiscourseType::Response;
        wisdom.frame_meta.verified = true;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        wisdom.frame_meta.created_at = now;

        for i in 0..16 {
            if let Some(sum) = &slot_sums[i]
                && slot_counts[i] > 0
            {
                let role = slot_roles[i].unwrap_or(SlotRole::Free(i as u8));
                let mut averaged = [0.0f32; SLOT_DIM];
                let n = slot_counts[i] as f64;
                for (a, &s) in averaged.iter_mut().zip(sum.iter()) {
                    *a = (s / n) as f32;
                }
                l2_normalize(&mut averaged);

                let mut slot = SlotData::new(role);
                slot.write_resolution(0, averaged);
                // Ignore error — we know the index is valid
                let _ = wisdom.write_slot(i, slot);

                wisdom.meta[i].certainty = self.config.wisdom_gamma;
                wisdom.meta[i].source = SlotSource::Memory;
                wisdom.meta[i].updated_at = now;
            }
        }

        wisdom
    }

    /// Returns the configuration.
    pub fn config(&self) -> &ConsolidationConfig {
        &self.config
    }
}

/// Union-Find: find root with path compression.
fn find(parent: &[usize], mut i: usize) -> usize {
    // Note: we don't do path compression in this immutable version,
    // we just follow parent pointers to the root.
    while parent[i] != i {
        i = parent[i];
    }
    i
}

/// Union-Find: merge two sets.
fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        // Simple union: always attach rb under ra
        parent[rb] = ra;
    }
}

/// L2-normalizes a vector in place.
fn l2_normalize(v: &mut [f32; SLOT_DIM]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::slot::{SlotData, SlotRole};

    fn make_gist(id: u64, strand: u64, value: f32) -> FrameGist {
        let mut vector = [value; SLOT_DIM];
        l2_normalize(&mut vector);
        FrameGist {
            vector,
            frame_id: id,
            strand_id: strand,
            created_at: id * 1000,
        }
    }

    fn make_frame_with_r0(id: u64, strand: u64, value: f32) -> TensorFrame {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        let mut vec = [value; SLOT_DIM];
        l2_normalize(&mut vec);
        slot.write_resolution(0, vec);
        frame.write_slot(0, slot).unwrap();
        frame.frame_meta.frame_id = id;
        frame.frame_meta.strand_id = strand;
        frame.frame_meta.global_certainty = 0.5;
        frame
    }

    #[test]
    fn empty_gists_no_clusters() {
        let engine = ConsolidationEngine::with_defaults();
        let hnsw = HnswIndex::new();
        let clusters = engine.find_clusters(0, &hnsw, &[]);
        assert!(clusters.is_empty());
    }

    #[test]
    fn small_cluster_skipped() {
        let engine = ConsolidationEngine::new(ConsolidationConfig {
            min_cluster_size: 10, // Require large clusters
            ..Default::default()
        });

        let mut hnsw = HnswIndex::new();
        let gists: Vec<FrameGist> = (1..=5)
            .map(|i| make_gist(i, 0, 0.5))
            .collect();
        for g in &gists {
            hnsw.insert(g).unwrap();
        }

        let clusters = engine.find_clusters(0, &hnsw, &gists);
        // 5 similar gists but min_cluster_size is 10 → no clusters
        assert!(clusters.is_empty());
    }

    #[test]
    fn cluster_detection_similar_frames() {
        let config = ConsolidationConfig {
            min_cluster_size: 3,
            similarity_threshold: 0.9,
            query_k: 20,
            ..Default::default()
        };
        let engine = ConsolidationEngine::new(config);

        let mut hnsw = HnswIndex::new();

        // Create a cluster of similar gists (all value=0.5)
        let similar_gists: Vec<FrameGist> = (1..=5)
            .map(|i| make_gist(i, 0, 0.5))
            .collect();
        for g in &similar_gists {
            hnsw.insert(g).unwrap();
        }

        // Create a distinct gist (value=0.9, different direction after normalize)
        let distinct = make_gist(100, 0, 0.9);
        hnsw.insert(&distinct).unwrap();

        let mut all_gists = similar_gists;
        all_gists.push(distinct);

        let clusters = engine.find_clusters(0, &hnsw, &all_gists);

        // Should find at least 1 cluster of the 5 similar frames
        // (The 5 identical gists should cluster together)
        assert!(
            !clusters.is_empty(),
            "should find at least one cluster"
        );

        let biggest = clusters.iter().max_by_key(|c| c.member_frame_ids.len()).unwrap();
        assert!(
            biggest.member_frame_ids.len() >= 3,
            "largest cluster should have >= 3 members, got {}",
            biggest.member_frame_ids.len()
        );
    }

    #[test]
    fn wisdom_frame_has_high_gamma() {
        let engine = ConsolidationEngine::with_defaults();
        let frames: Vec<TensorFrame> = (1..=5)
            .map(|i| make_frame_with_r0(i, 0, 0.5))
            .collect();
        let frame_refs: Vec<&TensorFrame> = frames.iter().collect();

        let cluster = FrameCluster {
            member_frame_ids: (1..=5).collect(),
            centroid: [0.5; SLOT_DIM],
            average_certainty: 0.5,
        };

        let wisdom = engine.create_wisdom_frame(&cluster, &frame_refs, 0, 100);
        assert_eq!(wisdom.frame_meta.frame_id, 100);
        assert_eq!(wisdom.frame_meta.strand_id, 0);
        assert_eq!(wisdom.frame_meta.global_certainty, 0.95);
        assert!(wisdom.frame_meta.verified);
    }

    #[test]
    fn wisdom_frame_averaged_r0() {
        let engine = ConsolidationEngine::with_defaults();

        // Create frames with slightly different R0 vectors
        let frames: Vec<TensorFrame> = (1..=3)
            .map(|i| {
                let mut frame = TensorFrame::new();
                let mut slot = SlotData::new(SlotRole::Agent);
                let mut vec = [0.0f32; SLOT_DIM];
                vec[0] = i as f32; // Different first element
                vec[1] = 1.0; // Same second element
                slot.write_resolution(0, vec);
                frame.write_slot(0, slot).unwrap();
                frame.frame_meta.frame_id = i;
                frame
            })
            .collect();
        let refs: Vec<&TensorFrame> = frames.iter().collect();

        let cluster = FrameCluster {
            member_frame_ids: vec![1, 2, 3],
            centroid: [0.5; SLOT_DIM],
            average_certainty: 0.5,
        };

        let wisdom = engine.create_wisdom_frame(&cluster, &refs, 0, 100);

        // Check that slot 0 exists with an R0 vector
        let slot = wisdom.slots[0].as_ref().expect("slot 0 should exist");
        let r0 = slot.resolutions[0].expect("R0 should exist");

        // The average of [1, 2, 3] in dim 0 is 2.0 (before normalize)
        // After normalization it won't be exactly 2.0 but the ratio
        // r0[0]/r0[1] should be approximately 2.0/1.0 = 2.0
        let ratio = r0[0] / r0[1];
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "ratio should be ~2.0, got {ratio}"
        );
    }

    #[test]
    fn wisdom_frame_multiple_slots() {
        let engine = ConsolidationEngine::with_defaults();

        let frames: Vec<TensorFrame> = (1..=3)
            .map(|i| {
                let mut frame = TensorFrame::new();
                // Slot 0: Agent
                let mut s0 = SlotData::new(SlotRole::Agent);
                s0.write_resolution(0, [0.1 * i as f32; SLOT_DIM]);
                frame.write_slot(0, s0).unwrap();
                // Slot 1: Predicate
                let mut s1 = SlotData::new(SlotRole::Predicate);
                s1.write_resolution(0, [0.2 * i as f32; SLOT_DIM]);
                frame.write_slot(1, s1).unwrap();
                frame.frame_meta.frame_id = i;
                frame
            })
            .collect();
        let refs: Vec<&TensorFrame> = frames.iter().collect();

        let cluster = FrameCluster {
            member_frame_ids: vec![1, 2, 3],
            centroid: [0.5; SLOT_DIM],
            average_certainty: 0.5,
        };

        let wisdom = engine.create_wisdom_frame(&cluster, &refs, 0, 100);

        // Both slots should be present
        assert!(wisdom.slots[0].is_some(), "slot 0 should exist");
        assert!(wisdom.slots[1].is_some(), "slot 1 should exist");
        assert!(wisdom.slots[2].is_none(), "slot 2 should not exist");
    }

    #[test]
    fn union_find_basic() {
        let mut parent = vec![0, 1, 2, 3, 4];
        union(&mut parent, 0, 1);
        union(&mut parent, 2, 3);
        union(&mut parent, 0, 3);

        assert_eq!(find(&parent, 0), find(&parent, 1));
        assert_eq!(find(&parent, 0), find(&parent, 2));
        assert_eq!(find(&parent, 0), find(&parent, 3));
        assert_ne!(find(&parent, 0), find(&parent, 4));
    }
}
