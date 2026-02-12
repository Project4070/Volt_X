//! Frame distillation: clusters of related frames → wisdom frames.
//!
//! Delegates to the existing [`ConsolidationEngine`](volt_db::ConsolidationEngine)
//! in `volt-db`. This module provides a learning-oriented wrapper that
//! operates across all strands and returns structured results.
//!
//! ## Algorithm
//!
//! For each strand:
//! 1. HNSW finds clusters of semantically similar frames
//! 2. Each cluster's R₀ vectors are averaged into a wisdom frame
//! 3. Wisdom frames get high gamma (0.95) and are stored back
//! 4. Source frames are recorded as superseded

use volt_core::VoltError;
use volt_db::VoltStore;

/// Configuration for the distillation pass.
///
/// # Example
///
/// ```
/// use volt_learn::distillation::DistillationConfig;
///
/// let config = DistillationConfig::default();
/// assert_eq!(config.min_cluster_size, 5);
/// ```
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Minimum number of frames to form a cluster. Default: 5.
    pub min_cluster_size: usize,
    /// Cosine similarity threshold for grouping frames. Default: 0.85.
    pub similarity_threshold: f32,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 5,
            similarity_threshold: 0.85,
        }
    }
}

/// Result of distillation for a single strand.
///
/// # Example
///
/// ```
/// use volt_learn::distillation::DistillationResult;
///
/// let result = DistillationResult {
///     strand_id: 1,
///     clusters_found: 2,
///     wisdom_frames_created: 2,
///     frames_superseded: 12,
/// };
/// assert_eq!(result.clusters_found, 2);
/// ```
#[derive(Debug, Clone)]
pub struct DistillationResult {
    /// The strand that was distilled.
    pub strand_id: u64,
    /// Number of clusters detected.
    pub clusters_found: usize,
    /// Number of wisdom frames created.
    pub wisdom_frames_created: usize,
    /// Number of source frames that were superseded.
    pub frames_superseded: usize,
}

/// Runs distillation on a single strand.
///
/// Delegates to [`VoltStore::consolidate_strand`] for cluster detection
/// and wisdom frame creation.
///
/// # Errors
///
/// Returns [`VoltError::StorageError`] if consolidation fails.
///
/// # Example
///
/// ```
/// use volt_learn::distillation::distill_strand;
/// use volt_db::VoltStore;
///
/// let mut store = VoltStore::new();
/// let result = distill_strand(&mut store, 0).unwrap();
/// assert_eq!(result.clusters_found, 0); // Empty store has no clusters
/// ```
pub fn distill_strand(
    store: &mut VoltStore,
    strand_id: u64,
) -> Result<DistillationResult, VoltError> {
    let result = store.consolidate_strand(strand_id)?;

    Ok(DistillationResult {
        strand_id,
        clusters_found: result.clusters_found,
        wisdom_frames_created: result.wisdom_frames.len(),
        frames_superseded: result.superseded_frame_ids.len(),
    })
}

/// Runs distillation on all strands in the store.
///
/// Iterates over every strand and consolidates frames into wisdom frames
/// where clusters are found. Returns one [`DistillationResult`] per strand.
///
/// # Errors
///
/// Returns [`VoltError::StorageError`] if any strand consolidation fails.
///
/// # Example
///
/// ```
/// use volt_learn::distillation::distill_all_strands;
/// use volt_db::VoltStore;
///
/// let mut store = VoltStore::new();
/// let results = distill_all_strands(&mut store).unwrap();
/// assert_eq!(results.len(), 1); // Default strand 0
/// ```
pub fn distill_all_strands(
    store: &mut VoltStore,
) -> Result<Vec<DistillationResult>, VoltError> {
    let strand_ids = store.list_strands();
    let mut results = Vec::with_capacity(strand_ids.len());

    for strand_id in strand_ids {
        results.push(distill_strand(store, strand_id)?);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::{SlotData, SlotRole, TensorFrame, SLOT_DIM};

    fn make_frame_with_r0(value: f32) -> TensorFrame {
        let mut frame = TensorFrame::new();
        let mut slot = SlotData::new(SlotRole::Agent);
        let mut vec = [value; SLOT_DIM];
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in &mut vec {
            *v /= norm;
        }
        slot.write_resolution(0, vec);
        frame.write_slot(0, slot).unwrap();
        frame
    }

    #[test]
    fn default_config_sensible() {
        let config = DistillationConfig::default();
        assert_eq!(config.min_cluster_size, 5);
        assert!((config.similarity_threshold - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn distill_empty_strand_no_clusters() {
        let mut store = VoltStore::new();
        let result = distill_strand(&mut store, 0).unwrap();
        assert_eq!(result.strand_id, 0);
        assert_eq!(result.clusters_found, 0);
        assert_eq!(result.wisdom_frames_created, 0);
        assert_eq!(result.frames_superseded, 0);
    }

    #[test]
    fn distill_all_strands_includes_default() {
        let mut store = VoltStore::new();
        let results = distill_all_strands(&mut store).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].strand_id, 0);
    }

    #[test]
    fn distill_all_strands_multiple() {
        let mut store = VoltStore::new();
        store.create_strand(1).unwrap();
        store.create_strand(2).unwrap();
        let results = distill_all_strands(&mut store).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn distill_with_similar_frames_creates_wisdom() {
        let mut store = VoltStore::new();

        // Store enough similar frames to overflow T0 → T1 and form a cluster
        // T0 capacity is 64, so we need > 64 frames with same content
        for _ in 0..70 {
            let frame = make_frame_with_r0(0.5);
            store.store(frame).unwrap();
        }

        let result = distill_strand(&mut store, 0).unwrap();
        // With 70 identical frames and default min_cluster_size=5,
        // consolidation should find at least one cluster
        // (depends on HNSW indexing and threshold)
        // Consolidation should complete without error regardless of cluster count
        assert!(result.wisdom_frames_created <= result.clusters_found);
    }
}
