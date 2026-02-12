//! Strand graduation: detect novel topic clusters and promote to new strands.
//!
//! When a strand accumulates a large cluster of frames about a topic
//! that is distinct from the strand's overall content, those frames
//! "graduate" into a new strand. This enables specialization over time.
//!
//! ## Algorithm
//!
//! 1. For each strand, group learning events by frame_id
//! 2. Query HNSW for clusters of related frames within the strand
//! 3. If a cluster has ≥ `min_cluster_frames` members, check whether
//!    the cluster centroid is dissimilar to the overall strand centroid
//! 4. If dissimilar enough, create a new strand and migrate the frames

use std::collections::HashMap;

use volt_core::{VoltError, SLOT_DIM};
use volt_db::VoltStore;

use crate::event::LearningEvent;

/// Configuration for strand graduation.
///
/// # Example
///
/// ```
/// use volt_learn::graduation::GraduationConfig;
///
/// let config = GraduationConfig::default();
/// assert_eq!(config.min_cluster_frames, 50);
/// ```
#[derive(Debug, Clone)]
pub struct GraduationConfig {
    /// Minimum frames in a cluster to trigger graduation. Default: 50.
    pub min_cluster_frames: usize,
    /// Cosine similarity threshold for frames within a cluster. Default: 0.7.
    pub internal_similarity_threshold: f32,
    /// Maximum new strands to create per sleep cycle. Default: 3.
    pub max_new_strands_per_cycle: usize,
}

impl Default for GraduationConfig {
    fn default() -> Self {
        Self {
            min_cluster_frames: 50,
            internal_similarity_threshold: 0.7,
            max_new_strands_per_cycle: 3,
        }
    }
}

/// Result of a strand graduation check.
///
/// # Example
///
/// ```
/// use volt_learn::graduation::GraduationResult;
///
/// let result = GraduationResult {
///     new_strands_created: vec![],
///     frames_migrated: 0,
/// };
/// assert!(result.new_strands_created.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct GraduationResult {
    /// IDs of newly created strands.
    pub new_strands_created: Vec<u64>,
    /// Total number of frames migrated to new strands.
    pub frames_migrated: usize,
}

/// Computes the centroid (average) of a set of R₀ gist vectors.
fn compute_centroid(vectors: &[[f32; SLOT_DIM]]) -> [f32; SLOT_DIM] {
    let mut centroid = [0.0f32; SLOT_DIM];
    if vectors.is_empty() {
        return centroid;
    }
    for vec in vectors {
        for (c, &v) in centroid.iter_mut().zip(vec.iter()) {
            *c += v;
        }
    }
    let n = vectors.len() as f32;
    for c in &mut centroid {
        *c /= n;
    }
    // L2 normalize
    let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for c in &mut centroid {
            *c /= norm;
        }
    }
    centroid
}

/// Computes cosine similarity between two vectors.
fn cosine_similarity(a: &[f32; SLOT_DIM], b: &[f32; SLOT_DIM]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Generates the next available strand ID.
fn next_strand_id(store: &VoltStore) -> u64 {
    let existing = store.list_strands();
    existing.iter().max().copied().unwrap_or(0) + 1
}

/// Checks all strands for graduation candidates and promotes them.
///
/// Analyzes frame clusters within each strand. When a cluster of
/// `min_cluster_frames` or more frames is internally cohesive but
/// dissimilar to the strand's overall centroid, those frames are
/// migrated to a newly created strand.
///
/// # Errors
///
/// Returns [`VoltError::StorageError`] or [`VoltError::LearnError`]
/// if cluster analysis or frame migration fails.
///
/// # Example
///
/// ```
/// use volt_learn::graduation::{check_graduation, GraduationConfig};
/// use volt_db::VoltStore;
///
/// let mut store = VoltStore::new();
/// let config = GraduationConfig::default();
/// let result = check_graduation(&mut store, &[], &config).unwrap();
/// assert!(result.new_strands_created.is_empty());
/// ```
pub fn check_graduation(
    store: &mut VoltStore,
    events: &[LearningEvent],
    config: &GraduationConfig,
) -> Result<GraduationResult, VoltError> {
    if events.is_empty() {
        return Ok(GraduationResult {
            new_strands_created: Vec::new(),
            frames_migrated: 0,
        });
    }

    // Group events by strand
    let mut strand_events: HashMap<u64, Vec<&LearningEvent>> = HashMap::new();
    for event in events {
        strand_events.entry(event.strand_id).or_default().push(event);
    }

    let mut new_strands = Vec::new();
    let mut total_migrated = 0;

    for strand_evts in strand_events.values() {
        if new_strands.len() >= config.max_new_strands_per_cycle {
            break;
        }

        // Collect unique frame IDs and their gists
        let mut frame_gists: Vec<(u64, [f32; SLOT_DIM])> = Vec::new();

        for event in strand_evts {
            // Check if we already have this frame
            if frame_gists.iter().any(|(id, _)| *id == event.frame_id) {
                continue;
            }

            if let Some(frame) = store.get_by_id(event.frame_id) {
                // Extract R₀ gist: average of active slot R₀ vectors
                let mut gist = [0.0f32; SLOT_DIM];
                let mut count = 0;
                for slot_opt in &frame.slots {
                    if let Some(slot) = slot_opt
                        && let Some(r0) = &slot.resolutions[0]
                    {
                        for (g, &v) in gist.iter_mut().zip(r0.iter()) {
                            *g += v;
                        }
                        count += 1;
                    }
                }
                if count > 0 {
                    for g in &mut gist {
                        *g /= count as f32;
                    }
                    let norm: f32 =
                        gist.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 1e-10 {
                        for g in &mut gist {
                            *g /= norm;
                        }
                    }
                    frame_gists.push((event.frame_id, gist));
                }
            }
        }

        if frame_gists.len() < config.min_cluster_frames {
            continue;
        }

        // Compute strand centroid from all gists
        let all_vectors: Vec<[f32; SLOT_DIM]> =
            frame_gists.iter().map(|(_, g)| *g).collect();
        let strand_centroid = compute_centroid(&all_vectors);

        // Find clusters via simple greedy approach:
        // Group frames that are similar to each other but dissimilar
        // to the overall strand centroid.
        let mut candidate_cluster: Vec<(u64, [f32; SLOT_DIM])> = Vec::new();

        for &(frame_id, ref gist) in &frame_gists {
            let sim_to_centroid =
                cosine_similarity(gist, &strand_centroid);

            // If this frame is not well-represented by the strand centroid,
            // it might belong to a novel sub-topic
            if sim_to_centroid < config.internal_similarity_threshold {
                // Check if it's similar to the existing candidate cluster
                if candidate_cluster.is_empty() {
                    candidate_cluster.push((frame_id, *gist));
                } else {
                    let cluster_centroid: Vec<[f32; SLOT_DIM]> =
                        candidate_cluster.iter().map(|(_, g)| *g).collect();
                    let cc = compute_centroid(&cluster_centroid);
                    let sim_to_cluster = cosine_similarity(gist, &cc);
                    if sim_to_cluster >= config.internal_similarity_threshold {
                        candidate_cluster.push((frame_id, *gist));
                    }
                }
            }
        }

        // Graduate if the cluster is large enough
        if candidate_cluster.len() >= config.min_cluster_frames
            && new_strands.len() < config.max_new_strands_per_cycle
        {
            let new_id = next_strand_id(store) + new_strands.len() as u64;
            store.create_strand(new_id).map_err(|_| {
                // Strand might already exist if IDs collide
                VoltError::LearnError {
                    message: format!(
                        "graduation: failed to create strand {new_id}"
                    ),
                }
            })?;

            let mut migrated = 0;
            for (frame_id, _) in &candidate_cluster {
                match store.reassign_frame_strand(*frame_id, new_id) {
                    Ok(()) => migrated += 1,
                    Err(_) => {
                        // Frame might be in T0 or already moved — skip
                        continue;
                    }
                }
            }

            if migrated > 0 {
                new_strands.push(new_id);
                total_migrated += migrated;
            }
        }
    }

    Ok(GraduationResult {
        new_strands_created: new_strands,
        frames_migrated: total_migrated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_core::meta::DiscourseType;
    use volt_core::MAX_SLOTS;

    fn make_event(frame_id: u64, strand_id: u64) -> LearningEvent {
        LearningEvent {
            frame_id,
            strand_id,
            query_type: DiscourseType::Query,
            gamma_scores: [0.8; MAX_SLOTS],
            convergence_iterations: 5,
            ghost_activations: 0,
            timestamp: frame_id * 1000,
        }
    }

    #[test]
    fn default_config_sensible() {
        let config = GraduationConfig::default();
        assert_eq!(config.min_cluster_frames, 50);
        assert!(config.internal_similarity_threshold > 0.0);
        assert!(config.max_new_strands_per_cycle > 0);
    }

    #[test]
    fn empty_events_no_graduation() {
        let mut store = VoltStore::new();
        let config = GraduationConfig::default();
        let result = check_graduation(&mut store, &[], &config).unwrap();
        assert!(result.new_strands_created.is_empty());
        assert_eq!(result.frames_migrated, 0);
    }

    #[test]
    fn few_events_no_graduation() {
        let mut store = VoltStore::new();
        let config = GraduationConfig::default();
        let events: Vec<LearningEvent> =
            (1..10).map(|i| make_event(i, 0)).collect();
        let result =
            check_graduation(&mut store, &events, &config).unwrap();
        assert!(result.new_strands_created.is_empty());
    }

    #[test]
    fn centroid_of_single_vector() {
        let v = [1.0f32; SLOT_DIM];
        let centroid = compute_centroid(&[v]);
        // Should be L2 normalized version of [1.0; SLOT_DIM]
        let expected_norm = (SLOT_DIM as f32).sqrt();
        let expected_val = 1.0 / expected_norm;
        assert!(
            (centroid[0] - expected_val).abs() < 1e-5,
            "expected {expected_val}, got {}",
            centroid[0]
        );
    }

    #[test]
    fn centroid_of_empty() {
        let centroid = compute_centroid(&[]);
        assert!(centroid.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn cosine_similarity_identical() {
        let mut v = [0.0f32; SLOT_DIM];
        v[0] = 1.0;
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let mut a = [0.0f32; SLOT_DIM];
        let mut b = [0.0f32; SLOT_DIM];
        a[0] = 1.0;
        b[1] = 1.0;
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = [0.0f32; SLOT_DIM];
        let mut b = [0.0f32; SLOT_DIM];
        b[0] = 1.0;
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn max_strands_per_cycle_respected() {
        let config = GraduationConfig {
            max_new_strands_per_cycle: 0,
            ..GraduationConfig::default()
        };
        let mut store = VoltStore::new();
        let events: Vec<LearningEvent> =
            (1..100).map(|i| make_event(i, 0)).collect();
        let result =
            check_graduation(&mut store, &events, &config).unwrap();
        assert!(result.new_strands_created.is_empty());
    }
}
