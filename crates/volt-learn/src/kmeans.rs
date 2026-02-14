//! Mini-batch k-means clustering with k-means++ initialization.
//!
//! Used for codebook initialization: cluster code embeddings into
//! 65,536 centroids that become the VQ-VAE codebook entries.
//!
//! # Algorithm
//!
//! 1. **K-means++ init**: Select `k` initial centroids with probability
//!    proportional to squared distance from the nearest existing centroid.
//! 2. **Mini-batch update**: For each batch of vectors, assign to nearest
//!    centroid and update with learning rate `eta = 1/count`.
//! 3. **Convergence**: Stop when mean centroid movement falls below
//!    `tolerance` or `max_iterations` is reached.
//!
//! # Example
//!
//! ```no_run
//! use volt_learn::kmeans::{KMeansConfig, mini_batch_kmeans};
//! use volt_core::SLOT_DIM;
//!
//! let vectors: Vec<[f32; SLOT_DIM]> = vec![[0.1; SLOT_DIM]; 1000];
//! let config = KMeansConfig {
//!     k: 10,
//!     batch_size: 256,
//!     max_iterations: 50,
//!     tolerance: 1e-5,
//!     seed: 42,
//! };
//! let result = mini_batch_kmeans(&vectors, &config).unwrap();
//! assert_eq!(result.centroids.len(), 10);
//! ```

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use volt_core::{VoltError, SLOT_DIM};

/// Configuration for mini-batch k-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters (centroids to produce).
    pub k: usize,
    /// Vectors per mini-batch update.
    pub batch_size: usize,
    /// Maximum number of full passes over the data.
    pub max_iterations: usize,
    /// Convergence threshold: stop when mean centroid movement is below this.
    pub tolerance: f32,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 65_536,
            batch_size: 8192,
            max_iterations: 50,
            tolerance: 1e-5,
            seed: 42,
        }
    }
}

/// Result metadata from a k-means run.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// The final centroids.
    pub centroids: Vec<[f32; SLOT_DIM]>,
    /// Number of iterations actually performed.
    pub iterations: usize,
    /// Final mean centroid movement (last iteration).
    pub final_movement: f32,
}

/// Squared L2 distance between two vectors.
fn l2_distance_sq(a: &[f32; SLOT_DIM], b: &[f32; SLOT_DIM]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// L2-normalize a vector in place. Returns the original norm.
fn normalize_in_place(v: &mut [f32; SLOT_DIM]) -> f32 {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

/// Find the index of the nearest centroid to `vector` (brute-force).
fn assign_nearest(vector: &[f32; SLOT_DIM], centroids: &[[f32; SLOT_DIM]]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    for (i, centroid) in centroids.iter().enumerate() {
        let dist = l2_distance_sq(vector, centroid);
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    best_idx
}

/// K-means++ initialization: select `k` centroids from `vectors`.
///
/// Uses a subsample of `vectors` if the set is very large (>200K)
/// to keep initialization tractable at high k values.
fn kmeans_plus_plus_init(
    vectors: &[[f32; SLOT_DIM]],
    k: usize,
    rng: &mut StdRng,
) -> Vec<[f32; SLOT_DIM]> {
    // For large datasets, subsample to keep init fast
    // (k-means++ is O(n*k) per centroid, totaling O(n*k^2))
    let max_init_samples = 200_000;
    let init_vectors: Vec<[f32; SLOT_DIM]> = if vectors.len() > max_init_samples {
        // Reservoir-sample max_init_samples vectors
        let mut sample = vectors[..max_init_samples].to_vec();
        for (i, vec) in vectors.iter().enumerate().skip(max_init_samples) {
            let j = rng.random_range(0..=i);
            if j < max_init_samples {
                sample[j] = *vec;
            }
        }
        sample
    } else {
        vectors.to_vec()
    };

    let n = init_vectors.len();
    let mut centroids = Vec::with_capacity(k);

    // Pick first centroid uniformly at random
    let first_idx = rng.random_range(0..n);
    centroids.push(init_vectors[first_idx]);

    // min_dist[i] = min distance from init_vectors[i] to any existing centroid
    let mut min_dist: Vec<f32> = vec![f32::MAX; n];

    for c in 1..k {
        // Parallel: update min_dist with the last-added centroid
        let last_centroid = &centroids[c - 1];
        min_dist
            .par_iter_mut()
            .zip(init_vectors.par_iter())
            .for_each(|(d, vec)| {
                let dist = l2_distance_sq(vec, last_centroid);
                if dist < *d {
                    *d = dist;
                }
            });

        // Sequential: compute total weight (preserves deterministic centroid selection)
        let total_weight: f64 = min_dist.iter().map(|&d| d as f64).sum();

        // Weighted random selection
        if total_weight < 1e-30 {
            // All remaining vectors are at distance 0 from centroids;
            // just pick randomly to fill remaining slots
            let idx = rng.random_range(0..n);
            centroids.push(init_vectors[idx]);
        } else {
            let threshold = rng.random_range(0.0..total_weight);
            let mut cumulative: f64 = 0.0;
            let mut chosen = 0;
            for (i, &d) in min_dist.iter().enumerate() {
                cumulative += d as f64;
                if cumulative >= threshold {
                    chosen = i;
                    break;
                }
            }
            centroids.push(init_vectors[chosen]);
        }

        // Progress for large k
        if k >= 1000 && c % 5000 == 0 {
            eprintln!("[kmeans++] Initialized {c}/{k} centroids");
        }
    }

    centroids
}

/// Run mini-batch k-means clustering.
///
/// Clusters `vectors` into `config.k` groups and returns L2-normalized
/// centroids suitable for [`volt_bus::codebook::Codebook::from_entries`].
///
/// # Errors
///
/// - Empty `vectors` input
/// - Fewer unique vectors than `config.k`
/// - Centroids diverge (NaN/Inf detected)
///
/// # Example
///
/// ```no_run
/// use volt_learn::kmeans::{KMeansConfig, mini_batch_kmeans};
/// use volt_core::SLOT_DIM;
///
/// let vectors: Vec<[f32; SLOT_DIM]> = vec![[0.1; SLOT_DIM]; 1000];
/// let config = KMeansConfig { k: 10, ..Default::default() };
/// let result = mini_batch_kmeans(&vectors, &config).unwrap();
/// assert_eq!(result.centroids.len(), 10);
/// ```
pub fn mini_batch_kmeans(
    vectors: &[[f32; SLOT_DIM]],
    config: &KMeansConfig,
) -> Result<KMeansResult, VoltError> {
    if vectors.is_empty() {
        return Err(VoltError::LearnError {
            message: "k-means: input vectors are empty".into(),
        });
    }
    if vectors.len() < config.k {
        return Err(VoltError::LearnError {
            message: format!(
                "k-means: need at least {} vectors for k={}, got {}",
                config.k,
                config.k,
                vectors.len()
            ),
        });
    }

    let mut rng = StdRng::seed_from_u64(config.seed);

    // Step 1: K-means++ initialization
    eprintln!(
        "[kmeans] Initializing {} centroids via k-means++ on {} vectors",
        config.k,
        vectors.len()
    );
    let mut centroids = kmeans_plus_plus_init(vectors, config.k, &mut rng);
    eprintln!("[kmeans] Initialization complete");

    // Normalize initial centroids
    for c in centroids.iter_mut() {
        normalize_in_place(c);
    }

    // Per-centroid assignment count (for learning rate)
    let mut counts: Vec<u64> = vec![1; config.k]; // start at 1 to avoid div-by-zero

    let n = vectors.len();
    let mut final_movement = f32::MAX;
    let mut iteration = 0;

    // Step 2: Mini-batch iterations
    for iter in 0..config.max_iterations {
        iteration = iter + 1;

        // Save old centroids for convergence check
        let old_centroids = centroids.clone();

        // Generate shuffled indices for this epoch
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            indices.swap(i, j);
        }

        // Process in batches
        for batch_start in (0..n).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(n);
            let batch_indices = &indices[batch_start..batch_end];

            // Parallel: assign each vector to its nearest centroid
            let assignments: Vec<usize> = batch_indices
                .par_iter()
                .map(|&idx| assign_nearest(&vectors[idx], &centroids))
                .collect();

            // Sequential: update centroids (order-dependent due to learning rate)
            for (batch_pos, &nearest) in assignments.iter().enumerate() {
                let idx = batch_indices[batch_pos];
                counts[nearest] += 1;
                let eta = 1.0 / counts[nearest] as f32;
                let centroid = &mut centroids[nearest];
                for (d, c) in centroid.iter_mut().enumerate() {
                    *c = (1.0 - eta) * *c + eta * vectors[idx][d];
                }
            }
        }

        // Normalize centroids after each epoch
        for c in centroids.iter_mut() {
            normalize_in_place(c);
        }

        // Parallel: check for NaN/Inf
        let bad_centroid = centroids
            .par_iter()
            .enumerate()
            .find_any(|(_, c)| c.iter().any(|x| !x.is_finite()));
        if let Some((i, _)) = bad_centroid {
            return Err(VoltError::LearnError {
                message: format!(
                    "k-means diverged at iteration {iteration}: centroid {i} contains NaN/Inf"
                ),
            });
        }

        // Parallel: convergence check — mean centroid movement
        let total_movement: f64 = centroids
            .par_iter()
            .zip(old_centroids.par_iter())
            .map(|(new, old)| l2_distance_sq(new, old) as f64)
            .sum();
        final_movement = (total_movement / config.k as f64).sqrt() as f32;

        eprintln!(
            "[kmeans] Iteration {iteration}/{}, mean centroid movement: {final_movement:.8}",
            config.max_iterations
        );

        if final_movement < config.tolerance {
            eprintln!("[kmeans] Converged at iteration {iteration}");
            break;
        }
    }

    Ok(KMeansResult {
        centroids,
        iterations: iteration,
        final_movement,
    })
}

/// Compute mean L2 distance from each vector to its nearest centroid.
///
/// This measures quantization error: how well the centroids represent
/// the original vector distribution.
pub fn mean_quantization_error(
    vectors: &[[f32; SLOT_DIM]],
    centroids: &[[f32; SLOT_DIM]],
) -> f32 {
    if vectors.is_empty() || centroids.is_empty() {
        return 0.0;
    }
    let total: f64 = vectors
        .par_iter()
        .map(|v| {
            let nearest = assign_nearest(v, centroids);
            l2_distance_sq(v, &centroids[nearest]).sqrt() as f64
        })
        .sum();
    (total / vectors.len() as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a vector with a known pattern seeded by `id`.
    fn test_vector(id: u64) -> [f32; SLOT_DIM] {
        let mut rng = StdRng::seed_from_u64(id);
        let mut v = [0.0f32; SLOT_DIM];
        for x in v.iter_mut() {
            *x = rng.random_range(-1.0..1.0);
        }
        normalize_in_place(&mut v);
        v
    }

    /// Generate vectors clustered around `n_clusters` centers.
    fn clustered_vectors(
        n_clusters: usize,
        per_cluster: usize,
        spread: f32,
        seed: u64,
    ) -> Vec<[f32; SLOT_DIM]> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut vectors = Vec::with_capacity(n_clusters * per_cluster);

        // Generate cluster centers
        let centers: Vec<[f32; SLOT_DIM]> = (0..n_clusters)
            .map(|i| test_vector(seed + i as u64 * 1000))
            .collect();

        for center in &centers {
            for _ in 0..per_cluster {
                let mut v = *center;
                for x in v.iter_mut() {
                    *x += rng.random_range(-spread..spread);
                }
                normalize_in_place(&mut v);
                vectors.push(v);
            }
        }
        vectors
    }

    #[test]
    fn kmeans_known_clusters() {
        // 10 well-separated clusters, 100 points each
        let vectors = clustered_vectors(10, 100, 0.05, 12345);
        let config = KMeansConfig {
            k: 10,
            batch_size: 256,
            max_iterations: 30,
            tolerance: 1e-6,
            seed: 42,
        };
        let result = mini_batch_kmeans(&vectors, &config).unwrap();
        assert_eq!(result.centroids.len(), 10);

        // Each centroid should be close to a cluster center
        let error = mean_quantization_error(&vectors, &result.centroids);
        // In 256-dim space, even tight clusters (spread=0.05) have substantial
        // L2 distance from centroids due to normalization on the unit hypersphere
        assert!(
            error < 0.8,
            "quantization error {error:.4} should be < 0.8 for tight clusters"
        );
    }

    #[test]
    fn kmeans_single_cluster() {
        let vectors: Vec<[f32; SLOT_DIM]> = (0..100).map(test_vector).collect();
        let config = KMeansConfig {
            k: 1,
            batch_size: 64,
            max_iterations: 10,
            tolerance: 1e-6,
            seed: 42,
        };
        let result = mini_batch_kmeans(&vectors, &config).unwrap();
        assert_eq!(result.centroids.len(), 1);
    }

    #[test]
    fn kmeans_k_equals_n() {
        let vectors: Vec<[f32; SLOT_DIM]> = (0..10).map(test_vector).collect();
        let config = KMeansConfig {
            k: 10,
            batch_size: 10,
            max_iterations: 5,
            tolerance: 1e-6,
            seed: 42,
        };
        let result = mini_batch_kmeans(&vectors, &config).unwrap();
        assert_eq!(result.centroids.len(), 10);
        // With k=n, quantization error should be very small
        let error = mean_quantization_error(&vectors, &result.centroids);
        assert!(error < 0.5, "error {error:.4} should be small for k=n");
    }

    #[test]
    fn kmeans_empty_vectors_errors() {
        let vectors: Vec<[f32; SLOT_DIM]> = vec![];
        let config = KMeansConfig {
            k: 10,
            ..Default::default()
        };
        assert!(mini_batch_kmeans(&vectors, &config).is_err());
    }

    #[test]
    fn kmeans_too_few_vectors_errors() {
        let vectors: Vec<[f32; SLOT_DIM]> = vec![test_vector(0); 5];
        let config = KMeansConfig {
            k: 10,
            ..Default::default()
        };
        assert!(mini_batch_kmeans(&vectors, &config).is_err());
    }

    #[test]
    fn kmeans_deterministic() {
        let vectors = clustered_vectors(5, 50, 0.1, 99);
        let config = KMeansConfig {
            k: 5,
            batch_size: 64,
            max_iterations: 10,
            tolerance: 1e-6,
            seed: 42,
        };
        let result1 = mini_batch_kmeans(&vectors, &config).unwrap();
        let result2 = mini_batch_kmeans(&vectors, &config).unwrap();
        assert_eq!(result1.iterations, result2.iterations);
        for (c1, c2) in result1.centroids.iter().zip(result2.centroids.iter()) {
            for (a, b) in c1.iter().zip(c2.iter()) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "centroids differ: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn kmeans_centroids_are_normalized() {
        let vectors = clustered_vectors(5, 100, 0.1, 77);
        let config = KMeansConfig {
            k: 5,
            batch_size: 128,
            max_iterations: 10,
            tolerance: 1e-6,
            seed: 42,
        };
        let result = mini_batch_kmeans(&vectors, &config).unwrap();
        for (i, c) in result.centroids.iter().enumerate() {
            let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "centroid {i} norm = {norm}, expected ~1.0"
            );
        }
    }

    #[test]
    fn mean_quantization_error_zero_for_identical() {
        let v = test_vector(0);
        let vectors = vec![v; 100];
        let centroids = vec![v];
        let error = mean_quantization_error(&vectors, &centroids);
        assert!(error < 1e-6, "error should be ~0 for identical vectors");
    }

    #[test]
    fn l2_distance_sq_correctness() {
        let a = [1.0f32; SLOT_DIM];
        let b = [0.0f32; SLOT_DIM];
        let dist = l2_distance_sq(&a, &b);
        assert!((dist - SLOT_DIM as f32).abs() < 1e-4);
    }
}
