//! Codebook initialization from code corpus.
//!
//! Implements the Phase 0.3 pipeline: encode code files through
//! [`StubTranslator`], collect slot embeddings, run k-means clustering,
//! and produce an initialized [`Codebook`] with centroids covering
//! code's embedding space.
//!
//! # Pipeline
//!
//! ```text
//! The Stack JSONL → StackCorpusReader → Translator.encode()
//!     → extract R0 slot vectors → subsample → mini-batch k-means
//!     → Codebook::from_entries(centroids) → save()
//! ```
//!
//! # Example
//!
//! ```no_run
//! use std::path::PathBuf;
//! use volt_learn::codebook_init::{CodebookInitConfig, init_codebook_from_corpus};
//! use volt_learn::kmeans::KMeansConfig;
//! use volt_translate::StubTranslator;
//!
//! let config = CodebookInitConfig {
//!     corpus_path: PathBuf::from("D:/VoltData/phase0/the_stack_sample"),
//!     max_files: 100_000,
//!     kmeans_sample_size: 500_000,
//!     kmeans_config: KMeansConfig { k: 65_536, ..Default::default() },
//!     output_path: PathBuf::from("checkpoints/codebook_code.bin"),
//!     log_interval: 10_000,
//! };
//! let translator = StubTranslator::new();
//! let result = init_codebook_from_corpus(&config, &translator).unwrap();
//! println!("Mean quantization error: {:.4}", result.mean_quantization_error);
//! ```

use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use volt_bus::codebook::Codebook;
use volt_core::{TensorFrame, VoltError, MAX_SLOTS, SLOT_DIM};
use volt_translate::Translator;

use crate::kmeans::{self, KMeansConfig};
use crate::stack_corpus::StackCorpusReader;

/// Configuration for codebook initialization from a code corpus.
#[derive(Debug, Clone)]
pub struct CodebookInitConfig {
    /// Path to a JSONL file or directory of JSONL files.
    pub corpus_path: PathBuf,
    /// Maximum number of code files to process (0 = unlimited).
    pub max_files: usize,
    /// Number of vectors to subsample for k-means (0 = use all).
    pub kmeans_sample_size: usize,
    /// K-means clustering configuration.
    pub kmeans_config: KMeansConfig,
    /// Output path for the codebook binary file.
    pub output_path: PathBuf,
    /// Print progress every N files (0 = no progress output).
    pub log_interval: usize,
}

impl Default for CodebookInitConfig {
    fn default() -> Self {
        Self {
            corpus_path: PathBuf::from("D:/VoltData/phase0/the_stack_sample"),
            max_files: 1_000_000,
            kmeans_sample_size: 2_000_000,
            kmeans_config: KMeansConfig::default(),
            output_path: PathBuf::from("checkpoints/codebook_code.bin"),
            log_interval: 10_000,
        }
    }
}

/// Result from a codebook initialization run.
#[derive(Debug, Clone)]
pub struct CodebookInitResult {
    /// Number of code files successfully processed.
    pub files_processed: usize,
    /// Number of files skipped due to encoding errors.
    pub files_skipped: usize,
    /// Total vectors collected (before subsampling).
    pub vectors_collected: usize,
    /// Vectors actually used for k-means.
    pub vectors_used_for_kmeans: usize,
    /// Number of k-means iterations performed.
    pub kmeans_iterations: usize,
    /// Mean L2 distance from vectors to nearest centroid.
    pub mean_quantization_error: f32,
    /// Path where the codebook was saved.
    pub codebook_path: PathBuf,
}

/// Extract non-zero R0 slot vectors from a TensorFrame.
///
/// Pulls the resolution-0 (discourse level) embedding from each
/// filled slot, skipping zero/near-zero vectors.
///
/// # Example
///
/// ```
/// use volt_core::TensorFrame;
/// use volt_learn::codebook_init::extract_slot_vectors;
///
/// let frame = TensorFrame::new();
/// let vectors = extract_slot_vectors(&frame);
/// assert!(vectors.is_empty()); // empty frame has no slot vectors
/// ```
pub fn extract_slot_vectors(frame: &TensorFrame) -> Vec<[f32; SLOT_DIM]> {
    let mut vectors = Vec::new();
    for slot in frame.slots.iter().take(MAX_SLOTS).flatten() {
        if let Some(vec) = &slot.resolutions[0] {
            let norm_sq: f32 = vec.iter().map(|x| x * x).sum();
            if norm_sq > 1e-10 {
                vectors.push(*vec);
            }
        }
    }
    vectors
}

/// Subsample `target_size` vectors from `vectors` using Fisher-Yates partial shuffle.
fn subsample(
    vectors: &[[f32; SLOT_DIM]],
    target_size: usize,
    seed: u64,
) -> Vec<[f32; SLOT_DIM]> {
    if vectors.len() <= target_size {
        return vectors.to_vec();
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..vectors.len()).collect();
    for i in 0..target_size {
        let j = rng.random_range(i..vectors.len());
        indices.swap(i, j);
    }
    indices[..target_size]
        .iter()
        .map(|&i| vectors[i])
        .collect()
}

/// Open a corpus reader from a path (file or directory).
fn open_corpus(path: &Path) -> Result<StackCorpusReader, VoltError> {
    if path.is_dir() {
        StackCorpusReader::from_directory(path)
    } else {
        StackCorpusReader::from_file(path)
    }
}

/// Run the full codebook initialization pipeline.
///
/// 1. Stream-read code files from The Stack JSONL
/// 2. Encode each through [`StubTranslator`]
/// 3. Collect R0 slot vectors
/// 4. Subsample for k-means
/// 5. Run mini-batch k-means
/// 6. Build and save [`Codebook`]
/// 7. Validate quantization error
///
/// # Errors
///
/// - Corpus path not found or unreadable
/// - No vectors collected (all files failed encoding)
/// - K-means failure (too few vectors, divergence)
/// - Codebook save failure
///
/// # Example
///
/// ```no_run
/// use volt_learn::codebook_init::{CodebookInitConfig, init_codebook_from_corpus};
/// use volt_translate::StubTranslator;
///
/// let config = CodebookInitConfig::default();
/// let translator = StubTranslator::new();
/// let result = init_codebook_from_corpus(&config, &translator).unwrap();
/// assert!(result.mean_quantization_error < 0.1);
/// ```
pub fn init_codebook_from_corpus(
    config: &CodebookInitConfig,
    translator: &dyn Translator,
) -> Result<CodebookInitResult, VoltError> {
    // Step 1: Open corpus
    eprintln!(
        "[codebook-init] Reading corpus from {}",
        config.corpus_path.display()
    );
    let reader = open_corpus(&config.corpus_path)?;

    // Step 2: Encode files and collect vectors
    let mut all_vectors: Vec<[f32; SLOT_DIM]> = Vec::new();
    let mut files_processed: usize = 0;
    let mut files_skipped: usize = 0;

    for entry_result in reader {
        // Check file limit
        if config.max_files > 0 && files_processed >= config.max_files {
            break;
        }

        let entry = match entry_result {
            Ok(e) => e,
            Err(_) => {
                files_skipped += 1;
                continue;
            }
        };

        // Skip very short or very long files
        if entry.content.len() < 50 || entry.content.len() > 50_000 {
            files_skipped += 1;
            continue;
        }

        // Encode through StubTranslator
        match translator.encode(&entry.content) {
            Ok(output) => {
                let slot_vecs = extract_slot_vectors(&output.frame);
                all_vectors.extend_from_slice(&slot_vecs);
                files_processed += 1;
            }
            Err(_) => {
                files_skipped += 1;
                continue;
            }
        }

        // Progress
        if config.log_interval > 0 && files_processed.is_multiple_of(config.log_interval) {
            eprintln!(
                "[codebook-init] Encoded {files_processed} files ({} vectors, skipped {files_skipped})",
                all_vectors.len()
            );
        }
    }

    let vectors_collected = all_vectors.len();
    eprintln!(
        "[codebook-init] Encoding complete: {files_processed} files, \
         {vectors_collected} vectors, {files_skipped} skipped"
    );

    if vectors_collected < config.kmeans_config.k {
        return Err(VoltError::LearnError {
            message: format!(
                "collected only {vectors_collected} vectors, need at least {} for k-means (k={})",
                config.kmeans_config.k, config.kmeans_config.k
            ),
        });
    }

    // Step 3: Subsample for k-means
    let kmeans_vectors = if config.kmeans_sample_size > 0 {
        let sample = subsample(&all_vectors, config.kmeans_sample_size, config.kmeans_config.seed);
        eprintln!(
            "[codebook-init] Subsampled {} vectors for k-means (from {vectors_collected})",
            sample.len()
        );
        sample
    } else {
        eprintln!("[codebook-init] Using all {vectors_collected} vectors for k-means");
        all_vectors.clone()
    };
    let vectors_used_for_kmeans = kmeans_vectors.len();

    // Step 4: Run k-means
    eprintln!(
        "[codebook-init] Running k-means (k={}, batch_size={}, max_iter={})",
        config.kmeans_config.k, config.kmeans_config.batch_size, config.kmeans_config.max_iterations
    );
    let kmeans_result = kmeans::mini_batch_kmeans(&kmeans_vectors, &config.kmeans_config)?;
    eprintln!(
        "[codebook-init] K-means finished in {} iterations (movement: {:.8})",
        kmeans_result.iterations, kmeans_result.final_movement
    );

    // Step 5: Build codebook
    eprintln!(
        "[codebook-init] Building codebook ({} entries)",
        kmeans_result.centroids.len()
    );
    let codebook = Codebook::from_entries(kmeans_result.centroids)?;

    // Step 6: Save codebook
    if let Some(parent) = config.output_path.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent).map_err(|e| VoltError::LearnError {
            message: format!(
                "failed to create output directory {}: {e}",
                parent.display()
            ),
        })?;
    }
    codebook.save(&config.output_path)?;
    eprintln!(
        "[codebook-init] Saved codebook to {}",
        config.output_path.display()
    );

    // Step 7: Validate quantization error on all collected vectors
    eprintln!(
        "[codebook-init] Validating quantization error on {vectors_collected} vectors"
    );
    // Reload codebook to verify save/load roundtrip, then use lookup for validation.
    // We use the k-means centroids directly since Codebook.entries is private.
    let loaded_codebook = Codebook::load(&config.output_path)?;
    let mut total_error: f64 = 0.0;
    for v in &all_vectors {
        let (_, quantized) = loaded_codebook.quantize(v).unwrap_or((0, *v));
        let dist: f32 = v
            .iter()
            .zip(quantized.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            .sqrt();
        total_error += dist as f64;
    }
    let mean_error = (total_error / all_vectors.len().max(1) as f64) as f32;

    eprintln!("[codebook-init] Mean L2 quantization error: {mean_error:.6}");

    Ok(CodebookInitResult {
        files_processed,
        files_skipped,
        vectors_collected,
        vectors_used_for_kmeans,
        kmeans_iterations: kmeans_result.iterations,
        mean_quantization_error: mean_error,
        codebook_path: config.output_path.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use volt_translate::StubTranslator;

    #[test]
    fn extract_slot_vectors_empty_frame() {
        let frame = TensorFrame::new();
        let vectors = extract_slot_vectors(&frame);
        assert!(vectors.is_empty());
    }

    #[test]
    fn extract_slot_vectors_filled_frame() {
        let translator = StubTranslator::new();
        let output = translator
            .encode("the quick brown fox jumps over the lazy dog")
            .unwrap();
        let vectors = extract_slot_vectors(&output.frame);
        // Should have vectors for filled slots
        assert!(
            !vectors.is_empty(),
            "should extract vectors from filled slots"
        );
        assert!(
            vectors.len() <= MAX_SLOTS,
            "can't have more than MAX_SLOTS vectors"
        );
        // All vectors should be non-zero and finite
        for v in &vectors {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(norm > 0.1, "vector should be non-zero");
            assert!(v.iter().all(|x| x.is_finite()), "vector should be finite");
        }
    }

    #[test]
    fn subsample_smaller_than_target() {
        let vectors: Vec<[f32; SLOT_DIM]> = vec![[1.0; SLOT_DIM]; 10];
        let result = subsample(&vectors, 100, 42);
        assert_eq!(result.len(), 10); // returns all if smaller than target
    }

    #[test]
    fn subsample_exact_size() {
        let vectors: Vec<[f32; SLOT_DIM]> = vec![[1.0; SLOT_DIM]; 100];
        let result = subsample(&vectors, 50, 42);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn subsample_deterministic() {
        let vectors: Vec<[f32; SLOT_DIM]> = (0..100)
            .map(|i| {
                let mut v = [0.0f32; SLOT_DIM];
                v[0] = i as f32;
                v
            })
            .collect();
        let s1 = subsample(&vectors, 30, 42);
        let s2 = subsample(&vectors, 30, 42);
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a[0], b[0]);
        }
    }
}
