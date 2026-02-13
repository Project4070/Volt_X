//! Phase 0.3 integration tests: Codebook Initialization from Code Corpus.
//!
//! All tests use synthetic data (no external files required).
//! Tests run on 8MB stack threads for Windows TensorFrame compatibility.

use std::io::Write;
use std::path::PathBuf;

use volt_learn::codebook_init::{
    extract_slot_vectors, init_codebook_from_corpus, CodebookInitConfig,
};
use volt_learn::kmeans::{mini_batch_kmeans, mean_quantization_error, KMeansConfig};
use volt_learn::stack_corpus::StackCorpusReader;
use volt_translate::Translator;

/// Run a closure on a thread with 8MB stack (Windows TensorFrame safety).
fn with_large_stack<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    std::thread::Builder::new()
        .name("test-large-stack".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(f)
        .expect("failed to spawn test thread")
        .join()
        .expect("test thread panicked")
}

/// Create a temporary JSONL file with synthetic code entries.
fn create_synthetic_corpus(count: usize) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    for i in 0..count {
        let content = format!(
            "def function_{i}(x, y):\n    result = x + y * {i}\n    return result\n\n\
             class Handler_{i}:\n    def process(self, data):\n        return data[{i}]\n"
        );
        let entry = serde_json::json!({
            "content": content,
            "language": "python",
            "path": format!("repo/module_{i}.py"),
            "size": content.len()
        });
        writeln!(f, "{}", serde_json::to_string(&entry).unwrap()).unwrap();
    }
    f.flush().unwrap();
    f
}

#[test]
fn phase_0_3_stack_corpus_reader_basic() {
    let corpus = create_synthetic_corpus(50);
    let reader = StackCorpusReader::from_file(corpus.path()).unwrap();
    let entries: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(entries.len(), 50);
    assert!(entries[0].content.contains("def function_0"));
    assert_eq!(entries[0].language, "python");
}

#[test]
fn phase_0_3_stack_corpus_reader_directory() {
    let dir = tempfile::tempdir().unwrap();

    // Create two JSONL shard files
    for shard in 0..2 {
        let path = dir.path().join(format!("shard_{shard:02}.jsonl"));
        let mut f = std::fs::File::create(&path).unwrap();
        for i in 0..10 {
            let idx = shard * 10 + i;
            let content = format!("def func_{idx}(): pass\n# padding to reach minimum length for the filter threshold of fifty bytes ok");
            let entry = serde_json::json!({
                "content": content,
                "language": "python",
                "path": format!("file_{idx}.py"),
                "size": content.len()
            });
            writeln!(f, "{}", serde_json::to_string(&entry).unwrap()).unwrap();
        }
    }

    let reader = StackCorpusReader::from_directory(dir.path()).unwrap();
    let entries: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(entries.len(), 20);
}

#[test]
fn phase_0_3_extract_slot_vectors_from_encoded_frame() {
    with_large_stack(|| {
        let translator = volt_translate::StubTranslator::new();
        let output = translator
            .encode("def calculate total price quantity discount rate")
            .unwrap();
        let vectors = extract_slot_vectors(&output.frame);

        // StubTranslator fills slots based on word count (7 words = up to 7 slots)
        assert!(!vectors.is_empty(), "should extract at least one vector");
        assert!(vectors.len() <= 16, "cannot exceed MAX_SLOTS");

        // Vectors should be L2-normalized (from word_to_vector)
        for v in &vectors {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "vector norm {norm} should be ~1.0"
            );
        }
    });
}

#[test]
fn phase_0_3_kmeans_quality_on_synthetic_data() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use volt_core::SLOT_DIM;

    // Generate 5000 vectors from 32 clusters
    let mut rng = StdRng::seed_from_u64(12345);
    let n_clusters = 32;
    let per_cluster = 156; // 32 * 156 = 4992

    // Generate cluster centers
    let centers: Vec<[f32; SLOT_DIM]> = (0..n_clusters)
        .map(|_| {
            let mut v = [0.0f32; SLOT_DIM];
            for x in v.iter_mut() {
                *x = rng.random_range(-1.0..1.0);
            }
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= norm;
            }
            v
        })
        .collect();

    // Generate points around each center
    let mut vectors = Vec::with_capacity(n_clusters * per_cluster);
    for center in &centers {
        for _ in 0..per_cluster {
            let mut v = *center;
            for x in v.iter_mut() {
                *x += rng.random_range(-0.05..0.05);
            }
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= norm;
            }
            vectors.push(v);
        }
    }

    let config = KMeansConfig {
        k: 32,
        batch_size: 512,
        max_iterations: 30,
        tolerance: 1e-6,
        seed: 42,
    };

    let result = mini_batch_kmeans(&vectors, &config).unwrap();
    assert_eq!(result.centroids.len(), 32);

    let error = mean_quantization_error(&vectors, &result.centroids);
    // In 256-dim space on the unit hypersphere, even tight clusters
    // (spread=0.05) yield L2 quantization errors around 0.5-0.6
    assert!(
        error < 0.8,
        "mean quantization error {error:.4} should be < 0.8 for tight clusters"
    );
}

#[test]
fn phase_0_3_full_pipeline_synthetic() {
    with_large_stack(|| {
        // Create a small synthetic corpus
        let corpus = create_synthetic_corpus(100);
        let output_dir = tempfile::tempdir().unwrap();
        let output_path = output_dir.path().join("test_codebook.bin");

        let config = CodebookInitConfig {
            corpus_path: PathBuf::from(corpus.path()),
            max_files: 100,
            kmeans_sample_size: 0, // use all
            kmeans_config: KMeansConfig {
                k: 16, // small k for test
                batch_size: 64,
                max_iterations: 20,
                tolerance: 1e-6,
                seed: 42,
            },
            output_path: output_path.clone(),
            log_interval: 50,
        };

        let translator = volt_translate::StubTranslator::new();
        let result = init_codebook_from_corpus(&config, &translator).unwrap();

        // Verify results
        assert!(result.files_processed > 0, "should process some files");
        assert!(result.vectors_collected > 0, "should collect vectors");
        assert!(result.kmeans_iterations > 0, "should run k-means");
        assert!(
            result.mean_quantization_error.is_finite(),
            "quantization error should be finite"
        );
        assert!(
            result.mean_quantization_error >= 0.0,
            "quantization error should be non-negative"
        );

        // Verify codebook file was created
        assert!(output_path.exists(), "codebook file should exist");

        // Verify codebook can be loaded
        let codebook = volt_bus::codebook::Codebook::load(&output_path).unwrap();
        assert_eq!(codebook.len(), 16, "codebook should have 16 entries");
    });
}

#[test]
fn phase_0_3_codebook_roundtrip_after_init() {
    with_large_stack(|| {
        let corpus = create_synthetic_corpus(50);
        let output_dir = tempfile::tempdir().unwrap();
        let output_path = output_dir.path().join("roundtrip_codebook.bin");

        let config = CodebookInitConfig {
            corpus_path: PathBuf::from(corpus.path()),
            max_files: 50,
            kmeans_sample_size: 0,
            kmeans_config: KMeansConfig {
                k: 8,
                batch_size: 32,
                max_iterations: 10,
                tolerance: 1e-6,
                seed: 42,
            },
            output_path: output_path.clone(),
            log_interval: 0,
        };

        let translator = volt_translate::StubTranslator::new();
        let result = init_codebook_from_corpus(&config, &translator).unwrap();

        // Load codebook and verify quantization works
        let codebook = volt_bus::codebook::Codebook::load(&output_path).unwrap();

        // Encode a test string and quantize
        let translator = volt_translate::StubTranslator::new();
        let output = translator
            .encode("import numpy as np from collections import defaultdict")
            .unwrap();
        let vectors = extract_slot_vectors(&output.frame);

        // Every vector should quantize successfully
        for v in &vectors {
            let (id, quantized) = codebook.quantize(v).unwrap();
            assert!(id < 8, "codebook id should be < k=8");
            let norm: f32 = quantized.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "quantized vector should be normalized"
            );
        }

        eprintln!(
            "Roundtrip test: {} files, {} vectors, error={:.4}",
            result.files_processed, result.vectors_collected, result.mean_quantization_error
        );
    });
}
