//! CLI binary for codebook initialization from code corpus.
//!
//! Reads The Stack JSONL files, encodes through StubTranslator,
//! runs k-means clustering, and saves an initialized codebook.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin codebook-init
//! cargo run --bin codebook-init -- --corpus D:/VoltData/phase0/the_stack_sample
//! ```

use std::path::PathBuf;

use volt_learn::codebook_init::{CodebookInitConfig, init_codebook_from_corpus};
use volt_learn::kmeans::KMeansConfig;

fn main() {
    // Parse simple CLI args (--corpus, --output, --max-files, --k)
    let args: Vec<String> = std::env::args().collect();

    let mut corpus_path = PathBuf::from("D:/VoltData/phase0/the_stack_sample");
    let mut output_path = PathBuf::from("checkpoints/codebook_code.bin");
    let mut max_files: usize = 1_000_000;
    let mut k: usize = 65_536;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                i += 1;
                if i < args.len() {
                    corpus_path = PathBuf::from(&args[i]);
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output_path = PathBuf::from(&args[i]);
                }
            }
            "--max-files" => {
                i += 1;
                if i < args.len() {
                    max_files = args[i].parse().unwrap_or(max_files);
                }
            }
            "--k" => {
                i += 1;
                if i < args.len() {
                    k = args[i].parse().unwrap_or(k);
                }
            }
            "--help" | "-h" => {
                eprintln!("Usage: codebook-init [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --corpus <PATH>      Path to JSONL file or directory (default: D:/VoltData/phase0/the_stack_sample)");
                eprintln!("  --output <PATH>      Output codebook path (default: checkpoints/codebook_code.bin)");
                eprintln!("  --max-files <N>      Max code files to process (default: 1000000)");
                eprintln!("  --k <N>              Number of codebook entries (default: 65536)");
                eprintln!("  --help               Show this help");
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}. Use --help for usage.");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!("=== Volt X Codebook Initialization ===");
    eprintln!("  Corpus:    {}", corpus_path.display());
    eprintln!("  Output:    {}", output_path.display());
    eprintln!("  Max files: {max_files}");
    eprintln!("  k:         {k}");
    eprintln!();

    let config = CodebookInitConfig {
        corpus_path,
        max_files,
        kmeans_sample_size: 2_000_000,
        kmeans_config: KMeansConfig {
            k,
            batch_size: 8192,
            max_iterations: 50,
            tolerance: 1e-5,
            seed: 42,
        },
        output_path,
        log_interval: 10_000,
    };

    // Run on a thread with large stack (TensorFrame is ~64KB, Windows default is 1MB)
    let result = std::thread::Builder::new()
        .name("codebook-init".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(move || init_codebook_from_corpus(&config))
        .expect("failed to spawn init thread")
        .join()
        .expect("init thread panicked");

    match result {
        Ok(r) => {
            eprintln!();
            eprintln!("=== Codebook Initialization Complete ===");
            eprintln!("  Files processed:        {}", r.files_processed);
            eprintln!("  Files skipped:          {}", r.files_skipped);
            eprintln!("  Vectors collected:      {}", r.vectors_collected);
            eprintln!("  Vectors used (k-means): {}", r.vectors_used_for_kmeans);
            eprintln!("  K-means iterations:     {}", r.kmeans_iterations);
            eprintln!("  Mean quant. error:      {:.6}", r.mean_quantization_error);
            eprintln!("  Saved to:               {}", r.codebook_path.display());
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    }
}
