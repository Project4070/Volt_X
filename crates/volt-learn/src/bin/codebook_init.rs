//! CLI binary for codebook initialization from code corpus.
//!
//! Reads The Stack JSONL files, encodes through a translator,
//! runs k-means clustering, and saves an initialized codebook.
//!
//! # Usage
//!
//! ```bash
//! # With learned encoder (recommended — requires code-training feature):
//! cargo run --release -p volt-learn --features code-training --bin codebook-init -- \
//!   --corpus D:/VoltData/phase0/the_stack_sample/python/python_sample.jsonl \
//!   --tokenizer checkpoints/code_tokenizer.json \
//!   --encoder checkpoints/code_encoder.safetensors \
//!   --decoder checkpoints/code_decoder.safetensors \
//!   --max-files 100000
//!
//! # With stub translator (no GPU, no trained model needed):
//! cargo run --release -p volt-learn --bin codebook-init
//! ```

use std::path::PathBuf;

use volt_learn::codebook_init::{CodebookInitConfig, init_codebook_from_corpus};
use volt_learn::kmeans::KMeansConfig;
use volt_translate::{StubTranslator, Translator};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut corpus_path = PathBuf::from("D:/VoltData/phase0/the_stack_sample");
    let mut output_path = PathBuf::from("checkpoints/codebook_code.bin");
    let mut max_files: usize = 1_000_000;
    let mut k: usize = 65_536;
    let mut tokenizer_path: Option<PathBuf> = None;
    let mut encoder_path: Option<PathBuf> = None;
    let mut decoder_path: Option<PathBuf> = None;

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
            "--tokenizer" => {
                i += 1;
                if i < args.len() {
                    tokenizer_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--encoder" => {
                i += 1;
                if i < args.len() {
                    encoder_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--decoder" => {
                i += 1;
                if i < args.len() {
                    decoder_path = Some(PathBuf::from(&args[i]));
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
                eprintln!("  --tokenizer <PATH>   BPE tokenizer JSON (enables learned encoder)");
                eprintln!("  --encoder <PATH>     Encoder safetensors weights");
                eprintln!("  --decoder <PATH>     Decoder safetensors weights");
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

    // Build translator
    let translator: Box<dyn Translator + Send> = build_translator(
        tokenizer_path,
        encoder_path,
        decoder_path,
    );

    eprintln!("=== Volt X Codebook Initialization ===");
    eprintln!("  Corpus:     {}", corpus_path.display());
    eprintln!("  Output:     {}", output_path.display());
    eprintln!("  Max files:  {max_files}");
    eprintln!("  k:          {k}");
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
        .spawn(move || init_codebook_from_corpus(&config, translator.as_ref()))
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

/// Build the appropriate translator based on CLI flags.
///
/// When `--tokenizer`, `--encoder`, and `--decoder` are all provided
/// (and the `code-training` feature is enabled), uses the learned
/// CNN encoder. Otherwise falls back to the hash-based StubTranslator.
fn build_translator(
    tokenizer_path: Option<PathBuf>,
    encoder_path: Option<PathBuf>,
    decoder_path: Option<PathBuf>,
) -> Box<dyn Translator + Send> {
    let use_learned = tokenizer_path.is_some();

    if !use_learned {
        eprintln!("[codebook-init] Using StubTranslator (hash-based)");
        return Box::new(StubTranslator::new());
    }

    // All three paths required for learned translator
    let tok = tokenizer_path.expect("--tokenizer required");
    let enc = encoder_path.unwrap_or_else(|| {
        eprintln!("ERROR: --encoder required when --tokenizer is provided");
        std::process::exit(1);
    });
    let dec = decoder_path.unwrap_or_else(|| {
        eprintln!("ERROR: --decoder required when --tokenizer is provided");
        std::process::exit(1);
    });

    #[cfg(feature = "code-training")]
    {
        eprintln!("[codebook-init] Loading LearnedTranslator...");
        eprintln!("  Tokenizer: {}", tok.display());
        eprintln!("  Encoder:   {}", enc.display());
        eprintln!("  Decoder:   {}", dec.display());
        match volt_translate::learned::LearnedTranslator::load(&tok, &enc, &dec) {
            Ok(t) => {
                eprintln!("[codebook-init] LearnedTranslator loaded successfully");
                Box::new(t)
            }
            Err(e) => {
                eprintln!("ERROR: failed to load learned translator: {e}");
                std::process::exit(1);
            }
        }
    }

    #[cfg(not(feature = "code-training"))]
    {
        let _ = (tok, enc, dec);
        eprintln!("ERROR: --tokenizer requires the code-training feature.");
        eprintln!("Recompile with: cargo run --features code-training --bin codebook-init");
        std::process::exit(1);
    }
}
