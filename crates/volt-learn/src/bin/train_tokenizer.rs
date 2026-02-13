//! BPE tokenizer training CLI for Volt X code training.
//!
//! Trains a Byte-Pair Encoding tokenizer on The Stack Python corpus
//! and saves it as a JSON file for use with the code encoder/decoder.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p volt-learn --features code-training --bin train-tokenizer -- \
//!   --corpus "D:\VoltData\phase0\the_stack_sample" \
//!   --vocab-size 32768 \
//!   --output "checkpoints/code_tokenizer.json" \
//!   --max-files 50000
//! ```

use std::path::{Path, PathBuf};
use tokenizers::models::bpe::BPE;
use tokenizers::models::bpe::trainer::BpeTrainer;
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config = parse_args(&args);

    eprintln!("=== Volt X BPE Tokenizer Training ===");
    eprintln!("Corpus:     {}", config.corpus.display());
    eprintln!("Vocab size: {}", config.vocab_size);
    eprintln!("Max files:  {}", config.max_files);
    eprintln!("Output:     {}", config.output.display());
    eprintln!();

    // Collect text files from corpus
    eprintln!("Scanning corpus for text files...");
    let files = collect_corpus_files(&config.corpus, config.max_files);
    eprintln!("Found {} files to process", files.len());

    if files.is_empty() {
        eprintln!("ERROR: No files found in corpus directory");
        std::process::exit(1);
    }

    // Build tokenizer with BPE model
    let trainer = BpeTrainer::builder()
        .vocab_size(config.vocab_size)
        .min_frequency(2)
        .special_tokens(vec![
            AddedToken::from("<pad>", true),
            AddedToken::from("<unk>", true),
            AddedToken::from("<bos>", true),
            AddedToken::from("<eos>", true),
        ])
        .show_progress(true)
        .build();

    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));

    eprintln!("Training BPE tokenizer...");
    let mut trainer_wrapper = TrainerWrapper::BpeTrainer(trainer);
    tokenizer
        .train_from_files(&mut trainer_wrapper, files.clone())
        .expect("tokenizer training failed");

    eprintln!("Vocabulary size: {}", tokenizer.get_vocab_size(true));

    // Ensure output directory exists
    if let Some(parent) = config.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    tokenizer
        .save(&config.output, true)
        .expect("failed to save tokenizer");

    eprintln!("Tokenizer saved to: {}", config.output.display());

    // Quick validation
    let test_code = "def hello_world():\n    print(\"Hello, world!\")";
    let encoding = tokenizer.encode(test_code, false).unwrap();
    eprintln!(
        "\nValidation: \"{}\" → {} tokens",
        test_code,
        encoding.get_ids().len()
    );
    eprintln!("Token IDs: {:?}", &encoding.get_ids()[..encoding.get_ids().len().min(20)]);

    let decoded = tokenizer.decode(encoding.get_ids(), true).unwrap();
    eprintln!("Decoded:   \"{}\"", decoded);
}

struct TrainConfig {
    corpus: PathBuf,
    vocab_size: usize,
    max_files: usize,
    output: PathBuf,
}

fn parse_args(args: &[String]) -> TrainConfig {
    let mut corpus = PathBuf::from("D:\\VoltData\\phase0\\the_stack_sample");
    let mut vocab_size = 32768usize;
    let mut max_files = 50000usize;
    let mut output = PathBuf::from("checkpoints/code_tokenizer.json");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                i += 1;
                corpus = PathBuf::from(&args[i]);
            }
            "--vocab-size" => {
                i += 1;
                vocab_size = args[i].parse().expect("invalid vocab-size");
            }
            "--max-files" => {
                i += 1;
                max_files = args[i].parse().expect("invalid max-files");
            }
            "--output" => {
                i += 1;
                output = PathBuf::from(&args[i]);
            }
            "--help" | "-h" => {
                eprintln!("Usage: train-tokenizer [OPTIONS]");
                eprintln!("  --corpus <PATH>      Corpus directory (default: D:\\VoltData\\phase0\\the_stack_sample)");
                eprintln!("  --vocab-size <N>     Vocabulary size (default: 32768)");
                eprintln!("  --max-files <N>      Max files to process (default: 50000)");
                eprintln!("  --output <PATH>      Output tokenizer JSON (default: checkpoints/code_tokenizer.json)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    TrainConfig {
        corpus,
        vocab_size,
        max_files,
        output,
    }
}

/// Collect file paths from the corpus directory.
///
/// Recursively walks the directory tree. For JSONL files, extracts code
/// content to temporary text files. For plain text/Python files, uses
/// them directly.
fn collect_corpus_files(corpus_dir: &Path, max_files: usize) -> Vec<String> {
    let mut files = Vec::new();

    // Check if corpus is a single JSONL file
    if corpus_dir.is_file() {
        return extract_code_from_jsonl(corpus_dir, max_files);
    }

    // Recursively walk directory for JSONL or Python files
    walk_dir_recursive(corpus_dir, max_files, &mut files);
    files
}

/// Recursively walk a directory, collecting JSONL/py/txt files.
fn walk_dir_recursive(dir: &Path, max_files: usize, files: &mut Vec<String>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("Cannot read directory {}: {e}", dir.display());
            return;
        }
    };

    for entry in entries.filter_map(|e| e.ok()) {
        if files.len() >= max_files {
            break;
        }
        let path = entry.path();
        if path.is_dir() {
            walk_dir_recursive(&path, max_files, files);
        } else if path.is_file() {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "jsonl" {
                eprintln!("  Found JSONL: {}", path.display());
                let mut extracted = extract_code_from_jsonl(&path, max_files - files.len());
                files.append(&mut extracted);
            } else if ext == "py" || ext == "txt" {
                files.push(path.to_string_lossy().to_string());
            }
        }
    }
}

/// Extract code fields from a JSONL file into temporary text files.
fn extract_code_from_jsonl(path: &Path, max_entries: usize) -> Vec<String> {
    use std::io::{BufRead, BufReader, Write};

    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Cannot open {}: {e}", path.display());
            return Vec::new();
        }
    };

    // Create a temp directory for extracted code
    let temp_dir = std::env::temp_dir().join("volt_tokenizer_train");
    std::fs::create_dir_all(&temp_dir).ok();

    let reader = BufReader::new(file);
    let mut files = Vec::new();
    let mut count = 0;

    for line in reader.lines() {
        if count >= max_entries {
            break;
        }
        let Ok(line) = line else { continue };
        let Ok(obj) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };

        let code = obj
            .get("content")
            .or_else(|| obj.get("code"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if code.len() < 50 || code.len() > 50_000 {
            continue;
        }

        let temp_path = temp_dir.join(format!("code_{count}.txt"));
        if let Ok(mut f) = std::fs::File::create(&temp_path)
            && f.write_all(code.as_bytes()).is_ok()
        {
            files.push(temp_path.to_string_lossy().to_string());
            count += 1;
        }

        if count % 5000 == 0 && count > 0 {
            eprintln!("  Extracted {count} code samples...");
        }
    }

    eprintln!("  Extracted {count} code samples from {}", path.display());
    files
}
