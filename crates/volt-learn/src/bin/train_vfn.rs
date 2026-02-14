//! Scaled VFN flow matching training CLI for Volt X Phase 2.1.
//!
//! Trains the ScaledVfn (~51M params) on (query, solution) code pairs
//! using time-conditioned flow matching.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p volt-learn --features vfn-training --bin train-vfn -- \
//!   --data "D:\VoltData\humaneval\humaneval_problems.jsonl" \
//!   --data "D:\VoltData\mbpp\mbpp_problems.jsonl" \
//!   --tokenizer "checkpoints\code_tokenizer.json" \
//!   --encoder "checkpoints\code_encoder.safetensors" \
//!   --decoder "checkpoints\code_decoder.safetensors" \
//!   --output "checkpoints\scaled_vfn.safetensors" \
//!   --epochs 10 --batch-size 32 --lr 1e-4 --device cuda
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use candle_nn::VarMap;

use volt_learn::code_pairs::{load_datasets, problems_to_frame_pairs};
use volt_soft::scaled_vfn::{ScaledVfn, ScaledVfnConfig};
use volt_soft::training::scaled_flow_matching::{train_scaled_vfn, ScaledFlowConfig};
use volt_translate::learned::LearnedTranslator;
use volt_translate::Translator;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config = parse_args(&args);

    eprintln!("=== Volt X Scaled VFN Training (Phase 2.1) ===");
    eprintln!("Data files: {}", config.data_paths.len());
    for path in &config.data_paths {
        eprintln!("  - {}", path.display());
    }
    eprintln!("Tokenizer:  {}", config.tokenizer.display());
    eprintln!("Encoder:    {}", config.encoder.display());
    eprintln!("Decoder:    {}", config.decoder.display());
    eprintln!("Output:     {}", config.output.display());
    eprintln!("Epochs:     {}", config.epochs);
    eprintln!("Batch size: {}", config.batch_size);
    eprintln!("LR:         {}", config.lr);
    eprintln!("Warmup:     {} steps", config.warmup_steps);
    eprintln!("Hidden dim: {}", config.hidden_dim);
    eprintln!("Num blocks: {}", config.num_blocks);
    eprintln!("Device:     {}", config.device_name);
    eprintln!();

    // Select device
    let device = select_device(&config.device_name);
    eprintln!("Using device: {:?}", device);

    // Load datasets
    eprintln!("Loading datasets...");
    let dataset = load_datasets(&config.data_paths).unwrap_or_else(|e| {
        eprintln!("ERROR: failed to load datasets: {e}");
        std::process::exit(1);
    });
    eprintln!("Loaded {} problems total", dataset.len());

    // Load LearnedTranslator
    eprintln!("Loading LearnedTranslator...");
    let translator = LearnedTranslator::load(
        &config.tokenizer,
        &config.encoder,
        &config.decoder,
    )
    .unwrap_or_else(|e| {
        eprintln!("ERROR: failed to load translator: {e}");
        std::process::exit(1);
    });
    eprintln!("Translator loaded");

    // Encode problems to FramePairs
    eprintln!("Encoding problems to FramePairs...");
    let encode_start = Instant::now();
    let problems: Vec<_> = dataset.iter().collect();
    let pairs = problems_to_frame_pairs(
        &problems.iter().map(|p| (*p).clone()).collect::<Vec<_>>(),
        &translator as &dyn Translator,
    );
    let encode_time = encode_start.elapsed();
    eprintln!(
        "Encoded {} pairs in {:.1}s ({:.1} problems/sec)",
        pairs.len(),
        encode_time.as_secs_f64(),
        dataset.len() as f64 / encode_time.as_secs_f64(),
    );

    if pairs.is_empty() {
        eprintln!("ERROR: no valid FramePairs generated");
        std::process::exit(1);
    }

    // Create ScaledVfn
    let vfn_config = ScaledVfnConfig {
        hidden_dim: config.hidden_dim,
        num_blocks: config.num_blocks,
        ..ScaledVfnConfig::default()
    };
    let var_map = VarMap::new();
    let vfn = ScaledVfn::new_trainable(&vfn_config, &var_map, &device).unwrap_or_else(|e| {
        eprintln!("ERROR: failed to create ScaledVfn: {e}");
        std::process::exit(1);
    });
    eprintln!("ScaledVfn created: {} parameters", vfn.param_count());

    // Ensure output directory exists
    if let Some(parent) = config.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Training config
    let flow_config = ScaledFlowConfig {
        max_lr: config.lr,
        min_lr: config.lr * 0.01,
        warmup_steps: config.warmup_steps,
        epochs: config.epochs,
        batch_size: config.batch_size,
        log_interval: 50,
        ..ScaledFlowConfig::default()
    };

    // Train with per-epoch checkpointing
    eprintln!();
    eprintln!("Starting training...");
    let train_start = Instant::now();

    let result = train_scaled_vfn_with_checkpoints(
        &vfn,
        &var_map,
        &pairs,
        &flow_config,
        &device,
        &config.output,
    );

    let train_time = train_start.elapsed();

    match result {
        Ok(total_steps) => {
            // Save final model
            if let Err(e) = vfn.save(&config.output) {
                eprintln!("ERROR: failed to save final model: {e}");
                std::process::exit(1);
            }
            eprintln!();
            eprintln!("Final model saved: {}", config.output.display());
            eprintln!(
                "Training complete: {} steps in {:.1}s",
                total_steps,
                train_time.as_secs_f64()
            );
        }
        Err(e) => {
            eprintln!("ERROR: training failed: {e}");
            std::process::exit(1);
        }
    }
}

/// Train with per-epoch checkpoint saving.
///
/// We run the training loop manually (epoch by epoch) so we can
/// save checkpoints between epochs. This is more robust than running
/// the full `train_scaled_vfn` and losing progress on crash.
fn train_scaled_vfn_with_checkpoints(
    vfn: &ScaledVfn,
    var_map: &VarMap,
    pairs: &[volt_soft::training::FramePair],
    config: &ScaledFlowConfig,
    device: &Device,
    output: &std::path::Path,
) -> Result<usize, volt_core::VoltError> {
    // Run one-epoch configs in a loop so we can checkpoint between
    let mut total_steps = 0;

    for epoch in 0..config.epochs {
        let epoch_start = Instant::now();

        let epoch_config = ScaledFlowConfig {
            epochs: 1,
            seed: config.seed.wrapping_add(epoch as u64),
            ..config.clone()
        };

        let result = train_scaled_vfn(vfn, var_map, pairs, &epoch_config, device)?;
        total_steps += result.total_steps;

        let epoch_time = epoch_start.elapsed();
        eprintln!(
            "[Epoch {}/{}] Train: {:.4} | Valid: {:.4} | Time: {:.1}s | Total steps: {}",
            epoch + 1,
            config.epochs,
            result.final_train_loss,
            result.final_valid_loss,
            epoch_time.as_secs_f64(),
            total_steps,
        );

        // Save checkpoint
        let checkpoint_path = output.with_file_name(format!(
            "{}_epoch_{}.safetensors",
            output
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy(),
            epoch + 1
        ));
        if let Err(e) = vfn.save(&checkpoint_path) {
            eprintln!("WARNING: failed to save checkpoint: {e}");
        } else {
            eprintln!("Checkpoint saved: {}", checkpoint_path.display());
        }
    }

    Ok(total_steps)
}

/// Select compute device.
fn select_device(name: &str) -> Device {
    match name {
        "cuda" | "gpu" => match Device::cuda_if_available(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("WARNING: CUDA not available ({e}), falling back to CPU");
                Device::Cpu
            }
        },
        _ => Device::Cpu,
    }
}

struct TrainConfig {
    data_paths: Vec<PathBuf>,
    tokenizer: PathBuf,
    encoder: PathBuf,
    decoder: PathBuf,
    output: PathBuf,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    warmup_steps: usize,
    hidden_dim: usize,
    num_blocks: usize,
    device_name: String,
}

fn parse_args(args: &[String]) -> TrainConfig {
    let mut data_paths = Vec::new();
    let mut tokenizer = PathBuf::from("checkpoints/code_tokenizer.json");
    let mut encoder = PathBuf::from("checkpoints/code_encoder.safetensors");
    let mut decoder = PathBuf::from("checkpoints/code_decoder.safetensors");
    let mut output = PathBuf::from("checkpoints/scaled_vfn.safetensors");
    let mut epochs = 10usize;
    let mut batch_size = 32usize;
    let mut lr = 1e-4;
    let mut warmup_steps = 2000usize;
    let mut hidden_dim = 2048usize;
    let mut num_blocks = 6usize;
    let mut device_name = "cpu".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                data_paths.push(PathBuf::from(&args[i]));
            }
            "--tokenizer" => {
                i += 1;
                tokenizer = PathBuf::from(&args[i]);
            }
            "--encoder" => {
                i += 1;
                encoder = PathBuf::from(&args[i]);
            }
            "--decoder" => {
                i += 1;
                decoder = PathBuf::from(&args[i]);
            }
            "--output" => {
                i += 1;
                output = PathBuf::from(&args[i]);
            }
            "--epochs" => {
                i += 1;
                epochs = args[i].parse().expect("invalid epochs");
            }
            "--batch-size" => {
                i += 1;
                batch_size = args[i].parse().expect("invalid batch-size");
            }
            "--lr" => {
                i += 1;
                lr = args[i].parse().expect("invalid lr");
            }
            "--warmup" => {
                i += 1;
                warmup_steps = args[i].parse().expect("invalid warmup");
            }
            "--hidden-dim" => {
                i += 1;
                hidden_dim = args[i].parse().expect("invalid hidden-dim");
            }
            "--num-blocks" => {
                i += 1;
                num_blocks = args[i].parse().expect("invalid num-blocks");
            }
            "--device" => {
                i += 1;
                device_name = args[i].to_string();
            }
            "--help" | "-h" => {
                eprintln!("Usage: train-vfn [OPTIONS]");
                eprintln!("  --data <PATH>         JSONL dataset file (repeatable)");
                eprintln!("  --tokenizer <PATH>    BPE tokenizer JSON");
                eprintln!("  --encoder <PATH>      Frozen encoder safetensors");
                eprintln!("  --decoder <PATH>      Frozen decoder safetensors");
                eprintln!("  --output <PATH>       Output safetensors path");
                eprintln!("  --epochs <N>          Training epochs (default: 10)");
                eprintln!("  --batch-size <N>      Batch size (default: 32)");
                eprintln!("  --lr <FLOAT>          Peak learning rate (default: 1e-4)");
                eprintln!("  --warmup <N>          Warmup steps (default: 2000)");
                eprintln!("  --hidden-dim <N>      VFN hidden dimension (default: 2048)");
                eprintln!("  --num-blocks <N>      Number of residual blocks (default: 6)");
                eprintln!("  --device <cpu|cuda>   Compute device (default: cpu)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if data_paths.is_empty() {
        eprintln!("ERROR: at least one --data path is required");
        eprintln!("Run with --help for usage");
        std::process::exit(1);
    }

    TrainConfig {
        data_paths,
        tokenizer,
        encoder,
        decoder,
        output,
        epochs,
        batch_size,
        lr,
        warmup_steps,
        hidden_dim,
        num_blocks,
        device_name,
    }
}
