//! Phase 2 Scaled VFN training CLI for Volt X.
//!
//! Trains the scaled Vector Field Network (~50M params) using Flow Matching
//! on code problem (query, solution) frame pairs.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p volt-learn --features code-training --bin train-vfn -- \
//!   --data "D:\VoltData\phase0\humaneval\humaneval.jsonl" \
//!   --output "checkpoints/scaled_vfn.bin" \
//!   --hidden-dim 1536 --num-blocks 10 \
//!   --steps 10000 --batch-size 32 --lr 1e-4 \
//!   --stage simple_functions
//! ```
//!
//! ## Curriculum Stages
//!
//! - `simple_functions` — Stage 1: single-operation functions
//! - `loops_conditionals` — Stage 2: loops and conditionals
//! - `multi_function` — Stage 3: multi-function programs
//! - `algorithmic` — Stage 4: algorithmic reasoning
//! - `all` — No filtering (default)

use std::path::PathBuf;
use std::time::Instant;

use volt_learn::code_dataset::CodeDataset;
use volt_learn::code_frame_pairs::{
    generate_code_frame_pairs, CodeFramePairConfig, CurriculumStageFilter,
};
use volt_soft::scaled_vfn::ScaledVfnConfig;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config = parse_args(&args);

    eprintln!("=== Volt X VFN Training (Phase 2.1) ===");
    eprintln!("Data:       {}", config.data.display());
    eprintln!("Output:     {}", config.output.display());
    eprintln!("Hidden dim: {}", config.hidden_dim);
    eprintln!("Blocks:     {}", config.num_blocks);
    eprintln!("Steps:      {}", config.steps);
    eprintln!("Batch size: {}", config.batch_size);
    eprintln!("LR:         {}", config.lr);
    eprintln!("Stage:      {}", config.stage);
    eprintln!();

    // Load dataset
    eprintln!("Loading code dataset...");
    let start = Instant::now();
    let dataset = CodeDataset::from_file(&config.data).unwrap_or_else(|e| {
        eprintln!("ERROR: failed to load dataset: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "Loaded {} problems in {:.1}s",
        dataset.len(),
        start.elapsed().as_secs_f32()
    );

    // Generate frame pairs
    eprintln!("Generating frame pairs...");
    let start = Instant::now();
    let stage_filter = match config.stage.as_str() {
        "simple_functions" => Some(CurriculumStageFilter::SimpleFunctions),
        "loops_conditionals" => Some(CurriculumStageFilter::LoopsAndConditionals),
        "multi_function" => Some(CurriculumStageFilter::MultiFunctionPrograms),
        "algorithmic" => Some(CurriculumStageFilter::AlgorithmicReasoning),
        "all" => None,
        other => {
            eprintln!("ERROR: unknown stage '{other}'. Valid: simple_functions, loops_conditionals, multi_function, algorithmic, all");
            std::process::exit(1);
        }
    };

    let pair_config = CodeFramePairConfig {
        resolution: 0,
        seed: 42,
        max_pairs: 0,
        stage_filter,
    };

    let pairs = generate_code_frame_pairs(&dataset, &pair_config).unwrap_or_else(|e| {
        eprintln!("ERROR: failed to generate frame pairs: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "Generated {} frame pairs in {:.1}s",
        pairs.len(),
        start.elapsed().as_secs_f32()
    );

    // Configure VFN
    let vfn_config = ScaledVfnConfig {
        hidden_dim: config.hidden_dim,
        num_blocks: config.num_blocks,
        io_dim: volt_core::SLOT_DIM,
    };
    eprintln!(
        "VFN architecture: {} params ({:.1}M)",
        vfn_config.param_count(),
        vfn_config.param_count() as f64 / 1_000_000.0
    );

    // Phase 2 training requires GPU feature for candle
    #[cfg(feature = "code-training")]
    {
        use candle_core::Device;
        use candle_nn::VarMap;
        use volt_soft::training::phase2::{train_phase2, CurriculumStage, Phase2Config};
        use volt_soft::training::FramePair;

        // Convert CodeFramePairs to volt_soft FramePairs
        let soft_pairs: Vec<FramePair> = pairs
            .iter()
            .map(|p| FramePair {
                question: p.question.clone(),
                answer: p.answer.clone(),
            })
            .collect();

        // Split into train/validation (90/10)
        let val_count = (soft_pairs.len() / 10).max(1).min(100);
        let (val_pairs, train_pairs) = soft_pairs.split_at(val_count);

        eprintln!(
            "Split: {} train, {} validation",
            train_pairs.len(),
            val_pairs.len()
        );

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        eprintln!("Using device: {:?}", device);

        let var_map = VarMap::new();

        let stage = match config.stage.as_str() {
            "simple_functions" => CurriculumStage::SimpleFunctions,
            "loops_conditionals" => CurriculumStage::LoopsAndConditionals,
            "multi_function" => CurriculumStage::MultiFunctionPrograms,
            "algorithmic" => CurriculumStage::AlgorithmicReasoning,
            _ => CurriculumStage::SimpleFunctions,
        };

        let phase2_config = Phase2Config {
            vfn_config,
            learning_rate: config.lr,
            weight_decay: 0.01,
            num_steps: config.steps,
            batch_size: config.batch_size,
            seed: 42,
            resolution: 0,
            stage,
            attention_loss_weight: 0.3,
            joint_attention: true,
            validation_interval: 500,
            validation_pairs: val_pairs.len().min(50),
            warmup_steps: 100,
            max_grad_norm: 1.0,
        };

        eprintln!("\nStarting training...");
        let start = Instant::now();
        let result =
            train_phase2(&var_map, train_pairs, val_pairs, &phase2_config, &device)
                .unwrap_or_else(|e| {
                    eprintln!("ERROR: training failed: {e}");
                    std::process::exit(1);
                });

        let elapsed = start.elapsed().as_secs_f32();
        eprintln!("\n=== Training Complete ===");
        eprintln!("Steps:      {}", result.steps_completed);
        eprintln!("Final loss: {:.6}", result.final_loss);
        eprintln!(
            "Time:       {:.1}s ({:.1} steps/s)",
            elapsed,
            result.steps_completed as f32 / elapsed
        );

        if !result.validation_metrics.is_empty() {
            eprintln!("\nValidation History:");
            for m in &result.validation_metrics {
                eprintln!(
                    "  Step {:5}: MSE={:.4}  cos_sim={:.4}  dir_acc={:.2}%",
                    m.step,
                    m.avg_mse_loss,
                    m.avg_cosine_similarity,
                    m.directional_accuracy * 100.0
                );
            }
        }

        let save_path = config.output.with_extension("safetensors");
        eprintln!("\nSaving checkpoint to {}...", save_path.display());
        var_map.save(&save_path).unwrap_or_else(|e| {
            eprintln!("ERROR: failed to save checkpoint: {e}");
            std::process::exit(1);
        });
        eprintln!("Done.");
    }

    #[cfg(not(feature = "code-training"))]
    {
        let _ = pairs;
        let _ = vfn_config;
        eprintln!("ERROR: Phase 2 training requires --features code-training");
        eprintln!(
            "Run: cargo run --release -p volt-learn --features code-training --bin train-vfn -- ..."
        );
        std::process::exit(1);
    }
}

struct CliConfig {
    data: PathBuf,
    output: PathBuf,
    hidden_dim: usize,
    num_blocks: usize,
    steps: usize,
    batch_size: usize,
    lr: f64,
    stage: String,
}

fn parse_args(args: &[String]) -> CliConfig {
    let mut config = CliConfig {
        data: PathBuf::from("humaneval.jsonl"),
        output: PathBuf::from("checkpoints/scaled_vfn.bin"),
        hidden_dim: 1536,
        num_blocks: 10,
        steps: 10_000,
        batch_size: 32,
        lr: 1e-4,
        stage: "all".to_string(),
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                config.data = PathBuf::from(&args[i]);
            }
            "--output" => {
                i += 1;
                config.output = PathBuf::from(&args[i]);
            }
            "--hidden-dim" => {
                i += 1;
                config.hidden_dim = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("ERROR: invalid --hidden-dim value");
                    std::process::exit(1);
                });
            }
            "--num-blocks" => {
                i += 1;
                config.num_blocks = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("ERROR: invalid --num-blocks value");
                    std::process::exit(1);
                });
            }
            "--steps" => {
                i += 1;
                config.steps = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("ERROR: invalid --steps value");
                    std::process::exit(1);
                });
            }
            "--batch-size" => {
                i += 1;
                config.batch_size = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("ERROR: invalid --batch-size value");
                    std::process::exit(1);
                });
            }
            "--lr" => {
                i += 1;
                config.lr = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("ERROR: invalid --lr value");
                    std::process::exit(1);
                });
            }
            "--stage" => {
                i += 1;
                config.stage = args[i].clone();
            }
            "--help" | "-h" => {
                eprintln!("Usage: train-vfn [options]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --data PATH        JSONL dataset file");
                eprintln!("  --output PATH      Checkpoint output path");
                eprintln!("  --hidden-dim N     Hidden dimension (default: 1536)");
                eprintln!("  --num-blocks N     Number of residual blocks (default: 10)");
                eprintln!("  --steps N          Training steps (default: 10000)");
                eprintln!("  --batch-size N     Batch size (default: 32)");
                eprintln!("  --lr FLOAT         Learning rate (default: 1e-4)");
                eprintln!("  --stage STAGE      Curriculum stage: simple_functions, loops_conditionals,");
                eprintln!("                     multi_function, algorithmic, all (default: all)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                eprintln!("Use --help for usage information.");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    config
}
