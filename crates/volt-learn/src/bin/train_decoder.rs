//! Code decoder training CLI for Volt X Phase 1.
//!
//! Trains the autoregressive decoder to reconstruct code tokens from
//! TensorFrame slot vectors. Uses a frozen encoder to produce slot embeddings,
//! then trains the decoder with teacher-forced cross-entropy reconstruction loss.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p volt-learn --features code-training --bin train-decoder -- \
//!   --data "D:\VoltData\phase1\codesearchnet\data\codesearchnet_python_train.jsonl" \
//!   --tokenizer "checkpoints/code_tokenizer.json" \
//!   --encoder "checkpoints/code_encoder.safetensors" \
//!   --output "checkpoints/code_decoder.safetensors" \
//!   --epochs 10 --batch-size 64 --lr 3e-4
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::Optimizer;
use tokenizers::Tokenizer;

// Note: MAX_SLOTS not needed — decoder cross-attends to per-token features directly
use volt_learn::codesearchnet::CsnDataset;
use volt_translate::code_decoder::{CodeDecoder, CodeDecoderConfig};
use volt_translate::code_encoder::{CodeEncoder, CodeEncoderConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config = parse_args(&args);

    eprintln!("=== Volt X Decoder Training (Phase 1.2) ===");
    eprintln!("Data:       {}", config.data.display());
    eprintln!("Tokenizer:  {}", config.tokenizer.display());
    eprintln!("Encoder:    {}", config.encoder.display());
    eprintln!("Output:     {}", config.output.display());
    eprintln!("Epochs:     {}", config.epochs);
    eprintln!("Batch size: {}", config.batch_size);
    eprintln!("LR:         {}", config.lr);
    eprintln!("Max len:    {} tokens", config.max_decode_len);
    eprintln!("Device:     {:?}", config.device_name);
    eprintln!();

    // Select device
    let device = select_device(&config.device_name);
    eprintln!("Using device: {:?}", device);

    // Load tokenizer
    eprintln!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&config.tokenizer).unwrap_or_else(|e| {
        eprintln!("ERROR: failed to load tokenizer: {e}");
        std::process::exit(1);
    });

    // Load dataset
    eprintln!("Loading CodeSearchNet dataset...");
    let dataset = CsnDataset::from_file(&config.data).unwrap_or_else(|e| {
        eprintln!("ERROR: failed to load dataset: {e}");
        std::process::exit(1);
    });
    let (train_set, valid_set) = dataset.split_train_valid(0.9);
    eprintln!(
        "Dataset: {} train, {} valid",
        train_set.len(),
        valid_set.len()
    );

    // Load frozen encoder
    eprintln!("Loading frozen encoder...");
    let enc_config = CodeEncoderConfig::default();
    let encoder = CodeEncoder::load(&enc_config, &config.encoder, &device).unwrap_or_else(|e| {
        eprintln!("ERROR: failed to load encoder: {e}");
        std::process::exit(1);
    });
    eprintln!("Encoder loaded ({} params, frozen)", encoder.param_count());

    // Create decoder
    let dec_config = CodeDecoderConfig {
        max_seq_len: config.max_decode_len,
        ..CodeDecoderConfig::default()
    };
    let decoder = CodeDecoder::new_random(&dec_config, &device).unwrap_or_else(|e| {
        eprintln!("ERROR: failed to create decoder: {e}");
        std::process::exit(1);
    });
    eprintln!("Decoder parameters: {}", decoder.param_count());

    // Optimizer (only decoder params — encoder is frozen)
    let mut optimizer =
        candle_nn::AdamW::new(decoder.var_map().all_vars(), candle_nn::ParamsAdamW {
            lr: config.lr,
            weight_decay: 0.01,
            ..Default::default()
        })
        .expect("failed to create optimizer");

    // Ensure output directory exists
    if let Some(parent) = config.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let steps_per_epoch = train_set.len() / config.batch_size;
    let total_steps = steps_per_epoch * config.epochs;
    let mut global_step = 0usize;

    eprintln!();
    eprintln!(
        "Training: {} epochs × {} steps = {} total steps",
        config.epochs, steps_per_epoch, total_steps
    );
    eprintln!();

    let mut rng_state = 42u64;

    for epoch in 1..=config.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f64;
        let mut epoch_samples = 0usize;

        // Shuffle indices
        let mut indices: Vec<usize> = (0..train_set.len()).collect();
        shuffle(&mut indices, &mut rng_state);

        for step in 0..steps_per_epoch {
            let step_start = Instant::now();
            global_step += 1;

            // Learning rate schedule: linear warmup + cosine decay
            let lr = compute_lr(global_step, config.warmup_steps, total_steps, config.lr);
            optimizer.set_learning_rate(lr);

            // Get batch
            let batch_start = step * config.batch_size;
            let batch_indices: Vec<usize> = indices
                [batch_start..batch_start + config.batch_size]
                .to_vec();
            let batch = train_set.batch(&batch_indices);

            if batch.is_empty() {
                continue;
            }

            // Tokenize code
            let token_batches = tokenize_batch(&tokenizer, &batch, enc_config.max_seq_len);
            if token_batches.is_empty() {
                continue;
            }

            let actual_batch = token_batches.len();
            let enc_max_len = token_batches.iter().map(|ids| ids.len()).max().unwrap_or(1);

            // Create encoder input tensor
            let enc_tensor = ids_to_tensor(&token_batches, enc_max_len, &device);

            // Forward through frozen encoder → get slot embeddings
            let enc_output = match encoder.forward(&enc_tensor) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("  [step {global_step}] encoder forward error: {e}");
                    continue;
                }
            };

            // Detach encoder features — encoder is frozen, no gradients flow back.
            // Pass per-token CNN features [batch, seq_len, 256] directly to decoder
            // cross-attention. Aggregating into 16 slot averages destroyed all
            // positional/token identity — that was why accuracy was stuck at 33%.
            let context = enc_output.features.detach();

            // Target token IDs for reconstruction
            // Truncate to decoder max_len
            let target_len = enc_max_len.min(config.max_decode_len);
            let target_ids = truncate_and_pad(&token_batches, target_len, &device);

            // Create shifted-right input for teacher forcing: [0, t1, t2, ..., t_{n-1}]
            let shifted_input = shift_right(&target_ids, &device);

            // Decoder forward (autoregressive with teacher forcing)
            let logits = match decoder.forward(&context, &shifted_input) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("  [step {global_step}] decoder forward error: {e}");
                    continue;
                }
            };

            // Flatten for cross-entropy: [batch*target_len, vocab] vs [batch*target_len]
            let vocab_size = dec_config.vocab_size;
            let logits_flat = match logits.reshape((actual_batch * target_len, vocab_size)) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("  [step {global_step}] reshape error: {e}");
                    continue;
                }
            };
            let targets_flat = match target_ids.reshape(actual_batch * target_len) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("  [step {global_step}] reshape error: {e}");
                    continue;
                }
            };

            // Cross-entropy loss
            let loss = match candle_nn::loss::cross_entropy(&logits_flat, &targets_flat) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("  [step {global_step}] loss error: {e}");
                    continue;
                }
            };

            // Backward + step
            if let Err(e) = optimizer.backward_step(&loss) {
                eprintln!("  [step {global_step}] backward error: {e}");
                continue;
            }

            let loss_val: f32 = loss.to_scalar().unwrap_or(f32::NAN);
            epoch_loss += loss_val as f64;
            epoch_samples += actual_batch;

            let step_time = step_start.elapsed();
            let samples_per_sec = actual_batch as f64 / step_time.as_secs_f64();

            // Progress every 10 steps
            if global_step.is_multiple_of(10) || step == 0 {
                // Compute token accuracy on this batch
                let accuracy = compute_batch_accuracy(&logits, &target_ids);

                eprintln!(
                    "[Epoch {}/{}] Step {}/{} | Loss: {:.4} | Acc: {:.2}% | LR: {:.2e} | {:.1} samples/sec",
                    epoch,
                    config.epochs,
                    step + 1,
                    steps_per_epoch,
                    loss_val,
                    accuracy * 100.0,
                    lr,
                    samples_per_sec,
                );
            }
        }

        // Epoch summary
        let epoch_time = epoch_start.elapsed();
        let n_batches = epoch_samples as f64 / config.batch_size as f64;
        let avg_loss = if n_batches > 0.0 {
            epoch_loss / n_batches
        } else {
            0.0
        };

        // Validation loss
        let (valid_loss, valid_acc) = compute_validation_loss(
            &encoder,
            &decoder,
            &tokenizer,
            &valid_set,
            config.batch_size,
            &enc_config,
            &dec_config,
            &device,
        );

        eprintln!(
            "[Epoch {}/{}] Complete | Train Loss: {:.4} | Valid Loss: {:.4} | Valid Acc: {:.2}% | Time: {:.1}s",
            epoch,
            config.epochs,
            avg_loss,
            valid_loss,
            valid_acc * 100.0,
            epoch_time.as_secs_f64(),
        );

        // Save checkpoint
        let checkpoint_path = config
            .output
            .with_file_name(format!(
                "{}_epoch_{}.safetensors",
                config
                    .output
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy(),
                epoch
            ));
        if let Err(e) = decoder.save(&checkpoint_path) {
            eprintln!("WARNING: failed to save checkpoint: {e}");
        } else {
            eprintln!("Checkpoint saved: {}", checkpoint_path.display());
        }
    }

    // Save final model
    if let Err(e) = decoder.save(&config.output) {
        eprintln!("ERROR: failed to save final model: {e}");
        std::process::exit(1);
    }
    eprintln!();
    eprintln!("Final model saved: {}", config.output.display());
    eprintln!("Decoder training complete.");
}

/// Compute token-level accuracy for a batch.
fn compute_batch_accuracy(logits: &Tensor, targets: &Tensor) -> f64 {
    let preds = match logits.argmax(candle_core::D::Minus1) {
        Ok(p) => p,
        Err(_) => return 0.0,
    };
    let eq = match preds.eq(targets) {
        Ok(e) => e,
        Err(_) => return 0.0,
    };
    let correct: f64 = match eq.to_dtype(DType::F32).and_then(|t| t.mean_all()) {
        Ok(t) => match t.to_scalar::<f32>() {
            Ok(v) => v as f64,
            Err(_) => return 0.0,
        },
        Err(_) => return 0.0,
    };
    correct
}

/// Compute validation loss and accuracy.
#[allow(clippy::too_many_arguments)]
fn compute_validation_loss(
    encoder: &CodeEncoder,
    decoder: &CodeDecoder,
    tokenizer: &Tokenizer,
    valid_set: &CsnDataset,
    batch_size: usize,
    enc_config: &CodeEncoderConfig,
    dec_config: &CodeDecoderConfig,
    device: &Device,
) -> (f64, f64) {
    let steps = (valid_set.len() / batch_size).clamp(1, 50);
    let mut total_loss = 0.0f64;
    let mut total_acc = 0.0f64;
    let mut count = 0;

    for step in 0..steps {
        let batch_start = step * batch_size;
        let batch_end = (batch_start + batch_size).min(valid_set.len());
        let batch_indices: Vec<usize> = (batch_start..batch_end).collect();
        let batch = valid_set.batch(&batch_indices);

        if batch.is_empty() {
            continue;
        }

        let token_batches = tokenize_batch(tokenizer, &batch, enc_config.max_seq_len);
        if token_batches.is_empty() {
            continue;
        }

        let actual_batch = token_batches.len();
        let enc_max_len = token_batches.iter().map(|ids| ids.len()).max().unwrap_or(1);
        let enc_tensor = ids_to_tensor(&token_batches, enc_max_len, device);

        let enc_output = match encoder.forward(&enc_tensor) {
            Ok(o) => o,
            Err(_) => continue,
        };

        // Per-token features for decoder cross-attention
        let context = enc_output.features.detach();

        let target_len = enc_max_len.min(dec_config.max_seq_len);
        let target_ids = truncate_and_pad(&token_batches, target_len, device);

        // Teacher-forced input: shifted right
        let shifted_input = shift_right(&target_ids, device);

        let logits = match decoder.forward(&context, &shifted_input) {
            Ok(l) => l,
            Err(_) => continue,
        };

        let vocab_size = dec_config.vocab_size;
        let logits_flat = match logits.reshape((actual_batch * target_len, vocab_size)) {
            Ok(l) => l,
            Err(_) => continue,
        };
        let targets_flat = match target_ids.reshape(actual_batch * target_len) {
            Ok(t) => t,
            Err(_) => continue,
        };

        if let Ok(loss) = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)
            && let Ok(val) = loss.to_scalar::<f32>()
            && val.is_finite()
        {
            total_loss += val as f64;
            let acc = compute_batch_accuracy(&logits, &target_ids);
            total_acc += acc;
            count += 1;
        }
    }

    if count > 0 {
        (total_loss / count as f64, total_acc / count as f64)
    } else {
        (f64::NAN, 0.0)
    }
}

/// Tokenize a batch of code records.
fn tokenize_batch(
    tokenizer: &Tokenizer,
    batch: &[&volt_learn::codesearchnet::CsnRecord],
    max_len: usize,
) -> Vec<Vec<u32>> {
    let mut all_ids = Vec::new();

    for record in batch {
        let Ok(encoding) = tokenizer.encode(record.code.as_str(), false) else {
            continue;
        };

        let ids: Vec<u32> = encoding.get_ids().iter().take(max_len).copied().collect();
        if ids.is_empty() {
            continue;
        }
        all_ids.push(ids);
    }

    all_ids
}

/// Pad token ID sequences and create a tensor.
fn ids_to_tensor(ids_batch: &[Vec<u32>], max_len: usize, device: &Device) -> Tensor {
    let batch_size = ids_batch.len();
    let mut data = vec![0u32; batch_size * max_len];

    for (i, ids) in ids_batch.iter().enumerate() {
        for (j, &id) in ids.iter().enumerate().take(max_len) {
            data[i * max_len + j] = id;
        }
    }

    Tensor::from_vec(data, (batch_size, max_len), device)
        .expect("failed to create token tensor")
}

/// Truncate sequences to target_len and create a u32 tensor.
fn truncate_and_pad(
    ids_batch: &[Vec<u32>],
    target_len: usize,
    device: &Device,
) -> Tensor {
    let batch_size = ids_batch.len();
    let mut data = vec![0u32; batch_size * target_len];

    for (i, ids) in ids_batch.iter().enumerate() {
        for (j, &id) in ids.iter().enumerate().take(target_len) {
            data[i * target_len + j] = id;
        }
    }

    Tensor::from_vec(data, (batch_size, target_len), device)
        .expect("failed to create target tensor")
}

/// Shift token IDs right by 1 position, prepending 0 (BOS).
///
/// Input:  `[t1, t2, t3, ..., tn]`
/// Output: `[0,  t1, t2, ..., t_{n-1}]`
fn shift_right(target_ids: &Tensor, device: &Device) -> Tensor {
    let dims = target_ids.dims();
    let batch = dims[0];
    let seq_len = dims[1];

    if seq_len <= 1 {
        // Edge case: just return zeros (BOS only)
        return Tensor::zeros((batch, seq_len), DType::U32, device)
            .expect("failed to create BOS tensor");
    }

    // BOS column: [batch, 1] of zeros
    let bos = Tensor::zeros((batch, 1), DType::U32, device)
        .expect("failed to create BOS tensor");

    // All but last token: [batch, seq_len - 1]
    let prefix = target_ids
        .narrow(1, 0, seq_len - 1)
        .expect("failed to narrow target_ids");

    // Concat: [batch, 1] + [batch, seq_len-1] = [batch, seq_len]
    Tensor::cat(&[&bos, &prefix], 1).expect("failed to concat shifted input")
}

/// Cosine learning rate schedule with linear warmup.
fn compute_lr(step: usize, warmup_steps: usize, total_steps: usize, max_lr: f64) -> f64 {
    if step <= warmup_steps {
        max_lr * (step as f64 / warmup_steps.max(1) as f64)
    } else {
        let progress = (step - warmup_steps) as f64 / (total_steps - warmup_steps).max(1) as f64;
        let min_lr = max_lr * 0.01;
        min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

/// Simple Fisher-Yates shuffle.
fn shuffle(data: &mut [usize], rng_state: &mut u64) {
    for i in (1..data.len()).rev() {
        *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (*rng_state >> 33) as usize % (i + 1);
        data.swap(i, j);
    }
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
    data: PathBuf,
    tokenizer: PathBuf,
    encoder: PathBuf,
    output: PathBuf,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    warmup_steps: usize,
    max_decode_len: usize,
    device_name: String,
}

fn parse_args(args: &[String]) -> TrainConfig {
    let mut data =
        PathBuf::from("D:\\VoltData\\phase1\\codesearchnet\\data\\codesearchnet_python_train.jsonl");
    let mut tokenizer = PathBuf::from("checkpoints/code_tokenizer.json");
    let mut encoder = PathBuf::from("checkpoints/code_encoder.safetensors");
    let mut output = PathBuf::from("checkpoints/code_decoder.safetensors");
    let mut epochs = 10usize;
    let mut batch_size = 64usize;
    let mut lr = 3e-4;
    let mut warmup_steps = 1000usize;
    let mut max_decode_len = 256usize;
    let mut device_name = "cpu".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                data = PathBuf::from(&args[i]);
            }
            "--tokenizer" => {
                i += 1;
                tokenizer = PathBuf::from(&args[i]);
            }
            "--encoder" => {
                i += 1;
                encoder = PathBuf::from(&args[i]);
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
            "--max-len" => {
                i += 1;
                max_decode_len = args[i].parse().expect("invalid max-len");
            }
            "--device" => {
                i += 1;
                device_name = args[i].to_string();
            }
            "--help" | "-h" => {
                eprintln!("Usage: train-decoder [OPTIONS]");
                eprintln!("  --data <PATH>         CodeSearchNet JSONL file");
                eprintln!("  --tokenizer <PATH>    BPE tokenizer JSON");
                eprintln!("  --encoder <PATH>      Frozen encoder safetensors");
                eprintln!("  --output <PATH>       Output safetensors path");
                eprintln!("  --epochs <N>          Training epochs (default: 10)");
                eprintln!("  --batch-size <N>      Batch size (default: 64)");
                eprintln!("  --lr <FLOAT>          Max learning rate (default: 3e-4)");
                eprintln!("  --warmup <N>          Warmup steps (default: 1000)");
                eprintln!("  --max-len <N>         Max decode length (default: 256)");
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

    TrainConfig {
        data,
        tokenizer,
        encoder,
        output,
        epochs,
        batch_size,
        lr,
        warmup_steps,
        max_decode_len,
        device_name,
    }
}
