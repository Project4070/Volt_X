//! Contrastive encoder training CLI for Volt X Phase 1.
//!
//! Trains the CNN code encoder on CodeSearchNet (code, docstring) pairs
//! using InfoNCE contrastive loss + role classification loss.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p volt-learn --features code-training --bin train-encoder -- \
//!   --data "D:\VoltData\phase1\codesearchnet\data\codesearchnet_python_train.jsonl" \
//!   --tokenizer "checkpoints/code_tokenizer.json" \
//!   --output "checkpoints/code_encoder.safetensors" \
//!   --epochs 10 --batch-size 128 --lr 5e-4
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::Optimizer;
use tokenizers::Tokenizer;

use volt_learn::codesearchnet::CsnDataset;
use volt_learn::contrastive::infonce_loss;
use volt_learn::role_labels::label_code_tokens;
use volt_translate::code_encoder::{CodeEncoder, CodeEncoderConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config = parse_args(&args);

    eprintln!("=== Volt X Encoder Training (Phase 1.1) ===");
    eprintln!("Data:       {}", config.data.display());
    eprintln!("Tokenizer:  {}", config.tokenizer.display());
    eprintln!("Output:     {}", config.output.display());
    eprintln!("Epochs:     {}", config.epochs);
    eprintln!("Batch size: {}", config.batch_size);
    eprintln!("LR:         {}", config.lr);
    eprintln!("Warmup:     {} steps", config.warmup_steps);
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

    // Create encoder
    let enc_config = CodeEncoderConfig::default();
    let encoder = CodeEncoder::new_random(&enc_config, &device).unwrap_or_else(|e| {
        eprintln!("ERROR: failed to create encoder: {e}");
        std::process::exit(1);
    });
    eprintln!("Encoder parameters: {}", encoder.param_count());

    // Optimizer
    let mut optimizer =
        candle_nn::AdamW::new(encoder.var_map().all_vars(), candle_nn::ParamsAdamW {
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
    let contrastive_weight = 0.8;
    let role_weight = 0.2;

    for epoch in 1..=config.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f64;
        let mut epoch_contrastive = 0.0f64;
        let mut epoch_role = 0.0f64;
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

            if batch.len() < 2 {
                continue;
            }

            // Tokenize code and docstrings
            let (code_ids, code_role_labels) =
                tokenize_batch_with_roles(&tokenizer, &batch, true, enc_config.max_seq_len);
            let (doc_ids, _) =
                tokenize_batch_with_roles(&tokenizer, &batch, false, enc_config.max_seq_len);

            let actual_batch = code_ids.len();
            if actual_batch < 2 {
                continue;
            }

            // Find max seq lengths in this batch
            let code_max_len = code_ids.iter().map(|ids| ids.len()).max().unwrap_or(1);
            let doc_max_len = doc_ids.iter().map(|ids| ids.len()).max().unwrap_or(1);

            // Pad and create tensors
            let code_tensor = ids_to_tensor(&code_ids, code_max_len, &device);
            let doc_tensor = ids_to_tensor(&doc_ids, doc_max_len, &device);

            // Forward pass for code
            let code_output = match encoder.forward(&code_tensor) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("  [step {global_step}] forward error: {e}");
                    continue;
                }
            };

            // Forward pass for docstrings
            let doc_output = match encoder.forward(&doc_tensor) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("  [step {global_step}] forward error: {e}");
                    continue;
                }
            };

            // Contrastive loss on summary vectors
            let contrastive_loss =
                match infonce_loss(&code_output.summary, &doc_output.summary, 0.07) {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("  [step {global_step}] contrastive loss error: {e}");
                        continue;
                    }
                };

            // Role classification loss
            let role_label_tensor = role_labels_to_tensor(
                &code_role_labels,
                code_max_len,
                &device,
            );

            // Get role logits from features (need to compute from features → role_head)
            // The role_probs from forward() are after softmax — we need the logits.
            // Actually role_probs are softmax output, but cross_entropy in candle expects
            // raw logits. Let's use log of probs as a workaround, or better, use the
            // features → role_head directly.
            // Since role_probs = softmax(role_logits), we need role_logits.
            // We can get this by applying role_head to features.
            // But we don't expose role_head directly... We have role_probs.
            // For cross-entropy: candle's cross_entropy expects logits (unnormalized).
            // Workaround: use NLL loss with log(role_probs) = log_softmax(role_logits).
            // Or: use role_probs.log() as log_probs and compute NLL manually.
            //
            // Actually, let's compute role loss using the role_probs directly:
            // NLL = -sum(log(role_probs[correct_label])) / N
            let role_loss =
                match compute_role_loss_from_probs(&code_output.role_probs, &role_label_tensor) {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("  [step {global_step}] role loss error: {e}");
                        continue;
                    }
                };

            // Combined loss
            let combined = match (&contrastive_loss * contrastive_weight)
                .and_then(|cl| (&role_loss * role_weight).map(|rl| (cl, rl)))
                .and_then(|(cl, rl)| cl + rl)
            {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("  [step {global_step}] loss combination error: {e}");
                    continue;
                }
            };

            // Backward + step
            if let Err(e) = optimizer.backward_step(&combined) {
                eprintln!("  [step {global_step}] backward error: {e}");
                continue;
            }

            let loss_val: f32 = combined.to_scalar().unwrap_or(f32::NAN);
            let cl_val: f32 = contrastive_loss.to_scalar().unwrap_or(f32::NAN);
            let rl_val: f32 = role_loss.to_scalar().unwrap_or(f32::NAN);
            epoch_loss += loss_val as f64;
            epoch_contrastive += cl_val as f64;
            epoch_role += rl_val as f64;
            epoch_samples += actual_batch;

            let step_time = step_start.elapsed();
            let samples_per_sec = actual_batch as f64 / step_time.as_secs_f64();

            // Progress every 10 steps
            if global_step.is_multiple_of(10) || step == 0 {
                eprintln!(
                    "[Epoch {}/{}] Step {}/{} | Loss: {:.4} (C: {:.4} R: {:.4}) | LR: {:.2e} | {:.1} samples/sec",
                    epoch,
                    config.epochs,
                    step + 1,
                    steps_per_epoch,
                    loss_val,
                    cl_val,
                    rl_val,
                    lr,
                    samples_per_sec,
                );
            }
        }

        // Epoch summary
        let epoch_time = epoch_start.elapsed();
        let avg_loss = if epoch_samples > 0 {
            epoch_loss / (epoch_samples as f64 / config.batch_size as f64)
        } else {
            0.0
        };
        let avg_contrastive = if epoch_samples > 0 {
            epoch_contrastive / (epoch_samples as f64 / config.batch_size as f64)
        } else {
            0.0
        };
        let avg_role = if epoch_samples > 0 {
            epoch_role / (epoch_samples as f64 / config.batch_size as f64)
        } else {
            0.0
        };

        // Validation loss
        let valid_loss = compute_validation_loss(
            &encoder,
            &tokenizer,
            &valid_set,
            config.batch_size,
            &enc_config,
            &device,
        );

        eprintln!(
            "[Epoch {}/{}] Complete | Train Loss: {:.4} (C: {:.4} R: {:.4}) | Valid Loss: {:.4} | Time: {:.1}s",
            epoch,
            config.epochs,
            avg_loss,
            avg_contrastive,
            avg_role,
            valid_loss,
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
        if let Err(e) = encoder.save(&checkpoint_path) {
            eprintln!("WARNING: failed to save checkpoint: {e}");
        } else {
            eprintln!("Checkpoint saved: {}", checkpoint_path.display());
        }
    }

    // Save final model
    if let Err(e) = encoder.save(&config.output) {
        eprintln!("ERROR: failed to save final model: {e}");
        std::process::exit(1);
    }
    eprintln!();
    eprintln!("Final model saved: {}", config.output.display());
    eprintln!("Training complete.");
}

/// Compute NLL role classification loss from softmax probabilities.
///
/// Since we have role_probs (after softmax) and candle's NLL expects
/// log-probabilities, we compute log(probs + eps) then use `nll()`.
fn compute_role_loss_from_probs(
    role_probs: &Tensor,
    labels: &Tensor,
) -> Result<Tensor, candle_core::Error> {
    let batch = role_probs.dims()[0];
    let seq_len = role_probs.dims()[1];
    let num_roles = role_probs.dims()[2];

    // Flatten: [batch*seq_len, 16]
    let probs_flat = role_probs.reshape((batch * seq_len, num_roles))?;
    let labels_flat = labels
        .reshape(batch * seq_len)?
        .to_dtype(DType::U32)?;

    // log(probs + eps) to get log-probabilities
    let log_probs = (probs_flat + 1e-8)?.log()?;

    // NLL: -mean(log_prob[i, label[i]])
    candle_nn::loss::nll(&log_probs, &labels_flat)
}

/// Compute validation loss (contrastive only) on the validation set.
fn compute_validation_loss(
    encoder: &CodeEncoder,
    tokenizer: &Tokenizer,
    valid_set: &CsnDataset,
    batch_size: usize,
    enc_config: &CodeEncoderConfig,
    device: &Device,
) -> f64 {
    let steps = (valid_set.len() / batch_size).clamp(1, 50);
    let mut total_loss = 0.0f64;
    let mut count = 0;

    for step in 0..steps {
        let batch_start = step * batch_size;
        let batch_end = (batch_start + batch_size).min(valid_set.len());
        let batch_indices: Vec<usize> = (batch_start..batch_end).collect();
        let batch = valid_set.batch(&batch_indices);

        if batch.len() < 2 {
            continue;
        }

        let (code_ids, _) =
            tokenize_batch_with_roles(tokenizer, &batch, true, enc_config.max_seq_len);
        let (doc_ids, _) =
            tokenize_batch_with_roles(tokenizer, &batch, false, enc_config.max_seq_len);

        if code_ids.len() < 2 {
            continue;
        }

        let code_max_len = code_ids.iter().map(|ids| ids.len()).max().unwrap_or(1);
        let doc_max_len = doc_ids.iter().map(|ids| ids.len()).max().unwrap_or(1);

        let code_tensor = ids_to_tensor(&code_ids, code_max_len, device);
        let doc_tensor = ids_to_tensor(&doc_ids, doc_max_len, device);

        let code_out = match encoder.forward(&code_tensor) {
            Ok(o) => o,
            Err(_) => continue,
        };
        let doc_out = match encoder.forward(&doc_tensor) {
            Ok(o) => o,
            Err(_) => continue,
        };

        if let Ok(loss) = infonce_loss(&code_out.summary, &doc_out.summary, 0.07)
            && let Ok(val) = loss.to_scalar::<f32>()
            && val.is_finite()
        {
            total_loss += val as f64;
            count += 1;
        }
    }

    if count > 0 {
        total_loss / count as f64
    } else {
        f64::NAN
    }
}

/// Tokenize a batch, returning (token_ids_per_sample, role_labels_per_sample).
///
/// If `use_code` is true, tokenizes the code field; otherwise the docstring.
fn tokenize_batch_with_roles(
    tokenizer: &Tokenizer,
    batch: &[&volt_learn::codesearchnet::CsnRecord],
    use_code: bool,
    max_len: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut all_ids = Vec::new();
    let mut all_labels = Vec::new();

    for record in batch {
        let text = if use_code {
            &record.code
        } else {
            &record.docstring
        };

        let Ok(encoding) = tokenizer.encode(text.as_str(), false) else {
            continue;
        };

        let ids: Vec<u32> = encoding.get_ids().iter().take(max_len).copied().collect();
        if ids.is_empty() {
            continue;
        }

        // Role labels (only meaningful for code)
        let labels = if use_code {
            let tokens: Vec<&str> = encoding
                .get_tokens()
                .iter()
                .take(max_len)
                .map(|s| s.as_str())
                .collect();
            label_code_tokens(&tokens)
        } else {
            vec![15u32; ids.len()] // No role for docstrings
        };

        all_ids.push(ids);
        all_labels.push(labels);
    }

    (all_ids, all_labels)
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

/// Pad role label sequences and create a tensor.
fn role_labels_to_tensor(
    labels_batch: &[Vec<u32>],
    max_len: usize,
    device: &Device,
) -> Tensor {
    let batch_size = labels_batch.len();
    let mut data = vec![15u32; batch_size * max_len]; // Default: no role (S15)

    for (i, labels) in labels_batch.iter().enumerate() {
        for (j, &label) in labels.iter().enumerate().take(max_len) {
            data[i * max_len + j] = label;
        }
    }

    Tensor::from_vec(data, (batch_size, max_len), device)
        .expect("failed to create label tensor")
}

/// Cosine learning rate schedule with linear warmup.
fn compute_lr(step: usize, warmup_steps: usize, total_steps: usize, max_lr: f64) -> f64 {
    if step <= warmup_steps {
        // Linear warmup
        max_lr * (step as f64 / warmup_steps.max(1) as f64)
    } else {
        // Cosine decay
        let progress = (step - warmup_steps) as f64 / (total_steps - warmup_steps).max(1) as f64;
        let min_lr = max_lr * 0.01; // Decay to 1% of max
        min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

/// Simple Fisher-Yates shuffle using a basic PRNG.
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
        "cuda" | "gpu" => {
            match Device::cuda_if_available(0) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("WARNING: CUDA not available ({e}), falling back to CPU");
                    Device::Cpu
                }
            }
        }
        _ => Device::Cpu,
    }
}

struct TrainConfig {
    data: PathBuf,
    tokenizer: PathBuf,
    output: PathBuf,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    warmup_steps: usize,
    device_name: String,
}

fn parse_args(args: &[String]) -> TrainConfig {
    let mut data =
        PathBuf::from("D:\\VoltData\\phase1\\codesearchnet\\data\\codesearchnet_python_train.jsonl");
    let mut tokenizer = PathBuf::from("checkpoints/code_tokenizer.json");
    let mut output = PathBuf::from("checkpoints/code_encoder.safetensors");
    let mut epochs = 10usize;
    let mut batch_size = 128usize;
    let mut lr = 5e-4;
    let mut warmup_steps = 2000usize;
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
            "--device" => {
                i += 1;
                device_name = args[i].to_string();
            }
            "--help" | "-h" => {
                eprintln!("Usage: train-encoder [OPTIONS]");
                eprintln!("  --data <PATH>         CodeSearchNet JSONL file");
                eprintln!("  --tokenizer <PATH>    BPE tokenizer JSON");
                eprintln!("  --output <PATH>       Output safetensors path");
                eprintln!("  --epochs <N>          Training epochs (default: 10)");
                eprintln!("  --batch-size <N>      Batch size (default: 128)");
                eprintln!("  --lr <FLOAT>          Max learning rate (default: 5e-4)");
                eprintln!("  --warmup <N>          Warmup steps (default: 2000)");
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
        output,
        epochs,
        batch_size,
        lr,
        warmup_steps,
        device_name,
    }
}
