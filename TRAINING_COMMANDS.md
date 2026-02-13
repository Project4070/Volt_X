# Phase 1: Translator Training Commands

## Prerequisites

- BPE tokenizer training corpus: `D:\VoltData\phase0\the_stack_sample\` (6.5M Python files, 43.8 GB)
- CodeSearchNet training data: `D:\VoltData\phase1\codesearchnet\data\codesearchnet_python_train.jsonl` (100K records, 144 MB)
- GPU: RTX 5090 Mobile (24 GB VRAM)

## Step 0: Create checkpoints directory

```cmd
mkdir c:\Volt_X\checkpoints
```

## Step 1: Train BPE tokenizer (CPU only, ~10-30 min)

```cmd
cargo run --release -p volt-learn --features code-training --bin train-tokenizer -- ^
  --corpus "D:\VoltData\phase0\the_stack_sample" ^
  --vocab-size 32768 ^
  --output "c:\Volt_X\checkpoints\code_tokenizer.json" ^
  --max-files 50000
```

**Output:** `c:\Volt_X\checkpoints\code_tokenizer.json`

## Step 2: Train encoder (GPU, ~60 min on RTX 5090)

Contrastive training on (code, docstring) pairs with role grounding.

```cmd
cargo run --release -p volt-learn --features code-training --bin train-encoder -- ^
  --data "D:\VoltData\phase1\codesearchnet\data\codesearchnet_python_train.jsonl" ^
  --tokenizer "c:\Volt_X\checkpoints\code_tokenizer.json" ^
  --output "c:\Volt_X\checkpoints\code_encoder.safetensors" ^
  --epochs 10 ^
  --batch-size 128 ^
  --lr 5e-4 ^
  --warmup 200 ^
  --device cuda
```

**Output:** `c:\Volt_X\checkpoints\code_encoder.safetensors` + per-epoch checkpoints

## Step 3: Train decoder (GPU, ~2-4 hours)

Reconstruction training using frozen encoder embeddings.

```cmd
cargo run --release -p volt-learn --features code-training --bin train-decoder -- ^
  --data "D:\VoltData\phase1\codesearchnet\data\codesearchnet_python_train.jsonl" ^
  --tokenizer "c:\Volt_X\checkpoints\code_tokenizer.json" ^
  --encoder "c:\Volt_X\checkpoints\code_encoder.safetensors" ^
  --output "c:\Volt_X\checkpoints\code_decoder.safetensors" ^
  --epochs 10 ^
  --batch-size 16 ^
  --lr 3e-4 ^
  --warmup 200 ^
  --device cuda
```

**Output:** `c:\Volt_X\checkpoints\code_decoder.safetensors` + per-epoch checkpoints

## Progress Output

All binaries print progress to stderr:

```
[Epoch 1/10] Step 1/700 | Loss: 4.231 (C: 3.892 R: 4.378) | LR: 2.5e-4 | 45.2 samples/sec
[Epoch 1/10] Step 10/700 | Loss: 3.456 (C: 3.012 R: 3.645) | LR: 5.0e-4 | 46.1 samples/sec
...
[Epoch 1/10] Complete | Train Loss: 3.456 | Valid Loss: 3.612 | Time: 120.3s
Checkpoint saved: c:\Volt_X\checkpoints\code_encoder_epoch_1.safetensors
```

## CLI Help

Each binary supports `--help`:

```cmd
cargo run --release -p volt-learn --features code-training --bin train-tokenizer -- --help
cargo run --release -p volt-learn --features code-training --bin train-encoder -- --help
cargo run --release -p volt-learn --features code-training --bin train-decoder -- --help
```
