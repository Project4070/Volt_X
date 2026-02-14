# Volt X — Cloud Training & Parallelization Plan

This document explains **exactly** how to run Volt X's remaining training
phases on cloud GPU/CPU servers, what files to transfer, what commands to
run, and how to parallelize independent work to minimize total wall-clock
time.

**Audience:** Anyone with basic Linux/CLI experience and access to a cloud
GPU provider (Lambda Labs, RunPod, Vast.ai, AWS, GCP, etc.).

**Date:** 2026-02-14
**Current status:** Phases 0.1, 0.2, 0.4, 1.1, 1.2, 1.3 are complete.
Phase 0.3 (codebook k-means) and Phase 2 (VFN training) are ready to
launch on cloud instances simultaneously.

---

## Table of Contents

1. [Overview: What Needs to Be Trained](#1-overview-what-needs-to-be-trained)
2. [Dependency Graph: What Blocks What](#2-dependency-graph-what-blocks-what)
3. [Parallelization Strategy](#3-parallelization-strategy)
4. [Cloud Instance Recommendations](#4-cloud-instance-recommendations)
5. [One-Time Setup: Build Environment](#5-one-time-setup-build-environment)
6. [File Transfer Manifest](#6-file-transfer-manifest)
7. [Phase-by-Phase Execution Guide](#7-phase-by-phase-execution-guide)
   - [Phase 0.3: Codebook k-means](#phase-03-codebook-k-means-cloud-cpu)
   - [Phase 2: VFN Training](#phase-2-vfn-flow-matching-gpu-required)
   - [Phase 3: Hard Core Calibration](#phase-3-hard-core-calibration-light-gpu)
   - [Phase 4: Joint Alignment](#phase-4-joint-alignment-gpu-required)
   - [Phase 5: Scale & Benchmark](#phase-5-scale--benchmark-multi-gpu)
8. [Checkpoint Sync Workflow](#8-checkpoint-sync-workflow)
9. [Cost Estimates](#9-cost-estimates)
10. [Timeline: Serial vs. Parallel](#10-timeline-serial-vs-parallel)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Overview: What Needs to Be Trained

| Phase | Task | Parameters | Hardware | Est. Time (1× H100) |
|-------|------|-----------|----------|---------------------|
| 0.3 | Codebook k-means | 16M centroids | Cloud CPU (32+ cores) | ~1.5 hrs |
| 2.1 | VFN Flow Matching | 51M | GPU | 100–200 hrs |
| 2.2 | Slot Attention | 33K (joint w/ 2.1) | GPU | (included in 2.1) |
| 2.3 | Diffusion Controller | 10K (joint w/ 2.1) | GPU | 5–10 hrs |
| 3.1 | Intent Router Calibration | 1.3K | Light GPU | 1–2 hrs |
| 3.2 | Certainty (γ) Calibration | ~100 | Light GPU | 5–10 hrs |
| 3.3 | Safety Axiom Refinement | 1.3K | Light GPU | 3–5 hrs |
| 4.1 | End-to-End Fine-Tuning | 51M | GPU | 75–150 hrs |
| 4.2 | RLVF Alignment | 51M | GPU | 40–75 hrs |
| 4.3 | Sleep Consolidation | 51M | GPU | 15–25 hrs |
| 5.1 | VFN Scaling (200M→500M) | 500M | Multi-GPU | 400–750 hrs |
| 5.2 | Multi-Language Training | 500M + decoders | Multi-GPU | 150–300 hrs |
| 5.3 | Benchmark Publication | — (inference) | GPU | 100–200 hrs |

**Total remaining:** ~900–1,800 H100-hours

---

## 2. Dependency Graph: What Blocks What

```
 DONE ───────────────────────────────────────────────────────────
  0.1  VFN Checkpoint system       ✅
  0.2  Code Dataset Pipeline       ✅
  0.4  Code Attention Bias         ✅
  1.1  Code Encoder (5.1M)         ✅  → checkpoints/code_encoder.safetensors
  1.2  Code Decoder (6.7M)         ✅  → checkpoints/code_decoder.safetensors
  1.3  Role Grounding              ✅  (joint with 1.1)

 NOW (both on cloud, in parallel) ───────────────────────────────

  0.3  Codebook k-means ─────── Cloud CPU instance (32+ cores)
       │                         Output: checkpoints/codebook_code.bin
       │                         Takes ~1.5 hrs on 32-core cloud vs. 6+ hrs local
       │
       │  (no dependency between 0.3 and 2.x — they run simultaneously)
       │
  2.1  VFN Flow Matching ────── Cloud GPU instance (H100)
  2.2  Slot Attention    ────── (joint with 2.1)
  2.3  Diffusion Tuning ─────── (joint with 2.1)
       │                         Output: checkpoints/scaled_vfn.safetensors
       │
       ├─── needs codebook (0.3) AND VFN (2.x) ───┐
       │                                           ▼
       │    3.1  Intent Router ────────────────── can run in parallel
       │    3.2  Certainty Calibration ────────── can run in parallel
       │    3.3  Safety Refinement ────────────── can run in parallel
       │                                           │
       │         (all three are independent)        │
       │                                           ▼
       │    4.1  End-to-End Fine-Tuning ────────── sequential
       │         │
       │         ▼
       │    4.2  RLVF Alignment ────────────────── sequential
       │         │
       │         ▼
       │    4.3  Sleep Consolidation ───────────── sequential
       │                                           │
       │                                           ▼
       │    5.1  VFN Scaling (50M→500M) ────────── sequential
       │         │
       │         ▼
       │    5.2a Multi-Lang: JavaScript ────────── parallel ─┐
       │    5.2b Multi-Lang: Java ──────────────── parallel ─┤
       │    5.2c Multi-Lang: Rust+Go ───────────── parallel ─┤
       │                                                     ▼
       │    5.3  Benchmark & Publish ───────────── final step
```

**Key insight:** Phase 2 (VFN training) does NOT need the codebook. The
VFN trains on raw TensorFrames produced by the encoder. The codebook is
only needed starting at Phase 3 for HNSW retrieval. This means **Phase 0.3
and Phase 2 can run on separate cloud instances simultaneously**. Phase 0.3
runs on a cheap CPU-only instance (32+ cores), while Phase 2 runs on a GPU
instance (H100). Phase 0.3 finishes in ~1.5 hrs on cloud — long before
Phase 2 is done.

---

## 3. Parallelization Strategy

### What can run at the same time

| Time Period | Cloud CPU A | Cloud GPU A | Cloud GPU B | Cloud GPU C |
|-------------|-------------|-------------|-------------|-------------|
| **Now** | **0.3 codebook (CPU)** | **2.1 VFN training** | — | — |
| **~2 hrs later** | 0.3 done → download codebook | 2.1 continues... | — | — |
| **After 2.x** | — | 3.1 Router | 3.2 Calibration | 3.3 Safety |
| **After 3.x** | — | 4.1 → 4.2 → 4.3 (sequential) | — | — |
| **After 4.x** | — | 5.1 VFN scaling | — | — |
| **After 5.1** | — | 5.2a JavaScript | 5.2b Java | 5.2c Rust+Go |
| **After 5.2** | — | 5.3 Benchmarks | — | — |

### Rules

1. **Phase 0.3 and Phase 2 are independent** — run on separate cloud instances simultaneously
2. **Phase 0.3 goes on a cheap CPU instance** — 32+ cores, no GPU needed, finishes in ~1.5 hrs
3. **Phase 3 tasks (3.1, 3.2, 3.3) are independent** — run on 3 GPUs in parallel
4. **Phase 4 tasks (4.1 → 4.2 → 4.3) are sequential** — each needs the previous output
5. **Phase 5.2 languages are independent** — run on separate GPUs in parallel
6. **Always sync checkpoints back** after each phase completes

---

## 4. Cloud Instance Recommendations

### Phase 2 (VFN Training — the bottleneck)

| Provider | Instance | GPU | VRAM | Price/hr | Est. Total |
|----------|----------|-----|------|----------|------------|
| Lambda Labs | gpu_1x_h100_sxm5 | 1× H100 SXM | 80GB | ~$2.49 | $250–500 |
| RunPod | H100 SXM | 1× H100 SXM | 80GB | ~$2.69 | $270–540 |
| Vast.ai | H100 | 1× H100 | 80GB | ~$2.00 | $200–400 |
| AWS | p5.xlarge | 1× H100 | 80GB | ~$4.20 | $420–840 |

**Minimum requirement:** 24GB VRAM (RTX 4090 works, H100 is faster).
Batch size 32 at BF16 with 51M params fits in ~8GB VRAM.

### Phase 3 (Light calibration)

Any GPU with ≥16GB VRAM. Even a single RTX 4060 Ti works. Cheapest option.

### Phase 5 (Multi-GPU scaling)

| Provider | Instance | GPUs | Price/hr | Est. Total |
|----------|----------|------|----------|------------|
| Lambda Labs | gpu_4x_h100 | 4× H100 | ~$9.96 | $1,000–1,900 |
| RunPod | 4× H100 | 4× H100 | ~$10.76 | $1,100–2,000 |

---

## 5. One-Time Setup: Build Environment

The training binaries are written in Rust and use the `candle` ML
framework (Rust-native, no Python dependency). You need to build the
project on the cloud instance.

### Step 1: Provision a Linux instance with GPU

Any Ubuntu 22.04+ with CUDA 12.x and ≥24GB VRAM.

### Step 2: Install Rust toolchain

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Verify
rustc --version   # needs 1.85+ for edition 2024
cargo --version
```

### Step 3: Install system dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev git

# Verify CUDA
nvidia-smi          # should show GPU
nvcc --version      # needs CUDA 12.x
```

### Step 4: Clone the repository

```bash
git clone https://github.com/Project4070/Volt-X.git ~/volt-x
cd ~/volt-x
```

### Step 5: Build the training binaries

```bash
# Build VFN training binary (includes GPU support via candle-cuda)
cargo build --release -p volt-learn --features vfn-training --bin train-vfn

# Build codebook-init (CPU only, for Phase 0.3 if needed)
cargo build --release -p volt-learn --bin codebook-init

# Verify binaries exist
ls -la target/release/train-vfn
ls -la target/release/codebook-init
```

**Build time:** ~5–10 minutes on a cloud instance (first build downloads
and compiles all dependencies).

### Step 6: Create directory structure

```bash
mkdir -p ~/volt-x/checkpoints
mkdir -p ~/volt-x/data
mkdir -p ~/volt-x/logs
```

---

## 6. File Transfer Manifest

### What to upload to every cloud instance

These are the **trained artifacts from Phase 1** that all later phases
need. They are small and must be uploaded before any training starts.

| File | Size | Source Path (local Windows) | Dest Path (cloud Linux) |
|------|------|---------------------------|------------------------|
| BPE tokenizer | 2.3 MB | `checkpoints\code_tokenizer.json` | `~/volt-x/checkpoints/code_tokenizer.json` |
| Trained encoder | 20.6 MB | `checkpoints\code_encoder.safetensors` | `~/volt-x/checkpoints/code_encoder.safetensors` |
| Trained decoder | 21.0 MB | `checkpoints\code_decoder.safetensors` | `~/volt-x/checkpoints/code_decoder.safetensors` |
| **Subtotal** | **~44 MB** | | |

### Upload command (from local Windows PowerShell)

```powershell
# Using SCP (replace USER@HOST with your cloud instance)
scp checkpoints\code_tokenizer.json USER@HOST:~/volt-x/checkpoints/
scp checkpoints\code_encoder.safetensors USER@HOST:~/volt-x/checkpoints/
scp checkpoints\code_decoder.safetensors USER@HOST:~/volt-x/checkpoints/
```

### Datasets per phase

Each phase needs different datasets. Only upload what is needed for the
current phase to save transfer time.

#### Phase 2: VFN Training

| Dataset | Size | Source Path (D: drive) | Format |
|---------|------|----------------------|--------|
| HumanEval | 0.2 MB | `D:\VoltData\phase0\humaneval\data\humaneval.jsonl` | JSONL |
| MBPP | 0.1 MB | `D:\VoltData\phase0\mbpp\data\mbpp.jsonl` | JSONL |
| Combined (HE+MBPP) | ~0.3 MB | `D:\VoltData\phase0\code_training_combined.jsonl` | JSONL |
| APPS (intro, train) | varies | `D:\VoltData\phase2\apps\data\apps_introductory_train.jsonl` | JSONL |
| APPS (interview, train) | varies | `D:\VoltData\phase2\apps\data\apps_interview_train.jsonl` | JSONL |
| APPS (competition, train) | varies | `D:\VoltData\phase2\apps\data\apps_competition_train.jsonl` | JSONL |
| **Total** | **~1.3 GB** | | |

```powershell
# Upload Phase 2 datasets
scp D:\VoltData\phase0\humaneval\data\humaneval.jsonl USER@HOST:~/volt-x/data/
scp D:\VoltData\phase0\mbpp\data\mbpp.jsonl USER@HOST:~/volt-x/data/
scp D:\VoltData\phase0\code_training_combined.jsonl USER@HOST:~/volt-x/data/
scp D:\VoltData\phase2\apps\data\apps_introductory_train.jsonl USER@HOST:~/volt-x/data/
scp D:\VoltData\phase2\apps\data\apps_interview_train.jsonl USER@HOST:~/volt-x/data/
scp D:\VoltData\phase2\apps\data\apps_competition_train.jsonl USER@HOST:~/volt-x/data/
```

#### Phase 0.3: Codebook k-means (on cloud CPU instance)

| Dataset | Size | Source Path (D: drive) |
|---------|------|----------------------|
| The Stack Python (sample) | ~43.8 GB | `D:\VoltData\phase0\the_stack_sample\python\python_sample.jsonl` |

**Transfer strategy:** The dataset is 44GB — uploading from local may be
slow depending on your upload speed. Two options:

**Option A — Upload from local** (if you have ≥50 Mbps upload):

```powershell
# From local Windows PowerShell — will take ~2 hrs at 50 Mbps
scp D:\VoltData\phase0\the_stack_sample\python\python_sample.jsonl USER@CPU_HOST:~/volt-x/data/
```

**Option B — Download directly on cloud** (recommended, much faster):

```bash
# On the cloud CPU instance — download from HuggingFace directly
pip install datasets huggingface_hub
python3 -c "
from datasets import load_dataset
import json

ds = load_dataset('bigcode/the-stack', data_dir='data/python',
                  split='train', streaming=True, trust_remote_code=True)

with open('data/python_sample.jsonl', 'w') as f:
    for i, row in enumerate(ds):
        if i >= 6_600_000:
            break
        f.write(json.dumps({'content': row['content']}) + '\n')
print(f'Wrote {i+1} files')
"
```

Cloud instances typically have 1–10 Gbps network — this downloads in
minutes instead of hours.

**Also upload Phase 1 checkpoints** (needed by the encoder):

```powershell
scp checkpoints\code_tokenizer.json USER@CPU_HOST:~/volt-x/checkpoints/
scp checkpoints\code_encoder.safetensors USER@CPU_HOST:~/volt-x/checkpoints/
scp checkpoints\code_decoder.safetensors USER@CPU_HOST:~/volt-x/checkpoints/
```

#### Phase 3: Calibration

| Dataset | Size | Notes |
|---------|------|-------|
| HumanEval + MBPP combined | 0.3 MB | Already uploaded for Phase 2 |
| APPS (intro/interview/competition) | varies | Already uploaded for Phase 2 |
| Routing labels | ~5 MB | Generated from above datasets (script TBD) |
| Safety examples | ~2 MB | Generated adversarial + benign examples (script TBD) |
| **New uploads** | **~7 MB** | |

#### Phase 4: Joint Alignment

| Dataset | Size | Notes |
|---------|------|-------|
| Same as Phase 2+3 | — | Already on cloud |
| Codebook | ~64 MB | `checkpoints/codebook_code.bin` (from Phase 0.3) |
| VFN checkpoint | ~200 MB | `checkpoints/scaled_vfn.safetensors` (from Phase 2) |
| **New uploads** | **~264 MB** | |

#### Phase 5: Scaling

| Dataset | Size | Source Path |
|---------|------|-------------|
| The Stack Python (expanded) | 43.8 GB+ | `D:\VoltData\phase0\the_stack_sample\` |
| MultiPL-E (15 languages) | 2.9 MB | `D:\VoltData\phase5\multiple\` |
| The Stack per-language | ~20 GB each | Download directly to cloud (faster) |
| **Total** | **~100+ GB** | |

**Recommendation for Phase 5:** Download The Stack directly on the cloud
instance using HuggingFace CLI, rather than uploading from local machine.

```bash
# On cloud instance — download The Stack per-language directly
pip install huggingface_hub datasets
python -c "
from datasets import load_dataset
ds = load_dataset('bigcode/the-stack', data_dir='data/javascript',
                  split='train', streaming=True, trust_remote_code=True)
# ... save to JSONL
"
```

---

## 7. Phase-by-Phase Execution Guide

---

### Phase 0.3: Codebook k-means (cloud CPU)

**RUN ON CLOUD** — offload to a CPU instance with 32+ cores.

**What it does:** Takes 100K Python files from The Stack, encodes each
through the trained CNN encoder to get 16×256-dim slot vectors, then
runs mini-batch k-means with k=65,536 to find 65,536 cluster centroids.
These centroids become the "codebook" used for HNSW nearest-neighbor
lookup later.

**Hardware:** CPU-only cloud instance with 32+ cores and ≥64GB RAM.
No GPU needed. Examples:

| Provider | Instance | Cores | RAM | Price/hr |
| -------- | -------- | ----- | --- | -------- |
| AWS | c7i.8xlarge | 32 vCPU | 64 GB | ~$1.36 |
| AWS | c7i.16xlarge | 64 vCPU | 128 GB | ~$2.72 |
| GCP | c3-standard-44 | 44 vCPU | 176 GB | ~$1.85 |
| Hetzner | CCX63 | 48 vCPU | 192 GB | ~$0.85 |

**Duration:** ~1.5 hrs on 32-core cloud (vs. 6+ hrs on local machine).
The k-means step uses `rayon` for parallel centroid updates — more cores
= faster.

**Output:** `checkpoints/codebook_code.bin` (~64 MB)

#### Step 1: Provision instance and build

```bash
# On the cloud CPU instance
git clone https://github.com/Project4070/Volt-X.git ~/volt-x
cd ~/volt-x

# Install Rust if not present
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Build codebook-init (CPU only, no CUDA needed)
# Use --features code-training to enable the learned encoder
cargo build --release -p volt-learn --features code-training --bin codebook-init
```

#### Step 2: Get the data onto the instance

**Option A — Download The Stack directly on cloud** (recommended):

```bash
pip install datasets huggingface_hub
python3 -c "
from datasets import load_dataset
import json

ds = load_dataset('bigcode/the-stack', data_dir='data/python',
                  split='train', streaming=True, trust_remote_code=True)

with open('data/python_sample.jsonl', 'w') as f:
    for i, row in enumerate(ds):
        if i >= 6_600_000:
            break
        f.write(json.dumps({'content': row['content']}) + '\n')
print(f'Wrote {i+1} files')
"
```

**Option B — Upload from local** (if you already have it downloaded):

```powershell
# From local Windows PowerShell
scp D:\VoltData\phase0\the_stack_sample\python\python_sample.jsonl USER@CPU_HOST:~/volt-x/data/
```

#### Step 3: Upload Phase 1 checkpoints (needed by the encoder)

```powershell
# From local Windows PowerShell
scp checkpoints\code_tokenizer.json USER@CPU_HOST:~/volt-x/checkpoints/
scp checkpoints\code_encoder.safetensors USER@CPU_HOST:~/volt-x/checkpoints/
scp checkpoints\code_decoder.safetensors USER@CPU_HOST:~/volt-x/checkpoints/
```

#### Step 4: Run codebook initialization

```bash
cd ~/volt-x

cargo run --release -p volt-learn --features code-training --bin codebook-init -- \
  --corpus "data/python_sample.jsonl" \
  --tokenizer "checkpoints/code_tokenizer.json" \
  --encoder "checkpoints/code_encoder.safetensors" \
  --decoder "checkpoints/code_decoder.safetensors" \
  --max-files 100000 --k 65536 \
  --output "checkpoints/codebook_code.bin"
```

**What to expect:**

- Progress logs every 10,000 files processed
- Encoding phase: ~30–45 min (100K files through CNN encoder)
- k-means phase: ~30–60 min (65,536 clusters on ~1.6M vectors)
- Total: ~1–1.5 hrs on a 32-core instance

#### Step 5: Download the codebook

```powershell
# From local Windows PowerShell — download the result (~64 MB)
scp USER@CPU_HOST:~/volt-x/checkpoints/codebook_code.bin checkpoints\
```

Also copy it to wherever Phase 3 will run (or upload to S3):

```bash
# On the CPU instance — upload to S3 for later phases
aws s3 cp checkpoints/codebook_code.bin s3://volt-x-training/checkpoints/
```

**After downloading the codebook, terminate the CPU instance** to stop
billing. This instance is only needed for ~2 hours total.

---

### Phase 2: VFN Flow Matching (GPU required)

**CAN START IMMEDIATELY** — no dependency on Phase 0.3.

**What it does:** Trains the Scaled VFN (51 million parameters) to learn
a velocity field that drives TensorFrames from "problem description" toward
"working code solution". This is the core intelligence of the system.
Think of it as teaching the system: "given a description of what code
should do, drift toward the correct code."

**Hardware:** 1× GPU with ≥24GB VRAM (H100 recommended, RTX 4090 works)

**Duration:** 100–200 H100-hours (2–4 weeks wall-clock on 1× H100)

#### Files needed on cloud

| File | Purpose |
|------|---------|
| `checkpoints/code_tokenizer.json` | Tokenize input text |
| `checkpoints/code_encoder.safetensors` | Encode text → TensorFrame |
| `checkpoints/code_decoder.safetensors` | Decode TensorFrame → text (for validation) |
| `data/code_training_combined.jsonl` | HumanEval + MBPP combined (Stage 1) |
| `data/apps_introductory_train.jsonl` | APPS easy problems (Stage 2) |
| `data/apps_interview_train.jsonl` | APPS medium problems (Stage 2-3) |
| `data/apps_competition_train.jsonl` | APPS hard problems (Stage 3) |

#### Run command

```bash
cd ~/volt-x

# Stage 1: Simple problems (HumanEval + MBPP combined)
cargo run --release -p volt-learn --features vfn-training --bin train-vfn -- \
  --data "data/code_training_combined.jsonl" \
  --tokenizer "checkpoints/code_tokenizer.json" \
  --encoder "checkpoints/code_encoder.safetensors" \
  --decoder "checkpoints/code_decoder.safetensors" \
  --output "checkpoints/scaled_vfn_stage1.safetensors" \
  --epochs 10 --batch-size 32 --lr 1e-4 --warmup 2000 \
  --hidden-dim 2048 --num-blocks 6 \
  --device cuda

# Stage 2: Add APPS introductory + interview problems
#   (Curriculum training: start simple, add harder problems)
cargo run --release -p volt-learn --features vfn-training --bin train-vfn -- \
  --data "data/code_training_combined.jsonl" \
  --data "data/apps_introductory_train.jsonl" \
  --data "data/apps_interview_train.jsonl" \
  --tokenizer "checkpoints/code_tokenizer.json" \
  --encoder "checkpoints/code_encoder.safetensors" \
  --decoder "checkpoints/code_decoder.safetensors" \
  --output "checkpoints/scaled_vfn_stage2.safetensors" \
  --epochs 10 --batch-size 32 --lr 5e-5 --warmup 1000 \
  --hidden-dim 2048 --num-blocks 6 \
  --device cuda

# Stage 3: Add APPS competition problems (hardest)
cargo run --release -p volt-learn --features vfn-training --bin train-vfn -- \
  --data "data/code_training_combined.jsonl" \
  --data "data/apps_introductory_train.jsonl" \
  --data "data/apps_interview_train.jsonl" \
  --data "data/apps_competition_train.jsonl" \
  --tokenizer "checkpoints/code_tokenizer.json" \
  --encoder "checkpoints/code_encoder.safetensors" \
  --decoder "checkpoints/code_decoder.safetensors" \
  --output "checkpoints/scaled_vfn.safetensors" \
  --epochs 10 --batch-size 32 --lr 2e-5 --warmup 500 \
  --hidden-dim 2048 --num-blocks 6 \
  --device cuda
```

#### What to expect

- Each epoch produces a checkpoint: `scaled_vfn_epoch_1.safetensors`, etc.
- Training logs print loss every 50 steps
- Final output: `checkpoints/scaled_vfn.safetensors` (~200 MB)

#### After completion

Download the trained VFN back to your local machine:

```bash
# From local Windows PowerShell
scp USER@HOST:~/volt-x/checkpoints/scaled_vfn.safetensors checkpoints\
scp USER@HOST:~/volt-x/checkpoints/scaled_vfn_stage1.safetensors checkpoints\
```

---

### Phase 3: Hard Core Calibration (light GPU)

**Requires:** Phase 0.3 codebook + Phase 2 VFN checkpoint (both must be done)

**What it does:** Three independent calibration tasks that tune the
system's routing, confidence scoring, and safety checks for code tasks.

**Hardware:** 1× GPU with ≥16GB VRAM per task (cheap instances work)

**Duration:** ~10 H100-hours total, but **all three run in parallel**

#### 3.1 — Intent Router Calibration

Teaches the system to route incoming queries to the right handler
(e.g., "write a function" → CodeRunner, "fix this bug" → CodeDebugger).

```
Cloud GPU A:
  Input:  10K labeled routing queries (generated from datasets)
  Output: Calibrated strand routing vectors
  Time:   1–2 hours
```

#### 3.2 — Certainty (γ) Calibration

Makes the system's confidence scores meaningful. After calibration,
when the system says "I'm 80% sure this code is correct", it should
actually be correct ~80% of the time.

```
Cloud GPU B:
  Input:  VFN checkpoint + 2,700 problems with test suites
  Output: Calibration curve (maps raw γ → actual pass probability)
  Time:   5–10 hours
```

#### 3.3 — Safety Axiom Refinement

Adjusts the safety system to catch malicious code (file deletion,
data exfiltration) while not blocking legitimate code.

```
Cloud GPU C:
  Input:  2,638 labeled examples (1,138 benign + 1,500 adversarial)
  Output: Refined safety axiom vectors
  Time:   3–5 hours
```

#### Parallelization

All three tasks are **completely independent**. Spin up 3 cheap GPU
instances, run one task per instance, all finish within ~10 hours
wall-clock instead of ~20 hours sequential.

**Files needed (all three tasks):**

| File | Purpose | Size |
|------|---------|------|
| `checkpoints/scaled_vfn.safetensors` | Trained VFN (from Phase 2) | ~200 MB |
| `checkpoints/codebook_code.bin` | Codebook (from Phase 0.3) | ~64 MB |
| `checkpoints/code_tokenizer.json` | Tokenizer | 2.3 MB |
| `checkpoints/code_encoder.safetensors` | Encoder | 20.6 MB |
| `checkpoints/code_decoder.safetensors` | Decoder | 21.0 MB |
| `data/code_training_combined.jsonl` | HumanEval + MBPP combined | 0.3 MB |
| `data/apps_introductory_train.jsonl` | APPS easy problems | varies |
| `data/apps_interview_train.jsonl` | APPS medium problems | varies |
| `data/apps_competition_train.jsonl` | APPS hard problems | varies |
| Phase 3 labeled data (generated) | Routing/safety labels | ~10 MB |

**Note:** Phase 3 training binaries are not yet implemented in code.
They will need to be written before this phase can run. The architecture
and approach are defined in `CODE_TRAINING_PLAN.md` sections 3.1–3.3.

---

### Phase 4: Joint Alignment (GPU required)

**Requires:** All of Phase 2 + Phase 3 complete

**What it does:** Fine-tunes the entire pipeline end-to-end. Instead of
training components in isolation, this phase runs the full loop:
text → encode → VFN inference → decode → execute code → check test
results → backpropagate. It's like a dress rehearsal with real feedback.

**Hardware:** 1–2× H100 (more VRAM helps for larger batches)

**Duration:** 130–250 H100-hours (sequential: 4.1 → 4.2 → 4.3)

#### 4.1 — End-to-End Fine-Tuning (75–150 hrs)

```bash
# Inputs: All checkpoints from Phase 1-3 + datasets
# Output: Fine-tuned VFN + attention weights
# Training signal: 70% flow matching loss + 30% test pass rate (REINFORCE)
```

#### 4.2 — RLVF Alignment (40–75 hrs)

```bash
# Inputs: Fine-tuned model from 4.1 + APPS problems
# Output: RLVF-aligned VFN (calibrated confidence, fewer overconfident errors)
# Training signal: Shaped rewards based on test outcomes + γ accuracy
```

#### 4.3 — Sleep Consolidation Validation (15–25 hrs)

```bash
# Inputs: RLVF model from 4.2 + 5K diverse APPS problems
# Output: Post-sleep model with consolidated patterns
# Validates: Sleep improves performance, no catastrophic forgetting
```

**Files needed:**

All Phase 2+3 files, plus:

| File | Purpose | Size |
|------|---------|------|
| All Phase 3 outputs | Calibrated router, safety, γ | < 10 MB |
| Phase 2 VFN | Starting weights | ~200 MB |

**Note:** Phase 4 training binaries are not yet implemented. They will
extend the train-vfn pipeline with REINFORCE gradients from test
execution in a sandboxed environment.

---

### Phase 5: Scale & Benchmark (multi-GPU)

**Requires:** Phase 4 complete

**What it does:** Scales the VFN from 51M to 500M parameters, adds
support for multiple programming languages, and runs rigorous
benchmarks to measure performance.

**Hardware:** 4–8× H100 for scaling; individual GPUs for per-language training

**Duration:** 650–1,250 H100-hours total

#### 5.1 — VFN Scaling (400–750 hrs)

Progressively grows the VFN: 51M → 200M → 500M parameters using
knowledge distillation (smaller model teaches larger model).

```
Instances: 1× multi-GPU node (4× H100, 320GB total VRAM)
Strategy:  Data-parallel training with gradient accumulation
```

#### 5.2 — Multi-Language Training (150–300 hrs)

Extends from Python-only to JavaScript, Java, Rust, Go.

**This is embarrassingly parallel** — each language trains independently:

```
Cloud GPU A: 5.2a — JavaScript (The Stack JS subset, 20GB)
Cloud GPU B: 5.2b — Java (The Stack Java subset, 20GB)
Cloud GPU C: 5.2c — Rust + Go (The Stack Rust+Go subsets, 10GB each)

All three can run simultaneously.
Wall-clock: ~100 hrs (instead of 300 hrs sequential)
```

**Download data directly on cloud** (faster than uploading from local):

```bash
# On each cloud instance — download language-specific data
pip install datasets huggingface_hub
python3 -c "
from datasets import load_dataset
ds = load_dataset('bigcode/the-stack', data_dir='data/javascript',
                  split='train', streaming=True, trust_remote_code=True)
# Process and save to JSONL...
"
```

#### 5.3 — Benchmark Publication (100–200 hrs)

Run inference on benchmark suites (HumanEval, MBPP, APPS, MultiPL-E,
CLRS) and compare against transformer baselines.

---

## 8. Checkpoint Sync Workflow

Every time a training phase completes, sync the output checkpoint back
to your local machine AND to cloud storage for the next phase.

### Option A: Direct SCP (simplest)

```bash
# After Phase 2 completes on cloud:
# From local Windows PowerShell:
scp USER@CLOUD:~/volt-x/checkpoints/scaled_vfn.safetensors checkpoints\

# Before Phase 3 starts on cloud (possibly different instance):
scp checkpoints\scaled_vfn.safetensors USER@CLOUD3:~/volt-x/checkpoints/
scp checkpoints\codebook_code.bin USER@CLOUD3:~/volt-x/checkpoints/
```

### Option B: Cloud object storage (recommended for multi-instance)

Use an S3-compatible bucket as a central checkpoint store:

```bash
# Upload after each phase (on cloud instance)
aws s3 cp checkpoints/scaled_vfn.safetensors s3://volt-x-training/checkpoints/

# Download before next phase (on new cloud instance)
aws s3 cp s3://volt-x-training/checkpoints/ checkpoints/ --recursive
```

### Checkpoint sizes

| Checkpoint | Size | Produced by |
|-----------|------|-------------|
| `code_tokenizer.json` | 2.3 MB | Phase 1.1 (done) |
| `code_encoder.safetensors` | 20.6 MB | Phase 1.1 (done) |
| `code_decoder.safetensors` | 21.0 MB | Phase 1.2 (done) |
| `codebook_code.bin` | ~64 MB | Phase 0.3 (running) |
| `scaled_vfn.safetensors` | ~200 MB | Phase 2 |
| Phase 3 calibration files | < 10 MB | Phase 3 |
| Phase 4 fine-tuned VFN | ~200 MB | Phase 4 |
| 500M VFN (Phase 5) | ~2 GB | Phase 5.1 |
| Per-language decoders | ~40 MB each | Phase 5.2 |

**Total checkpoint storage:** < 3 GB across all phases.

---

## 9. Cost Estimates

### Optimistic scenario (efficient parallelization)

| Phase | Instances | Hours/Instance | Rate/hr | Cost |
|-------|-----------|----------------|---------|------|
| Phase 0.3 | 1× 32-core CPU | ~2 | $1.36 | $3 |
| Phase 2 | 1× H100 | ~150 | $2.50 | $375 |
| Phase 3 | 3× RTX 4090 | ~8 each | $0.50 | $12 |
| Phase 4 | 1× H100 | ~180 | $2.50 | $450 |
| Phase 5.1 | 4× H100 node | ~120 | $10.00 | $1,200 |
| Phase 5.2 | 3× H100 | ~80 each | $2.50 | $600 |
| Phase 5.3 | 1× H100 | ~100 | $2.50 | $250 |
| **Total** | | | | **~$2,893** |

### Conservative scenario (some idle time, retries)

| Phase | Cost |
|-------|------|
| Phase 0.3 | $5 |
| Phase 2 | $500 |
| Phase 3 | $25 |
| Phase 4 | $625 |
| Phase 5 | $3,350 |
| **Total** | **~$4,505** |

### Money-saving tips

1. **Use spot/interruptible instances** for Phase 2 and 5 (50–70% cheaper).
   The training has per-epoch checkpointing, so interruptions only lose
   the current epoch (~20 min of work).

2. **Use Vast.ai or RunPod** instead of AWS/GCP — typically 40–60% cheaper
   for the same hardware.

3. **Start with Phase 2 only** to validate the approach before committing
   to Phase 5 scaling costs.

4. **Phase 3 is cheap** — use the cheapest GPU instances available.

---

## 10. Timeline: Serial vs. Parallel

### Serial execution (one thing at a time, all local)

```
Week 1-2:   Phase 0.3 finishes locally, prep
Week 2-6:   Phase 2 VFN training
Week 6-9:   Phase 3 calibration
Week 9-15:  Phase 4 alignment
Week 15-23: Phase 5 scaling
────────────────────────────────
Total: ~23 weeks (~5.5 months)
```

### Parallel execution (cloud offloading)

```
Day 1:      Spin up Cloud CPU A (0.3) + Cloud GPU A (2.1) simultaneously
Day 1:      Phase 0.3 done in ~2 hrs → terminate CPU instance
Week 1-4:   Phase 2 continues on Cloud GPU A
Week 4-5:   Phase 3 (3 tasks on 3 cheap GPUs in parallel, ~1 week)
Week 5-8:   Phase 4 (sequential on Cloud GPU A)
Week 8-11:  Phase 5.1 scaling (4× H100 node)
Week 11-13: Phase 5.2 (3 languages on 3 GPUs in parallel, ~2 weeks)
Week 13-14: Phase 5.3 benchmarks
────────────────────────────────
Total: ~14 weeks (~3.5 months)
```

### Time saved: ~9 weeks (40% reduction)

The biggest wins come from:

1. **Phase 0.3 on cloud CPU** finishes in ~2 hrs instead of 6+ hrs locally
2. **Parallelizing Phase 3** across 3 GPUs (saves ~2 weeks)
3. **Parallelizing Phase 5.2** languages (saves ~4 weeks)
4. **Faster GPU hardware** in cloud vs. local (saves ~2 weeks)

---

## 11. Troubleshooting

### "CUDA not available" error

The `candle` framework needs CUDA at compile time. Make sure:
```bash
# Check CUDA is installed
nvcc --version        # Should show CUDA 12.x
echo $CUDA_HOME       # Should point to CUDA installation

# If not set:
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Rebuild with CUDA
cargo build --release -p volt-learn --features vfn-training --bin train-vfn
```

### Out of memory (OOM) on GPU

Reduce batch size:
```bash
# Instead of --batch-size 32, try:
--batch-size 16    # halves VRAM usage
--batch-size 8     # quarters VRAM usage
```

### Build fails: "edition 2024 not supported"

You need Rust 1.85+:
```bash
rustup update stable
rustc --version   # Must be 1.85.0 or newer
```

### Training loss not decreasing

- Check that checkpoints loaded correctly (encoder/decoder/tokenizer)
- Try lower learning rate: `--lr 5e-5` instead of `1e-4`
- Verify dataset files are valid JSONL with `head -1 data/code_training_combined.jsonl`

### Checkpoint file too large to SCP

Use compression:
```bash
# On cloud
gzip -k checkpoints/scaled_vfn.safetensors
# Transfer the .gz file, decompress locally
```

### Instance preempted (spot instance interrupted)

The training saves checkpoints every epoch. Find the latest:
```bash
ls -lt checkpoints/scaled_vfn_epoch_*.safetensors
# Resume from latest epoch by running the same command again
# (the binary starts fresh, but you can copy the latest epoch
#  checkpoint to the starting position)
```

---

## Quick Reference: Commands Cheat Sheet

```bash
# === UPLOAD (from local Windows PowerShell) ===

# Phase 1 checkpoints (needed by all phases)
scp checkpoints\code_tokenizer.json USER@HOST:~/volt-x/checkpoints/
scp checkpoints\code_encoder.safetensors USER@HOST:~/volt-x/checkpoints/
scp checkpoints\code_decoder.safetensors USER@HOST:~/volt-x/checkpoints/

# Phase 2 datasets
scp D:\VoltData\phase0\humaneval\data\humaneval.jsonl USER@HOST:~/volt-x/data/
scp D:\VoltData\phase0\mbpp\data\mbpp.jsonl USER@HOST:~/volt-x/data/
scp D:\VoltData\phase0\code_training_combined.jsonl USER@HOST:~/volt-x/data/
scp D:\VoltData\phase2\apps\data\apps_introductory_train.jsonl USER@HOST:~/volt-x/data/
scp D:\VoltData\phase2\apps\data\apps_interview_train.jsonl USER@HOST:~/volt-x/data/
scp D:\VoltData\phase2\apps\data\apps_competition_train.jsonl USER@HOST:~/volt-x/data/

# === BUILD ON CLOUD CPU INSTANCE (Phase 0.3) ===

cd ~/volt-x
cargo build --release -p volt-learn --features code-training --bin codebook-init


# === TRAIN PHASE 0.3 (on cloud CPU — run in parallel with Phase 2) ===

cargo run --release -p volt-learn --features code-training --bin codebook-init -- \
  --corpus "data/python_sample.jsonl" \
  --tokenizer "checkpoints/code_tokenizer.json" \
  --encoder "checkpoints/code_encoder.safetensors" \
  --decoder "checkpoints/code_decoder.safetensors" \
  --max-files 100000 --k 65536 \
  --output "checkpoints/codebook_code.bin"


# === BUILD ON CLOUD GPU INSTANCE (Phase 2) ===

cd ~/volt-x
cargo build --release -p volt-learn --features vfn-training --bin train-vfn


# === TRAIN PHASE 2 (on cloud GPU — run in parallel with Phase 0.3) ===

cargo run --release -p volt-learn --features vfn-training --bin train-vfn -- \
  --data "data/code_training_combined.jsonl" \
  --data "data/apps_introductory_train.jsonl" \
  --data "data/apps_interview_train.jsonl" \
  --tokenizer "checkpoints/code_tokenizer.json" \
  --encoder "checkpoints/code_encoder.safetensors" \
  --decoder "checkpoints/code_decoder.safetensors" \
  --output "checkpoints/scaled_vfn.safetensors" \
  --epochs 10 --batch-size 32 --lr 1e-4 --warmup 2000 \
  --hidden-dim 2048 --num-blocks 6 \
  --device cuda


# === DOWNLOAD RESULTS (from local Windows PowerShell) ===

# Codebook from CPU instance (~64 MB, available after ~2 hrs)
scp USER@CPU_HOST:~/volt-x/checkpoints/codebook_code.bin checkpoints\

# VFN from GPU instance (~200 MB, available after weeks)
scp USER@GPU_HOST:~/volt-x/checkpoints/scaled_vfn.safetensors checkpoints\
```

---

*This plan covers the full training pipeline from current state to
benchmark publication. Start Phase 0.3 on a cloud CPU and Phase 2 on a
cloud GPU simultaneously — they have no dependency on each other. Phase
0.3 finishes in ~2 hours; Phase 2 is the critical path at 2–4 weeks.*
