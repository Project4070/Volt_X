# Vast.ai Quick-Start Guide — Phase 0.3 + Phase 2.1

**Instance:** Single RTX 5090 (32 GB) + AMD EPYC Rome (192 cores), $0.20/hr
**Goal:** Minimize billable prep time. Every minute the instance is on costs money.
**Strategy:** Pre-compress and upload everything from local PC (good ethernet),
then build while extractions run in parallel on the instance.

**Estimated prep time:** ~30–60 minutes from SSH to training started (dominated
by uploading ~35 GB of compressed Stack data over your upstream connection).
**Estimated training cost:** $60–100 (300–500 GPU-hours for Phase 2.1).

---

## Pre-Flight Checklist (Do This BEFORE Renting)

Everything in this section is done on your local Windows machine while you
are NOT paying for cloud time.

### 1. Stage upload files into one directory

Create a staging directory so uploads are organized. Small files go into
`VoltStaging` (checkpoints + Phase 2 datasets). The two curated Stack
datasets are compressed separately for parallel upload.

```powershell
# Create staging directory
mkdir C:\VoltStaging
mkdir C:\VoltStaging\checkpoints
mkdir C:\VoltStaging\data

# Copy checkpoints (~44 MB total)
copy c:\Volt_X\checkpoints\code_tokenizer.json    C:\VoltStaging\checkpoints\
copy c:\Volt_X\checkpoints\code_encoder.safetensors C:\VoltStaging\checkpoints\
copy c:\Volt_X\checkpoints\code_decoder.safetensors C:\VoltStaging\checkpoints\

# Copy Phase 2 datasets (~1.3 GB total)
copy D:\VoltData\phase0\code_training_combined.jsonl          C:\VoltStaging\data\
copy D:\VoltData\phase0\humaneval\data\humaneval.jsonl        C:\VoltStaging\data\
copy D:\VoltData\phase0\mbpp\data\mbpp.jsonl                  C:\VoltStaging\data\
copy D:\VoltData\phase2\apps\data\apps_introductory_train.jsonl  C:\VoltStaging\data\
copy D:\VoltData\phase2\apps\data\apps_interview_train.jsonl     C:\VoltStaging\data\
copy D:\VoltData\phase2\apps\data\apps_competition_train.jsonl   C:\VoltStaging\data\
```

### 2. Compress the staging directory + Stack datasets

Three archives, uploaded in parallel:

- **VoltStaging.tar.gz** — checkpoints + Phase 2 data (~200–400 MB)
- **stack_python.tar.gz** — Phase 0 Python Stack, 44 GB → ~9–11 GB compressed
- **stack_multilang.tar.gz** — Phase 5 multilang Stack, 98 GB → ~20–25 GB compressed

```powershell
# ── Archive 1: Small files (fast, <1 min) ──
tar -czf C:\VoltStaging.tar.gz -C C:\ VoltStaging
# Result: C:\VoltStaging.tar.gz (~200-400 MB)
```

Archives 2 and 3 use `tar` for correct relative paths, then 7-Zip for
fast multi-threaded gzip compression. Run each in its own PowerShell
window to parallelize.

```powershell
# ── Archive 2: Phase 0 Python Stack (~44 GB → ~9-11 GB) ──
# Step 1: tar with -C to store just "python_sample.jsonl" (no full path)
tar -cf C:\stack_python.tar -C "D:\VoltData\phase0\the_stack_sample\python" python_sample.jsonl
# Step 2: compress with 7-Zip multi-threaded (~5-15 min)
& "C:\Program Files\7-Zip\7z.exe" a -tgzip -mx3 -mmt=on C:\stack_python.tar.gz C:\stack_python.tar
del C:\stack_python.tar
```

```powershell
# ── Archive 3: Phase 5 Multilang Stack (~98 GB → ~20-25 GB) ──
# Step 1: tar with -C to store "go/go_sample.jsonl", "java/java_sample.jsonl", etc.
tar -cf C:\stack_multilang.tar -C "D:\VoltData\phase5\the_stack_multilang" go/go_sample.jsonl java/java_sample.jsonl javascript/javascript_sample.jsonl python/python_sample.jsonl rust/rust_sample.jsonl
# Step 2: compress with 7-Zip multi-threaded (~10-30 min)
& "C:\Program Files\7-Zip\7z.exe" a -tgzip -mx3 -mmt=on C:\stack_multilang.tar.gz C:\stack_multilang.tar
del C:\stack_multilang.tar
```

> **Tip:** `-mx3` is "fast" compression — good ratio vs speed for JSONL
> text. `-mmt=on` uses all CPU cores. The uncompressed `tar` step is
> I/O-bound (just copies bytes), the `7z` step is CPU-bound.
>
> **Alternative:** If you have very fast ethernet (2.5+ Gbps) and want
> to skip compression time, upload the raw JSONL files directly via scp.
> The upload will be ~142 GB instead of ~33 GB but avoids the compression
> step entirely.

### 3. Verify files

```powershell
# Small staging archive — 9 files (3 checkpoints + 6 datasets)
dir C:\VoltStaging\checkpoints\   # 3 files: tokenizer, encoder, decoder
dir C:\VoltStaging\data\          # 6 files: combined, humaneval, mbpp, 3x apps

# Stack archives
dir C:\stack_python.tar.gz        # ~9-11 GB
dir C:\stack_multilang.tar.gz     # ~20-25 GB
```

### 4. Save this setup script locally

Create `C:\VoltStaging\cloud_setup.sh` with the following content.
This is the single script you will run on the cloud instance.

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/5] Installing Rust toolchain ==="
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
echo "Rust version: $(rustc --version)"

echo "=== [2/5] Installing system dependencies ==="
apt-get update -qq && apt-get install -y -qq build-essential pkg-config libssl-dev git > /dev/null 2>&1
echo "System deps installed."

echo "=== [3/5] Cloning repository ==="
git clone --depth 1 https://github.com/Project4070/Volt-X.git ~/volt-x
echo "Repo cloned."

echo "=== [4/5] Creating directory structure ==="
mkdir -p ~/volt-x/checkpoints ~/volt-x/data ~/volt-x/logs

echo "=== [5/5] Building training binaries (this takes ~5-10 min) ==="
cd ~/volt-x
# Single build compiles both binaries.
# vfn-training implies code-training, so codebook-init gets its deps too.
cargo build --release -p volt-learn --features vfn-training \
  --bin codebook-init --bin train-vfn 2>&1 | tail -5

echo ""
echo "============================================"
echo "  BUILD COMPLETE"
echo "  Binaries:"
ls -lh target/release/codebook-init target/release/train-vfn
echo "============================================"
```

---

## Step 1: Rent the Instance on Vast.ai

### Instance selection criteria

On https://vast.ai/search, filter for:

| Filter | Value |
|--------|-------|
| GPU | RTX 5090 × 1 |
| CPU cores | 128+ (EPYC Rome 192 preferred) |
| RAM | 64 GB+ |
| Disk | 250 GB+ free (need ~180 GB for both Stack datasets + archives) |
| CUDA | 12.8+ (required for Blackwell/RTX 5090) |
| Image | `nvidia/cuda:12.8.0-devel-ubuntu22.04` or similar with CUDA dev tools |
| Price | ≤$0.20/hr |
| Reliability | ≥95% (check host rating) |

**Image choice matters.** Pick a Docker image that already has CUDA
development tools (nvcc, headers). The `-devel` tag includes them.
If you pick a `-runtime` image, the Rust build will fail because candle
needs CUDA headers at compile time.

### Rent and note your connection details

After renting, Vast.ai gives you SSH connection info:
```
ssh -p <PORT> root@<HOST>
```

Save `HOST` and `PORT` — you need them for the next steps.

---

## Step 2: Connect and Run Parallel Setup

**CLOCK IS TICKING.** Open **four** local terminals (PowerShell).
All four run in parallel to minimize wall time.

### Terminal 1 — Upload small files + setup script, start build

```powershell
# Upload the compressed staging archive and setup script (~200-400 MB, fast)
# Replace <PORT> and <HOST> with your Vast.ai instance details
scp -P <PORT> C:\VoltStaging.tar.gz root@<HOST>:/root/
scp -P <PORT> C:\VoltStaging\cloud_setup.sh root@<HOST>:/root/
```

Then SSH in, extract, and start the build:

```powershell
ssh -p <PORT> root@<HOST>
```

```bash
# On the instance — extract small files, then build
cd /root
tar -xzf VoltStaging.tar.gz

# Start the build immediately (compiles while Stack uploads)
chmod +x /root/cloud_setup.sh
/root/cloud_setup.sh

# After build completes (~5-10 min), move small files into place:
cp /root/VoltStaging/checkpoints/* ~/volt-x/checkpoints/
cp /root/VoltStaging/data/* ~/volt-x/data/

echo "Small files in place:"
ls -lh ~/volt-x/checkpoints/
ls -lh ~/volt-x/data/
```

### Terminal 2 — Upload Phase 0 Python Stack (~9-11 GB compressed)

Start this immediately — don't wait for the build.

```powershell
# Uploads the Phase 0 Python-only Stack (~9-11 GB compressed → 44 GB)
scp -P <PORT> C:\stack_python.tar.gz root@<HOST>:/root/
```

Once the upload finishes, SSH in and extract:

```powershell
ssh -p <PORT> root@<HOST>
```

```bash
# Extract Phase 0 Stack into the data directory
mkdir -p ~/volt-x/data
cd ~/volt-x/data
tar -xzf /root/stack_python.tar.gz
# Result: ~/volt-x/data/python_sample.jsonl (~44 GB)
ls -lh python_sample.jsonl

# Clean up the archive to free disk space
rm /root/stack_python.tar.gz
```

### Terminal 3 — Upload Phase 5 Multilang Stack (~20-25 GB compressed)

Start this immediately — don't wait for the build.

```powershell
# Uploads the Phase 5 multilang Stack (~20-25 GB compressed → 98 GB)
scp -P <PORT> C:\stack_multilang.tar.gz root@<HOST>:/root/
```

Once the upload finishes, SSH in and extract:

```powershell
ssh -p <PORT> root@<HOST>
```

```bash
# Extract Phase 5 multilang Stack into a subdirectory
mkdir -p ~/volt-x/data/multilang
cd ~/volt-x/data/multilang
tar -xzf /root/stack_multilang.tar.gz
# Result: 5 language dirs in ~/volt-x/data/multilang/
#   go/go_sample.jsonl, java/java_sample.jsonl, javascript/javascript_sample.jsonl,
#   python/python_sample.jsonl, rust/rust_sample.jsonl
ls -lh */*.jsonl

# Clean up the archive to free disk space
rm /root/stack_multilang.tar.gz
```

### Terminal 4 — (Optional) Monitor progress

```powershell
ssh -p <PORT> root@<HOST>
```

```bash
# Watch disk usage as uploads arrive and extract
watch -n 10 'df -h / && echo "---" && ls -lh /root/*.tar.gz 2>/dev/null && echo "---" && du -sh ~/volt-x/data/* 2>/dev/null'
```

### What's happening in parallel

```text
Time    Terminal 1 (build)        Terminal 2 (Stack Python)   Terminal 3 (Stack Multilang)
─────   ─────────────────────     ──────────────────────      ───────────────────────────
0:00    Uploading small .tar.gz   scp stack_python.tar.gz     scp stack_multilang.tar.gz
0:01    SSH + extract + build     Uploading (~9-11 GB)...     Uploading (~20-25 GB)...
0:05    cargo build (compiling)   Uploading...                Uploading...
0:10    BUILD COMPLETE            Upload done, extracting     Uploading...
0:11    cp small files in place   Extract done (44 GB)        Uploading...
0:15    (ready)                   (done)                      Upload done, extracting
0:20    (ready)                   (done)                      Extract done (98 GB)
─────
~20-30 min total (depends on upload speed)
```

**Bottleneck:** The longest task is uploading `stack_multilang.tar.gz`
(~20–25 GB). At 500 Mbps upstream that's ~6 min; at 100 Mbps ~28 min.
The Rust build and Phase 0 Stack upload finish well before that.

---

## Step 3: Verify Everything Before Launching

Before starting training, run these checks. Takes 30 seconds.

```bash
# On the instance
cd ~/volt-x

# 1. Binaries exist?
ls -lh target/release/codebook-init target/release/train-vfn

# 2. GPU visible?
nvidia-smi

# 3. CUDA works?
nvcc --version

# 4. Checkpoints present? (should be 3 files, ~44 MB total)
ls -lh checkpoints/
# Expected:
#   code_tokenizer.json       (~2.3 MB)
#   code_encoder.safetensors  (~20.6 MB)
#   code_decoder.safetensors  (~21.0 MB)

# 5. Phase 2 data present? (should be 6 files)
ls -lh data/code_training_combined.jsonl
ls -lh data/humaneval.jsonl data/mbpp.jsonl
ls -lh data/apps_introductory_train.jsonl
ls -lh data/apps_interview_train.jsonl
ls -lh data/apps_competition_train.jsonl

# 6. Phase 0 Stack data present? (should be ~44 GB)
ls -lh data/python_sample.jsonl

# 7. Phase 5 multilang Stack present? (5 lang dirs, ~98 GB total)
ls -lh data/multilang/*/*.jsonl
# Expected:
#   go/go_sample.jsonl               (~19 GB)
#   java/java_sample.jsonl           (~24 GB)
#   javascript/javascript_sample.jsonl (~24 GB)
#   python/python_sample.jsonl       (~24 GB)
#   rust/rust_sample.jsonl           (~7.2 GB)

echo "All checks passed. Ready to train."
```

If any check fails, fix it before proceeding. Do NOT start training
with missing files — it will fail partway through and waste GPU-hours.

---

## Step 4: Launch Both Phases in tmux

Use `tmux` so training survives SSH disconnections. Both phases run
simultaneously — Phase 0.3 on CPU, Phase 2.1 on GPU.

```bash
# Install tmux if not present
apt-get install -y tmux

# Create a tmux session with two panes
tmux new-session -d -s training

# ── Pane 0: Phase 0.3 (CPU — codebook k-means) ──
tmux send-keys -t training "cd ~/volt-x && cargo run --release -p volt-learn --features code-training --bin codebook-init -- \
  --corpus data/python_sample.jsonl \
  --tokenizer checkpoints/code_tokenizer.json \
  --encoder checkpoints/code_encoder.safetensors \
  --decoder checkpoints/code_decoder.safetensors \
  --max-files 100000 --k 65536 \
  --output checkpoints/codebook_code.bin \
  2>&1 | tee logs/phase03.log" Enter

# ── Pane 1: Phase 2.1 Stage 1 (GPU — VFN flow matching) ──
tmux split-window -v -t training

tmux send-keys -t training "cd ~/volt-x && cargo run --release -p volt-learn --features vfn-training --bin train-vfn -- \
  --data data/code_training_combined.jsonl \
  --tokenizer checkpoints/code_tokenizer.json \
  --encoder checkpoints/code_encoder.safetensors \
  --decoder checkpoints/code_decoder.safetensors \
  --output checkpoints/scaled_vfn_stage1.safetensors \
  --epochs 10 --batch-size 32 --lr 1e-4 --warmup 2000 \
  --hidden-dim 2048 --num-blocks 6 \
  --device cuda \
  2>&1 | tee logs/phase21_stage1.log" Enter

# Attach to watch both
tmux attach -t training
```

### tmux controls

| Key | Action |
|-----|--------|
| `Ctrl+B` then `↑`/`↓` | Switch between panes |
| `Ctrl+B` then `D` | Detach (training continues in background) |
| `tmux attach -t training` | Re-attach later |

**Phase 0.3 finishes in ~1.5 hours.** It uses CPU only — all 192 EPYC
cores via rayon parallelism. The GPU is free for Phase 2.1 the entire
time.

**Phase 2.1 Stage 1 runs for days/weeks.** It uses the GPU. Monitor
loss in the logs.

---

## Step 5: After Phase 0.3 Completes (~1.5 hrs)

Phase 0.3 finishes quickly. Verify and secure the output immediately.

```bash
# On the instance
ls -lh ~/volt-x/checkpoints/codebook_code.bin
# Expected: ~64 MB
```

Download the codebook to your local machine right away (don't risk
losing it to an instance failure):

```powershell
# From local PowerShell
scp -P <PORT> root@<HOST>:~/volt-x/checkpoints/codebook_code.bin c:\Volt_X\checkpoints\
```

Phase 0.3 is now done. The CPU sits idle for the remainder — that's
fine, you're only paying $0.20/hr total.

---

## Step 6: Phase 2.1 — Curriculum Stages

Phase 2.1 uses curriculum training: 3 stages, each adding harder
problems. After Stage 1 completes, start Stage 2 manually.

### When Stage 1 finishes

```bash
# Verify Stage 1 output
ls -lh ~/volt-x/checkpoints/scaled_vfn_stage1.safetensors

# Start Stage 2 (add APPS introductory + interview)
cd ~/volt-x
cargo run --release -p volt-learn --features vfn-training --bin train-vfn -- \
  --data data/code_training_combined.jsonl \
  --data data/apps_introductory_train.jsonl \
  --data data/apps_interview_train.jsonl \
  --tokenizer checkpoints/code_tokenizer.json \
  --encoder checkpoints/code_encoder.safetensors \
  --decoder checkpoints/code_decoder.safetensors \
  --output checkpoints/scaled_vfn_stage2.safetensors \
  --epochs 10 --batch-size 32 --lr 5e-5 --warmup 1000 \
  --hidden-dim 2048 --num-blocks 6 \
  --device cuda \
  2>&1 | tee logs/phase21_stage2.log
```

### When Stage 2 finishes

```bash
# Verify Stage 2 output
ls -lh ~/volt-x/checkpoints/scaled_vfn_stage2.safetensors

# Start Stage 3 (add APPS competition — hardest problems)
cd ~/volt-x
cargo run --release -p volt-learn --features vfn-training --bin train-vfn -- \
  --data data/code_training_combined.jsonl \
  --data data/apps_introductory_train.jsonl \
  --data data/apps_interview_train.jsonl \
  --data data/apps_competition_train.jsonl \
  --tokenizer checkpoints/code_tokenizer.json \
  --encoder checkpoints/code_encoder.safetensors \
  --decoder checkpoints/code_decoder.safetensors \
  --output checkpoints/scaled_vfn.safetensors \
  --epochs 10 --batch-size 32 --lr 2e-5 --warmup 500 \
  --hidden-dim 2048 --num-blocks 6 \
  --device cuda \
  2>&1 | tee logs/phase21_stage3.log
```

### Automating all 3 stages

If you want all 3 stages to run unattended (so you don't have to SSH
back in between stages), use this combined script instead of the
individual commands above:

```bash
# Create ~/volt-x/run_phase2.sh
cat << 'SCRIPT' > ~/volt-x/run_phase2.sh
#!/usr/bin/env bash
set -euo pipefail
cd ~/volt-x

COMMON="--tokenizer checkpoints/code_tokenizer.json \
  --encoder checkpoints/code_encoder.safetensors \
  --decoder checkpoints/code_decoder.safetensors \
  --hidden-dim 2048 --num-blocks 6 --device cuda"

echo "=== STAGE 1: HumanEval + MBPP (lr=1e-4) ==="
cargo run --release -p volt-learn --features vfn-training --bin train-vfn -- \
  --data data/code_training_combined.jsonl \
  --output checkpoints/scaled_vfn_stage1.safetensors \
  --epochs 10 --batch-size 32 --lr 1e-4 --warmup 2000 \
  $COMMON 2>&1 | tee logs/phase21_stage1.log

echo "=== STAGE 2: + APPS intro + interview (lr=5e-5) ==="
cargo run --release -p volt-learn --features vfn-training --bin train-vfn -- \
  --data data/code_training_combined.jsonl \
  --data data/apps_introductory_train.jsonl \
  --data data/apps_interview_train.jsonl \
  --output checkpoints/scaled_vfn_stage2.safetensors \
  --epochs 10 --batch-size 32 --lr 5e-5 --warmup 1000 \
  $COMMON 2>&1 | tee logs/phase21_stage2.log

echo "=== STAGE 3: + APPS competition (lr=2e-5) ==="
cargo run --release -p volt-learn --features vfn-training --bin train-vfn -- \
  --data data/code_training_combined.jsonl \
  --data data/apps_introductory_train.jsonl \
  --data data/apps_interview_train.jsonl \
  --data data/apps_competition_train.jsonl \
  --output checkpoints/scaled_vfn.safetensors \
  --epochs 10 --batch-size 32 --lr 2e-5 --warmup 500 \
  $COMMON 2>&1 | tee logs/phase21_stage3.log

echo "=== ALL 3 STAGES COMPLETE ==="
ls -lh checkpoints/scaled_vfn*.safetensors
SCRIPT

chmod +x ~/volt-x/run_phase2.sh
```

Then in tmux, instead of the Stage 1 command, run:

```bash
~/volt-x/run_phase2.sh
```

All three stages execute back-to-back with no manual intervention.

---

## Step 7: Monitoring

### Check training progress

```bash
# Re-attach to tmux
tmux attach -t training

# Or tail logs from any SSH session:
tail -f ~/volt-x/logs/phase03.log       # Phase 0.3 (CPU)
tail -f ~/volt-x/logs/phase21_stage1.log # Phase 2.1 (GPU)
```

### Check GPU utilization

```bash
# One-shot
nvidia-smi

# Continuous (updates every 5 seconds)
watch -n 5 nvidia-smi
```

The RTX 5090 should show ~90–100% GPU utilization during VFN training.
If it shows <50%, the batch size may be too small or data loading is
the bottleneck.

### Check CPU utilization (Phase 0.3)

```bash
htop
# All 192 cores should be active during k-means (rayon parallelism)
```

### Periodic checkpoint backup

Don't wait until training is fully done. Download intermediate
checkpoints periodically to protect against instance failure.

```powershell
# From local PowerShell — run every few days during Phase 2.1
scp -P <PORT> root@<HOST>:~/volt-x/checkpoints/scaled_vfn_epoch_*.safetensors c:\Volt_X\checkpoints\
```

---

## Step 8: Download Final Results and Terminate

### After all Phase 2.1 stages complete

```powershell
# From local PowerShell — download all outputs

# Codebook (if not already downloaded after Phase 0.3)
scp -P <PORT> root@<HOST>:~/volt-x/checkpoints/codebook_code.bin c:\Volt_X\checkpoints\

# VFN checkpoints — all stages
scp -P <PORT> root@<HOST>:~/volt-x/checkpoints/scaled_vfn_stage1.safetensors c:\Volt_X\checkpoints\
scp -P <PORT> root@<HOST>:~/volt-x/checkpoints/scaled_vfn_stage2.safetensors c:\Volt_X\checkpoints\
scp -P <PORT> root@<HOST>:~/volt-x/checkpoints/scaled_vfn.safetensors c:\Volt_X\checkpoints\

# Training logs (for reference)
scp -P <PORT> root@<HOST>:~/volt-x/logs/*.log c:\Volt_X\logs\
```

### Verify downloads locally

```powershell
dir c:\Volt_X\checkpoints\
# Expected:
#   code_tokenizer.json        (~2.3 MB)  — already had
#   code_encoder.safetensors   (~20.6 MB) — already had
#   code_decoder.safetensors   (~21.0 MB) — already had
#   codebook_code.bin          (~64 MB)   — NEW from Phase 0.3
#   scaled_vfn_stage1.safetensors (~200 MB) — NEW
#   scaled_vfn_stage2.safetensors (~200 MB) — NEW
#   scaled_vfn.safetensors     (~200 MB)  — NEW (final)
```

### TERMINATE THE INSTANCE

**Do this immediately after downloading.** Every minute you delay costs
money.

Go to https://cloud.vast.ai/instances/ → click **Destroy** on your
instance. Do not just "Stop" it — stopped instances may still incur
storage charges on some providers.

---

## Troubleshooting

### Build fails: "CUDA not found" or "nvcc not found"

The Docker image is missing CUDA development tools. Either:
- Switch to an image with `-devel` tag (e.g., `nvidia/cuda:12.8.0-devel-ubuntu22.04`)
- Or install manually:
  ```bash
  export CUDA_HOME=/usr/local/cuda
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```

### Build fails: "edition 2024 not supported"

Rust is too old. Need 1.85+:
```bash
rustup update stable
rustc --version
```

### OOM on GPU

Reduce batch size:
```bash
--batch-size 16    # halves VRAM (default is 32)
--batch-size 8     # quarters VRAM
```

The model is 51M params — it should fit easily in 32 GB at batch 32.
OOM likely means something else is consuming VRAM.

### Upload is slow

If your upstream is slower than expected, consider:
- Uploading just `stack_python.tar.gz` first (needed for Phase 0.3)
  and starting training while `stack_multilang.tar.gz` continues uploading
- Using `rsync --partial --progress` instead of `scp` for resumable
  transfers (install rsync on Windows via Git Bash or WSL)

### Disk full during extraction

The compressed archives + extracted files coexist briefly. You need
~250 GB free. If tight on space, extract and delete archives one at a
time instead of in parallel.

### Instance preempted mid-training

The training saves checkpoints every epoch. Find the latest:
```bash
ls -lt ~/volt-x/checkpoints/scaled_vfn_epoch_*.safetensors
```

Rent a new instance, re-run setup, upload the latest checkpoint, and
resume training from that epoch.

### SSH disconnects

This is why we use tmux. Training continues in the background.
Just reconnect and re-attach:
```bash
ssh -p <PORT> root@<HOST>
tmux attach -t training
```

---

## Cost Summary

| Item | Duration | Rate | Cost |
|------|----------|------|------|
| Prep (upload + build + extract) | ~0.5 hr | $0.20/hr | $0.10 |
| Phase 0.3 (CPU k-means) | ~1.5 hr | (parallel) | $0.00 |
| Phase 2.1 Stage 1 | ~50–100 hr | $0.20/hr | $10–20 |
| Phase 2.1 Stage 2 | ~100–200 hr | $0.20/hr | $20–40 |
| Phase 2.1 Stage 3 | ~100–200 hr | $0.20/hr | $20–40 |
| **Total** | **~300–500 hr** | | **$60–100** |

Phase 0.3 costs nothing extra — it runs on CPU while the GPU handles
Phase 2.1. The prep phase costs ~$0.10 (30 min at $0.20/hr). All Stack
data is uploaded from your local machine — no HuggingFace dependency.

---

## File Reference

### Files uploaded from local machine → cloud

#### Archive 1 — VoltStaging.tar.gz (~200–400 MB compressed)

| File | Size | Local Path | Cloud Path |
|------|------|------------|------------|
| Tokenizer | 2.3 MB | `c:\Volt_X\checkpoints\code_tokenizer.json` | `~/volt-x/checkpoints/` |
| Encoder | 20.6 MB | `c:\Volt_X\checkpoints\code_encoder.safetensors` | `~/volt-x/checkpoints/` |
| Decoder | 21.0 MB | `c:\Volt_X\checkpoints\code_decoder.safetensors` | `~/volt-x/checkpoints/` |
| HumanEval | 0.2 MB | `D:\VoltData\phase0\humaneval\data\humaneval.jsonl` | `~/volt-x/data/` |
| MBPP | 0.1 MB | `D:\VoltData\phase0\mbpp\data\mbpp.jsonl` | `~/volt-x/data/` |
| Combined | ~0.3 MB | `D:\VoltData\phase0\code_training_combined.jsonl` | `~/volt-x/data/` |
| APPS intro | varies | `D:\VoltData\phase2\apps\data\apps_introductory_train.jsonl` | `~/volt-x/data/` |
| APPS interview | varies | `D:\VoltData\phase2\apps\data\apps_interview_train.jsonl` | `~/volt-x/data/` |
| APPS competition | varies | `D:\VoltData\phase2\apps\data\apps_competition_train.jsonl` | `~/volt-x/data/` |

#### Archive 2 — stack_python.tar.gz (~9–11 GB compressed)

| File | Size | Local Path | Cloud Path |
|------|------|------------|------------|
| Python Stack (6.6M files) | 43.8 GB | `D:\VoltData\phase0\the_stack_sample\python\python_sample.jsonl` | `~/volt-x/data/python_sample.jsonl` |

#### Archive 3 — stack_multilang.tar.gz (~20–25 GB compressed)

| File | Size | Local Path | Cloud Path |
|------|------|------------|------------|
| Python (multilang) | 24.0 GB | `D:\VoltData\phase5\the_stack_multilang\python\python_sample.jsonl` | `~/volt-x/data/multilang/python/` |
| JavaScript | 24.0 GB | `D:\VoltData\phase5\the_stack_multilang\javascript\javascript_sample.jsonl` | `~/volt-x/data/multilang/javascript/` |
| Java | 24.0 GB | `D:\VoltData\phase5\the_stack_multilang\java\java_sample.jsonl` | `~/volt-x/data/multilang/java/` |
| Go | 19.2 GB | `D:\VoltData\phase5\the_stack_multilang\go\go_sample.jsonl` | `~/volt-x/data/multilang/go/` |
| Rust | 7.2 GB | `D:\VoltData\phase5\the_stack_multilang\rust\rust_sample.jsonl` | `~/volt-x/data/multilang/rust/` |

#### Upload totals

| Metric | Size |
|------|------|
| Uncompressed | ~142 GB |
| Compressed | ~30–36 GB |

### Files produced by training (download cloud → local when done)

| File | Size | Cloud Path | Local Path |
|------|------|-----------|------------|
| Codebook | ~64 MB | `~/volt-x/checkpoints/codebook_code.bin` | `c:\Volt_X\checkpoints\` |
| VFN Stage 1 | ~200 MB | `~/volt-x/checkpoints/scaled_vfn_stage1.safetensors` | `c:\Volt_X\checkpoints\` |
| VFN Stage 2 | ~200 MB | `~/volt-x/checkpoints/scaled_vfn_stage2.safetensors` | `c:\Volt_X\checkpoints\` |
| VFN Final | ~200 MB | `~/volt-x/checkpoints/scaled_vfn.safetensors` | `c:\Volt_X\checkpoints\` |
