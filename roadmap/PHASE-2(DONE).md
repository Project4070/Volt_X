# Phase 2: The Soft Core (Months 3-4)

> Extracted from the master roadmap. Depends on: Phase 1 completed.

## Goal
Replace the stub processor with actual GPU-based RAR reasoning. The system should start producing meaningfully different outputs based on the complexity of the input.

---

## Milestone 2.1: VQ-VAE Codebook (Week 9-10) ✅ DONE

**Crate:** `volt-bus` (extend)

### What You Build
- 65,536-entry codebook: `[65536, 256]` float array
- HNSW index over codebook entries using `hnsw_rs` or custom implementation
- `quantize(vector) -> (codebook_id: u16, quantized_vector: [f32; 256])`
- `lookup(codebook_id) -> [f32; 256]`
- Codebook initialization: K-Means clustering over word embeddings from a pretrained model (download embeddings, cluster offline, save as binary)

### What You Test
- Quantize a vector -> lookup by ID -> cosine sim to original > 0.85
- HNSW query: 1000 random queries, each returns nearest codebook entry in < 0.5ms
- Codebook utilization after initializing from embeddings: > 80% of entries used

**Duration:** 2 weeks.

---

## Milestone 2.2: Real Forward Translator (Week 11-13) ✅ DONE

**Crate:** `volt-translate` (replace stub)

### What You Build
- Load a frozen LLM backbone (Qwen3-0.6B via `candle-transformers` Qwen2 module)
- Frame Projection Head: lightweight MLP (3 layers, 50M params) mapping LLM hidden states -> slot assignments + slot vectors
- Training pipeline:
  - Download PropBank/FrameNet data
  - For each annotated sentence: LLM hidden states -> Frame Projection Head -> predicted slots
  - Loss: slot assignment cross-entropy + codebook quantization commitment + round-trip reconstruction
  - Optimizer: AdamW, lr=1e-4, batch=32
  - Train on single GPU for ~7 days

### What You Test
- "The cat sat on the mat" -> S0=cat (AGENT), S1=sat (PRED), S2=mat (LOCATION)
- Slot assignment accuracy > 80% on PropBank validation set
- Round-trip BLEU > 0.70 (encode -> decode -> compare to original)
- Codebook utilization > 75%

**Duration:** 3 weeks (including training time).

---

## Milestone 2.3: Basic RAR Loop on CPU (Week 14-15) ✅ DONE

**Crate:** `volt-soft`

### What You Build
- First, implement RAR on CPU (not GPU yet). This lets you debug the algorithm without CUDA complexity.
- Slot-local VFN: a simple MLP (4 layers, 256->512->512->256). Randomly initialized. Not trained yet -- just verify the loop mechanics.
- RAR phases:
  - **Root:** apply VFN to each active slot independently
  - **Attend:** compute 16x16 attention matrix (Q, K, V projections + softmax)
  - **Refine:** state update + unit normalization + convergence check
- Convergence detection: per-slot ||delta_S|| < epsilon
- Budget enforcement: maximum 50 iterations
- Progressive freezing: converged slots skip Root phase

### What You Test
- Random input frame -> RAR loop runs -> eventually converges (all slots ||delta|| < epsilon)
- Easy input (few filled slots) -> converges in < 5 iterations
- Complex input (many filled slots) -> takes more iterations
- Frozen slots don't change between iterations
- Timing: 50 iterations on CPU < 500ms (not fast, but testable)

### What You Explicitly Do NOT Build Yet
- No GPU. CPU only.
- No trained VFN. Random weights.
- No ghost frames or bleed buffer.
- No diffusion noise.

### Why CPU First
GPU debugging is hell. CUDA errors are opaque. CPU debugging is transparent: you can step through every value in a debugger. Get the algorithm right on CPU, then port to GPU.

**Duration:** 2 weeks.

---

## Milestone 2.4: GPU Port + VFN Training (Week 16-18) ✅ DONE

**Crate:** `volt-soft` (extend)

### What You Build
- Port RAR loop to GPU using `cudarc` or `candle` CUDA backend
- Parallelize Root phase: all 16 slot VFN passes in one batched CUDA kernel
- Attention phase: batched matrix multiply on GPU
- Add diffusion noise injection (per-slot adaptive sigma)
- Train VFN via Flow Matching:
  - Generate (question, answer) frame pairs from training data
  - Linear interpolation path: `F(t) = (1-t)*F_q + t*F_a`
  - Train VFN to predict `(F_a - F_q)` at every t
  - Loss: MSE on drift direction
  - Train for ~2 weeks on single GPU

### What You Test
- GPU RAR produces same results as CPU RAR (bit-close, within float precision)
- GPU is > 10x faster than CPU implementation
- After VFN training: convergence rate > 80% on validation question-answer pairs
- Adaptive computation: simple questions converge in < 5 iterations, complex in > 10
- Diffusion: increasing sigma produces more diverse outputs for creative queries

**Duration:** 3 weeks.

---

## Phase 2 Checkpoint

At the end of Phase 2, you have a system that actually thinks. The GPU runs the RAR loop, slots converge at different rates, and the VFN produces meaningful drift directions. The outputs are noticeably better than Phase 1's heuristic translator.
