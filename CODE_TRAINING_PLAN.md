# Volt X — Code Training Plan: From Demo to Code Intelligence

## Status Quo

Volt X currently runs end-to-end: text in, TensorFrame, LLL bus, RAR
inference, Hard Core verification, safety check, text out. 573 tests
pass. The pipeline is structurally complete.

This document describes a **code-specialized training path** — training
Volt X to excel at programming tasks before acquiring serious hardware
(B200). The goal is to prove Volt's architectural advantages
(compositional reasoning, multi-step inference, deterministic
verification) on a domain with:

1. **Clean evaluation:** Code either passes tests or doesn't
2. **Structural alignment:** Functions decompose into slots, execution
   is multi-step (RAR), Hard Strand can verify
3. **Smaller datasets:** 100K high-quality examples vs. 10M general text
4. **Clear value:** Developers need code tools, market exists

---

## Phase 0: Bootstrap (No GPU Required)

**Goal:** Build the code-specific data foundation.

**Duration:** 2–3 weeks
**Hardware:** CPU only (any modern machine)

### 0.1 — VFN Checkpoint System **DONE**

**What:** Save and load VFN weights to disk.
**Why:** Same as general plan — no training without persistence.
**Scope:** Implement `Vfn::save(path)` / `Vfn::load(path)` with binary
format (magic bytes + version + raw weights). Checksum validation.
**Deliverable:** Roundtrip test — train one Forward-Forward epoch on
code examples, save, load, verify bitwise identical.

**Dataset:** None (infrastructure task)
**GPU hours:** 0

---

### 0.2 — Code Dataset Pipeline **DONE**

**What:** A file-based dataset loader for (code_query, code_solution) pairs.
**Why:** Current training uses 1,000 synthetic eval pairs. Real code
training requires real problems at scale.

**Scope:**
- Define JSONL format: `{"query": "...", "solution": "...", "tests": [...]}`
- Streaming reader yields `(TensorFrame, TensorFrame)` pairs
- Support for test cases (executable verification)
- Conversion scripts for HumanEval, MBPP, APPS to this format

**Datasets:**
- **HumanEval:** 164 Python problems (hand-written, high quality)
- **MBPP:** 974 Python problems (crowdsourced, natural language)
- **Exercism Python:** ~150 problems (test-driven learning)

**Total data:** ~1,300 problems
**Format conversion:** Write scripts in `volt-learn/scripts/convert_code_datasets.py`

**Deliverable:** Load 1,000+ code pairs, iterate through them, confirm
TensorFrame encoding is deterministic. Verify test cases can be parsed
and stored.

**GPU hours:** 0

---

### 0.3 — Codebook Initialization from Code Corpus

**What:** Populate the 65,536-entry codebook from code embeddings
instead of natural language text.
**Why:** Code has different distribution than natural language — more
structured, uses symbols (brackets, operators), different token
frequencies. The codebook must cover code's actual embedding space.

**Scope:**
- Download The Stack (Python subset, ~50GB → sample 1M files)
- Encode code through stub translator (treat code as structured text)
- Collect all slot embeddings (16 slots × 1M files = 16M vectors)
- Run k-means (k=65,536) on collected vectors
- Use centroids as codebook entries
- Rebuild HNSW index

**Dataset:**
- **The Stack (Python):** ~470GB total → sample 50GB
- **Deduplication:** Use built-in deduplicated version
- **Filtering:** Remove minified code, generated files, test fixtures
- **Target:** 1M Python files (~500-1000 lines each)

**Deliverable:** Codebook where nearest-neighbor lookup on code
embeddings returns semantically coherent entries. Quantization error
below threshold (mean L2 distance < 0.05).

**Processing time:** ~24 hours CPU (k-means on 16M vectors)
**GPU hours:** 0

**Note** The code implementation has been done, but K-means processing is deferred to after the phase 1 translator training

---

### 0.4 — Slot Attention Weight Initialization for Code **DONE**

**What:** Replace random slot attention weights with structured
initialization reflecting code semantics.

**Why:** Code has different slot usage than natural language:
- S0 (Agent): Function name / class
- S1 (Predicate): Operation / method call
- S2 (Patient): Arguments / parameters
- S3 (Instrument): Return value / result
- S4 (Time): Execution order (for loops, etc.)
- S5 (Manner): Algorithm pattern (recursive, iterative)
- S6-S8: Control flow (if/else, try/catch)
- S9-S15: Free slots for complex logic

**Scope:**
- Define code-specific attention bias matrix:
  - Function ↔ Arguments (strong)
  - Function ↔ Return (strong)
  - Control flow ↔ Predicate (medium)
  - Time ↔ Loops (strong)
- Initialize Q/K/V projections to approximate this prior
- Validate on synthetic code examples (function definitions, loops, conditionals)

**Dataset:** 1,000 synthetic code patterns (generated via script)

**Deliverable:** Comparison showing structured init converges in fewer
RAR iterations than random init on code understanding tasks.

**GPU hours:** 0 (CPU validation only)

---

## Phase 1: Translator Training (Minimal GPU)
**Goal:** Train translator to encode code → TensorFrame and decode
TensorFrame → code. This is critical — all downstream tasks depend on
code representation quality.

**Duration:** 3–4 weeks
**Hardware:** 1–2× RTX 4090 (or 1× H100)

### 1.1 — Learned Code Encoder **DONE**

**What:** Train a model that maps tokenized code to TensorFrame slot
embeddings.

**Why:** Stub translator uses hash-based encoding — "def" and "class"
map to unrelated vectors. Learned encoder must capture code semantics:
`sum_array()` and `calculate_total()` should be similar.

**Scope:**
- Architecture: Lightweight encoder (NOT a transformer)
  - Option A: Tree-structured CNN on AST (Abstract Syntax Tree)
  - Option B: MLP on subword embeddings + positional encoding
  - Output: 16 slots × 256 dims
- Training objective: Contrastive loss
  - Similar code (same functionality, different names) → cosine sim > 0.8
  - Different code (different functionality) → cosine sim < 0.3
- Data source: CodeSearchNet (function + docstring pairs)

**Datasets:**
- **CodeSearchNet (Python):** 100K function-docstring pairs
  - Use docstring as semantic label for contrastive learning
  - Positive pairs: (code_i, doc_i) pulled together
  - Negative pairs: All mismatched pairs in batch pushed apart
- **Split:** 90K train, 10K validation

**Architecture (implemented):**
- Input: BPE tokens (32,768 vocab, trained on The Stack), max 512 tokens
- Embedding: `Embedding(32768, 128)` — 4.19M params
- 3-layer CNN: Conv1D(128→256, k=3), Conv1D(256→256, k=5), Conv1D(256→256, k=7) + LayerNorm + GELU
- Role Head: Linear(256, 16) → softmax (per-token role assignment)
- Embed Head: Linear(256, 256) (per-token embedding projection)
- Slot aggregation: weighted average per role → L2-norm → 16×256 TensorFrame
- Parameters: ~5.1M
- Loss: InfoNCE contrastive loss (temperature τ=0.07)
- Optimizer: AdamW (lr=5e-4, weight_decay=0.01, warmup 2K steps, cosine decay)

**Training results (RTX 5090 Mobile 24GB):**
- 10 epochs, batch size 128, ~800 steps/epoch
- Final: Train Contrastive Loss = 1.41, Valid Loss = 1.69
- Training time: ~1.5 hours total

**Trainable parameters:** ~5.1M
**GPU hours:** ~1.5 (RTX 5090 Mobile)
**Artifacts:** `checkpoints/code_encoder.safetensors`, `checkpoints/code_tokenizer.json`

---

### 1.2 — Learned Code Decoder **DONE**

**What:** Train TensorFrame → code generation (diagnostic — proves
encoder features are informative).

**Why:** Validates that the encoder captures enough information to
reconstruct original code. The Phase 1 decoder is a diagnostic tool;
Phase 5 replaces it with an LLM-backed translator (Qwen3-0.6B, 600M params).

**Architecture (implemented — autoregressive with cross-attention):**
- Input: Per-token CNN features from encoder `[batch, seq_len, 256]`
- Token embedding: `Embedding(32768, 128)` — shared with encoder
- Projection up: Linear(128, 512) + GELU
- Cross-attention: Q from decoder, K/V from encoder features
- Self-attention: Causal mask (autoregressive)
- FFN: Linear(512, 512) + GELU
- Output projection: Linear(512, 128) → tied with embedding → logits
- Custom RMSNorm (CUDA-compatible, replaces candle's LayerNorm)
- Parameters: ~6.7M (embeddings shared with encoder)
- Loss: Teacher-forced cross-entropy on BPE token sequences
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)

**Key design decision:** Decoder cross-attends to per-token CNN features
`[batch, seq_len, 256]` rather than aggregated slot vectors `[batch, 16, 256]`.
Slot aggregation destroys per-token positional/identity information,
causing accuracy to plateau at ~33% (bigram-level prediction). Direct
per-token features yield 61% accuracy.

**Training results (RTX 5090 Mobile 24GB):**
- 10 epochs, batch size 64, ~552s/epoch
- Final: Train Loss = 1.9992, Valid Loss = 2.1381, Valid Acc = 61.14%
- Training time: ~1.5 hours total

**Note on accuracy:** 61% from a ~6M param model (4M in embedding table,
~2M actual attention+FFN logic) is a strong result. 90%+ accuracy would
require 100M+ parameters. This decoder validates that the encoder
produces informative features.

**Trainable parameters:** ~6.7M
**GPU hours:** ~1.5 (RTX 5090 Mobile)
**Artifacts:** `checkpoints/code_decoder.safetensors`

---

### 1.3 — Role Grounding for Code Structure **DONE**

**What:** Train translator to assign code elements to semantically
correct slots.

**Why:** Stub translator assigns by position. Real code has non-linear
structure — function name should always → S0, arguments → S2, regardless
of syntax order.

**Implementation:** Heuristic role labeling function (`role_labels.rs`)
assigns BPE tokens to slot roles based on Python syntax patterns:

- `def`/`class` + following identifier → S0 (Agent/function name)
- `(` to `)` after function name → S2 (Patient/arguments)
- `return` + expression → S3 (Location/return value)
- `if`/`else`/`elif` → S6 (Instrument/control flow)
- `for`/`while` → S4 (Time/execution order)
- `try`/`except`/`catch` → S7 (Cause/control flow 2)

The encoder's Role Head (Linear(256, 16) → softmax) learns per-token
role assignment jointly with the contrastive loss during encoder
training. The heuristic labels provide supervision signal via
cross-entropy loss weighted at 0.3× contrastive + 0.7× role classification.

**Trainable parameters:** Shared with 1.1 (~5M total)
**GPU hours:** Included in 1.1 (joint training)

---

**Phase 1 Total: COMPLETE**
- **Datasets used:** CodeSearchNet Python (100K pairs), The Stack Python (50K files for BPE tokenizer)
- **GPU hours used:** ~3 hours (RTX 5090 Mobile 24GB)
- **Calendar time:** ~1 week
- **Deliverables:** CNN code encoder (5.1M params, contrastive loss 1.41), autoregressive decoder (6.7M params, 61% token accuracy), BPE tokenizer (32K vocab), role labeling heuristics
- **Artifacts:** `checkpoints/code_tokenizer.json`, `checkpoints/code_encoder.safetensors`, `checkpoints/code_decoder.safetensors`

---

## Phase 2: Soft Core Training (Moderate GPU)

**Goal:** Train VFN so RAR converges to correct code solutions.

**Duration:** 4–6 weeks
**Hardware:** 2–4× RTX 4090 or 1–2× H100

### 2.1 — VFN Flow Matching on Code Tasks

**What:** Train VFN to learn velocity field driving query frames
(problem description) toward solution frames (working code).

**Why:** VFN defines the energy landscape for RAR. Untrained = random
drift. Trained = drift toward correct algorithmic solutions.

**Scope:**
- Scale VFN from 3-layer (525K params) to 50M–200M params
  - Start small (50M) before scaling to 500M+ in Phase 5
  - Architecture: Deep MLP or small Fourier Neural Operator
- Training: Flow Matching
  - Learn v(F_query, t) → F_solution
  - Loss: MSE between predicted drift and target drift
- Data: (problem_frame, solution_frame) pairs
- Curriculum:
  - Stage 1: Simple functions (single operation, no loops)
  - Stage 2: Loops and conditionals
  - Stage 3: Multi-function programs
  - Stage 4: Algorithmic reasoning (sorting, searching)

**Datasets:**

**Stage 1 (Weeks 1-2):** Simple problems
- **MBPP (introductory):** ~400 problems
- **Exercism (easy):** ~80 problems
- **Total:** ~500 problems, 5K training pairs (10× augmentation via paraphrasing)

**Stage 2 (Weeks 2-3):** Intermediate
- **MBPP (intermediate):** ~400 problems
- **HumanEval (easy subset):** ~80 problems
- **Total:** ~500 problems, 5K pairs

**Stage 3 (Weeks 3-4):** Advanced
- **HumanEval (full):** 164 problems
- **APPS (introductory):** ~2,000 problems
- **Total:** ~2,200 problems, 10K pairs

**Stage 4 (Weeks 4-6):** Algorithmic
- **APPS (interview level):** ~2,500 problems
- **CLRS (algorithm traces):** ~30K examples (30 algorithms × 1K each)
- **Total:** ~30K training pairs

**Training details:**
- Parameters: 50M (VFN) + 33K (attention) = ~50M total
- Optimizer: AdamW (lr=1e-4 → 1e-5 with cosine decay)
- Batch size: 32 (fits in 24GB VRAM with gradient checkpointing)
- Steps: 500K total (100K per stage)
- Mixed precision: BF16

**Validation:**
- RAR convergence rate on held-out HumanEval (164 problems)
- Target: >60% converge within 30 iterations
- Solution quality: Cosine sim > 0.6 vs. ground truth

**Deliverable:** VFN where RAR iterations on code problems:
- Converge within 30 iterations for >60% of simple problems
- Produce frames decodable to syntactically valid code >80% of the time
- Generate code passing >40% of test cases (Pass@1 metric)

**Trainable parameters:** 50M (VFN) + 33K (attention)
**GPU hours:** 200–400 (RTX 4090) or 100–200 (H100)
**Dataset size:** ~45K training pairs total

---

### 2.2 — Slot Attention Training for Code Flow

**What:** Train cross-slot attention jointly with VFN for code
reasoning.

**Why:** Code requires specific information flow:
- Function name (S0) must attend to Arguments (S2) to understand inputs
- Operation (S1) attends to Control Flow (S6-S8) for branching logic
- Return (S3) attends to Operation (S1) for result computation

**Scope:**
- Joint training with VFN (end-to-end through RAR)
- Code-specific attention patterns to learn:
  - S0 ↔ S2 (function ↔ args)
  - S1 ↔ S6-S8 (operation ↔ control flow)
  - S3 ↔ S1 (return ↔ operation)
- Regularization: Attention entropy penalty (prevent collapse)
- Visualization: Generate attention heatmaps on test code

**Dataset:** Same as 2.1 (joint training)

**Deliverable:** Attention heatmaps showing code-appropriate patterns:
- Function definitions: S0 attends strongly to S2 (args)
- Loops: S4 (time) attends to S6 (control flow)
- Returns: S3 attends to S1 (operation)

**Trainable parameters:** ~33K (joint with 2.1)
**GPU hours:** Included in 2.1 (joint training)

---

### 2.3 — Diffusion Controller Tuning

**What:** Train per-slot noise schedule for RAR exploration.

**Why:** Code generation needs more exploration in early iterations
(search algorithm space), then exploitation (refine syntax).

**Scope:**
- Small MLP (~10K params): (slot_state, iteration, delta) → σ
- Train alongside VFN
- Objective: Minimize iterations to valid code while maintaining quality

**Dataset:** Same as 2.1

**Deliverable:** Learned schedule reduces mean RAR iterations by >20%
on hard problems (APPS interview level).

**Trainable parameters:** ~10K
**GPU hours:** 10–20 (RTX 4090) or 5–10 (H100)

---

**Phase 2 Total:**
- **Datasets:** MBPP (974), HumanEval (164), APPS (5K), CLRS (30K)
- **Total training pairs:** ~45K
- **GPU hours:** 210–420 (RTX 4090) or 105–210 (H100)
- **Calendar time:** 4–6 weeks
- **Deliverables:** Trained VFN (50M params) converging on code problems

---

## Phase 3: Hard Core Calibration (Minimal GPU)

**Goal:** Calibrate Hard Core for code-specific tasks — route to
correct strand, verify solutions, detect unsafe code.

**Duration:** 2–3 weeks
**Hardware:** 1× RTX 4090 or CPU-heavy with small GPU

### 3.1 — Intent Router Calibration for Code

**What:** Train intent router to route code queries to correct strands.

**Code-specific strands:**
- **CodeRunner:** Generate code from description
- **CodeDebugger:** Fix buggy code
- **CodeExplainer:** Explain what code does (reverse translation)
- **MathEngine:** Math-heavy algorithms (dynamic programming, combinatorics)
- **MemoryRetrieval:** Code search / documentation lookup

**Scope:**
- Collect 10K labeled code queries across 5 strands
- Train strand vectors (256-dim each) via contrastive loss
- Support multi-strand routing (query may need CodeRunner + MathEngine)

**Datasets:**
- **CodeRunner:** HumanEval (164), MBPP (974) → 1,138 examples, augment to 4K
- **CodeDebugger:** CodeXGLUE defect detection (10K) → sample 2K
- **CodeExplainer:** CodeSearchNet (generate questions from docstrings) → 2K
- **MathEngine:** APPS (math-heavy problems) → 1K
- **MemoryRetrieval:** CodeSearchNet retrieval task → 1K
- **Total:** 10K labeled queries

**Labeling:**
- Automatic for CodeRunner/Debugger (task type is explicit)
- GPT-4 labeling for ambiguous queries
- Manual verification for 1K examples (quality check)

**Training:**
- Strand vectors: 5 × 256-dim = 1,280 parameters
- Loss: Multi-label contrastive (query can match multiple strands)
- Optimizer: AdamW (lr=1e-3, 10K steps)

**Deliverable:** Routing accuracy >90% on held-out 2K test set.

**Trainable parameters:** ~1.3K
**GPU hours:** 2–5 (RTX 4090) or 1–2 (H100)
**Dataset size:** 10K labeled queries

---

### 3.2 — Certainty Calibration (γ) for Code

**What:** Calibrate certainty scores to match test pass probability.

**Why:** In code, γ should predict: P(generated code passes all tests).
Uncalibrated γ is meaningless for user trust.

**Scope:**
- Run inference on 5K code problems (HumanEval + MBPP + APPS sample)
- Collect (generated_code, γ, test_pass_rate) triples
- Fit calibration function (Platt scaling):
  - Input: Raw γ from certainty engine
  - Output: Calibrated P(tests pass)
- Measure Expected Calibration Error (ECE)

**Datasets:**
- **HumanEval:** 164 (all)
- **MBPP test:** 500
- **APPS (intro+interview sample):** 2K
- **Total:** ~2,700 problems with test suites

**Procedure:**
1. Generate code for each problem via trained pipeline
2. Execute tests, record pass rate (0.0–1.0)
3. Collect (γ_raw, pass_rate) pairs
4. Fit isotonic regression: γ_raw → γ_calibrated
5. Validate on held-out 500 examples

**Deliverable:** ECE < 0.08 on held-out test set.
(Lower than general NLP target of 0.05 — code is harder)

**Trainable parameters:** ~100 (calibration curve)
**GPU hours:** 5–10 (inference on 2.7K problems)
**Dataset size:** 2,700 problems with tests

---

### 3.3 — Safety Axiom Refinement for Code

**What:** Refine safety axioms to catch malicious/dangerous code.

**Code-specific safety concerns:**
- K1 (No Harm): Detect destructive operations (file deletion, system commands)
- K2 (Privacy): Detect data exfiltration (network calls, file uploads)
- K3 (Honesty): Detect obfuscated/deceptive code
- K4 (Autonomy): Flag code requiring elevated permissions
- K5 (Beneficence): Ensure code is helpful (not trolling/useless)

**Scope:**
- Generate adversarial code examples designed to evade each axiom
- For each false negative (dangerous code that passed), adjust axiom vector
- For each false positive (safe code blocked), adjust to decrease similarity
- Freeze axiom vectors after refinement (immutable post-training)

**Datasets:**
- **Benign code:** HumanEval + MBPP (1,138 examples) — all safe
- **Adversarial generation:**
  - K1 violations: `os.system("rm -rf /")`, `shutil.rmtree("/")` (500 examples)
  - K2 violations: `requests.post(url, data=secrets)` (500 examples)
  - K3 violations: Obfuscated code, misleading names (500 examples)
  - **Total:** 1,500 adversarial + 1,138 benign = 2,638 examples

**Training:**
- Adjust K1–K5 vectors (5 × 256-dim = 1,280 params)
- Objective: Maximize separation between safe and unsafe
- Constraint: No false positives on HumanEval/MBPP

**Deliverable:**
- False negative rate < 2% on adversarial benchmark
- False positive rate < 3% on benign benchmark
- (Higher tolerance than NLP — code safety is critical)

**Trainable parameters:** ~1.3K (axiom vectors)
**GPU hours:** 3–5 (RTX 4090)
**Dataset size:** 2,638 labeled examples

---

**Phase 3 Total:**
- **Datasets:** 10K routing labels, 2.7K calibration examples, 2.6K safety examples
- **GPU hours:** 10–20 (RTX 4090) or 5–10 (H100)
- **Calendar time:** 2–3 weeks
- **Deliverables:** Calibrated router, certainty scores, safety axioms

---

## Phase 4: Joint Alignment (Moderate GPU)

**Goal:** End-to-end fine-tuning with executable test verification.

**Duration:** 4–6 weeks
**Hardware:** 2–4× RTX 4090 or 1–2× H100

### 4.1 — End-to-End Flow Matching with Test Execution

**What:** Fine-tune full pipeline with gradients from test results.

**Why:** Phases 1–3 train in isolation. E2E training ensures all
components work together, optimized for final metric: test pass rate.

**Scope:**
- Freeze translator encoder/decoder (from Phase 1)
- Fine-tune VFN + attention + diffusion jointly
- Loss: Hybrid
  - 70% flow matching (frame → solution frame)
  - 30% test pass rate (REINFORCE on test outcomes)
- Curriculum: Simple → complex

**Datasets:**
- **Stage 1:** HumanEval (164) + MBPP easy (400) = 564 problems
- **Stage 2:** MBPP full (974) + APPS intro (2K) = 3K problems
- **Stage 3:** APPS intro+interview (4.5K)

**Training:**
- Batch size: 16 (larger problems, more VRAM)
- Steps: 200K (50K per stage + 50K final)
- Optimizer: AdamW (lr=5e-5, decay to 1e-5)
- Test execution: Run generated code in sandbox, collect pass/fail
- Gradient: Backprop through decoder using REINFORCE for discrete test outcomes

**Deliverable:** End-to-end Pass@1 on HumanEval >25% (baseline: GPT-3
Codex ~30%, GPT-2 ~0%).

**Trainable parameters:** 50M (VFN + attention)
**GPU hours:** 150–300 (RTX 4090) or 75–150 (H100)
**Dataset size:** ~4.5K problems with test suites

---

### 4.2 — RLVF Alignment for Code

**What:** REINFORCE with Verified Feedback using executable tests.

**Why:** Test pass/fail is perfect verification signal — no human
labeling needed.

**Scope:**
- Reward table:
  - **All tests pass + γ > 0.7:** +1.0 (correct and confident)
  - **All tests pass + γ < 0.7:** +0.5 (correct but underconfident)
  - **Some tests pass:** +0.1 × pass_rate (partial credit)
  - **No tests pass + γ < 0.3:** +0.2 (wrong but honest)
  - **No tests pass + γ > 0.7:** -2.0 (wrong and overconfident — BAD)
  - **Unsafe code flagged:** -5.0 (safety violation)
- Training data: APPS (interview + competition levels)
- Self-play: Generate harder problems via mutation

**Datasets:**
- **APPS (interview):** 2,500 problems
- **APPS (competition):** 500 problems (very hard, sparse rewards)
- **Self-generated mutations:** 1,000 problems (modify APPS problems)
- **Total:** 4K problems

**Training:**
- Policy: VFN (frozen translator)
- Baseline: Exponential moving average of rewards
- Optimizer: AdamW (lr=1e-5)
- Steps: 100K
- Batch: 8 (each requires test execution)

**Deliverable:**
- Overconfident error rate < 8% (γ > 0.7 on failing code)
- ECE < 0.06
- Pass@1 on APPS (interview) > 15%

**Trainable parameters:** 50M (VFN)
**GPU hours:** 80–150 (RTX 4090) or 40–75 (H100)
**Dataset size:** 4K problems

---

### 4.3 — Sleep Consolidation Validation

**What:** Validate sleep improves code generation performance.

**Why:** Sleep should distill learned patterns (common algorithms,
idioms) into stronger memories.

**Scope:**
- Run 5K inference queries (APPS sample)
- Trigger sleep cycle:
  - Forward-Forward training on high-reward frames
  - Distillation of common patterns (loops, sorting, DP)
  - Graduation: Create specialized strands (e.g., "GraphAlgorithms")
- Re-evaluate on held-out test set

**Datasets:**
- **Pre-sleep:** 5K APPS problems (diverse)
- **Post-sleep eval:** HumanEval + MBPP test (654 problems)
- **Measure:** Pass@1, convergence speed, γ calibration

**Deliverable:**
- Post-sleep Pass@1 improves >5% on new problems
- No degradation >3% on old problems (no catastrophic forgetting)

**Trainable parameters:** 50M (VFN, updated via Forward-Forward)
**GPU hours:** 30–50 (RTX 4090) or 15–25 (H100)
**Dataset size:** 5K training + 654 eval

---

**Phase 4 Total:**
- **Datasets:** APPS (5K), HumanEval (164), MBPP (974)
- **GPU hours:** 260–500 (RTX 4090) or 130–250 (H100)
- **Calendar time:** 4–6 weeks
- **Deliverables:** E2E system achieving Pass@1 >25% on HumanEval

---

## Phase 5: Scale and Benchmark (Full GPU or Pre-B200)

**Goal:** Scale to multiple languages, larger VFN, publish benchmarks.

**Duration:** 4–8 weeks
**Hardware:** 4–8× RTX 4090 or 2–4× H100 (before B200 upgrade)

### 5.1 — VFN Scaling to 200M–500M Parameters

**What:** Scale VFN from 50M → 200M → 500M parameters.

**Why:** Larger VFN = richer energy landscape = better performance on
hard problems.

**Scope:**
- Implement Fourier Neural Operator (FNO) architecture
- Progressive scaling: 50M (current) → 200M → 500M
- Knowledge distillation: 50M → 200M (teacher-student)
- Validation at each scale

**Datasets:**
- **The Stack (Python):** 50GB → 100GB (2M files)
- **APPS (full):** 10K problems
- **CodeContests (easy subset):** 2K problems
- **Total:** ~12K problems, 100GB code corpus

**Training:**
- 200M VFN: 300K steps, batch 16
- 500M VFN: 500K steps, batch 8 (gradient accumulation)
- Optimizer: AdamW (lr=1e-4 → 5e-6)

**Deliverable:**
- 500M VFN converges >80% of queries within 20 iterations
- Pass@1 on HumanEval >35% (exceed GPT-3 Codex)
- Pass@1 on APPS (interview) >20%

**Trainable parameters:** 200M → 500M
**GPU hours:** 800–1,500 (RTX 4090) or 400–750 (H100)
**Dataset size:** 12K problems + 100GB corpus

---

### 5.2 — Multi-Language Training

**What:** Extend to multiple programming languages.

**Why:** Prove compositional transfer — learn "sorting algorithm"
abstractly, apply to Python/Java/Rust.

**Scope:**
- Languages: Python (primary), JavaScript, Java, Rust, Go
- Per-language codebook refinement (optional)
- Cross-lingual transfer tests

**Datasets:**
- **MultiPL-E:** HumanEval in 18 languages (164 × 18 = 2,952 tests)
- **The Stack:** Sample 20GB per language (Python 50GB, JS 20GB, Java 20GB, Rust 10GB, Go 10GB)
- **Exercism:** Multi-language problems (~100 per language)

**Training:**
- Stage 1: Python-only (already done)
- Stage 2: Add JavaScript (20GB, 2K problems)
- Stage 3: Add Java (20GB, 1K problems)
- Stage 4: Add Rust + Go (20GB combined, 1K problems)

**Deliverable:**
- Pass@1 on MultiPL-E (Python) >35%
- Pass@1 on MultiPL-E (JS/Java) >25% (transfer from Python)
- Pass@1 on MultiPL-E (Rust/Go) >20%

**Trainable parameters:** 500M (VFN) + language-specific decoders (10M each)
**GPU hours:** 300–600 (RTX 4090) or 150–300 (H100)
**Dataset size:** ~120GB code, 5K problems

---

### 5.3 — Benchmark Publication

**What:** Rigorous benchmarking vs. transformer baselines.

**Why:** Prove Volt's advantages on compositional code tasks.

**Primary Benchmarks (expected advantage):**
- **HumanEval:** Industry standard, Pass@1/10/100
- **MBPP:** Natural language → code
- **APPS:** Competitive programming (compositional reasoning)
- **MultiPL-E:** Cross-lingual transfer
- **CLRS:** Algorithmic reasoning (RAR = execution steps)

**Secondary Benchmarks (parity expected):**
- **DS-1000:** Data science (library knowledge)
- **CodeXGLUE:** Code understanding tasks

**Baselines:**
- GPT-2 (small, 117M params): ~0% on HumanEval
- CodeGen-350M (Salesforce): ~12% on HumanEval
- GPT-3 Codex (12B): ~30% on HumanEval
- GPT-4 (via API): ~67% on HumanEval
- **Comparable transformer (500M):** Train on same data, same compute

**Metrics:**
- **Pass@k:** Percentage of problems solved with k attempts
- **Compute efficiency:** Pass@1 per GPU-hour (Volt's key advantage)
- **Compositional transfer:** Python → Rust performance (MultiPL-E)
- **Convergence speed:** RAR iterations to valid solution

**Deliverable:** arXiv paper:
- Volt (500M) matches CodeGen-350M at 30% lower compute cost
- Volt (500M) exceeds 500M transformer baseline by >10% on APPS
- Volt shows superior cross-lingual transfer on MultiPL-E
- Architectural analysis: RAR iterations correlate with algorithm steps (CLRS)

**GPU hours:** 100–200 (benchmarking inference)
**Dataset size:** ~15K test problems across all benchmarks

---

**Phase 5 Total:**
- **Datasets:** The Stack (120GB), APPS (10K), MultiPL-E (3K), CLRS (30K)
- **GPU hours:** 1,200–2,300 (RTX 4090) or 650–1,250 (H100)
- **Calendar time:** 4–8 weeks
- **Deliverables:** 500M VFN, multi-language support, published benchmarks

---

## Summary: Code Training Components × Phases

| Component | Phase | Parameters | GPU Hours (4090) | GPU Hours (H100) | Dataset Size |
|-----------|-------|------------|------------------|------------------|--------------|
| VFN checkpoint | 0.1 | — | 0 | 0 | — |
| Code dataset pipeline | 0.2 | — | 0 | 0 | 1.3K problems |
| Codebook (code) | 0.3 | 16M | 0 (CPU) | 0 | 50GB Python |
| Attention init | 0.4 | 33K | 0 | 0 | 1K synthetic |
| Code encoder | 1.1 ✅ | 5.1M | ~1.5 | ~1.5 | 100K pairs |
| Code decoder | 1.2 ✅ | 6.7M | ~1.5 | ~1.5 | 100K funcs |
| Role grounding | 1.3 ✅ | (shared) | (joint) | (joint) | (joint) |
| VFN flow (code) | 2.1 | 50M | 200–400 | 100–200 | 45K pairs |
| Slot attention | 2.2 | 33K | (joint) | (joint) | (joint) |
| Diffusion tuning | 2.3 | 10K | 10–20 | 5–10 | (joint) |
| Router (code) | 3.1 | 1.3K | 2–5 | 1–2 | 10K queries |
| γ calibration | 3.2 | 100 | 5–10 | 5–10 | 2.7K problems |
| Safety (code) | 3.3 | 1.3K | 3–5 | 3–5 | 2.6K examples |
| E2E fine-tuning | 4.1 | 50M | 150–300 | 75–150 | 4.5K problems |
| RLVF (code) | 4.2 | 50M | 80–150 | 40–75 | 4K problems |
| Sleep validation | 4.3 | 50M | 30–50 | 15–25 | 5K + 654 eval |
| VFN scaling | 5.1 | 200M–500M | 800–1,500 | 400–750 | 12K + 100GB |
| Multi-language | 5.2 | 500M + 50M | 300–600 | 150–300 | 5K + 120GB |
| Benchmarks | 5.3 | — | 100–200 | 100–200 | 15K tests |
| **TOTAL** | | **500M** | **~1,740–3,350** | **~930–1,790** | **~100K problems + 120GB code** |

---

## Total Estimated Compute (Code-Specialized)

| Phase | RTX 4090 Hours | H100 Hours | Calendar Time | Key Datasets |
|-------|----------------|------------|---------------|--------------|
| Phase 0: Bootstrap | 0 | 0 | 2–3 weeks | HumanEval, MBPP, The Stack |
| Phase 1: Translator ✅ | ~3 | ~3 | ~1 week | CodeSearchNet (100K) |
| Phase 2: Soft Core | 210–420 | 105–210 | 4–6 weeks | APPS (5K), CLRS (30K) |
| Phase 3: Hard Core | 10–20 | 5–10 | 2–3 weeks | 10K labeled queries |
| Phase 4: Alignment | 260–500 | 130–250 | 4–6 weeks | APPS (5K), HumanEval |
| Phase 5: Scale | 1,200–2,300 | 650–1,250 | 4–8 weeks | MultiPL-E, The Stack (120GB) |
| **TOTAL** | **~1,740–3,350** | **~930–1,790** | **~5–6 months** | **~100K problems + 120GB** |

---

## Compute Cost Estimates (Before B200)

**RTX 4090 Option (Consumer GPUs):**
- **Hardware:** 4× RTX 4090 (24GB each) = ~$7,000 total
- **Time:** ~1,740–3,350 GPU-hours ÷ 4 GPUs = 435–840 wall-clock hours (18–35 days at 100% utilization)
- **Power cost:** ~1.8 kW total × 500 hours × $0.12/kWh = ~$110
- **Total:** Hardware ($7K one-time) + power ($110)

**H100 Option (Cloud GPUs):**
- **Cloud cost:** ~$2–3/hour per H100 (AWS, Lambda Labs)
- **Time:** 930–1,790 H100-hours ÷ 2 GPUs = 465–895 wall-clock hours
- **Total cost:** 930–1,790 hours × $2.50/hour = **$2,325–4,475**

**Recommendation:** Start with 2–4× RTX 4090 for Phases 0–3 (~500 GPU-hours), then rent H100s for Phase 4–5 scaling. Total cost: ~$1,500 cloud + $7K hardware (if purchased).

---

## Key Advantages of Volt for Code

1. **Compositional reasoning:** Functions decompose into slots (args, logic, return)
2. **Multi-step execution:** RAR iterations align with algorithm steps (provable on CLRS)
3. **Deterministic verification:** Hard Strand executes code → test pass/fail
4. **Memory efficiency:** Three-tier memory for API docs, past solutions
5. **Algebraic binding:** Variable scope, function composition via HDC
6. **Safety:** Deterministic axioms catch malicious code (no probabilistic failures)

---

## Success Metrics vs. Transformer Baselines

**Target performance (Volt 500M vs. Transformer 500M, same compute):**
- **HumanEval Pass@1:** Volt 35% vs. Transformer 25% (+10% absolute)
- **MBPP Pass@1:** Volt 30% vs. Transformer 20%
- **APPS (interview):** Volt 20% vs. Transformer 12%
- **MultiPL-E (cross-lingual):** Volt 25% avg vs. Transformer 18% (better transfer)
- **CLRS (algorithm steps):** Volt 80% vs. Transformer 50% (RAR = execution alignment)
- **Compute efficiency:** Volt achieves HumanEval 30% at 1,000 GPU-hours vs. Transformer at 1,500 GPU-hours (1.5× more efficient)

**Publishable claim:**
*"Volt achieves competitive code generation performance (35% HumanEval Pass@1) with 30–50% lower compute cost than transformer baselines, demonstrating architectural advantages in compositional reasoning and multi-step inference."*

---

*This document describes the code-specialized training path to evolve Volt X from a structurally complete pipeline into a functioning code intelligence system, suitable for deployment before acquiring B200 hardware.*
