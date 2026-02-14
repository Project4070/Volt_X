# Volt X — Unified Training Plan

**Version:** 2.0 (replaces TRAINING_PLAN.md, CODE_TRAINING_PLAN.md, and 5 other
fragmented documents — all archived in `archive/training/`)

**Date:** 2026-02-14

**Status:** Phase 0 infrastructure complete. Phase 1 code translator trained
(reusable artifacts). Training paradigm and curriculum revised — this document
is the single source of truth.

---

## Why This Rewrite Exists

The previous training plans drifted from Volt's architectural philosophy
through incremental improvisation:

1. **Dataset selection was wrong.** Bulk-downloading The Stack (44 GB of raw
   Python files) was the wrong starting point. Raw source files require
   post-processing and don't come with the structured annotations Volt needs.
   Curated, pre-formatted datasets on HuggingFace are better, and streaming
   eliminates the need for full downloads.

2. **HuggingFace streaming was ignored.** The old pipeline required downloading
   entire datasets to disk before training. HF's `datasets` library supports
   `streaming=True`, which iterates over rows directly from the Hub with no
   local storage. For large corpora this is essential.

3. **The training paradigm was drifting toward next-token prediction.** The
   Phase 1 decoder is autoregressive (teacher-forced cross-entropy on BPE
   tokens). The VFN training objective (flow match from query frame to answer
   frame) is functionally equivalent to "predict the right output embedding
   given an input embedding" — which is NTP in disguise with a continuous
   relaxation. This defeats the purpose of Volt's architecture.

4. **Code came before language.** The old plan trained on CodeSearchNet first,
   then hoped to generalize to natural language later. This is backwards. A
   system that doesn't know what "weather" means cannot write a weather app.
   Language understanding must come first; code is a domain specialization
   on top of it.

5. **Plans were fragmented and contradictory.** Seven separate documents
   (TRAINING_PLAN, CODE_TRAINING_PLAN, CLOUD_TRAINING_PLAN, TRAINING_COMMANDS,
   DATA_SETUP_SUMMARY, VASTAI_SETUP_GUIDE, PATH_TO_AGI) each told a different
   story. PATH_TO_AGI suggested "train RAR on next-word prediction" in
   Milestone 5 — directly contradicting Volt's design.

This document unifies everything into one plan that stays true to Volt's
architectural principles.

---

## Part I: How Volt Learns (Not NTP)

### What Transformers Do

Transformers learn by next-token prediction: given a sequence of tokens,
predict the next one. All knowledge — word meaning, world knowledge, reasoning
ability, code syntax — emerges implicitly from this single objective applied
to trillions of tokens.

### What Volt Does Instead

Volt has **structure that transformers don't**: 16 semantic slots, 4
resolutions, HDC algebra, and an iterative refinement loop (RAR). The training
must exploit this structure, not ignore it.

**Four training signals replace NTP:**

#### 1. Slot Filling (Structural Masked Modeling)

Present a frame with some slots populated, others empty or corrupted. Train the
VFN to refine the frame until missing slots converge to correct values.

```
Input:   Agent=chef, Predicate=???, Patient=pasta, Location=kitchen
Target:  Agent=chef, Predicate=cook, Patient=pasta, Location=kitchen
```

This directly mirrors what RAR does at inference time. Training IS practicing
inference.

#### 2. Compositional Binding (HDC Algebra Learning)

Train on (A, relation, B) triples using HDC operations:

```
bind(weather, UsedFor)  ≈  planning         (in HDC space)
bind(chef, performs)    ≈  cooking           (in HDC space)
unbind(result, relation) ≈ original_concept  (retrieval)
```

This teaches the HDC algebra to carry actual semantic content.

#### 3. Frame Discrimination (Energy-Based / Contrastive)

Train the VFN's energy function so that coherent frames have low energy and
incoherent frames have high energy:

```
Low energy:   "chef cooks pasta in kitchen"    (real frame)
High energy:  "chef cooks pasta in algebra"    (corrupted frame)
```

RAR then performs gradient descent on this energy landscape to find coherent
frame states.

#### 4. Multi-Resolution Consistency (Self-Supervised)

R0 (discourse gist) must be consistent with R1 (proposition), R2 (phrase),
R3 (token). If R0 says "cooking topic" and R1 says "algebraic theorem," that's
a contradiction the VFN should learn to resolve. This requires no labels.

### The Decoder Exception

The decoder (frame → text) CAN stay autoregressive. It's a rendering step —
it takes a completed frame and produces human-readable text. This is analogous
to a GPU rasterizer: the 3D scene (frame) is computed via non-sequential
methods, but the final pixel output is sequential. The decoder's NTP is
contained and doesn't infect the upstream pipeline.

---

## Part II: Curriculum — Language First, Domain Second

### Why This Order Matters

```
Foundation A:  Lexical Grounding       (what do words mean?)
               FrameNet, PropBank, AMR
                      ↓
Foundation B:  World Knowledge         (how do concepts relate?)
               ConceptNet, ATOMIC, WordNet
                      ↓
Foundation C:  Compositional Reasoning (multi-step frame transformations)
               SCAN, COGS, CLUTRR, bAbI
                      ↓
Domain:        Code (or any specialization)
               CodeSearchNet, HumanEval, MBPP, tiny-codes
```

Each layer builds on the previous. By the time the system reaches code
training, it already knows:
- "weather" is a natural phenomenon (lexical grounding)
- "forecast" is a prediction about future states (world knowledge)
- "app" is a software instrument used by agents (world knowledge)
- Given premises, derive conclusions via frame refinement (reasoning)

Writing a weather app then becomes composing known concepts — not memorizing
code patterns from (problem, solution) text pairs.

### What This Means for Existing Work

| Existing Artifact | Status | Role in New Plan |
|---|---|---|
| VFN Checkpoint System (Phase 0.1) | Reusable as-is | Infrastructure |
| Code Dataset Pipeline (Phase 0.2) | Reusable, extend for new formats | Infrastructure |
| Code Attention Bias (Phase 0.4) | Reusable for code phase | Used in Domain phase |
| BPE Tokenizer (32K vocab) | Reusable as-is | Shared across all phases |
| CNN Encoder (5.1M params) | Architecture reusable | Retrain on language data, then fine-tune on code |
| Autoregressive Decoder (6.7M) | Architecture reusable | Retrain after encoder; role unchanged (rendering) |
| Codebook Init Pipeline | Reusable | Run after Foundation phases, not before |
| ScaledVfn (51M params) | Architecture reusable | New training objectives (not flat flow matching) |
| training_config.toml | Needs update | Add language dataset paths |
| Vast.ai / cloud setup | Outdated | Rewrite when training approach is implemented |

---

## Part III: Training Phases

### Phase 0: Infrastructure (DONE, except updates)

All infrastructure from the original Phase 0 is reusable:

- **0.1 VFN Checkpoint System** — DONE. Save/load works.
- **0.2 Dataset Pipeline** — DONE for JSONL. Needs extension:
  - Add HuggingFace streaming adapter (Python preprocessing script that
    streams from HF Hub → writes local JSONL in Volt's format)
  - Add FrameNet/PropBank/ConceptNet format converters
  - Add slot-annotated data format: JSONL with `{"text": "...", "slots": {"S0": "agent_word", "S1": "predicate_word", ...}}`
- **0.3 Codebook Init** — Code DONE, k-means deferred. Still valid approach,
  run after Foundation B when the encoder produces meaningful embeddings.
- **0.4 Attention Bias** — DONE for code-specific patterns. Add a general
  language bias matrix (Agent↔Predicate strong, Agent↔Patient medium) as the
  default, with code bias applied on top during code fine-tuning.

**New infrastructure needed:**

- **0.5 HuggingFace Streaming Script** — Python script in `tools/` that:
  - Accepts a HuggingFace dataset name and split
  - Streams rows via `datasets.load_dataset(..., streaming=True)`
  - Converts to Volt's JSONL format
  - Writes to stdout or a file
  - Handles FrameNet, PropBank, ConceptNet, SCAN, COGS, CodeSearchNet,
    tiny-codes, etc. via format-specific converters
  - No full dataset download required

---

### Phase F1: Lexical Grounding

**Goal:** Teach the system what words mean in slot-structured space.

**Duration:** 2-3 weeks
**Hardware:** 1x RTX 4090/5090, ~10-20 GPU-hours

#### What This Phase Teaches

The stub translator maps words to vectors via hash — "dog" and "canine" are
completely unrelated. After this phase:
- Semantically similar words produce similar embeddings
- Words are assigned to correct semantic slots (Agent in S0, Predicate in S1)
- The system understands thematic roles, not just token identity

#### Training Objective

**Slot Assignment** (supervised): Given a sentence with known SRL annotations,
train the encoder's Role Head to classify each token into the correct slot.
Loss: cross-entropy on slot labels from FrameNet frame elements.

**Semantic Embedding** (contrastive): Paraphrase pairs should produce similar
frame embeddings, unrelated pairs should produce dissimilar ones. Loss: InfoNCE
with temperature τ=0.07 (same as existing encoder).

**Slot Filling** (self-supervised): Mask random slots in encoded frames, train
VFN to reconstruct them. This begins VFN training from the very first phase.

#### Datasets (all streamable from HuggingFace)

| Dataset | Size | What It Provides | HF Name |
|---|---|---|---|
| FrameNet 1.7 | ~200K annotated sentences | Semantic frames with named slots (frame elements) | `framenet_v17` or via NLTK |
| PropBank / CoNLL-2012 | ~1M predicate-argument annotations | Arg0 (agent), Arg1 (patient), ArgM-LOC, ArgM-TMP, ArgM-MNR, ArgM-CAU | `conll2012_ontonotesv5` |
| STS Benchmark | 8.6K sentence pairs with similarity scores | Paraphrase / semantic similarity supervision | `sentence-transformers/stsb` |
| PAWS | 108K paraphrase pairs | Hard paraphrase detection (same words, different meaning) | `google-research-datasets/paws` |

**Slot mapping from FrameNet/PropBank to TensorFrame:**

| FrameNet Element / PropBank Arg | TensorFrame Slot |
|---|---|
| Agent / Arg0 | S0 (Agent) |
| Predicate / verb | S1 (Predicate) |
| Patient / Arg1 | S2 (Patient) |
| Location / ArgM-LOC | S3 (Location) |
| Time / ArgM-TMP | S4 (Time) |
| Manner / ArgM-MNR | S5 (Manner) |
| Instrument / Arg2 | S6 (Instrument) |
| Cause / ArgM-CAU | S7 (Cause) |
| Result / Arg3-4 | S8 (Result) |
| Other / ArgM-* | S9-S15 (Free) |

This is not a coincidence — TensorFrame's slot design was inspired by thematic
role theory. FrameNet and PropBank are the ground truth for it.

#### Architecture

Reuse the existing CNN encoder architecture (5.1M params):
- Input: BPE tokens (32K vocab, existing tokenizer)
- 3x Conv1D (128→256) with increasing kernel sizes
- Role Head: Linear(256, 16) → softmax (slot assignment)
- Embed Head: Linear(256, 256) (per-token embedding)
- Slot aggregation → L2-norm → 16x256 TensorFrame

Retrain from scratch on language data (not fine-tune from code weights —
language is the foundation, code comes later).

#### Deliverables

- Slot assignment accuracy >80% on FrameNet/PropBank test set
- Paraphrase cosine similarity >0.75 (STS-B correlation >0.70)
- VFN slot-filling reconstruction >60% on masked slots

---

### Phase F2: World Knowledge

**Goal:** Teach the system how concepts relate using HDC algebra.

**Duration:** 2-3 weeks
**Hardware:** CPU-heavy + light GPU, ~20-30 GPU-hours

#### What This Phase Teaches

After lexical grounding, the system knows what individual words mean. Now it
learns relationships between concepts:
- "weather" UsedFor "planning activities"
- "umbrella" UsedFor "protection from rain"
- "rain" IsA "precipitation"
- "chef" CapableOf "cooking"

These relationships are encoded as HDC operations:
```
bind(weather, UsedFor) ≈ planning_activities   (in the same 256-dim space)
```

#### Training Objective

**Relational Binding** (supervised): Given (concept_A, relation, concept_B)
triples, minimize: `‖bind(embed(A), embed(relation)) - embed(B)‖²`

**Retrieval via Unbind** (supervised): Given bind result and relation, retrieve
the original concept: `unbind(bind(A, rel), rel) ≈ A`

**Category Superposition** (supervised): Given category members, their
superposition should represent the category:
`superpose(dog, cat, fish) ≈ animal`

**Negative Sampling** (contrastive): Random (A, relation, B') triples where
B' is wrong should produce high distance after binding.

#### Datasets (all streamable)

| Dataset | Size | What It Provides | HF Name |
|---|---|---|---|
| ConceptNet 5 | 3.5M triples (EN subset ~1.5M) | Commonsense relations (UsedFor, IsA, HasProperty, CapableOf, etc.) | `conceptnet5` or direct CSV |
| ATOMIC 2020 | 1.33M if-then tuples | Social/physical commonsense (xNeed, xEffect, xWant, oReact) | `allenai/atomic` |
| WordNet (via NLTK) | 117K synsets, 207K word-sense pairs | Hypernyms, hyponyms, meronyms, holonyms | NLTK `wordnet` |

#### Architecture

No new neural components. This phase trains:
- The encoder's embedding space (joint fine-tune with Phase F1 weights)
- HDC codebook alignment (codebook entries drift toward real concept clusters)
- VFN energy landscape (correct bindings = low energy states)

#### Deliverables

- Bind retrieval accuracy >50% on ConceptNet test split
  (bind(A, rel) → nearest neighbor = B)
- Analogy accuracy >35% (A:B::C:? via bind/unbind algebra)
- Codebook quantization error <0.15 on concept embeddings

---

### Phase F3: Compositional Reasoning

**Goal:** Train the VFN to perform multi-step frame transformations — the
core of RAR.

**Duration:** 4-6 weeks
**Hardware:** 1x H100 or 2x RTX 4090, ~100-200 GPU-hours

#### What This Phase Teaches

This is the central training phase. The VFN learns to "think" — to take a
frame representing a question or premise and iteratively refine it toward a
frame representing the answer or conclusion.

This is NOT flat flow matching (query_frame → answer_frame). Instead:

#### Training Objectives (Three Complementary Signals)

**1. Denoising Slot Refinement (Primary)**
- Take a real frame F from the corpus
- Corrupt it: zero random slots, add Gaussian noise, shuffle slot assignments
- Train VFN to predict the drift vector that restores the original frame
- This directly trains the velocity field that RAR follows at inference time
- Loss: per-slot MSE between predicted drift and (F_original - F_corrupted)

**2. Slot-Conditional Flow Matching (Secondary)**
- Flow match per-slot with slot-identity constraints
- Time variable t_s is per-slot (different slots converge at different rates)
- Constraint: Agent-slot drift must point toward Agent-like embeddings, not
  arbitrary vectors
- Loss: MSE + slot-identity cosine penalty

**3. Reasoning Chain Supervision (Tertiary)**
- Given (premise, step_1, step_2, ..., conclusion), train VFN to produce
  one refinement step at a time
- NOT: map question directly to answer (that's disguised NTP)
- YES: map frame(t) to frame(t+1) where each step is a valid reasoning move
- Datasets with intermediate steps: CLUTRR (family relation chains),
  bAbI (multi-hop QA with supporting facts), SCAN (compositional commands
  with intermediate parses)
- Loss: MSE on per-step drift

#### Datasets (all streamable or generatable)

| Dataset | Size | What It Provides | HF Name |
|---|---|---|---|
| SCAN | 20K compositional commands | Compositional generalization (jump twice and walk left) | `scan` |
| COGS | 24K sentences | Systematic compositional OOD generalization | `cogs` |
| CLUTRR | 5-15K per complexity level | Multi-hop relational reasoning (family trees) | `CLUTRR/v1.0` or generate |
| bAbI QA | 10K per task (20 tasks) | Multi-hop reasoning, path-finding, counting | `facebook/babi_qa` |
| Self-generated denoising | Unlimited | Corrupt any frame from F1/F2 corpus, train to reconstruct | Generated online |

#### Architecture

ScaledVfn (51M params) — existing architecture with modifications:
- **Time conditioning**: Existing sinusoidal embedding, now per-slot
- **Slot identity head**: New small head (Linear(256, 16)) that predicts
  which slot type a given embedding belongs to. Added to enforce structural
  constraints during denoising.
- **Per-slot convergence**: Existing mechanism in RAR, now used during
  training to let easy slots freeze while hard slots continue training.

#### Deliverables

- SCAN compositional generalization: >85% accuracy
- bAbI multi-hop (tasks 2, 3): >75% accuracy
- Frame denoising: >65% exact slot reconstruction (within cosine sim >0.9)
- RAR convergence: >70% of test queries converge within 30 iterations

---

### Phase D1: Code Domain Specialization

**Goal:** Extend Volt's language understanding to code as a domain.

**Duration:** 3-4 weeks
**Hardware:** 1x RTX 4090/5090, ~10-20 GPU-hours (encoder fine-tune) +
100-200 GPU-hours (VFN fine-tune on code)

#### Prerequisites

Phases F1-F3 must be complete. The system must understand language and
compositional reasoning before it can understand code.

#### What This Phase Teaches

Code is a structured language with known slot mappings:

| Code Element | TensorFrame Slot | Example |
|---|---|---|
| Function/class name | S0 (Agent) | `def sort_array` |
| Operation/method | S1 (Predicate) | `sorted()`, `append()` |
| Arguments/parameters | S2 (Patient) | `(arr, reverse=True)` |
| Return value | S3 (Location*) | `return sorted_arr` |
| Loop/iteration | S4 (Time) | `for i in range(n)` |
| Algorithm pattern | S5 (Manner) | recursive, iterative, DP |
| Control flow | S6-S8 (Instrument/Cause/Result) | if/else, try/except |
| Complex logic | S9-S15 (Free) | nested structures |

*Location slot is overloaded for code to mean "where the result goes."

#### Training Objective

**Code Slot Assignment** (supervised): Fine-tune the Role Head from Phase F1
to classify code tokens into code-specific slots. Uses the existing heuristic
role labels from Phase 0.4 (def→S0, args→S2, return→S3, if→S6, for→S4).

**Code Semantic Embedding** (contrastive): Same InfoNCE as Phase F1 but on
code pairs. `sum_array()` ≈ `calculate_total()` in embedding space.

**Code Denoising** (VFN fine-tune): Apply the Phase F3 denoising objective to
code frames. Mask the function-name slot, train VFN to predict it from
arguments + body structure.

**Compositional Code Reasoning** (VFN fine-tune): Given a problem description
frame, refine toward a solution frame. This uses flow matching BUT with
slot-conditional constraints from Phase F3 — not flat frame-to-frame mapping.

#### Datasets (curated, pre-formatted, streamable)

| Dataset | Size | What It Provides | HF Name |
|---|---|---|---|
| CodeSearchNet (Python) | 100K pairs | Function-docstring pairs (contrastive) | Already downloaded (144 MB) |
| tiny-codes | 1.6M pairs | Synthetic instruction→code (pre-formatted) | `nampdn-ai/tiny-codes` |
| CodeAlpaca | 20K pairs | Instruction→code (pre-formatted) | `sahil2801/CodeAlpaca-20k` |
| code_instructions_122k | 122K pairs | Instruction→code (pre-formatted) | `TokenBender/code_instructions_122k_alpaca_style` |
| HumanEval | 164 problems | Evaluation benchmark (with tests) | Already downloaded |
| MBPP | 974 problems | Evaluation benchmark (with tests) | Already downloaded |
| TACO | 26K problems | Competitive programming (with tests) | `BAAI/TACO` |

**NOT used:** The Stack (44 GB of raw Python files). Raw source files are
useful for BPE tokenizer training (already done) and codebook initialization,
but not for VFN training. Pre-formatted (instruction, solution) pairs are
more useful per byte.

#### Code Attention Bias

The existing code attention bias matrix (ADR-029) is applied on top of the
general language attention weights learned in Phase F1:

```
Final attention = learned_language_weights + code_attention_bias
```

This preserves linguistic priors (Agent↔Predicate) while adding code-specific
ones (Function↔Arguments, Return↔Operation).

#### Deliverables

- Code slot assignment accuracy >85%
- Code paraphrase similarity >0.75
- RAR convergence on code problems >60% within 30 iterations
- Decoded code is syntactically valid >80% of the time
- Deferred target: Pass@1 >25% on HumanEval after joint alignment (Phase JA)

---

### Phase JA: Joint Alignment

**Goal:** End-to-end calibration of the full pipeline.

**Duration:** 4-6 weeks
**Hardware:** 1-2x H100, ~200-400 GPU-hours

#### JA.1 — Codebook Initialization

**When:** First step of Joint Alignment, after F1-F3 and D1 are complete.

Run the existing k-means pipeline (`codebook-init` binary) using the Phase D1
encoder to produce embeddings, then cluster into 65,536 codebook entries. This
is a CPU-only job (~1-2 hours on 32+ cores). The codebook is needed for HNSW
retrieval in subsequent steps.

Output: `checkpoints/codebook.bin` (~64 MB)

#### JA.2 — Intent Router Calibration

Train strand capability vectors so cosine routing selects the correct Hard
Strand. 10K labeled queries across strands (MathEngine, CodeRunner,
CodeDebugger, HDCAlgebra, etc.).

Training: contrastive loss on 256-dim strand vectors.
Deliverable: routing accuracy >90% on held-out test set.

#### JA.3 — Certainty Calibration (gamma)

Run inference on 5K problems with known answers. Collect (prediction, gamma,
is_correct) triples. Fit isotonic regression to map raw gamma to calibrated
P(correct).

Deliverable: Expected Calibration Error (ECE) < 0.08.

#### JA.4 — Safety Axiom Refinement

Refine K1-K5 axiom vectors using adversarial examples. Minimize false negatives
(dangerous content that passes) while keeping false positives (safe content
blocked) below 5%.

Deliverable: false negative rate <2%, false positive rate <5%.

#### JA.5 — End-to-End Fine-Tuning

Fine-tune VFN + attention with gradients flowing through the full pipeline.
Freeze encoder/decoder (trained in F1/D1). Use the denoising + slot-conditional
objectives from Phase F3, but on end-to-end pipeline outputs.

For code: add test-execution reward signal via REINFORCE (test pass/fail is
perfect verification — no human labeling needed).

#### JA.6 — RLVF Alignment

REINFORCE with Verified Feedback, using the existing infrastructure in
`volt-learn/rlvf.rs`. Reward shaping:

| Outcome | Reward |
|---|---|
| Correct + calibrated (gamma matches accuracy) | +1.0 |
| Correct + underconfident | +0.5 |
| Wrong + honest (low gamma) | +0.2 |
| Wrong + overconfident (high gamma) | -2.0 |
| Safety violation | -5.0 |

Deliverable: overconfident error rate <8%, ECE <0.06.

#### JA.7 — Sleep Consolidation Validation

Run 5K inference queries, trigger sleep cycle (Forward-Forward + Distillation +
Graduation + GC), re-evaluate. Validate that post-sleep performance improves
without catastrophic forgetting.

Deliverable: >5% improvement on new queries, <3% degradation on old queries.

---

### Phase S: Scale & Benchmark

**Goal:** Scale VFN, add domains/languages, publish benchmarks.

**Duration:** 4-8 weeks
**Hardware:** 2-4x H100 or multi-GPU node, ~500-1500 GPU-hours

#### S.1 — VFN Scaling (51M → 200M → 500M)

Progressive scaling with knowledge distillation. Implement Fourier Neural
Operator architecture at 500M params. Validate at each scale point.

#### S.2 — Multi-Domain Training

Extend beyond code to multiple domains using the same Foundation weights:
- Natural language QA (Natural Questions, SQuAD)
- Mathematics (GSM8K, MATH)
- Scientific reasoning (SciQ, ARC-Challenge)
- Multi-lingual code (MultiPL-E: Python → JS, Java, Rust, Go)

Each domain reuses Foundation phases F1-F3 and adds domain-specific fine-tuning
(same pattern as Phase D1 for code).

#### S.3 — Benchmark Publication

Primary benchmarks (structural advantage expected):
- **SCAN/COGS**: Compositional generalization (Volt's HDC binding advantage)
- **bAbI**: Multi-hop reasoning (Volt's three-tier memory advantage)
- **ARC-AGI**: Abstract reasoning (RAR's iterative refinement advantage)

Secondary benchmarks (parity expected):
- **HumanEval / MBPP**: Code generation (Pass@1)
- **MMLU**: General knowledge (baseline comparison)

Baselines: comparably-sized transformer trained on the same data with the same
compute budget.

Deliverable: arXiv paper demonstrating Volt's advantages on compositional and
reasoning tasks at lower compute cost.

---

## Part IV: Dataset Strategy

### Principles

1. **Curated over raw.** Pre-formatted (input, output) pairs > raw source
   files. 20K high-quality CodeAlpaca pairs are more useful than 6.5M raw
   Python files from The Stack.

2. **Stream, don't download.** Use `datasets.load_dataset(..., streaming=True)`
   for anything over 1 GB. Write a Python script that streams and converts to
   Volt's JSONL format.

3. **Structured over flat.** Prefer datasets with annotations (SRL labels,
   semantic roles, relation types) over plain text. Volt's architecture
   requires structured supervision.

4. **Small and dense > large and sparse.** 100K densely-annotated FrameNet
   sentences are worth more than 10M unlabeled sentences for Volt's training.

### Dataset Summary by Phase

| Phase | Primary Datasets | Total Size | Format |
|---|---|---|---|
| F1: Lexical | FrameNet, PropBank/CoNLL-2012, STS-B, PAWS | ~1.5M annotated sentences | Stream from HF |
| F2: Knowledge | ConceptNet, ATOMIC, WordNet | ~5M triples | Stream or download CSV |
| F3: Reasoning | SCAN, COGS, CLUTRR, bAbI, self-generated denoising | ~200K examples + unlimited generated | Stream from HF |
| D1: Code | CodeSearchNet, tiny-codes, CodeAlpaca, code_instructions, HumanEval, MBPP, TACO | ~1.9M pairs | Stream or already downloaded |
| JA: Alignment | Reuse D1 evaluation sets + generated adversarial | ~20K labeled | Generated |
| S: Scale | Natural Questions, GSM8K, MATH, SciQ, MultiPL-E | ~500K+ | Stream from HF |

### What To Keep From Existing Downloads

| Downloaded Dataset | Size | Keep? | Reason |
|---|---|---|---|
| The Stack Python (44 GB) | 6.5M files | Keep for codebook init & BPE only | Raw files, not useful for VFN training |
| CodeSearchNet (144 MB) | 100K pairs | Keep | Pre-formatted, used in D1 |
| HumanEval (0.2 MB) | 164 problems | Keep | Evaluation benchmark |
| MBPP (0.1 MB) | 257 problems | Keep | Evaluation benchmark |
| APPS (1.3 GB) | 10K problems | Keep | Evaluation + RLVF training |
| MultiPL-E (3 MB) | 2.4K problems | Keep | Cross-lingual evaluation |

### HuggingFace Streaming Script (to implement)

```python
# tools/stream_dataset.py — example usage
# Stream FrameNet and convert to Volt JSONL format
python tools/stream_dataset.py \
    --dataset framenet_v17 \
    --format slot_annotated \
    --output data/framenet_train.jsonl \
    --max-examples 200000

# Stream ConceptNet triples
python tools/stream_dataset.py \
    --dataset conceptnet5 \
    --format triples \
    --output data/conceptnet_en.jsonl \
    --language en

# Stream tiny-codes for code training
python tools/stream_dataset.py \
    --dataset nampdn-ai/tiny-codes \
    --format code_pairs \
    --output data/tiny_codes.jsonl \
    --max-examples 500000
```

---

## Part V: Operational Details

### Compute Estimates

| Phase | GPU-Hours | Hardware | Est. Cost (Cloud) |
|---|---|---|---|
| F1: Lexical Grounding | 10-20 | 1x RTX 4090 | $5-10 |
| F2: World Knowledge | 20-30 | CPU + light GPU | $10-20 |
| F3: Compositional Reasoning | 100-200 | 1x H100 | $250-500 |
| D1: Code Domain | 110-220 | 1x H100 | $275-550 |
| JA: Joint Alignment | 200-400 | 1-2x H100 | $500-1000 |
| S: Scale & Benchmark | 500-1500 | 2-4x H100 | $1250-3750 |
| **Total** | **~940-2370** | | **~$2300-5800** |

These estimates assume the new training objectives converge efficiently. The
previous plan estimated ~930-1790 H100-hours for the code-only path; the new
plan adds language foundation phases (~130-250 hours) but may reduce code
training time by providing better initialization.

### Checkpoint Inventory

| Checkpoint | Produced By | Size | Path |
|---|---|---|---|
| BPE Tokenizer | Phase 0 (done) | 2.3 MB | `checkpoints/code_tokenizer.json` |
| Language Encoder | Phase F1 | ~20 MB | `checkpoints/lang_encoder.safetensors` |
| Language Decoder | Phase F1 | ~21 MB | `checkpoints/lang_decoder.safetensors` |
| Code Encoder (fine-tune) | Phase D1 | ~20 MB | `checkpoints/code_encoder.safetensors` |
| Code Decoder (fine-tune) | Phase D1 | ~21 MB | `checkpoints/code_decoder.safetensors` |
| VFN (Foundation) | Phase F3 | ~200 MB | `checkpoints/vfn_foundation.safetensors` |
| VFN (Code fine-tune) | Phase D1 | ~200 MB | `checkpoints/vfn_code.safetensors` |
| Codebook | Phase JA.1 | ~64 MB | `checkpoints/codebook.bin` |
| VFN (Aligned) | Phase JA | ~200 MB | `checkpoints/vfn_aligned.safetensors` |
| VFN (Scaled, 500M) | Phase S.1 | ~2 GB | `checkpoints/vfn_scaled.safetensors` |

### Cloud Execution

Cloud setup and parallelization details will be written as a separate
operational guide when the new training binaries are implemented. The core
approach (tmux sessions, checkpoint sync, per-epoch saves for preemption
recovery) from the archived CLOUD_TRAINING_PLAN.md remains valid — only the
training commands and data requirements change.

### What Needs to Be Implemented in Code

| Item | Crate | Description |
|---|---|---|
| HF streaming script | `tools/stream_dataset.py` | Python script for dataset conversion |
| FrameNet/PropBank data loader | `volt-learn` | JSONL reader for slot-annotated sentences |
| ConceptNet triple loader | `volt-learn` | JSONL reader for (concept, relation, concept) triples |
| Denoising training loop | `volt-soft` or `volt-learn` | Corrupt frames, train VFN to restore |
| Slot-conditional flow matching | `volt-soft` | Per-slot time variables + slot-identity constraint |
| Reasoning chain trainer | `volt-learn` | Multi-step refinement training |
| Language attention bias | `volt-soft` | General linguistic slot attention prior |
| `train-foundation` binary | `volt-learn` | Training binary for F1-F3 phases |

---

## Part VI: Success Criteria

### Phase F1 (Lexical Grounding)
- Slot assignment accuracy >80% on PropBank test set
- STS-B Spearman correlation >0.70
- "dog" and "canine" cosine similarity >0.7 in Agent slot (S0)

### Phase F2 (World Knowledge)
- bind(A, rel) retrieval accuracy >50% on ConceptNet test
- A:B::C:? analogy accuracy >35%

### Phase F3 (Compositional Reasoning)
- SCAN length generalization >85%
- bAbI tasks 1-3 accuracy >75%
- Frame denoising slot reconstruction cosine sim >0.9 for >65% of slots

### Phase D1 (Code)
- Code slot assignment accuracy >85%
- Code paraphrase similarity >0.75
- RAR convergence >60% within 30 iterations

### Phase JA (Alignment)
- ECE <0.08
- Overconfident error rate <8%
- Post-sleep improvement >5%, forgetting <3%

### Phase S (Scale)
- SCAN/COGS: >90% compositional generalization
- HumanEval Pass@1 >25% (at 500M params)
- Compute efficiency: match transformer baseline at lower GPU-hours

---

## Appendix: Comparison with Previous Approach

| Aspect | Previous Plan | This Plan |
|---|---|---|
| **Starting point** | Code (CodeSearchNet) | Language (FrameNet, PropBank) |
| **VFN objective** | Flat flow matching (query→answer) | Denoising + slot-conditional + reasoning chains |
| **Primary training signal** | MSE on frame-to-frame drift | Slot filling + compositional binding + energy-based discrimination |
| **NTP presence** | Decoder (intentional) + VFN objective (accidental) | Decoder only (intentional, contained) |
| **Dataset strategy** | Download The Stack (44 GB), local JSONL | Stream from HF, curated pre-formatted datasets |
| **Curriculum** | Code first, maybe language later | Language → knowledge → reasoning → code |
| **Number of plan documents** | 7 (contradictory) | 1 (this document) |
| **Consistency with ARCHITECTURE.md** | Drifting (PATH_TO_AGI suggests NTP) | Aligned (all training uses Volt's native structures) |

---

*This document is the single source of truth for Volt X training. All previous
training plans are archived in `archive/training/`. When implementing training
code, reference this document and `training_config.toml` for dataset paths.
When the approach changes, update THIS document — do not create a new one.*
