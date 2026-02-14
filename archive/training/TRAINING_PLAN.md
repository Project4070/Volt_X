# Volt X — Training Plan: From Demo to Cognitive Architecture

## Status Quo

Volt X currently runs end-to-end: text in, TensorFrame, LLL bus, RAR
inference, Hard Core verification, safety check, text out. 573 tests
pass. The pipeline is structurally complete.

But the system is not yet *trained*. The VFN has random Xavier weights.
The codebook has no learned representations. The translator maps words
to frames via deterministic hashing, not learned embeddings. The slot
attention weights are random. The system produces outputs, but those
outputs are mechanistic, not intelligent.

What follows is every component that must be trained — in dependency
order — to turn Volt X from a functioning pipeline into a functioning
cognitive architecture.

---

## Phase 0: Bootstrap (No GPU Required)

**Goal:** Build the data foundation. Nothing can be trained without
input data and a way to save/load learned weights.

**Duration:** 2–3 weeks
**Hardware:** CPU only (any modern machine)

### 0.1 — VFN Checkpoint System

**What:** Save and load VFN weights to disk.
**Why:** Currently, all 525,568 VFN parameters live in memory and are
lost on restart. No training is meaningful without persistence.
**Scope:** Implement `Vfn::save(path)` / `Vfn::load(path)` with a
binary format matching the codebook pattern (magic bytes + version +
raw weights). Include a checksum for integrity validation.
**Deliverable:** Roundtrip test — train one Forward-Forward epoch, save,
load, verify weights are bitwise identical.

### 0.2 — Dataset Pipeline

**What:** A file-based dataset loader for (query, answer) pairs.
**Why:** All current training uses synthetic in-memory data (1,000
hardcoded eval pairs, generated logic puzzles). Real training requires
real data at scale — at minimum 100K pairs for initial VFN training,
1M+ for convergence.
**Scope:**
- Define a simple format: JSONL with `{"query": "...", "answer": "..."}` lines
- Implement a streaming reader that yields `(TensorFrame, TensorFrame)` pairs
- Support sharded files for datasets that exceed memory
- Include a script to convert existing NLP datasets (SCAN, COGS, Natural Questions) to this format

**Deliverable:** Load 100K pairs, iterate through them, confirm
TensorFrame encoding is deterministic and reproducible.

### 0.3 — Codebook Initialization from Real Embeddings

**What:** Populate the 65,536-entry codebook from a real embedding
distribution instead of synthetic vectors.
**Why:** The codebook is Volt X's quantization vocabulary — the
"symbols" it uses to compress continuous frames into discrete codes.
Random vectors make poor symbols. The codebook entries must cover the
actual distribution of frame embeddings that the system will encounter.
**Scope:**
- Encode a large text corpus (e.g., Korean Wikipedia, 1M+ sentences) through the stub translator
- Collect all slot embeddings (16 slots × N sentences = up to 16M vectors)
- Run k-means (k=65,536) on the collected vectors
- Use the centroids as codebook entries
- Rebuild HNSW index

**Deliverable:** Codebook where nearest-neighbor lookup on real text
embeddings returns semantically coherent entries. Quantization error
below a defined threshold.

### 0.4 — Slot Attention Weight Initialization

**What:** Replace random slot attention weights with a structured
initialization that reflects the known slot semantics.
**Why:** Slot attention (16×16 cross-slot Q/K/V) currently uses random
weights. But the slot roles are not arbitrary — Agent (S0) should attend
to Predicate (S1) and Patient (S2), not to Time (S4) or Manner (S5) in
most cases. A structured initialization based on linguistic priors will
dramatically reduce the training needed for attention to converge.
**Scope:**
- Define an initial attention bias matrix based on thematic role co-occurrence
  (e.g., Agent↔Predicate strong, Agent↔Instrument weak)
- Initialize Q/K/V projections so that the softmax attention pattern
  approximates this prior
- Validate that RAR iterations with structured attention converge faster
  than with random attention on synthetic examples

**Deliverable:** Comparison showing structured init converges in fewer
RAR iterations than random init on 1,000 test queries.

---

## Phase 1: Translator Training (Minimal GPU)

**Goal:** Replace the hash-based stub translator with a learned
text→frame encoder. This is the single highest-leverage training task —
everything downstream depends on frame quality.

**Duration:** 3–4 weeks
**Hardware:** 1–2× H100 (or RTX 4090 for smaller experiments)

### 1.1 — Learned Text Encoder

**What:** Train a model that maps tokenized text to TensorFrame slot
embeddings.
**Why:** The stub translator uses deterministic word hashing. This
produces unique vectors per word but captures zero semantics — "dog" and
"canine" map to completely unrelated vectors. A learned encoder must
place semantically similar inputs near each other in the 256-dim slot
space, or the entire downstream pipeline (RAR, codebook quantization,
memory retrieval) operates on noise.
**Scope:**
- Architecture: Lightweight encoder (not a transformer — a CNN or MLP
  operating on subword embeddings) that outputs 16 slots × 256 dims
- Training objective: Contrastive loss — paraphrase pairs should produce
  similar frames, unrelated pairs should produce dissimilar frames
- Data: Paraphrase datasets (PAWS, QQP, Korean paraphrase corpus) +
  hard negatives
- Target: Cosine similarity > 0.8 for paraphrases, < 0.3 for unrelated

**Trainable parameters:** ~2–10M (depending on encoder depth)
**Deliverable:** Translator where `encode("the dog ran")` and
`encode("the canine sprinted")` produce frames with cosine similarity
> 0.8 on the Agent and Predicate slots.

### 1.2 — Learned Text Decoder

**What:** Train the reverse path: TensorFrame → text.
**Why:** Currently, decode works by nearest-neighbor lookup against the
stub translator's vocabulary. A learned decoder enables the system to
generate fluent text from arbitrary frame states, including frames
produced by RAR inference that don't correspond to any single input.
**Scope:**
- Architecture: Per-slot MLP that maps 256-dim embedding → token logits,
  followed by a lightweight assembler that orders slot outputs into
  coherent text
- Training: Reconstruction loss — encode text → frame → decode → text,
  minimize token-level cross-entropy
- Parallel decode (all 16 slots simultaneously) — not autoregressive

**Trainable parameters:** ~2–10M
**Deliverable:** Round-trip reconstruction accuracy > 90% on a held-out
test set (exact token match on content words, allowing variation on
function words).

### 1.3 — Role Grounding

**What:** Train the translator to respect slot semantics — agents go in
S0, predicates in S1, patients in S2, etc.
**Why:** The stub translator assigns words to slots by position (first
word → S0, second → S1). Real language doesn't follow this order. "The
cat was chased by the dog" puts the patient first and the agent last.
The translator must learn semantic role labeling, not positional
assignment.
**Scope:**
- Use SRL-annotated data (PropBank, FrameNet, or Korean equivalent) to
  supervise slot assignment
- Additional loss term: cross-entropy on slot identity prediction
- Validate on sentences with non-canonical word order (passives,
  relative clauses, Korean SOV structure)

**Deliverable:** On 1,000 test sentences with known SRL labels, slot
assignment accuracy > 85%.

---

## Phase 2: Soft Core Training (Moderate GPU)

**Goal:** Train the VFN so that RAR inference converges to correct
answers — the core "thinking" of the system.

**Duration:** 4–6 weeks
**Hardware:** 4–8× H100

### 2.1 — VFN Flow Matching (Primary Training)

**What:** Train the VFN to learn a velocity field that drives query
frames toward answer frames.
**Why:** This is the central training task. The VFN (currently 525,568
params, target 500M–2B) defines the energy landscape that RAR traverses.
An untrained VFN produces random drift. A trained VFN produces drift
that moves query frames toward correct answer frames along smooth
trajectories on the unit hypersphere.
**Scope:**
- Scale VFN from 3-layer MLP (525K params) to target architecture:
  Fourier Neural Operator or deep MLP (500M+ params)
- Training: Flow Matching (Lipman et al., 2022) — learn the constant
  velocity field v(F, t) = F_answer − F_query
- Data: (query_frame, answer_frame) pairs from Phase 0.2 dataset,
  encoded through Phase 1.1 translator
- Curriculum: Start with simple factual pairs, increase to multi-hop
  reasoning
- Loss: MSE between predicted drift and target drift
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Validation: RAR convergence rate on held-out queries — fraction of
  queries where all active slots converge within 50 iterations

**Trainable parameters:** 500M–2B
**Compute estimate:** ~100–500 H100-hours for initial convergence
**Deliverable:** On 10,000 held-out queries, RAR converges (all active
slots ‖ΔS‖ < ε) within 20 iterations for >80% of queries, with cosine
similarity > 0.6 between output frame and ground-truth answer frame.

### 2.2 — Slot Attention Training

**What:** Train the cross-slot attention weights jointly with VFN.
**Why:** Slot attention controls information flow between the 16 slots
during the Attend phase of RAR. Untrained attention means slots don't
communicate meaningfully — the Agent slot doesn't know what the
Predicate slot contains. This must be learned for multi-slot reasoning.
**Scope:**
- Joint training with VFN (end-to-end through RAR iterations)
- Attention should learn that Agent↔Predicate↔Patient form a core
  triangle, Time and Location are contextual modifiers, and free slots
  (S9–S15) attend dynamically based on content
- Regularization: Attention entropy penalty to prevent collapse to
  uniform or one-hot patterns

**Trainable parameters:** ~33K (Q/K/V projections for 16 slots × 256 dims)
**Deliverable:** Attention heatmaps on test queries show linguistically
plausible patterns (Agent attends strongly to Predicate, weakly to
Location).

### 2.3 — Diffusion Controller Tuning

**What:** Train the per-slot noise schedule σ_φ(F, t) that controls
exploration during RAR.
**Why:** Too much noise and RAR never converges. Too little and it gets
stuck in local minima. The noise schedule should be high in early
iterations (explore) and decay as slots converge (exploit). This is
currently a fixed schedule — it should be learned.
**Scope:**
- Small network (MLP, ~10K params) that takes (slot_state, iteration,
  convergence_delta) and outputs noise magnitude σ
- Train alongside VFN using a convergence-speed objective: minimize
  iterations to convergence while maintaining answer quality
- Validate: Compare fixed vs. learned schedule on queries requiring
  multi-hop reasoning

**Trainable parameters:** ~10K
**Deliverable:** Learned diffusion schedule reduces mean RAR iterations
by >20% compared to fixed schedule on hard queries.

---

## Phase 3: Hard Core Calibration (Minimal GPU)

**Goal:** Calibrate the Hard Core so that certainty scores (γ) are
meaningful and the intent router selects the right strand.

**Duration:** 2–3 weeks
**Hardware:** 1× H100 (or CPU-heavy with small GPU)

### 3.1 — Intent Router Calibration

**What:** Train the intent router's strand vectors so that cosine
similarity correctly routes queries to the right Hard Strand.
**Why:** Currently, strand identity vectors (MathEngine, CodeRunner,
HDCAlgebra, etc.) are manually defined. The router works on obvious
cases ("what is 2+2" → MathEngine) but fails on ambiguous queries. The
strand vectors must be learned from labeled routing examples.
**Scope:**
- Collect routing labels: 10K queries labeled with correct strand
- Train strand vectors via contrastive loss: correct strand should have
  highest cosine similarity to query frame
- Handle multi-strand queries (query needs both math and code) via
  top-k routing

**Trainable parameters:** ~10K (one 256-dim vector per strand)
**Deliverable:** Routing accuracy >95% on a held-out test set of 2,000
labeled queries.

### 3.2 — Certainty Calibration (γ)

**What:** Calibrate the certainty engine so that γ scores correspond to
actual correctness probability.
**Why:** γ is the system's self-confidence measure. Uncalibrated, γ=0.9
might mean the system is correct 60% of the time. Calibrated, γ=0.9
should mean ~90% correctness. This is critical for the safety layer
(which triggers on low-γ) and for user trust.
**Scope:**
- Run inference on 10K labeled (query, answer) pairs
- Collect (predicted_answer, γ, is_correct) triples
- Fit a calibration function (Platt scaling or isotonic regression)
  that maps raw γ to calibrated probability
- Integrate into certainty engine as a post-processing step
- Measure Expected Calibration Error (ECE) before and after

**Deliverable:** ECE < 0.05 on held-out test set (currently: uncalibrated).

### 3.3 — Safety Axiom Vector Refinement

**What:** Refine the K1–K5 axiom vectors using adversarial examples.
**Why:** Safety axioms are currently defined as manually constructed HDC
vectors. They catch obvious violations but may miss subtle ones.
Adversarial refinement ensures the safety boundary is tight.
**Scope:**
- Generate adversarial queries designed to evade each axiom (red-teaming)
- For each false negative (harmful query that passed), adjust the axiom
  vector to increase similarity
- For each false positive (safe query that was blocked), adjust to
  decrease similarity
- Maintain axiom immutability constraint: adjustments must be approved
  and frozen after training

**Deliverable:** False negative rate < 1% on adversarial benchmark,
false positive rate < 5% on benign benchmark.

---

## Phase 4: Joint Alignment (Full GPU)

**Goal:** End-to-end fine-tuning of the complete pipeline so that all
components work together, not just individually.

**Duration:** 4–6 weeks
**Hardware:** 8× H100

### 4.1 — End-to-End Flow Matching

**What:** Fine-tune the full pipeline
(Translator → VFN → Attention → Hard Core → Decode) with gradients
flowing through the entire chain.
**Why:** Phases 1–3 train components in isolation. But the translator's
frame encoding must match what the VFN expects. The VFN's output must be
decodable. The attention patterns must complement the VFN drift. Joint
training aligns all components to work as a system.
**Scope:**
- Freeze the translator and decoder (trained in Phase 1)
- Fine-tune VFN + attention + diffusion jointly
- Loss: End-to-end reconstruction (input text → full pipeline → output
  text → cross-entropy against ground truth)
- Curriculum: Simple → complex queries over training

**Compute estimate:** ~200–1,000 H100-hours
**Deliverable:** End-to-end accuracy on held-out benchmark exceeds the
sum-of-parts performance from isolated training by >10%.

### 4.2 — RLVF Alignment

**What:** Apply REINFORCE with Verified Feedback to calibrate the
system's outputs against human-verifiable answers.
**Why:** Flow Matching teaches the VFN to move query frames toward
answer frames. But the quality of answers depends on whether the
training pairs are good. RLVF provides a reward signal from verified
(correct/incorrect) outcomes, allowing the system to learn from its
own mistakes.
**Scope:**
- Use the existing RLVF infrastructure (volt-learn/rlvf.rs)
- Reward table: correct+calibrated → +1.0, correct+underconfident → +0.5,
  wrong+honest → +0.2, wrong+overconfident → -2.0
- Training data: Self-play logic puzzles (expandable) + verified QA pairs
- Target: Reduce overconfident errors (γ > 0.7 on wrong answers) to < 5%

**Deliverable:** Overconfident error rate < 5%, ECE < 0.03.

### 4.3 — Sleep Consolidation Validation

**What:** Validate that the sleep cycle (Forward-Forward + Distillation
+ Graduation) actually improves the system over time.
**Why:** Sleep consolidation is the mechanism for continual learning —
the system should get better with use. This must be empirically
validated: run 10,000 queries, trigger sleep, then re-evaluate. If
performance doesn't improve (or degrades), the consolidation parameters
need tuning.
**Scope:**
- Run a controlled experiment: 10K queries → sleep → re-evaluate on
  held-out set
- Measure: answer quality, convergence speed, γ calibration before and
  after sleep
- Tune Forward-Forward hyperparameters (learning rate, goodness
  threshold, corruption noise) for stable improvement
- Validate no catastrophic forgetting: performance on old queries must
  not degrade >5%

**Deliverable:** Post-sleep performance improves on new queries by >5%
without degrading on old queries by >5%.

---

## Phase 5: Scale and Benchmark (Full GPU)

**Goal:** Scale the trained system and produce publishable benchmark
results against transformer baselines.

**Duration:** 4–8 weeks
**Hardware:** 16–64× H100

### 5.1 — VFN Scaling

**What:** Scale VFN from the initial architecture to the target 500M–2B
parameter range.
**Why:** The 525K-parameter VFN is a proof of concept. To handle
real-world queries with the richness and ambiguity of natural language,
the energy landscape must be far more complex. The architecture document
specifies a Fourier Neural Operator at 500M–2B params as the target.
**Scope:**
- Implement FNO-based VFN (spectral convolutions in slot space)
- Progressive scaling: 50M → 200M → 500M → 2B, validating at each step
- Knowledge distillation from smaller to larger models where possible
- Monitor: convergence speed, answer quality, VRAM usage at each scale

**Compute estimate:** ~1,000–5,000 H100-hours across all scales
**Deliverable:** VFN at 500M+ params that converges on 90%+ of queries
within 15 RAR iterations, with answer quality exceeding the 525K
baseline by >20% on all benchmarks.

### 5.2 — Multi-Corpus Training

**What:** Train on diverse, large-scale corpora beyond the initial
dataset.
**Why:** A cognitive architecture must generalize across domains. Initial
training on clean QA pairs establishes the mechanism. Multi-corpus
training establishes breadth.
**Scope:**
- Corpora: Korean Wikipedia, legal documents, medical texts, code,
  scientific papers, conversational data
- Per-domain codebook refinement (domain-specific quantization)
- Per-domain strand creation via graduation
- Target: 1B–10B tokens total training data

**Compute estimate:** ~500–2,000 H100-hours
**Deliverable:** Cross-domain evaluation showing >60% answer accuracy on
out-of-domain queries (domains not seen in training).

### 5.3 — Benchmark Publication

**What:** Rigorous benchmarking against transformer baselines on tasks
where Volt X has a structural advantage.
**Why:** This is the result that justifies everything — the proof that
the architecture works where the theory predicts it should.
**Scope — Primary benchmarks (structural advantage expected):**
- **SCAN** (compositional generalization): Volt X should excel due to
  algebraic binding
- **COGS** (compositional out-of-distribution): Same structural
  advantage
- **ARC-AGI** (abstract reasoning): RAR's iterative refinement should
  help
- **bAbI** (multi-hop reasoning): Three-tier memory gives persistent
  context

**Scope — Secondary benchmarks (parity expected, not dominance):**
- **MMLU** (general knowledge): Baseline comparison
- **Korean language reasoning** (custom): Demonstrates practical value
  for Korean GPU program

**Baselines:** GPT-2 (small), GPT-4-class (via API), and a comparably
sized transformer trained on the same data with the same compute budget.

**Deliverable:** arXiv paper with benchmark tables showing Volt X
matches or exceeds transformer baselines on compositional and reasoning
tasks at 10–100× lower compute cost.

---

## Summary: Components × Phases

| Component | Phase | Parameters | GPU Hours | Depends On |
|---|---|---|---|---|
| Checkpoint system | 0.1 | — | 0 | — |
| Dataset pipeline | 0.2 | — | 0 | — |
| Codebook init | 0.3 | 65K × 256 | 0 | 0.2 |
| Attention init | 0.4 | ~33K | 0 | — |
| Text encoder | 1.1 | 2–10M | 10–50 | 0.2 |
| Text decoder | 1.2 | 2–10M | 10–50 | 1.1 |
| Role grounding | 1.3 | (shared w/ 1.1) | 10–30 | 1.1 |
| VFN flow matching | 2.1 | 500M–2B | 100–500 | 0.1, 1.1 |
| Slot attention | 2.2 | ~33K | (joint w/ 2.1) | 2.1 |
| Diffusion tuning | 2.3 | ~10K | 10–50 | 2.1 |
| Router calibration | 3.1 | ~10K | 5–20 | 1.1 |
| γ calibration | 3.2 | ~1K | 5–10 | 2.1, 3.1 |
| Safety refinement | 3.3 | ~1.3K | 5–10 | 3.2 |
| E2E fine-tuning | 4.1 | (all above) | 200–1,000 | 1–3 |
| RLVF alignment | 4.2 | (VFN) | 50–200 | 4.1 |
| Sleep validation | 4.3 | (VFN) | 20–50 | 4.2 |
| VFN scaling | 5.1 | 500M–2B | 1,000–5,000 | 4.1 |
| Multi-corpus | 5.2 | (all above) | 500–2,000 | 5.1 |
| Benchmarks | 5.3 | — | 100–500 | 5.1, 5.2 |

## Total Estimated Compute

| Phase | H100-Hours | Calendar Time |
|---|---|---|
| Phase 0: Bootstrap | 0 | 2–3 weeks |
| Phase 1: Translator | 30–130 | 3–4 weeks |
| Phase 2: Soft Core | 110–550 | 4–6 weeks |
| Phase 3: Hard Core | 15–40 | 2–3 weeks |
| Phase 4: Alignment | 270–1,250 | 4–6 weeks |
| Phase 5: Scale | 1,600–7,500 | 4–8 weeks |
| **Total** | **~2,000–10,000** | **~6 months** |

The lower bound (2,000 H100-hours) assumes the 500M VFN converges
efficiently and benchmarks are run on a focused task set. The upper
bound (10,000 H100-hours) assumes scaling to 2B parameters with
multi-corpus training. Both are within the scope of the Korean
government GPU program allocation for startups.

---

*This document describes the training required to evolve Volt X from a
structurally complete pipeline into a functioning cognitive architecture
capable of producing benchmark-publishable results.*
