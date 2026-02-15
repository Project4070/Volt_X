# Volt X — Development History

A chronological record of Volt X's development from initial commit to
current state. Derived from git history (`c29b996..HEAD`).

---

## Day 1 — Feb 9, 2026: Project Bootstrap + Phase 1 Complete

### Project Initialization (`c29b996`)

The Rust workspace was created with 10 crates following a strict
one-directional dependency graph:

```text
core <- bus <- soft/hard/db <- translate/learn/safety/ledger <- server
```

Initial documentation included ARCHITECTURE.md (full technical reference),
DECISIONS.md (ADR log), the Master Blueprint, SVGs.md (visual diagrams),
and per-phase roadmap files. All crates started as stubs with `todo!()`
placeholders.

### Milestone 1.1 — TensorFrame (`e5cc1fa`)

The foundational data structure: `[S=16 x R=4 x D=256]` Tensor Frames.
Implemented serialization (rkyv + serde), certainty-based frame merging,
per-slot L2 normalization, and comprehensive benchmarks. 25 unit tests,
11 integration tests, 26 doc tests. Merge ops at ~23us, normalization
at ~15us.

### Milestone 1.2 — HDC Algebra (`ff1bfe6`)

FFT-based circular convolution for bind/unbind, additive superposition,
cyclic permute, and cosine similarity. Thread-local FFT planner for
performance. All four HDC property requirements verified:
`unbind(bind(a,b),a) ~ b`, constituents detectable in superposition,
bound pairs dissimilar to inputs, different shifts orthogonal.
Bind at ~1.6us (6x faster than 10us target). 47 tests.

### Milestone 1.3 — Stub Translator + HTTP Server (`135ad73`)

First end-to-end text-in/text-out pipeline. StubTranslator with heuristic
word-to-slot mapping (Agent/Predicate/Patient), deterministic hash-based
encoding. Axum 0.8 server with `/health` and `/api/think` endpoints.
57 tests including 100-concurrent-request stress test.

### Milestone 1.4 — n8n Integration (`33e9c58`)

Debug output with per-slot state reporting and timing instrumentation.
n8n workflow for chat UI: Chat Trigger -> HTTP Request -> Switch ->
Debug Panel -> Chat Reply.

**Phase 1 delivered in a single day.** 168 tests passing.

---

## Day 2 — Feb 10, 2026: Phase 1 Audit + Phase 2 + Phase 3

### Phase 1 Audit & Fixes (`a5dc78a`, `06b4415`)

PR #1 identified that Soft Core and Hard Core stubs weren't wired into
the `/api/think` pipeline. Fixed by adding `process_stub()` and
`verify_stub()` pass-throughs and wiring the full
Encode -> Soft Core -> Hard Core -> Bus similarity -> Decode pipeline.
Also fixed `unwrap()` in `normalize_all()` and added Clone for
StubTranslator. 180 tests.

### Milestone 2.1 — VQ-VAE Codebook (`789537b`)

65,536-entry codebook with HNSW index (hnsw_rs). Quantize via nearest
neighbor, lookup by u16 ID. VXCB binary persistence format. Python
initializer using GloVe + PCA + Mini-Batch K-Means.

### Milestone 2.3 — RAR Loop on CPU (`789537b`)

Vector Field Network (VFN): 3-layer MLP (256->512->512->256) with Xavier
init. Cross-slot attention with Q/K/V projections over 16x16 slot matrix.
RAR loop: Root -> Attend -> Refine with per-slot convergence detection,
progressive freezing, and budget enforcement. 288ms for 50 iterations
(target <500ms).

### Milestone 2.4 — GPU Port + Flow Matching (`45b3151`)

GPU-accelerated RAR via candle (feature-gated behind `gpu`). Per-slot
adaptive diffusion noise injection. Flow Matching VFN training with
AdamW optimizer and synthetic pair generation. 208+ tests passing.

### Milestone 2.2 — LLM Forward Translator (`7cc3e4a`)

Qwen3-0.6B frozen backbone with trained Frame Projection Head. 3-layer
MLP (1024->4096->4096->4096) mapping LLM hidden states to 16-class slot
roles + 256-dim embeddings. Feature-gated behind `llm`. Python training
pipeline with CoNLL-2012 SRL support.

### Phase 2 Audit Fixes (`0c032dd`)

Closed 6 gaps: reconstruction loss metric, BLEU-4 scorer, codebook
utilization tracking, Criterion RAR benchmark (CPU+GPU), VFN convergence
rate test (>50% on synthetic data).

### Milestone 3.1 — Intent Router + MathEngine (`f42d515`)

Cosine-similarity-based intent routing with 256-dim capability vectors.
MathEngine Hard Strand parsing arithmetic from frame slots.

### Milestone 3.2 — Hard Core Pipeline (`72fe78f`)

CertaintyEngine (min-rule gamma propagation), ProofConstructor (chain
recording), HDCAlgebra strand (exposes bind/unbind/superpose/permute
via op codes), CodeRunner (sandboxed WASM via wasmtime with fuel limits),
HardCorePipeline integrating all components. 142 tests.

**Phases 2 and 3.1-3.2 delivered on day 2.**

---

## Day 3 — Feb 11, 2026: Phase 3 Complete + Phase 4 Complete

### Milestone 3.3 — Safety Layer (`e305c78`)

Deterministic safety with K1-K5 axiom vectors in HDC space.
TransitionMonitor checks frames against axioms, ViolationScorer
computes aggregate safety, Omega Veto provides hardware-level halt
that cannot be overridden. SafetyLayer wraps full pipeline with
pre/post checks. Replaced Phase 1 stubs with real RAR inference and
safety-wrapped Hard Core in the server. Proof chains and safety scores
now flow to the API response.

### Milestone 4.1 — VoltDB T0 + T1 (`b0a9345`)

T0 WorkingMemory: 64-frame ring buffer with FIFO eviction. T1
StrandStore: strand-organized HashMap with JSON persistence (dedicated
8MB-stack threads to handle 64KB TensorFrames on Windows). VoltStore
facade with auto-eviction, strand management, monotonic frame IDs.
Retrieval by ID <0.1ms across 1000 frames.

### Milestone 4.2 — HNSW + Ghost Bleed (`eee778f`)

Per-strand HNSW semantic search, B-tree temporal index, Ghost Bleed
Buffer with auto-refresh engine, and ghost-aware RAR cross-attention.
Ghost gists flow as plain `[f32; 256]` arrays between volt-db and
volt-soft, preserving the architecture boundary.

### Milestone 4.3 — T2 + GC + Consolidation (`c89d923`)

Full three-tier persistent memory. CompressedFrame, GistFrame,
Tombstone with custom binary codec (serde can't handle `[f32; 256]`).
BloomFilter with splitmix64 double-hashing. Per-strand WAL with CRC32
checksums. LSM-Tree (Memtable + mmap'd SortedRuns + compaction).
GcEngine with 4-tier decay (Full -> Compressed -> Gist -> Tombstone).
ConsolidationEngine for cluster detection and wisdom frame creation.

### Memory Integration (`7eead7a`)

Wired VoltDB into the server: frames stored after inference, ghost gists
influence future RAR passes (alpha=0.1). Fixed StubTranslator to write
R0 discourse for gist extraction and HNSW indexing.

**Phases 3 and 4 delivered on day 3.**

---

## Day 4 — Feb 12, 2026: Phase 5 + Phase 6.1

### Milestone 5.1 — Learning Event Logging (`c49c640`)

LearningEvent captures per-inference telemetry (frame_id, strand_id,
gamma scores, convergence iterations, ghost activations). EventBuffer
with bounded FIFO eviction (10K default). EventLogger with JSON
persistence and per-strand statistics. Wired into `/api/think` with
best-effort logging. 65 tests.

### Milestone 5.2 — Sleep Consolidation (`08015a7`)

SleepScheduler with idle detection and background thread. Forward-Forward
layer-local goodness optimization for VFN training (no backprop through
full network). Frame distillation via HNSW clustering. Strand graduation
for novel topic detection and migration.

### Milestone 5.3 — RLVF Joint Alignment (`1ad921e`)

REINFORCE with shaped rewards: correct+calibrated (+1.0), wrong+honest
(+0.2), wrong+overconfident (-2.0), safety violation (-5.0). EvalDataset
with 1000 QA pairs across 4 categories. CalibrationResult with ECE
metric. 5 self-play logic puzzle types. RLVF runs as optional phase
after Forward-Forward in sleep cycles.

### Milestone 6.1 — Module System (`78ef8cd`)

ModuleInfo/ModuleType in volt-core. ActionCore trait in volt-translate.
`catch_unwind` panic safety in IntentRouter. WeatherStrand example
module (feature-gated). ModuleRegistry with `GET /api/modules`. CLI
`volt modules list|install|uninstall` subcommands.

### Decoder Fix (`2cf8a9a`)

Fixed math engine outputting "[slot1] [slot6] 15." instead of clean "15".
Result slot (S8) detected and extracted cleanly. Also added PATH_TO_AGI.md,
web UI (index.html, app.js, style.css), and CLI chat client.

**Phase 5 and 6.1 delivered on day 4. All 6 implementation phases of the
runtime architecture were complete in 4 days.**

---

## Day 5 — Feb 13, 2026: Training Infrastructure

### Training Plan (`321f893`)

Comprehensive training plan documenting all trainable components across
6 training phases with parameter counts, compute estimates, and
dependency ordering. Created for GPU resource application (NIPA/KAIT).

### Phase 0 — Bootstrap Infrastructure (`a178c52`)

- **0.1**: VFN checkpoint save/load with binary format + checksum
- **0.2**: Code dataset pipeline (CodeDataset, CodeProblem JSONL reader)
- **0.3**: Codebook init pipeline (StackCorpusReader, mini-batch k-means)
- **0.4**: Code attention bias (16x16 additive bias matrix)

### Phase 1 — Learned Translator (`a178c52`)

CNN encoder (5.1M params): 3-layer Conv1D + role head + embed head,
trained on CodeSearchNet Python 100K pairs. Autoregressive decoder
(6.7M params): cross-attention to per-token features, 61% token
accuracy. BPE tokenizer (32K vocab) from The Stack Python sample.
LearnedTranslator implementing the Translator trait.

### Codebook Init Refactor (`09da1ec`)

`codebook_init::init_codebook_from_corpus` now accepts `&dyn Translator`
instead of hardcoding StubTranslator, enabling learned encoder use.
CLI gains `--tokenizer/--encoder/--decoder` flags.

---

## Day 6 — Feb 14, 2026: VFN Training + Cloud Plans + Unification

### Phase 2 — VFN Training Infra (`a5cbcd3`)

ScaledVfn (51M params) with time-conditioned flow matching. code_pairs
module for dataset-to-FramePair conversion. `train-vfn` CLI binary.
Cloud training plan with parallelization strategy.

### Vast.ai Setup Guide (`425253c`)

Step-by-step guide for running Phase 0.3 + 2.1 on a single Vast.ai
instance (RTX 5090 + EPYC Rome 192-core, $0.20/hr). Optimized for
minimal billable prep time (~15 min) with parallel setup tracks.

### The Great Unification (`4a7b294`)

**Critical turning point.** Recognized that the training approach was
drifting from Volt's philosophy through incremental improvisation
across 7 separate contradictory documents. Archived all of them:

- TRAINING_PLAN.md, CODE_TRAINING_PLAN.md, CLOUD_TRAINING_PLAN.md,
  TRAINING_COMMANDS.md, DATA_SETUP_SUMMARY.md, VASTAI_SETUP_GUIDE.md,
  PATH_TO_AGI.md

Created `TRAINING.md` v2.0 as single source of truth. Key pivots:

1. **Language before code** — can't write a weather app without knowing
   what weather means
2. **Not NTP** — slot filling, compositional binding, frame discrimination,
   multi-resolution consistency instead of next-token prediction
3. **Curated datasets** — FrameNet/PropBank/ConceptNet over raw Stack
4. **HF streaming** — no more 44GB bulk downloads
5. **VFN objective** — denoising + slot-conditional flow matching +
   reasoning chains, not flat frame-to-frame mapping

---

## Day 7 — Feb 15, 2026: Spiral Curriculum

### TRAINING.md v3.0 — Spiral Revision

The sequential plan (F1->F2->F3->D1) was replaced with a **spiral
curriculum** inspired by developmental learning theory. Six stages
mirror human cognitive development:

| Stage              | Analogy     | Active Slots | VFN Iterations |
| ---                | ---         | ---          | ---            |
| 1: Naming          | Infant      | 2            | 1              |
| 2: Sentences       | Toddler     | 3-4          | 1-3            |
| 3: Context         | Preschooler | 8            | 3-10           |
| 4: Narrative       | School-age  | 12           | 10-20          |
| 5: Hypotheticals   | Teenager    | 16           | up to 30       |
| 6: Specialization  | Adult       | 16           | up to 30       |

All four training signals (slot filling, binding, discrimination,
multi-resolution consistency) present from Stage 1. Metric-gated
advancement prevents moving forward without mastery. 20% rehearsal
from earlier stages prevents catastrophic forgetting. The VFN gets
~5x more training time compared to the sequential plan.

---

## Summary Statistics

| Metric                  | Value                                          |
| ---                     | ---                                            |
| Development span        | 7 days (Feb 9-15, 2026)                        |
| Total commits           | 38                                             |
| Pull requests           | 4                                              |
| Files changed           | 176                                            |
| Lines added             | ~64,600                                        |
| Crates implemented      | 10 of 10                                       |
| Runtime phases complete | All 6 (Phase 1-6)                              |
| Training infra phases   | Phase 0 complete, training plan v3.0 finalized |
| Tests passing           | 500+ across workspace                          |
| ADRs recorded           | 30 (ADR-001 through ADR-030)                   |

## Architecture Delivered

```text
volt-core       TensorFrame [16x4x256], SlotData, VoltError, ModuleInfo
volt-bus        HDC algebra (bind/unbind/superpose/permute), VQ-VAE Codebook
volt-soft       VFN, SlotAttention, RAR loop (CPU+GPU), Flow Matching, Diffusion
volt-hard       IntentRouter, MathEngine, CodeRunner, HDCAlgebra, CertaintyEngine
volt-db         T0 ring buffer, T1 strand store, T2 LSM-tree, HNSW, Ghost Bleed, WAL, GC
volt-translate  StubTranslator, LlmTranslator, LearnedTranslator (CNN), Decoder
volt-learn      EventLogger, Sleep (FF + Distillation + Graduation), RLVF, Dataset pipelines
volt-safety     K1-K5 axioms, TransitionMonitor, ViolationScorer, Omega Veto
volt-server     Axum HTTP, /api/think, /api/modules, Web UI, CLI chat
volt-ledger     (Stub — P2P sharing, deferred to post-training)
```

## Training Plan Evolution

```text
v1.0 (Code-First)     Code datasets -> VFN flow matching -> hope for generalization
        |
        v  "Can't write a weather app without knowing what weather means"
        |
v2.0 (Sequential)     F1 Lexical -> F2 Knowledge -> F3 Reasoning -> D1 Code
        |
        v  "VFN starves for 6 weeks while encoder trains alone"
        |
v3.0 (Spiral)         Stage 1-6 developmental curriculum, all signals from day 1
```
