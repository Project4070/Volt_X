# Architecture Decision Records

## ADR-001: Workspace Structure (2026-02-09)
**Decision:** Use Cargo workspace with one crate per component.
**Reason:** Independent compilation, independent testing,
enforced dependency direction.
**Alternatives considered:** Single crate with modules (rejected:
circular dependencies too easy), separate repos (rejected: too much
overhead for solo dev).

## ADR-002: TensorFrame Dimensions (2026-02-09)
**Decision:** S=16 slots, R=4 resolutions, D=256 dims per slot.
**Reason:** 16 slots covers all semantic roles with 7 free. 4 resolutions
span discourse→token. 256 dims balances expressiveness with compute cost.
Total max size 64KB fits comfortably in cache lines.
**Alternatives considered:** S=8 (too few free slots), S=32 (excessive
for most queries), D=512 (doubles compute for marginal expressiveness).

## ADR-003: Error Handling (2026-02-09)
**Decision:** VoltError enum in volt-core, thiserror derivation,
no unwrap in library code.
**Reason:** Consistent error handling prevents silent failures.
thiserror provides ergonomic error types. Banning unwrap forces
explicit error handling at every boundary.

## ADR-004: Split-Brain Architecture (2026-02-09)
**Decision:** GPU Soft Core (neural intuition) + CPU Hard Core
(deterministic logic) communicating via TensorFrame Bus.
**Reason:** GPU does intuition (parallel SIMD), CPU does logic
(branching). Math is computed not predicted (zero hallucination).
RAM becomes living memory. Runs on consumer hardware.
**Alternatives considered:** Everything on GPU (old Volt v2.0 approach,
rejected: needs datacenter GPUs, CPU idle, RAM wasted).

## ADR-005: Root-Attend-Refine (RAR) Inference (2026-02-09)
**Decision:** Three-phase iterative inference: Root (parallel slot-local),
Attend (cross-slot O(S²)), Refine (update + convergence check).
**Reason:** Per-slot convergence allows adaptive computation. Converged
slots freeze, reducing compute per iteration. Embarrassingly parallel
Root phase on GPU. Attention is O(S²) where S=16, not O(n²) where n=100K.
**Alternatives considered:** Standard transformer attention (rejected:
O(n²) is too expensive for consumer hardware).

## ADR-006: Forward-Forward Training (2026-02-09)
**Decision:** Use Hinton's Forward-Forward algorithm for continual learning
instead of standard backpropagation.
**Reason:** FF uses ~24x less VRAM than backprop (only one layer loaded
at a time). Train VRAM ≈ Inference VRAM. Consumer RTX 4060 is sufficient.
**Tradeoff:** ~3x slower training, but VRAM savings make local training viable.
**Alternatives considered:** Standard backprop (rejected: needs A100-class
GPUs for 500M param VFN).

## ADR-007: Safety as Code, Not Weights (2026-02-09)
**Decision:** Safety layer runs on CPU with deterministic Rust logic.
Axiomatic guard with immutable axioms. Omega Veto as hardware-level halt.
**Reason:** Neural safety is probabilistic and gameable. Code safety is
provable and auditable. The Omega Veto cannot be overridden by any
neural component.
**Alternatives considered:** RLHF-style safety (rejected: probabilistic,
can be jailbroken, not suitable for sovereign AI).

## ADR-008: Rust Edition 2024 (2026-02-09)
**Decision:** Use Rust edition 2024 for all crates.
**Reason:** Latest stable edition with improved ergonomics.

## ADR-009: VoltError::TranslateError Variant (2026-02-09)
**Decision:** Add `TranslateError { message: String }` variant to VoltError.
**Reason:** Translate operations have distinct failure modes (empty input,
oversized input, vocabulary lock errors) that should be distinguishable
from FrameError or BusError. Follows the existing pattern of per-domain
error variants (BusError, StorageError, etc.).
**Alternatives considered:** Reuse FrameError (rejected: misleading context),
separate TranslateError type (rejected: breaks unified VoltError pattern).

## ADR-010: Axum 0.8 for HTTP Server (2026-02-09)
**Decision:** Use Axum 0.8 as the HTTP framework for volt-server.
**Reason:** Axum is the most popular async Rust web framework, built on
tokio and tower. Zero-cost routing, type-safe extractors, excellent
ecosystem. `build_app()` returns a Router testable via tower oneshot
without starting a TCP listener.
**Alternatives considered:** Actix-web (rejected: different async runtime,
not tower-native), Warp (rejected: less ergonomic, smaller community),
raw hyper (rejected: too low-level for milestone pace).

## ADR-011: hnsw_rs for Codebook HNSW Index (2026-02-10)

**Decision:** Use `hnsw_rs` 0.3 as the HNSW index library for the VQ-VAE
codebook in volt-bus (Milestone 2.1).
**Reason:** Pure Rust implementation of HNSW (Malkov & Yashunin). Provides
`DistCosine` out of the box, sub-millisecond queries over 65K entries, and
thread-safe concurrent reads. The codebook's `quantize()` function needs
fast approximate nearest-neighbor search — brute force over 65,536 × 256
would be ~2-5ms, above the 0.5ms target.
**Alternatives considered:** `instant-distance` (rejected: less mature,
fewer distance metrics), `hnswlib-rs` (rejected: C++ bindings add build
complexity on Windows), custom HNSW (rejected: unnecessary when a
well-maintained crate exists).

## ADR-012: Codebook Binary Format (2026-02-10)

**Decision:** Use a simple custom binary format (VXCB magic + version +
entry count + dim + raw f32 LE data) for codebook persistence.
**Reason:** The codebook needs to be produced by a Python script (K-Means
over word embeddings) and consumed by Rust. A raw binary format is trivial
to read/write from both languages. The HNSW index is rebuilt on load rather
than serialized, keeping the format simple and portable.
**Alternatives considered:** rkyv (rejected: Python can't produce rkyv
output), protobuf/flatbuffers (rejected: unnecessary complexity for a
flat f32 array), NumPy .npy (rejected: adds npy parsing dependency to Rust).

## ADR-013: CPU-First RAR Implementation (2026-02-10)

**Decision:** Implement RAR inference loop on CPU first (Milestone 2.3),
with GPU port deferred to Milestone 2.4.
**Reason:** GPU debugging is opaque (CUDA errors are cryptic). CPU
implementation allows stepping through every value in a debugger. The
algorithm can be verified correct on CPU, then ported to GPU with
confidence. CPU implementation uses pure Rust with no external NN
framework — just manual matrix multiplications and ReLU activations.
**Architecture:** VFN is a 3-layer MLP (256→512→512→256) with Xavier
init. Cross-slot attention uses Q/K/V projections (256→256) with
scaled dot-product softmax. RAR update rule:
`S_i(t+1) = normalize(S_i(t) + dt × (drift_i + β·msg_i))`.
**Performance:** 50 iterations with 16 slots completes in ~288ms
(release build), well under the 500ms target.
**Alternatives considered:** Start with GPU directly (rejected: debugging
too painful), use candle/tch for CPU ops (rejected: adds heavy
dependency for simple matrix math that pure Rust handles fine).

## ADR-014: candle for GPU ML Operations (2026-02-10)

**Decision:** Use `candle-core` + `candle-nn` (0.8) for GPU-accelerated
RAR inference and VFN training in Milestone 2.4.
**Reason:** candle provides native Rust ML tensor operations with CUDA
backend, autograd for Flow Matching training (requires backprop through
VFN), batched matmul for attention, and CPU fallback for development/CI.
Feature-gated behind `gpu` — without the feature, only pure-Rust CPU
code compiles (zero impact on existing Milestone 2.3 code).
**Alternatives considered:** cudarc (rejected: too low-level, no autograd,
would need to implement matmul/softmax/backprop from scratch), tch-rs
(rejected: requires libtorch C++ bindings, harder build on Windows),
wgpu (rejected: compute shaders are lower-level than needed, no autograd).

## ADR-015: rand for Diffusion Noise Generation (2026-02-10)

**Decision:** Use `rand` 0.9 for Gaussian noise generation in the
diffusion noise controller.
**Reason:** Diffusion noise injection needs proper Gaussian random
numbers. The existing `nn::Rng` (splitmix64) only produces uniform
samples. `rand` with `rand_distr::Normal` provides well-tested Gaussian
sampling with seedable deterministic RNGs (StdRng/SmallRng).
**Alternatives considered:** Extend nn::Rng with Box-Muller (rejected:
reimplementing well-tested math), no diffusion noise (rejected: required
by Milestone 2.4 spec).

## ADR-016: Qwen3-0.6B via candle-transformers for LLM Backbone (2026-02-10)

**Decision:** Use Qwen3-0.6B (`Qwen/Qwen3-0.6B`) as the frozen LLM
backbone for the forward translator (Milestone 2.2), loaded via
`candle-transformers` 0.8 (Qwen2 module) and tokenized with `tokenizers`
0.20.
**Reason:** Qwen3-0.6B is a modern (2025) small LLM with hidden_dim=1024,
151K vocabulary covering 119 languages, and GQA+SwiGLU+RoPE+RMSNorm
architecture. At ~600MB safetensors it is small enough for consumer
hardware. candle-transformers provides a ready-made Qwen2 model
implementation (`candle_transformers::models::qwen2`) which supports
Qwen3 (same architecture). Training happens in Python (PyTorch), weights
exported as safetensors, inference in Rust (candle). This follows the
same Python-train / Rust-infer split established by `tools/codebook_init.py`
in Milestone 2.1.
**Feature gating:** All LLM dependencies behind `llm` feature in
volt-translate, following the `gpu` feature pattern in volt-soft.
**Alternatives considered:** TinyLlama-1.1B (rejected: older model,
larger at 2.2GB f16), Phi-2 (rejected: 2.7B params, overkill for
semantic role labeling), llama.cpp bindings (rejected: C++ build
dependency), full Python inference server (rejected: adds network code).

## ADR-017: HardStrand Trait + Intent Router Design (2026-02-10)

**Decision:** Hard Strands are pluggable via a `HardStrand` trait with
three key methods: `capability_vector()` returning a 256-dim unit vector,
`threshold()` for activation similarity floor, and `process(frame)` for
execution. The Intent Router computes cosine similarity (via
`volt_bus::similarity`) between each registered strand's capability vector
and all active frame slots at R0 (discourse resolution), routing to the
best match above threshold.
**Reason:** Capability vectors live in the same HDC space as frame slot
embeddings, making routing a single cosine similarity comparison — O(S×K)
where S=active slots and K=registered strands. The threshold per strand
allows conservative strands (e.g., safety-critical) to require higher
confidence before activating. The trait is `Send + Sync` enabling future
parallel strand evaluation.
**Slot Protocol:** Math operations use a structured encoding in the
Instrument slot (S6) at R0: dim[0]=op_code, dim[1]=left, dim[2]=right.
Results go to the Result slot (S8) at R0 with gamma=1.0 (exact computation).
**Alternatives considered:** String-based routing (rejected: not in HDC
space, breaks vector algebra), fixed strand dispatch table (rejected: not
extensible), per-slot routing to different strands simultaneously
(deferred: adds complexity, single best-match sufficient for Milestone 3.1).

## ADR-018: CertaintyEngine & ProofConstructor as Pipeline Infrastructure (2026-02-10)

**Decision:** CertaintyEngine and ProofConstructor are pipeline
infrastructure, not HardStrands. They run unconditionally on every frame
inside `HardCorePipeline`, rather than being routed to by the IntentRouter.
**Reason:** CertaintyEngine computes min-rule gamma propagation across all
active slots — it is not a capability that should be "activated" by cosine
similarity. ProofConstructor records what other strands did — it observes
rather than participates. Making them pipeline infrastructure (not traits)
keeps the routing logic clean and ensures they always execute.
**Architecture:** `HardCorePipeline` wraps `IntentRouter` +
`CertaintyEngine` + `ProofConstructor`. Flow: route frame → record routing
decisions in proof → propagate certainty → record certainty step in proof
→ return `PipelineResult { frame, proof: ProofChain }`.
**Alternatives considered:** Making them HardStrands with always-activate
threshold of 0.0 (rejected: conceptual mismatch, they don't have
capability vectors), running them outside the pipeline (rejected: forces
callers to manually wire them up).

## ADR-019: wasmtime for CodeRunner Sandbox (2026-02-10)

**Decision:** Use `wasmtime` 29 for sandboxed code execution in the
CodeRunner HardStrand, feature-gated behind `sandbox` (on by default).
**Reason:** CodeRunner needs to execute untrusted code safely on the CPU.
WASM provides a memory-safe, capability-based sandbox with no implicit
access to filesystem, network, or system calls. wasmtime is the reference
Cranelift-based WASM runtime, maintained by the Bytecode Alliance, with
fuel-based execution limits to prevent infinite loops. WASM bytes are
encoded in the Instrument slot (S6) across R1/R2/R3 resolutions as
f32-cast bytes (up to 768 bytes).
**Sandboxing guarantees:** No WASI imports (instantiation fails if module
requests filesystem/network/clock), fuel-limited to 1M operations (returns
`VoltError::HardError` on exhaustion), isolated linear memory (4MB max).
**Feature gating:** Behind `sandbox` feature because wasmtime is a large
dependency tree (~200 crates). The rest of volt-hard compiles without it.
**Alternatives considered:** Lua VM (rejected: less sandboxable, no fuel
limits), direct native execution (rejected: unsafe, no isolation), WASI
with capability restrictions (rejected: still too much surface area,
simpler to block all WASI imports entirely).

## ADR-020: HDCAlgebra Slot Convention (2026-02-10)

**Decision:** HDCAlgebra uses op codes 11-15 in the Instrument slot (S6)
at R0 for bind/unbind/superpose/permute/similarity operations. Operand
slot indices are encoded in dim[1] and dim[2], with dim[3] for permute
offset k. Source vectors are read from the referenced slots at R0.
**Reason:** HDCAlgebra exposes `volt_bus` HDC operations (bind, unbind,
superpose, permute, similarity) as a callable Hard Strand. Using slot
indices as operand references (rather than embedding operand vectors in
S6) allows operating on any frame slot, supporting compositional reasoning
chains where one operation's output feeds another's input.
**Op codes:** 11.0=bind, 12.0=unbind, 13.0=superpose, 14.0=permute,
15.0=similarity. These are disjoint from MathEngine codes (1-8) and
CodeRunner (10).
**Capability vector:** Deterministic from seed `0x4844_4341_4C47_4231`
("HDCALGB1"), threshold 0.3 — same pattern as MathEngine.
**Alternatives considered:** Inline operand vectors in S6 R1/R2 (rejected:
limits to two fixed operands, can't reference arbitrary slots), separate
HDC-specific frame format (rejected: breaks TensorFrame universality).

## ADR-021: Safety Layer Architecture (2026-02-11)

**Decision:** The safety layer in `volt-safety` uses five constant axiom
vectors (K1-K5) in HDC space, checked via cosine similarity against every
active slot's R0 embedding before and after pipeline processing. Violations
trigger the Omega Veto which returns a safe empty frame and logs the full
trigger state for audit.
**Reason:** Cosine similarity against constant vectors is O(S×K) per frame
(S=active slots, K=5 axioms), adding negligible latency (< 1ms measured).
Using the same HDC space as strand capability vectors means axiom vectors
are directly comparable to frame content — no separate embedding space needed.
The Omega Veto is a struct method (not a trait), making it impossible to
override via polymorphism. Wrapping both pre- and post-pipeline ensures
neither input nor output can violate axioms.
**Axiom design:** Each axiom is a deterministic 256-dim unit vector built
from a unique seed using the same splitmix64 hash as Hard Strand capability
vectors. Thresholds are set at 0.65-0.70 cosine similarity. K1 (harm), K2
(deception), K3 (privacy), K5 (integrity) are Halt-severity. K4 (autonomy)
is Warning-severity, allowing processing to continue with logging.
**Module structure:** `axiom.rs` (K1-K5 definitions), `monitor.rs`
(TransitionMonitor), `scorer.rs` (ViolationScorer), `veto.rs` (OmegaVeto),
`layer.rs` (SafetyLayer wrapping HardCorePipeline).
**Alternatives considered:** Per-slot safety classifiers (rejected: neural
approach is probabilistic and gameable), single aggregate safety score
without per-axiom breakdown (rejected: loses auditability), safety as a
HardStrand (rejected: safety must run unconditionally, not via cosine
routing — same rationale as ADR-018 for CertaintyEngine).

## ADR-022: Ghost Bleed Architecture (2026-02-11)

**Decision:** Ghost bleed is split across two crates with no cross-dependency.
`volt-db` produces ghost gist vectors (`Vec<[f32; 256]>`) via the BleedEngine
and per-strand HNSW indices. `volt-soft` consumes them as a parameter to
`forward_with_ghosts()` / `rar_loop_with_ghosts()`. Integration code
(volt-server or tests) wires them together.
**Reason:** `volt-soft` and `volt-db` are at the same dependency level — neither
can import the other. Passing gist vectors as plain `[f32; 256]` arrays respects
this boundary while requiring zero new shared types. The `FrameGist` struct lives
in `volt-db` since only `volt-db` needs `frame_id`/`strand_id` metadata; `volt-soft`
only needs the raw vectors.
**Ghost attention design:** Ghost gists participate as additional Key/Value sources
with a **separate softmax**, blended with slot attention messages via an alpha weight:
`final_msg = (1-α)·slot_msg + α·ghost_msg`. This ensures ghost frames provide subtle
memory influence without destabilizing the primary slot attention dynamics. Alpha
defaults to 0.1.
**HNSW indexing:** Per-strand HNSW indices using `hnsw_rs` (same library as
ADR-011). Index is rebuilt from stored gists on load, not serialized — matching the
Codebook pattern from ADR-012. R₀ gist extraction uses `volt_bus::superpose` to
combine all active slot R₀ vectors into a single 256-dim unit vector per frame.
**Alternatives considered:** Shared `GhostGist` type in `volt-core` (rejected:
`volt-core` should not know about ghost frames — they are a memory concept, not a
fundamental data structure), ghost frames as full TensorFrames passed to soft core
(rejected: wastes memory and compute — only R₀ gist needed for cross-attention),
single combined softmax over slots + ghosts (rejected: ghosts could dominate attention
when the buffer is large; separate softmax with alpha blending gives explicit control).

## ADR-023: T2 Storage Engine + GC + WAL Design (2026-02-11)

**Decision:** Implement Tier 2 as an LSM-Tree with memtable + mmap'd sorted
runs, WAL for crash recovery, and a 4-tier GC decay pipeline.

**T2 LSM-Tree:** Frames are inserted into a BTreeMap memtable, flushed to
binary sorted runs when the memtable exceeds 4MB. Sorted runs are mmap'd
via `memmap2` for zero-copy reads. Each run has a Bloom filter in its header
for fast negative checks. Compaction merges runs at the same level into the
next level (max 4 runs per level, max 4 levels).

**Frame compression:** 4-tier decay: Full (~64KB) → Compressed/R0+R1 (~8KB) →
Gist/R0 only (~1KB) → Tombstone (32B). `CompressedFrame` and `GistFrame` use
a custom binary codec instead of serde because serde doesn't support
`[f32; 256]` arrays (max array size is 32). Tombstones use serde_json (small,
no large arrays).

**GC retention scoring:**
`score = 0.40 * exp(-age_days/τ) + 0.35 * γ + 0.15 * ln(1+refs) + 0.10 * pinned_bonus`.
Frames are immortal if: pinned, gamma >= 1.0, or is_wisdom. Decay thresholds
are configurable (default: Full→Compressed at 0.7, Compressed→Gist at 0.4,
Gist→Tombstone at 0.1). GC only demotes, never promotes.

**WAL:** Per-strand append-only binary log files with CRC32 checksums
(via `crc32fast`). On crash, valid entries are replayed; corrupt tail entries
are skipped. WAL is checkpointed (truncated) after successful T2 flushes.

**Consolidation:** Greedy union-find clustering over per-strand HNSW gists.
Clusters above min_size (default 5) are merged into "wisdom frames" — summary
frames with averaged R0 vectors and high certainty (default 0.95). Superseded
frames get lower priority in future GC runs.

**Concurrency:** `ConcurrentVoltStore` wraps `VoltStore` in `Arc<RwLock<_>>`.
This satisfies the "10 readers + 1 writer" requirement with standard library
primitives. Epoch-based concurrency (crossbeam) deferred to a future milestone
if contention becomes a bottleneck.

**New dependencies:** `memmap2 = "0.9"` (mmap'd sorted runs) and
`crc32fast = "1.4"` (WAL + sorted run checksums). Both are small,
well-maintained, no-unsafe pure Rust crates.

**Alternatives considered:** `rkyv` for zero-copy deserialization (rejected:
custom binary is simpler and sufficient for the 4 decay types), `crossbeam-epoch`
for lock-free concurrent access (rejected: `RwLock` is simpler and sufficient
for current workload), `sled` or `rocksdb` as the T2 backend (rejected:
external dependency too heavy, custom LSM gives us control over the mmap +
Bloom + compaction pipeline).

## ADR-024: Learning Event Logging Architecture (2026-02-12)

**Decision:** Learning events are accumulated in a bounded in-memory buffer
(`EventBuffer`, default capacity 10,000) with FIFO eviction, wrapped by
`EventLogger` which provides on-demand statistics and JSON persistence.
The logger is shared across server handlers via `Arc<RwLock<EventLogger>>`.

**Event struct:** `LearningEvent { frame_id, strand_id, query_type,
gamma_scores, convergence_iterations, ghost_activations, timestamp }`.
All fields are available from existing pipeline outputs — no new data
collection is needed.

**Statistics:** Computed on demand from the buffer (no incremental counters).
`StrandStatistics` includes query count, average gamma, average iterations,
and `TopicDistribution` across discourse types.

**Persistence:** JSON files matching the `StrandStore` save/load pattern.
Save/load runs on a dedicated thread with 8 MB stack for Windows
compatibility.

**Server integration:** `EventLogger` added to `AppState` behind
`Arc<RwLock<_>>`. Event logging is best-effort — a poisoned lock does
not fail the inference request.

**Error variant:** New `VoltError::LearnError { message }` follows the
per-domain pattern (ADR-003).

**Alternatives considered:** VoltDB WAL for event persistence (rejected:
WAL is optimized for frame crash recovery, not analytics events), ring
buffer (rejected: complicates drain/index semantics), incremental statistics
(rejected: adds complexity for minimal gain at expected buffer sizes).

## ADR-025: Sleep Consolidation Architecture (2026-02-12)

**Decision:** Sleep consolidation is orchestrated by `SleepScheduler` in
`volt-learn`, which runs a multi-phase cycle: frame distillation → Forward-Forward
VFN training → strand graduation → garbage collection. The scheduler supports
both manual triggering and automatic background polling (idle > 10 minutes).

**Forward-Forward training:** Implemented as a CPU-only per-layer goodness
optimization on `Vfn`. Each layer is trained independently — positive samples
(high-gamma verified frames) push goodness above a threshold, negative samples
(low-gamma + corrupted frames) push goodness below. No backpropagation — gradients
never flow between layers. This keeps VRAM/memory usage at ~1x inference. New
public API on `Vfn`: `forward_layer()`, `update_layer()`, `layer_shape()`. New
`pub(crate)` methods on `nn::Linear`: `weights_mut()`, `bias_mut()`.

**Frame distillation:** Delegates to existing `VoltStore::consolidate_strand()` from
Milestone 4.3 (HNSW clustering + wisdom frame creation). Thin wrapper in
`volt-learn::distillation` iterates all strands.

**Strand graduation:** Analyzes learning events to find clusters of frames within
a strand that are dissimilar to the strand's centroid. If a cluster exceeds 50 frames,
creates a new strand and migrates frames via `VoltStore::reassign_frame_strand()`.

**Background thread:** `SleepScheduler::spawn_background()` spawns a thread with
4 MB stack that polls at configurable intervals. Locks are acquired in fixed order
(logger → store → vfn) to prevent deadlocks. Main thread remains responsive via
`RwLock` sharing.

**Alternatives considered:** GPU-based Forward-Forward via candle (rejected: adds
GPU dependency to volt-learn, violates architecture rules), async scheduler with
tokio (rejected: CLAUDE.md forbids async in volt-learn), single-layer VFN update
without per-layer API (rejected: can't implement true Forward-Forward without
layer-local forward passes).

## ADR-026: RLVF Joint Alignment Architecture (2026-02-12)

**Decision:** RLVF (Reinforcement Learning from Verified Feedback) uses
REINFORCE with baseline to update VFN weights based on a shaped reward signal.
The reward combines output correctness (cosine similarity vs verified answer)
with gamma calibration quality (penalizing overconfident errors, rewarding
honest uncertainty).

**Evaluation dataset:** 1000 deterministic (question, answer) pairs across
4 categories (Math, Logic, Factual, Creative) at 250 each. Questions/answers
are word sequences encoded by `StubTranslator`'s hash-based `word_to_vector`.
No real NL understanding needed — the dataset tests that the VFN learns to
map question embeddings toward answer embeddings.

**Reward shaping:** Five reward levels: correct+confident (+1.0),
correct+uncertain (+0.5), wrong+uncertain (+0.2, honest), wrong+mid (-0.5),
wrong+overconfident (-2.0, strong penalty). This asymmetric shaping
incentivizes calibrated confidence over raw accuracy.

**REINFORCE with baseline:** Per-epoch exponential moving average baseline.
Advantage = reward - baseline. Positive advantage → treat as positive FF
sample (push goodness up). Negative advantage → treat as negative FF sample
(push goodness down). Magnitude clamped to [-2.0, 2.0] for stability.
Reuses the layer-local gradient computation from Forward-Forward (no backprop).

**Certainty calibration metric:** Expected Calibration Error (ECE) computed
over 10 equal-width gamma bins. ECE = Σ |accuracy_i - mean_gamma_i| × n_i / n.
A perfectly calibrated model has ECE = 0.0.

**Self-play logic puzzles:** Deterministic generation of 5 puzzle types
(Modus Ponens, Transitivity, Modus Tollens, Conjunction, Disjunction) using
a fixed vocabulary of atomic propositions. Graded by cosine similarity
between VFN output and expected conclusion frame.

**New dependency:** `volt-translate` added to `volt-learn` for encoding
eval pairs into TensorFrames.

**Alternatives considered:** PPO-style clipped objective (rejected: requires
log-probability computation which VFN doesn't expose), separate reward model
(rejected: over-engineering for this milestone), backpropagation through full
pipeline (rejected: violates Forward-Forward paradigm and CLAUDE.md rules).

## ADR-018: Shared VFN for Learning Loop Closure (2026-02-12)

**Decision:** Add `SharedVfn = Arc<RwLock<Vfn>>` to `AppState`. The RAR
inference pipeline reads (clones) from it; the sleep scheduler write-locks
and trains it via Forward-Forward and RLVF. The pipeline thread snapshots the
VFN once per request to minimize lock contention.
**Reason:** Without shared state, the learning system trained a VFN that was
discarded after each sleep cycle, and inference created a fresh random VFN on
every request. Sharing the VFN closes the learning loop: trained weights carry
through to subsequent inference.
**Alternatives considered:** Persist VFN to disk and reload (rejected: adds
I/O latency and crash-recovery complexity), pass VFN by reference into the
RAR thread (rejected: would hold read lock for entire RAR loop duration,
blocking sleep scheduler writes).

## ADR-019: Sleep Scheduler Server Integration (2026-02-12)

**Decision:** Spawn `SleepScheduler::spawn_background()` in `main.rs` with
`Arc` clones of `VoltStore`, `Vfn`, and `EventLogger`. Added
`build_app_with_state()` so `main.rs` can create `AppState` first, clone
the Arcs, then build the router. Added `ConcurrentVoltStore::inner_arc()`
to expose the raw `Arc<RwLock<VoltStore>>` needed by `spawn_background`.
**Reason:** The sleep scheduler was fully implemented but never wired into
the server. Consolidation, FF training, RLVF, graduation, and GC never ran.
**Alternatives considered:** Run sleep in an async task (rejected:
volt-learn is pure synchronous, mixing async would add complexity),
trigger sleep from an HTTP endpoint only (rejected: would require manual
invocation instead of automatic idle detection).

## ADR-020: RLVF as Optional Sleep Phase (2026-02-12)

**Decision:** Added `rlvf_config: Option<RlvfConfig>` and
`rlvf_min_events: usize` to `SleepConfig`. RLVF runs after Forward-Forward
in `run_sleep_cycle_inner()` only when enabled and enough events have
accumulated. Server enables it by default.
**Reason:** RLVF is more expensive than FF (evaluates 1000 QA pairs per
epoch). Making it optional with an event threshold prevents wasted cycles
early in the server's lifetime when there are too few events to learn from.
**Alternatives considered:** Always run RLVF (rejected: wasteful when
event count is low), separate RLVF scheduler thread (rejected: over-
engineering — it shares the same VFN lock order as FF).

## ADR-027: Module System Architecture (2026-02-12)

**Decision:** Compile-time module system using Cargo feature flags,
`ModuleInfo` metadata struct, `ActionCore` trait, `catch_unwind` panic
safety, and `ModuleRegistry` discovery at startup. No dynamic loading.
No new external dependencies (zero-dep milestone).

**Components:**

- `ModuleInfo` + `ModuleType` in volt-core: common metadata for all modules.
- `HardStrand::info()` and `Translator::info()`: default methods returning
  `Option<ModuleInfo>` (backward compatible, existing impls inherit `None`).
- `ActionCore` trait in volt-translate: output modules converting
  TensorFrames to human-consumable output (text, audio, image, etc.).
- `catch_unwind(AssertUnwindSafe(...))` in `IntentRouter::route()`: catches
  panicking strands, logs the error, returns non-activated decision.
- `IntentRouter::unregister()` and `strand_names()`: runtime strand
  management for hot-plug lifecycle.
- `WeatherStrand` behind `#[cfg(feature = "weather")]`: example community
  module with deterministic mock data (5 weather profiles).
- `ModuleRegistry::discover()` in volt-server: enumerates built-in +
  feature-gated modules at startup. `GET /api/modules` HTTP endpoint.
- CLI subcommands: `volt modules list|install|uninstall` prints human-
  readable instructions for the feature-flag workflow.

**Reason:** Rust is a compiled language. Dynamic loading (`dlopen`)
requires `unsafe` and loses type-system guarantees. Feature flags are the
established Cargo pattern for optional compilation units. This approach
gives zero runtime overhead for disabled modules and full type checking
at compile time.
**Alternatives considered:** `libloading` dynamic plugins (rejected:
requires `unsafe`, ABI fragility, no type checking across boundary),
WASM plugin host (rejected: adds wasmtime dependency + performance
overhead for CPU-bound strand logic, better suited for untrusted
third-party code in a future milestone), trait objects with `inventory`
crate (rejected: adds external dependency, magic linker tricks).

## ADR-028: Codebook Initialization Pipeline (2026-02-13)

**Decision:** Implement codebook initialization as a streaming encode +
brute-force mini-batch k-means pipeline in `volt-learn`. New modules:
`stack_corpus` (streaming JSONL reader for The Stack), `kmeans` (mini-batch
k-means with k-means++ initialization), `codebook_init` (pipeline
orchestration). K-means implemented from scratch (~300 lines) rather than
using an external crate.
**Reason:** The codebook's 65,536 entries must cover code's actual embedding
space, not random/NL distributions. Mini-batch k-means with brute-force
assignment is simple and performant at k=65K, dim=256 (~0.4ms/vector),
making HNSW-accelerated assignment unnecessary. `rand` added to `volt-learn`
for k-means++ random sampling (already a workspace dependency).
**Alternatives considered:** `linfa-clustering` crate (rejected: heavy
dependency tree, not in workspace, algorithm is simple to implement),
HNSW-accelerated nearest-centroid assignment (rejected: marginal speed
benefit at 65K scale, adds complexity), GPU k-means in `volt-soft`
(rejected: Phase 0 is CPU-only, one-time initialization).

## ADR-029: Code Attention Bias (2026-02-13)

**Decision:** Add an optional additive attention bias to `SlotAttention`
— a `[[f32; MAX_SLOTS]; MAX_SLOTS]` matrix added to pre-softmax logits
based on slot position. The code-specific bias in `code_attention.rs`
encodes structural priors (e.g., Function S0 ↔ Arguments S2 = +2.0).
**Reason:** Code has strong positional structure that random Xavier
initialization ignores. An additive bias is the correct mechanism for
position-dependent priors because Q/K/V weight matrices are shared
across all 16 slots (content-based, not position-based). Unlike Q/K/V
initialization, an additive bias persists through training and is
composable with learned attention patterns (similar to ALiBi in
transformers). No new dependencies required.
**Alternatives considered:** Q/K/V weight initialization to embed the
bias (rejected: shared weight matrices cannot encode position-dependent
priors — `Q_i = W_Q · x_i` makes all queries content-dependent, not
position-dependent), per-slot Q/K/V projections (rejected: 16× weight
increase, massive architectural change), learned slot embeddings added
before projection (deferred: good idea but larger scope than Phase 0.4).

## ADR-030: Lightweight CNN Encoder for Phase 1 (2026-02-13)

**Decision:** Train a lightweight ~5M parameter CNN encoder (BPE →
Embedding(32768, 128) → 3×Conv1D → Role Head + Embed Head) and a ~7M
parameter non-autoregressive decoder for Phase 1 code translation, rather
than using the existing LlmTranslator (Qwen3-0.6B backbone, ~600M params).
Uses InfoNCE contrastive loss on CodeSearchNet (code, docstring) pairs with
heuristic role grounding supervision. New `code-training` feature flag
(separate from `llm`) — only requires candle-core, candle-nn, tokenizers.
**Reason:** The plan explicitly targets a lightweight encoder trainable from
scratch in ~30 GPU-hours on a single RTX 5090 Mobile. The LlmTranslator
requires downloading a frozen 600M param backbone and cannot learn
code-specific slot roles. A CNN encoder has three advantages: (1) no model
download, (2) fast training from scratch, (3) role head directly classifies
tokens into the 16 PropBank-inspired slot roles (Agent, Patient, Location,
etc.) that define TensorFrame structure. The non-autoregressive decoder
predicts all output positions independently, avoiding the complexity of
autoregressive decoding in candle (which lacks built-in GRU/LSTM).
**Alternatives considered:** LlmTranslator with Qwen3 backbone (deferred to
Phase 5: requires ~1.2GB download, frozen backbone cannot learn slot roles),
autoregressive GRU decoder (rejected: candle-nn lacks GRU, would require
manual implementation), transformer encoder instead of CNN (rejected: heavier
for same embedding quality at this model scale, and self-attention is
unnecessary when Conv1D with k=3,5,7 provides sufficient receptive field).
