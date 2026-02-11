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
