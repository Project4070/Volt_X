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
