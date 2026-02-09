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
