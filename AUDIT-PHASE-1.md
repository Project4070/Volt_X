# Volt X Phase 1 Audit Report

**Date:** 2026-02-09
**Auditor:** Claude (automated code audit)
**Scope:** Phase 1 — "The Skeleton" (Milestones 1.1 through 1.4)
**Verdict:** SUBSTANTIALLY COMPLETE with 1 architectural gap and 5 minor issues

---

## Executive Summary

Phase 1 aims to deliver: *"A dumb system that takes text in and text out, with TensorFrames flowing through the entire pipeline. The Soft Core is a stub (copies input to output). The Hard Core is a stub (passes through). But the Frame Bus works, the data structures work, the server works, and n8n can talk to it."*

The individual components (TensorFrame, LLL algebra, translator, HTTP server) are well-implemented and thoroughly tested. However, **the end-to-end pipeline does not match the Phase 1 checkpoint specification**. The Soft Core and Hard Core stubs are not wired into the pipeline, and the Frame Bus algebra (despite being fully implemented and tested) is never invoked during an actual `/api/think` request.

| Area | Rating | Summary |
|------|--------|---------|
| Milestone 1.1: TensorFrame | **PASS** | All data structures correct, 62 tests |
| Milestone 1.2: LLL Algebra | **PASS** | All 5 ops implemented, 47 tests, perf targets met |
| Milestone 1.3: Translator + Server | **PASS** | Encode/decode works, HTTP API functional, 68 tests |
| Milestone 1.4: n8n Integration | **PASS** | Workflow JSON exists, slot_states/timing_ms in response |
| Phase 1 Checkpoint Pipeline | **FAIL** | Soft/Hard Core stubs not in pipeline, Bus not invoked |
| Code Standards | **MOSTLY PASS** | 1 unwrap in lib code, 1 missing Clone derive |

---

## Milestone 1.1: TensorFrame Data Structure — PASS

**Crate:** `volt-core` | **Tests:** 62 (25 unit + 11 integration + 26 doc tests)

### Requirements vs Implementation

| Requirement | Status | Evidence |
|-------------|--------|----------|
| TensorFrame with S=16, R=4, D=256 | PASS | `frame.rs:34-47`, constants in `lib.rs` |
| SlotData, SlotMeta, FrameMeta structs | PASS | `slot.rs`, `meta.rs` |
| SlotRole enum | PASS | 9 fixed roles + `Free(u8)` for S9-S15 |
| Serde serialization | PASS | Feature-gated, custom big-array helpers |
| rkyv zero-copy serialization | PASS | Feature-gated, roundtrip + zero-copy tests pass |
| Frame create/clone/read/write/merge | PASS | All operations return `Result<T, VoltError>` |
| Unit normalization | PASS | Per-slot and batch L2 normalization |
| Empty frame < 100 bytes | PASS | `data_size_bytes()` returns 0 for empty |
| Full frame = 64KB | PASS | 16 x 4 x 256 x 4 = 65,536 bytes verified |

### Issues

1. **`unwrap()` in library code** (`frame.rs:483`): The `normalize_all()` method contains `.unwrap()` on an `Option<&SlotData>` inside a branch already guarded by `is_some()`. While logically safe, it violates the CLAUDE.md rule: *"No unwrap() in library code."*

   **Fix:** Replace the `.as_ref().unwrap()` pattern with a `let Some(ref slot) = self.slots[slot_idx]` guard or use an `if let` chain.

2. **Serde roundtrip test `#[ignore]`d**: `serialization_test.rs` has `serde_roundtrip_is_bit_identical()` marked `#[ignore]` due to stack overflow with large nested arrays. The rkyv roundtrip test works fine.

   **Fix:** Use `Box<TensorFrame>` or heap allocation in the test, or document this as a known limitation with serde for full frames.

---

## Milestone 1.2: LLL Algebra — PASS

**Crate:** `volt-bus` | **Tests:** 47 (28 unit + 10 integration + 9 doc tests)

### Requirements vs Implementation

| Requirement | Status | Evidence |
|-------------|--------|----------|
| FFT-based `bind(a, b) -> c` | PASS | `fft.rs` uses `rustfft` 6.2, circular convolution |
| Unbinding `unbind(c, a) -> b_approx` | PASS | Circular correlation in frequency domain |
| Superposition `superpose(vec) -> normalized_sum` | PASS | Element-wise sum + L2 normalize |
| Permutation `permute(a, k)` | PASS | Cyclic shift with `rem_euclid` |
| Cosine similarity `sim(a, b) -> f32` | PASS | Dot product / (norm_a * norm_b) |
| Operations on 256-dim vectors | PASS | All use `[f32; SLOT_DIM]` |
| Batch operations on entire frames | PASS | `bind_frames`, `unbind_frames`, `similarity_frames` |

### Milestone Requirements Verified

- `unbind(bind(a, b), a) ~ b` with similarity > 0.85 — **Verified in tests**
- `sim(superpose([a, b]), a) > 0` — **Verified**
- `sim(bind(a, b), a) ~ 0` — **Verified (|sim| < 0.1)**
- `sim(permute(a, 1), permute(a, 2)) ~ 0` — **Verified (|sim| < 0.1)**
- Bind on 256 dims < 10us — **Achieved ~1.6us (6x faster)**

### Issues

None. This milestone is fully compliant.

---

## Milestone 1.3: Stub Translator + HTTP Server — PASS

**Crates:** `volt-translate` (45 tests) + `volt-server` (19 tests)

### Translator Requirements

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Stub forward translator (positional heuristic) | PASS | `stub.rs`: word 0 -> Agent, word 1 -> Predicate, word 2 -> Patient |
| Deterministic word-to-vector (hash -> seed -> 256-dim) | PASS | FNV-1a hash in `encode.rs`, L2-normalized |
| Stub reverse translator (template output) | PASS | Vocab-based nearest-neighbor decode |
| Translator trait with encode/decode/decode_slots | PASS | `lib.rs:54-103` |

### Server Requirements

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Axum HTTP server | PASS | Axum 0.8, `build_app()` returns testable Router |
| `POST /api/think` accepts `{"text": "..."}` | PASS | Returns text, gamma, strand_id, iterations |
| `GET /health` | PASS | Returns `{"status": "ok", "version": "0.1.0"}` |
| 100 concurrent requests without crash | PASS | `concurrent_requests_do_not_crash` test |
| Empty/huge/invalid input -> graceful error | PASS | 400 Bad Request with error messages |

### Issues

1. **`StubTranslator` missing `Clone` derive** (`stub.rs`): Contains `RwLock<Vec<VocabEntry>>` and does not derive Clone. Violates CLAUDE.md rule: *"All structs must derive Debug, Clone."*

   **Fix:** Add `impl Clone for StubTranslator` that clones the inner Vec (or restructure to make `#[derive(Clone)]` work).

---

## Milestone 1.4: n8n Integration — PASS

**Crate:** `volt-server` (extended) + `n8n/` directory

### Requirements vs Implementation

| Requirement | Status | Evidence |
|-------------|--------|----------|
| ThinkResponse includes `slot_states` | PASS | `models.rs:82-95`: index, role, word, certainty, source, resolution_count |
| ThinkResponse includes `timing_ms` | PASS | `models.rs:108-116`: encode_ms, decode_ms, total_ms |
| `decode_slots()` on Translator trait | PASS | Returns `Vec<(usize, SlotRole, String)>` |
| n8n workflow JSON | PASS | `n8n/volt-x-chat.workflow.json` |
| Workflow: Chat Trigger -> HTTP Request -> Switch -> Debug Panel -> Chat Reply | PASS | All 6 nodes present with correct wiring |

### Issues

None for this milestone specifically.

---

## Phase 1 Checkpoint: End-to-End Pipeline — FAIL

### Expected Pipeline (from PHASE-1.md)

```
n8n -> HTTP -> Translate -> TensorFrame -> Bus -> Stub Process -> Translate Back -> HTTP -> n8n
```

### Actual Pipeline (from `routes.rs:45-118`)

```
HTTP -> Translate(encode) -> TensorFrame -> Translate(decode) -> HTTP
```

### What Is Missing

**1. Soft Core Stub Not in Pipeline**

Phase 1 states: *"The Soft Core is a stub (copies input to output)."*

`volt-soft/src/lib.rs` contains only documentation and TODO comments referencing Milestone 3.1. There is no callable stub function (not even an identity pass-through). The server (`routes.rs`) never references `volt-soft`.

**Remediation:** Add a stub function to volt-soft:
```rust
pub fn process_stub(frame: &TensorFrame) -> TensorFrame {
    frame.clone() // copies input to output
}
```
Then call it in `routes.rs` between encode and decode.

**2. Hard Core Stub Not in Pipeline**

Phase 1 states: *"The Hard Core is a stub (passes through)."*

`volt-hard/src/lib.rs` contains only documentation and TODO comments referencing Milestone 3.2. There is no callable stub function. The server (`routes.rs`) never references `volt-hard`.

**Remediation:** Add a stub function to volt-hard:
```rust
pub fn verify_stub(frame: &TensorFrame) -> TensorFrame {
    frame.clone() // passes through unchanged
}
```
Then call it in `routes.rs` after the soft core stub.

**3. Frame Bus Not Invoked in Pipeline**

Phase 1 states: *"the Frame Bus works"* (in the context of the pipeline).

The LLL algebra in `volt-bus` is fully implemented and tested in isolation, but it is never called during an actual `/api/think` request. The bus is imported by `volt-server` via the dependency chain but never used.

**Remediation:** While the bus algebra isn't expected to *transform* the frame in Phase 1, the pipeline should at minimum demonstrate that the bus is functional by either:
- Having the soft/hard stubs use bus operations (even trivially, e.g., `bind(frame, identity)`)
- Or routing the frame explicitly through the bus layer

**4. Impact Assessment**

This is a structural gap, not a functional one. All the building blocks work individually:
- TensorFrame: correct
- LLL algebra: correct and performant
- Translator: encode/decode roundtrips work
- HTTP server: handles requests properly
- n8n: workflow is wired

But the Phase 1 checkpoint explicitly describes a pipeline where TensorFrame flows through Bus -> Soft Core -> Hard Core, and that pipeline does not exist. Subsequent phases that "improve one component of this working pipeline" will have nothing to slot into because the pipeline skeleton is incomplete.

---

## Code Standards Compliance

### Passing

| Standard | Status |
|----------|--------|
| Rust edition 2024 | PASS — all crates |
| `cargo clippy -- -D warnings` | PASS — zero warnings |
| All public functions have doc comments with examples | PASS |
| All structs derive Debug, Clone | MOSTLY PASS — 1 violation (StubTranslator) |
| No unwrap() in library code | MOSTLY PASS — 1 violation (frame.rs:483) |
| VoltError used everywhere | PASS |
| thiserror for error derivation | PASS |
| Error messages include context | PASS |
| Dependencies flow one direction | PASS |
| No circular dependencies | PASS |
| volt-core imports no other volt-* crate | PASS |
| volt-server is the leaf crate | PASS |
| No async in volt-core or volt-bus | PASS |
| No GPU code outside volt-soft | PASS |
| No network code outside volt-ledger/volt-server | PASS |
| Naming conventions (kebab-case, snake_case, PascalCase) | PASS |

### Test Suite Summary

| Crate | Unit | Integration | Doc | Total |
|-------|------|-------------|-----|-------|
| volt-core | 25 | 12 | 26 | 63 |
| volt-bus | 28 | 10 | 9 | 47 |
| volt-translate | 23 | 13 | 9 | 45 |
| volt-server | 0 | 10 | 9 | 19 |
| **Total** | **76** | **45** | **53** | **174** |

All 174 tests pass. Zero failures.

### Additional Issues

1. **Workspace integration tests not running**: `tests/integration/smoke.rs` and `tests/integration/frame_roundtrip.rs` exist at the workspace root but are never compiled or executed. The workspace Cargo.toml is a virtual manifest (no `[package]` section), so these files are dead code.

   **Fix:** Either move these tests into a dedicated test crate added to the workspace members, or delete them (the smoke test coverage already exists in individual crate tests).

2. **Unused import warning in test code**: `crates/volt-bus/tests/algebra_test.rs:11` imports `SlotData` but never uses it. This produces a compiler warning during `cargo test` (though not during `cargo clippy` on release profile).

   **Fix:** Remove the unused `SlotData` import.

3. **Serde roundtrip test permanently `#[ignore]`d**: `serialization_test.rs` has `serde_roundtrip_is_bit_identical` marked `#[ignore]` due to stack overflow. Since the rkyv roundtrip works, this isn't blocking, but it means serde serialization of full frames is untested in CI.

   **Fix:** Allocate the test frame on the heap (`Box::new(TensorFrame::new())`), or test with a partially-filled frame that doesn't overflow the stack.

---

## Complete Findings Summary

### Critical (blocks Phase 1 completion)

| # | Issue | Location | Remediation |
|---|-------|----------|-------------|
| C1 | Soft Core stub missing from pipeline | `volt-soft/src/lib.rs`, `routes.rs` | Add pass-through stub, wire into `/api/think` |
| C2 | Hard Core stub missing from pipeline | `volt-hard/src/lib.rs`, `routes.rs` | Add pass-through stub, wire into `/api/think` |
| C3 | Frame Bus not invoked in pipeline | `routes.rs` | Route frame through bus (even as identity) |

### Minor (code standards violations)

| # | Issue | Location | Remediation |
|---|-------|----------|-------------|
| M1 | `unwrap()` in library code | `volt-core/src/frame.rs:483` | Replace with `if let` or match |
| M2 | `StubTranslator` missing Clone | `volt-translate/src/stub.rs` | Add Clone implementation |
| M3 | Dead workspace integration tests | `tests/integration/` | Move to test crate or delete |
| M4 | Unused import in test | `volt-bus/tests/algebra_test.rs:11` | Remove `SlotData` import |
| M5 | Serde roundtrip test `#[ignore]`d | `volt-core/tests/serialization_test.rs` | Fix stack overflow or test partial frame |

---

## Recommended Remediation Order

1. **C1 + C2 + C3** (pipeline completion): Add stub pass-through functions to `volt-soft` and `volt-hard`. Update `routes.rs` to call: `encode -> soft_core_stub -> hard_core_stub -> decode`. This completes the Phase 1 checkpoint pipeline and takes approximately one focused session.

2. **M1** (unwrap fix): One-line change in `frame.rs:483`.

3. **M2** (Clone derive): Add Clone implementation to `StubTranslator`.

4. **M3** (dead tests): Move or delete `tests/integration/`.

5. **M4 + M5** (test hygiene): Quick cleanup.

---

## Conclusion

Phase 1 has produced high-quality, well-tested individual components. The TensorFrame data structure, LLL algebra, stub translator, and HTTP server all meet their milestone specifications. The code quality is strong: zero clippy warnings, comprehensive doc comments, proper error handling, and 174 passing tests.

The single blocking gap is the pipeline integration. The Phase 1 checkpoint explicitly requires TensorFrame to flow through *"Bus -> Stub Process"* between translation stages. Currently, the frame goes directly from encoder to decoder with no intermediate processing. Adding the three stub pass-through calls (soft core, hard core, and a bus touchpoint) to `routes.rs` would close this gap and deliver on the Phase 1 promise: a dumb but complete end-to-end pipeline ready for subsequent phases to improve.
