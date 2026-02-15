# Phase 5 Audit — Integration Gaps & Remaining Work

> Audit date: 2026-02-12
> Test suite: 1,058 tests, all passing

---

## Gap 1: VFN Identity Problem (Critical) — RESOLVED

**Crates:** volt-soft ↔ volt-learn ↔ volt-server

The Soft Core previously instantiated a fresh `Vfn::new_random(42)` for every
request in `routes.rs`. The learning system (Forward-Forward + RLVF) trained a
*separate* VFN instance. These two never connected.

**Fix applied:** Created `SharedVfn = Arc<RwLock<Vfn>>` in `AppState`. The RAR
loop reads (clones) from it; the sleep scheduler write-locks and trains it.
The pipeline thread in `routes.rs` now calls `rar_loop_with_ghosts` directly
with the snapshotted VFN instead of the convenience wrapper that created a
fresh random VFN each time.

**Files changed:**

- `crates/volt-server/src/state.rs` — added `SharedVfn` type + `vfn` field
- `crates/volt-server/src/routes.rs` — snapshot VFN from state, call `rar_loop_with_ghosts` directly
- `crates/volt-db/src/store.rs` — added `ConcurrentVoltStore::inner_arc()`

---

## Gap 2: Sleep Scheduler Not Spawned (Critical) — RESOLVED

**Crates:** volt-learn ↔ volt-server

`SleepScheduler::spawn_background()` existed and was tested, but `main.rs` /
`build_app()` never called it.

**Fix applied:** `main.rs` now creates `AppState` explicitly, spawns the sleep
scheduler with shared references to `ConcurrentVoltStore`, `EventLogger`, and
`SharedVfn`, then builds the Axum router via `build_app_with_state()`.
Graceful shutdown signals the scheduler on server exit.

**Files changed:**

- `crates/volt-server/src/main.rs` — spawn sleep scheduler, graceful shutdown
- `crates/volt-server/src/lib.rs` — added `build_app_with_state()`

---

## Gap 3: RLVF Not Triggered (Medium) — RESOLVED

**Crates:** volt-learn ↔ volt-server

`train_rlvf()` was standalone with no trigger.

**Fix applied:** Added RLVF as an optional phase in the sleep cycle, gated by
`rlvf_config: Option<RlvfConfig>` and `rlvf_min_events: usize` in
`SleepConfig`. When enabled and enough events have accumulated, the sleep cycle
generates the eval dataset, creates a translator, and runs RLVF after
Forward-Forward training. The server enables it by default in `main.rs`.

**Files changed:**

- `crates/volt-learn/src/sleep.rs` — added RLVF config fields, RLVF phase in
  `run_sleep_cycle_inner()`, `SleepHandle::touch()` for activity signaling

---

## Gap 4: Hard Core Strand Set Is Static (Low) — DEFERRED

**Crates:** volt-hard ↔ volt-learn

`IntentRouter` registers `MathEngine` + `HdcAlgebra` at startup.
`check_graduation()` can create new strands in VoltDB, but they are never
registered into the running router.

**Deferred:** Phase 6 scope. Graduation works at the data level today.

---

## Gap 5: Pre-existing Windows Stack Overflow (Low) — RESOLVED

**Crate:** volt-core

`serde_roundtrip_is_bit_identical` overflowed on Windows (64KB TensorFrame +
serde_json recursive descent on 1MB default stack).

**Fix applied:** Wrapped the test body in `std::thread::Builder::new()
.stack_size(8 * 1024 * 1024)` — same pattern used in volt-db and volt-learn.

**Files changed:**

- `crates/volt-core/tests/serialization_test.rs`

---

## Summary

| Gap | Status | Priority |
| --- | ------ | -------- |
| Gap 1: Shared VFN | Resolved | Critical |
| Gap 2: Sleep scheduler | Resolved | Critical |
| Gap 3: RLVF in sleep cycle | Resolved | Medium |
| Gap 4: Dynamic strand registration | Deferred (Phase 6) | Low |
| Gap 5: Serde test stack overflow | Resolved | Low |
