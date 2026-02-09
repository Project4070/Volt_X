# Milestone 0: The Test Harness (Week 0 — before everything)

> Extracted from the master roadmap. This milestone must be completed before Phase 1 begins.

## What You Build

```
volt-x/
├── Cargo.toml              (workspace root)
├── crates/
│   ├── volt-core/          (data structures: TensorFrame, SlotRole, etc.)
│   ├── volt-bus/           (LLL algebra: bind, unbind, superpose, permute)
│   ├── volt-soft/          (GPU Soft Core: RAR loop)
│   ├── volt-hard/          (CPU Hard Core: intent router, hard strands)
│   ├── volt-db/            (VoltDB: storage engine)
│   ├── volt-translate/     (input/output translators)
│   ├── volt-learn/         (continual learning engine)
│   ├── volt-safety/        (safety layer, omega veto)
│   ├── volt-ledger/        (intelligence commons)
│   └── volt-server/        (Axum HTTP server, n8n integration)
├── tests/
│   ├── integration/        (end-to-end tests)
│   └── benchmarks/         (performance tracking)
└── n8n/
    └── workflows/          (n8n workflow JSON exports)
```

### Why This Structure Matters
- Each crate compiles independently. A bug in `volt-ledger` does not prevent `volt-core` from compiling.
- Each crate has its own unit tests. You can run `cargo test -p volt-core` without touching anything else.
- Dependencies flow one way: core <- bus <- soft/hard/db <- translate <- server. No circular dependencies.
- New crates can be added without modifying existing ones.

## What You Test
- `cargo test` passes on empty crates (all compile, zero tests, zero failures)
- `cargo clippy` produces zero warnings
- CI pipeline runs on every commit (GitHub Actions, ~2 minutes)

## Duration
1 day.
