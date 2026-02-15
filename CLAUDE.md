# CLAUDE.md — Instructions for Claude Code

## Project: Volt X

Volt X is a cognitive architecture implementing a stateful AI operating
system. It uses Tensor Frames (not flat vectors), Root-Attend-Refine
inference (not transformers), and three-tier memory (not context windows).

## Project Status

All 6 runtime implementation phases are **complete** (Phases 1-6).
Training infrastructure (Phase 0) is complete. The project is now in
the **training phase** — implementing the spiral curriculum described
in `TRAINING.md` v3.0.

See `Dev_History.md` for a chronological record of the full build.

## Context Documents — Read Before Working

When starting a new session or task, read the relevant documents below
to load context efficiently.

### Always Read

- `ARCHITECTURE.md` — Full technical design reference
- `DECISIONS.md` — Past architectural decisions and rationale (ADR-001 through ADR-030)

### Training (current focus)

- `TRAINING.md` — **Unified training plan v3.0** (single source of truth)
  - Spiral curriculum: 6 developmental stages (Naming → Specialization)
  - All 4 training signals from day one, progressively complex
  - Supersedes all previous training documents (archived in `archive/training/`)

### Development History

- `Dev_History.md` — Chronological record of all development (Feb 9-15, 2026)

### Roadmap (all implementation phases complete)

These files document what was built. All milestones are done:

- `roadmap/MILESTONE-0.md` — Test harness setup
- `roadmap/PHASE-1(DONE).md` — TensorFrame, LLL, Stub Server
- `roadmap/PHASE-2(DONE).md` — Codebook, Translator, RAR, GPU
- `roadmap/PHASE-3(DONE).md` — Router, Strands, Safety
- `roadmap/PHASE-4.md` — VoltDB T0/T1/T2, HNSW, GC
- `roadmap/PHASE-5.md` — Learning Events, Sleep, RLVF
- `roadmap/PHASE-6.md` — Module System (6.1 done; 6.2+ deferred to post-training)

### Other Reference (read when needed)

- `VOLT X — Master Blueprint.md` — Full system design narrative
- `SVGs.md` — Visual architecture diagrams (large file, read in 500-line chunks)
- `Appendix.md` — Spiral curriculum design rationale

### Training Data

**IMPORTANT:** All training data is stored on **D: drive** (NOT C: drive):

- Base location: `D:\VoltData\`
- Configuration: `c:\Volt_X\training_config.toml`

When implementing training code, always reference datasets via the paths
in `training_config.toml`. Prefer HuggingFace streaming over bulk downloads.
Never hardcode C: drive paths for data.

## Code Standards

### Language: Rust (edition 2024)

- All code must pass `cargo clippy -- -D warnings` with zero warnings
- All public functions must have doc comments with examples
- All structs must derive Debug, Clone. Serialize/Deserialize where
  data crosses boundaries
- No unwrap() in library code. Use Result<T, VoltError> everywhere.
  unwrap() allowed only in tests and benches
- No unsafe code without a comment explaining why it is necessary
  and a tracking issue to remove it

### Naming Conventions

- Crate names: volt-{component} (kebab-case)
- Module names: snake_case
- Types: PascalCase
- Functions: snake_case
- Constants: SCREAMING_SNAKE_CASE
- The word "Frame" always means TensorFrame, never video frame
- The word "Strand" always means a VoltDB strand, never a thread

### Error Handling

- Define VoltError enum in volt-core, re-export from all crates
- Use thiserror for error derivation
- Error messages must include context:
  "failed to load strand #{id} from T1: {inner_error}"
- Never silently swallow errors

### Testing Requirements

- Every public function must have at least one unit test
- Every crate must have integration tests in tests/ directory
- Tests must not depend on external services (no network, no GPU in
  unit tests)
- GPU tests are separate: `cargo test --features gpu`
- Use proptest for property-based testing on core data structures
- Benchmark critical paths with criterion

### Architecture Rules

- Dependencies flow one direction: core ← bus ← soft/hard/db ←
  translate/learn/safety ← server
- No circular dependencies between crates
- No crate may import from volt-server (server is the leaf)
- volt-core may not import from any other volt-* crate
- All cross-crate communication happens through TensorFrame

### When Writing New Code

1. Write the test first (or at least the test signature)
2. Implement the minimum code to pass the test
3. Run clippy and fix all warnings
4. Add doc comments
5. Run the full test suite: cargo test --workspace

### When Modifying Existing Code

1. Read the existing tests to understand current behavior
2. Add a test for the new behavior before modifying code
3. Ensure all existing tests still pass
4. Update doc comments if behavior changed
5. If this changes an architectural decision, update DECISIONS.md

### Performance Expectations

- TensorFrame creation: < 1μs
- LLL bind operation (256 dims): < 10μs
- HNSW query (65K entries): < 500μs
- Single RAR iteration (16 slots, CPU): < 1ms
- Full inference (simple query): < 50ms
- Full inference (complex query): < 300ms

### Things NOT to Do

- Do not add new external dependencies without justification in DECISIONS.md
- Do not use async in volt-core or volt-bus (pure synchronous logic)
- Do not put GPU code in any crate except volt-soft
- Do not put network code in any crate except volt-ledger and volt-server
