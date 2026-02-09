# CLAUDE.md — Instructions for Claude Code

## Project: Volt X
Volt X is a cognitive architecture implementing a stateful AI operating 
system. It uses Tensor Frames (not flat vectors), Root-Attend-Refine 
inference (not transformers), and three-tier memory (not context windows).

## Context Documents — Read Before Working

When starting a new session or task, read the relevant documents below
to load context efficiently. Do NOT read the full master roadmap
(`VOLT X — Roadmap.md`) — use the per-phase extracts instead.

### Always Read

- `ARCHITECTURE.md` — Full technical design reference
- `DECISIONS.md` — Past architectural decisions and rationale

### Roadmap (read only the current phase)

- `roadmap/MILESTONE-0.md` — Test harness setup (Week 0)
- `roadmap/PHASE-1.md` — The Skeleton: TensorFrame, LLL, Stub Server (Months 1-2)
- `roadmap/PHASE-2.md` — The Soft Core: Codebook, Translator, RAR, GPU (Months 3-4)
- `roadmap/PHASE-3.md` — The Hard Core: Router, Strands, Safety (Months 5-6)
- `roadmap/PHASE-4.md` — Memory: VoltDB T0/T1/T2, HNSW, GC (Months 7-8)
- `roadmap/PHASE-5.md` — Learning: Events, Sleep, RLVF (Months 9-10)
- `roadmap/PHASE-6.md` — Ecosystem: Modules, Commons, P2P (Months 11-12)

### Other Reference (read when needed)

- `VOLT X — Master Blueprint.md` — Full system design narrative
- `SVGs.md` — Visual architecture diagrams (large file, read in 500-line chunks)
- `VOLT X — Roadmap.md` — Full master roadmap (prefer per-phase files above)

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
- Do not implement features from future milestones. Stub them with 
  todo!() and a comment referencing the milestone number
