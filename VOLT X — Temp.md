

# The Volt X Development Pipeline

## The Reality: Solo Developer + Claude Code + VS Code

You're one person building something that would normally require a team. The pipeline needs to compensate for the absence of code reviewers, QA engineers, and DevOps staff. Claude Code becomes your pair programmer, code reviewer, and documentation writer. The pipeline automates everything that a team would do manually.

The core principle: never rely on your memory. Rely on the pipeline. If a test doesn't enforce a behavior, that behavior will break eventually. If a decision isn't documented, it will be re-debated. If a pattern isn't linted, it will drift.

---

## Workspace Structure

```
volt-x/
├── .cargo/
│   └── config.toml              # Shared build settings, link flags
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions: test + lint + bench on every push
├── .vscode/
│   ├── settings.json            # Workspace settings for VS Code + rust-analyzer
│   ├── tasks.json               # Build/test/bench tasks bound to hotkeys
│   └── extensions.json          # Recommended extensions list
├── CLAUDE.md                    # Instructions for Claude Code (critical file)
├── ARCHITECTURE.md              # The master blueprint (condensed for dev reference)
├── DECISIONS.md                 # Architecture Decision Records (ADR log)
├── Cargo.toml                   # Workspace root
├── crates/
│   ├── volt-core/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── frame.rs         # TensorFrame struct
│   │   │   ├── slot.rs          # SlotData, SlotRole, SlotMeta
│   │   │   ├── meta.rs          # FrameMeta
│   │   │   └── tests.rs         # Unit tests (same file or mod tests)
│   │   └── benches/
│   │       └── frame_bench.rs   # Criterion benchmarks
│   ├── volt-bus/                # LLL algebra
│   ├── volt-soft/               # GPU Soft Core (RAR)
│   ├── volt-hard/               # CPU Hard Core
│   ├── volt-db/                 # VoltDB storage engine
│   ├── volt-translate/          # Translators
│   ├── volt-learn/              # Continual learning
│   ├── volt-safety/             # Safety layer
│   ├── volt-ledger/             # Intelligence Commons
│   └── volt-server/             # Axum HTTP server
├── tests/
│   └── integration/
│       ├── smoke.rs             # Does the system start and respond?
│       ├── pipeline.rs          # End-to-end: text in → frame → text out
│       ├── memory.rs            # Store frame → recall frame
│       └── safety.rs            # Safety invariants hold
├── scripts/
│   ├── dev.sh                   # Start dev server with hot reload
│   ├── bench.sh                 # Run all benchmarks, compare to baseline
│   └── train.sh                 # Training scripts (Phase 2+)
├── data/
│   ├── codebook/                # Pre-computed codebook binaries
│   ├── models/                  # Trained model checkpoints
│   └── eval/                    # Evaluation datasets
└── n8n/
    └── volt-test-bench.json     # n8n workflow export
```

---

## The CLAUDE.md File — The Most Important File in the Repo

This file tells Claude Code how to behave in your project. It is the single highest-leverage file you will write.

```markdown
# CLAUDE.md — Instructions for Claude Code

## Project: Volt X
Volt X is a cognitive architecture implementing a stateful AI operating 
system. It uses Tensor Frames (not flat vectors), Root-Attend-Refine 
inference (not transformers), and three-tier memory (not context windows).

## Architecture Reference
Read ARCHITECTURE.md for the full design. Read DECISIONS.md for past 
architectural decisions and their rationale.

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
```

This file shapes every Claude Code interaction. When you ask Claude to write code, it follows these rules. When you ask it to review code, it checks against these rules. It is the team culture encoded as text.

---

## VS Code Configuration

### .vscode/settings.json

```json
{
  "rust-analyzer.check.command": "clippy",
  "rust-analyzer.check.extraArgs": ["--", "-D", "warnings"],
  "rust-analyzer.cargo.features": "all",
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "editor.tabSize": 4,
  "files.insertFinalNewline": true,
  "files.trimTrailingWhitespace": true,
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  },
  "search.exclude": {
    "/target": true,
    "/data/models": true
  },
  "todo-tree.regex.regex": "(TODO|FIXME|HACK|MILESTONE|DECISION):",
  "todo-tree.highlights.customHighlight": {
    "MILESTONE": {
      "icon": "milestone",
      "foreground": "#22d3ee",
      "type": "tag"
    },
    "DECISION": {
      "icon": "book",
      "foreground": "#f59e0b",
      "type": "tag"
    }
  }
}
```

### .vscode/tasks.json

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "test-all",
      "type": "shell",
      "command": "cargo test --workspace",
      "group": { "kind": "test", "isDefault": true },
      "problemMatcher": ["$rustc"],
      "presentation": { "reveal": "always", "panel": "dedicated" }
    },
    {
      "label": "test-crate",
      "type": "shell",
      "command": "cargo test -p ${input:crateName}",
      "problemMatcher": ["$rustc"]
    },
    {
      "label": "clippy",
      "type": "shell",
      "command": "cargo clippy --workspace -- -D warnings",
      "group": "build",
      "problemMatcher": ["$rustc"]
    },
    {
      "label": "bench",
      "type": "shell",
      "command": "cargo bench --workspace",
      "problemMatcher": []
    },
    {
      "label": "serve-dev",
      "type": "shell",
      "command": "cargo watch -x 'run -p volt-server'",
      "isBackground": true,
      "problemMatcher": []
    }
  ],
  "inputs": [
    {
      "id": "crateName",
      "type": "pickString",
      "description": "Which crate to test?",
      "options": [
        "volt-core", "volt-bus", "volt-soft", "volt-hard",
        "volt-db", "volt-translate", "volt-learn",
        "volt-safety", "volt-ledger", "volt-server"
      ]
    }
  ]
}
```

### Keybindings (add to keybindings.json)

| Shortcut | Action | Why |
|---|---|---|
| `Ctrl+Shift+T` | Run test-all task | Test everything after any change |
| `Ctrl+Shift+Y` | Run clippy task | Check lint before committing |
| `Ctrl+Shift+B` | Run bench task | Track performance regressions |
| `Ctrl+Shift+D` | Run serve-dev task | Hot-reload dev server |

---

## The Daily Workflow

Here is exactly how a development session works:

### Starting a Session

```
1. Open VS Code in volt-x/ workspace
2. Terminal 1: `cargo watch -x 'run -p volt-server'` (dev server)
3. Terminal 2: open for ad-hoc commands
4. Open DECISIONS.md — remind yourself of recent decisions
5. Open the current milestone's tracking issue — see what's next
```

### The Development Loop (per feature)

```svg
<svg viewBox="0 0 1000 400" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="wfBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f"/>
      <stop offset="100%" style="stop-color:#0a0a1a"/>
    </linearGradient>
  </defs>
  <rect width="1000" height="400" fill="url(#wfBg)" rx="12"/>
  <text x="500" y="30" text-anchor="middle" fill="#e2e8f0" font-size="15" font-weight="bold">The Development Loop (repeat for every feature)</text>

  <!-- Step 1 -->
  <rect x="30" y="55" width="175" height="70" rx="10" fill="#1a0050" stroke="#8b5cf6" stroke-width="2"/>
  <text x="118" y="80" text-anchor="middle" fill="#ddd6fe" font-size="10" font-weight="bold">1. THINK</text>
  <text x="118" y="97" text-anchor="middle" fill="#a78bfa" font-size="8">Ask Claude Code:</text>
  <text x="118" y="112" text-anchor="middle" fill="#8b5cf6" font-size="7">"How should X work?"</text>
  <line x1="205" y1="90" x2="235" y2="90" stroke="#94a3b8" stroke-width="1.5"/>
  <polygon points="232,86 240,90 232,94" fill="#94a3b8"/>

  <!-- Step 2 -->
  <rect x="240" y="55" width="175" height="70" rx="10" fill="#002020" stroke="#ef4444" stroke-width="2"/>
  <text x="328" y="80" text-anchor="middle" fill="#fca5a5" font-size="10" font-weight="bold">2. TEST FIRST</text>
  <text x="328" y="97" text-anchor="middle" fill="#f87171" font-size="8">Write the test that</text>
  <text x="328" y="112" text-anchor="middle" fill="#f87171" font-size="7">defines "done"</text>
  <line x1="415" y1="90" x2="445" y2="90" stroke="#94a3b8" stroke-width="1.5"/>
  <polygon points="442,86 450,90 442,94" fill="#94a3b8"/>

  <!-- Step 3 -->
  <rect x="450" y="55" width="175" height="70" rx="10" fill="#001a10" stroke="#10b981" stroke-width="2"/>
  <text x="538" y="80" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">3. IMPLEMENT</text>
  <text x="538" y="97" text-anchor="middle" fill="#34d399" font-size="8">Write minimum code</text>
  <text x="538" y="112" text-anchor="middle" fill="#059669" font-size="7">to pass the test</text>
  <line x1="625" y1="90" x2="655" y2="90" stroke="#94a3b8" stroke-width="1.5"/>
  <polygon points="652,86 660,90 652,94" fill="#94a3b8"/>

  <!-- Step 4 -->
  <rect x="660" y="55" width="155" height="70" rx="10" fill="#0a0a05" stroke="#f59e0b" stroke-width="2"/>
  <text x="738" y="80" text-anchor="middle" fill="#fde68a" font-size="10" font-weight="bold">4. VERIFY</text>
  <text x="738" y="97" text-anchor="middle" fill="#fcd34d" font-size="8">cargo test + clippy</text>
  <text x="738" y="112" text-anchor="middle" fill="#d97706" font-size="7">ALL tests pass?</text>
  <line x1="815" y1="90" x2="845" y2="90" stroke="#94a3b8" stroke-width="1.5"/>
  <polygon points="842,86 850,90 842,94" fill="#94a3b8"/>

  <!-- Step 5 -->
  <rect x="850" y="55" width="120" height="70" rx="10" fill="#001a20" stroke="#22d3ee" stroke-width="2"/>
  <text x="910" y="80" text-anchor="middle" fill="#a5f3fc" font-size="10" font-weight="bold">5. COMMIT</text>
  <text x="910" y="97" text-anchor="middle" fill="#67e8f9" font-size="8">Conventional</text>
  <text x="910" y="112" text-anchor="middle" fill="#0891b2" font-size="7">commit message</text>

  <!-- Fail loop -->
  <path d="M 738 125 Q 738 160 490 165 Q 328 170 328 125" stroke="#ef4444" stroke-width="1.5" fill="none" stroke-dasharray="5"/>
  <polygon points="331,128 325,120 319,128" fill="#ef4444"/>
  <text x="520" y="175" text-anchor="middle" fill="#ef4444" font-size="8">Test fails? → fix and re-test</text>

  <!-- Claude Code interactions -->
  <rect x="30" y="210" width="940" height="165" rx="12" fill="#0a0a15" stroke="#334155" stroke-width="1.5"/>
  <text x="500" y="235" text-anchor="middle" fill="#e2e8f0" font-size="12" font-weight="bold">How to Use Claude Code at Each Step</text>

  <rect x="50" y="250" width="200" height="55" rx="6" fill="#1a0050" stroke="#8b5cf6" stroke-width="1"/>
  <text x="150" y="268" text-anchor="middle" fill="#c4b5fd" font-size="9" font-weight="bold">Step 1: THINK</text>
  <text x="150" y="284" text-anchor="middle" fill="#a78bfa" font-size="7">"I need to implement X.</text>
  <text x="150" y="296" text-anchor="middle" fill="#a78bfa" font-size="7">What's the best approach?"</text>

  <rect x="270" y="250" width="200" height="55" rx="6" fill="#200020" stroke="#ef4444" stroke-width="1"/>
  <text x="370" y="268" text-anchor="middle" fill="#fca5a5" font-size="9" font-weight="bold">Step 2: TEST FIRST</text>
  <text x="370" y="284" text-anchor="middle" fill="#f87171" font-size="7">"Write tests for the X</text>
  <text x="370" y="296" text-anchor="middle" fill="#f87171" font-size="7">feature based on our spec."</text>

  <rect x="490" y="250" width="200" height="55" rx="6" fill="#001a10" stroke="#10b981" stroke-width="1"/>
  <text x="590" y="268" text-anchor="middle" fill="#6ee7b7" font-size="9" font-weight="bold">Step 3: IMPLEMENT</text>
  <text x="590" y="284" text-anchor="middle" fill="#34d399" font-size="7">"Implement X to pass</text>
  <text x="590" y="296" text-anchor="middle" fill="#34d399" font-size="7">the tests we just wrote."</text>

  <rect x="710" y="250" width="240" height="55" rx="6" fill="#0a0a05" stroke="#f59e0b" stroke-width="1"/>
  <text x="830" y="268" text-anchor="middle" fill="#fde68a" font-size="9" font-weight="bold">Step 4-5: VERIFY + COMMIT</text>
  <text x="830" y="284" text-anchor="middle" fill="#fcd34d" font-size="7">"Review this code for issues.</text>
  <text x="830" y="296" text-anchor="middle" fill="#fcd34d" font-size="7">Any edge cases I missed?"</text>

  <text x="500" y="340" text-anchor="middle" fill="#94a3b8" font-size="9" font-weight="bold">Each Claude Code interaction is scoped and specific. Never ask "build the whole feature."</text>
  <text x="500" y="358" text-anchor="middle" fill="#64748b" font-size="8">Always: one question, one answer, one action. Then verify. Then next question.</text>
</svg>
```

### The Claude Code Interaction Pattern

The key to productive Claude Code usage is scoped questions with context. Never ask "implement VoltDB." Instead:

Pattern 1: Design Question
```
"I'm working on milestone 4.1 (VoltDB Tier 0+T1). I need to implement 
the strand-organized HashMap for T1 storage. Looking at the TensorFrame 
struct in volt-core/src/frame.rs, what's the best way to organize 
strands for fast lookup by strand_id and fast iteration within a strand?"
```

Pattern 2: Test-First Request
```
"Write tests for the VoltDB T1 strand storage. It should support:
1. Create strand by ID
2. Append frame to strand
3. Get frame by frame_id
4. Get most recent N frames from strand
5. List all strands
Test with 3 strands, 100 frames each."
```

Pattern 3: Implementation Request
```
"The tests in volt-db/src/tests.rs define the T1 storage interface.
Implement the StrandStore struct to pass all tests. Use a 
HashMap<u64, Vec<TensorFrame>> internally. Follow the patterns 
in CLAUDE.md."
```

Pattern 4: Review Request
```
"Review the StrandStore implementation I just wrote. Check for:
- Does it follow CLAUDE.md rules?
- Any unwrap() in library code?
- Error handling complete?
- Edge cases: empty strand, duplicate frame_id, concurrent access?
- Performance: any O(n) operations that should be O(1)?"
```

Pattern 5: Refactor Request
```
"The StrandStore works but I realize the lookup is O(n) per strand.
Refactor to add a secondary HashMap<u64, (u64, usize)> that maps 
frame_id → (strand_id, index_in_vec). Keep all tests passing."
```

---

## Git Workflow

### Branching Strategy

For solo development, keep it simple:

```
main          → always compiles, all tests pass, stable
  ├── dev     → your working branch, may have WIP commits
  └── m/X.Y   → milestone branches (e.g., m/1.1, m/2.3)
```

Each milestone gets its own branch off `dev`. When the milestone is complete (all tests pass, all criteria met), merge into `dev`. Periodically merge `dev` into `main` when you have a stable checkpoint.

### Commit Message Convention

```
<type>(<crate>): <description>

Types:
  feat     - new feature
  fix      - bug fix
  test     - adding or fixing tests
  refactor - restructuring without changing behavior
  perf     - performance improvement
  docs     - documentation only
  chore    - build, CI, tooling changes

Examples:
  feat(volt-core): implement TensorFrame slot write/read
  test(volt-bus): add property tests for bind/unbind roundtrip
  fix(volt-db): prevent panic on empty strand query
  perf(volt-soft): batch 16 VFN passes in single CUDA kernel
  docs(volt-hard): add MathEngine capability description
  refactor(volt-db): extract HNSW index into separate module
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

set -e

echo "Running clippy..."
cargo clippy --workspace -- -D warnings

echo "Running tests..."
cargo test --workspace --quiet

echo "Checking formatting..."
cargo fmt --check

echo "All checks passed."
```

This runs automatically before every commit. If anything fails, the commit is rejected. You cannot commit broken code.

---

## CI Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, dev, "m/"]
  pull_request:
    branches: [main, dev]

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: "-D warnings"

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - uses: Swatinem/rust-cache@v2
      
      - name: Format check
        run: cargo fmt --check
      
      - name: Clippy
        run: cargo clippy --workspace -- -D warnings
      
      - name: Tests
        run: cargo test --workspace
      
      - name: Doc tests
        run: cargo test --workspace --doc

  bench:
    runs-on: ubuntu-latest
    needs: check
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      
      - name: Benchmarks
        run: cargo bench --workspace -- --output-format bencher | tee bench_output.txt
      
      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: cargo
          output-file-path: bench_output.txt
          auto-push: true
```

This gives you:
- Every push: format + lint + test (fails fast, blocks broken merges)
- Every merge to main: benchmarks (tracks performance over time)
- Benchmark history visible as a graph in GitHub Pages

---

## The DECISIONS.md Log

Every architectural decision gets recorded. This prevents re-debating settled questions and provides context for Claude Code:

```markdown
# Architecture Decision Records

## ADR-001: Workspace Structure (2026-02-09)
Decision: Use Cargo workspace with one crate per component.
Reason: Independent compilation, independent testing, 
enforced dependency direction.
Alternatives considered: Single crate with modules (rejected: 
circular dependencies too easy), separate repos (rejected: too much 
overhead for solo dev).

## ADR-002: TensorFrame Dimensions (2026-02-09)
Decision: S=16 slots, R=4 resolutions, D=256 dims per slot.
Reason: 16 slots covers all semantic roles with 7 free. 
4 resolutions span discourse→token. 256 dims balances expressiveness 
with compute cost. Total max size 64KB fits comfortably in cache lines.
Alternatives considered: S=8 (too few free slots), S=32 (excessive 
for most queries), D=512 (doubles compute for marginal expressiveness).

## ADR-003: Error Handling (2026-02-09)
Decision: VoltError enum in volt-core, thiserror derivation, 
no unwrap in library code.
Reason: Consistent error handling prevents silent failures. 
thiserror provides ergonomic error types. Banning unwrap forces 
explicit error handling at every boundary.
```

When you work with Claude Code, you can say: "Check DECISIONS.md — we already decided X about Y for reason Z."

---

## Quality Gates: What Must Be True Before Moving On

Each milestone has explicit quality gates. You do not start the next milestone until all gates pass:

```echarts
{
  "backgroundColor": "#0f172a",
  "title": {
    "text": "Quality Gates Per Phase",
    "left": "center",
    "textStyle": { "color": "#e2e8f0", "fontSize": 14 }
  },
  "tooltip": { "trigger": "axis", "backgroundColor": "#1e293b", "textStyle": { "color": "#e2e8f0" } },
  "radar": {
    "indicator": [
      { "name": "All Tests\nPass", "max": 1 },
      { "name": "Zero Clippy\nWarnings", "max": 1 },
      { "name": "Benchmarks\nMeet Target", "max": 1 },
      { "name": "Doc Comments\nComplete", "max": 1 },
      { "name": "Integration\nTest Works", "max": 1 },
      { "name": "DECISIONS.md\nUpdated", "max": 1 }
    ],
    "shape": "polygon",
    "splitArea": { "areaStyle": { "color": ["#0f172a", "#1e293b"] } },
    "axisLine": { "lineStyle": { "color": "#334155" } },
    "splitLine": { "lineStyle": { "color": "#1e293b" } },
    "axisName": { "color": "#94a3b8", "fontSize": 10 }
  },
  "series": [
    {
      "type": "radar",
      "data": [
        {
          "value": [1, 1, 1, 1, 1, 1],
          "name": "Must be ALL GREEN to proceed",
          "lineStyle": { "color": "#22c55e", "width": 2 },
          "itemStyle": { "color": "#22c55e" },
          "areaStyle": { "color": "rgba(34,197,94,0.2)" }
        }
      ]
    }
  ]
}
```

Gate 1: All Tests Pass. `cargo test --workspace` exits with code 0. No skipped tests. No ignored tests (unless explicitly documented with reason and tracking issue).

Gate 2: Zero Clippy Warnings. `cargo clippy --workspace -- -D warnings` produces no output. Not "warnings acknowledged" — zero warnings.

Gate 3: Benchmarks Meet Target. Performance targets from CLAUDE.md are met or exceeded. If a benchmark regresses, investigate and fix before proceeding.

Gate 4: Doc Comments Complete. Every public function, struct, and enum has a doc comment. `cargo doc --workspace --no-deps` produces no warnings.

Gate 5: Integration Test Works. The end-to-end integration test (text in → frame → process → text out) passes with the new feature integrated.

Gate 6: DECISIONS.md Updated. Any architectural decisions made during the milestone are recorded with rationale.

---

## The Weekly Rhythm

```
Monday:    Plan the week. Review milestone progress. Update tracking issue.
           Ask Claude Code: "Review our progress on milestone X.Y. What's left?"

Tue-Thu:   Build. Follow the development loop. Aim for 1-2 features per day.
           Each feature: think → test → implement → verify → commit.

Friday:    Review + polish. Run full benchmark suite. Update documentation.
           Ask Claude Code: "Review all code changed this week. Any issues?"
           Write weekly dev log (even just 5 bullet points).

Weekend:   Optional: let sleep consolidation run (when it exists).
           Think about next week's approach. Read relevant papers/docs.
```

---

## The Emergency Protocol: When Things Break

When you hit a wall (and you will), follow this exact sequence:

```
1. Stop writing code.
2. Write down what you expected to happen and what actually happened.
3. Run `cargo test --workspace` — identify exactly which tests fail.
4. Ask Claude Code: "I'm stuck on X. Expected Y, got Z. 
   Here are the failing tests. What's wrong?"
5. If Claude's suggestion doesn't work after 2 attempts, step back:
   - Is the test correct? (Maybe the test is wrong, not the code.)
   - Is the design correct? (Maybe the approach needs to change.)
   - Is this the right milestone order? (Maybe you need something 
     from a later milestone now.)
6. If stepping back reveals a design issue: update DECISIONS.md, 
   revise the approach, and continue.
7. Never "hack around" a problem. Either fix it properly or 
   document it as a known issue with a tracking item.
```

The pipeline exists so that when things go wrong (and they will), you have a systematic way to find and fix the problem instead of flailing. Every test is a checkpoint. Every decision is recorded. Every commit is reversible. The pipeline is your safety net.

Build the net first. Then do the tightrope walk.