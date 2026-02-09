

# Volt X Development Roadmap

## The Engineering Philosophy: Build the Skeleton, Then Add Organs

The cardinal sin of ambitious projects is building everything at once. You end up with 50 half-working components that all depend on each other, and a bug in any one of them makes the whole system untestable. This roadmap follows a different principle:

Every milestone produces a working, testable system. At no point do you have code that "will work once the other parts are done." Each step adds one capability to an already-functioning system. If you stop at any milestone, you have something useful.

The order is chosen to maximize the feedback loop speed — you want to see results immediately so you can course-correct before bad decisions calcify.

---

## Roadmap Overview

```echarts
{
  "backgroundColor": "#0f172a",
  "title": {
    "text": "Volt X Development Roadmap: 12 Months",
    "subtext": "24 Milestones across 6 Phases — every milestone is a working system",
    "left": "center",
    "textStyle": { "color": "#e2e8f0", "fontSize": 16 },
    "subtextStyle": { "color": "#64748b" }
  },
  "tooltip": { "trigger": "axis", "backgroundColor": "#1e293b", "textStyle": { "color": "#e2e8f0" } },
  "xAxis": {
    "type": "category",
    "data": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12"],
    "axisLabel": { "color": "#94a3b8" },
    "axisLine": { "lineStyle": { "color": "#334155" } },
    "name": "Month",
    "nameTextStyle": { "color": "#64748b" }
  },
  "yAxis": {
    "type": "value",
    "show": false
  },
  "series": [
    {
      "name": "Phase 1: Skeleton",
      "type": "bar",
      "stack": "total",
      "data": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "itemStyle": { "color": "#8b5cf6" },
      "label": { "show": false }
    },
    {
      "name": "Phase 2: Soft Core",
      "type": "bar",
      "stack": "total",
      "data": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      "itemStyle": { "color": "#22d3ee" }
    },
    {
      "name": "Phase 3: Hard Core",
      "type": "bar",
      "stack": "total",
      "data": [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
      "itemStyle": { "color": "#f59e0b" }
    },
    {
      "name": "Phase 4: Memory",
      "type": "bar",
      "stack": "total",
      "data": [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
      "itemStyle": { "color": "#ec4899" }
    },
    {
      "name": "Phase 5: Learning",
      "type": "bar",
      "stack": "total",
      "data": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
      "itemStyle": { "color": "#10b981" }
    },
    {
      "name": "Phase 6: Ecosystem",
      "type": "bar",
      "stack": "total",
      "data": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
      "itemStyle": { "color": "#f97316" }
    }
  ],
  "legend": {
    "data": ["Phase 1: Skeleton", "Phase 2: Soft Core", "Phase 3: Hard Core", "Phase 4: Memory", "Phase 5: Learning", "Phase 6: Ecosystem"],
    "bottom": 5,
    "textStyle": { "color": "#94a3b8" }
  }
}
```

---

## The Golden Rule: Test Harness First

Before writing a single line of Volt code, build the test harness. This is the scaffold that every subsequent milestone will be tested against.

### Milestone 0: The Test Harness (Week 0 — before everything)

What you build:

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

Why this structure matters:
- Each crate compiles independently. A bug in `volt-ledger` does not prevent `volt-core` from compiling.
- Each crate has its own unit tests. You can run `cargo test -p volt-core` without touching anything else.
- Dependencies flow one way: core ← bus ← soft/hard/db ← translate ← server. No circular dependencies.
- New crates can be added without modifying existing ones.

What you test:
- `cargo test` passes on empty crates (all compile, zero tests, zero failures)
- `cargo clippy` produces zero warnings
- CI pipeline runs on every commit (GitHub Actions, ~2 minutes)

Duration: 1 day.

---

## Phase 1: The Skeleton (Months 1-2)

The goal of Phase 1 is: a dumb system that takes text in and text out, with TensorFrames flowing through the entire pipeline. The Soft Core is a stub (copies input to output). The Hard Core is a stub (passes through). But the Frame Bus works, the data structures work, the server works, and n8n can talk to it.

### Milestone 1.1: TensorFrame Data Structure (Week 1-2)

Crate: `volt-core`

What you build:
- `TensorFrame` struct with 16 slots, 4 resolutions, 256 dims
- `SlotData`, `SlotMeta`, `FrameMeta` structs
- `SlotRole` enum
- Serialization/deserialization via `serde` and `rkyv` (zero-copy)
- Frame creation, cloning, slot read/write, merge operations
- Unit normalization (per-slot, per-resolution)

What you test:
- Create a frame, write to slot 3 at R₁, read it back → identical
- Merge two frames → slots from both present, conflicts resolved by γ
- Serialize to bytes, deserialize → bit-identical
- `rkyv` zero-copy: archived bytes are usable without parsing
- Frame size: empty frame < 100 bytes, full frame = 64 KB exactly

What you explicitly do NOT build yet:
- No GPU code. Frames are CPU-only for now.
- No codebook. Slots hold raw float arrays.
- No HNSW index.

Why this order: Everything else depends on TensorFrame. If this data structure is wrong, everything built on top is wrong. Get it right first.

Duration: 2 weeks.

### Milestone 1.2: LLL Algebra (Week 3-4)

Crate: `volt-bus`

What you build:
- FFT-based binding: `bind(a, b) → c` using `rustfft` crate
- Unbinding via involution: `unbind(c, a) → b_approx`
- Superposition: `superpose(vec![a, b, c]) → normalized_sum`
- Permutation: `permute(a, k) → cyclic_shift_by_k`
- Cosine similarity: `sim(a, b) → f32`
- Operations work on individual slot vectors (256 dims)
- Batch operations: apply bind/unbind to entire frame (all slots)

What you test:
- `unbind(bind(a, b), a)` ≈ `b` with cosine similarity > 0.85
- `sim(superpose([a, b]), a)` > 0 (constituent detectable)
- `sim(bind(a, b), a)` ≈ 0 (bound pair is dissimilar to inputs)
- `sim(permute(a, 1), permute(a, 2))` ≈ 0 (different shifts are orthogonal)
- Performance: bind on 256 dims < 10μs

Why this order: The algebra is the grammar of thought. It must be correct before the Soft Core tries to use it.

Duration: 2 weeks.

### Milestone 1.3: Stub Translator + HTTP Server (Week 5-6)

Crate: `volt-translate` + `volt-server`

What you build:
- Stub Forward Translator: Takes raw text, splits into words, assigns first word to S₀ (AGENT), verb to S₁ (PREDICATE), object to S₂ (PATIENT). No ML. Just heuristic word-to-slot mapping. Each word is encoded as a random but deterministic vector (hash of word → seed → random 256-dim vector, normalized).
- Stub Reverse Translator: Takes TensorFrame, reads slot metadata, produces template: "Agent [S₀] does [S₁] to [S₂]."
- Axum HTTP Server:
  - `POST /api/think` → accepts JSON `{ "text": "..." }`, returns JSON `{ "text": "...", "gamma": [...], "strand_id": 0, "iterations": 1 }`
  - Health endpoint: `GET /health`

What you test:
- Send "The cat sat on the mat" → get back a response with 3 filled slots
- Round-trip: encode → decode → readable text (even if clumsy)
- Server starts, responds to curl, handles 100 concurrent requests without crash
- Invalid input (empty string, huge input, binary garbage) → graceful error, not panic

What you explicitly do NOT build yet:
- No LLM backbone. The translator is a dumb heuristic.
- No GPU. Everything runs on CPU.

Why this order: You now have a testable end-to-end system. Text goes in, TensorFrame flows through the pipeline, text comes out. Every subsequent milestone improves one component of this working pipeline.

Duration: 2 weeks.

### Milestone 1.4: n8n Integration (Week 7-8)

Crate: `volt-server` (extend)

What you build:
- n8n workflow: Chat Trigger → HTTP Request to `localhost:8080/api/think` → Switch (text/tool/error) → Chat Reply
- Debug output: the JSON response includes `slot_states`, `timing_ms`, `strand_id`
- n8n displays these in a "Debug Panel" node (Set node formatting)

What you test:
- Open n8n, type in chat, get response
- Response includes slot breakdown visible in n8n execution log
- Handle errors gracefully (server down → n8n shows error, not crash)

Checkpoint: At the end of Phase 1, you have a working chat system. It's stupid (heuristic translator, no reasoning), but the entire pipeline works: n8n → HTTP → Translate → TensorFrame → Bus → Stub Process → Translate Back → HTTP → n8n. Every subsequent phase improves one part of this pipeline without breaking the others.

---

## Phase 2: The Soft Core (Months 3-4)

The goal: replace the stub processor with actual GPU-based RAR reasoning. The system should start producing meaningfully different outputs based on the complexity of the input.

### Milestone 2.1: VQ-VAE Codebook (Week 9-10)

Crate: `volt-bus` (extend)

What you build:
- 65,536-entry codebook: `[65536, 256]` float array
- HNSW index over codebook entries using `hnsw_rs` or custom implementation
- `quantize(vector) → (codebook_id: u16, quantized_vector: [f32; 256])`
- `lookup(codebook_id) → [f32; 256]`
- Codebook initialization: K-Means clustering over word embeddings from a pretrained model (download embeddings, cluster offline, save as binary)

What you test:
- Quantize a vector → lookup by ID → cosine sim to original > 0.85
- HNSW query: 1000 random queries, each returns nearest codebook entry in < 0.5ms
- Codebook utilization after initializing from embeddings: > 80% of entries used

Duration: 2 weeks.

### Milestone 2.2: Real Forward Translator (Week 11-13)

Crate: `volt-translate` (replace stub)

What you build:
- Load a frozen LLM backbone (start small: `TinyLlama-1.1B` or `Phi-2`) via `candle` or `llama.cpp` bindings
- Frame Projection Head: lightweight MLP (3 layers, 50M params) mapping LLM hidden states → slot assignments + slot vectors
- Training pipeline:
  - Download PropBank/FrameNet data
  - For each annotated sentence: LLM hidden states → Frame Projection Head → predicted slots
  - Loss: slot assignment cross-entropy + codebook quantization commitment + round-trip reconstruction
  - Optimizer: AdamW, lr=1e-4, batch=32
  - Train on single GPU for ~7 days

What you test:
- "The cat sat on the mat" → S₀=cat (AGENT), S₁=sat (PRED), S₂=mat (LOCATION)
- Slot assignment accuracy > 80% on PropBank validation set
- Round-trip BLEU > 0.70 (encode → decode → compare to original)
- Codebook utilization > 75%

Duration: 3 weeks (including training time).

### Milestone 2.3: Basic RAR Loop on CPU (Week 14-15)

Crate: `volt-soft`

What you build:
- First, implement RAR on CPU (not GPU yet). This lets you debug the algorithm without CUDA complexity.
- Slot-local VFN: a simple MLP (4 layers, 256→512→512→256). Randomly initialized. Not trained yet — just verify the loop mechanics.
- RAR phases:
  - Root: apply VFN to each active slot independently
  - Attend: compute 16×16 attention matrix (Q, K, V projections + softmax)
  - Refine: state update + unit normalization + convergence check
- Convergence detection: per-slot ‖ΔS‖ < ε
- Budget enforcement: maximum 50 iterations
- Progressive freezing: converged slots skip Root phase

What you test:
- Random input frame → RAR loop runs → eventually converges (all slots ‖Δ‖ < ε)
- Easy input (few filled slots) → converges in < 5 iterations
- Complex input (many filled slots) → takes more iterations
- Frozen slots don't change between iterations
- Timing: 50 iterations on CPU < 500ms (not fast, but testable)

What you explicitly do NOT build yet:
- No GPU. CPU only.
- No trained VFN. Random weights.
- No ghost frames or bleed buffer.
- No diffusion noise.

Why CPU first: GPU debugging is hell. CUDA errors are opaque. CPU debugging is transparent: you can step through every value in a debugger. Get the algorithm right on CPU, then port to GPU.

Duration: 2 weeks.

### Milestone 2.4: GPU Port + VFN Training (Week 16-18)

Crate: `volt-soft` (extend)

What you build:
- Port RAR loop to GPU using `cudarc` or `candle` CUDA backend
- Parallelize Root phase: all 16 slot VFN passes in one batched CUDA kernel
- Attention phase: batched matrix multiply on GPU
- Add diffusion noise injection (per-slot adaptive σ)
- Train VFN via Flow Matching:
  - Generate (question, answer) frame pairs from training data
  - Linear interpolation path: `F(t) = (1-t)·F_q + t·F_a`
  - Train VFN to predict `(F_a - F_q)` at every t
  - Loss: MSE on drift direction
  - Train for ~2 weeks on single GPU

What you test:
- GPU RAR produces same results as CPU RAR (bit-close, within float precision)
- GPU is > 10× faster than CPU implementation
- After VFN training: convergence rate > 80% on validation question-answer pairs
- Adaptive computation: simple questions converge in < 5 iterations, complex in > 10
- Diffusion: increasing σ produces more diverse outputs for creative queries

Checkpoint: At the end of Phase 2, you have a system that actually thinks. The GPU runs the RAR loop, slots converge at different rates, and the VFN produces meaningful drift directions. The outputs are noticeably better than Phase 1's heuristic translator.

Duration: 3 weeks.

---

## Phase 3: The Hard Core (Months 5-6)

The goal: add deterministic CPU tools and verification. The system should now be able to do math exactly, execute code, verify its own outputs, and produce proof chains.

### Milestone 3.1: Intent Router + First Hard Strand (Week 19-20)

Crate: `volt-hard`

What you build:
- Intent Router: receives TensorFrame from Soft Core, computes cosine similarity against registered Hard Strand capability vectors, routes to best match
- MathEngine Hard Strand: implements `HardStrand` trait, handles arithmetic, algebra, basic calculus
- Integration: Soft Core → Intent Router → MathEngine → result injected back into frame

What you test:
- "What is 847 × 392?" → MathEngine activates → exact answer 331,824 → γ = 1.0
- "Tell me about cats" → no Hard Strand activates → passes through Soft Core only
- Router correctly distinguishes math queries from non-math queries (>95% accuracy on 100 test cases)
- MathEngine returns in < 1ms

Duration: 2 weeks.

### Milestone 3.2: More Hard Strands (Week 21-23)

Crate: `volt-hard` (extend)

What you build:
- CodeRunner: sandboxed code execution via `wasmtime`. Takes code from frame, runs it, returns stdout/stderr.
- HDCAlgebra: exposes bind/unbind/superpose as a callable Hard Strand for compositional reasoning
- CertaintyEngine: min-rule propagation across frame slots. Computes frame-level γ.
- ProofConstructor: records which Hard Strands were called, what they returned, builds proof chain.

What you test:
- CodeRunner: `print(2+2)` → output "4" in sandboxed environment. Malicious code (file access, network) → blocked.
- CertaintyEngine: frame with γ=[1.0, 0.8, 0.6] → global γ = 0.6
- ProofConstructor: after processing, proof chain has >= 2 steps, each with source and γ

Duration: 3 weeks.

### Milestone 3.3: Safety Layer (Week 24-25)

Crate: `volt-safety`

What you build:
- Axiomatic Guard: 5 hardcoded invariants (K₁-K₅) as constant vectors
- Transition Monitor: checks every frame transition against invariants (inner product)
- Violation Scorer: computes violation score, triggers warning or halt
- Omega Veto: when triggered, freezes all processing, returns safe default, logs state
- Integration: Safety layer wraps the entire Soft Core → Hard Core pipeline

What you test:
- Normal query → safety layer passes through, no interference
- Query touching K₁ (harm) → violation detected → Omega Veto fires → safe default response
- Omega Veto logs include full frame state at time of trigger
- Safety layer adds < 1ms latency to normal queries
- Cannot bypass safety by crafting special frame structures (adversarial testing)

Checkpoint: At the end of Phase 3, you have a verified AI. It thinks (GPU), acts (CPU tools), and verifies (certainty + proofs + safety). The outputs now include γ scores and proof chains. Math is exact. Code execution is sandboxed. Safety cannot be bypassed.

Duration: 2 weeks.

---

## Phase 4: Memory (Months 7-8)

The goal: add persistent memory so the system remembers across conversations. This is where Volt stops being a chatbot and becomes a personal AI.

### Milestone 4.1: VoltDB Tier 0 + Tier 1 (Week 26-28)

Crate: `volt-db`

What you build:
- Tier 0 (Working Memory): in-memory ring buffer of 64 TensorFrames. LRU eviction.
- Tier 1 (Short-Term Memory): strand-organized HashMap<StrandId, Vec<TensorFrame>>. Frames persist in RAM across queries.
- Basic strand management: create strand, switch strand, list strands
- Frame storage: every processed frame auto-stores in T0, evicts to T1 when T0 fills
- Basic retrieval: get frame by ID, get frames by strand, get most recent N frames

What you test:
- Ask 100 questions → all 100 frames stored → retrievable by ID
- T0 fills at 64 → oldest frame moves to T1 → still retrievable from T1
- Switch strand → new queries go to new strand → old strand intact
- Restart server → T1 persists (serialize to disk on shutdown, reload on start)
- Retrieval by ID: < 0.1ms

What you explicitly do NOT build yet:
- No HNSW index (linear scan is fine for thousands of frames)
- No Tier 2 compression
- No ghost frames or bleed
- No GC

Duration: 3 weeks.

### Milestone 4.2: HNSW Indexing + Ghost Bleed (Week 29-31)

Crate: `volt-db` (extend)

What you build:
- HNSW index over all frame R₀ gists within each strand
- Semantic retrieval: `query_similar(frame_gist, k=10) → top-10 similar frames`
- B-tree temporal index: `query_range(start_time, end_time) → frames in range`
- Ghost Bleed Buffer: array of ~1000 R₀ gists loaded into a buffer accessible by Soft Core
- Bleed Engine: on every new frame, query HNSW → update ghost buffer with top relevant gists
- Integration with RAR Attend phase: ghost frames participate as additional Key/Value in cross-attention

What you test:
- Ask about Rust → get back relevant frames from Coding strand (not Cooking strand)
- Ask about something from 2 weeks ago → HNSW finds it → ghost appears → full frame loads on page fault
- Ghost buffer refreshes when topic changes (new frame gist differs from previous)
- Semantic retrieval accuracy: > 80% of top-10 results are genuinely relevant (manual evaluation on 50 queries)

Duration: 3 weeks.

### Milestone 4.3: Tier 2 + GC + Consolidation (Week 32-34)

Crate: `volt-db` (extend)

What you build:
- Tier 2 (Long-Term Memory): compressed frame storage (R₀ only, 1 KB per frame) on disk via `mmap`
- LSM-Tree structure: memtable → flush to sorted run → periodic compaction
- GC pipeline: retention score computation → 4-tier decay (full → compressed → gist → tombstone)
- Bloom filters on LSM runs for fast negative checks
- MVCC: `crossbeam-epoch` for lock-free readers, per-strand Mutex for writers
- WAL: append-only log per strand for crash recovery
- Consolidation: batch of similar frames → distilled summary frame (simple averaging for now, no FF yet)

What you test:
- Store 1 million frames → retrieval still < 5ms via HNSW + B-tree
- GC correctly tombstones old, low-γ, unreferenced frames
- Immortal frames (high γ, user-pinned) survive GC indefinitely
- Crash test: kill process mid-write → restart → no data loss beyond current frame
- Concurrent read/write: 10 reader threads + 1 writer thread → no deadlocks, no corruption
- Memory usage grows sublinearly: 1M frames < 2 GB RAM + 1 GB disk

Checkpoint: At the end of Phase 4, Volt remembers everything. Conversations from months ago are retrievable. Ghost frames subtly influence current reasoning. The system feels like it knows you.

Duration: 3 weeks.

---

## Phase 5: Learning (Months 9-10)

The goal: activate the self-improvement loop. The system should get measurably better over time through use.

### Milestone 5.1: Learning Event Logging (Week 35-36)

Crate: `volt-learn`

What you build:
- Learning event struct: `{ frame_id, strand_id, query_type, γ_scores, convergence_iterations, ghost_activations, timestamp }`
- Every inference run automatically logs a learning event to strand metadata
- Learning event buffer: accumulates events, flushable to disk
- Basic statistics: per-strand query count, average γ, average iteration count, topic distribution

What you test:
- After 100 queries, learning buffer has 100 events
- Statistics reflect actual usage patterns (more coding queries → coding strand dominates)
- Events survive restart (persisted alongside strand in VoltDB)

Duration: 2 weeks.

### Milestone 5.2: Sleep Consolidation (Week 37-39)

Crate: `volt-learn` (extend)

What you build:
- Sleep scheduler: triggers consolidation when system idle > 10 minutes OR on manual command
- Frame distillation: clusters of related frames within a strand → averaged into wisdom frames
- Forward-Forward VFN update:
  - Positive data: high-γ verified frames (goodness target: high)
  - Negative data: low-γ rejected frames, corrupted frames (goodness target: low)
  - Update one VFN layer at a time, discard activations between layers
  - VRAM usage: approximately 1× inference
- Strand graduation: when Mirror Module detects a cluster of > 50 frames about an unrecognized topic → promote to new strand

What you test:
- After sleep consolidation: VFN produces measurably lower loss on validation set (even if small improvement)
- Distillation: 50 raw frames about Rust lifetimes → 3-5 wisdom frames that retain key information
- Strand graduation: after 50+ cooking-related queries → "Cooking" strand automatically created
- VRAM during sleep: does not exceed 1.5× inference VRAM
- System remains responsive during consolidation (consolidation runs on background threads)

Duration: 3 weeks.

### Milestone 5.3: RLVF Joint Alignment (Week 40-42)

Crate: `volt-learn` (extend)

What you build:
- Evaluation dataset: 1000 (question, verified_answer) pairs across math, logic, factual, creative
- RLVF loop:
  1. Run full Volt pipeline on question → output frame with γ
  2. Compare to verified answer (cosine similarity of frame slots)
  3. Compute reward: correct + calibrated γ → positive; overconfident error → strong negative; honest uncertainty → mild
  4. Update VFN via policy gradient (REINFORCE with baseline)
- Certainty calibration metric: "when Volt says γ=0.9, is it correct 90% of the time?"
- Self-play for logic: automatically generate simple logic puzzles, run Volt, grade proof chains

What you test:
- After RLVF: certainty calibration improves (measured by reliability diagram)
- Self-play: Volt solves > 80% of generated logic puzzles with valid proofs
- No safety regression: re-run adversarial safety tests → all still pass
- Overall quality: human evaluation of 50 random queries shows improvement vs pre-RLVF

Checkpoint: At the end of Phase 5, Volt improves itself. Every conversation makes it better. Sleep consolidation refines its understanding. RLVF calibrates its confidence. It measurably gets smarter over weeks of use.

Duration: 3 weeks.

---

## Phase 6: Ecosystem (Months 11-12)

The goal: open the platform to the community and launch the Intelligence Commons.

### Milestone 6.1: Trait Specification + Module Hot-Plug (Week 43-44)

Crate: all crates (refine interfaces)

What you build:
- Finalize and document the three traits: `Translator`, `HardStrand`, `ActionCore`
- Module discovery: at startup, scan installed crates for trait implementations via feature flags
- Hot-plug infrastructure: `volt install <module-name>` downloads crate, compiles, registers with router
- Example module: `volt-strand-weather` (fetches weather API, demonstrates HardStrand implementation)

What you test:
- Install example module → restart → Intent Router correctly routes weather queries to it
- Uninstall module → restart → weather queries fall back to Soft Core
- Module with a bug (panic) → caught by Volt, logged, does not crash the system
- Documentation: another developer can read the docs and build a module in < 1 day

Duration: 2 weeks.

### Milestone 6.2: Community Translators + Action Cores (Week 45-46)

Crate: `volt-translate` (extend)

What you build:
- Vision Translator prototype: uses CLIP or SigLIP to encode images → frame slots
- Speech Action Core prototype: uses Bark or Piper for TTS from frame text output
- Package both as example community modules with full documentation
- Module publishing guide: how to publish to crates.io / Volt module registry

What you test:
- Send image → Vision Translator → TensorFrame with object labels in AGENT/PATIENT slots
- Text response → Speech Action Core → audio file
- Both work as hot-pluggable modules (install/uninstall)

Duration: 2 weeks.

### Milestone 6.3: Intelligence Commons Layer 0 (Week 47-48)

Crate: `volt-ledger`

What you build:
- Local event log: append-only, Merkle-hashed entries
- Ed25519 keypair generation and management (self-sovereign identity)
- Strand export: serialize strand → encrypt → sign → shareable binary
- Strand import: verify signature → decrypt → merge into VoltDB
- Basic fact logging: each verified frame (γ > 0.95) logged with provenance

What you test:
- Export Coding strand → import on different Volt instance → all frames intact
- Tampered export (modified bytes) → import rejects (signature verification fails)
- Event log is append-only (cannot modify past entries)
- 10,000 logged events → Merkle root computable in < 100ms

Duration: 2 weeks.

### Milestone 6.4: P2P Mesh + Settlement Prototype (Week 49-52)

Crate: `volt-ledger` (extend)

What you build:
- libp2p integration: peer discovery, gossip protocol for fact sharing
- CRDT-based event log synchronization between peers (eventual consistency)
- Module registry: content-addressed (CID) module binaries shared via P2P
- Settlement prototype: simple DAG-based micropayment tracking (not production-grade — proof of concept)

What you test:
- Two Volt instances on same network discover each other
- Shared fact (γ > 0.95) propagates from instance A to instance B
- Module published by A → discoverable and installable by B
- Settlement: A uses B's module → usage event logged → settlement ledger shows credit

Checkpoint: At the end of Phase 6, Volt is an open platform. Community members can build modules, share knowledge, trade strands, and participate in a decentralized intelligence economy.

Duration: 4 weeks.

---

## Anti-Patterns to Avoid

These are the specific traps that kill ambitious projects. Each one is addressed by the milestone structure:

```echarts
{
  "backgroundColor": "#0f172a",
  "title": {
    "text": "Development Anti-Patterns and How the Roadmap Avoids Them",
    "left": "center",
    "textStyle": { "color": "#e2e8f0", "fontSize": 14 }
  },
  "tooltip": { "trigger": "item", "backgroundColor": "#1e293b", "textStyle": { "color": "#e2e8f0" } },
  "xAxis": { "show": false },
  "yAxis": {
    "type": "category",
    "data": [
      "Premature\nOptimization",
      "Big Bang\nIntegration",
      "GPU Before\nCPU Debug",
      "Training Before\nInference",
      "Ecosystem Before\nCore",
      "Network Before\nLocal"
    ],
    "axisLabel": { "color": "#94a3b8", "fontSize": 10 },
    "axisLine": { "lineStyle": { "color": "#334155" } }
  },
  "series": [
    {
      "name": "Risk if ignored",
      "type": "bar",
      "data": [8, 10, 9, 7, 8, 6],
      "itemStyle": { "color": "#ef4444", "borderRadius": [0, 4, 4, 0] },
      "label": { "show": true, "position": "right", "color": "#fca5a5", "fontSize": 9, "formatter": ["Stub first, optimize later (M1.3→M2.4)", "Working system at every milestone", "RAR on CPU first (M2.3), GPU second (M2.4)", "End-to-end stub pipeline first (M1.3)", "Core works solo before opening ecosystem (M6.1)", "Local-first always. P2P last (M6.4)"] }
    }
  ]
}
```

Premature optimization: Milestone 1.3 builds a stub translator (dumb heuristics). This is intentional. You need the pipeline working before you need it working fast. The real translator arrives in M2.2.

Big bang integration: Every milestone produces a testable system. At no point do you have "I need to finish 3 more components before I can test anything."

GPU before CPU debug: Milestone 2.3 implements RAR on CPU first. Only after the algorithm is correct do you port to GPU in M2.4. GPU debugging adds 10× the development time. Never debug algorithms and CUDA simultaneously.

Training before inference: The inference pipeline works end-to-end (with stubs) before any training begins. This means you can test every training improvement against the real inference pipeline immediately.

Ecosystem before core: Community modules arrive in Phase 6. The core is complete and stable by Phase 5. You never build an ecosystem on a shifting foundation.

Network before local: The ledger and P2P are the last things built (M6.3-6.4). Volt works fully offline before any network features exist. Network problems never block local development.

---

## Dependency Graph

The milestone ordering ensures that dependencies always flow forward, never backward:

```svg
<svg viewBox="0 0 1000 500" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="depBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f"/>
      <stop offset="100%" style="stop-color:#0a0a1a"/>
    </linearGradient>
    <marker id="depA" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#475569"/></marker>
  </defs>
  <rect width="1000" height="500" fill="url(#depBg)" rx="12"/>
  <text x="500" y="28" text-anchor="middle" fill="#e2e8f0" font-size="14" font-weight="bold">Dependency Graph: Each milestone depends only on completed ones</text>

  <!-- M0 -->
  <rect x="420" y="45" width="160" height="30" rx="6" fill="#1e293b" stroke="#475569" stroke-width="1.5"/>
  <text x="500" y="65" text-anchor="middle" fill="#94a3b8" font-size="9" font-weight="bold">M0: Test Harness</text>

  <!-- Phase 1 -->
  <rect x="200" y="95" width="130" height="30" rx="6" fill="#1a0050" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="265" y="115" text-anchor="middle" fill="#c4b5fd" font-size="8" font-weight="bold">M1.1: TensorFrame</text>

  <rect x="370" y="95" width="130" height="30" rx="6" fill="#1a0050" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="435" y="115" text-anchor="middle" fill="#c4b5fd" font-size="8" font-weight="bold">M1.2: LLL Algebra</text>

  <rect x="540" y="95" width="140" height="30" rx="6" fill="#1a0050" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="610" y="115" text-anchor="middle" fill="#c4b5fd" font-size="8" font-weight="bold">M1.3: Stub + Server</text>

  <rect x="720" y="95" width="120" height="30" rx="6" fill="#1a0050" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="780" y="115" text-anchor="middle" fill="#c4b5fd" font-size="8" font-weight="bold">M1.4: n8n</text>

  <line x1="500" y1="75" x2="265" y2="95" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="500" y1="75" x2="435" y2="95" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="330" y1="110" x2="540" y2="110" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="500" y1="110" x2="540" y2="110" stroke="#475569" stroke-width="1"/>
  <line x1="680" y1="110" x2="720" y2="110" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>

  <!-- Phase 2 -->
  <rect x="140" y="165" width="130" height="30" rx="6" fill="#083344" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="205" y="185" text-anchor="middle" fill="#67e8f9" font-size="8" font-weight="bold">M2.1: Codebook</text>

  <rect x="310" y="165" width="150" height="30" rx="6" fill="#083344" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="385" y="185" text-anchor="middle" fill="#67e8f9" font-size="8" font-weight="bold">M2.2: Real Translator</text>

  <rect x="500" y="165" width="140" height="30" rx="6" fill="#083344" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="570" y="185" text-anchor="middle" fill="#67e8f9" font-size="8" font-weight="bold">M2.3: RAR on CPU</text>

  <rect x="680" y="165" width="160" height="30" rx="6" fill="#083344" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="760" y="185" text-anchor="middle" fill="#67e8f9" font-size="8" font-weight="bold">M2.4: GPU + VFN Train</text>

  <line x1="435" y1="125" x2="205" y2="165" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="270" y1="180" x2="310" y2="180" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="610" y1="125" x2="385" y2="165" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="460" y1="180" x2="500" y2="180" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="640" y1="180" x2="680" y2="180" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>

  <!-- Phase 3 -->
  <rect x="200" y="235" width="160" height="30" rx="6" fill="#1a1000" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="280" y="255" text-anchor="middle" fill="#fcd34d" font-size="8" font-weight="bold">M3.1: Router + Math</text>

  <rect x="400" y="235" width="170" height="30" rx="6" fill="#1a1000" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="485" y="255" text-anchor="middle" fill="#fcd34d" font-size="8" font-weight="bold">M3.2: More Strands</text>

  <rect x="610" y="235" width="150" height="30" rx="6" fill="#1a1000" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="685" y="255" text-anchor="middle" fill="#fcd34d" font-size="8" font-weight="bold">M3.3: Safety Layer</text>

  <line x1="760" y1="195" x2="280" y2="235" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="360" y1="250" x2="400" y2="250" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="570" y1="250" x2="610" y2="250" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>

  <!-- Phase 4 -->
  <rect x="150" y="305" width="170" height="30" rx="6" fill="#1a0020" stroke="#ec4899" stroke-width="1.5"/>
  <text x="235" y="325" text-anchor="middle" fill="#f9a8d4" font-size="8" font-weight="bold">M4.1: VoltDB T0+T1</text>

  <rect x="360" y="305" width="180" height="30" rx="6" fill="#1a0020" stroke="#ec4899" stroke-width="1.5"/>
  <text x="450" y="325" text-anchor="middle" fill="#f9a8d4" font-size="8" font-weight="bold">M4.2: HNSW + Ghost</text>

  <rect x="580" y="305" width="190" height="30" rx="6" fill="#1a0020" stroke="#ec4899" stroke-width="1.5"/>
  <text x="675" y="325" text-anchor="middle" fill="#f9a8d4" font-size="8" font-weight="bold">M4.3: T2 + GC + MVCC</text>

  <line x1="685" y1="265" x2="235" y2="305" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="320" y1="320" x2="360" y2="320" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="540" y1="320" x2="580" y2="320" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>

  <!-- Phase 5 -->
  <rect x="150" y="375" width="170" height="30" rx="6" fill="#001a10" stroke="#10b981" stroke-width="1.5"/>
  <text x="235" y="395" text-anchor="middle" fill="#34d399" font-size="8" font-weight="bold">M5.1: Event Logging</text>

  <rect x="360" y="375" width="180" height="30" rx="6" fill="#001a10" stroke="#10b981" stroke-width="1.5"/>
  <text x="450" y="395" text-anchor="middle" fill="#34d399" font-size="8" font-weight="bold">M5.2: Sleep Consol.</text>

  <rect x="580" y="375" width="180" height="30" rx="6" fill="#001a10" stroke="#10b981" stroke-width="1.5"/>
  <text x="670" y="395" text-anchor="middle" fill="#34d399" font-size="8" font-weight="bold">M5.3: RLVF Align</text>

  <line x1="675" y1="335" x2="235" y2="375" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="320" y1="390" x2="360" y2="390" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="540" y1="390" x2="580" y2="390" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>

  <!-- Phase 6 -->
  <rect x="100" y="445" width="160" height="30" rx="6" fill="#1a0a00" stroke="#f97316" stroke-width="1.5"/>
  <text x="180" y="465" text-anchor="middle" fill="#fb923c" font-size="8" font-weight="bold">M6.1: Traits + Plug</text>

  <rect x="300" y="445" width="160" height="30" rx="6" fill="#1a0a00" stroke="#f97316" stroke-width="1.5"/>
  <text x="380" y="465" text-anchor="middle" fill="#fb923c" font-size="8" font-weight="bold">M6.2: Vision + TTS</text>

  <rect x="500" y="445" width="160" height="30" rx="6" fill="#1a0a00" stroke="#f97316" stroke-width="1.5"/>
  <text x="580" y="465" text-anchor="middle" fill="#fb923c" font-size="8" font-weight="bold">M6.3: Ledger L0</text>

  <rect x="700" y="445" width="170" height="30" rx="6" fill="#1a0a00" stroke="#f97316" stroke-width="1.5"/>
  <text x="785" y="465" text-anchor="middle" fill="#fb923c" font-size="8" font-weight="bold">M6.4: P2P + Settle</text>

  <line x1="670" y1="405" x2="180" y2="445" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="260" y1="460" x2="300" y2="460" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="460" y1="460" x2="500" y2="460" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
  <line x1="660" y1="460" x2="700" y2="460" stroke="#475569" stroke-width="1" marker-end="url(#depA)"/>
</svg>
```

Every arrow points downward and rightward. No milestone depends on anything that hasn't been built yet. Every milestone can be tested against the already-working system from the previous milestone.

---

## Final Note: The One-Person Development Reality

You're building this solo, at least initially. The roadmap is designed for that reality:

Weeks 1-8 (Phase 1): Pure Rust engineering. No ML. No GPU. You're building data structures, a web server, and piping them together. This is fast, testable, and you can see progress daily.

Weeks 9-18 (Phase 2): First ML work. Training the translator and VFN. This is slower (training takes days), but each training run tests against the already-working pipeline.

Weeks 19-25 (Phase 3): Back to pure Rust engineering. Building tools. The most satisfying phase — you're adding visible capabilities rapidly.

Weeks 26-34 (Phase 4): Database engineering. Different skill set. But every feature is immediately testable by asking the system "do you remember what we talked about yesterday?"

Weeks 35-42 (Phase 5): The magical phase where the system starts improving itself. You can measure improvement week over week.

Weeks 43-52 (Phase 6): Opening to the world. Writing docs, building examples, launching the network.

At month 6, you have a fully functional personal AI that thinks, verifies, uses tools, remembers everything, and runs on your laptop. Everything after month 6 is scaling, ecosystem, and polish.

That's the target. Month 6. A working Volt X on your machine, that remembers you, that gets smarter every night, that can't be taken away from you.

The dangerous game begins.