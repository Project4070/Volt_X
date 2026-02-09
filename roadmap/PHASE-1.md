# Phase 1: The Skeleton (Months 1-2)

> Extracted from the master roadmap. Depends on: Milestone 0 completed.

## Goal
A dumb system that takes text in and text out, with TensorFrames flowing through the entire pipeline. The Soft Core is a stub (copies input to output). The Hard Core is a stub (passes through). But the Frame Bus works, the data structures work, the server works, and n8n can talk to it.

---

## Milestone 1.1: TensorFrame Data Structure (Week 1-2)

**Crate:** `volt-core`

### What You Build
- `TensorFrame` struct with 16 slots, 4 resolutions, 256 dims
- `SlotData`, `SlotMeta`, `FrameMeta` structs
- `SlotRole` enum
- Serialization/deserialization via `serde` and `rkyv` (zero-copy)
- Frame creation, cloning, slot read/write, merge operations
- Unit normalization (per-slot, per-resolution)

### What You Test
- Create a frame, write to slot 3 at R1, read it back -> identical
- Merge two frames -> slots from both present, conflicts resolved by gamma
- Serialize to bytes, deserialize -> bit-identical
- `rkyv` zero-copy: archived bytes are usable without parsing
- Frame size: empty frame < 100 bytes, full frame = 64 KB exactly

### What You Explicitly Do NOT Build Yet
- No GPU code. Frames are CPU-only for now.
- No codebook. Slots hold raw float arrays.
- No HNSW index.

### Why This Order
Everything else depends on TensorFrame. If this data structure is wrong, everything built on top is wrong. Get it right first.

**Duration:** 2 weeks.

**Status:** ✅ **COMPLETED** (Commit e5cc1fa)

- All required features implemented: serialization (rkyv + serde), merge operations, normalization
- 65 tests passing (25 unit + 11 integration + 3 serialization + 26 doc tests)
- Zero clippy warnings
- Performance targets exceeded (merge ~23µs, normalize ~15µs)

---

## Milestone 1.2: LLL Algebra (Week 3-4)

**Crate:** `volt-bus`

### What You Build
- FFT-based binding: `bind(a, b) -> c` using `rustfft` crate
- Unbinding via involution: `unbind(c, a) -> b_approx`
- Superposition: `superpose(vec![a, b, c]) -> normalized_sum`
- Permutation: `permute(a, k) -> cyclic_shift_by_k`
- Cosine similarity: `sim(a, b) -> f32`
- Operations work on individual slot vectors (256 dims)
- Batch operations: apply bind/unbind to entire frame (all slots)

### What You Test
- `unbind(bind(a, b), a)` ~ `b` with cosine similarity > 0.85
- `sim(superpose([a, b]), a)` > 0 (constituent detectable)
- `sim(bind(a, b), a)` ~ 0 (bound pair is dissimilar to inputs)
- `sim(permute(a, 1), permute(a, 2))` ~ 0 (different shifts are orthogonal)
- Performance: bind on 256 dims < 10us

### Why This Order
The algebra is the grammar of thought. It must be correct before the Soft Core tries to use it.

**Duration:** 2 weeks.

---

## Milestone 1.3: Stub Translator + HTTP Server (Week 5-6)

**Crates:** `volt-translate` + `volt-server`

### What You Build
- **Stub Forward Translator:** Takes raw text, splits into words, assigns first word to S0 (AGENT), verb to S1 (PREDICATE), object to S2 (PATIENT). No ML. Just heuristic word-to-slot mapping. Each word is encoded as a random but deterministic vector (hash of word -> seed -> random 256-dim vector, normalized).
- **Stub Reverse Translator:** Takes TensorFrame, reads slot metadata, produces template: "Agent [S0] does [S1] to [S2]."
- **Axum HTTP Server:**
  - `POST /api/think` -> accepts JSON `{ "text": "..." }`, returns JSON `{ "text": "...", "gamma": [...], "strand_id": 0, "iterations": 1 }`
  - Health endpoint: `GET /health`

### What You Test
- Send "The cat sat on the mat" -> get back a response with 3 filled slots
- Round-trip: encode -> decode -> readable text (even if clumsy)
- Server starts, responds to curl, handles 100 concurrent requests without crash
- Invalid input (empty string, huge input, binary garbage) -> graceful error, not panic

### What You Explicitly Do NOT Build Yet
- No LLM backbone. The translator is a dumb heuristic.
- No GPU. Everything runs on CPU.

### Why This Order
You now have a testable end-to-end system. Text goes in, TensorFrame flows through the pipeline, text comes out. Every subsequent milestone improves one component of this working pipeline.

**Duration:** 2 weeks.

---

## Milestone 1.4: n8n Integration (Week 7-8)

**Crate:** `volt-server` (extend)

### What You Build
- n8n workflow: Chat Trigger -> HTTP Request to `localhost:8080/api/think` -> Switch (text/tool/error) -> Chat Reply
- Debug output: the JSON response includes `slot_states`, `timing_ms`, `strand_id`
- n8n displays these in a "Debug Panel" node (Set node formatting)

### What You Test
- Open n8n, type in chat, get response
- Response includes slot breakdown visible in n8n execution log
- Handle errors gracefully (server down -> n8n shows error, not crash)

**Duration:** 2 weeks.

---

## Phase 1 Checkpoint

At the end of Phase 1, you have a working chat system. It's stupid (heuristic translator, no reasoning), but the entire pipeline works:

**n8n -> HTTP -> Translate -> TensorFrame -> Bus -> Stub Process -> Translate Back -> HTTP -> n8n**

Every subsequent phase improves one part of this pipeline without breaking the others.
