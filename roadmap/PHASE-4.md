# Phase 4: Memory (Months 7-8)

> Extracted from the master roadmap. Depends on: Phase 3 completed.

## Goal
Add persistent memory so the system remembers across conversations. This is where Volt stops being a chatbot and becomes a personal AI.

---

## Milestone 4.1: VoltDB Tier 0 + Tier 1 (Week 26-28) **[DONE]**

**Crate:** `volt-db`

### What You Build
- Tier 0 (Working Memory): in-memory ring buffer of 64 TensorFrames. LRU eviction.
- Tier 1 (Short-Term Memory): strand-organized HashMap<StrandId, Vec<TensorFrame>>. Frames persist in RAM across queries.
- Basic strand management: create strand, switch strand, list strands
- Frame storage: every processed frame auto-stores in T0, evicts to T1 when T0 fills
- Basic retrieval: get frame by ID, get frames by strand, get most recent N frames

### What You Test
- Ask 100 questions -> all 100 frames stored -> retrievable by ID
- T0 fills at 64 -> oldest frame moves to T1 -> still retrievable from T1
- Switch strand -> new queries go to new strand -> old strand intact
- Restart server -> T1 persists (serialize to disk on shutdown, reload on start)
- Retrieval by ID: < 0.1ms

### What You Explicitly Do NOT Build Yet
- No HNSW index (linear scan is fine for thousands of frames)
- No Tier 2 compression
- No ghost frames or bleed
- No GC

**Duration:** 3 weeks.

---

## Milestone 4.2: HNSW Indexing + Ghost Bleed (Week 29-31) **[DONE]**

**Crate:** `volt-db` (extend)

### What You Build
- HNSW index over all frame R0 gists within each strand
- Semantic retrieval: `query_similar(frame_gist, k=10) -> top-10 similar frames`
- B-tree temporal index: `query_range(start_time, end_time) -> frames in range`
- Ghost Bleed Buffer: array of ~1000 R0 gists loaded into a buffer accessible by Soft Core
- Bleed Engine: on every new frame, query HNSW -> update ghost buffer with top relevant gists
- Integration with RAR Attend phase: ghost frames participate as additional Key/Value in cross-attention

### What You Test
- Ask about Rust -> get back relevant frames from Coding strand (not Cooking strand)
- Ask about something from 2 weeks ago -> HNSW finds it -> ghost appears -> full frame loads on page fault
- Ghost buffer refreshes when topic changes (new frame gist differs from previous)
- Semantic retrieval accuracy: > 80% of top-10 results are genuinely relevant (manual evaluation on 50 queries)

**Duration:** 3 weeks.

---

## Milestone 4.3: Tier 2 + GC + Consolidation (Week 32-34)

**Crate:** `volt-db` (extend)

### What You Build
- Tier 2 (Long-Term Memory): compressed frame storage (R0 only, 1 KB per frame) on disk via `mmap`
- LSM-Tree structure: memtable -> flush to sorted run -> periodic compaction
- GC pipeline: retention score computation -> 4-tier decay (full -> compressed -> gist -> tombstone)
- Bloom filters on LSM runs for fast negative checks
- MVCC: `crossbeam-epoch` for lock-free readers, per-strand Mutex for writers
- WAL: append-only log per strand for crash recovery
- Consolidation: batch of similar frames -> distilled summary frame (simple averaging for now, no FF yet)

### What You Test
- Store 1 million frames -> retrieval still < 5ms via HNSW + B-tree
- GC correctly tombstones old, low-gamma, unreferenced frames
- Immortal frames (high gamma, user-pinned) survive GC indefinitely
- Crash test: kill process mid-write -> restart -> no data loss beyond current frame
- Concurrent read/write: 10 reader threads + 1 writer thread -> no deadlocks, no corruption
- Memory usage grows sublinearly: 1M frames < 2 GB RAM + 1 GB disk

**Duration:** 3 weeks.

---

## Phase 4 Checkpoint

At the end of Phase 4, Volt remembers everything. Conversations from months ago are retrievable. Ghost frames subtly influence current reasoning. The system feels like it knows you.
