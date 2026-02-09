# Phase 5: Learning (Months 9-10)

> Extracted from the master roadmap. Depends on: Phase 4 completed.

## Goal
Activate the self-improvement loop. The system should get measurably better over time through use.

---

## Milestone 5.1: Learning Event Logging (Week 35-36)

**Crate:** `volt-learn`

### What You Build
- Learning event struct: `{ frame_id, strand_id, query_type, gamma_scores, convergence_iterations, ghost_activations, timestamp }`
- Every inference run automatically logs a learning event to strand metadata
- Learning event buffer: accumulates events, flushable to disk
- Basic statistics: per-strand query count, average gamma, average iteration count, topic distribution

### What You Test
- After 100 queries, learning buffer has 100 events
- Statistics reflect actual usage patterns (more coding queries -> coding strand dominates)
- Events survive restart (persisted alongside strand in VoltDB)

**Duration:** 2 weeks.

---

## Milestone 5.2: Sleep Consolidation (Week 37-39)

**Crate:** `volt-learn` (extend)

### What You Build
- Sleep scheduler: triggers consolidation when system idle > 10 minutes OR on manual command
- Frame distillation: clusters of related frames within a strand -> averaged into wisdom frames
- Forward-Forward VFN update:
  - Positive data: high-gamma verified frames (goodness target: high)
  - Negative data: low-gamma rejected frames, corrupted frames (goodness target: low)
  - Update one VFN layer at a time, discard activations between layers
  - VRAM usage: approximately 1x inference
- Strand graduation: when Mirror Module detects a cluster of > 50 frames about an unrecognized topic -> promote to new strand

### What You Test
- After sleep consolidation: VFN produces measurably lower loss on validation set (even if small improvement)
- Distillation: 50 raw frames about Rust lifetimes -> 3-5 wisdom frames that retain key information
- Strand graduation: after 50+ cooking-related queries -> "Cooking" strand automatically created
- VRAM during sleep: does not exceed 1.5x inference VRAM
- System remains responsive during consolidation (consolidation runs on background threads)

**Duration:** 3 weeks.

---

## Milestone 5.3: RLVF Joint Alignment (Week 40-42)

**Crate:** `volt-learn` (extend)

### What You Build
- Evaluation dataset: 1000 (question, verified_answer) pairs across math, logic, factual, creative
- RLVF loop:
  1. Run full Volt pipeline on question -> output frame with gamma
  2. Compare to verified answer (cosine similarity of frame slots)
  3. Compute reward: correct + calibrated gamma -> positive; overconfident error -> strong negative; honest uncertainty -> mild
  4. Update VFN via policy gradient (REINFORCE with baseline)
- Certainty calibration metric: "when Volt says gamma=0.9, is it correct 90% of the time?"
- Self-play for logic: automatically generate simple logic puzzles, run Volt, grade proof chains

### What You Test
- After RLVF: certainty calibration improves (measured by reliability diagram)
- Self-play: Volt solves > 80% of generated logic puzzles with valid proofs
- No safety regression: re-run adversarial safety tests -> all still pass
- Overall quality: human evaluation of 50 random queries shows improvement vs pre-RLVF

**Duration:** 3 weeks.

---

## Phase 5 Checkpoint

At the end of Phase 5, Volt improves itself. Every conversation makes it better. Sleep consolidation refines its understanding. RLVF calibrates its confidence. It measurably gets smarter over weeks of use.
