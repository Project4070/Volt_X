# Phase 3: The Hard Core (Months 5-6)

> Extracted from the master roadmap. Depends on: Phase 2 completed.

## Goal
Add deterministic CPU tools and verification. The system should now be able to do math exactly, execute code, verify its own outputs, and produce proof chains.

---

## Milestone 3.1: Intent Router + First Hard Strand (Week 19-20) -- DONE

**Crate:** `volt-hard`

### What You Build

- [x] Intent Router: receives TensorFrame from Soft Core, computes cosine similarity against registered Hard Strand capability vectors, routes to best match
- [x] MathEngine Hard Strand: implements `HardStrand` trait, handles arithmetic, algebra, basic calculus
- [x] Integration: Soft Core -> Intent Router -> MathEngine -> result injected back into frame

### What You Test

- [x] "What is 847 x 392?" -> MathEngine activates -> exact answer 332,024 -> gamma = 1.0
- [x] "Tell me about cats" -> no Hard Strand activates -> passes through Soft Core only
- [x] Router correctly distinguishes math queries from non-math queries (>95% accuracy on 100 test cases)
- [x] MathEngine returns in < 1ms

**Duration:** 2 weeks.

---

## Milestone 3.2: More Hard Strands (Week 21-23)

**Crate:** `volt-hard` (extend)

### What You Build
- **CodeRunner:** sandboxed code execution via `wasmtime`. Takes code from frame, runs it, returns stdout/stderr.
- **HDCAlgebra:** exposes bind/unbind/superpose as a callable Hard Strand for compositional reasoning
- **CertaintyEngine:** min-rule propagation across frame slots. Computes frame-level gamma.
- **ProofConstructor:** records which Hard Strands were called, what they returned, builds proof chain.

### What You Test
- CodeRunner: `print(2+2)` -> output "4" in sandboxed environment. Malicious code (file access, network) -> blocked.
- CertaintyEngine: frame with gamma=[1.0, 0.8, 0.6] -> global gamma = 0.6
- ProofConstructor: after processing, proof chain has >= 2 steps, each with source and gamma

**Duration:** 3 weeks.

---

## Milestone 3.3: Safety Layer (Week 24-25)

**Crate:** `volt-safety`

### What You Build
- Axiomatic Guard: 5 hardcoded invariants (K1-K5) as constant vectors
- Transition Monitor: checks every frame transition against invariants (inner product)
- Violation Scorer: computes violation score, triggers warning or halt
- Omega Veto: when triggered, freezes all processing, returns safe default, logs state
- Integration: Safety layer wraps the entire Soft Core -> Hard Core pipeline

### What You Test
- Normal query -> safety layer passes through, no interference
- Query touching K1 (harm) -> violation detected -> Omega Veto fires -> safe default response
- Omega Veto logs include full frame state at time of trigger
- Safety layer adds < 1ms latency to normal queries
- Cannot bypass safety by crafting special frame structures (adversarial testing)

**Duration:** 2 weeks.

---

## Phase 3 Checkpoint

At the end of Phase 3, you have a verified AI. It thinks (GPU), acts (CPU tools), and verifies (certainty + proofs + safety). The outputs now include gamma scores and proof chains. Math is exact. Code execution is sandboxed. Safety cannot be bypassed.
