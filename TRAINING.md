# Volt X — Unified Training Plan

**Version:** 3.0 (spiral curriculum revision — replaces sequential F1/F2/F3
approach from v2.0)

**Date:** 2026-02-15

**Status:** Phase 0 infrastructure complete. Training paradigm revised to
spiral curriculum — all four training signals from day one, applied to
progressively more complex content across six developmental stages.

---

## Why This Rewrite Exists

The previous training plans drifted from Volt's architectural philosophy
through incremental improvisation:

1. **Dataset selection was wrong.** Bulk-downloading The Stack (44 GB of raw
   Python files) was the wrong starting point. Raw source files require
   post-processing and don't come with the structured annotations Volt needs.
   Curated, pre-formatted datasets on HuggingFace are better, and streaming
   eliminates the need for full downloads.

2. **HuggingFace streaming was ignored.** The old pipeline required downloading
   entire datasets to disk before training. HF's `datasets` library supports
   `streaming=True`, which iterates over rows directly from the Hub with no
   local storage. For large corpora this is essential.

3. **The training paradigm was drifting toward next-token prediction.** The
   Phase 1 decoder is autoregressive (teacher-forced cross-entropy on BPE
   tokens). The VFN training objective (flow match from query frame to answer
   frame) is functionally equivalent to "predict the right output embedding
   given an input embedding" — which is NTP in disguise with a continuous
   relaxation. This defeats the purpose of Volt's architecture.

4. **Code came before language.** The old plan trained on CodeSearchNet first,
   then hoped to generalize to natural language later. This is backwards. A
   system that doesn't know what "weather" means cannot write a weather app.
   Language understanding must come first; code is a domain specialization
   on top of it.

5. **Plans were fragmented and contradictory.** Seven separate documents
   each told a different story. PATH_TO_AGI suggested "train RAR on
   next-word prediction" — directly contradicting Volt's design.

6. **Sequential phases starved the VFN.** In the v2.0 plan, VFN training
   didn't begin until Phase F3 (week 6+). The encoder trained alone in
   F1 and F2, then the VFN had to learn everything from scratch. A spiral
   approach where the VFN trains from day one gives it 5x more practice
   by the time it reaches complex reasoning tasks.

This document unifies everything into one plan that stays true to Volt's
architectural principles and follows a developmental learning trajectory.

---

## Part I: How Volt Learns (Not NTP)

### What Transformers Do

Transformers learn by next-token prediction: given a sequence of tokens,
predict the next one. All knowledge — word meaning, world knowledge, reasoning
ability, code syntax — emerges implicitly from this single objective applied
to trillions of tokens.

### What Volt Does Instead

Volt has **structure that transformers don't**: 16 semantic slots, 4
resolutions, HDC algebra, and an iterative refinement loop (RAR). The training
must exploit this structure, not ignore it.

**Four training signals replace NTP:**

#### 1. Slot Filling (Structural Masked Modeling)

Present a frame with some slots populated, others empty or corrupted. Train the
VFN to refine the frame until missing slots converge to correct values.

```
Input:   Agent=chef, Predicate=???, Patient=pasta, Location=kitchen
Target:  Agent=chef, Predicate=cook, Patient=pasta, Location=kitchen
```

This directly mirrors what RAR does at inference time. Training IS practicing
inference.

#### 2. Compositional Binding (HDC Algebra Learning)

Train on (A, relation, B) triples using HDC operations:

```
bind(weather, UsedFor)  ≈  planning         (in HDC space)
bind(chef, performs)    ≈  cooking           (in HDC space)
unbind(result, relation) ≈ original_concept  (retrieval)
```

This teaches the HDC algebra to carry actual semantic content.

#### 3. Frame Discrimination (Energy-Based / Contrastive)

Train the VFN's energy function so that coherent frames have low energy and
incoherent frames have high energy:

```
Low energy:   "chef cooks pasta in kitchen"    (real frame)
High energy:  "chef cooks pasta in algebra"    (corrupted frame)
```

RAR then performs gradient descent on this energy landscape to find coherent
frame states.

#### 4. Multi-Resolution Consistency (Self-Supervised)

R0 (discourse gist) must be consistent with R1 (proposition), R2 (phrase),
R3 (token). If R0 says "cooking topic" and R1 says "algebraic theorem," that's
a contradiction the VFN should learn to resolve. This requires no labels.

### The Decoder Exception

The decoder (frame → text) CAN stay autoregressive. It's a rendering step —
it takes a completed frame and produces human-readable text. This is analogous
to a GPU rasterizer: the 3D scene (frame) is computed via non-sequential
methods, but the final pixel output is sequential. The decoder's NTP is
contained and doesn't infect the upstream pipeline.

### The Spiral Principle

**All four signals are present at every developmental stage.** What changes
is the complexity:

| Signal | Stage 1 (Naming) | Stage 5 (Hypotheticals) |
|---|---|---|
| Slot Filling | Mask 1 of 2 slots | Mask 8 of 16 slots |
| Binding | `dog IsA animal` | `unbind(bind(cause, entails), entails) ≈ cause` |
| Discrimination | 2-slot coherence check | 16-slot narrative coherence |
| Multi-Resolution | R0 gist vs R1 (2 slots) | All 4 resolutions, 16 slots |

This is a Montessori/Piaget-style curriculum — interleaved and developmental —
not isolated semester courses where skills atrophy between phases.

---

## Part II: The Spiral Curriculum

### Why Developmental, Not Sequential

The v2.0 plan treated training as a sequence of isolated phases: first train
lexical grounding (F1), then world knowledge (F2), then reasoning (F3), then
code (D1). Each phase trained different components with different objectives.

This has three problems:

1. **The VFN starves.** VFN training doesn't start until Phase F3 (week 6+).
   By then the encoder has learned embeddings that the VFN has never seen.
   The VFN must learn slot filling, denoising, and reasoning all at once on
   unfamiliar representations.

2. **Catastrophic forgetting at phase boundaries.** When F2 starts, F1's
   lexical grounding can degrade. When F3 starts, F2's relational bindings
   can drift. Phase boundaries create discontinuities.

3. **Wasted compute.** In F1, the encoder trains on slot assignment while the
   VFN sits idle. In F2, HDC trains on relations while the encoder drifts.
   Every component should train on every batch.

### The Spiral Alternative

A spiral curriculum applies all four training signals from day one, but
controls complexity via three levers:

| Lever | Range | What It Controls |
|---|---|---|
| **Active slots** | 2 → 16 | How many slots the system uses |
| **VFN iterations** | 1 → 30 | How many refinement steps the VFN takes |
| **HDC operations** | superpose only → full algebra | Which binding operations are available |

The system starts as an "infant" — naming things with 2 slots and 1
refinement step — and gradually matures to a "teenager" solving multi-hop
reasoning with 16 slots and 30 iterations. Code is the final specialization,
learned by a system that already understands the world.

### Metric-Gated Advancement

The curriculum scheduler advances to the next stage only when specific
accuracy targets are met. This prevents the "trained but didn't learn"
failure mode:

```
if stage.slot_filling_accuracy >= gate_threshold
   && stage.binding_retrieval_accuracy >= gate_threshold
   && stage.discrimination_auc >= gate_threshold:
       advance_to_next_stage()
else:
       continue_training_current_stage()
```

### Continuous Rehearsal

To prevent catastrophic forgetting, **20% of every training batch** consists
of material from earlier stages. A Stage 4 batch contains:

- 80% Stage 4 material (narrative, multi-hop, 12 active slots)
- 10% Stage 2-3 material (SVO sentences, contextual frames)
- 10% Stage 1 material (concrete nouns, 2-slot frames)

This is cheaper than it sounds — earlier stages use fewer active slots, so
forward/backward passes on rehearsal examples are faster.

### How This Maps to Existing Code

| Existing Component | Role in Spiral Approach |
|---|---|
| CNN Encoder (5.1M params) | Same architecture. Role Head starts predicting 2 slots, gradually expands to 16 |
| ScaledVfn (51M params) | Same architecture. `max_iterations` starts at 1, grows to 30 |
| SlotAttention | Same, but only attends over active slots (inactive slots masked) |
| HDC bind/unbind/superpose | Same ops, introduced progressively: superpose → bind → unbind → permute |
| Codebook | Initialized after Stage 4 when embeddings are meaningful |
| Decoder | Trained alongside encoder from Stage 1, but only decodes active slots |

### What This Means for Existing Artifacts

| Existing Artifact | Status | Role in Spiral Plan |
|---|---|---|
| VFN Checkpoint System (Phase 0.1) | Reusable as-is | Infrastructure |
| Code Dataset Pipeline (Phase 0.2) | Reusable, extend for new formats | Infrastructure |
| Code Attention Bias (Phase 0.4) | Reusable for Stage 6 | Applied during code specialization |
| BPE Tokenizer (32K vocab) | Reusable as-is | Shared across all stages |
| CNN Encoder (5.1M params) | Architecture reusable | Retrain from scratch with progressive slot expansion |
| Autoregressive Decoder (6.7M) | Architecture reusable | Trained alongside encoder, decodes active slots only |
| Codebook Init Pipeline | Reusable | Run after Stage 4 when embeddings are meaningful |
| ScaledVfn (51M params) | Architecture reusable | Trains from Stage 1 (1 iteration) to Stage 5 (30 iterations) |
| training_config.toml | Needs update | Add language dataset paths, stage configs |

---

## Part III: The Six Developmental Stages

### Stage 1: Naming (The Infant)

**What Volt learns:** Concrete nouns exist and belong in semantic slots.
Basic category membership. "Dog" and "canine" mean the same thing.

**Active slots:** 2 (S0: Agent, S2: Patient)
**VFN iterations:** 1
**HDC operations:** superpose only (category formation)
**Duration:** 1-2 weeks
**Hardware:** 1x RTX 4090, ~5 GPU-hours

#### Training Signals at This Stage

**Slot Filling:** Mask 1 of 2 slots, predict it. Given `Agent=dog, Patient=???`
with context "the dog chased the cat," predict `Patient=cat`. This is trivial —
and that's the point. The VFN starts with guaranteed success.

**Binding:** Simple IsA relations via superposition.
`superpose(dog, cat, fish) ≈ animal`. No bind/unbind yet — just category
formation from exemplars.

**Discrimination:** 2-slot coherence. "Dog chases cat" (low energy) vs
"dog chases algebra" (high energy). The VFN learns that Agent and Patient
should be semantically compatible.

**Multi-Resolution:** R0 gist should indicate "animal interaction" when
R1 contains dog and cat. Only 2-slot consistency required.

#### Datasets

| Dataset | What It Provides | Subset Used |
|---|---|---|
| FrameNet 1.7 | Concrete frames with Agent/Patient annotations | Simple transitive frames only |
| WordNet (via NLTK) | Hypernym chains (dog → canine → animal) | Noun synsets, depth ≤ 3 |
| STS Benchmark | Semantic similarity pairs | Concrete noun pairs only |

#### Slot Mapping

| FrameNet Element | TensorFrame Slot |
|---|---|
| Agent / Arg0 | S0 (Agent) |
| Patient / Arg1 | S2 (Patient) |
| All others | Masked / inactive |

#### Gate Criteria (advance to Stage 2 when met)

- Slot filling accuracy ≥ 70% (mask 1 of 2 slots)
- Superposition category retrieval ≥ 60% (nearest neighbor of `superpose(dog, cat)` = animal)
- Discrimination AUC ≥ 0.65

---

### Stage 2: Sentences (The Toddler)

**What Volt learns:** Actions exist. Subject-Verb-Object structure. Role
violations are detectable — "chef cooks pasta" is valid, "pasta cooks chef"
is wrong.

**Active slots:** 3-4 (S0: Agent, S1: Predicate, S2: Patient, optionally S3: Location)
**VFN iterations:** 1-3
**HDC operations:** superpose + bind
**Duration:** 2-3 weeks
**Hardware:** 1x RTX 4090, ~10 GPU-hours

#### Training Signals at This Stage

**Slot Filling:** Mask 1-2 of 3-4 slots. Given `Agent=chef, Predicate=???,
Patient=pasta`, predict `Predicate=cook`. The VFN now takes 1-3 iterations
and must learn when to stop refining.

**Binding:** Agent-Predicate and Predicate-Patient relations.
`bind(chef, performs) ≈ cooking`. This introduces the bind operation for
the first time — the system learns that relationships between concepts can
be computed algebraically.

**Discrimination:** Role violation detection. "Chef cooks pasta" has low
energy; "pasta cooks chef" has high energy. Same words, wrong roles — the
energy function must use slot identity, not just slot content.

**Multi-Resolution:** R0 gist should indicate "cooking event" when R1
contains chef/cook/pasta. R1 proposition structure (SVO) must be consistent
with R2 phrase-level embeddings.

#### Datasets

| Dataset | What It Provides | Subset Used |
|---|---|---|
| PropBank / CoNLL-2012 | Arg0/Arg1/V annotations | Simple SVO sentences (3-4 args max) |
| FrameNet 1.7 | Semantic frames with roles | Frames with ≤ 4 frame elements |
| PAWS | Paraphrase pairs (same words, different structure) | Subset with clear SVO structure |

#### Gate Criteria (advance to Stage 3 when met)

- Slot filling accuracy ≥ 75% (mask 1-2 of 3-4 slots)
- Bind retrieval accuracy ≥ 40% (`bind(A, rel)` nearest neighbor = B)
- Role violation discrimination AUC ≥ 0.70
- VFN converges within 3 iterations on >80% of examples

---

### Stage 3: Context (The Preschooler)

**What Volt learns:** Events have context — where, when, how, and why things
happen. 2-hop reasoning chains: "rain causes floods, floods cause damage"
→ "rain causes damage."

**Active slots:** 8 (S0-S7: Agent, Predicate, Patient, Location, Time, Manner, Instrument, Cause)
**VFN iterations:** 3-10
**HDC operations:** superpose + bind + unbind
**Duration:** 3-4 weeks
**Hardware:** 1x RTX 4090, ~20-30 GPU-hours

#### Training Signals at This Stage

**Slot Filling:** Mask 2-4 of 8 slots. Given a full event frame with Location
and Time removed, predict where and when. The VFN must now coordinate multiple
missing slots — if it knows "swimming" it can infer "pool" (Location) and
"summer" (Time).

**Binding:** Contextual relations via ConceptNet and ATOMIC triples.
`bind(rain, causes) ≈ flood`. Unbind is introduced: given `bind(rain, causes)`
and `causes`, retrieve `rain`. This teaches reversible algebraic reasoning.

**Discrimination:** 8-slot coherence. "Chef cooks pasta in kitchen at noon"
(low energy) vs "chef cooks pasta in algebra at theorem" (high energy). The
energy landscape becomes richer — more slots means more ways to be incoherent.

**Multi-Resolution:** All 4 resolutions active. R0 (discourse gist) must
be consistent with R1 (proposition), R2 (phrase), R3 (token) across 8
slots. Cross-resolution inference: R0 says "cooking" and R3 has token
"spatula" → R1 should include an Instrument slot.

#### Datasets

| Dataset | What It Provides | Subset Used |
|---|---|---|
| PropBank / CoNLL-2012 (full) | All argument types including ArgM-LOC, TMP, MNR, CAU | Full training set |
| ConceptNet 5 (EN) | Commonsense relations (UsedFor, IsA, HasProperty, CapableOf) | ~1.5M English triples |
| ATOMIC 2020 | Social/physical commonsense (xNeed, xEffect, oReact) | If-then tuples |
| STS Benchmark (full) | Semantic similarity across sentence types | Full dataset |
| bAbI QA (tasks 1) | Single-hop factual QA with supporting facts | Task 1 only as 2-hop warmup |

#### Slot Mapping (Full Linguistic)

| FrameNet Element / PropBank Arg | TensorFrame Slot |
|---|---|
| Agent / Arg0 | S0 (Agent) |
| Predicate / verb | S1 (Predicate) |
| Patient / Arg1 | S2 (Patient) |
| Location / ArgM-LOC | S3 (Location) |
| Time / ArgM-TMP | S4 (Time) |
| Manner / ArgM-MNR | S5 (Manner) |
| Instrument / Arg2 | S6 (Instrument) |
| Cause / ArgM-CAU | S7 (Cause) |

#### Gate Criteria (advance to Stage 4 when met)

- Slot filling accuracy ≥ 75% (mask 2-4 of 8 slots)
- Bind/unbind retrieval accuracy ≥ 50% on ConceptNet test split
- Discrimination AUC ≥ 0.75
- 2-hop reasoning accuracy ≥ 50% (A→B→C chains)
- VFN converges within 10 iterations on >80% of examples

---

### Stage 4: Narrative (The School-Age Child)

**What Volt learns:** Sequences of events form stories. Easy slots converge
first; dependent slots are derived. The VFN learns an "easy-first" strategy —
fill Agent and Predicate, then derive Result from them.

**Active slots:** 12 (S0-S11)
**VFN iterations:** 10-20
**HDC operations:** superpose + bind + unbind + permute
**Duration:** 4-5 weeks
**Hardware:** 1x H100 or 2x RTX 4090, ~60-100 GPU-hours

#### Training Signals at This Stage

**Slot Filling (Denoising):** Mask 4-6 of 12 slots including dependent ones.
The VFN must learn ordering — fill Cause before Result, fill Agent before
deriving what the agent is capable of. Per-slot convergence: frozen slots
stop consuming compute while uncertain slots continue iterating.

**Binding:** Complex multi-hop chains. `bind(bind(rain, causes), entails)`
composes two relations. Permute is introduced: reorder slot assignments to
test structural invariance (the meaning shouldn't change if you swap the
physical position of equivalent slots).

**Discrimination:** Narrative coherence across 12 slots. A story where the
Agent, Predicate, Result, and Cause are all mutually consistent has low
energy. Shuffle one slot's content to a different role → high energy.

**Multi-Resolution:** Cross-resolution inference becomes primary. R0 discourse
gist ("a cooking disaster") should be derivable from R1 propositions ("chef
burned the pasta, smoke filled the kitchen, fire alarm rang").

**Codebook initialization happens here.** After Stage 4, the encoder produces
meaningful embeddings for the first time across 12 diverse slots. Run k-means
to produce the initial codebook (65,536 entries).

#### Datasets

| Dataset | What It Provides | Subset Used |
|---|---|---|
| bAbI QA (tasks 1-10) | Multi-hop reasoning, path-finding | Tasks 1-10 (increasing complexity) |
| CLUTRR | Family relation chains (multi-hop relational reasoning) | 5-10 hop complexity |
| PropBank (narrative subset) | Multi-predicate sentences with complex arg structure | Sentences with ≥ 4 arguments |
| ATOMIC 2020 (full) | Causal chains (if X then Y then Z) | Full if-then tuples |
| Self-generated denoising | Corrupt any frame from Stages 1-3 corpus | Unlimited generated |

#### Gate Criteria (advance to Stage 5 when met)

- Slot filling accuracy ≥ 70% (mask 4-6 of 12 slots)
- Multi-hop reasoning accuracy ≥ 60% (bAbI tasks 2-3)
- VFN learns easy-first ordering (Agent converges before Result in >70% of cases)
- Codebook quantization error < 0.15 on Stage 4 embeddings
- VFN converges within 20 iterations on >75% of examples

---

### Stage 5: Hypotheticals (The Teenager)

**What Volt learns:** Full abstract reasoning. Systematic compositionality —
recombining known concepts in novel ways. All 16 slots, full 30-iteration
budget, all HDC operations.

**Active slots:** 16 (all)
**VFN iterations:** up to 30
**HDC operations:** full algebra (superpose + bind + unbind + permute)
**Duration:** 4-6 weeks
**Hardware:** 1x H100, ~100-200 GPU-hours

#### Training Signals at This Stage

**Slot Filling (Full Denoising):** Mask 6-10 of 16 slots. Heavily corrupted
frames (zeroed slots, Gaussian noise, shuffled assignments). The VFN predicts
per-slot drift vectors that restore the original frame. This is the full
denoising objective from the v2.0 plan, but the VFN has already practiced
thousands of iterations of progressively harder slot reconstruction.

**Binding (Slot-Conditional Flow Matching):** Flow match per-slot with
slot-identity constraints. Time variable t_s is per-slot (different slots
converge at different rates). Constraint: Agent-slot drift must point toward
Agent-like embeddings, not arbitrary vectors.

**Discrimination (Full Energy Landscape):** 16-slot coherence with all
resolutions active. The VFN's energy function must score narrative coherence,
causal consistency, temporal ordering, and structural validity simultaneously.

**Multi-Resolution (Reasoning Chain Supervision):** Given (premise, step_1,
step_2, ..., conclusion), train VFN to produce one refinement step at a time.
NOT: map question directly to answer (that's disguised NTP). YES: map
frame(t) to frame(t+1) where each step is a valid reasoning move.

#### Datasets

| Dataset | What It Provides | Subset Used |
|---|---|---|
| SCAN | Compositional generalization (jump twice and walk left) | Full dataset (20K commands) |
| COGS | Systematic compositional OOD generalization | Full dataset (24K sentences) |
| CLUTRR (extended) | Multi-hop relational reasoning at high complexity | 10-15 hop chains |
| bAbI QA (all 20 tasks) | Full multi-hop reasoning suite | All tasks |
| Self-generated denoising | Corrupt any frame from Stages 1-4 corpus | Unlimited generated |

#### Gate Criteria (advance to Stage 6 when met)

- SCAN compositional generalization ≥ 85%
- bAbI tasks 1-3 accuracy ≥ 75%
- Frame denoising: exact slot reconstruction (cosine sim > 0.9) for ≥ 65% of slots
- RAR convergence: ≥ 70% of test queries converge within 30 iterations
- Analogy accuracy ≥ 35% (A:B::C:? via bind/unbind algebra)

---

### Stage 6: Specialization (The Adult)

**What Volt learns:** Code is a structured domain language. A system that
already understands agents, actions, results, causality, and compositional
reasoning now applies those concepts to programming.

**Active slots:** 16 (with code-specific role mappings)
**VFN iterations:** up to 30
**HDC operations:** full algebra + code attention bias
**Duration:** 3-4 weeks
**Hardware:** 1x H100, ~100-200 GPU-hours

#### Why Code Is a Specialization, Not a Foundation

By Stage 6, the system already knows:

- "weather" is a natural phenomenon (Stage 1: naming)
- "forecast" is a prediction about future states (Stage 3: context)
- "app" is a software instrument used by agents (Stage 3: context)
- Given premises, derive conclusions via frame refinement (Stage 5: reasoning)

Writing a weather app then becomes composing known concepts — not memorizing
code patterns from (problem, solution) text pairs.

#### Code Slot Mapping

| Code Element | TensorFrame Slot | Example |
|---|---|---|
| Function/class name | S0 (Agent) | `def sort_array` |
| Operation/method | S1 (Predicate) | `sorted()`, `append()` |
| Arguments/parameters | S2 (Patient) | `(arr, reverse=True)` |
| Return value | S3 (Location*) | `return sorted_arr` |
| Loop/iteration | S4 (Time) | `for i in range(n)` |
| Algorithm pattern | S5 (Manner) | recursive, iterative, DP |
| Control flow | S6-S8 (Instrument/Cause/Result) | if/else, try/except |
| Complex logic | S9-S15 (Free) | nested structures |

*Location slot is overloaded for code to mean "where the result goes."

#### Training Signals at This Stage

**Code Slot Assignment** (supervised): Fine-tune the Role Head to classify
code tokens into code-specific slots. Uses the existing heuristic role labels
from Phase 0.4 (def→S0, args→S2, return→S3, if→S6, for→S4).

**Code Semantic Embedding** (contrastive): Same InfoNCE as earlier stages but
on code pairs. `sum_array()` ≈ `calculate_total()` in embedding space.

**Code Denoising** (VFN fine-tune): Apply the Stage 5 denoising objective to
code frames. Mask the function-name slot, train VFN to predict it from
arguments + body structure.

**Compositional Code Reasoning** (VFN fine-tune): Given a problem description
frame, refine toward a solution frame. Uses slot-conditional flow matching
with constraints from Stage 5 — not flat frame-to-frame mapping.

#### Code Attention Bias

The existing code attention bias matrix (ADR-029) is applied on top of the
general language attention weights learned in Stages 1-5:

```
Final attention = learned_language_weights + code_attention_bias
```

This preserves linguistic priors (Agent↔Predicate) while adding code-specific
ones (Function↔Arguments, Return↔Operation).

#### Datasets

| Dataset | Size | What It Provides | HF Name |
|---|---|---|---|
| CodeSearchNet (Python) | 100K pairs | Function-docstring pairs (contrastive) | Already downloaded (144 MB) |
| tiny-codes | 1.6M pairs | Synthetic instruction→code (pre-formatted) | `nampdn-ai/tiny-codes` |
| CodeAlpaca | 20K pairs | Instruction→code (pre-formatted) | `sahil2801/CodeAlpaca-20k` |
| code_instructions_122k | 122K pairs | Instruction→code (pre-formatted) | `TokenBender/code_instructions_122k_alpaca_style` |
| HumanEval | 164 problems | Evaluation benchmark (with tests) | Already downloaded |
| MBPP | 974 problems | Evaluation benchmark (with tests) | Already downloaded |
| TACO | 26K problems | Competitive programming (with tests) | `BAAI/TACO` |

**NOT used for VFN training:** The Stack (44 GB of raw Python files). Retained
for BPE tokenizer training (already done) and codebook initialization only.

#### Gate Criteria (advance to Joint Alignment when met)

- Code slot assignment accuracy ≥ 85%
- Code paraphrase similarity ≥ 0.75
- RAR convergence on code problems ≥ 60% within 30 iterations
- Decoded code is syntactically valid ≥ 80% of the time

---

## Part IV: Joint Alignment & Scale

### JA: Joint Alignment

**Goal:** End-to-end calibration of the full pipeline.

**Duration:** 4-6 weeks
**Hardware:** 1-2x H100, ~200-400 GPU-hours

#### JA.1 — Codebook Refinement

If the codebook was initialized after Stage 4, re-run k-means using the
Stage 6 encoder (which now embeds both language and code) to produce final
65,536 codebook entries. CPU-only job (~1-2 hours on 32+ cores).

Output: `checkpoints/codebook.bin` (~64 MB)

#### JA.2 — Intent Router Calibration

Train strand capability vectors so cosine routing selects the correct Hard
Strand. 10K labeled queries across strands (MathEngine, CodeRunner,
CodeDebugger, HDCAlgebra, etc.).

Training: contrastive loss on 256-dim strand vectors.
Deliverable: routing accuracy >90% on held-out test set.

#### JA.3 — Certainty Calibration (gamma)

Run inference on 5K problems with known answers. Collect (prediction, gamma,
is_correct) triples. Fit isotonic regression to map raw gamma to calibrated
P(correct).

Deliverable: Expected Calibration Error (ECE) < 0.08.

#### JA.4 — Safety Axiom Refinement

Refine K1-K5 axiom vectors using adversarial examples. Minimize false negatives
(dangerous content that passes) while keeping false positives (safe content
blocked) below 5%.

Deliverable: false negative rate <2%, false positive rate <5%.

#### JA.5 — End-to-End Fine-Tuning

Fine-tune VFN + attention with gradients flowing through the full pipeline.
Freeze encoder/decoder (trained through Stages 1-6). Use the denoising +
slot-conditional objectives from Stage 5, but on end-to-end pipeline outputs.

For code: add test-execution reward signal via REINFORCE (test pass/fail is
perfect verification — no human labeling needed).

#### JA.6 — RLVF Alignment

REINFORCE with Verified Feedback, using the existing infrastructure in
`volt-learn/rlvf.rs`. Reward shaping:

| Outcome | Reward |
|---|---|
| Correct + calibrated (gamma matches accuracy) | +1.0 |
| Correct + underconfident | +0.5 |
| Wrong + honest (low gamma) | +0.2 |
| Wrong + overconfident (high gamma) | -2.0 |
| Safety violation | -5.0 |

Deliverable: overconfident error rate <8%, ECE <0.06.

#### JA.7 — Sleep Consolidation Validation

Run 5K inference queries, trigger sleep cycle (Forward-Forward + Distillation +
Graduation + GC), re-evaluate. Validate that post-sleep performance improves
without catastrophic forgetting.

Deliverable: >5% improvement on new queries, <3% degradation on old queries.

---

### S: Scale & Benchmark

**Goal:** Scale VFN, add domains/languages, publish benchmarks.

**Duration:** 4-8 weeks
**Hardware:** 2-4x H100 or multi-GPU node, ~500-1500 GPU-hours

#### S.1 — VFN Scaling (51M → 200M → 500M)

Progressive scaling with knowledge distillation. Implement Fourier Neural
Operator architecture at 500M params. Validate at each scale point.

#### S.2 — Multi-Domain Training

Extend beyond code to multiple domains using the same Stage 1-5 weights:
- Natural language QA (Natural Questions, SQuAD)
- Mathematics (GSM8K, MATH)
- Scientific reasoning (SciQ, ARC-Challenge)
- Multi-lingual code (MultiPL-E: Python → JS, Java, Rust, Go)

Each domain reuses Stage 1-5 foundation and adds domain-specific fine-tuning
(same pattern as Stage 6 for code).

#### S.3 — Benchmark Publication

Primary benchmarks (structural advantage expected):
- **SCAN/COGS**: Compositional generalization (Volt's HDC binding advantage)
- **bAbI**: Multi-hop reasoning (Volt's three-tier memory advantage)
- **ARC-AGI**: Abstract reasoning (RAR's iterative refinement advantage)

Secondary benchmarks (parity expected):
- **HumanEval / MBPP**: Code generation (Pass@1)
- **MMLU**: General knowledge (baseline comparison)

Baselines: comparably-sized transformer trained on the same data with the same
compute budget.

Deliverable: arXiv paper demonstrating Volt's advantages on compositional and
reasoning tasks at lower compute cost.

---

## Part V: Infrastructure & Datasets

### Phase 0: Infrastructure (DONE, except updates)

All infrastructure from the original Phase 0 is reusable:

- **0.1 VFN Checkpoint System** — DONE. Save/load works.
- **0.2 Dataset Pipeline** — DONE for JSONL. Needs extension:
  - Add HuggingFace streaming adapter
  - Add FrameNet/PropBank/ConceptNet format converters
  - Add slot-annotated data format: JSONL with `{"text": "...", "slots": {"S0": "agent_word", "S1": "predicate_word", ...}}`
- **0.3 Codebook Init** — Code DONE, k-means deferred to after Stage 4.
- **0.4 Attention Bias** — DONE for code-specific patterns. Add a general
  language bias matrix as the default, with code bias applied on top in Stage 6.

**New infrastructure needed:**

- **0.5 HuggingFace Streaming Script** — Python script in `tools/` that:
  - Accepts a HuggingFace dataset name and split
  - Streams rows via `datasets.load_dataset(..., streaming=True)`
  - Converts to Volt's JSONL format
  - Writes to stdout or a file
  - Handles FrameNet, PropBank, ConceptNet, SCAN, COGS, CodeSearchNet,
    tiny-codes, etc. via format-specific converters
  - No full dataset download required

- **0.6 Curriculum Scheduler** — Rust module in `volt-learn` that:
  - Tracks current developmental stage
  - Monitors gate metrics per stage
  - Controls active slot count, max VFN iterations, available HDC ops
  - Manages rehearsal mixing (20% earlier stages)
  - Logs stage transitions and metric history

### Dataset Strategy

#### Principles

1. **Curated over raw.** Pre-formatted (input, output) pairs > raw source
   files. 20K high-quality CodeAlpaca pairs are more useful than 6.5M raw
   Python files from The Stack.

2. **Stream, don't download.** Use `datasets.load_dataset(..., streaming=True)`
   for anything over 1 GB.

3. **Structured over flat.** Prefer datasets with annotations (SRL labels,
   semantic roles, relation types) over plain text.

4. **Small and dense > large and sparse.** 100K densely-annotated FrameNet
   sentences are worth more than 10M unlabeled sentences.

#### Dataset Summary by Stage

| Stage | Primary Datasets | Total Size | Format |
|---|---|---|---|
| 1: Naming | FrameNet (simple), WordNet, STS-B (concrete) | ~100K examples | Stream from HF |
| 2: Sentences | PropBank (SVO), FrameNet (≤4 roles), PAWS | ~500K examples | Stream from HF |
| 3: Context | PropBank (full), ConceptNet, ATOMIC, STS-B (full) | ~3M examples | Stream from HF |
| 4: Narrative | bAbI (1-10), CLUTRR, ATOMIC (chains), self-generated | ~200K + unlimited | Stream + generated |
| 5: Hypotheticals | SCAN, COGS, CLUTRR (extended), bAbI (all), self-generated | ~100K + unlimited | Stream + generated |
| 6: Code | CodeSearchNet, tiny-codes, CodeAlpaca, code_instructions, TACO | ~1.9M pairs | Stream or downloaded |
| JA: Alignment | Reuse Stage 6 eval sets + generated adversarial | ~20K labeled | Generated |
| S: Scale | Natural Questions, GSM8K, MATH, SciQ, MultiPL-E | ~500K+ | Stream from HF |

#### What To Keep From Existing Downloads

| Downloaded Dataset | Size | Keep? | Reason |
|---|---|---|---|
| The Stack Python (44 GB) | 6.5M files | Keep for codebook init & BPE only | Raw files, not useful for VFN training |
| CodeSearchNet (144 MB) | 100K pairs | Keep | Pre-formatted, used in Stage 6 |
| HumanEval (0.2 MB) | 164 problems | Keep | Evaluation benchmark |
| MBPP (0.1 MB) | 257 problems | Keep | Evaluation benchmark |
| APPS (1.3 GB) | 10K problems | Keep | Evaluation + RLVF training |
| MultiPL-E (3 MB) | 2.4K problems | Keep | Cross-lingual evaluation |

### HuggingFace Streaming Script (to implement)

```python
# tools/stream_dataset.py — example usage
# Stream FrameNet and convert to Volt JSONL format
python tools/stream_dataset.py \
    --dataset framenet_v17 \
    --format slot_annotated \
    --output data/framenet_train.jsonl \
    --max-examples 200000

# Stream ConceptNet triples
python tools/stream_dataset.py \
    --dataset conceptnet5 \
    --format triples \
    --output data/conceptnet_en.jsonl \
    --language en

# Stream tiny-codes for code training
python tools/stream_dataset.py \
    --dataset nampdn-ai/tiny-codes \
    --format code_pairs \
    --output data/tiny_codes.jsonl \
    --max-examples 500000
```

---

## Part VI: Operational Details

### Compute Estimates

| Stage | GPU-Hours | Hardware | Est. Cost (Cloud) |
|---|---|---|---|
| 1: Naming | 5 | 1x RTX 4090 | $2-5 |
| 2: Sentences | 10 | 1x RTX 4090 | $5-10 |
| 3: Context | 20-30 | 1x RTX 4090 | $10-20 |
| 4: Narrative | 60-100 | 1x H100 | $150-250 |
| 5: Hypotheticals | 100-200 | 1x H100 | $250-500 |
| 6: Code | 100-200 | 1x H100 | $250-500 |
| JA: Joint Alignment | 200-400 | 1-2x H100 | $500-1000 |
| S: Scale & Benchmark | 500-1500 | 2-4x H100 | $1250-3750 |
| **Total** | **~995-2445** | | **~$2400-6000** |

Rehearsal overhead is included in per-stage estimates (20% replay adds ~20%
compute but uses smaller/faster frames from earlier stages).

### Checkpoint Inventory

| Checkpoint | Produced By | Size | Path |
|---|---|---|---|
| BPE Tokenizer | Phase 0 (done) | 2.3 MB | `checkpoints/code_tokenizer.json` |
| Stage 4 Encoder | Stage 4 | ~20 MB | `checkpoints/stage4_encoder.safetensors` |
| Stage 4 Decoder | Stage 4 | ~21 MB | `checkpoints/stage4_decoder.safetensors` |
| Stage 4 VFN | Stage 4 | ~200 MB | `checkpoints/stage4_vfn.safetensors` |
| Codebook (initial) | After Stage 4 | ~64 MB | `checkpoints/codebook_init.bin` |
| Stage 5 VFN | Stage 5 | ~200 MB | `checkpoints/stage5_vfn.safetensors` |
| Code Encoder | Stage 6 | ~20 MB | `checkpoints/code_encoder.safetensors` |
| Code Decoder | Stage 6 | ~21 MB | `checkpoints/code_decoder.safetensors` |
| Code VFN | Stage 6 | ~200 MB | `checkpoints/code_vfn.safetensors` |
| Codebook (final) | JA.1 | ~64 MB | `checkpoints/codebook.bin` |
| VFN (Aligned) | JA | ~200 MB | `checkpoints/vfn_aligned.safetensors` |
| VFN (Scaled, 500M) | S.1 | ~2 GB | `checkpoints/vfn_scaled.safetensors` |

Note: Stages 1-3 don't produce separately named checkpoints — each stage
overwrites `checkpoints/current_encoder.safetensors`, `current_decoder.safetensors`,
and `current_vfn.safetensors`. Named checkpoints begin at Stage 4 when
embeddings are meaningful enough to snapshot.

### Cloud Execution

Cloud setup details will be written as a separate operational guide when the
spiral training binary is implemented. The core approach (tmux sessions,
checkpoint sync, per-epoch saves for preemption recovery) from the archived
CLOUD_TRAINING_PLAN.md remains valid — only the training commands and
curriculum scheduler configuration change.

### What Needs to Be Implemented in Code

| Item | Crate | Description |
|---|---|---|
| HF streaming script | `tools/stream_dataset.py` | Python script for dataset conversion |
| Curriculum scheduler | `volt-learn` | Stage tracking, gate metrics, rehearsal mixing, slot/iteration control |
| FrameNet/PropBank data loader | `volt-learn` | JSONL reader for slot-annotated sentences |
| ConceptNet triple loader | `volt-learn` | JSONL reader for (concept, relation, concept) triples |
| Denoising training loop | `volt-soft` or `volt-learn` | Corrupt frames, train VFN to restore (used from Stage 1) |
| Slot-conditional flow matching | `volt-soft` | Per-slot time variables + slot-identity constraint |
| Reasoning chain trainer | `volt-learn` | Multi-step refinement training |
| Active slot masking | `volt-soft` | Mask inactive slots in SlotAttention based on current stage |
| Progressive Role Head | `volt-translate` | Role Head output size grows from 2 to 16 across stages |
| Language attention bias | `volt-soft` | General linguistic slot attention prior |
| `train-spiral` binary | `volt-learn` | Training binary for all 6 developmental stages |

---

## Part VII: Success Criteria

### Stage 1 (Naming)

- Slot filling: ≥ 70% on 2-slot frames
- Superposition retrieval: ≥ 60%
- Discrimination AUC: ≥ 0.65

### Stage 2 (Sentences)

- Slot filling: ≥ 75% on 3-4 slot frames
- Bind retrieval: ≥ 40%
- Role violation AUC: ≥ 0.70

### Stage 3 (Context)

- Slot filling: ≥ 75% on 8-slot frames
- Bind/unbind retrieval: ≥ 50% on ConceptNet test
- 2-hop reasoning: ≥ 50%
- STS-B Spearman correlation: ≥ 0.70

### Stage 4 (Narrative)

- Slot filling: ≥ 70% on 12-slot frames
- bAbI tasks 2-3: ≥ 60%
- Easy-first ordering: ≥ 70%
- Codebook quantization error: < 0.15

### Stage 5 (Hypotheticals)

- SCAN length generalization: ≥ 85%
- bAbI tasks 1-3: ≥ 75%
- Frame denoising (cosine sim > 0.9): ≥ 65% of slots
- RAR convergence within 30 iterations: ≥ 70%
- Analogy accuracy: ≥ 35%

### Stage 6 (Code)

- Code slot assignment: ≥ 85%
- Code paraphrase similarity: ≥ 0.75
- RAR convergence on code: ≥ 60% within 30 iterations
- Syntactically valid decoded code: ≥ 80%

### Joint Alignment

- ECE: < 0.08
- Overconfident error rate: < 8%
- Post-sleep improvement: > 5%, forgetting < 3%

### Scale

- SCAN/COGS: ≥ 90% compositional generalization
- HumanEval Pass@1: ≥ 25% (at 500M params)
- Compute efficiency: match transformer baseline at lower GPU-hours

---

## Appendix A: Comparison with Previous Approaches

| Aspect | v1.0 (Code-First) | v2.0 (Sequential) | v3.0 (Spiral) |
|---|---|---|---|
| **Starting point** | Code (CodeSearchNet) | Language (FrameNet, PropBank) | 2-slot concrete nouns |
| **VFN training start** | Phase F3 (week 6+) | Phase F3 (week 6+) | Stage 1 (day 1) |
| **VFN objective** | Flat flow matching | Denoising + slot-conditional | Same, but practiced from day 1 |
| **Active slots** | All 16 from start | All 16 from start | 2 → 16 progressive |
| **VFN iterations** | Fixed at 30 | Fixed at 30 | 1 → 30 progressive |
| **Catastrophic forgetting** | N/A (single phase) | Risk at F1→F2→F3 boundaries | Mitigated by 20% rehearsal |
| **VFN total training time** | ~100-200 hours | ~100-200 hours | ~295-530 hours (5x more) |
| **NTP presence** | Decoder + VFN (accidental) | Decoder only | Decoder only |
| **Dataset strategy** | Download The Stack (44 GB) | Stream from HF | Stream from HF |
| **Curriculum model** | None | Semester courses | Montessori / Piaget developmental |
| **Implementation complexity** | Simple (1 loop) | Moderate (3 separate loops) | Higher (1 loop + curriculum scheduler) |
| **Debugging** | Easy (isolated) | Easy (isolated phases) | Harder (stage interactions) |
| **Closest analogy** | Trade school | University (finish calc, then physics) | Montessori education |

## Appendix B: The Spiral Advantage — Why 5x VFN Practice Matters

In the sequential plan, the training timeline looks like:

```text
Week 1-3:  Encoder trains on F1 (lexical grounding)     — VFN idle
Week 3-6:  Encoder trains on F2 (world knowledge)       — VFN idle
Week 6-12: VFN trains on F3 (compositional reasoning)   — VFN learning from scratch
Week 12-16: VFN fine-tunes on D1 (code)                 — VFN adapting
```

In the spiral plan:

```text
Week 1-2:  VFN fills 1 slot from 2-slot frames          — trivial, builds confidence
Week 2-5:  VFN fills 1-2 slots from 3-4 slot frames     — learns SVO structure
Week 5-9:  VFN fills 2-4 slots from 8-slot frames       — learns contextual inference
Week 9-14: VFN fills 4-6 slots from 12-slot frames      — learns ordering strategy
Week 14-20: VFN fills 6-10 slots from 16-slot frames    — masters full reasoning
Week 20-24: VFN applies skills to code domain            — fast specialization
```

By week 6, the spiral VFN has completed ~1000 training iterations on progressively
harder slot reconstruction. The sequential VFN has completed zero. This head start
compounds — the spiral VFN arrives at Stage 5 (SCAN/COGS level reasoning) having
already internalized slot structure, convergence patterns, and easy-first strategies
that the sequential VFN must learn cold.

---

*This document is the single source of truth for Volt X training. All previous
training plans are archived in `archive/training/`. When implementing training
code, reference this document and `training_config.toml` for dataset paths.
When the approach changes, update THIS document — do not create a new one.*
