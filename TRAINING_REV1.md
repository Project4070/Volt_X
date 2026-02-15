# Volt X — Developmental Training Plan (Spiral Curriculum)

**Version:** 3.0 (Developmental / Spiral Approach)
**Refers to:** `TRAINING.md` (previous sequential approach)
**Status:** PROPOSED REVISION

---

## The Philosophy: "Growing Up" Instead of "Loading Data"

The previous training plan (`TRAINING.md`) treated learning as a series of university courses: first you take Linguistics 101, then Knowledge 202, then Reasoning 303, and finally Code 404.

This revision proposes a **developmental approach**, modeled after human child development and Yann LeCun's JEPA (Joint Embedding Predictive Architecture). A child doesn't learn "all nouns" before learning "any verbs." They learn "ball" (noun), then "ball rolls" (verb), then "I push ball" (causation), then "if I drop ball it falls" (physics).

We replace distinct *Phases* with a single **Spiral Curriculum**. The model loops through the same four training signals from day one, but the complexity of the content, the number of active slots, and the depth of reasoning (VFN iterations) grow over time.

### Alignment with LeCun's JEPA

Volt's architecture is naturally a JEPA. The convergence is striking:
1.  **Prediction in Representation Space:** Volt predicts slot embeddings (256-dim vectors), not tokens.
2.  **Energy-Based Discrimination:** Coherent frames have low energy; incoherent frames have high energy.
3.  **Hierarchical Prediction:** Volt predicts at multiple resolutions (R0 Gist → R3 Token).
4.  **No Generative Reconstruction:** We don't reconstruct pixels/tokens; we reconstruct semantic slots.

This plan embraces that identity.

---

## Part I: The Spiral Curriculum

Instead of switching datasets, we switch **Developmental Stages**. A `CurriculumScheduler` controls the complexity of the task.

### The Scheduler

```rust
struct CurriculumScheduler {
    current_stage: u8,           // 1-6
    active_slots: Vec<SlotRole>, // Which slots are "unlocked"
    active_resolutions: Vec<Resolution>, // R0-R3
    max_vfn_iterations: u32,     // RAR budget (starts at 1, grows to 30)
    dataset_filters: StageFilter,// Controls which data rows are eligible
    rehearsal_ratio: f32,        // Fraction of batch from prior stages (prevents forgetting)
}
```

Advancement is **metric-gated**, not time-gated. Volt moves to Stage 3 only when Stage 2 metrics are met.

### The Six Developmental Stages

#### Stage 1: "Naming" (The Infant)
*   **Goal:** Learn that concrete nouns exist and belong in Agent/Patient slots.
*   **Active Slots:** S0 (Agent), S2 (Patient). (2 of 16)
*   **Active Resolutions:** R0 (Gist) only.
*   **VFN Iterations:** 1 (Single pass).
*   **Vocabulary:** ~500 concrete nouns, ~100 properties.
*   **What it learns:**
    *   `S0=dog` is valid. `S0=run` is invalid (wrong slot).
    *   `dog` is similar to `cat` (embedding space).
*   **Datasets:** WordNet hypernyms, ConceptNet HasProperty.
*   **Signals:**
    *   *Slot Filling:* Fill 1 missing slot from 1 given slot.
    *   *Binding:* Superposition only (`dog + cat ≈ animal`).

#### Stage 2: "Sentences" (The Toddler)
*   **Goal:** Learn actions and SVO structure.
*   **Active Slots:** S0 (Agent), S1 (Predicate), S2 (Patient). (3 of 16)
*   **Active Resolutions:** R0 + R1 (Proposition).
*   **VFN Iterations:** 3.
*   **Vocabulary:** ~2,000 (verbs, common objects).
*   **What it learns:**
    *   "Chef cooks pasta" (Agent-Predicate-Patient).
    *   "Pasta cooks chef" is high energy (role violation).
*   **Datasets:** PropBank (Arg0+V+Arg1 only), ConceptNet CapableOf/UsedFor.
*   **Signals:**
    *   *Binding:* `bind(chef, CapableOf) ≈ cooking`.
    *   *Multi-Res:* R0 "cooking topic" aligns with R1 "chef cooks pasta".

#### Stage 3: "Where and When" (The Preschooler)
*   **Goal:** Modifiers, location, time, simple causation.
*   **Active Slots:** S0–S7 (Agent through Cause). (8 of 16)
*   **Active Resolutions:** R0–R2 (Phrase level).
*   **VFN Iterations:** 5-10.
*   **Vocabulary:** ~5,000.
*   **What it learns:**
    *   "Chef cooks pasta *in kitchen*" (Location).
    *   "Rain *causes* flood" (Causation chain).
*   **Datasets:** Full FrameNet, PropBank (ArgM-LOC/TMP/MNR), bAbI (Task 1).
*   **Signals:**
    *   *Binding:* 2-hop chains (`rain → causes → flood`).
    *   *Discrimination:* "Chef cooks pasta *in algebra*" = High Energy.

#### Stage 4: "Stories" (The School Child)
*   **Goal:** Multi-step reasoning, results, instruments, full frames.
*   **Active Slots:** S0–S8 (All semantic roles).
*   **Active Resolutions:** All (R0–R3).
*   **VFN Iterations:** 10-15 (First real RAR usage).
*   **What it learns:**
    *   Iterative refinement: Fill easy slots → Fill dependent slots → Consistency check.
    *   Analogy: `dog:animal :: car:vehicle`.
*   **Datasets:** CLUTRR (3-5 hops), bAbI (Tasks 2-8), SCAN (simple commands), ATOMIC.
*   **Signals:**
    *   *Slot Filling:* Mask 3-4 slots, reconstruct all.

#### Stage 5: "Abstract Thinking" (The Teenager)
*   **Goal:** Compositional generalization, logic, hypothetical reasoning.
*   **Active Slots:** All 16 (S9-S15 used for complex abstract structures).
*   **VFN Iterations:** 30 (Full budget).
*   **What it learns:**
    *   Systematic compositionality (COGS).
    *   Length generalization (SCAN).
    *   Complex multi-hop deduction (bAbI all tasks).
*   **Datasets:** SCAN (full), COGS, CLUTRR (8-15 hops), Self-generated denoising.
*   **Key:** This is where previous "Phase F3" happens, but now the VFN has been practicing slot reconstruction for weeks.

#### Stage 6: "Specialization" (The Adult / Code)
*   **Goal:** Learn Code as a structured domain language.
*   **Active Slots:** All 16 (mapped to code constructs).
*   **What it learns:**
    *   `def sort(arr)`: S0=sort, S2=arr.
    *   Recursion, control flow, algorithms.
*   **Datasets:** CodeSearchNet, tiny-codes, HumanEval, MBPP.
*   **Note:** This replaces "Phase D1". Code is just another language, learned after the brain is fully formed.

---

## Part II: Technical Implementation

### 1. The Training Loop
Instead of 3 separate scripts, we have one `train-spiral` binary.

```python
# Conceptual loop
scheduler = CurriculumScheduler()
while scheduler.stage <= 6:
    batch = get_batch(scheduler.active_datasets, rehearsal_ratio=0.2)

    # 1. Mask slots based on stage difficulty
    masked_frame = mask_frame(batch.frame, difficulty=scheduler.stage)

    # 2. VFN Forward (Iterative)
    # Only attend to active slots! Mask the rest.
    prediction = vfn.forward(masked_frame, steps=scheduler.max_vfn_iterations)

    # 3. JEPA / Denoising Loss
    loss = mse(prediction, batch.target_frame)
    loss += energy_loss(prediction) # Push down energy of correct frame

    # 4. Update
    optimizer.step(loss)

    # 5. Check advancement
    if scheduler.check_metrics(validation_set):
        scheduler.advance_stage()
```

### 2. Component Adaptation

| Component | Adaptation for Spiral |
|---|---|
| **CNN Encoder** | Role Head output grows. Starts predicting only S0/S2, eventually all 16. Gradients for unused slots are zeroed. |
| **ScaledVfn** | `max_iterations` is dynamic. Slot Attention mask grows. |
| **Decoder** | Trained alongside, but only decodes active slots. |
| **HDC Ops** | `superpose` (Stage 1) → `bind` (Stage 2) → `unbind` (Stage 4) → `permute` (Stage 5). |

### 3. Rehearsal Strategy (The Spiral)
To prevent catastrophic forgetting (e.g., forgetting what "dog" is while learning Python), **20% of every batch** consists of data from previous stages.
*   In Stage 4, we still see Stage 1 noun-binding tasks.
*   This keeps the "foundation" solid while building the "penthouse."

---

## Part III: Dataset Strategy (Mapped to Stages)

| Stage | Primary Datasets | Size (Rows) |
|---|---|---|
| **1. Naming** | WordNet (hypernyms), ConceptNet (HasProperty) | ~200K |
| **2. Sentences** | PropBank (simple), ConceptNet (Action) | ~500K |
| **3. Where/When** | FrameNet (full), PropBank (full), bAbI (simple) | ~1M |
| **4. Stories** | CLUTRR, ATOMIC, SCAN (simple), bAbI (med) | ~2M |
| **5. Abstract** | SCAN (hard), COGS, CLUTRR (hard), Self-Gen | ~2M+ |
| **6. Code** | CodeSearchNet, tiny-codes, HumanEval | ~2M |

**Streaming is still key.** We use the same `tools/stream_dataset.py` infrastructure, but the `StageFilter` selects which rows to yield.

---

## Part IV: Tradeoffs vs. Sequential Plan

| Feature | Sequential (`TRAINING.md`) | Spiral (`TRAINING_REV1.md`) |
|---|---|---|
| **Complexity** | Low (isolated phases) | High (dynamic scheduler) |
| **VFN Training** | Late (Phase F3) | Day 1 (Stage 1) |
| **Forgetting** | High risk at boundaries | Mitigated by rehearsal |
| **Compute** | Wasted early (Encoder only) | Efficient (Train all always) |
| **Analogy** | University Courses | Child Development |
| **Debuggability** | Easy | Harder (interactions) |

**Why Spiral wins:** The VFN needs practice. In the sequential plan, the VFN (the "brain") sits idle for weeks while the Encoder (the "eyes") learns to see. In the spiral plan, the brain starts practicing with simple concepts immediately.

---

## Part V: Execution Plan

1.  **Refactor Data Pipeline:** Implement `StageFilter` to slice datasets by complexity (e.g., filter PropBank for only SVO rows for Stage 2).
2.  **Implement Scheduler:** Build `CurriculumScheduler` in `volt-learn`.
3.  **Update Models:** Ensure `ScaledVfn` and `CodeEncoder` handle dynamic slot masking/activation.
4.  **Launch Stage 1:** Verify "Naming" convergence before unlocking Stage 2.

This approach is riskier to implement but promises a much more robust final model.
