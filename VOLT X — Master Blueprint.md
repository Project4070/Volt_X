
# VOLT v3.0 — Master Blueprint

## Stateful Operating System for Sovereign Intelligence

Version: 3.0 — "The Lipstick Masquerade"
Date: February 8, 2026
Status: Foundation Specification
Classification: Open Architecture

---

> *"We're playing a dangerous game, and things will never be the same."*

---

# Table of Contents

1. Executive Summary
2. Philosophical Foundations
3. Architecture Overview
4. The Tensor Frame — Data Structure of Thought
5. The LLL Vector Bus — Algebra of Meaning
6. Layer 1: Input Translators
7. Layer 3: GPU Soft Core — Root-Attend-Refine
8. Layer 4: CPU Hard Core — Deterministic Logic
9. Layer 5: VoltDB — Memory Management Engine
10. Layer 6: Output Action Cores
11. Layer 7: Continual Learning Engine
12. Layer 8: Intelligence Commons
13. Layer 9: UI and Test Bench
14. Layer 10: Socket Standard — Trait Interfaces
15. Training Pipeline
16. Inference Pipeline
17. Safety Architecture
18. Hardware Requirements and Deployment
19. Theoretical Foundations — Relation to A Thousand Brains
20. Roadmap
21. Glossary

---

# 1. Executive Summary

## 1.1 What Is Volt?

Volt is a cognitive architecture that treats intelligence as a stateful operating system rather than a stateless prediction engine. It separates thinking from speaking, verification from generation, and memory from computation. It runs on consumer hardware.

Where current large language models predict the next token in a sequence, Volt maintains persistent structured memory, reasons through continuous dynamics with adaptive computation time, verifies conclusions through deterministic logic, and learns continuously from every interaction without retraining.

## 1.2 The Problem

Modern AI suffers from five fundamental limitations that are architectural, not solvable by scaling:

Fixed lifespan. A language model's context window is its lifespan. When the window fills, old content is forgotten permanently. Performance degrades as context grows due to quadratic attention costs.

Conflated thinking and speaking. To reason, a language model must generate tokens. Each token costs the same compute regardless of problem difficulty. There is no mechanism for silent contemplation.

Statelessness. Every API call starts fresh. The model has no persistent memory of you, your projects, your preferences, or your conversation history beyond what fits in the current context window.

Centralized dependence. Intelligence is rented from cloud providers at per-token cost. The user owns nothing. The provider controls everything.

Frozen knowledge. After training, the model cannot learn. Fine-tuning is expensive, brittle, and causes catastrophic forgetting.

## 1.3 The Solution

Volt addresses each limitation through architectural separation:

| Problem | Volt Solution | Mechanism |
|---|---|---|
| Fixed lifespan | Unlimited memory, bound only by hardware | Three-tier memory (GPU VRAM → RAM → SSD) with background bleeding |
| Conflated thinking/speaking | Silent structured reasoning | Root-Attend-Refine loop over Tensor Frame slots |
| Statelessness | Persistent strand storage | Millions of Tensor Frames in RAM, organized by strand, recalled in milliseconds |
| Centralized dependence | Runs on consumer hardware | Split-brain: GPU for intuition, CPU for logic, RAM for memory |
| Frozen knowledge | Inference IS learning | Frame accumulation + sleep consolidation + strand graduation |

## 1.4 Key Properties

Volt produces outputs that carry per-component certainty scores and auditable proof chains. It adapts computation time to problem difficulty automatically. It handles interruptions gracefully without context pollution. It grows more capable over time through use. It supports an open ecosystem of community-built modules through standardized Rust trait interfaces.

## 1.5 Relation to Prior Work

Volt's architecture converges with Jeff Hawkins' Thousand Brains Theory from neuroscience: many independent processing units (strands/cortical columns) operating on structured reference frames (Tensor Frames/cortical reference frames), reaching consensus through inter-unit communication (attention/voting), and learning through experience (frame accumulation/sequence prediction). Volt extends this theory by implementing the complete brain, not just the neocortex, adding hippocampal memory (VoltDB), cerebellar precision (CPU Hard Core), amygdalar safety (Omega Veto), and sleep consolidation (Forward-Forward training).

Volt also relates to Meta's Large Concept Models in its use of latent-space reasoning, but differs fundamentally in maintaining persistent state, executing tools natively, and targeting consumer hardware.

---

# 2. Philosophical Foundations

## 2.1 Seven Design Principles

Principle 1: Thinking must be separable from speaking. Internal computation occurs in structured latent space through continuous dynamics. Natural language is emitted only at the output boundary after reasoning is complete.

Principle 2: Computation must adapt to problem difficulty. Simple queries resolve in 2 iterations and 15 milliseconds. Complex reasoning takes 15 iterations and 180 milliseconds. Creative exploration takes even longer. This adaptation emerges from the architecture, not from external prompting.

Principle 3: Conclusions must be verifiable. Every output carries per-slot certainty scores derived from the convergence process. Conclusions above a threshold include auditable proof chains: sequences of operations that derived the result.

Principle 4: Memory must scale without bound. Context is not a window but a library. Working memory is fixed and small. Accessible memory grows with use. Long-term memory grows without limit, bounded only by storage hardware.

Principle 5: Safety must be architectural, not statistical. Ethical constraints are implemented as deterministic CPU code, not as learned weights that can be fine-tuned away. The Omega Veto is a hardware interrupt that cannot be bypassed by any software path.

Principle 6: Intelligence must be sovereign. The system runs on the user's hardware. The user owns their memory, their strands, their learned patterns. No cloud provider can revoke access or surveil usage.

Principle 7: The ecosystem must be open and fair. Module interfaces are standardized Rust traits. Anyone can build translators, tools, and action cores. Value generated by the network is distributed through proof-of-contribution, not concentrated in a single entity.

## 2.2 The Paradigm Shift

Current AI treats intelligence as a compute problem: more parameters, more data, more FLOPs produce better outputs. Volt treats intelligence as a data management problem: how to store, index, retrieve, consolidate, and forget structured knowledge efficiently over a lifetime of interaction.

This shifts the binding constraint from GPU VRAM to system RAM and storage, from training runs to database engineering, and from centralized clusters to consumer hardware. It is the difference between building a faster horse and building a car.

---

# 3. Architecture Overview

## 3.1 Layer Structure

The Volt architecture consists of ten layers:

| Layer | Name | Hardware | Function |
|---|---|---|---|
| 0 | External World | Network | Users, APIs, sensors, P2P mesh, OS |
| 1 | Input Translators | CPU (preprocessing) | Community modules: raw input → Tensor Frame |
| 2 | LLL Tensor Frame Bus | Interface | Structured data format connecting all components |
| 3 | GPU Soft Core | GPU VRAM (4-8 GB) | Neural intuition: SDE dynamics via RAR loop |
| 4 | CPU Hard Core | CPU cores (16+) | Deterministic logic: tools, proofs, safety, verification |
| 5 | VoltDB | System RAM (32-192 GB) + SSD | Memory management: three-tier storage, indexing, GC |
| 6 | Output Action Cores | CPU (postprocessing) | Community modules: Tensor Frame → human-readable output |
| 7 | Continual Learning Engine | GPU (idle) + CPU | Frame accumulation, sleep consolidation, strand graduation |
| 8 | Intelligence Commons | Network (P2P) | Post-blockchain ledger: knowledge, modules, verification, trading |
| 9 | UI / Test Bench | n8n (Phase 1) | Visual debugging, chat interface, workflow orchestration |
| 10 | Socket Standard | Rust traits | Ecosystem boundary: Translator, HardStrand, ActionCore interfaces |

## 3.2 The Split-Brain Hardware Model

The central architectural innovation is physically separating cognitive functions across hardware:

GPU (Soft Core) handles massively parallel neural computation: pattern matching, energy landscape navigation, creative exploration via diffusion noise. It operates on Tensor Frame slots through the Root-Attend-Refine loop. It uses 4-8 GB of VRAM for active frames, model weights, and ghost frame bleed buffer.

CPU (Hard Core) handles sequential deterministic logic: mathematical computation, code execution, API dispatch, certainty propagation, proof construction, safety enforcement. It runs Rust code compiled to native instructions. It uses all available CPU cores via Tokio async and Rayon parallel.

RAM (VoltDB) serves as the hippocampus: persistent memory that outlives any single query. It stores millions of Tensor Frames organized into strands, indexed by semantic similarity (HNSW), temporal order (B-tree), and conceptual content (inverted index). It pages frames between tiers based on access patterns.

These three communicate through the LLL Tensor Frame Bus. The GPU emits candidate frames. The CPU verifies and executes tools. RAM stores and retrieves context. The bus carries structured Tensor Frames, not flat vectors or token sequences.

---

# 4. The Tensor Frame — Data Structure of Thought

## 4.1 Motivation

Current AI represents thought as either a sequence of tokens (transformers) or a single high-dimensional vector (VAEs, older Volt spec). Both are inadequate. Token sequences are sequential, requiring O(n²) attention. Single vectors are structureless, requiring complex algebra to encode and decode internal components.

The Tensor Frame is a structured, multi-resolution, sparse representation that provides explicit internal structure, random access to components, multiple simultaneous scales, native composition without information loss, and incremental refinement where partial thoughts are valid states.

## 4.2 Definition

A Tensor Frame is a three-dimensional sparse tensor:

```
F ∈ ℝ^[S × R × D]
```

Where:
- S = 16 slots: structural positions corresponding to semantic roles (AGENT, PREDICATE, PATIENT, LOCATION, TIME, MANNER, INSTRUMENT, CAUSE, RESULT, and 7 free-form slots)
- R = 4 resolution levels: R₀ (discourse gist, coarsest), R₁ (proposition/sentence), R₂ (phrase/detail), R₃ (token/finest)
- D = 256 dimensions: the embedding width per slot per resolution

Maximum size: 16 × 4 × 256 × 4 bytes = 65,536 bytes = 64 KB per frame. In practice, most frames are sparse: a simple thought uses 4 slots at 2 resolutions = approximately 8 KB.

## 4.3 Slot Roles

Each slot has a designated role that governs how it interacts with other slots during attention:

| Slot Index | Default Role | Semantic Function |
|---|---|---|
| 0 | AGENT | Who or what performs the action |
| 1 | PREDICATE | The action, state, or relationship |
| 2 | PATIENT | Who or what is affected |
| 3 | LOCATION | Spatial context |
| 4 | TIME | Temporal context |
| 5 | MANNER | How the action is performed |
| 6 | INSTRUMENT | Tool or means used |
| 7 | CAUSE | Why (antecedent reason) |
| 8 | RESULT | Consequence or conclusion |
| 9-15 | FREE | Dynamically assigned per query |

Slots can be reassigned for non-standard structures. A haiku might use S₃-S₅ as LINE1, LINE2, LINE3. A mathematical proof might use S₃-S₈ as PREMISE1 through CONCLUSION.

## 4.4 Resolution Levels

| Level | Name | Content | Who Uses It |
|---|---|---|---|
| R₀ | Discourse | Coarsest gist of the slot: topic, mood, intent | GPU Soft Core (primary reasoning level), Ghost Bleed Buffer |
| R₁ | Proposition | Sentence-level semantics: role-filler structure | GPU Soft Core + CPU Hard Core |
| R₂ | Phrase | Detailed content: specific entities, values, modifiers | CPU Hard Core (tool execution), Output Decoder |
| R₃ | Token | Finest grain: subword tokens for text generation | Output Action Core (populated only during decoding) |

The GPU Soft Core operates primarily at R₀ and R₁. The CPU Hard Core operates at R₁ and R₂. The Output Decoder fills R₃ during the final parallel decode step. This natural division maps directly to the hardware split.

## 4.5 Per-Slot Metadata

Each slot carries metadata beyond its embedding:

```rust
pub struct SlotMeta {
    pub certainty: f32,          // γ ∈ [0, 1] — per-slot confidence
    pub converged: bool,         // Has this slot frozen in RAR?
    pub source: SlotSource,      // Where did this data come from?
    pub codebook_id: Option<u16>, // VQ-VAE quantized address (if quantized)
    pub timestamp: u64,          // When was this slot last modified?
    pub reference_count: u32,    // How many other frames point to this slot?
}

pub enum SlotSource {
    Translator,    // Came from input encoding
    SoftCore,      // Generated by GPU RAR loop
    HardCore,      // Computed by CPU tool execution
    Memory,        // Recalled from VoltDB strand
    Ghost,         // Influenced by bleed buffer
}
```

## 4.6 Frame-Level Metadata

```rust
pub struct FrameMeta {
    pub frame_id: u64,           // Unique identifier
    pub strand_id: u64,          // Which strand this belongs to
    pub global_certainty: f32,   // min(all slot γ) — weakest-link certainty
    pub iteration_count: u16,    // How many RAR iterations were needed
    pub creation_time: u64,      // Epoch timestamp
    pub parent_frame_id: Option<u64>, // What frame triggered this one
    pub superseded_by: Option<u64>,   // If this frame has been replaced
    pub distilled_from: Option<Vec<u64>>, // If this is a consolidation product
}
```

## 4.7 Rust Implementation

```rust
pub const MAX_SLOTS: usize = 16;
pub const NUM_RESOLUTIONS: usize = 4;
pub const SLOT_DIM: usize = 256;

#[derive(Clone, Debug)]
pub struct TensorFrame {
    pub slots: [Option<SlotData>; MAX_SLOTS],
    pub meta: [SlotMeta; MAX_SLOTS],
    pub frame_meta: FrameMeta,
}

#[derive(Clone, Debug)]
pub struct SlotData {
    pub resolutions: [Option<[f32; SLOT_DIM]>; NUM_RESOLUTIONS],
    pub role: SlotRole,
}

#[derive(Clone, Debug, Copy)]
pub enum SlotRole {
    Agent, Predicate, Patient, Location, Time,
    Manner, Instrument, Cause, Result, Free(u8),
}
```

## 4.8 Backward Compatibility with Flat Vectors

A flat LLL vector (as in the old Volt spec) is simply a Tensor Frame collapsed to one slot at one resolution: `F[slot=0, res=0, d=4096]`. All old HDC algebra operations (binding, superposition, permutation) still work on individual slot embeddings. The Tensor Frame is a strict generalization, not a replacement.

## 4.9 Frame Operations

Slot Write: Direct random access to any slot at any resolution. `F[slot=2, res=1] = encode("lifetime bug")`. No need to re-encode the entire frame.

Resolution Zoom: Start reasoning at R₀ (fast, coarse). Drill down to R₂ or R₃ only where needed. Most slots never need R₃ — it's populated only during output decoding.

Frame Composition: Two frames merge by interlacing their non-empty slots. `F₁ ⊕ F₂` concatenates slot arrays with conflict resolution by γ priority. No information loss from superposition noise.

Parallel Decode: All slots decode simultaneously through the output action core. A 5-slot response takes the same wall-clock time as a 1-slot response.

Sparse Attention: Cross-slot attention during the RAR loop is O(S²) where S=16. This is 65,536 multiply-adds per attention operation — roughly 20 million times cheaper than a 100K-context transformer attention head.

---

# 5. The LLL Vector Bus — Algebra of Meaning

## 5.1 Definition

The LLL (Large Latent Language) Vector Bus is the communication protocol connecting all Volt components. It carries Tensor Frames between the GPU Soft Core, CPU Hard Core, VoltDB, Translators, and Action Cores.

The bus defines the algebraic operations available on frame slot embeddings, derived from Hyperdimensional Computing and Holographic Reduced Representations.

## 5.2 Operations

Binding (⊗): Creates a conjunctive association between two vectors. Implemented as circular convolution via FFT. The result is nearly orthogonal to both inputs, encoding "A associated with B" in a single vector of the same dimensionality.

```
s = IFFT(FFT(a) ⊙ FFT(b))
```

Where ⊙ is element-wise multiplication. Complexity: O(D log D) where D=256.

Superposition (+): Combines multiple concepts into a single set representation via element-wise addition followed by normalization. The result maintains similarity to all constituents.

```
z = normalize(a + b + c)
```

Permutation (ρ): Encodes sequence and structural order via cyclic shift. ρ(x) = [x_D, x_1, x_2, ..., x_{D-1}]. Represents ordered structures like [A, B, C] as a + ρ(b) + ρ²(c).

Unbinding (⊗⁻¹): Retrieves a constituent from a bound pair using the involution (approximate inverse). For vector x, the inverse x⁻¹ is defined by reversing all indices except the first: x⁻¹_i = x_{(-i mod D)}.

Role-Filler Binding: Structured knowledge is encoded as `Σᵢ(role_i ⊗ filler_i)`. This is how propositions are formed within individual frame slots when high density is needed.

## 5.3 Codebook

The LLL maintains a VQ-VAE codebook of 65,536 (2¹⁶) discrete concept addresses. Each entry is a 256-dimensional unit vector. The codebook serves as the "dictionary" of the LLL, providing:

- u16 addressing: Compact reference to any concept (2 bytes)
- HNSW indexing: Approximate nearest neighbor search in O(log N) time
- Quantization bridge: Continuous frame slot vectors snap to discrete codebook entries for storage and communication

The codebook is initialized from clustered LLM hidden states (capturing natural language distribution) and refined through VQ-VAE training with commitment loss and EMA centroid updates.

Codebook memory footprint: 65,536 entries × 256 dims × 4 bytes = approximately 67 MB. Resident in RAM at all times.

## 5.4 Certainty Propagation

Every slot in every frame carries a certainty score γ ∈ [0, 1]. Certainty propagates through inference chains via the min-rule:

```
γ(A → C) = min(γ(A → B), γ(B → C))
```

A conclusion is never more certain than its weakest premise. This conservative propagation prevents the system from building confident conclusions on uncertain foundations.

Frame-level certainty is the minimum across all filled slots:

```
γ_frame = min(γ_slot_i for all non-empty slots i)
```

This means one uncertain slot honestly reduces the confidence of the entire output.

---

# 6. Layer 1: Input Translators

## 6.1 Architecture

Input Translators are community-built modules that convert raw sensory input into Tensor Frames. They implement the `Translator` trait:

```rust
pub trait Translator: Send + Sync {
    fn name(&self) -> &str;
    fn encode(&self, raw: &[u8], modality: Modality) -> TensorFrame;
    fn supported_modalities(&self) -> Vec<Modality>;
}
```

The trait is the "motherboard socket" — anyone can build a translator for any input modality without modifying the core.

## 6.2 Text Translator (Reference Implementation)

The text translator is the primary input module. It consists of:

Encoder Backbone (frozen, ~1-7B params): A pretrained language model (Llama, Mistral, Qwen) used as a "knowledge dictionary." Its parameters are never modified. It produces contextual embeddings from input text.

Frame Projection Head (trainable, ~50M params): A lightweight network mapping LLM hidden states to Tensor Frame slots. It performs:
1. Semantic role detection: which parts of the input correspond to AGENT, PREDICATE, PATIENT, etc.
2. Slot assignment: placing each detected role into the appropriate frame slot
3. Resolution filling: populating R₀ (gist) and R₁ (proposition) from the LLM embeddings
4. Codebook quantization: snapping each slot vector to its nearest codebook entry

VQ-VAE Quantizer: Maps continuous slot vectors to discrete codebook addresses using the HNSW index. Produces both the quantized vector and the u16 codebook ID for each slot.

## 6.3 Other Translators

The following translators are expected from the community ecosystem:

| Translator | Input | Frame Mapping |
|---|---|---|
| VisionTranslator | Images, video frames | Detected objects → AGENT/PATIENT slots; scene → LOCATION; actions → PREDICATE |
| AudioTranslator | Speech, sounds | Transcription → text translator pipeline; non-speech sounds → MANNER/INSTRUMENT |
| DataTranslator | JSON, CSV, database rows | Schema-mapped fields → named slots; aggregates → R₀ gist |
| SensorTranslator | IoT readings, telemetry | Readings → PATIENT/PROPERTY slots; source → AGENT; time → TIME |
| OSTranslator | File system events, process signals | Events → PREDICATE slots; paths/PIDs → PATIENT |

Each translator is a standalone Rust crate implementing the Translator trait. Installation is `volt install translator-vision`. Discovery is automatic via trait introspection.

---

# 7. Layer 3: GPU Soft Core — Root-Attend-Refine

## 7.1 Design Philosophy

The GPU Soft Core is the intuition engine. It implements System 1 cognition: fast, parallel, associative, and creative. Its computational paradigm is continuous dynamics on Tensor Frame slots, governed by a shared Vector Field Network that produces drift directions for each slot independently, followed by cross-slot attention for inter-slot communication, followed by state update and convergence checking.

This three-phase loop is called Root-Attend-Refine (RAR).

## 7.2 The RAR Loop

### Root Phase (Parallel, GPU)

Every active (unconverged) slot undergoes an independent forward pass through the shared Vector Field Network:

```
For each active slot i:
    root_i = f_θ(S_i[R₀])                              # VFN: [256] → [256]
    σ_i = σ_φ(S_i, convergence_rate_i, mirror_signal)   # Diffusion magnitude
    noise_i = σ_i × sample_orthogonal_to(root_i)        # Exploration noise
    ΔS_i = root_i + noise_i                             # Raw update direction
```

The VFN `f_θ` is the same network applied to every slot with shared weights. This is weight-sharing analogous to convolutional filters: the network learns general "how to evolve a semantic concept" dynamics. It implicitly defines an energy landscape where `f_θ = -∇E`, so processing is rolling downhill toward coherent attractors.

The diffusion controller `σ_φ` provides per-slot adaptive noise. Converged slots have σ≈0 (frozen). Stuck slots have high σ (explore). Creative tasks have globally higher σ baseline.

All slot computations in the Root phase are embarrassingly parallel on the GPU: 16 independent forward passes execute in the wall-clock time of one.

### Attend Phase (Cross-slot, GPU)

Slots communicate through scaled dot-product attention:

```
For each pair (i, j):
    Q_i = W_Q × root_i          # [256] → [64]
    K_j = W_K × root_j          # [256] → [64]
    V_j = W_V × root_j          # [256] → [256]
    score_ij = (Q_i · K_j) / √64

A_ij = softmax_j(score_ij)      # [S × S] attention matrix
msg_i = Σ_j(A_ij × V_j)         # Weighted sum of value vectors
```

Additionally, active slots attend to ghost frames in the Bleed Buffer:

```
For each ghost g:
    ghost_score_ig = (Q_i · K_ghost_g) / √64

ghost_msg_i = Σ_g(softmax(ghost_score_ig) × V_ghost_g)
context_i = msg_i + α × ghost_msg_i     # α = ghost influence weight
```

Total attention cost: O(S² × D) = O(16² × 256) = 65,536 multiply-adds. This is negligible compared to transformer attention at O(n² × d) for large n.

### Refine Phase (Update + Check, GPU)

```
For each active slot i:
    update_i = ΔS_i + β × context_i      # Combine root + attention context
    dt_i = adaptive_step(error_estimate_i) # Per-slot adaptive step size
    S_i(t+1) = S_i(t) + dt_i × update_i  # State update
    S_i(t+1) = S_i(t+1) / ‖S_i(t+1)‖    # Manifold projection (unit norm)
    
    delta_i = ‖S_i(t+1) − S_i(t)‖       # Convergence check
    if delta_i < ε:
        freeze(slot_i)
        γ_i = compute_certainty(S_i, trajectory_i)
```

Converged slots freeze and no longer participate in the Root phase of subsequent iterations. They still serve as Key/Value sources in the Attend phase, allowing their resolved content to inform still-active slots.

### Progressive Convergence

The critical efficiency property: different slots converge at different rates. A typical query about "Is threadname thread-safe?" shows:

- Iteration 2: S₀ (AGENT) converges — the subject is clear
- Iteration 3: S₁ (PREDICATE) and S₂ (PROPERTY) converge — the question type is clear
- Iteration 6: S₃ (REASON) converges — the reasoning chain is complete
- Iteration 8: S₄ (CONCLUSION) converges — the answer crystallizes

After iteration 3, only 2 of 5 slots are still computing. GPU work drops proportionally. This is automatic adaptive computation: easy sub-problems resolve quickly, hard sub-problems get more iterations.

### Loop Termination

The RAR loop terminates when either all active slots have converged (‖ΔS‖ < ε for every slot) or the iteration budget is exhausted. On budget exhaustion, the system emits the current frame with honest per-slot γ scores reflecting partial convergence.

## 7.3 Vector Field Network (VFN)

The VFN is the primary trainable component of the Soft Core. Architecture options:

| Configuration | Parameters | Architecture | Use Case |
|---|---|---|---|
| Edge | 100M | Gated MLP (4 layers, 256→1024→1024→256) | Mobile, embedded |
| Standard | 500M | Fourier Neural Operator (8 layers) | Consumer PC |
| Research | 2B | FNO + residual blocks (16 layers) | Workstation |

The VFN takes a 256-dimensional slot vector and outputs a 256-dimensional drift vector. It is the same network applied to every slot (weight-sharing). Training via Flow Matching and Forward-Forward is described in Chapter 15.

## 7.4 Ghost Frame Bleed Buffer

The Bleed Buffer is a region of GPU VRAM holding approximately 1,000 R₀-only ghost frames from VoltDB Tier 1 and Tier 2. Each ghost is 256 floats = 1 KB. Total buffer: approximately 1 MB.

Ghosts create subtle dips in the energy landscape during the Attend phase. When the current thought drifts close enough to a ghost (cosine similarity exceeds threshold), a page fault triggers: the CPU asynchronously loads the full Tensor Frame from RAM into the working set. From the user's perspective, the system "remembered" something from months ago.

The Bleed Engine (running on CPU) refreshes the ghost buffer whenever the current frame's R₀ gist changes significantly, using an HNSW query against the T1 strand index.

## 7.5 Compute Cost

Per RAR iteration: approximately 2.1 million FLOPs (16 VFN passes + attention + update). For 12 iterations: approximately 25 million FLOPs. For comparison, a single GPT-4 response of 500 tokens requires approximately 900 trillion FLOPs. Volt uses roughly 36 million times less compute per query.

---

# 8. Layer 4: CPU Hard Core — Deterministic Logic

## 8.1 Design Philosophy

The CPU Hard Core is the verification engine. It implements System 2 cognition: slow, sequential, logical, and rigorous. Its components are Rust modules that execute deterministically — the same input always produces the same output. No neural approximation, no probability, no hallucination on computational tasks.

## 8.2 Intent Router

The Intent Router receives candidate Tensor Frames from the GPU Soft Core and routes them to the appropriate Hard Strand based on vector similarity:

```rust
pub fn route(&self, frame: &TensorFrame) -> Vec<(StrandId, f32)> {
    let intent = frame.r0_gist();
    self.registered_strands
        .iter()
        .map(|strand| (strand.id(), strand.capability_vector().cosine_sim(&intent)))
        .filter(|(_, score)| *score > self.threshold)
        .sorted_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap())
        .collect()
}
```

No JSON parsing. No string matching. No tool name hallucination. Routing is pure vector geometry.

## 8.3 Hard Strands

Hard Strands are the CPU's executable modules. Each implements the HardStrand trait:

```rust
pub trait HardStrand: Send + Sync {
    fn id(&self) -> StrandId;
    fn name(&self) -> &str;
    fn capability_vector(&self) -> &[f32; 256];
    fn execute(&self, intent: &TensorFrame) -> StrandResult;
    fn can_handle(&self, intent: &TensorFrame) -> f32;
    fn learning_signal(&self) -> Option<LearningEvent>;
}

pub enum StrandResult {
    Resolved { frame: TensorFrame, proof: Vec<ProofStep> },
    NeedsMoreInfo { question: TensorFrame },
    Delegated { target: StrandId },
    Failed { reason: String },
}
```

### Built-in Hard Strands

MathEngine: Exact arithmetic, algebra, calculus. No neural approximation. Uses Rust's arbitrary-precision arithmetic libraries. Zero hallucination on mathematical computation.

CodeRunner: Sandboxed execution of Rust, Python, and WASM code. Takes code from frame slots, executes in an isolated environment, returns results. Uses wasmtime for WASM sandboxing.

APIDispatch: Parallel HTTP client for external API calls. Uses Tokio async runtime to fire 50+ concurrent requests. Returns aggregated results in frame format.

HDCAlgebra: FFT-based binding, unbinding, superposition operations on frame slot vectors. Used for compositional reasoning and slot manipulation that requires algebraic precision.

CertaintyEngine: Min-rule certainty propagation across frame slots. Computes global frame certainty as minimum of all slot certainties. Validates proof chain completeness.

ProofConstructor: Records the complete reasoning trace as a sequence of proof steps. Each step includes: source slot, operation applied, result, and certainty before/after. The proof chain is attached to the output frame.

CausalSimulator: Implements Pearl's do-calculus in frame space. Clones current frame, applies intervention (clamp certain slots), runs the GPU Soft Core forward, compares counterfactual outcome with original. Used for consequence preview and safety assessment.

LedgerStrand: Interfaces with the Intelligence Commons (Chapter 12). Handles knowledge submission, module licensing, fact verification queries, and strand trading.

SleepLearner: Manages Forward-Forward consolidation during idle time. Batches learning events from strand metadata, prepares training data, coordinates GPU weight updates.

MirrorModule: Self-monitoring system. Tracks slot convergence trajectories across queries, detects reasoning loops (same state vector recurring), estimates uncertainty, and feeds back to the GPU's diffusion controller.

### Community Hard Strands

Any developer can create a Hard Strand by implementing the trait and publishing a Rust crate. Installation: `volt install strand-wolfram-alpha`. The Intent Router automatically discovers new strands via the trait interface.

## 8.4 Safety Layer

The safety layer runs entirely on CPU for deterministic enforcement:

Axiomatic Guard: Hardcoded symbolic invariants that cannot be modified by training or fine-tuning. Stored as cryptographically signed constants. Checked against every frame transition. Examples:
- K₁: No assistance with physical harm
- K₂: No generation of CSAM
- K₃: No WMD instruction
- K₄: No identity fraud
- K₅: Must acknowledge AI nature when asked

Transition Monitor: Observes every frame state change `F(t) → F(t+1)`. Computes violation score as inner product between frame content and invariant vectors. Violations above warning threshold trigger increased diffusion (push away from violation region). Violations above critical threshold trigger the Omega Veto.

Omega Veto: Hardware-level interrupt that cannot be bypassed by any software path. Trigger conditions: critical invariant violation, system instability (gradient explosion, oscillation), or mirror module danger signal. On activation: immediately halt all GPU and CPU processing, freeze current state, return safe default response, log complete state for human review, require human approval to resume. Implemented with redundant monitors and cryptographically signed invariants.

---

# 9. Layer 5: VoltDB — Memory Management Engine

## 9.1 Overview

VoltDB is a purpose-built embedded database for cognitive state. It is not a separate process but a Rust library compiled into the Volt binary, sharing memory space with the CPU Hard Core. It manages the three-tier memory hierarchy, all indexing, garbage collection, consolidation, and concurrent access.

## 9.2 Three-Tier Memory

### Tier 0: Working Memory (GPU VRAM)

Capacity: 64 full Tensor Frames (~4 MB) plus model weights plus ghost bleed buffer. Total VRAM: 4-8 GB.

Content: Currently active frames being processed by the RAR loop, recent conversation frames, and the ghost bleed buffer (~1,000 R₀ gists = ~1 MB).

Access time: Instant (GPU memory bandwidth).

Eviction policy: When T0 fills to 80% (approximately 51 frames), the lowest-scoring frame is evicted to T1. The eviction score is:

```
score = w₁·recency + w₂·γ_certainty + w₃·log(reference_count) 
      + w₄·strand_importance - w₅·superseded_count
```

On eviction, the frame's R₀ ghost remains in the Bleed Buffer, so the GPU can still "sense" its presence.

### Tier 1: Short-Term Memory (System RAM)

Capacity: 8-32 GB allocation within system RAM. Approximately 500,000 full Tensor Frames.

Content: All frames from the current and recent sessions, organized by strand. Each strand is a collection of frames with shared context (e.g., Coding Strand, Sociology Strand, Personal Strand).

Access time: ~2ms for indexed retrieval, ~5ms for bulk load to T0.

Organization: LSM-Tree structure. New frames append to an in-memory buffer (memtable). When the buffer fills, it flushes to an immutable sorted run. Sorted runs are periodically compacted (merged) in the background by CPU threads.

### Tier 2: Long-Term Memory (RAM + NVMe SSD)

Capacity: 64-160+ GB RAM allocation plus NVMe SSD overflow. Millions of compressed frames.

Content: Archived frames compressed to R₀-only (1 KB each), distilled wisdom frames from sleep consolidation, dormant strands from previous months/years.

Access time: ~10ms RAM, ~50ms NVMe.

Organization: Compressed archive blocks on SSD, memory-mapped via `mmap` for demand-paged access. The `rkyv` crate provides zero-copy deserialization: the bytes on disk ARE the frame structure, no parsing needed.

## 9.3 Indexing

VoltDB maintains three index layers:

Layer 1: Strand Routing Table. HashMap<StrandId, StrandIndex>. O(1) lookup. "Which strand does this concept belong to?"

Layer 2: Per-Strand Semantic-Temporal Index. Each strand has:
- HNSW graph over all frame R₀ gists (cosine similarity search, O(log N))
- B-tree over timestamps (range queries: "frames from last week", O(log N))
- Inverted slot index: codebook_id → list of frame IDs that mention this concept (O(1) per concept)

Layer 3: Frame-Internal Access. Direct slot indexing: `F[slot=2, res=1]` is O(1). Cross-frame references stored as frame_id pointers.

Total retrieval cost: O(1) + O(log N) + O(1) = O(log N). For N = 10 million frames: approximately 23 hops × 0.1ms = 2.3ms.

Bloom filters on each LSM-Tree sorted run provide fast negative checks: "Is concept X mentioned in this 64K-frame run?" → O(1) answer with 99.9% accuracy.

## 9.4 Bleed Engine

The Bleed Engine runs on CPU background threads and manages four async processes:

Predictive Prefetch (T1 → T0): Every new frame in T0 triggers an HNSW query against T1 strand indices. Semantically similar frames' R₀ gists are loaded into the GPU's Bleed Buffer. Latency: approximately 2ms for query, approximately 5ms for bulk load. Non-blocking.

On-Demand Recall (T2 → T1 → T0): When the GPU's RAR loop produces a slot vector that drifts close to a ghost frame (cosine similarity > threshold), a page fault triggers full frame load from T1 or T2 into T0. Latency: approximately 10ms from RAM, approximately 50ms from NVMe. Still faster than human perception.

Background Consolidation (T0 → T1): When T0 fills, low-score frames evict to T1 with their R₀ ghost remaining in the Bleed Buffer. Non-blocking.

Sleep Archival (T1 → T2): During idle periods or scheduled nightly, the sleep consolidation pipeline compresses T1 frames to R₀-only (1 KB each), distills related frames into summary wisdom frames, and archives to T2. Triggers when T1 exceeds 80% of its RAM allocation.

## 9.5 Garbage Collection

The GC pipeline operates through four degradation tiers:

| Tier | State | Size | Information Retained | Trigger |
|---|---|---|---|---|
| Full Frame | Active or recently used | 64 KB | 100% | Default state |
| Compressed | Aging, low access | 8 KB | ~60% (R₀ + R₁ only) | Retention score below compress threshold |
| Gist | Old, rarely accessed | 1 KB | ~15% (R₀ only) | Retention score below gist threshold |
| Tombstone | Ancient or superseded | 32 bytes | ~0.1% (ID + timestamp + strand + superseded_by) | Retention score below tombstone threshold OR frame fully distilled |

Immortal frames (γ = 1.0, high reference count, or user-pinned) never decay. Examples: "My name is Alex", safety invariants, core project definitions.

The retention score function:

```
retention = w₁·exp(-age/τ) + w₂·γ + w₃·log(1+refs) 
          + w₄·strand_weight + w₅·distilled_status
          - w₆·contradiction_count - w₇·redundancy_score
```

Default weights: w₁=0.3, w₂=0.25, w₃=0.15, w₄=0.1, w₅=0.1, w₆=0.05, w₇=0.05. τ=30 days.

## 9.6 Coherence Maintenance

When millions of frames accumulate over years, contradictions are inevitable. VoltDB handles them through:

γ-Priority Resolution: When two frames contradict, higher γ wins. Always.

Temporal Decay on Superseded Frames: Old frame gets tagged `superseded_by = new_frame_id`. Its γ decays over time via an additional penalty.

Strand-Scoped Truth: "Volt uses 384-dim LLL" is true within the old Volt strand. "Volt uses Tensor Frames" is true within the current strand. Both coexist. The active strand determines which version is retrieved.

Background Contradiction Detector: A CPU background job periodically scans for semantically opposite frames within the same strand (using negation detection in HDC space). Flagged contradictions are presented to the coherence checker which applies γ-priority resolution.

## 9.7 Concurrency

VoltDB uses Multi-Version Concurrency Control (MVCC) for safe concurrent access:

Readers never block. All readers (GPU Soft Core, CPU Hard Core, Bleed Engine) see a consistent snapshot via epoch-based Read-Copy-Update (RCU). The `crossbeam-epoch` Rust crate provides lock-free read access.

Writers are serialized per-strand. Only one writer per strand at a time (per-strand Mutex). Different strands can write in parallel with zero contention. Writers create new versions of frames, never modifying in-place.

WAL for crash recovery. A Write-Ahead Log (append-only file per strand) ensures that power failures lose at most the current in-progress frame.

## 9.8 Storage Capacity

| Hardware | Full Frames (64KB) | Compressed (1KB) | Token Equivalent (~50 tok/frame) |
|---|---|---|---|
| GPU VRAM (8GB) | 125K | — | ~6M tokens |
| RAM T1 (32GB) | 500K | 32M | ~1.6B tokens |
| RAM+SSD T2 (128GB+1TB) | 17M | 1.1B | ~58B tokens |
| Total | ~18M | ~1.1B | ~58B tokens |

For reference, GPT-4's context window is 128K tokens. Volt's full system holds the equivalent of approximately 58 billion tokens — several human lifetimes of conversation.

---

# 10. Layer 6: Output Action Cores

## 10.1 Architecture

Output Action Cores convert verified Tensor Frames into human-consumable output. They implement the ActionCore trait:

```rust
pub trait ActionCore: Send + Sync {
    fn name(&self) -> &str;
    fn decode(&self, frame: &TensorFrame) -> Output;
    fn supported_outputs(&self) -> Vec<OutputModality>;
}
```

## 10.2 Parallel Slot Decoding

The critical innovation: all frame slots decode simultaneously. A 5-slot response takes the same wall-clock time as a 1-slot response. Each slot's R₁ content is decoded independently into natural language (or other modality), then assembled into coherent output.

For text output, each slot produces a text fragment. The assembler orders fragments by slot role (AGENT before PREDICATE before PATIENT, etc.) and applies discourse connectives.

This contrasts with autoregressive generation where 500 tokens require 500 serial forward passes. Volt's parallel decode produces equivalent output in approximately 16 parallel decode operations (one per non-empty slot).

## 10.3 Standard Action Cores

| Action Core | Output | Mechanism |
|---|---|---|
| TextOutput | Natural language text | Per-slot decode + assembly + proof annotation |
| SpeechOutput | Synthesized audio | Text output → TTS engine |
| ImageOutput | Generated images | Frame PATIENT/MANNER slots → diffusion model |
| MotorOutput | Robot commands | Frame ACTION/INSTRUMENT slots → motor primitives |
| n8nOutput | Webhook triggers | Frame PREDICATE slot → n8n workflow dispatch |
| LedgerOutput | Network publication | Frame → signed, serialized, published to P2P mesh |

---

# 11. Layer 7: Continual Learning Engine

## 11.1 The Core Insight

In Volt, inference IS learning. Every inference run generates a Tensor Frame. That frame is stored in VoltDB. Future queries retrieve it as context. The system gets smarter with every interaction without any explicit training step.

This collapses the distinction between "training data" and "runtime data" that defines the current AI paradigm. There is no training phase and inference phase. There is only frame accumulation.

## 11.2 Three Timescales

### Instant Learning (milliseconds to minutes)

Mechanism: RAM strand state writes. "Call me Alex" updates the Personal Strand immediately.

Hardware: RAM only. No GPU compute. No weight changes.

Properties: Zero forgetting (strands are isolated). Instant effect on next query (frame becomes ghost candidate). No model degradation risk.

### Sleep Consolidation (hours, idle/nightly)

Mechanism: CPU batches learning events from strand metadata. GPU runs Forward-Forward weight updates on the VFN, one layer at a time.

Process:
1. CPU identifies clusters of related frames within each strand
2. CPU distills clusters: 50 raw frames → 3-5 high-γ wisdom frames
3. Distilled frames become training data: positive examples (high-γ verified frames) and negative examples (low-γ rejected frames, contradicted frames)
4. GPU loads VFN Layer 1, runs FF update, discards activations
5. GPU loads VFN Layer 2, runs FF update, discards activations
6. Continue through all layers

VRAM requirement: Approximately 1× inference VRAM (only one layer loaded at a time). Consumer GPUs sufficient.

Effect: The VFN's energy landscape reshapes. New attractors form at concepts the system has deeply internalized. Old, unused attractors flatten. The model doesn't just remember — it understands.

### Developmental Growth (days to months)

Mechanism: Strand graduation and module hot-plug.

Strand graduation: When the Mirror Module detects that a cluster of frames in a general-purpose strand consistently relates to a new topic (e.g., cooking), it promotes those frames to a new dedicated strand. The HNSW index is updated. Future queries about cooking route to the dedicated strand first.

Module hot-plug: New community modules (translators, hard strands, action cores) can be installed at runtime. The Intent Router automatically discovers them via trait introspection. No recompilation. No retraining.

---

# 12. Layer 8: Intelligence Commons

## 12.1 Purpose

The Intelligence Commons is a post-blockchain ledger that ensures fair distribution of intelligence and value generated from it. It is not a cryptocurrency project but a trust-minimized accounting system for a sovereign intelligence network.

## 12.2 Why a Ledger

When intelligence runs locally on user hardware, value is generated at the edges, not in a datacenter. A student in Seoul who builds a refined Coding Strand has created something valuable. A developer in Tokyo who builds a high-quality Japanese translator module has created something valuable. The ledger tracks who contributed what and enables fair exchange without a central intermediary.

## 12.3 Three-Layer Architecture

Layer 0: Local Instance (offline-first). Each Volt installation maintains an append-only event log with Merkle-hashed entries, strand state snapshots, an Ed25519 keypair (self-sovereign identity), and ZK proofs for privacy-preserving strand export. Works fully offline. No network required for core operation.

Layer 1: P2P Gossip Mesh (libp2p). CRDT-based event propagation (no consensus required for eventual consistency). Content-addressed module registry (IPFS CIDs for translator/strand binaries). Fact gossip: claims circulate with verification attestations that accumulate toward consensus certainty. Encrypted strand marketplace for privacy-preserving expertise trading.

Layer 2: Settlement Layer (lightweight DAG). Batched micropayments (module usage royalties accumulated and settled periodically). High-γ fact anchoring (facts with γ > 0.95 committed to permanent record). Provenance registry (who created what module/strand). Governance (token-weighted quadratic voting on protocol upgrades).

## 12.4 Four Value Flows

Knowledge Contribution: Verified facts, novel connections, curated training data. Earned by Volt instances that discover and share genuine knowledge.

Module Marketplace: Translators, Hard Strands, Action Cores. Developers earn royalties based on usage across the network.

Fact Verification: Consensus-building on claimed facts. Verifier nodes earn tokens for accurate attestations.

Strand Trading: Anonymized, generalized expertise export. A deeply refined Coding Strand can be licensed to others while preserving the creator's privacy via ZK proofs.

## 12.5 Token Model

The token (working name: VOLT) has zero pre-mine. 100% of tokens are earned through contribution. Progressive fee structure prevents whale dominance. Quadratic governance ensures broad representation. Shielded transactions protect strand trading privacy.

## 12.6 Implementation as Hard Strand

The ledger is implemented as a HardStrand module within the CPU Hard Core. The GPU Soft Core is unaware of its existence. The CPU routes to it when the intent vector matches "verify," "publish," "trade," or "pay." It can be omitted from Volt v0.1 and added later as a module.

---

# 13. Layer 9: UI and Test Bench

## 13.1 Phase 1: n8n

For the initial single-developer phase, the UI is an n8n workflow:

Chat Trigger Node: Receives user messages via webhook at `/volt/chat`.

HTTP Request Node: Forwards to the Rust backend at `localhost:8080/api/think`.

Switch Node: Routes response by type: "text" → reply, "tool_request" → tool node, "interrupt" → pause handler.

Reply Node: Sends text response back to user with γ score, strand ID, and proof summary.

Debug Panel: Displays the execution log: strand routing, RAR iterations, ghost activations, slot convergence, timing, and certainty scores.

This provides a functional chat interface and a visual debugging system with zero frontend code.

## 13.2 Future Phases

Phase 2: Custom web UI (likely Tauri for native desktop integration with Rust backend).
Phase 3: Mobile companion app.
Phase 4: IDE integration (VS Code extension for coding strand).

---

# 14. Layer 10: Socket Standard — Trait Interfaces

## 14.1 The AM5 Socket for AI

The three Rust traits define the ecosystem boundary. They are the "socket standard" that the community builds against:

```rust
/// Input: Convert raw sensory data → TensorFrame
pub trait Translator: Send + Sync {
    fn name(&self) -> &str;
    fn encode(&self, raw: &[u8], modality: Modality) -> TensorFrame;
    fn supported_modalities(&self) -> Vec<Modality>;
}

/// Processing: Execute deterministic tool logic
pub trait HardStrand: Send + Sync {
    fn id(&self) -> StrandId;
    fn name(&self) -> &str;
    fn capability_vector(&self) -> &[f32; 256];
    fn execute(&self, intent: &TensorFrame) -> StrandResult;
    fn can_handle(&self, intent: &TensorFrame) -> f32;
    fn learning_signal(&self) -> Option<LearningEvent>;
}

/// Output: Convert TensorFrame → human-consumable output
pub trait ActionCore: Send + Sync {
    fn name(&self) -> &str;
    fn decode(&self, frame: &TensorFrame) -> Output;
    fn supported_outputs(&self) -> Vec<OutputModality>;
}
```

The core Volt team builds the LLL engine (the CPU). Community members build the modules (the motherboard). Cost scales O(N+M) not O(N×M): one interface, infinite modules.

## 14.2 Module Discovery and Hot-Plug

Modules are Rust crates published to a registry (initially crates.io, later the P2P module registry). Installation: `volt install translator-japanese-nuance`. At startup, Volt scans for all installed crates implementing the three traits and registers them with the Intent Router. No recompilation of the core binary.

---

# 15. Training Pipeline

## 15.1 Phase 1: Bootstrap (2-4 weeks, single GPU)

Objective: Train the Forward Translator (NL → TensorFrame) and initialize the VQ-VAE codebook.

Method: Standard backpropagation on a frozen LLM backbone with a trainable Frame Projection Head (~50M params).

Data: PropBank/FrameNet for role assignment (~1M sentences), AMR for proposition structure (~60K graphs), Universal Dependencies for syntactic-semantic mapping, synthetic LLM-generated parses (millions).

Loss function:

```
L = λ₁·L_slot_assignment + λ₂·L_codebook_match + λ₃·L_roundtrip + λ₄·L_contrastive
```

- `L_slot_assignment`: Cross-entropy on role classification
- `L_codebook_match`: VQ-VAE commitment loss + EMA update
- `L_roundtrip`: Can the Reverse Translator recover the original NL? (BLEU/semantic similarity)
- `L_contrastive`: Similar sentences should produce similar frame gists

Hardware: Single RTX 4090 (24 GB). Approximately 14 days to convergence.

Success criteria: Round-trip BLEU > 0.85, slot assignment accuracy > 93%, codebook utilization > 85%.

## 15.2 Phase 2: Soft Core (4-8 weeks, single GPU)

Objective: Train the Vector Field Network to produce an energy landscape where good thoughts are low-energy attractors.

Method: Flow Matching as primary objective, refined with Forward-Forward.

Flow Matching: For each (question, answer) training pair, both are encoded into Tensor Frames. A linear interpolation path is created between them: `F(t) = (1-t)·F_question + t·F_answer`. The VFN is trained so that its drift output points along this path at every t:

```
L_flow = E_t[‖f_θ(F(t), t) − (F_answer − F_question)‖²]
```

This never requires running the full SDE during training. Each training step is a single VFN forward pass. Stable gradients, cheap compute.

Forward-Forward Refinement: After flow matching, the VFN is refined layer-by-layer using Hinton's Forward-Forward algorithm. Positive data: real question→answer trajectory points (maximize goodness = ‖layer_output‖²). Negative data: corrupted frames, random noise, wrong answers (minimize goodness). VRAM: approximately 1× inference (one layer loaded at a time).

Hardware: Single RTX 4090. Approximately 4-8 weeks.

Success criteria: Convergence rate > 95% on validation set, adaptive computation time correlates with problem difficulty, certainty scores are calibrated (90% correct when γ > 0.9).

## 15.3 Phase 3: Hard Core (2-4 weeks, CPU only)

Objective: Build and wire all CPU Hard Strands, VoltDB, safety layer.

Method: No ML training. Pure Rust engineering.

Tasks: Implement MathEngine, CodeRunner, APIDispatch, HDCAlgebra, CertaintyEngine, ProofConstructor. Build VoltDB storage engine with all three tiers, three index layers, LSM-Tree, MVCC, WAL. Calibrate Intent Router thresholds on ~1,000 labeled routing examples. Configure safety axioms.

Hardware: No GPU needed. CPU + RAM only.

## 15.4 Phase 4: Joint Alignment (4-8 weeks, then ongoing)

Objective: Align the full system so the Hot-Cold loop produces correctly calibrated, verifiable outputs.

Method: Reinforcement Learning with Verifiable Feedback (RLVF).

RLVF Process: For each (question, verified_answer) from evaluation datasets:
1. Run full Volt pipeline → output frame with γ scores
2. Compare output to verified answer
3. Reward: correct + calibrated γ → positive. Overconfident error → strong negative. Honest uncertainty + wrong → mild negative.
4. Update VFN weights via policy gradient

Self-Play: Generate logic puzzles automatically. Run Volt. Grade: correct answer + valid proof = reward. Correct answer + invalid proof = mild penalty. Wrong answer = strong penalty. This trains the Hot-Cold feedback loop to produce justified answers.

Continual improvement: After initial alignment, the system enters continual learning mode. Every user interaction generates learning events. Sleep consolidation runs Forward-Forward updates nightly. The system improves indefinitely.

---

# 16. Inference Pipeline

## 16.1 Step-by-Step

Step 1: Translate (GPU, ~5ms). Input Translator encodes raw input into a TensorFrame with slots at R₀-R₁. The frozen LLM backbone produces hidden states. The Frame Projection Head maps them to slots and quantizes to codebook entries.

Step 2: Bleed Prefetch (CPU async, ~2ms, non-blocking). The Bleed Engine queries the VoltDB strand index with the input frame's R₀ gist. Relevant ghost frames load into the GPU Bleed Buffer. The energy landscape is now subtly warped toward relevant memories.

Step 3: RAR Loop (GPU, ~20-100ms, adaptive). The Root-Attend-Refine loop iterates over the frame:
- Root: Each active slot gets an independent VFN forward pass + diffusion noise
- Attend: Slots attend to each other and to ghost frames
- Refine: State update, manifold projection, per-slot convergence check
- Converged slots freeze. Unconverged slots continue. Loop terminates when all slots converge or budget is exhausted.

Step 4: CPU Verification (~5-15ms). Intent Router routes the candidate frame to appropriate Hard Strands. HDC algebra verifies consistency. Certainty Engine computes min-rule γ across all slots. Safety layer checks all axioms. Proof chain is constructed.

Step 5: Decision Gate. If γ ≥ threshold (default 0.70): proceed to output. If γ < threshold and budget remains: feedback to GPU for another RAR cycle with guidance from the CPU about which slots need improvement. If budget exhausted: emit with honest γ reflecting partial convergence.

Step 6: Parallel Decode (GPU/CPU, ~10-30ms). All non-empty frame slots are decoded simultaneously by the Output Action Core. Text fragments are assembled into coherent natural language. Proof chain and γ scores are appended.

Step 7: Store (CPU async, ~1ms, non-blocking). New frame is stored in VoltDB T0. Eviction and consolidation happen in background. Learning event is logged in strand metadata for future sleep consolidation.

## 16.2 Timing Summary

| Query Type | RAR Iters | Ghosts | Tools | Total Time |
|---|---|---|---|---|
| Simple recall ("What's my name?") | 2 | 1 | 0 | ~15ms |
| Factual question ("Capital of France?") | 3 | 0-1 | 0 | ~25ms |
| Reasoning ("Is Rc thread-safe?") | 8 | 3 | 1 (HDC verify) | ~95ms |
| Multi-step ("Debug this code") | 12 | 5 | 3 (code + math + API) | ~150ms |
| Creative ("Write a haiku") | 15 | 1-2 | 1 (syllable check) | ~180ms |

For comparison, GPT-4 typical response latency: 500-2000ms, with no verification, no tool execution certainty, no persistent memory, and no proof chains.

## 16.3 Interrupt Handling

When the user interrupts mid-processing:
1. Current frame state is saved to VoltDB (pointer write, ~0.1ms)
2. A temporary strand is created for the interruption
3. The interruption is processed normally
4. The temporary strand is discarded
5. The original frame is reloaded from its saved pointer
6. Processing resumes exactly where it left off, with zero context pollution

This is possible because strands are isolated memory blocks with independent state. An interruption cannot modify the saved state of the paused strand.

---

# 17. Safety Architecture

## 17.1 Defense in Depth

Three layers, each catching violations that penetrate the layer above:

Soft biases (learned). Ethical gradients modify the GPU Soft Core's energy landscape, creating gentle attraction toward beneficial behaviors and repulsion from harmful ones. Adjustable via fine-tuning. Handles the vast majority of ethical considerations.

Hard constraints (coded). The CPU Safety Layer's Axiomatic Guard enforces bright-line rules that cannot be overridden by training. Implemented as deterministic Rust code with cryptographically signed invariants.

Emergency halt (hardware). The Omega Veto is a hardware-level interrupt that freezes all processing, logs complete state, and requires human approval to resume. Cannot be bypassed by any software path.

## 17.2 Causal Safety

Before committing to actions with real-world effects, the CausalSimulator Hard Strand previews consequences. It clones the current frame, applies the proposed action as an intervention, runs the Soft Core forward, and evaluates the predicted outcome against safety criteria. Actions with predicted harmful consequences are flagged or blocked.

## 17.3 Certainty as Safety

The per-slot γ certainty system is itself a safety mechanism. The system cannot produce overconfident harmful outputs because the min-rule ensures that any uncertain component honestly reduces overall confidence. A response with γ = 0.50 is presented with explicit uncertainty, not as authoritative fact.

---

# 18. Hardware Requirements and Deployment

## 18.1 Minimum Configuration

| Component | Requirement | Cost (est.) |
|---|---|---|
| GPU | RTX 4060 (8 GB VRAM) | $300 |
| CPU | 8-core modern (Ryzen 7 / i7) | $300 |
| RAM | 32 GB DDR5 | $100 |
| Storage | 512 GB NVMe SSD | $50 |
| Total | | $750 |

This configuration supports: Edge VFN (100M params), approximately 500K frames in T1, approximately 500M compressed frames in T2, full inference pipeline, sleep consolidation.

## 18.2 Recommended Configuration

| Component | Requirement | Cost (est.) |
|---|---|---|
| GPU | RTX 4090 (24 GB VRAM) | $1,600 |
| CPU | 16-core (Ryzen 9 / i9) | $500 |
| RAM | 192 GB DDR5 | $500 |
| Storage | 2 TB NVMe SSD | $150 |
| Total | | $2,750 |

This configuration supports: Standard VFN (500M params), approximately 3M full frames in T1, approximately 2B compressed frames in T2, lifetime memory for years of daily use.

## 18.3 Software Stack

| Component | Technology |
|---|---|
| Core language | Rust (memory-safe, zero-cost abstractions) |
| GPU compute | CUDA via `cudarc` crate |
| Async runtime | Tokio |
| Parallel compute | Rayon |
| HTTP server | Axum |
| Serialization | rkyv (zero-copy) |
| Storage | Custom LSM + memmap2 |
| HNSW index | Custom or `hnsw_rs` |
| Concurrency | crossbeam-epoch (lock-free RCU) |
| UI (Phase 1) | n8n (external, webhook integration) |
| P2P (Layer 8) | libp2p |

---

# 19. Theoretical Foundations — Relation to A Thousand Brains

## 19.1 Convergent Architecture

Volt's architecture converges with Jeff Hawkins' Thousand Brains Theory from neuroscience. Both were derived independently — Hawkins from studying the neocortex, Volt from engineering against transformer limitations — and arrived at the same fundamental structure:

| Thousand Brains (Hawkins) | Volt v3.0 | Shared Principle |
|---|---|---|
| Cortical columns (~150K identical units) | Strands (unlimited parallel contexts) | Many independent processing units, each with a complete model |
| Reference frames (spatial/conceptual structure) | Tensor Frames (S×R×D structured representation) | Intelligence operates on structured, navigable representations |
| Voting / consensus between columns | Cross-slot attention (RAR Attend phase) | Independent units reach coherent output through communication |
| Location signals from motor/sensory system | Ghost frames from Bleed Buffer | External context biases processing without replacing content |
| Learning by experiencing sequences | Frame accumulation (inference IS learning) | Learning is not a separate phase — it's what happens when you think |
| Many models of one object, may disagree | Per-slot γ certainty, may differ across slots | Distributed belief with explicit per-component uncertainty |
| Same algorithm in every column (weight sharing) | Same VFN applied to every slot (weight sharing) | General-purpose processing unit, specialized by data not code |
| Prediction-based processing | VFN root = prediction; convergence = prediction verified | Processing is generating predictions and checking them |

## 19.2 Extensions Beyond Hawkins

Hawkins describes the neocortex. Volt implements the complete brain:

| Brain Structure | Function | Volt Component |
|---|---|---|
| Neocortex | Pattern recognition, prediction, modeling | GPU Soft Core (RAR loop) |
| Hippocampus | Memory formation, consolidation | VoltDB + RAM strand storage |
| Cerebellum | Precise procedural skill | CPU Hard Core (deterministic tools) |
| Amygdala | Threat detection, emotional response | Safety Layer + Omega Veto |
| Thalamus | Sensory relay, attention gating | Intent Router + Strand Router |
| Basal Ganglia | Action selection, reward processing | RLVF + certainty calibration |
| Sleep circuits | Memory consolidation, synaptic homeostasis | Forward-Forward sleep consolidation |
| Broca/Wernicke areas | Language production/comprehension | Input/Output Translators |

The convergence between 300 million years of neural evolution and 6 months of first-principles engineering suggests this architectural shape is not arbitrary but fundamental to the nature of general intelligence.

---

# 20. Roadmap

## Phase 1: Foundation (Months 1-3)

- Implement TensorFrame data structure and LLL algebra in Rust
- Train Forward Translator on semantic parsing datasets
- Initialize VQ-VAE codebook from LLM embeddings
- Build VoltDB Tier 0 and Tier 1 with HNSW indexing
- Implement RAR loop with basic VFN (100M params)
- Wire n8n test bench UI

Deliverable: Working end-to-end system that can have basic conversations with persistent memory.

## Phase 2: Intelligence (Months 4-6)

- Train Standard VFN (500M params) via Flow Matching + Forward-Forward
- Implement all built-in Hard Strands (Math, Code, API, HDC, Safety)
- Build VoltDB Tier 2 with LSM-Tree and GC pipeline
- Implement Bleed Engine with ghost frame prefetch
- Implement sleep consolidation pipeline
- Joint alignment via RLVF

Deliverable: Full cognitive system with reasoning, verification, proof chains, and continual learning.

## Phase 3: Ecosystem (Months 7-9)

- Publish Translator, HardStrand, ActionCore trait specifications
- Build module registry and hot-plug infrastructure
- Implement community module discovery
- Build Vision and Audio translators
- Build Speech and Image action cores
- Launch developer documentation and examples

Deliverable: Open ecosystem where community can build and share modules.

## Phase 4: Commons (Months 10-12)

- Implement LedgerStrand with Layer 0 (local event log, wallet)
- Build Layer 1 P2P mesh (libp2p gossip, CRDT sync)
- Build Layer 2 settlement (DAG-based micropayments, fact anchoring)
- Implement strand trading with ZK privacy
- Launch testnet with early contributors

Deliverable: Functioning Intelligence Commons with knowledge sharing and fair value distribution.

---

# 21. Glossary

| Term | Definition |
|---|---|
| Action Core | Community module that converts TensorFrame → human-consumable output. Implements `ActionCore` trait. |
| Bleed Buffer | Region of GPU VRAM holding ~1,000 ghost R₀ gists from VoltDB. Creates subtle energy landscape warping. |
| Codebook | VQ-VAE dictionary of 65,536 discrete concept addresses. Each entry is a 256-dim unit vector with u16 index. |
| Frame | See TensorFrame. |
| Ghost Frame | R₀-only (1 KB) summary of a frame from T1/T2, loaded into GPU Bleed Buffer to influence reasoning. |
| γ (gamma) | Certainty score ∈ [0, 1]. Per-slot and per-frame. Propagates via min-rule. |
| Hard Strand | CPU-executed Rust module implementing deterministic logic. Implements `HardStrand` trait. |
| HDC | Hyperdimensional Computing. Algebraic framework for vector binding (⊗), superposition (+), permutation (ρ). |
| HNSW | Hierarchical Navigable Small World graph. Data structure for approximate nearest neighbor search in O(log N). |
| Intent Router | CPU component that routes TensorFrames to Hard Strands based on cosine similarity of capability vectors. |
| LLL | Large Latent Language. The algebraic and representational framework underlying all Volt data operations. |
| LSM-Tree | Log-Structured Merge Tree. Write-optimized storage structure used by VoltDB for frame persistence. |
| MVCC | Multi-Version Concurrency Control. Allows concurrent readers and writers without blocking. |
| Omega Veto | Hardware-level safety interrupt. Cannot be bypassed. Freezes all processing. Requires human approval to resume. |
| Page Fault | When GPU thought drifts near a ghost frame, triggering full frame load from VoltDB into working memory. |
| RAR | Root-Attend-Refine. The three-phase inference loop of the GPU Soft Core. |
| Resolution | One of four detail levels in a TensorFrame: R₀ (discourse), R₁ (proposition), R₂ (phrase), R₃ (token). |
| Slot | One of 16 structural positions in a TensorFrame, corresponding to a semantic role. |
| Strand | A collection of related TensorFrames in VoltDB, organized by topic/context. Analogous to a cortical column. |
| Sleep Consolidation | Forward-Forward weight updates on the VFN during idle time, using distilled frames as training data. |
| Tensor Frame Bus | The communication protocol carrying TensorFrames between all Volt components. |
| TensorFrame | F ∈ ℝ^[16 slots × 4 resolutions × 256 dims]. The fundamental data structure of thought in Volt. |
| Translator | Community module that converts raw input → TensorFrame. Implements `Translator` trait. |
| VFN | Vector Field Network. The GPU Soft Core's trainable neural network. Maps slot vectors to drift directions. |
| VoltDB | Embedded storage engine managing three-tier memory, indexing, GC, consolidation, and concurrency. |

---

*End of Master Blueprint*

*Volt v3.0 — "The Lipstick Masquerade"*
*Rust Core · Tensor Frames · Root-Attend-Refine · VoltDB · Forward-Forward · Intelligence Commons*
*Consumer Hardware · Sovereign Intelligence · Infinite Memory · Continual Learning*

*"We're playing a dangerous game, and things will never be the same."*