# Volt X — Architecture Reference

## Overview

Volt X is a stateful cognitive operating system built on three innovations:

1. **Tensor Frames** — Structured 3D thought representation `[S=16 × R=4 × D=256]`
2. **Split-Brain Architecture** — GPU Soft Core (intuition) + CPU Hard Core (logic)
3. **Three-Tier Living Memory** — T0 (working) → T1 (strand) → T2 (archive)

## The Fundamental Unit: TensorFrame

```
F ∈ ℝ^[S=16 slots × R=4 resolutions × D=256 dims] — 64KB max per frame
```

### Slots (S=16)
Each slot holds a semantic role:
- S₀: Agent (who)
- S₁: Predicate (what action)
- S₂: Patient (affected entity)
- S₃: Location (where)
- S₄: Time (when)
- S₅: Manner (how)
- S₆: Instrument (with what)
- S₇: Cause (why)
- S₈: Result (outcome)
- S₉–S₁₅: Free (domain-specific extensions)

### Resolutions (R=4)
Multi-scale representation, coarse to fine:
- R₀: Discourse (topic gist)
- R₁: Proposition (sentence-level)
- R₂: Phrase (detail)
- R₃: Token (BPE subwords, populated at decode time)

### Key Properties
- **Sparse**: Most slots at most resolutions are empty
- **Addressable**: Direct read/write to F[slot, resolution]
- **Composable**: Frame merge via slot concatenation + metadata merge
- **Per-slot certainty (γ)**: Each slot carries independent confidence

## Split-Brain Architecture

### Layer 0: External World
User input, APIs, sensors, OS events, P2P mesh, module repository.

### Layer 1: Input Translators
Community modules implementing `trait Translator { fn encode(&self, raw: &[u8]) -> TensorFrame }`.
Modalities: Text, Vision, Audio, Data, Sensor, OS Events, Custom.

### Layer 2: LLL Tensor Frame Bus
The universal communication protocol. HDC algebra operations:
- ⊗ Bind (FFT circular convolution)
- + Superpose (additive)
- ρ Permute (role rotation)
- ⊗⁻¹ Unbind (inverse for retrieval)
- Per-slot γ certainty propagation
- Codebook quantization (u16, 65K entries)

### Layer 3: GPU Soft Core ("Right Brain")
Neural intuition engine. SIMD parallel, CUDA.

**Components:**
- State Buffer: Current TensorFrame in VRAM
- Vector Field Network (VFN): 500M–2B params, Fourier Neural Operator
- Diffusion Controller: Per-slot orthogonal noise for exploration
- ODE Solver: RK4/DOPRI5 adaptive stepping
- Manifold Projector: Unit hypersphere normalization
- Convergence Detector: Per-slot ‖ΔS‖ < ε check
- Retention Cache: Warm-start from recently-converged attractors
- Ghost Frame Bleed Buffer: ~1000 R₀ gists from T1/T2

**SDE dynamics:** `dF = −∇E(F,t)dt + σ_φ(F,t)dW_t`

**VRAM Budget:** 4–8 GB (consumer GPU sufficient)

### Layer 4: CPU Hard Core ("Left Brain")
Deterministic logic engine. Rust + Tokio + Rayon.

**Components:**
- Intent Router: Cosine similarity → route to best Hard Strand
- Hard Strands (impl HardStrand): MathEngine, CodeRunner, APIDispatch
- HDC Algebra: FFT Bind/Unbind verification
- Certainty Engine: Min-rule γ propagation
- Safety Layer: Axiomatic Guard + Transition Monitor + Omega Veto
- Proof Constructor: Traceable reasoning chains
- Causal Simulation: do(X=x) counterfactual reasoning
- Mirror Module: Feedback to Soft Core (diffusion adjustment)

### Layer 5: VoltDB (Three-Tier Memory)
- **T0 (Working Memory)**: 64 frames, VRAM/RAM, instant access
- **T1 (Strand Storage)**: Millions of frames in RAM, strand-organized
- **T2 (Archive)**: Compressed on disk, holographic retrieval
- HNSW index for approximate nearest neighbor queries
- WAL for crash recovery
- Ghost frame management for Bleed Buffer

### Layer 6: Output Translators
Parallel frame decode: all slots simultaneously → assemble response.
~30x faster than autoregressive for long outputs.

## Inference Pipeline (Step by Step)

1. **Forward Translator** (GPU, ~5ms): NL → TensorFrame (R₀-R₁ slots)
2. **Bleed Engine Prefetch** (CPU async, ~2ms): HNSW query → ghost frames
3. **GPU Soft Core RAR Loop** (~20-100ms): Root → Attend → Refine, repeat
4. **CPU Hard Core Verification** (~5-15ms): Route, verify, safety, proof
5. **Parallel Frame Decode** (~10-30ms): All slots → text simultaneously
6. **VoltDB Store** (CPU async, ~1ms): Frame → T0, eviction → T1

**Total:** ~130ms (simple) to ~250ms (complex)

## RAR: Root-Attend-Refine

Each iteration:
- **Root** (parallel): Slot-local VFN forward passes. 16 independent passes.
  Converged slots skip. Diffusion noise per slot.
- **Attend** (cross-slot): Slot-to-slot attention O(S²×D) = O(16²×256).
  Ghost frame cross-attention for memory influence.
- **Refine** (update): S_i(t+1) = S_i(t) + dt × (ΔS_i + β·msg_i).
  Manifold projection. Per-slot convergence check.

Converged slots freeze (still participate in Attend as K/V).
Loop continues until all converged OR budget exhausted.

**Compute per iteration:** ~2.1M FLOPs (negligible on GPU).

## Continual Learning

Three timescales:
1. **Instant** (ms): Strand vector updates in RAM. Every inference IS learning.
2. **Sleep** (hours): Forward-Forward weight updates. Layer-local, low VRAM.
3. **Developmental** (days): Strand graduation + community modules.

## Dependency Graph

```
volt-core (no deps)
  ↑
volt-bus (core)
  ↑
volt-soft (core, bus)     volt-hard (core, bus)     volt-db (core, bus)
  ↑                         ↑                         ↑
volt-translate (core, bus, db)
volt-learn (core, bus, db, soft)
volt-safety (core, bus, hard)
volt-ledger (core, bus, db)
  ↑
volt-server (ALL — leaf crate, no one imports this)
```

## Training Phases

1. **Bootstrap** (~2-4 weeks): Translator training, codebook init, role grounding
2. **Soft Core** (~4-8 weeks): VFN energy landscape, diffusion, convergence
3. **Hard Core** (~2-4 weeks): Rust engineering, safety axioms, VoltDB
4. **Joint Alignment** (~4-8 weeks): End-to-end calibration, continual learning

All phases run on consumer hardware (RTX 4090 sufficient, RTX 4060 viable).
