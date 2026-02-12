# Phase 6: Ecosystem (Months 11-12)

> Extracted from the master roadmap. Depends on: Phase 5 completed.

## Goal
Open the platform to the community and launch the Intelligence Commons.

---

## Milestone 6.1: Trait Specification + Module Hot-Plug (Week 43-44) -- COMPLETE

**Crate:** all crates (refine interfaces)

### What Was Built

- [x] Finalized and documented three traits: `Translator`, `HardStrand`, `ActionCore`
- [x] `ModuleInfo` + `ModuleType` in volt-core for module metadata
- [x] `ActionCore` trait in volt-translate with `TextAction` default impl
- [x] Default `info()` methods on `HardStrand` and `Translator` (backward compatible)
- [x] Module discovery: `ModuleRegistry::discover()` scans feature-gated modules at startup
- [x] Hot-plug infrastructure: `volt modules list|install|uninstall` CLI + `GET /api/modules` endpoint
- [x] `catch_unwind` panic safety in `IntentRouter::route()` + `unregister()`/`strand_names()`
- [x] Example module: `WeatherStrand` (deterministic mock data, behind `weather` feature flag)

### What Was Tested

- [x] Install weather module (enable feature) -> router routes weather queries to it
- [x] Uninstall module (disable feature) -> weather queries fall back to Soft Core
- [x] Module with a bug (panic) -> caught by router, logged, does not crash the system
- [x] `ModuleRegistry` discovers built-in + feature-gated modules
- [x] `GET /api/modules` returns JSON with module metadata

**Duration:** 2 weeks. **ADR:** ADR-027.

---

## Milestone 6.2: Community Translators + Action Cores (Week 45-46)

**Crate:** `volt-translate` (extend)

### What You Build
- Vision Translator prototype: uses CLIP or SigLIP to encode images -> frame slots
- Speech Action Core prototype: uses Bark or Piper for TTS from frame text output
- Package both as example community modules with full documentation
- Module publishing guide: how to publish to crates.io / Volt module registry

### What You Test
- Send image -> Vision Translator -> TensorFrame with object labels in AGENT/PATIENT slots
- Text response -> Speech Action Core -> audio file
- Both work as hot-pluggable modules (install/uninstall)

**Duration:** 2 weeks.

---

## Milestone 6.3: Intelligence Commons Layer 0 (Week 47-48)

**Crate:** `volt-ledger`

### What You Build
- Local event log: append-only, Merkle-hashed entries
- Ed25519 keypair generation and management (self-sovereign identity)
- Strand export: serialize strand -> encrypt -> sign -> shareable binary
- Strand import: verify signature -> decrypt -> merge into VoltDB
- Basic fact logging: each verified frame (gamma > 0.95) logged with provenance

### What You Test
- Export Coding strand -> import on different Volt instance -> all frames intact
- Tampered export (modified bytes) -> import rejects (signature verification fails)
- Event log is append-only (cannot modify past entries)
- 10,000 logged events -> Merkle root computable in < 100ms

**Duration:** 2 weeks.

---

## Milestone 6.4: P2P Mesh + Settlement Prototype (Week 49-52)

**Crate:** `volt-ledger` (extend)

### What You Build
- libp2p integration: peer discovery, gossip protocol for fact sharing
- CRDT-based event log synchronization between peers (eventual consistency)
- Module registry: content-addressed (CID) module binaries shared via P2P
- Settlement prototype: simple DAG-based micropayment tracking (not production-grade -- proof of concept)

### What You Test
- Two Volt instances on same network discover each other
- Shared fact (gamma > 0.95) propagates from instance A to instance B
- Module published by A -> discoverable and installable by B
- Settlement: A uses B's module -> usage event logged -> settlement ledger shows credit

**Duration:** 4 weeks.

---

## Phase 6 Checkpoint

At the end of Phase 6, Volt is an open platform. Community members can build modules, share knowledge, trade strands, and participate in a decentralized intelligence economy.
