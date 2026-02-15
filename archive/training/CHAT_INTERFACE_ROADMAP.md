# Chat Interface Roadmap for Volt X

**Status Date**: 2026-02-12
**Current Phase**: Phase 6.1 Complete (Module System)

## Executive Summary

Volt X has a **fully functional cognitive backend** with text-in/text-out capabilities via `POST /api/think`, persistent memory (VoltDB), learning (RLVF), and safety verification. However, it lacks the **user-facing infrastructure** needed for a modern chat interface experience.

**Bottom line**: You're **~1-2 weeks of frontend work** away from a usable chat interface.

---

## Current State: What Already Works ✅

The server at `crates/volt-server/src/routes.rs` provides:

- **Text in → Text out**: `POST /api/think` accepts JSON `{"text": "..."}` and returns processed text
- **Full cognitive pipeline**: Input → Encode → RAR (Soft Core) → Hard Core + Safety → Memory → Decode → Output
- **Persistent memory**: Verified frames are stored in VoltDB (T0/T1/T2), so the system "remembers" across requests
- **Learning**: RLVF alignment and sleep consolidation run in the background
- **Introspection**: Rich debug info (proof steps, gamma scores, slot states, timing)
- **Module system**: Hot-pluggable translators, strands, and action cores (Phase 6.1)

### Current API

**Endpoint**: `POST /api/think`

**Request**:
```json
{
  "text": "What is 10 + 20?"
}
```

**Response**:
```json
{
  "text": "10 + 20 = 30",
  "gamma": [1.0, 1.0, 1.0],
  "strand_id": 0,
  "iterations": 2,
  "slot_states": [...],
  "proof_steps": [
    {
      "strand_name": "math_engine",
      "description": "10 + 20 = 30",
      "similarity": 0.95,
      "gamma_after": 1.0,
      "activated": true
    }
  ],
  "safety_score": 0.0,
  "memory_frame_count": 42,
  "ghost_count": 5,
  "timing_ms": {
    "encode_ms": 0.5,
    "decode_ms": 0.3,
    "total_ms": 52.1
  }
}
```

---

## Missing Features for Chat Interface ❌

### 1. Conversational Context Management (CRITICAL)

**Current Issue**: Each request is independent. No conversation history or session tracking.

**What's Needed**:
- **Session/Conversation IDs**: Track multi-turn dialogues
  - Add `conversation_id: Option<String>` to `ThinkRequest`
  - Store conversation history in `AppState` (HashMap<ConversationId, Vec<Frame>>)
- **Conversation Strands**: VoltDB already has `strand_id`, but it's not tied to user sessions
  - Proposal: Use strand_id to represent conversations
  - Each conversation gets a unique strand in T1
- **Context Window**: Distinguish "this conversation" from "all past conversations"
  - Inject recent messages (last 5-10 frames) as context during RAR
  - Use ghost attention to pull from conversation-specific strand
- **Message History API**:
  - `GET /api/conversations` — list all conversations
  - `GET /api/conversations/:id/history` — retrieve past messages
  - `POST /api/conversations/:id/message` — send message in existing conversation
  - `POST /api/conversations` — start new conversation

**Priority**: **CRITICAL**
**Estimated Effort**: 2-3 days
**Implementation Notes**:
- Extend `ThinkRequest` and `ThinkResponse` models
- Add conversation state management to `AppState`
- Wire conversation frames into ghost attention mechanism

---

### 2. Streaming Responses (HIGH PRIORITY)

**Current Issue**: Client waits ~50-300ms for full pipeline, then gets complete response at once.

**What's Needed**:
- **Server-Sent Events (SSE)** endpoint: `GET /api/think/stream`
  - Stream events as RAR iterations complete
  - Send incremental slot updates as they converge
- **WebSocket** alternative: `ws://localhost:8080/api/chat`
  - Bidirectional communication for real-time interaction
- **Incremental decoding**: Stream slot words as they converge during RAR iterations
  - Modify RAR loop to yield after each iteration
  - Decode converged slots immediately
- **Progress updates**: Show "thinking..." states
  ```json
  {"type": "status", "message": "RAR iteration 3/15, 8 slots converged"}
  {"type": "slot", "role": "Agent", "word": "cat", "certainty": 0.82}
  {"type": "complete", "text": "cat sat mat.", "gamma": [0.82, 0.81, 0.80]}
  ```

**Priority**: **HIGH**
**Estimated Effort**: 3-5 days
**Implementation Notes**:
- Axum supports SSE via `Sse` response type
- Refactor `rar_loop_with_ghosts` to accept a callback for intermediate results
- Consider adding `streaming: bool` flag to `ThinkRequest`

---

### 3. User Interface (HIGH PRIORITY)

**Current Issue**: No frontend at all. Users must manually `curl` the API.

#### Option A: CLI Chat Client (Quick Win)

**What to Build**:
```bash
$ volt chat
Volt X Chat (v0.1.0)
Type 'exit' to quit, 'clear' to reset conversation.

You: Hello
[Volt thinks... 45ms]
Volt: cat sat mat.
     [γ: 0.82 | strands: certainty_engine | memory: 42 frames]

You: What's 10 + 20?
[Volt thinks... 98ms]
Volt: 10 + 20 = 30
     [γ: 1.0 | strands: math_engine | memory: 43 frames]

You: /debug
[Debug mode: ON - showing full proof chains]

You: exit
Goodbye!
```

**Features**:
- Maintains conversation state in memory (conversation_id generation)
- Shows abbreviated introspection (gamma, active strands, memory count)
- Calls `POST /api/think` for each message
- Commands: `/debug`, `/clear`, `/help`, `/modules`, `/export`
- Optional: ANSI colors for better UX

**Priority**: **HIGH**
**Estimated Effort**: 1-2 days
**Location**: New binary crate `volt-cli` or subcommand in `volt-server`

#### Option B: Web UI (Full Experience)

**What to Build**:
```
┌─────────────────────────────────────────────────┐
│  Volt X Chat                          [⚙️]  [?] │
├─────────────────────────────────────────────────┤
│                                                 │
│  You: Hello                                     │
│                                                 │
│  Volt: cat sat mat.                            │
│  [γ: 0.82 | 45ms | strands: certainty_engine] │
│                                                 │
│  You: What's 10 + 20?                          │
│                                                 │
│  Volt: 10 + 20 = 30                            │
│  [γ: 1.0 | 98ms | strands: math_engine]       │
│  [▼ Show proof chain]                          │
│                                                 │
├─────────────────────────────────────────────────┤
│  Type a message...                      [Send] │
└─────────────────────────────────────────────────┘
```

**Tech Stack Options**:
1. **Simple**: Plain HTML + Vanilla JS (no build step)
2. **Modern**: React/Vue/Svelte + Vite
3. **Rust**: Leptos/Dioxus (full-stack Rust, SSR optional)

**Features**:
- Chat message bubbles (user vs Volt)
- Input box with send button (Enter to send)
- "Thinking..." spinner during requests
- Collapsible debug panel for proof steps, slot states
- Conversation history sidebar (list of past conversations)
- Settings panel (adjust RAR iterations, ghost bleed alpha, etc.)
- Optional: dark mode toggle

**Priority**: **HIGH**
**Estimated Effort**: 3-5 days (vanilla), 5-7 days (modern framework)
**Location**: `crates/volt-server/static/` (serve via Axum `ServeDir`)

---

### 4. Better I/O Translators (Planned in Phase 6.2)

**Current State**: Text-only translator with basic word-to-vector encoding.

**Planned in Milestone 6.2** (Week 45-46):
- **Vision Translator**: Image input (CLIP/SigLIP → TensorFrame)
  - Upload image → extract objects/labels → populate AGENT/PATIENT slots
  - Example: "Describe this image" → [Agent: cat, Predicate: sits, Patient: mat]
- **Speech Action Core**: Audio output (TTS from frame text)
  - Text response → Bark/Piper TTS → MP3/WAV download
  - Example: "Read this aloud" button in web UI

**Multimodal Chat Examples**:
```
You: [uploads cat.jpg]
Volt: This image shows a cat sitting on a mat.

You: Generate audio for "hello world"
Volt: [🔊 audio player] hello_world.mp3

You: What do you see in this diagram?
Volt: A flowchart with 5 nodes showing the RAR inference loop.
```

**Priority**: **MEDIUM**
**Estimated Effort**: 2 weeks (per Phase 6.2 schedule)
**Status**: 🚧 Planned (not started)

---

### 5. Action Execution / Tool Use (Planned in Phase 6.2)

**Current Issue**: Volt can only *think*, not *do*.

**What's Needed**:
- **ActionCore modules**: Execute commands, API calls, file operations
  - Already scaffolded via `ActionCore` trait in `volt-translate/src/action.rs`
  - Example modules:
    - `WeatherActionCore`: Fetch real-time weather data (currently mock)
    - `CalculatorActionCore`: Perform exact arithmetic (delegate to math_engine)
    - `WebSearchActionCore`: Query search engines, return results
    - `FileActionCore`: Read/write files (with safety constraints)
- **Tool use protocol**: Like OpenAI function calling
  - Volt decides to use a tool → calls ActionCore → incorporates result into frame
  - Example:
    ```
    You: What's the weather in Tokyo?
    [Volt thinks... routes to WeatherStrand... calls WeatherActionCore]
    Volt: Tokyo is currently 15°C, partly cloudy.
    ```

**Priority**: **MEDIUM**
**Estimated Effort**: 2 weeks (per Phase 6.2 schedule)
**Status**: 🚧 Planned (ActionCore trait exists, modules not implemented)

---

### 6. User Identity & Multi-User Support (Planned in Phase 6.3)

**Current Issue**: No authentication, everyone shares the same memory.

**Planned in Milestone 6.3** (Week 47-48):
- **Ed25519 keypairs**: Self-sovereign identity (for P2P Intelligence Commons)
- **Per-user memory isolation**: Separate VoltDB instances or strand namespacing
  - Option A: Separate `VoltStore` per user (high memory cost)
  - Option B: Prefix strand IDs with user ID (`user_123:strand_0`)
- **Privacy controls**: "Don't share this conversation" flag
  - Private conversations not exported to Intelligence Commons
  - Local-only strands vs shareable strands
- **Session management**: JWT or session cookies
  - `POST /api/auth/login` → JWT token
  - Include `Authorization: Bearer <token>` in requests
  - AppState tracks user_id → VoltStore mappings

**Priority**: **LOW** (for single-user deployments)
**Estimated Effort**: 2 weeks (per Phase 6.3 schedule)
**Status**: 🚧 Planned (not started)

---

### 7. Community Modules & Sharing (Planned in Phase 6.3-6.4)

**Current State**: Module system exists (Phase 6.1 complete), but no sharing infrastructure.

**Planned**:
- **Milestone 6.3** (Week 47-48): Intelligence Commons Layer 0
  - Local event log: append-only, Merkle-hashed entries
  - Strand export/import: serialize → encrypt → sign → share
  - Fact logging: verified frames (γ > 0.95) logged with provenance
- **Milestone 6.4** (Week 49-52): P2P Mesh + Settlement
  - libp2p integration: peer discovery, gossip protocol
  - CRDT-based event log sync (eventual consistency)
  - Module registry: content-addressed binaries shared via P2P
  - Settlement prototype: DAG-based micropayment tracking

**Use Cases**:
- Share a trained coding strand with the community
- Install a community-built "SQL expert" strand
- Earn credits when others use your modules
- Subscribe to a shared knowledge base (e.g., medical facts)

**Priority**: **LOW** (for local single-user deployments)
**Estimated Effort**: 6 weeks total (per Phase 6.3-6.4 schedule)
**Status**: 🚧 Planned (not started)

---

## Recommended Implementation Path

### Phase 1: Minimal Viable Chat (1 week)

**Goal**: Make Volt interactable today with minimal effort.

**Tasks**:
1. **Add conversation tracking** (2 days)
   - Extend `ThinkRequest` with `conversation_id: Option<String>`
   - Store conversation history in `AppState`
   - Inject recent frames into ghost attention
   - Add `POST /api/conversations` and `GET /api/conversations/:id/history`

2. **Build CLI chat client** (2 days)
   - New binary `volt-cli` with `chat` subcommand
   - Interactive REPL: read input → call `/api/think` → display response
   - Show abbreviated introspection (gamma, strands, timing)
   - Commands: `/debug`, `/clear`, `/export`

3. **Test & polish** (1 day)
   - End-to-end testing of multi-turn conversations
   - Fix any memory leaks or performance issues
   - Update documentation

**Deliverable**: `volt chat` command that feels like ChatGPT in the terminal.

---

### Phase 2: Web Interface (1 week)

**Goal**: User-friendly web UI with streaming responses.

**Tasks**:
1. **Simple web UI** (3 days)
   - HTML + vanilla JS (or React if preferred)
   - Chat message bubbles, input box, send button
   - Display gamma scores, proof steps (collapsible)
   - Serve static files via Axum `ServeDir`

2. **Add SSE streaming** (2 days)
   - New endpoint: `GET /api/think/stream?text=...&conversation_id=...`
   - Refactor RAR loop to yield intermediate results
   - Stream slot updates as they converge
   - Update web UI to consume SSE events

3. **Polish & deploy** (2 days)
   - Responsive design (mobile-friendly)
   - Loading states, error handling
   - Deploy instructions (Docker, systemd, etc.)

**Deliverable**: Web UI at `http://localhost:8080/` with real-time streaming.

---

### Phase 3: Multimodal & Actions (2 weeks)

**Goal**: Vision, speech, and tool use (per Phase 6.2).

**Tasks** (see Milestone 6.2 in `roadmap/PHASE-6.md`):
1. Vision Translator (CLIP/SigLIP)
2. Speech Action Core (Bark/Piper TTS)
3. Tool use examples (weather, calculator, web search)

---

### Phase 4: Multi-User & P2P (4+ weeks)

**Goal**: Authentication, sharing, Intelligence Commons (per Phase 6.3-6.4).

**Tasks** (see Milestones 6.3-6.4 in `roadmap/PHASE-6.md`):
1. User auth & per-user memory
2. Strand export/import with signatures
3. P2P mesh & fact sharing
4. Module marketplace & settlement

---

## Feature Priority Matrix

| Feature | Priority | Effort | Status | Target Week |
|---------|----------|--------|--------|-------------|
| Conversation tracking | **CRITICAL** | 2-3 days | ❌ Not started | Week 1 |
| CLI chat client | **HIGH** | 1-2 days | ❌ Not started | Week 1 |
| Web UI (basic) | **HIGH** | 3-5 days | ❌ Not started | Week 2 |
| SSE streaming | **HIGH** | 3-5 days | ❌ Not started | Week 2 |
| Vision Translator | **MEDIUM** | 1 week | 🚧 Planned (6.2) | Week 45-46 |
| Speech Action Core | **MEDIUM** | 1 week | 🚧 Planned (6.2) | Week 45-46 |
| Tool use / Actions | **MEDIUM** | 2 weeks | 🚧 Planned (6.2) | Week 45-46 |
| User auth | **LOW** | 2 weeks | 🚧 Planned (6.3) | Week 47-48 |
| P2P mesh | **LOW** | 4 weeks | 🚧 Planned (6.4) | Week 49-52 |

---

## Quick Start: Try the API Today

Even without a chat UI, you can interact with Volt right now:

```bash
# Start the server
cd c:\Volt_X
cargo run --release --bin volt-server

# In another terminal, send a request
curl -X POST http://localhost:8080/api/think \
  -H "Content-Type: application/json" \
  -d '{"text": "What is 10 + 20?"}'

# Response:
# {
#   "text": "10 + 20 = 30",
#   "gamma": [1.0, 1.0, 1.0],
#   "iterations": 2,
#   ...
# }
```

---

## Open Questions

1. **Conversation Persistence**: Should conversations persist to disk (VoltDB T2) or stay in memory?
   - **Recommendation**: Store in T1 as strands, auto-archive old conversations to T2
   - Add `GET /api/conversations` to list all past conversations

2. **Streaming Format**: SSE vs WebSocket?
   - **Recommendation**: SSE for simplicity (unidirectional, auto-reconnect)
   - WebSocket if bidirectional features needed (e.g., interrupt mid-inference)

3. **UI Framework**: Vanilla JS vs React/Vue/Svelte vs Rust (Leptos/Dioxus)?
   - **Recommendation**: Start with vanilla JS (no build step, easy to modify)
   - Migrate to React/Svelte later if complexity grows

4. **Multi-User Priority**: Single-user (localhost) vs multi-tenant (cloud)?
   - **Recommendation**: Focus on localhost first (simpler, no auth needed)
   - Add multi-user in Phase 6.3 when Intelligence Commons is ready

---

## References

- Current API: [`crates/volt-server/src/routes.rs`](crates/volt-server/src/routes.rs)
- Request/Response models: [`crates/volt-server/src/models.rs`](crates/volt-server/src/models.rs)
- Phase 6 roadmap: [`roadmap/PHASE-6.md`](roadmap/PHASE-6.md)
- Architecture overview: [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## Next Steps

To get started on a chat interface:

1. **Read this document** to understand the gaps
2. **Choose a path**: CLI-first (fast) or Web UI (full experience)
3. **Start with conversation tracking** (foundation for everything else)
4. **Build incrementally**: Each feature should work standalone
5. **Test with real usage**: Multi-turn conversations, complex queries, edge cases

**Questions?** Review the roadmap files in `roadmap/` or check `ARCHITECTURE.md` for technical details.
