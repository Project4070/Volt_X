# Volt X User Guide

**Version 0.1.0** — A Cognitive Architecture / Stateful AI OS

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Web UI](#web-ui)
3. [CLI Chat Client](#cli-chat-client)
4. [What Volt X Can Do](#what-volt-x-can-do)
5. [Understanding the Output](#understanding-the-output)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- Windows, macOS, or Linux
- Rust toolchain installed (for building from source)

### Building Volt X

```bash
# Clone the repository (if you haven't already)
cd c:\Volt_X

# Build the release version (optimized)
cargo build --release

# The server binary will be at:
# target/release/volt-server.exe (Windows)
# target/release/volt-server (Linux/macOS)
```

### Starting the Server

```bash
# Start the HTTP server on port 8080
cargo run --release --bin volt-server
or
cargo run --release --bin volt-server --all-features

# Or run the built binary directly:
.\target\release\volt-server.exe
```

The server will start and you'll see:

```
Initializing VoltDB with 3-tier memory (T0/T1/T2)...
HNSW index initialized: 0 frames indexed
VoltDB initialized successfully
Volt X Server starting on 0.0.0.0:8080...
Server running at http://0.0.0.0:8080
```

**The server is now ready!** Open your browser or CLI client to start chatting.

---

## Web UI

### Accessing the Web Interface

Open your web browser and navigate to:

```
http://localhost:8080
```

You'll see the Volt X chat interface with:
- **Sidebar**: List of your conversations
- **Main area**: Chat messages and input box
- **Header controls**: Streaming toggle and debug mode

### Web UI Features

#### 1. **Creating a New Conversation**

Click the **"+ New"** button in the sidebar to start a fresh conversation. Each conversation has its own memory context.

#### 2. **Sending Messages**

Type your message in the input box at the bottom and either:
- Press **Enter** to send
- Press **Shift+Enter** to add a new line without sending
- Click the **"Send"** button

#### 3. **Streaming Mode** (⚡ Streaming)

**Default mode** — Shows real-time progress as Volt X thinks:

- "Preparing conversation..." → Setting up memory
- "Encoding..." → Converting your text to TensorFrame
- "Thinking..." → Running the RAR inference loop
- Final response appears when complete

**Toggle streaming off** to get instant responses (batch mode).

#### 4. **Debug Mode** (⚙ Debug)

Click **"⚙ Debug"** to toggle detailed technical information:

**Normal view:**
```
15
γ: 1.00 • 25ms
```

**Debug view:**
```
15
γ: 1.00 • 25ms
Iterations: 50
Memory: 1 frames, 0 ghosts
Safety: 0.000
Timing: encode=0.3ms, decode=0.0ms
Proof Chain:
→ math_engine (sim: 1.00, γ: 1.00) ✓
→ certainty_engine (sim: 1.00, γ: 1.00) ✓
Slots (3 active):
[1] Predicate: "[capability tag]" (γ=0.90, Translator)
[6] Instrument: "[operation data]" (γ=1.00, Translator)
[8] Result: "15" (γ=1.00, HardCore)
```

**Debug info explained:**
- **γ (gamma)**: Certainty score (0-1, higher is more certain)
- **Iterations**: How many RAR refinement loops ran
- **Memory**: Number of frames and ghosts in working memory
- **Safety**: Safety check score from Omega Veto
- **Proof Chain**: Which Hard Strands processed the frame
  - `sim`: Similarity score (how well the strand matched)
  - `γ`: Certainty after processing
  - `✓`: Strand activated and processed
  - `✗`: Strand evaluated but didn't activate
- **Slots**: Active TensorFrame slots with their contents

#### 5. **Conversation History**

Click on any conversation in the sidebar to view its history. The most recent conversation is at the top.

**Conversation metadata shows:**
- Conversation ID (last 6 digits)
- Message count
- Last message time (e.g., "5m ago", "2h ago")

#### 6. **Dark Mode**

The UI automatically switches to dark mode if your system preferences are set to dark mode.

---

## CLI Chat Client

### Starting the CLI Client

```bash
# Start the interactive CLI chat client
cargo run --release --bin volt-chat

# Or run the built binary:
.\target\release\volt-chat.exe
```

You'll see:

```
Volt X Chat (v0.1.0)
Type /help for commands, /exit to quit

You:
```

### CLI Commands

The CLI client supports several slash commands:

#### `/help` — Show help

```
You: /help
```

Shows all available commands and their descriptions.

#### `/clear` — Start new conversation

```
You: /clear
```

Clears the current conversation and starts a fresh one. Previous conversation history is preserved on the server.

#### `/debug` — Toggle debug mode

```
You: /debug
Debug mode: ON
```

Shows detailed technical output (proof chain, slots, timing) for each response.

#### `/list` — List all conversations

```
You: /list
Conversations:
  1. Conversation abc123 (5 messages, 2m ago)
  2. Conversation def456 (12 messages, 1h ago)
  3. Conversation ghi789 (3 messages, 1d ago)
```

#### `/exit` or `/quit` — Exit the client

```
You: /exit
Goodbye!
```

Exits the CLI client. The server keeps running.

### CLI Options

```bash
# Connect to a specific server URL
cargo run --release --bin volt-chat -- --url http://192.168.1.100:8080

# Resume a specific conversation
cargo run --release --bin volt-chat -- --conversation 123456789
```

### CLI Output Format

**Compact mode (default):**
```
You: What is 10 + 5?
Volt: 15
     [γ: 1.00 | conv: 1 | 25ms]
```

**Debug mode:**
```
You: What is 10 + 5?
Volt: 15
     [γ: 1.00 | conv: 1 | 25ms]

Debug Info:
  Iterations: 50
  Memory: 1 frames, 0 ghosts
  Safety: 0.000
  Timing: encode=0.3ms, decode=0.0ms

Proof Chain:
  → math_engine (sim: 1.00, γ: 1.00) ✓
  → certainty_engine (sim: 1.00, γ: 1.00) ✓

Slots (3 active):
  [1] Predicate: "[capability tag]" (γ=0.90, Translator)
  [6] Instrument: "[operation data]" (γ=1.00, Translator)
  [8] Result: "15" (γ=1.00, HardCore)
```

---

## What Volt X Can Do

### Exact Arithmetic (Math Engine)

Volt X has a deterministic math engine for exact computation.

**Supported operations:**

```bash
# Addition
You: 10 + 5
Volt: 15

# Subtraction
You: 100 - 37
Volt: 63

# Multiplication
You: 847 * 392
Volt: 332024

# Division
You: 100 / 4
Volt: 25

# Power/exponentiation
You: 2 ^ 10
Volt: 1024

# Alternative syntax also works:
You: 10 × 5    # Unicode multiplication
Volt: 50

You: 20 ÷ 4    # Unicode division
Volt: 5

You: 2 ** 8    # Double asterisk for power
Volt: 256
```

**Math features:**
- **Exact computation**: No floating-point errors for integer operations
- **Instant response**: < 1ms processing time
- **Certainty γ=1.0**: Math results always have maximum certainty
- **No hallucinations**: Only returns results it can compute exactly

### Hyperdimensional Computing (HDC Algebra)

Volt X can perform HDC vector operations (bind, unbind, superpose).

**Note:** HDC features are available but require specific slot-level encoding. This will be exposed through natural language in future updates.

### What Volt X Currently CANNOT Do

**Volt X is NOT a general-purpose chatbot.** It's a cognitive architecture optimized for:
- Exact reasoning (math, logic)
- Verifiable computation
- Deterministic output

**Current limitations:**
- ❌ No general conversation (e.g., "tell me about cats" won't work)
- ❌ No text generation or creative writing
- ❌ No knowledge base queries (no "what is X?" questions)
- ❌ No code generation or debugging help

**These features are planned** via integration with LLMs and additional Hard Strands (see roadmap Phase 6+).

---

## Understanding the Output

### Certainty (γ - Gamma)

The **γ score** represents how certain Volt X is about its answer:

- **γ = 1.00**: Maximum certainty (exact computation, verified)
- **γ = 0.90-0.99**: High certainty (strong evidence)
- **γ = 0.70-0.89**: Medium certainty (reasonable confidence)
- **γ = 0.50-0.69**: Low certainty (uncertain, needs verification)
- **γ < 0.50**: Very uncertain (likely unreliable)

**For math operations, γ is always 1.00** because the result is computed exactly.

### Timing Information

- **encode_ms**: Time to convert your text to TensorFrame
- **decode_ms**: Time to convert TensorFrame back to text
- **total_ms**: Total end-to-end processing time

**Typical timings:**
- Simple math: 20-50ms
- Complex operations: 50-300ms

### Proof Chain

The proof chain shows which Hard Strands evaluated your request:

```
→ math_engine (sim: 1.00, γ: 1.00) ✓
→ certainty_engine (sim: 1.00, γ: 1.00) ✓
```

- **sim (similarity)**: How well the strand's capability matched your input (0-1)
- **γ (gamma)**: Certainty score after processing
- **✓**: Strand activated and processed the frame
- **✗**: Strand evaluated but similarity was below threshold (didn't activate)

**How routing works:**
1. Your input is encoded to a TensorFrame
2. Intent Router compares the frame to all registered Hard Strands
3. The strand with highest similarity (above threshold) is activated
4. The strand processes the frame and returns the result

### Slot States

TensorFrame has 16 slots (S0-S15) with semantic roles:

- **S0 (Agent)**: "Who" is acting
- **S1 (Predicate)**: "What" is happening (capability tag for routing)
- **S2 (Patient)**: "What" is being acted upon
- **S6 (Instrument)**: "How" / tool being used (math operations go here)
- **S8 (Result)**: The computed result

**Example for "10 + 5":**
```
[1] Predicate: "[capability tag]" (γ=0.90, Translator)
[6] Instrument: "[operation data]" (γ=1.00, Translator)
[8] Result: "15" (γ=1.00, HardCore)
```

Slot 1 contains the math engine's capability vector (for routing), slot 6 contains the operation data (1.0=add, 10, 5), and slot 8 contains the result (15).

---

## Troubleshooting

### "Connection refused" or "Cannot connect"

**Problem:** The server isn't running.

**Solution:**
```bash
# Start the server
cargo run --release --bin volt-server
```

### "Echoes my input back"

**Problem:** You're asking Volt X to do something it doesn't support (e.g., general conversation).

**Solution:** Try a math expression:
```
You: 10 + 5
Volt: 15
```

Volt X currently only has specialized capabilities (math, HDC). General conversation requires LLM integration (planned).

### "Math engine not activating" (similarity -0.00)

**Problem:** The translator isn't properly tagging frames with capability vectors.

**Solution:** This was fixed in the latest build. Rebuild:
```bash
cargo build --release
```

Ensure you're using the latest version with the capability vector tagging fix.

### Server crashes or panics

**Problem:** Stack overflow or memory issue (TensorFrame is ~64KB).

**Solution:** This is handled automatically in the latest build (8MB thread stacks). If you still see crashes:

1. Check Windows event logs
2. Increase stack size in routes.rs if needed
3. Report the issue with full error message

### Web UI stuck at "Preparing..."

**Problem:** Streaming endpoint is hanging or the server isn't responding.

**Solution:**
1. Check server logs for errors
2. Try toggling streaming mode off (⚡ button)
3. Refresh the page
4. Restart the server

### "No similarity match" in debug output

**Problem:** The Intent Router isn't finding a matching Hard Strand.

**Solution:**
1. Check `/api/modules` to see installed strands
2. Verify your input matches a supported capability
3. Use debug mode to see similarity scores

---

## API Endpoints

For programmatic access, Volt X exposes a REST API:

### `GET /health`
Health check endpoint.

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### `POST /api/think`
Process text (batch mode).

```bash
curl -X POST http://localhost:8080/api/think \
  -H "Content-Type: application/json" \
  -d '{"text": "10 + 5"}'
```

Response:
```json
{
  "text": "15",
  "gamma": [1.0, 1.0, 1.0],
  "conversation_id": 1,
  "strand_id": 0,
  "iterations": 50,
  "slot_states": [...],
  "proof_steps": [...],
  "safety_score": 0.0,
  "memory_frame_count": 1,
  "ghost_count": 0,
  "timing_ms": {
    "encode_ms": 0.3,
    "decode_ms": 0.0,
    "total_ms": 25.0
  }
}
```

### `POST /api/think/stream`
Process text with SSE streaming.

```bash
curl -X POST http://localhost:8080/api/think/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "10 + 5"}' \
  --no-buffer
```

Returns Server-Sent Events (SSE):
```
data: {"type":"status","data":"Preparing conversation..."}

data: {"type":"encoding"}

data: {"type":"thinking"}

data: {"type":"complete","data":{...ThinkResponse...}}
```

### `POST /api/conversations`
Create a new conversation.

```bash
curl -X POST http://localhost:8080/api/conversations
```

Response:
```json
{
  "conversation_id": 123456789
}
```

### `GET /api/conversations`
List all conversations.

```bash
curl http://localhost:8080/api/conversations
```

Response:
```json
{
  "conversations": [
    {
      "id": 123456789,
      "created_at": 1704067200000000,
      "last_message_at": 1704070800000000,
      "message_count": 5
    }
  ]
}
```

### `GET /api/conversations/{id}/history`
Get conversation history.

```bash
curl http://localhost:8080/api/conversations/123456789/history
```

Response:
```json
{
  "conversation_id": 123456789,
  "messages": [
    {
      "frame_id": 1,
      "text": "10 + 5",
      "gamma": [0.9, 1.0, 1.0],
      "timestamp": 1704067200000000
    },
    {
      "frame_id": 2,
      "text": "15",
      "gamma": [1.0, 1.0, 1.0],
      "timestamp": 1704067205000000
    }
  ]
}
```

### `GET /api/modules`
List all installed modules.

```bash
curl http://localhost:8080/api/modules
```

Response:
```json
{
  "modules": [
    {
      "name": "math_engine",
      "module_type": "HardStrand",
      "version": "0.1.0",
      "description": "Exact arithmetic computation (add, sub, mul, div, pow)"
    },
    {
      "name": "stub_translator",
      "module_type": "Translator",
      "version": "0.1.0",
      "description": "Heuristic word-to-slot translator"
    }
  ]
}
```

---

## Examples

### Example 1: Simple Arithmetic

```
You: 42 + 17
Volt: 59
     [γ: 1.00 | conv: 1 | 23ms]
```

### Example 2: Complex Calculation

```
You: 847 * 392
Volt: 332024
     [γ: 1.00 | conv: 1 | 27ms]
```

### Example 3: Power Operations

```
You: 2 ^ 16
Volt: 65536
     [γ: 1.00 | conv: 1 | 24ms]
```

### Example 4: Division

```
You: 1024 / 32
Volt: 32
     [γ: 1.00 | conv: 1 | 25ms]
```

### Example 5: Multi-turn Conversation

```
You: 100 + 50
Volt: 150
     [γ: 1.00 | conv: 1 | 24ms]

You: 150 * 2
Volt: 300
     [γ: 1.00 | conv: 1 | 26ms]

You: 300 / 3
Volt: 100
     [γ: 1.00 | conv: 1 | 23ms]
```

Each message in the same conversation shares memory context (conversation ID stays the same).

---

## Next Steps

**Want to extend Volt X?**

- Add new Hard Strands for custom capabilities
- Integrate external LLMs via the module system
- Contribute to the open-source project

**Learn more:**
- Read `ARCHITECTURE.md` for technical details
- Check `DECISIONS.md` for design rationale
- See `roadmap/` for the development roadmap

**Community:**
- Report bugs on GitHub Issues
- Suggest features via Discussions
- Contribute code via Pull Requests

---

**Volt X** — A cognitive architecture that thinks, not hallucinates.
