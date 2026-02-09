

<svg viewBox="0 0 1000 500" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="diagBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0a0a1a" />
      <stop offset="100%" style="stop-color:#1a1020" />
    </linearGradient>
  </defs>

  <rect width="1000" height="500" fill="url(#diagBg)" rx="12"/>
  <text x="500" y="38" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="bold">The Diagnosis: Same DNA, Missing Catalyst</text>

  <!-- LEFT: Old Volt -->
  <rect x="30" y="60" width="440" height="410" rx="12" fill="#1a0a0a" stroke="#7f1d1d" stroke-width="1" opacity="0.9"/>
  <text x="250" y="90" text-anchor="middle" fill="#fca5a5" font-size="15" font-weight="bold">Volt v2.0 (Previous Plans)</text>
  <text x="250" y="112" text-anchor="middle" fill="#f87171" font-size="11">The Architecture Was Already Split-Brain</text>

  <!-- Hot Brain box -->
  <rect x="55" y="130" width="180" height="60" rx="8" fill="#1e1e3a" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="145" y="155" text-anchor="middle" fill="#fde68a" font-size="11" font-weight="bold">🔥 Hot Brain (CMFE)</text>
  <text x="145" y="173" text-anchor="middle" fill="#fcd34d" font-size="9">SDE Dynamics, Energy Landscape</text>

  <!-- Cold Brain box -->
  <rect x="265" y="130" width="180" height="60" rx="8" fill="#1e1e3a" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="355" y="155" text-anchor="middle" fill="#93c5fd" font-size="11" font-weight="bold">❄️ Cold Brain (HGR)</text>
  <text x="355" y="173" text-anchor="middle" fill="#93c5fd" font-size="9">Graph Reasoning, Proofs</text>

  <!-- Arrow between -->
  <line x1="235" y1="160" x2="265" y2="160" stroke="#a78bfa" stroke-width="1.5"/>
  <polygon points="262,156 270,160 262,164" fill="#a78bfa"/>

  <!-- LLL box -->
  <rect x="55" y="210" width="390" height="50" rx="8" fill="#1e1e3a" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="250" y="235" text-anchor="middle" fill="#ddd6fe" font-size="11" font-weight="bold">LLL: 384-dim Hilbert Space, VQ-VAE, HDC Algebra</text>
  <text x="250" y="250" text-anchor="middle" fill="#a78bfa" font-size="9">65K Codebook, Role-Filler Binding, Strand Architecture</text>

  <!-- Memory box -->
  <rect x="55" y="275" width="390" height="45" rx="8" fill="#1e1e3a" stroke="#6b7280" stroke-width="1.5"/>
  <text x="250" y="298" text-anchor="middle" fill="#d1d5db" font-size="11">Titans Memory + Holographic Compression + Neo4j + FAISS</text>
  <text x="250" y="313" text-anchor="middle" fill="#9ca3af" font-size="9">Three-tier storage, retrieval-augmented decompression</text>

  <!-- The Problem -->
  <rect x="55" y="340" width="390" height="115" rx="8" fill="#2a0a0a" stroke="#ef4444" stroke-width="2"/>
  <text x="250" y="365" text-anchor="middle" fill="#fca5a5" font-size="12" font-weight="bold">❌ THE PROBLEM</text>
  <text x="250" y="388" text-anchor="middle" fill="#f87171" font-size="10">• Both brains still run on GPU only</text>
  <text x="250" y="406" text-anchor="middle" fill="#f87171" font-size="10">• "Cold Brain" is cold in name, not in hardware</text>
  <text x="250" y="424" text-anchor="middle" fill="#f87171" font-size="10">• Tools/logic still approximated neurally</text>
  <text x="250" y="442" text-anchor="middle" fill="#f87171" font-size="10">• Still needs datacenter GPUs to run</text>

  <!-- RIGHT: New Volt -->
  <rect x="520" y="60" width="450" height="410" rx="12" fill="#001a10" stroke="#065f46" stroke-width="1" opacity="0.9"/>
  <text x="745" y="90" text-anchor="middle" fill="#6ee7b7" font-size="15" font-weight="bold">Volt Evolved (Rimac Moment)</text>
  <text x="745" y="112" text-anchor="middle" fill="#34d399" font-size="11">The Killer Idea: Hardware-Level Brain Split</text>

  <!-- GPU box -->
  <rect x="545" y="130" width="190" height="60" rx="8" fill="#0c1e3a" stroke="#a78bfa" stroke-width="2"/>
  <text x="640" y="155" text-anchor="middle" fill="#ddd6fe" font-size="11" font-weight="bold">🧠 GPU = Soft Core</text>
  <text x="640" y="173" text-anchor="middle" fill="#c4b5fd" font-size="9">Neural intuition ONLY</text>

  <!-- CPU box -->
  <rect x="755" y="130" width="190" height="60" rx="8" fill="#1e2a0c" stroke="#f59e0b" stroke-width="2"/>
  <text x="850" y="155" text-anchor="middle" fill="#fde68a" font-size="11" font-weight="bold">⚙️ CPU = Hard Core</text>
  <text x="850" y="173" text-anchor="middle" fill="#fcd34d" font-size="9">Deterministic Rust logic</text>

  <!-- RAM box -->
  <rect x="545" y="210" width="400" height="50" rx="8" fill="#0c2a3a" stroke="#22d3ee" stroke-width="2"/>
  <text x="745" y="235" text-anchor="middle" fill="#a5f3fc" font-size="11" font-weight="bold">💾 192GB RAM = Strand Storage (Living Memory)</text>
  <text x="745" y="250" text-anchor="middle" fill="#67e8f9" font-size="9">Millions of context strands, O(1) pointer swap, instant recall</text>

  <!-- LLL bus -->
  <rect x="545" y="275" width="400" height="45" rx="8" fill="#1e1e3a" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="745" y="298" text-anchor="middle" fill="#ddd6fe" font-size="11">LLL Vector Bus + Forward-Forward Training</text>
  <text x="745" y="313" text-anchor="middle" fill="#a78bfa" font-size="9">Train VRAM ≈ Inference VRAM • Consumer GPUs sufficient</text>

  <!-- The Solution -->
  <rect x="545" y="340" width="400" height="115" rx="8" fill="#0a2a15" stroke="#10b981" stroke-width="2"/>
  <text x="745" y="365" text-anchor="middle" fill="#6ee7b7" font-size="12" font-weight="bold">✅ THE BREAKTHROUGH</text>
  <text x="745" y="388" text-anchor="middle" fill="#34d399" font-size="10">• GPU does intuition, CPU does logic (actual split)</text>
  <text x="745" y="406" text-anchor="middle" fill="#34d399" font-size="10">• RAM becomes the brain's memory, not dead weight</text>
  <text x="745" y="424" text-anchor="middle" fill="#34d399" font-size="10">• Math is computed, not predicted (zero hallucination)</text>
  <text x="745" y="442" text-anchor="middle" fill="#34d399" font-size="10">• Runs on YOUR laptop. AI PC becomes real.</text>

  <!-- Big Arrow -->
  <line x1="480" y1="265" x2="515" y2="265" stroke="#e879f9" stroke-width="3" filter="url(#softGlow)"/>
  <polygon points="512,258 525,265 512,272" fill="#e879f9"/>
  <text x="500" y="255" text-anchor="middle" fill="#e879f9" font-size="9" font-weight="bold">CATALYST</text>
</svg>

<svg viewBox="0 0 900 400" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="oldBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1a0a0a" />
      <stop offset="100%" style="stop-color:#2a1020" />
    </linearGradient>
  </defs>

  <rect width="900" height="400" fill="url(#oldBg)" rx="12"/>
  <text x="450" y="32" text-anchor="middle" fill="#fca5a5" font-size="16" font-weight="bold">Old Volt v2.0: Sophisticated Graph RAG (Everything on GPU)</text>

  <!-- Big GPU box -->
  <rect x="40" y="50" width="820" height="310" rx="14" fill="#1a0020" stroke="#7f1d1d" stroke-width="2" stroke-dasharray="5"/>
  <text x="860" y="75" text-anchor="end" fill="#7f1d1d" font-size="11">ALL ON GPU</text>

  <!-- NL Input -->
  <rect x="70" y="80" width="120" height="45" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="1.5"/>
  <text x="130" y="100" text-anchor="middle" fill="#a5b4fc" font-size="10" font-weight="bold">NL Input</text>
  <text x="130" y="115" text-anchor="middle" fill="#818cf8" font-size="8">Tokenizer</text>

  <!-- Forward Translator -->
  <rect x="220" y="80" width="140" height="45" rx="8" fill="#1e293b" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="290" y="100" text-anchor="middle" fill="#c4b5fd" font-size="10" font-weight="bold">Translator (~1B)</text>
  <text x="290" y="115" text-anchor="middle" fill="#a78bfa" font-size="8">NL → LLL (GPU)</text>

  <line x1="190" y1="102" x2="220" y2="102" stroke="#6366f1" stroke-width="1.5"/>
  <polygon points="217,98 225,102 217,106" fill="#6366f1"/>

  <!-- Router -->
  <rect x="395" y="80" width="110" height="45" rx="8" fill="#1e293b" stroke="#10b981" stroke-width="1.5"/>
  <text x="450" y="100" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">Router</text>
  <text x="450" y="115" text-anchor="middle" fill="#34d399" font-size="8">Energy-Aware</text>

  <line x1="360" y1="102" x2="395" y2="102" stroke="#8b5cf6" stroke-width="1.5"/>
  <polygon points="392,98 400,102 392,106" fill="#8b5cf6"/>

  <!-- Hot Brain -->
  <rect x="70" y="160" width="350" height="170" rx="10" fill="#2a1500" stroke="#f59e0b" stroke-width="2"/>
  <text x="245" y="185" text-anchor="middle" fill="#fde68a" font-size="13" font-weight="bold">🔥 Hot Brain (CMFE) — ON GPU</text>

  <rect x="90" y="200" width="140" height="30" rx="5" fill="#3a2000" stroke="#d97706" stroke-width="1"/>
  <text x="160" y="220" text-anchor="middle" fill="#fcd34d" font-size="9">Vector Field Net (500M-2B)</text>

  <rect x="90" y="238" width="140" height="30" rx="5" fill="#3a2000" stroke="#d97706" stroke-width="1"/>
  <text x="160" y="258" text-anchor="middle" fill="#fcd34d" font-size="9">SDE Solver (RK4/DOPRI5)</text>

  <rect x="250" y="200" width="150" height="30" rx="5" fill="#3a2000" stroke="#d97706" stroke-width="1"/>
  <text x="325" y="220" text-anchor="middle" fill="#fcd34d" font-size="9">Diffusion Injector</text>

  <rect x="250" y="238" width="150" height="30" rx="5" fill="#3a2000" stroke="#d97706" stroke-width="1"/>
  <text x="325" y="258" text-anchor="middle" fill="#fcd34d" font-size="9">Manifold Projector</text>

  <rect x="90" y="278" width="310" height="35" rx="5" fill="#2a1000" stroke="#b45309" stroke-width="1"/>
  <text x="245" y="300" text-anchor="middle" fill="#fbbf24" font-size="9">Titans Memory (Differentiable Neural Store) — ON GPU</text>

  <!-- Cold Brain -->
  <rect x="455" y="160" width="380" height="170" rx="10" fill="#001530" stroke="#3b82f6" stroke-width="2"/>
  <text x="645" y="185" text-anchor="middle" fill="#93c5fd" font-size="13" font-weight="bold">❄️ Cold Brain (HGR) — ALSO ON GPU</text>

  <rect x="475" y="200" width="155" height="30" rx="5" fill="#001a3a" stroke="#2563eb" stroke-width="1"/>
  <text x="553" y="220" text-anchor="middle" fill="#93c5fd" font-size="9">HDC Encoder (FFT Bind)</text>

  <rect x="475" y="238" width="155" height="30" rx="5" fill="#001a3a" stroke="#2563eb" stroke-width="1"/>
  <text x="553" y="258" text-anchor="middle" fill="#93c5fd" font-size="9">Message Passing (×K)</text>

  <rect x="650" y="200" width="165" height="30" rx="5" fill="#001a3a" stroke="#2563eb" stroke-width="1"/>
  <text x="733" y="220" text-anchor="middle" fill="#93c5fd" font-size="9">Graph Constructor</text>

  <rect x="650" y="238" width="165" height="30" rx="5" fill="#001a3a" stroke="#2563eb" stroke-width="1"/>
  <text x="733" y="258" text-anchor="middle" fill="#93c5fd" font-size="9">Certainty + Proof</text>

  <rect x="475" y="278" width="340" height="35" rx="5" fill="#001030" stroke="#1d4ed8" stroke-width="1"/>
  <text x="645" y="300" text-anchor="middle" fill="#60a5fa" font-size="9">HNSW Index (65K Codebook) + Neo4j + FAISS — ALSO ON GPU/CPU MIX</text>

  <!-- Challenge-Response arrows -->
  <line x1="420" y1="240" x2="455" y2="240" stroke="#e879f9" stroke-width="2"/>
  <polygon points="452,236 460,240 452,244" fill="#e879f9"/>
  <line x1="455" y1="260" x2="420" y2="260" stroke="#e879f9" stroke-width="2"/>
  <polygon points="423,256 415,260 423,264" fill="#e879f9"/>

  <!-- Output -->
  <rect x="540" y="80" width="140" height="45" rx="8" fill="#1e293b" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="610" y="100" text-anchor="middle" fill="#c4b5fd" font-size="10" font-weight="bold">Reverse Trans</text>
  <text x="610" y="115" text-anchor="middle" fill="#a78bfa" font-size="8">LLL → NL (GPU)</text>

  <rect x="710" y="80" width="120" height="45" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="1.5"/>
  <text x="770" y="100" text-anchor="middle" fill="#a5b4fc" font-size="10" font-weight="bold">NL Output</text>
  <text x="770" y="115" text-anchor="middle" fill="#818cf8" font-size="8">+ γ + Proof</text>

  <line x1="505" y1="102" x2="540" y2="102" stroke="#10b981" stroke-width="1.5"/>
  <line x1="680" y1="102" x2="710" y2="102" stroke="#8b5cf6" stroke-width="1.5"/>

  <!-- Problem callout -->
  <rect x="250" y="370" width="400" height="25" rx="6" fill="#4a0000" stroke="#ef4444" stroke-width="1.5"/>
  <text x="450" y="388" text-anchor="middle" fill="#fca5a5" font-size="11" font-weight="bold">⚠️ Everything runs on GPU — CPU idle, RAM wasted, needs H100s</text>
</svg>




<svg viewBox="0 0 1000 480" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="threadBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0a0a1a" />
      <stop offset="100%" style="stop-color:#1a0a2e" />
    </linearGradient>
    <filter id="tGlow">
      <feGaussianBlur stdDeviation="2" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <rect width="1000" height="480" fill="url(#threadBg)" rx="12"/>
  <text x="500" y="35" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="bold">Three Severed Threads — Reunited in Volt</text>

  <!-- Thread 1: Continual Learning -->
  <rect x="30" y="60" width="300" height="390" rx="12" fill="#1a1000" stroke="#f59e0b" stroke-width="2"/>
  <text x="180" y="88" text-anchor="middle" fill="#fde68a" font-size="14" font-weight="bold">🧬 Thread 1</text>
  <text x="180" y="108" text-anchor="middle" fill="#fbbf24" font-size="12" font-weight="bold">Continual Learning</text>

  <rect x="50" y="125" width="260" height="55" rx="8" fill="#2a1a00" stroke="#d97706" stroke-width="1"/>
  <text x="180" y="148" text-anchor="middle" fill="#fcd34d" font-size="10" font-weight="bold">AlphaGo Era (2016-2019)</text>
  <text x="180" y="165" text-anchor="middle" fill="#fbbf24" font-size="9">AI evolves by playing itself. Natural.</text>

  <line x1="180" y1="180" x2="180" y2="200" stroke="#d97706" stroke-width="1.5"/>
  <polygon points="176,197 180,205 184,197" fill="#d97706"/>

  <rect x="50" y="205" width="260" height="55" rx="8" fill="#2a0a0a" stroke="#ef4444" stroke-width="1.5"/>
  <text x="180" y="228" text-anchor="middle" fill="#fca5a5" font-size="10" font-weight="bold">ChatGPT Era (2022+)</text>
  <text x="180" y="245" text-anchor="middle" fill="#f87171" font-size="9">❌ KILLED. Models are frozen at training.</text>

  <line x1="180" y1="260" x2="180" y2="280" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4"/>

  <rect x="50" y="280" width="260" height="45" rx="8" fill="#1a1000" stroke="#f59e0b" stroke-width="1" stroke-dasharray="4"/>
  <text x="180" y="300" text-anchor="middle" fill="#fbbf24" font-size="9">The assumption that AI should keep</text>
  <text x="180" y="315" text-anchor="middle" fill="#fbbf24" font-size="9">learning just... vanished from discourse.</text>

  <rect x="50" y="340" width="260" height="90" rx="8" fill="#1a2000" stroke="#10b981" stroke-width="2"/>
  <text x="180" y="363" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">⚡ Volt Resurrection</text>
  <text x="180" y="383" text-anchor="middle" fill="#34d399" font-size="9">Forward-Forward local learning</text>
  <text x="180" y="399" text-anchor="middle" fill="#34d399" font-size="9">Sleep consolidation (CPU offline)</text>
  <text x="180" y="415" text-anchor="middle" fill="#34d399" font-size="9">Strand evolution in RAM</text>

  <!-- Thread 2: Modular AI -->
  <rect x="350" y="60" width="300" height="390" rx="12" fill="#001020" stroke="#22d3ee" stroke-width="2"/>
  <text x="500" y="88" text-anchor="middle" fill="#a5f3fc" font-size="14" font-weight="bold">🧩 Thread 2</text>
  <text x="500" y="108" text-anchor="middle" fill="#22d3ee" font-size="12" font-weight="bold">Modular AI</text>

  <rect x="370" y="125" width="260" height="55" rx="8" fill="#002030" stroke="#0891b2" stroke-width="1"/>
  <text x="500" y="148" text-anchor="middle" fill="#67e8f9" font-size="10" font-weight="bold">Current State (External)</text>
  <text x="500" y="165" text-anchor="middle" fill="#22d3ee" font-size="9">n8n, LangChain, MCP — bolted on outside</text>

  <line x1="500" y1="180" x2="500" y2="200" stroke="#0891b2" stroke-width="1.5"/>
  <polygon points="496,197 500,205 504,197" fill="#0891b2"/>

  <rect x="370" y="205" width="260" height="55" rx="8" fill="#1a0030" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="500" y="228" text-anchor="middle" fill="#ddd6fe" font-size="10" font-weight="bold">The Problem</text>
  <text x="500" y="245" text-anchor="middle" fill="#c4b5fd" font-size="9">Modularity is an afterthought, not native</text>

  <line x1="500" y1="260" x2="500" y2="280" stroke="#a78bfa" stroke-width="1.5" stroke-dasharray="4"/>

  <rect x="370" y="280" width="260" height="45" rx="8" fill="#001020" stroke="#22d3ee" stroke-width="1" stroke-dasharray="4"/>
  <text x="500" y="300" text-anchor="middle" fill="#67e8f9" font-size="9">The LLM doesn't "know" it has tools.</text>
  <text x="500" y="315" text-anchor="middle" fill="#67e8f9" font-size="9">It's told via system prompts.</text>

  <rect x="370" y="340" width="260" height="90" rx="8" fill="#002020" stroke="#10b981" stroke-width="2"/>
  <text x="500" y="363" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">⚡ Volt Internalization</text>
  <text x="500" y="383" text-anchor="middle" fill="#34d399" font-size="9">Modularity IS the architecture</text>
  <text x="500" y="399" text-anchor="middle" fill="#34d399" font-size="9">Rust Trait = module interface</text>
  <text x="500" y="415" text-anchor="middle" fill="#34d399" font-size="9">CPU natively routes to tools</text>

  <!-- Thread 3: Old Volt DNA -->
  <rect x="670" y="60" width="300" height="390" rx="12" fill="#100020" stroke="#8b5cf6" stroke-width="2"/>
  <text x="820" y="88" text-anchor="middle" fill="#ddd6fe" font-size="14" font-weight="bold">📜 Thread 3</text>
  <text x="820" y="108" text-anchor="middle" fill="#a78bfa" font-size="12" font-weight="bold">Old Volt DNA</text>

  <rect x="690" y="125" width="260" height="55" rx="8" fill="#200040" stroke="#7c3aed" stroke-width="1"/>
  <text x="820" y="148" text-anchor="middle" fill="#c4b5fd" font-size="10" font-weight="bold">150 Pages of Specs</text>
  <text x="820" y="165" text-anchor="middle" fill="#a78bfa" font-size="9">LLL, Hot/Cold Brain, Strands, Safety</text>

  <line x1="820" y1="180" x2="820" y2="200" stroke="#7c3aed" stroke-width="1.5"/>
  <polygon points="816,197 820,205 824,197" fill="#7c3aed"/>

  <rect x="690" y="205" width="260" height="55" rx="8" fill="#2a0020" stroke="#ec4899" stroke-width="1.5"/>
  <text x="820" y="228" text-anchor="middle" fill="#fbcfe8" font-size="10" font-weight="bold">The Gap</text>
  <text x="820" y="245" text-anchor="middle" fill="#f9a8d4" font-size="9">No hardware grounding, no ecosystem</text>

  <line x1="820" y1="260" x2="820" y2="280" stroke="#ec4899" stroke-width="1.5" stroke-dasharray="4"/>

  <rect x="690" y="280" width="260" height="45" rx="8" fill="#100020" stroke="#8b5cf6" stroke-width="1" stroke-dasharray="4"/>
  <text x="820" y="300" text-anchor="middle" fill="#c4b5fd" font-size="9">Beautiful theory. No engine.</text>
  <text x="820" y="315" text-anchor="middle" fill="#c4b5fd" font-size="9">"A faster horse" in fancy math.</text>

  <rect x="690" y="340" width="260" height="90" rx="8" fill="#100030" stroke="#10b981" stroke-width="2"/>
  <text x="820" y="363" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">⚡ Volt Salvage</text>
  <text x="820" y="383" text-anchor="middle" fill="#34d399" font-size="9">LLL algebra → Vector Bus protocol</text>
  <text x="820" y="399" text-anchor="middle" fill="#34d399" font-size="9">Hot Brain → GPU Soft Core</text>
  <text x="820" y="415" text-anchor="middle" fill="#34d399" font-size="9">Cold Brain → CPU Hard Core + RAM</text>

  <!-- Bottom convergence -->
  <text x="500" y="470" text-anchor="middle" fill="#e2e8f0" font-size="12" font-weight="bold" filter="url(#tGlow)">Volt v3.0 = Continual Learning + Internalized Modularity + Grounded Theory</text>
</svg>

<svg viewBox="0 0 1000 550" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="learnBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0a0a1a" />
      <stop offset="100%" style="stop-color:#1a1a0a" />
    </linearGradient>
  </defs>

  <rect width="1000" height="550" fill="url(#learnBg)" rx="12"/>
  <text x="500" y="35" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="bold">Continual Learning in Volt: Three Timescales</text>
  <text x="500" y="55" text-anchor="middle" fill="#64748b" font-size="10">Inspired by biological learning: awake plasticity → sleep consolidation → developmental growth</text>

  <!-- Timescale 1: Instant (Strand Learning) -->
  <rect x="30" y="75" width="300" height="445" rx="12" fill="#001a20" stroke="#22d3ee" stroke-width="2"/>
  <text x="180" y="100" text-anchor="middle" fill="#a5f3fc" font-size="13" font-weight="bold">⚡ Instant Learning</text>
  <text x="180" y="118" text-anchor="middle" fill="#67e8f9" font-size="10">Milliseconds to minutes</text>
  <text x="180" y="136" text-anchor="middle" fill="#22d3ee" font-size="9" font-style="italic">"I just learned your name"</text>

  <rect x="50" y="155" width="260" height="35" rx="6" fill="#0c3a4a" stroke="#22d3ee" stroke-width="1"/>
  <text x="180" y="178" text-anchor="middle" fill="#cffafe" font-size="10" font-weight="bold">Hardware: RAM (Strand Storage)</text>

  <rect x="50" y="200" width="260" height="25" rx="4" fill="#083344" stroke="#0891b2" stroke-width="1"/>
  <text x="180" y="218" text-anchor="middle" fill="#67e8f9" font-size="9">Mechanism: Strand vector update</text>

  <rect x="50" y="235" width="260" height="25" rx="4" fill="#083344" stroke="#0891b2" stroke-width="1"/>
  <text x="180" y="253" text-anchor="middle" fill="#67e8f9" font-size="9">No GPU/CPU needed — pure memory write</text>

  <rect x="50" y="270" width="260" height="25" rx="4" fill="#083344" stroke="#0891b2" stroke-width="1"/>
  <text x="180" y="288" text-anchor="middle" fill="#67e8f9" font-size="9">Zero forgetting — strands are isolated</text>

  <text x="60" y="320" fill="#94a3b8" font-size="10" font-weight="bold">How it works:</text>
  <text x="60" y="340" fill="#67e8f9" font-size="9">1. User says "Call me Alex"</text>
  <text x="60" y="358" fill="#67e8f9" font-size="9">2. Personal Strand #04 updated:</text>
  <text x="70" y="376" fill="#22d3ee" font-size="9" font-family="monospace">strand.user_name = "Alex"</text>
  <text x="60" y="396" fill="#67e8f9" font-size="9">3. Persists across sessions</text>
  <text x="60" y="416" fill="#67e8f9" font-size="9">4. No weight update. Pure state.</text>

  <rect x="50" y="435" width="260" height="40" rx="6" fill="#0a2a20" stroke="#10b981" stroke-width="1"/>
  <text x="180" y="452" text-anchor="middle" fill="#6ee7b7" font-size="9" font-weight="bold">Biological analog:</text>
  <text x="180" y="468" text-anchor="middle" fill="#34d399" font-size="9">Working memory / Short-term memory</text>

  <rect x="50" y="485" width="260" height="25" rx="4" fill="#002a20" stroke="#059669" stroke-width="1.5"/>
  <text x="180" y="503" text-anchor="middle" fill="#6ee7b7" font-size="9" font-weight="bold">Old Volt: ✅ Titans memory (over-engineered)</text>

  <!-- Timescale 2: Sleep (FF Consolidation) -->
  <rect x="350" y="75" width="300" height="445" rx="12" fill="#1a1000" stroke="#f59e0b" stroke-width="2"/>
  <text x="500" y="100" text-anchor="middle" fill="#fde68a" font-size="13" font-weight="bold">🌙 Sleep Learning</text>
  <text x="500" y="118" text-anchor="middle" fill="#fcd34d" font-size="10">Hours (overnight)</text>
  <text x="500" y="136" text-anchor="middle" fill="#fbbf24" font-size="9" font-style="italic">"I got better at Rust overnight"</text>

  <rect x="370" y="155" width="260" height="35" rx="6" fill="#2a1a00" stroke="#f59e0b" stroke-width="1"/>
  <text x="500" y="178" text-anchor="middle" fill="#fef3c7" font-size="10" font-weight="bold">Hardware: CPU + GPU (idle time)</text>

  <rect x="370" y="200" width="260" height="25" rx="4" fill="#1a1000" stroke="#d97706" stroke-width="1"/>
  <text x="500" y="218" text-anchor="middle" fill="#fcd34d" font-size="9">Mechanism: Forward-Forward weight update</text>

  <rect x="370" y="235" width="260" height="25" rx="4" fill="#1a1000" stroke="#d97706" stroke-width="1"/>
  <text x="500" y="253" text-anchor="middle" fill="#fcd34d" font-size="9">Train layer-by-layer, no backprop graph</text>

  <rect x="370" y="270" width="260" height="25" rx="4" fill="#1a1000" stroke="#d97706" stroke-width="1"/>
  <text x="500" y="288" text-anchor="middle" fill="#fcd34d" font-size="9">CPU generates replay data from strands</text>

  <text x="380" y="320" fill="#94a3b8" font-size="10" font-weight="bold">How it works:</text>
  <text x="380" y="340" fill="#fcd34d" font-size="9">1. During day: accumulate "learning"</text>
  <text x="390" y="358" fill="#fcd34d" font-size="9">events in strand metadata</text>
  <text x="380" y="378" fill="#fcd34d" font-size="9">2. At night (or idle): CPU shuffles</text>
  <text x="390" y="396" fill="#fcd34d" font-size="9">and batches these events</text>
  <text x="380" y="416" fill="#fcd34d" font-size="9">3. GPU runs FF update, layer-by-layer</text>
  <text x="380" y="436" fill="#fcd34d" font-size="9">4. Weights improve. Model evolves.</text>

  <rect x="370" y="455" width="260" height="40" rx="6" fill="#2a2000" stroke="#10b981" stroke-width="1"/>
  <text x="500" y="472" text-anchor="middle" fill="#6ee7b7" font-size="9" font-weight="bold">Biological analog:</text>
  <text x="500" y="488" text-anchor="middle" fill="#34d399" font-size="9">Sleep consolidation / Memory replay</text>

  <rect x="370" y="498" width="260" height="22" rx="4" fill="#2a1a00" stroke="#b45309" stroke-width="1"/>
  <text x="500" y="514" text-anchor="middle" fill="#fbbf24" font-size="9" font-weight="bold">Old Volt: ❌ No training plan for local</text>

  <!-- Timescale 3: Developmental (Architecture Evolution) -->
  <rect x="670" y="75" width="300" height="445" rx="12" fill="#100020" stroke="#8b5cf6" stroke-width="2"/>
  <text x="820" y="100" text-anchor="middle" fill="#ddd6fe" font-size="13" font-weight="bold">🌱 Developmental Growth</text>
  <text x="820" y="118" text-anchor="middle" fill="#c4b5fd" font-size="10">Days to months</text>
  <text x="820" y="136" text-anchor="middle" fill="#a78bfa" font-size="9" font-style="italic">"I grew a new module for cooking"</text>

  <rect x="690" y="155" width="260" height="35" rx="6" fill="#200040" stroke="#8b5cf6" stroke-width="1"/>
  <text x="820" y="178" text-anchor="middle" fill="#ddd6fe" font-size="10" font-weight="bold">Hardware: RAM + Ecosystem</text>

  <rect x="690" y="200" width="260" height="25" rx="4" fill="#100020" stroke="#7c3aed" stroke-width="1"/>
  <text x="820" y="218" text-anchor="middle" fill="#c4b5fd" font-size="9">Mechanism: Module graduation + plugin</text>

  <rect x="690" y="235" width="260" height="25" rx="4" fill="#100020" stroke="#7c3aed" stroke-width="1"/>
  <text x="820" y="253" text-anchor="middle" fill="#c4b5fd" font-size="9">New strands spawn from experience</text>

  <rect x="690" y="270" width="260" height="25" rx="4" fill="#100020" stroke="#7c3aed" stroke-width="1"/>
  <text x="820" y="288" text-anchor="middle" fill="#c4b5fd" font-size="9">Community modules hot-plugged</text>

  <text x="700" y="320" fill="#94a3b8" font-size="10" font-weight="bold">How it works:</text>
  <text x="700" y="340" fill="#c4b5fd" font-size="9">1. User talks about cooking a lot</text>
  <text x="700" y="358" fill="#c4b5fd" font-size="9">2. System notices: no Cooking Strand</text>
  <text x="700" y="378" fill="#c4b5fd" font-size="9">3. Promotes temp context → new Strand</text>
  <text x="700" y="398" fill="#c4b5fd" font-size="9">4. OR: downloads community module</text>
  <text x="710" y="416" fill="#a78bfa" font-size="9" font-family="monospace">volt install translator-recipe</text>
  <text x="700" y="436" fill="#c4b5fd" font-size="9">5. New capability, zero retraining</text>

  <rect x="690" y="455" width="260" height="40" rx="6" fill="#200030" stroke="#10b981" stroke-width="1"/>
  <text x="820" y="472" text-anchor="middle" fill="#6ee7b7" font-size="9" font-weight="bold">Biological analog:</text>
  <text x="820" y="488" text-anchor="middle" fill="#34d399" font-size="9">Brain development / Neuroplasticity</text>

  <rect x="690" y="498" width="260" height="22" rx="4" fill="#200030" stroke="#6d28d9" stroke-width="1"/>
  <text x="820" y="514" text-anchor="middle" fill="#a78bfa" font-size="9" font-weight="bold">Old Volt: ✅ Mirror Module (Level 4 learning)</text>
</svg>

<svg viewBox="0 0 1000 600" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="modBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0a0a1a" />
      <stop offset="100%" style="stop-color:#0a1a2e" />
    </linearGradient>
  </defs>

  <rect width="1000" height="600" fill="url(#modBg)" rx="12"/>
  <text x="500" y="35" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="bold">Modular AI: External (Current) vs. Internal (Volt)</text>

  <!-- LEFT: External -->
  <rect x="25" y="60" width="460" height="510" rx="12" fill="#1a0a0a" stroke="#7f1d1d" stroke-width="1"/>
  <text x="255" y="88" text-anchor="middle" fill="#fca5a5" font-size="14" font-weight="bold">❌ External Modularity (LangChain/n8n + LLM)</text>

  <!-- LLM box -->
  <rect x="120" y="110" width="230" height="80" rx="10" fill="#1e293b" stroke="#ef4444" stroke-width="2"/>
  <text x="235" y="135" text-anchor="middle" fill="#fca5a5" font-size="12" font-weight="bold">LLM (Frozen Black Box)</text>
  <text x="235" y="155" text-anchor="middle" fill="#f87171" font-size="9">"You have access to these tools:"</text>
  <text x="235" y="170" text-anchor="middle" fill="#f87171" font-size="9">[system prompt injection]</text>

  <!-- JSON arrows -->
  <line x1="235" y1="190" x2="235" y2="220" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4"/>
  <text x="280" y="210" fill="#f87171" font-size="8">JSON text ↕</text>

  <!-- Orchestrator -->
  <rect x="100" y="220" width="270" height="50" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="235" y="245" text-anchor="middle" fill="#fde68a" font-size="11" font-weight="bold">Orchestrator (n8n / LangChain)</text>
  <text x="235" y="260" text-anchor="middle" fill="#fcd34d" font-size="8">Parses JSON, routes to tools, manages state</text>

  <!-- Tool boxes -->
  <line x1="150" y1="270" x2="150" y2="300" stroke="#f59e0b" stroke-width="1"/>
  <line x1="235" y1="270" x2="235" y2="300" stroke="#f59e0b" stroke-width="1"/>
  <line x1="320" y1="270" x2="320" y2="300" stroke="#f59e0b" stroke-width="1"/>

  <rect x="95" y="300" width="100" height="40" rx="6" fill="#1e293b" stroke="#6b7280" stroke-width="1"/>
  <text x="145" y="325" text-anchor="middle" fill="#d1d5db" font-size="9">Calculator</text>

  <rect x="210" y="300" width="100" height="40" rx="6" fill="#1e293b" stroke="#6b7280" stroke-width="1"/>
  <text x="260" y="325" text-anchor="middle" fill="#d1d5db" font-size="9">Search API</text>

  <rect x="325" y="300" width="100" height="40" rx="6" fill="#1e293b" stroke="#6b7280" stroke-width="1"/>
  <text x="375" y="325" text-anchor="middle" fill="#d1d5db" font-size="9">Code Exec</text>

  <!-- Problems -->
  <rect x="55" y="365" width="380" height="185" rx="8" fill="#1c0808" stroke="#991b1b" stroke-width="1"/>
  <text x="245" y="390" text-anchor="middle" fill="#fca5a5" font-size="11" font-weight="bold">Problems:</text>
  <text x="70" y="415" fill="#f87171" font-size="10">• LLM doesn't KNOW it has tools — it's told via text</text>
  <text x="70" y="435" fill="#f87171" font-size="10">• Tool selection is token prediction (can hallucinate)</text>
  <text x="70" y="455" fill="#f87171" font-size="10">• JSON parsing fails silently</text>
  <text x="70" y="475" fill="#f87171" font-size="10">• No type safety — wrong args crash at runtime</text>
  <text x="70" y="495" fill="#f87171" font-size="10">• Adding a tool = rewriting system prompt</text>
  <text x="70" y="515" fill="#f87171" font-size="10">• O(N×M) — every tool needs its own prompt eng.</text>
  <text x="70" y="535" fill="#f87171" font-size="10">• State lives outside the model — fragmented memory</text>

  <!-- RIGHT: Internal -->
  <rect x="515" y="60" width="460" height="510" rx="12" fill="#001a10" stroke="#065f46" stroke-width="1"/>
  <text x="745" y="88" text-anchor="middle" fill="#6ee7b7" font-size="14" font-weight="bold">✅ Internal Modularity (Volt)</text>

  <!-- GPU Soft Core -->
  <rect x="540" y="110" width="190" height="80" rx="10" fill="#1a0040" stroke="#7c3aed" stroke-width="2"/>
  <text x="635" y="135" text-anchor="middle" fill="#ddd6fe" font-size="11" font-weight="bold">🧠 GPU Soft Core</text>
  <text x="635" y="155" text-anchor="middle" fill="#a78bfa" font-size="9">Emits intent vectors</text>
  <text x="635" y="170" text-anchor="middle" fill="#a78bfa" font-size="9">(not tool names!)</text>

  <!-- LLL Bus -->
  <rect x="540" y="200" width="400" height="24" rx="12" fill="#2d1b69" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="740" y="217" text-anchor="middle" fill="#ddd6fe" font-size="9" font-weight="bold">⚡ LLL Vector Bus (typed, compiled interface)</text>

  <!-- CPU Hard Core -->
  <rect x="760" y="110" width="190" height="80" rx="10" fill="#1a2000" stroke="#d97706" stroke-width="2"/>
  <text x="855" y="135" text-anchor="middle" fill="#fef3c7" font-size="11" font-weight="bold">⚙️ CPU Hard Core</text>
  <text x="855" y="155" text-anchor="middle" fill="#fcd34d" font-size="9">Routes by vector similarity</text>
  <text x="855" y="170" text-anchor="middle" fill="#fcd34d" font-size="9">(not string matching!)</text>

  <!-- Module trait boxes -->
  <line x1="600" y1="224" x2="600" y2="254" stroke="#8b5cf6" stroke-width="1.5"/>
  <line x1="740" y1="224" x2="740" y2="254" stroke="#8b5cf6" stroke-width="1.5"/>
  <line x1="880" y1="224" x2="880" y2="254" stroke="#8b5cf6" stroke-width="1.5"/>

  <rect x="540" y="255" width="110" height="55" rx="6" fill="#0c3a1a" stroke="#22c55e" stroke-width="1.5"/>
  <text x="595" y="275" text-anchor="middle" fill="#86efac" font-size="9" font-weight="bold">impl HardStrand</text>
  <text x="595" y="292" text-anchor="middle" fill="#4ade80" font-size="8">MathEngine</text>
  <text x="595" y="303" text-anchor="middle" fill="#22c55e" font-size="7">Rust, compiled</text>

  <rect x="680" y="255" width="120" height="55" rx="6" fill="#0c3a1a" stroke="#22c55e" stroke-width="1.5"/>
  <text x="740" y="275" text-anchor="middle" fill="#86efac" font-size="9" font-weight="bold">impl HardStrand</text>
  <text x="740" y="292" text-anchor="middle" fill="#4ade80" font-size="8">CodeRunner</text>
  <text x="740" y="303" text-anchor="middle" fill="#22c55e" font-size="7">Rust, compiled</text>

  <rect x="830" y="255" width="110" height="55" rx="6" fill="#0c3a1a" stroke="#22c55e" stroke-width="1.5"/>
  <text x="885" y="275" text-anchor="middle" fill="#86efac" font-size="9" font-weight="bold">impl Translator</text>
  <text x="885" y="292" text-anchor="middle" fill="#4ade80" font-size="8">VisionModule</text>
  <text x="885" y="303" text-anchor="middle" fill="#22c55e" font-size="7">Community, plugged</text>

  <!-- Hot-pluggable -->
  <rect x="540" y="325" width="400" height="30" rx="6" fill="#0a2a15" stroke="#059669" stroke-width="1" stroke-dasharray="4"/>
  <text x="740" y="345" text-anchor="middle" fill="#34d399" font-size="9">↓ Hot-pluggable: any new module auto-discovered via Trait ↓</text>

  <!-- Benefits -->
  <rect x="540" y="370" width="400" height="185" rx="8" fill="#0a2015" stroke="#065f46" stroke-width="1"/>
  <text x="740" y="395" text-anchor="middle" fill="#6ee7b7" font-size="11" font-weight="bold">Benefits:</text>
  <text x="555" y="418" fill="#34d399" font-size="10">• Model KNOWS its tools — they're part of its brain</text>
  <text x="555" y="438" fill="#34d399" font-size="10">• Tool routing is vector similarity (no hallucination)</text>
  <text x="555" y="458" fill="#34d399" font-size="10">• Type-safe: Rust compiler catches bad args</text>
  <text x="555" y="478" fill="#34d399" font-size="10">• Adding a tool = implementing a Trait. Auto-discovered.</text>
  <text x="555" y="498" fill="#34d399" font-size="10">• O(N+M) — one interface, infinite modules</text>
  <text x="555" y="518" fill="#34d399" font-size="10">• State lives IN the model (Strands) — unified memory</text>
  <text x="555" y="538" fill="#34d399" font-size="10">• Modules can TEACH the model (continual learning!)</text>
</svg>












<svg viewBox="0 0 1200 600" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="tfBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f" />
      <stop offset="100%" style="stop-color:#0f0a20" />
    </linearGradient>
    <filter id="tg">
      <feGaussianBlur stdDeviation="2" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <rect width="1200" height="600" fill="url(#tfBg)" rx="12"/>
  <text x="600" y="38" text-anchor="middle" fill="#e2e8f0" font-size="19" font-weight="bold">The Shape of Thought: Three Paradigms</text>

  <!-- Paradigm 1: Token (1D) -->
  <rect x="30" y="65" width="360" height="505" rx="14" fill="#1a0a0a" stroke="#7f1d1d" stroke-width="1.5"/>
  <text x="210" y="95" text-anchor="middle" fill="#fca5a5" font-size="15" font-weight="bold">Token Stream (GPT)</text>
  <text x="210" y="115" text-anchor="middle" fill="#f87171" font-size="10">1D — Sequential — Autoregressive</text>

  <!-- Token sequence visualization -->
  <rect x="60" y="140" width="55" height="35" rx="6" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="88" y="163" text-anchor="middle" fill="#fca5a5" font-size="9">The</text>
  <line x1="115" y1="158" x2="130" y2="158" stroke="#ef4444" stroke-width="1"/>
  <rect x="130" y="140" width="55" height="35" rx="6" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="158" y="163" text-anchor="middle" fill="#fca5a5" font-size="9">cat</text>
  <line x1="185" y1="158" x2="200" y2="158" stroke="#ef4444" stroke-width="1"/>
  <rect x="200" y="140" width="55" height="35" rx="6" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="228" y="163" text-anchor="middle" fill="#fca5a5" font-size="9">sat</text>
  <line x1="255" y1="158" x2="270" y2="158" stroke="#ef4444" stroke-width="1"/>
  <rect x="270" y="140" width="55" height="35" rx="6" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="298" y="163" text-anchor="middle" fill="#fca5a5" font-size="9">on</text>
  <line x1="325" y1="158" x2="340" y2="158" stroke="#ef4444" stroke-width="1"/>
  <rect x="340" y="140" width="40" height="35" rx="6" fill="#2a0505" stroke="#991b1b" stroke-width="1" stroke-dasharray="3"/>
  <text x="360" y="163" text-anchor="middle" fill="#7f1d1d" font-size="9">?</text>

  <text x="210" y="205" text-anchor="middle" fill="#f87171" font-size="10">One word at a time. Left to right.</text>
  <text x="210" y="225" text-anchor="middle" fill="#f87171" font-size="10">To "remember" token 1 at token 1000,</text>
  <text x="210" y="245" text-anchor="middle" fill="#f87171" font-size="10">must attend across the entire sequence.</text>

  <text x="55" y="285" fill="#fca5a5" font-size="10" font-weight="bold">Problems:</text>
  <text x="55" y="305" fill="#f87171" font-size="9">• O(n²) attention for context</text>
  <text x="55" y="323" fill="#f87171" font-size="9">• Thinking = Speaking (no silent thought)</text>
  <text x="55" y="341" fill="#f87171" font-size="9">• Structure is implicit (must be learned)</text>
  <text x="55" y="359" fill="#f87171" font-size="9">• No parallel generation (strictly serial)</text>
  <text x="55" y="377" fill="#f87171" font-size="9">• Can't "look inside" a partial thought</text>
  <text x="55" y="395" fill="#f87171" font-size="9">• Lossy: syntax carries no semantics</text>

  <rect x="55" y="420" width="300" height="50" rx="8" fill="#1c0808" stroke="#991b1b" stroke-width="1"/>
  <text x="205" y="442" text-anchor="middle" fill="#fca5a5" font-size="10" font-weight="bold">Data shape: [seq_len, d_model]</text>
  <text x="205" y="460" text-anchor="middle" fill="#f87171" font-size="9">2D matrix, but semantically 1D (just a list)</text>

  <rect x="55" y="490" width="300" height="55" rx="8" fill="#1c0505" stroke="#7f1d1d" stroke-width="1"/>
  <text x="205" y="512" text-anchor="middle" fill="#f87171" font-size="9">Output: P(next_token | previous_tokens)</text>
  <text x="205" y="530" text-anchor="middle" fill="#991b1b" font-size="9">One scalar distribution per step</text>

  <!-- Paradigm 2: Vector (0D point) -->
  <rect x="420" y="65" width="360" height="505" rx="14" fill="#0a0a1a" stroke="#4338ca" stroke-width="1.5"/>
  <text x="600" y="95" text-anchor="middle" fill="#a5b4fc" font-size="15" font-weight="bold">Flat Vector (Old LLL)</text>
  <text x="600" y="115" text-anchor="middle" fill="#818cf8" font-size="10">0D — Point in Space — Holographic</text>

  <!-- Vector visualization (a single point with radiating dims) -->
  <circle cx="600" cy="175" r="30" fill="#1e1b4b" stroke="#6366f1" stroke-width="2"/>
  <text x="600" y="180" text-anchor="middle" fill="#c7d2fe" font-size="10" font-weight="bold">h ∈ ℝ⁴⁰⁹⁶</text>
  <line x1="600" y1="145" x2="600" y2="125" stroke="#6366f1" stroke-width="1" opacity="0.5"/>
  <line x1="630" y1="175" x2="650" y2="175" stroke="#6366f1" stroke-width="1" opacity="0.5"/>
  <line x1="570" y1="175" x2="550" y2="175" stroke="#6366f1" stroke-width="1" opacity="0.5"/>
  <line x1="621" y1="154" x2="636" y2="139" stroke="#6366f1" stroke-width="1" opacity="0.5"/>
  <line x1="579" y1="196" x2="564" y2="211" stroke="#6366f1" stroke-width="1" opacity="0.5"/>
  <line x1="621" y1="196" x2="636" y2="211" stroke="#6366f1" stroke-width="1" opacity="0.5"/>
  <line x1="579" y1="154" x2="564" y2="139" stroke="#6366f1" stroke-width="1" opacity="0.5"/>

  <text x="600" y="235" text-anchor="middle" fill="#818cf8" font-size="10">Entire thought = one point.</text>
  <text x="600" y="255" text-anchor="middle" fill="#818cf8" font-size="10">Structure compressed via HDC algebra.</text>

  <text x="445" y="285" fill="#a5b4fc" font-size="10" font-weight="bold">Problems:</text>
  <text x="445" y="305" fill="#818cf8" font-size="9">• No internal structure visible</text>
  <text x="445" y="323" fill="#818cf8" font-size="9">• Unbinding is noisy (~20 roles max)</text>
  <text x="445" y="341" fill="#818cf8" font-size="9">• Composition depth limited to ~3</text>
  <text x="445" y="359" fill="#818cf8" font-size="9">• Can't partially evaluate a thought</text>
  <text x="445" y="377" fill="#818cf8" font-size="9">• All-or-nothing: either converged or not</text>
  <text x="445" y="395" fill="#818cf8" font-size="9">• Still needs a decoder to become useful</text>

  <rect x="445" y="420" width="300" height="50" rx="8" fill="#1e1b4b" stroke="#4338ca" stroke-width="1"/>
  <text x="595" y="442" text-anchor="middle" fill="#a5b4fc" font-size="10" font-weight="bold">Data shape: [d=4096]</text>
  <text x="595" y="460" text-anchor="middle" fill="#818cf8" font-size="9">0D point: no axes to inspect</text>

  <rect x="445" y="490" width="300" height="55" rx="8" fill="#1e1b4b" stroke="#4338ca" stroke-width="1"/>
  <text x="595" y="512" text-anchor="middle" fill="#818cf8" font-size="9">Output: Single state h* after convergence</text>
  <text x="595" y="530" text-anchor="middle" fill="#4338ca" font-size="9">Must fully decode to use</text>

  <!-- Paradigm 3: Tensor Frame (Volt v3) -->
  <rect x="810" y="65" width="360" height="505" rx="14" fill="#001a15" stroke="#059669" stroke-width="2"/>
  <text x="990" y="95" text-anchor="middle" fill="#6ee7b7" font-size="15" font-weight="bold">Tensor Frame (Volt v3)</text>
  <text x="990" y="115" text-anchor="middle" fill="#34d399" font-size="10">3D — Structured — Multi-resolution</text>

  <!-- Tensor visualization (3D grid) -->
  <!-- Back face -->
  <rect x="880" y="135" width="100" height="70" rx="4" fill="#0a2a15" stroke="#059669" stroke-width="0.8" opacity="0.4"/>
  <!-- Side face (parallelogram) -->
  <polygon points="980,135 1020,120 1020,190 980,205" fill="#0c3a1a" stroke="#059669" stroke-width="0.8" opacity="0.4"/>
  <!-- Front face -->
  <rect x="920" y="150" width="100" height="70" rx="4" fill="#0a3a20" stroke="#22c55e" stroke-width="1.2"/>
  <!-- Grid lines on front face -->
  <line x1="945" y1="150" x2="945" y2="220" stroke="#22c55e" stroke-width="0.5" opacity="0.5"/>
  <line x1="970" y1="150" x2="970" y2="220" stroke="#22c55e" stroke-width="0.5" opacity="0.5"/>
  <line x1="995" y1="150" x2="995" y2="220" stroke="#22c55e" stroke-width="0.5" opacity="0.5"/>
  <line x1="920" y1="174" x2="1020" y2="174" stroke="#22c55e" stroke-width="0.5" opacity="0.5"/>
  <line x1="920" y1="197" x2="1020" y2="197" stroke="#22c55e" stroke-width="0.5" opacity="0.5"/>
  <!-- Labels -->
  <text x="970" y="167" text-anchor="middle" fill="#86efac" font-size="8" font-weight="bold">Slots</text>
  <text x="1030" y="167" fill="#4ade80" font-size="7" transform="rotate(-30,1030,167)">Depth</text>
  <text x="917" y="190" fill="#4ade80" font-size="7" transform="rotate(90,917,190)">Dims</text>

  <text x="990" y="247" text-anchor="middle" fill="#34d399" font-size="10">Structured, inspectable, composable.</text>
  <text x="990" y="267" text-anchor="middle" fill="#34d399" font-size="10">Each slot holds a concept at a resolution.</text>

  <text x="835" y="290" fill="#6ee7b7" font-size="10" font-weight="bold">Advantages:</text>
  <text x="835" y="310" fill="#34d399" font-size="9">• Slots = explicit structure (no HDC needed)</text>
  <text x="835" y="328" fill="#34d399" font-size="9">• Depth = multi-resolution (word → discourse)</text>
  <text x="835" y="346" fill="#34d399" font-size="9">• Can read/write individual slots</text>
  <text x="835" y="364" fill="#34d399" font-size="9">• Partial thoughts are valid (sparse slots)</text>
  <text x="835" y="382" fill="#34d399" font-size="9">• Output entire frame at once (not one token)</text>
  <text x="835" y="400" fill="#34d399" font-size="9">• Composition = slot concatenation</text>

  <rect x="835" y="420" width="300" height="50" rx="8" fill="#0a2a15" stroke="#059669" stroke-width="1.5"/>
  <text x="985" y="442" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">Data shape: [S slots, R res, D dims]</text>
  <text x="985" y="460" text-anchor="middle" fill="#34d399" font-size="9">3D tensor: addressable, structured, multi-scale</text>

  <rect x="835" y="490" width="300" height="55" rx="8" fill="#0a2a15" stroke="#059669" stroke-width="1.5"/>
  <text x="985" y="512" text-anchor="middle" fill="#34d399" font-size="9">Output: Entire structured frame in one step</text>
  <text x="985" y="530" text-anchor="middle" fill="#059669" font-size="9">Parallel decode: all slots simultaneously</text>
</svg>


<svg viewBox="0 0 1200 750" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="frameBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f" />
      <stop offset="100%" style="stop-color:#0a1a10" />
    </linearGradient>
  </defs>

  <rect width="1200" height="750" fill="url(#frameBg)" rx="12"/>
  <text x="600" y="38" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="bold">The Tensor Frame: Anatomy of a Thought</text>
  <text x="600" y="60" text-anchor="middle" fill="#64748b" font-size="10">F ∈ ℝ^[S=16 slots × R=4 resolutions × D=256 dims] — 16KB per frame</text>

  <!-- The main frame visualization -->
  <rect x="50" y="85" width="750" height="630" rx="14" fill="#0a0a15" stroke="#334155" stroke-width="1"/>

  <!-- Column headers (Slots) -->
  <text x="75" y="108" fill="#94a3b8" font-size="10" font-weight="bold">Slot →</text>
  <text x="160" y="108" text-anchor="middle" fill="#22d3ee" font-size="9" font-weight="bold">S₀</text>
  <text x="230" y="108" text-anchor="middle" fill="#22d3ee" font-size="9" font-weight="bold">S₁</text>
  <text x="300" y="108" text-anchor="middle" fill="#22d3ee" font-size="9" font-weight="bold">S₂</text>
  <text x="370" y="108" text-anchor="middle" fill="#22d3ee" font-size="9" font-weight="bold">S₃</text>
  <text x="440" y="108" text-anchor="middle" fill="#22d3ee" font-size="9" font-weight="bold">S₄</text>
  <text x="510" y="108" text-anchor="middle" fill="#22d3ee" font-size="9" font-weight="bold">S₅</text>
  <text x="580" y="108" text-anchor="middle" fill="#64748b" font-size="9">S₆</text>
  <text x="650" y="108" text-anchor="middle" fill="#64748b" font-size="9">...</text>
  <text x="720" y="108" text-anchor="middle" fill="#64748b" font-size="9">S₁₅</text>

  <!-- Row labels (Resolutions) -->
  <text x="75" y="130" fill="#94a3b8" font-size="10" font-weight="bold">Res ↓</text>

  <!-- Resolution 0: Discourse (coarsest) -->
  <text x="85" y="165" fill="#8b5cf6" font-size="9" font-weight="bold">R₀</text>
  <text x="85" y="178" fill="#7c3aed" font-size="7">Discourse</text>
  <text x="85" y="189" fill="#6d28d9" font-size="7">(coarsest)</text>

  <rect x="130" y="140" width="130" height="60" rx="6" fill="#1a0050" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="195" y="165" text-anchor="middle" fill="#c4b5fd" font-size="9" font-weight="bold">Topic Gist</text>
  <text x="195" y="180" text-anchor="middle" fill="#a78bfa" font-size="8">"Rust + bug + fix"</text>
  <text x="195" y="193" text-anchor="middle" fill="#7c3aed" font-size="7">d=256 floats</text>

  <rect x="270" y="140" width="130" height="60" rx="6" fill="#1a0050" stroke="#8b5cf6" stroke-width="1"/>
  <text x="335" y="165" text-anchor="middle" fill="#c4b5fd" font-size="9" font-weight="bold">Mood/Intent</text>
  <text x="335" y="180" text-anchor="middle" fill="#a78bfa" font-size="8">"question + urgent"</text>

  <rect x="410" y="140" width="130" height="60" rx="6" fill="#0a0030" stroke="#6d28d9" stroke-width="0.8" stroke-dasharray="3"/>
  <text x="475" y="175" text-anchor="middle" fill="#6d28d9" font-size="8">sparse (unused)</text>

  <rect x="550" y="140" width="80" height="60" rx="6" fill="#0a0030" stroke="#6d28d9" stroke-width="0.8" stroke-dasharray="3"/>
  <rect x="640" y="140" width="50" height="60" rx="6" fill="#0a0030" stroke="#6d28d9" stroke-width="0.8" stroke-dasharray="3"/>
  <rect x="700" y="140" width="60" height="60" rx="6" fill="#0a0030" stroke="#6d28d9" stroke-width="0.8" stroke-dasharray="3"/>

  <!-- Resolution 1: Proposition -->
  <text x="85" y="240" fill="#f59e0b" font-size="9" font-weight="bold">R₁</text>
  <text x="85" y="253" fill="#d97706" font-size="7">Proposition</text>
  <text x="85" y="264" fill="#b45309" font-size="7">(sentence)</text>

  <rect x="130" y="215" width="130" height="60" rx="6" fill="#1a1000" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="195" y="238" text-anchor="middle" fill="#fef3c7" font-size="9" font-weight="bold">AGENT</text>
  <text x="195" y="253" text-anchor="middle" fill="#fcd34d" font-size="8">"user (you)"</text>
  <text x="195" y="268" text-anchor="middle" fill="#d97706" font-size="7">γ = 1.0</text>

  <rect x="270" y="215" width="130" height="60" rx="6" fill="#1a1000" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="335" y="238" text-anchor="middle" fill="#fef3c7" font-size="9" font-weight="bold">PREDICATE</text>
  <text x="335" y="253" text-anchor="middle" fill="#fcd34d" font-size="8">"has_bug"</text>
  <text x="335" y="268" text-anchor="middle" fill="#d97706" font-size="7">γ = 0.85</text>

  <rect x="410" y="215" width="130" height="60" rx="6" fill="#1a1000" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="475" y="238" text-anchor="middle" fill="#fef3c7" font-size="9" font-weight="bold">PATIENT</text>
  <text x="475" y="253" text-anchor="middle" fill="#fcd34d" font-size="8">"lifetime_code"</text>
  <text x="475" y="268" text-anchor="middle" fill="#d97706" font-size="7">γ = 0.92</text>

  <rect x="550" y="215" width="130" height="60" rx="6" fill="#1a1000" stroke="#f59e0b" stroke-width="1"/>
  <text x="615" y="238" text-anchor="middle" fill="#fef3c7" font-size="9" font-weight="bold">MANNER</text>
  <text x="615" y="253" text-anchor="middle" fill="#fcd34d" font-size="8">"borrow_check"</text>

  <rect x="690" y="215" width="70" height="60" rx="6" fill="#0a0a00" stroke="#92400e" stroke-width="0.8" stroke-dasharray="3"/>

  <!-- Resolution 2: Phrase -->
  <text x="85" y="315" fill="#22d3ee" font-size="9" font-weight="bold">R₂</text>
  <text x="85" y="328" fill="#0891b2" font-size="7">Phrase</text>
  <text x="85" y="339" fill="#0e7490" font-size="7">(detail)</text>

  <rect x="130" y="290" width="130" height="60" rx="6" fill="#002530" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="195" y="313" text-anchor="middle" fill="#cffafe" font-size="8" font-weight="bold">user.identity</text>
  <text x="195" y="328" text-anchor="middle" fill="#67e8f9" font-size="7">"Alex, developer, Seoul"</text>
  <text x="195" y="343" text-anchor="middle" fill="#0891b2" font-size="7">from Personal Strand</text>

  <rect x="270" y="290" width="130" height="60" rx="6" fill="#002530" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="335" y="313" text-anchor="middle" fill="#cffafe" font-size="8" font-weight="bold">bug.type</text>
  <text x="335" y="328" text-anchor="middle" fill="#67e8f9" font-size="7">"E0502: borrow"</text>
  <text x="335" y="343" text-anchor="middle" fill="#0891b2" font-size="7">"mutable + immutable"</text>

  <rect x="410" y="290" width="130" height="60" rx="6" fill="#002530" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="475" y="313" text-anchor="middle" fill="#cffafe" font-size="8" font-weight="bold">code.context</text>
  <text x="475" y="328" text-anchor="middle" fill="#67e8f9" font-size="7">"fn process(&amp;mut self)"</text>
  <text x="475" y="343" text-anchor="middle" fill="#0891b2" font-size="7">"struct Handler"</text>

  <rect x="550" y="290" width="130" height="60" rx="6" fill="#002530" stroke="#22d3ee" stroke-width="1"/>
  <text x="615" y="313" text-anchor="middle" fill="#cffafe" font-size="8" font-weight="bold">solution.hint</text>
  <text x="615" y="328" text-anchor="middle" fill="#67e8f9" font-size="7">"clone or refactor"</text>

  <rect x="690" y="290" width="70" height="60" rx="6" fill="#001520" stroke="#0e7490" stroke-width="0.8" stroke-dasharray="3"/>

  <!-- Resolution 3: Token (finest) -->
  <text x="85" y="390" fill="#10b981" font-size="9" font-weight="bold">R₃</text>
  <text x="85" y="403" fill="#059669" font-size="7">Token</text>
  <text x="85" y="414" fill="#047857" font-size="7">(finest)</text>

  <rect x="130" y="365" width="130" height="60" rx="6" fill="#0a2015" stroke="#10b981" stroke-width="1"/>
  <text x="195" y="390" text-anchor="middle" fill="#6ee7b7" font-size="8">"Alex" "wants" "help"</text>
  <text x="195" y="408" text-anchor="middle" fill="#34d399" font-size="7">BPE subwords for output</text>
  <text x="195" y="420" text-anchor="middle" fill="#059669" font-size="7">only populated at decode</text>

  <rect x="270" y="365" width="130" height="60" rx="6" fill="#0a2015" stroke="#10b981" stroke-width="1"/>
  <text x="335" y="390" text-anchor="middle" fill="#6ee7b7" font-size="8">"cannot" "borrow"</text>
  <text x="335" y="408" text-anchor="middle" fill="#34d399" font-size="7">"*self" "as" "mutable"</text>

  <rect x="410" y="365" width="130" height="60" rx="6" fill="#0a2015" stroke="#10b981" stroke-width="1"/>
  <text x="475" y="390" text-anchor="middle" fill="#6ee7b7" font-size="8">"fn" "process" "(&amp;mut"</text>
  <text x="475" y="408" text-anchor="middle" fill="#34d399" font-size="7">"self)" "{" "..."</text>

  <rect x="550" y="365" width="130" height="60" rx="6" fill="#0a2015" stroke="#10b981" stroke-width="1"/>
  <rect x="690" y="365" width="70" height="60" rx="6" fill="#051a10" stroke="#047857" stroke-width="0.8" stroke-dasharray="3"/>

  <!-- Metadata row -->
  <text x="85" y="465" fill="#ec4899" font-size="9" font-weight="bold">Meta</text>

  <rect x="130" y="445" width="130" height="45" rx="6" fill="#1a0020" stroke="#ec4899" stroke-width="1"/>
  <text x="195" y="467" text-anchor="middle" fill="#fbcfe8" font-size="8">γ=1.0 | strand=#04</text>
  <text x="195" y="482" text-anchor="middle" fill="#f472b6" font-size="7">source: Personal</text>

  <rect x="270" y="445" width="130" height="45" rx="6" fill="#1a0020" stroke="#ec4899" stroke-width="1"/>
  <text x="335" y="467" text-anchor="middle" fill="#fbcfe8" font-size="8">γ=0.85 | needs_verify</text>
  <text x="335" y="482" text-anchor="middle" fill="#f472b6" font-size="7">source: Soft Core</text>

  <rect x="410" y="445" width="130" height="45" rx="6" fill="#1a0020" stroke="#ec4899" stroke-width="1"/>
  <text x="475" y="467" text-anchor="middle" fill="#fbcfe8" font-size="8">γ=0.92 | from_input</text>
  <text x="475" y="482" text-anchor="middle" fill="#f472b6" font-size="7">source: Translator</text>

  <rect x="550" y="445" width="130" height="45" rx="6" fill="#1a0020" stroke="#ec4899" stroke-width="1"/>
  <text x="615" y="467" text-anchor="middle" fill="#fbcfe8" font-size="8">γ=0.70 | speculative</text>
  <text x="615" y="482" text-anchor="middle" fill="#f472b6" font-size="7">source: Hot Brain</text>

  <!-- Resolution arrows on left -->
  <line x1="115" y1="150" x2="115" y2="420" stroke="#475569" stroke-width="1"/>
  <text x="108" y="285" fill="#475569" font-size="7" transform="rotate(-90,108,285)">Coarse → Fine</text>

  <!-- FRAME OPERATIONS (right side) -->
  <rect x="830" y="85" width="340" height="630" rx="14" fill="#0a0a15" stroke="#334155" stroke-width="1"/>
  <text x="1000" y="112" text-anchor="middle" fill="#e2e8f0" font-size="14" font-weight="bold">Frame Operations</text>

  <!-- Operation 1: Slot Write -->
  <rect x="855" y="130" width="290" height="70" rx="8" fill="#1a1000" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="1000" y="152" text-anchor="middle" fill="#fde68a" font-size="11" font-weight="bold">Slot Write (Direct Access)</text>
  <text x="1000" y="170" text-anchor="middle" fill="#fcd34d" font-size="9">F[slot=2, res=1] = encode("lifetime bug")</text>
  <text x="1000" y="188" text-anchor="middle" fill="#d97706" font-size="8">No need to re-encode entire thought</text>

  <!-- Operation 2: Resolution Zoom -->
  <rect x="855" y="210" width="290" height="70" rx="8" fill="#002530" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="1000" y="232" text-anchor="middle" fill="#cffafe" font-size="11" font-weight="bold">Resolution Zoom</text>
  <text x="1000" y="250" text-anchor="middle" fill="#67e8f9" font-size="9">R₀ (gist) → R₂ (detail) on demand</text>
  <text x="1000" y="268" text-anchor="middle" fill="#0891b2" font-size="8">Coarse reasoning first, refine where needed</text>

  <!-- Operation 3: Frame Merge -->
  <rect x="855" y="290" width="290" height="70" rx="8" fill="#0a2015" stroke="#10b981" stroke-width="1.5"/>
  <text x="1000" y="312" text-anchor="middle" fill="#6ee7b7" font-size="11" font-weight="bold">Frame Composition</text>
  <text x="1000" y="330" text-anchor="middle" fill="#34d399" font-size="9">F₁ ⊕ F₂ = concat slots + merge metadata</text>
  <text x="1000" y="348" text-anchor="middle" fill="#059669" font-size="8">Two thoughts become one without loss</text>

  <!-- Operation 4: Parallel Decode -->
  <rect x="855" y="370" width="290" height="70" rx="8" fill="#1a0050" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="1000" y="392" text-anchor="middle" fill="#ddd6fe" font-size="11" font-weight="bold">Parallel Decode</text>
  <text x="1000" y="410" text-anchor="middle" fill="#c4b5fd" font-size="9">ALL slots decoded simultaneously</text>
  <text x="1000" y="428" text-anchor="middle" fill="#8b5cf6" font-size="8">Not one token at a time — entire response frame</text>

  <!-- Operation 5: Sparse Attention -->
  <rect x="855" y="450" width="290" height="70" rx="8" fill="#1a0020" stroke="#ec4899" stroke-width="1.5"/>
  <text x="1000" y="472" text-anchor="middle" fill="#fbcfe8" font-size="11" font-weight="bold">Sparse Frame Attention</text>
  <text x="1000" y="490" text-anchor="middle" fill="#f9a8d4" font-size="9">Attend slot-to-slot, not token-to-token</text>
  <text x="1000" y="508" text-anchor="middle" fill="#ec4899" font-size="8">O(S²) where S=16 ≪ n=100K tokens</text>

  <!-- Operation 6: Progressive Refinement -->
  <rect x="855" y="530" width="290" height="70" rx="8" fill="#0a0a15" stroke="#94a3b8" stroke-width="1.5"/>
  <text x="1000" y="552" text-anchor="middle" fill="#e2e8f0" font-size="11" font-weight="bold">Progressive Refinement</text>
  <text x="1000" y="570" text-anchor="middle" fill="#94a3b8" font-size="9">Think at R₀ first (fast, coarse)</text>
  <text x="1000" y="588" text-anchor="middle" fill="#64748b" font-size="8">Refine to R₃ only where needed (lazy)</text>

  <!-- Memory comparison -->
  <rect x="855" y="615" width="290" height="80" rx="8" fill="#0a0a10" stroke="#475569" stroke-width="1"/>
  <text x="1000" y="637" text-anchor="middle" fill="#94a3b8" font-size="10" font-weight="bold">Memory per Frame:</text>
  <text x="1000" y="658" text-anchor="middle" fill="#e2e8f0" font-size="9">16 slots × 4 res × 256 dims × 4 bytes</text>
  <text x="1000" y="675" text-anchor="middle" fill="#22d3ee" font-size="11" font-weight="bold">= 64 KB per thought</text>
  <text x="1000" y="690" text-anchor="middle" fill="#64748b" font-size="8">(vs 2.6MB/token KV cache in transformers)</text>

  <!-- Sparsity note -->
  <rect x="130" y="510" width="630" height="80" rx="10" fill="#0a0a10" stroke="#334155" stroke-width="1"/>
  <text x="445" y="535" text-anchor="middle" fill="#94a3b8" font-size="11" font-weight="bold">Key Property: Sparsity</text>
  <text x="445" y="558" text-anchor="middle" fill="#e2e8f0" font-size="10">Most slots at most resolutions are EMPTY (sparse tensor).</text>
  <text x="445" y="578" text-anchor="middle" fill="#94a3b8" font-size="9">A simple thought uses 4 slots × 2 resolutions = 8KB. Complex reasoning fills more.</text>

  <!-- Connection to old LLL -->
  <rect x="130" y="610" width="630" height="90" rx="10" fill="#0a001a" stroke="#6d28d9" stroke-width="1.5"/>
  <text x="445" y="635" text-anchor="middle" fill="#ddd6fe" font-size="11" font-weight="bold">Backward Compatibility with Old LLL</text>
  <text x="445" y="658" text-anchor="middle" fill="#c4b5fd" font-size="9">A flat LLL vector is just a Frame collapsed to [1 slot, 1 resolution, 4096 dims]</text>
  <text x="445" y="678" text-anchor="middle" fill="#a78bfa" font-size="9">HDC binding: AGENT ⊗ CAT = writing to F[slot=AGENT, res=1] = encode(CAT)</text>
  <text x="445" y="695" text-anchor="middle" fill="#8b5cf6" font-size="8">All old algebra still works — but now you have STRUCTURE to skip it when possible</text>
</svg>

<svg viewBox="0 0 1100 520" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="outBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f" />
      <stop offset="100%" style="stop-color:#0a1a10" />
    </linearGradient>
  </defs>

  <rect width="1100" height="520" fill="url(#outBg)" rx="12"/>
  <text x="550" y="35" text-anchor="middle" fill="#e2e8f0" font-size="17" font-weight="bold">Output: Token Stream vs. Frame Emission</text>

  <!-- LEFT: Token stream -->
  <rect x="25" y="55" width="500" height="435" rx="12" fill="#1a0a0a" stroke="#7f1d1d" stroke-width="1.5"/>
  <text x="275" y="82" text-anchor="middle" fill="#fca5a5" font-size="14" font-weight="bold">❌ Autoregressive (GPT / LCM)</text>
  <text x="275" y="100" text-anchor="middle" fill="#f87171" font-size="10">Serial emission — one unit at a time</text>

  <!-- Timeline -->
  <line x1="60" y1="135" x2="490" y2="135" stroke="#991b1b" stroke-width="1"/>
  <text x="60" y="128" fill="#f87171" font-size="8">t=0</text>
  <text x="490" y="128" fill="#f87171" font-size="8">t=N</text>

  <!-- Token boxes appearing sequentially -->
  <rect x="65" y="145" width="50" height="30" rx="4" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="90" y="165" text-anchor="middle" fill="#fca5a5" font-size="8">The</text>
  <text x="90" y="185" fill="#991b1b" font-size="7">t=1</text>

  <rect x="125" y="145" width="50" height="30" rx="4" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="150" y="165" text-anchor="middle" fill="#fca5a5" font-size="8">issue</text>
  <text x="150" y="185" fill="#991b1b" font-size="7">t=2</text>

  <rect x="185" y="145" width="50" height="30" rx="4" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="210" y="165" text-anchor="middle" fill="#fca5a5" font-size="8">is</text>
  <text x="210" y="185" fill="#991b1b" font-size="7">t=3</text>

  <rect x="245" y="145" width="50" height="30" rx="4" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="270" y="165" text-anchor="middle" fill="#fca5a5" font-size="8">that</text>
  <text x="270" y="185" fill="#991b1b" font-size="7">t=4</text>

  <rect x="305" y="145" width="50" height="30" rx="4" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="330" y="165" text-anchor="middle" fill="#fca5a5" font-size="8">you</text>
  <text x="330" y="185" fill="#991b1b" font-size="7">t=5</text>

  <rect x="365" y="145" width="55" height="30" rx="4" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="393" y="165" text-anchor="middle" fill="#fca5a5" font-size="8">borrow</text>
  <text x="393" y="185" fill="#991b1b" font-size="7">t=6</text>

  <rect x="430" y="145" width="40" height="30" rx="4" fill="#1a0505" stroke="#7f1d1d" stroke-width="1" stroke-dasharray="3"/>
  <text x="450" y="165" text-anchor="middle" fill="#7f1d1d" font-size="8">...</text>
  <text x="450" y="185" fill="#7f1d1d" font-size="7">t=N</text>

  <!-- Problems list -->
  <text x="50" y="220" fill="#fca5a5" font-size="10" font-weight="bold">What happens at each step:</text>
  <text x="50" y="245" fill="#f87171" font-size="9">1. Run ENTIRE model forward pass (billions of FLOPs)</text>
  <text x="50" y="265" fill="#f87171" font-size="9">2. Produce probability distribution over vocab (50K+)</text>
  <text x="50" y="285" fill="#f87171" font-size="9">3. Sample ONE token from distribution</text>
  <text x="50" y="305" fill="#f87171" font-size="9">4. Append to KV cache (grows O(n))</text>
  <text x="50" y="325" fill="#f87171" font-size="9">5. Repeat. For every. Single. Word.</text>

  <rect x="50" y="350" width="450" height="55" rx="8" fill="#1c0808" stroke="#991b1b" stroke-width="1.5"/>
  <text x="275" y="372" text-anchor="middle" fill="#fca5a5" font-size="10" font-weight="bold">Cost: N forward passes for N tokens</text>
  <text x="275" y="392" text-anchor="middle" fill="#f87171" font-size="9">A 500-word response = 500 full model evaluations</text>

  <rect x="50" y="420" width="450" height="50" rx="8" fill="#1c0808" stroke="#7f1d1d" stroke-width="1"/>
  <text x="275" y="442" text-anchor="middle" fill="#f87171" font-size="9">Even LCM (Meta) just predicts "next sentence" — still serial.</text>
  <text x="275" y="460" text-anchor="middle" fill="#991b1b" font-size="8">P(concept_{t+1} | concept_t). Still a river flowing one way.</text>

  <!-- RIGHT: Frame emission -->
  <rect x="555" y="55" width="520" height="435" rx="12" fill="#001a10" stroke="#065f46" stroke-width="2"/>
  <text x="815" y="82" text-anchor="middle" fill="#6ee7b7" font-size="14" font-weight="bold">✅ Frame Emission (Volt)</text>
  <text x="815" y="100" text-anchor="middle" fill="#34d399" font-size="10">Parallel emission — entire structure at once</text>

  <!-- Timeline -->
  <line x1="590" y1="135" x2="1040" y2="135" stroke="#059669" stroke-width="1"/>
  <text x="590" y="128" fill="#34d399" font-size="8">t=0</text>
  <text x="750" y="128" fill="#34d399" font-size="8">t=1 (DONE)</text>

  <!-- Frame emitted all at once -->
  <rect x="600" y="145" width="420" height="40" rx="6" fill="#0a2a15" stroke="#22c55e" stroke-width="2"/>
  <!-- Slot fills -->
  <rect x="605" y="150" width="95" height="30" rx="4" fill="#0c3a1a" stroke="#22c55e" stroke-width="0.8"/>
  <text x="653" y="170" text-anchor="middle" fill="#86efac" font-size="7" font-weight="bold">AGENT slot</text>
  <rect x="705" y="150" width="95" height="30" rx="4" fill="#0c3a1a" stroke="#22c55e" stroke-width="0.8"/>
  <text x="753" y="170" text-anchor="middle" fill="#86efac" font-size="7" font-weight="bold">PRED slot</text>
  <rect x="805" y="150" width="95" height="30" rx="4" fill="#0c3a1a" stroke="#22c55e" stroke-width="0.8"/>
  <text x="853" y="170" text-anchor="middle" fill="#86efac" font-size="7" font-weight="bold">PATIENT slot</text>
  <rect x="905" y="150" width="110" height="30" rx="4" fill="#0c3a1a" stroke="#22c55e" stroke-width="0.8"/>
  <text x="960" y="170" text-anchor="middle" fill="#86efac" font-size="7" font-weight="bold">SOLUTION slot</text>

  <text x="815" y="200" text-anchor="middle" fill="#34d399" font-size="9">↑ All slots filled in ONE forward pass ↑</text>

  <!-- Then parallel decode -->
  <text x="600" y="230" fill="#6ee7b7" font-size="10" font-weight="bold">Then, parallel decode each slot to text:</text>

  <rect x="600" y="245" width="100" height="65" rx="6" fill="#0a2a15" stroke="#10b981" stroke-width="1"/>
  <text x="650" y="263" text-anchor="middle" fill="#6ee7b7" font-size="8" font-weight="bold">Slot 0 → Text</text>
  <text x="650" y="280" text-anchor="middle" fill="#34d399" font-size="7">"The issue is"</text>
  <text x="650" y="295" text-anchor="middle" fill="#059669" font-size="7">parallel ↓</text>

  <rect x="710" y="245" width="100" height="65" rx="6" fill="#0a2a15" stroke="#10b981" stroke-width="1"/>
  <text x="760" y="263" text-anchor="middle" fill="#6ee7b7" font-size="8" font-weight="bold">Slot 1 → Text</text>
  <text x="760" y="280" text-anchor="middle" fill="#34d399" font-size="7">"you borrow"</text>
  <text x="760" y="295" text-anchor="middle" fill="#059669" font-size="7">parallel ↓</text>

  <rect x="820" y="245" width="100" height="65" rx="6" fill="#0a2a15" stroke="#10b981" stroke-width="1"/>
  <text x="870" y="263" text-anchor="middle" fill="#6ee7b7" font-size="8" font-weight="bold">Slot 2 → Text</text>
  <text x="870" y="280" text-anchor="middle" fill="#34d399" font-size="7">"self mutably"</text>
  <text x="870" y="295" text-anchor="middle" fill="#059669" font-size="7">parallel ↓</text>

  <rect x="930" y="245" width="100" height="65" rx="6" fill="#0a2a15" stroke="#10b981" stroke-width="1"/>
  <text x="980" y="263" text-anchor="middle" fill="#6ee7b7" font-size="8" font-weight="bold">Slot 3 → Text</text>
  <text x="980" y="280" text-anchor="middle" fill="#34d399" font-size="7">"try cloning"</text>
  <text x="980" y="295" text-anchor="middle" fill="#059669" font-size="7">parallel ↓</text>

  <!-- Assembly -->
  <rect x="600" y="325" width="430" height="35" rx="6" fill="#002a15" stroke="#22c55e" stroke-width="1.5"/>
  <text x="815" y="348" text-anchor="middle" fill="#86efac" font-size="10">Assemble: "The issue is you borrow self mutably. Try cloning."</text>

  <rect x="600" y="375" width="430" height="55" rx="8" fill="#0a2a20" stroke="#059669" stroke-width="1.5"/>
  <text x="815" y="397" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">Cost: 1 frame pass + S parallel decodes</text>
  <text x="815" y="417" text-anchor="middle" fill="#34d399" font-size="9">500-word response ≈ 16 parallel slot decodes, not 500 serial passes</text>

  <rect x="600" y="445" width="430" height="35" rx="8" fill="#001a10" stroke="#047857" stroke-width="1"/>
  <text x="815" y="468" text-anchor="middle" fill="#34d399" font-size="9" font-weight="bold">~30x faster than autoregressive for long outputs</text>
</svg>


<svg viewBox="0 0 900 580" xmlns="http://www.w3.org/2000/svg" font-family="'Courier New', monospace">
  <defs>
    <linearGradient id="codeBg2" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1a1b26" />
      <stop offset="100%" style="stop-color:#1a1b26" />
    </linearGradient>
  </defs>

  <rect width="900" height="580" fill="url(#codeBg2)" rx="12"/>
  <rect x="0" y="0" width="900" height="36" fill="#16161e" rx="12"/>
  <rect x="0" y="20" width="900" height="16" fill="#16161e"/>
  <circle cx="20" cy="18" r="6" fill="#ff5f57"/>
  <circle cx="40" cy="18" r="6" fill="#febc2e"/>
  <circle cx="60" cy="18" r="6" fill="#28c840"/>
  <text x="450" y="22" text-anchor="middle" fill="#565f89" font-size="11">volt_core/src/frame.rs — The Tensor Frame</text>

  <text x="20" y="58" fill="#565f89" font-size="11">// ⚡ Volt Tensor Frame — the fundamental unit of thought</text>

  <text x="20" y="82" fill="#bb9af7" font-size="11">pub const</text><text x="105" y="82" fill="#ff9e64" font-size="11"> MAX_SLOTS: usize = 16;</text>
  <text x="20" y="98" fill="#bb9af7" font-size="11">pub const</text><text x="105" y="98" fill="#ff9e64" font-size="11"> NUM_RESOLUTIONS: usize = 4;</text>
  <text x="20" y="114" fill="#bb9af7" font-size="11">pub const</text><text x="105" y="114" fill="#ff9e64" font-size="11"> SLOT_DIM: usize = 256;</text>

  <text x="20" y="142" fill="#7aa2f7" font-size="11">#[derive(Clone, Debug)]</text>
  <text x="20" y="158" fill="#bb9af7" font-size="11">pub struct</text><text x="115" y="158" fill="#7dcfff" font-size="11"> TensorFrame {</text>

  <text x="40" y="178" fill="#565f89" font-size="11">/// Structured thought: [slots × resolutions × dims]</text>
  <text x="40" y="194" fill="#565f89" font-size="11">/// Sparse: most slots are None (empty)</text>
  <text x="40" y="212" fill="#c0caf5" font-size="11">pub slots: [Option&lt;SlotData&gt;; MAX_SLOTS],</text>

  <text x="40" y="236" fill="#565f89" font-size="11">/// Per-slot metadata (certainty, source, timestamp)</text>
  <text x="40" y="254" fill="#c0caf5" font-size="11">pub meta: [SlotMeta; MAX_SLOTS],</text>

  <text x="40" y="278" fill="#565f89" font-size="11">/// Frame-level: which strand, discourse type, global γ</text>
  <text x="40" y="296" fill="#c0caf5" font-size="11">pub frame_meta: FrameMeta,</text>
  <text x="20" y="314" fill="#7dcfff" font-size="11">}</text>

  <text x="20" y="342" fill="#7aa2f7" font-size="11">#[derive(Clone, Debug)]</text>
  <text x="20" y="358" fill="#bb9af7" font-size="11">pub struct</text><text x="115" y="358" fill="#7dcfff" font-size="11"> SlotData {</text>
  <text x="40" y="378" fill="#565f89" font-size="11">/// Multi-resolution embeddings for this slot</text>
  <text x="40" y="396" fill="#c0caf5" font-size="11">pub resolutions: [Option&lt;[f32; SLOT_DIM]&gt;; NUM_RESOLUTIONS],</text>
  <text x="40" y="418" fill="#565f89" font-size="11">/// R0=discourse, R1=proposition, R2=phrase, R3=token</text>
  <text x="40" y="436" fill="#c0caf5" font-size="11">pub role: SlotRole,</text>
  <text x="40" y="454" fill="#565f89" font-size="11">/// Codebook address (if quantized)</text>
  <text x="40" y="472" fill="#c0caf5" font-size="11">pub codebook_id: Option&lt;u16&gt;,</text>
  <text x="20" y="490" fill="#7dcfff" font-size="11">}</text>

  <text x="20" y="518" fill="#bb9af7" font-size="11">pub enum</text><text x="100" y="518" fill="#7dcfff" font-size="11"> SlotRole {</text>
  <text x="40" y="536" fill="#c0caf5" font-size="11">Agent, Predicate, Patient, Location, Time,</text>
  <text x="40" y="554" fill="#c0caf5" font-size="11">Manner, Instrument, Cause, Result, Free(u8),</text>
  <text x="20" y="572" fill="#7dcfff" font-size="11">}</text>
</svg>

<svg viewBox="0 0 1100 520" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="trBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f"/>
      <stop offset="100%" style="stop-color:#0a0a1a"/>
    </linearGradient>
  </defs>
  <rect width="1100" height="520" fill="url(#trBg)" rx="12"/>
  <text x="550" y="35" text-anchor="middle" fill="#e2e8f0" font-size="17" font-weight="bold">Training Volt: Four Phases</text>
  <text x="550" y="55" text-anchor="middle" fill="#64748b" font-size="10">Each phase builds on the previous one. Total: ~3-6 months on consumer hardware.</text>

  <!-- Phase 1 -->
  <rect x="30" y="75" width="240" height="420" rx="12" fill="#100020" stroke="#8b5cf6" stroke-width="2"/>
  <text x="150" y="100" text-anchor="middle" fill="#ddd6fe" font-size="13" font-weight="bold">Phase 1</text>
  <text x="150" y="118" text-anchor="middle" fill="#a78bfa" font-size="11" font-weight="bold">Bootstrap</text>
  <text x="150" y="136" text-anchor="middle" fill="#8b5cf6" font-size="9">~2-4 weeks</text>

  <rect x="45" y="152" width="210" height="28" rx="4" fill="#1a0050" stroke="#7c3aed" stroke-width="0.8"/>
  <text x="150" y="171" text-anchor="middle" fill="#c4b5fd" font-size="8">Extract knowledge from pretrained LLMs</text>
  <rect x="45" y="186" width="210" height="28" rx="4" fill="#1a0050" stroke="#7c3aed" stroke-width="0.8"/>
  <text x="150" y="205" text-anchor="middle" fill="#c4b5fd" font-size="8">Train Translators (NL ↔ Frame)</text>
  <rect x="45" y="220" width="210" height="28" rx="4" fill="#1a0050" stroke="#7c3aed" stroke-width="0.8"/>
  <text x="150" y="239" text-anchor="middle" fill="#c4b5fd" font-size="8">Initialize VQ-VAE codebook</text>
  <rect x="45" y="254" width="210" height="28" rx="4" fill="#1a0050" stroke="#7c3aed" stroke-width="0.8"/>
  <text x="150" y="273" text-anchor="middle" fill="#c4b5fd" font-size="8">Ground role vectors &amp; primitives</text>

  <text x="50" y="310" fill="#a78bfa" font-size="9" font-weight="bold">What you get:</text>
  <text x="50" y="328" fill="#8b5cf6" font-size="8">• Working Translators (in/out)</text>
  <text x="50" y="344" fill="#8b5cf6" font-size="8">• Grounded codebook (65K)</text>
  <text x="50" y="360" fill="#8b5cf6" font-size="8">• Role-Filler vocabulary</text>
  <text x="50" y="376" fill="#8b5cf6" font-size="8">• Frame ↔ NL round-trip works</text>

  <rect x="45" y="400" width="210" height="35" rx="6" fill="#0a0030" stroke="#6d28d9" stroke-width="1"/>
  <text x="150" y="418" text-anchor="middle" fill="#a78bfa" font-size="8" font-weight="bold">Training method: Standard backprop</text>
  <text x="150" y="432" text-anchor="middle" fill="#7c3aed" font-size="7">on frozen LLM backbone + thin adapters</text>

  <rect x="45" y="448" width="210" height="30" rx="4" fill="#050015" stroke="#4c1d95" stroke-width="0.8"/>
  <text x="150" y="468" text-anchor="middle" fill="#6d28d9" font-size="8">GPU: RTX 4090 sufficient (24GB)</text>

  <!-- Phase 2 -->
  <rect x="290" y="75" width="240" height="420" rx="12" fill="#001520" stroke="#0891b2" stroke-width="2"/>
  <text x="410" y="100" text-anchor="middle" fill="#a5f3fc" font-size="13" font-weight="bold">Phase 2</text>
  <text x="410" y="118" text-anchor="middle" fill="#22d3ee" font-size="11" font-weight="bold">Soft Core</text>
  <text x="410" y="136" text-anchor="middle" fill="#0891b2" font-size="9">~4-8 weeks</text>

  <rect x="305" y="152" width="210" height="28" rx="4" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="410" y="171" text-anchor="middle" fill="#67e8f9" font-size="8">Train VFN energy landscape</text>
  <rect x="305" y="186" width="210" height="28" rx="4" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="410" y="205" text-anchor="middle" fill="#67e8f9" font-size="8">Learn diffusion controller</text>
  <rect x="305" y="220" width="210" height="28" rx="4" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="410" y="239" text-anchor="middle" fill="#67e8f9" font-size="8">Train manifold projector</text>
  <rect x="305" y="254" width="210" height="28" rx="4" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="410" y="273" text-anchor="middle" fill="#67e8f9" font-size="8">Learn convergence dynamics</text>

  <text x="310" y="310" fill="#67e8f9" font-size="9" font-weight="bold">What you get:</text>
  <text x="310" y="328" fill="#0891b2" font-size="8">• Energy landscape with attractors</text>
  <text x="310" y="344" fill="#0891b2" font-size="8">• Adaptive computation time</text>
  <text x="310" y="360" fill="#0891b2" font-size="8">• Creative exploration (diffusion)</text>
  <text x="310" y="376" fill="#0891b2" font-size="8">• Thought convergence works</text>

  <rect x="305" y="400" width="210" height="35" rx="6" fill="#001a20" stroke="#0e7490" stroke-width="1"/>
  <text x="410" y="418" text-anchor="middle" fill="#67e8f9" font-size="8" font-weight="bold">Training method: Flow Matching +</text>
  <text x="410" y="432" text-anchor="middle" fill="#0891b2" font-size="7">Forward-Forward layer-local updates</text>

  <rect x="305" y="448" width="210" height="30" rx="4" fill="#001015" stroke="#0e7490" stroke-width="0.8"/>
  <text x="410" y="468" text-anchor="middle" fill="#0891b2" font-size="8">GPU: RTX 4090 sufficient (24GB)</text>

  <!-- Phase 3 -->
  <rect x="550" y="75" width="240" height="420" rx="12" fill="#0a0a05" stroke="#f59e0b" stroke-width="2"/>
  <text x="670" y="100" text-anchor="middle" fill="#fde68a" font-size="13" font-weight="bold">Phase 3</text>
  <text x="670" y="118" text-anchor="middle" fill="#fbbf24" font-size="11" font-weight="bold">Hard Core</text>
  <text x="670" y="136" text-anchor="middle" fill="#d97706" font-size="9">~2-4 weeks</text>

  <rect x="565" y="152" width="210" height="28" rx="4" fill="#1a1000" stroke="#d97706" stroke-width="0.8"/>
  <text x="670" y="171" text-anchor="middle" fill="#fcd34d" font-size="8">Wire Hard Strands (Rust, no training)</text>
  <rect x="565" y="186" width="210" height="28" rx="4" fill="#1a1000" stroke="#d97706" stroke-width="0.8"/>
  <text x="670" y="205" text-anchor="middle" fill="#fcd34d" font-size="8">Calibrate Intent Router</text>
  <rect x="565" y="220" width="210" height="28" rx="4" fill="#1a1000" stroke="#d97706" stroke-width="0.8"/>
  <text x="670" y="239" text-anchor="middle" fill="#fcd34d" font-size="8">Configure safety axioms (code, not ML)</text>
  <rect x="565" y="254" width="210" height="28" rx="4" fill="#1a1000" stroke="#d97706" stroke-width="0.8"/>
  <text x="670" y="273" text-anchor="middle" fill="#fcd34d" font-size="8">Build VoltDB storage engine</text>

  <text x="570" y="310" fill="#fcd34d" font-size="9" font-weight="bold">What you get:</text>
  <text x="570" y="328" fill="#d97706" font-size="8">• Working tool execution</text>
  <text x="570" y="344" fill="#d97706" font-size="8">• Safety layer active</text>
  <text x="570" y="360" fill="#d97706" font-size="8">• Memory management online</text>
  <text x="570" y="376" fill="#d97706" font-size="8">• Full Hot-Cold loop works</text>

  <rect x="565" y="400" width="210" height="35" rx="6" fill="#0a0a00" stroke="#92400e" stroke-width="1"/>
  <text x="670" y="418" text-anchor="middle" fill="#fcd34d" font-size="8" font-weight="bold">Training method: Mostly NONE</text>
  <text x="670" y="432" text-anchor="middle" fill="#d97706" font-size="7">Rust engineering, not ML training</text>

  <rect x="565" y="448" width="210" height="30" rx="4" fill="#0a0800" stroke="#78350f" stroke-width="0.8"/>
  <text x="670" y="468" text-anchor="middle" fill="#d97706" font-size="8">GPU: NOT NEEDED (CPU only)</text>

  <!-- Phase 4 -->
  <rect x="810" y="75" width="260" height="420" rx="12" fill="#001a10" stroke="#10b981" stroke-width="2"/>
  <text x="940" y="100" text-anchor="middle" fill="#6ee7b7" font-size="13" font-weight="bold">Phase 4</text>
  <text x="940" y="118" text-anchor="middle" fill="#34d399" font-size="11" font-weight="bold">Joint Alignment</text>
  <text x="940" y="136" text-anchor="middle" fill="#059669" font-size="9">~4-8 weeks (then ongoing)</text>

  <rect x="825" y="152" width="230" height="28" rx="4" fill="#0a2a15" stroke="#059669" stroke-width="0.8"/>
  <text x="940" y="171" text-anchor="middle" fill="#34d399" font-size="8">End-to-end system calibration</text>
  <rect x="825" y="186" width="230" height="28" rx="4" fill="#0a2a15" stroke="#059669" stroke-width="0.8"/>
  <text x="940" y="205" text-anchor="middle" fill="#34d399" font-size="8">Hot-Cold feedback loop tuning</text>
  <rect x="825" y="220" width="230" height="28" rx="4" fill="#0a2a15" stroke="#059669" stroke-width="0.8"/>
  <text x="940" y="239" text-anchor="middle" fill="#34d399" font-size="8">Certainty calibration (γ accuracy)</text>
  <rect x="825" y="254" width="230" height="28" rx="4" fill="#0a2a15" stroke="#059669" stroke-width="0.8"/>
  <text x="940" y="273" text-anchor="middle" fill="#34d399" font-size="8">Continual learning loop activation</text>

  <text x="830" y="310" fill="#34d399" font-size="9" font-weight="bold">What you get:</text>
  <text x="830" y="328" fill="#059669" font-size="8">• Full system works end-to-end</text>
  <text x="830" y="344" fill="#059669" font-size="8">• Correct certainty scores</text>
  <text x="830" y="360" fill="#059669" font-size="8">• Hot-Cold loop converges</text>
  <text x="830" y="376" fill="#059669" font-size="8">• Self-improvement activated</text>

  <rect x="825" y="400" width="230" height="35" rx="6" fill="#002a15" stroke="#047857" stroke-width="1"/>
  <text x="940" y="418" text-anchor="middle" fill="#34d399" font-size="8" font-weight="bold">Training method: RLVF + self-play +</text>
  <text x="940" y="432" text-anchor="middle" fill="#059669" font-size="7">continual FF consolidation</text>

  <rect x="825" y="448" width="230" height="30" rx="4" fill="#001a10" stroke="#047857" stroke-width="0.8"/>
  <text x="940" y="468" text-anchor="middle" fill="#059669" font-size="8">GPU: RTX 4090 (ongoing, idle time)</text>

  <!-- Arrows between phases -->
  <line x1="270" y1="280" x2="290" y2="280" stroke="#94a3b8" stroke-width="2"/>
  <polygon points="287,276 295,280 287,284" fill="#94a3b8"/>
  <line x1="530" y1="280" x2="550" y2="280" stroke="#94a3b8" stroke-width="2"/>
  <polygon points="547,276 555,280 547,284" fill="#94a3b8"/>
  <line x1="790" y1="280" x2="810" y2="280" stroke="#94a3b8" stroke-width="2"/>
  <polygon points="807,276 815,280 807,284" fill="#94a3b8"/>
</svg>

<svg viewBox="0 0 1100 380" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="ffBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f"/>
      <stop offset="100%" style="stop-color:#0a0a1a"/>
    </linearGradient>
  </defs>
  <rect width="1100" height="380" fill="url(#ffBg)" rx="12"/>
  <text x="550" y="32" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="bold">Forward-Forward vs. Backprop: VRAM During Training</text>

  <!-- LEFT: Backprop -->
  <rect x="40" y="55" width="480" height="290" rx="12" fill="#1a0a0a" stroke="#ef4444" stroke-width="1.5"/>
  <text x="280" y="82" text-anchor="middle" fill="#fca5a5" font-size="13" font-weight="bold">Standard Backprop</text>

  <!-- Stack of layers with activations stored -->
  <rect x="70" y="100" width="420" height="28" rx="4" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="280" y="119" text-anchor="middle" fill="#fca5a5" font-size="9">Layer 1: weights (200MB) + STORED activations (800MB)</text>
  <rect x="70" y="133" width="420" height="28" rx="4" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="280" y="152" text-anchor="middle" fill="#fca5a5" font-size="9">Layer 2: weights (200MB) + STORED activations (800MB)</text>
  <rect x="70" y="166" width="420" height="28" rx="4" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="280" y="185" text-anchor="middle" fill="#fca5a5" font-size="9">Layer 3: weights (200MB) + STORED activations (800MB)</text>
  <text x="280" y="210" fill="#ef4444" font-size="12">···</text>
  <rect x="70" y="220" width="420" height="28" rx="4" fill="#2a1010" stroke="#ef4444" stroke-width="1"/>
  <text x="280" y="239" text-anchor="middle" fill="#fca5a5" font-size="9">Layer N: weights (200MB) + STORED activations (800MB)</text>
  <rect x="70" y="256" width="420" height="28" rx="4" fill="#2a0505" stroke="#991b1b" stroke-width="1"/>
  <text x="280" y="275" text-anchor="middle" fill="#f87171" font-size="9">+ Optimizer states (Adam: 2× weights)</text>

  <rect x="70" y="300" width="420" height="30" rx="6" fill="#1c0505" stroke="#7f1d1d" stroke-width="1.5"/>
  <text x="280" y="320" text-anchor="middle" fill="#fca5a5" font-size="10" font-weight="bold">Total: ~24GB for 500M param VFN → Needs A100</text>

  <!-- RIGHT: Forward-Forward -->
  <rect x="560" y="55" width="500" height="290" rx="12" fill="#001a10" stroke="#10b981" stroke-width="1.5"/>
  <text x="810" y="82" text-anchor="middle" fill="#6ee7b7" font-size="13" font-weight="bold">Forward-Forward (Hinton)</text>

  <!-- Only one layer loaded at a time -->
  <rect x="590" y="100" width="440" height="28" rx="4" fill="#0a2a15" stroke="#22c55e" stroke-width="1"/>
  <text x="810" y="119" text-anchor="middle" fill="#86efac" font-size="9">Layer 1: weights (200MB) + activations (800MB) → UPDATE → DISCARD</text>

  <rect x="590" y="138" width="440" height="28" rx="4" fill="#062015" stroke="#059669" stroke-width="0.8" stroke-dasharray="3"/>
  <text x="810" y="157" text-anchor="middle" fill="#34d399" font-size="9" opacity="0.6">Layer 2: (not loaded yet — waiting for Layer 1 to finish)</text>

  <rect x="590" y="176" width="440" height="28" rx="4" fill="#051a10" stroke="#047857" stroke-width="0.8" stroke-dasharray="3"/>
  <text x="810" y="195" text-anchor="middle" fill="#059669" font-size="9" opacity="0.4">Layer 3: (not loaded yet)</text>

  <text x="810" y="225" fill="#059669" font-size="12" opacity="0.3">···</text>

  <rect x="590" y="240" width="440" height="28" rx="4" fill="#051a10" stroke="#047857" stroke-width="0.8" stroke-dasharray="3"/>
  <text x="810" y="259" text-anchor="middle" fill="#059669" font-size="9" opacity="0.3">Layer N: (not loaded yet)</text>

  <text x="810" y="290" text-anchor="middle" fill="#34d399" font-size="9">Only ONE layer in VRAM at a time. Sequential, not parallel.</text>

  <rect x="590" y="300" width="440" height="30" rx="6" fill="#002a15" stroke="#047857" stroke-width="1.5"/>
  <text x="810" y="320" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">Total: ~1GB for 500M VFN → RTX 4060 is fine</text>

  <!-- Bottom comparison -->
  <rect x="40" y="350" width="1020" height="22" rx="4" fill="#0a0a15" stroke="#334155" stroke-width="0.8"/>
  <text x="550" y="366" text-anchor="middle" fill="#94a3b8" font-size="9">Tradeoff: FF is ~3× slower (sequential layers) but uses ~24× less VRAM. For consumer GPUs, this is the only viable path.</text>
</svg>

<svg viewBox="0 0 1100 800" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="infBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f"/>
      <stop offset="100%" style="stop-color:#0a0a1a"/>
    </linearGradient>
  </defs>
  <rect width="1100" height="800" fill="url(#infBg)" rx="12"/>
  <text x="550" y="32" text-anchor="middle" fill="#e2e8f0" font-size="17" font-weight="bold">Inference Pipeline: Step by Step</text>
  <text x="550" y="52" text-anchor="middle" fill="#64748b" font-size="10">User: "I have a Rust lifetime error in my handler struct. How do I fix it?"</text>

  <!-- Step 1: Translate -->
  <rect x="40" y="72" width="1020" height="80" rx="10" fill="#0a2a15" stroke="#10b981" stroke-width="1.5"/>
  <text x="60" y="93" fill="#6ee7b7" font-size="11" font-weight="bold">Step 1: Forward Translator (GPU, ~5ms)</text>
  <text x="60" y="112" fill="#34d399" font-size="9">Frozen LLM backbone produces hidden states → Frame Projection Head maps to TensorFrame slots</text>
  <text x="60" y="132" fill="#059669" font-size="8">Output Frame: S₀=[AGENT: user, γ=1.0] S₁=[PRED: has_error, γ=0.9] S₂=[PATIENT: handler_lifetime, γ=0.85] S₃=[MANNER: fix_request, γ=0.95]</text>
  <text x="60" y="145" fill="#047857" font-size="7">Only R₀ and R₁ filled. R₂-R₃ empty (will be filled during reasoning or decoding).</text>

  <!-- Step 2: Bleed Check -->
  <rect x="40" y="162" width="1020" height="62" rx="10" fill="#0a0025" stroke="#e879f9" stroke-width="1.5"/>
  <text x="60" y="183" fill="#f0abfc" font-size="11" font-weight="bold">Step 2: Bleed Engine Prefetch (CPU async, ~2ms, non-blocking)</text>
  <text x="60" y="202" fill="#d946ef" font-size="9">Input Frame R₀ gist → HNSW query against T1 strand index → Load 47 ghost frames about "Rust lifetimes"</text>
  <text x="60" y="218" fill="#a855f7" font-size="8">Ghosts enter GPU Bleed Buffer. Energy landscape now has subtle dips near "borrow checker" and "ownership" attractors.</text>

  <!-- Step 3: GPU Soft Core -->
  <rect x="40" y="234" width="1020" height="145" rx="10" fill="#080020" stroke="#7c3aed" stroke-width="2"/>
  <text x="60" y="255" fill="#ddd6fe" font-size="11" font-weight="bold">Step 3: GPU Soft Core — SDE Dynamics (~20-100ms, adaptive)</text>

  <text x="60" y="278" fill="#c4b5fd" font-size="9">Iteration 1: VFN computes drift for each slot. S₀-S₂ converge quickly (known entities). S₃ (solution) is still foggy.</text>
  <text x="60" y="296" fill="#c4b5fd" font-size="9">Iteration 5: S₃ drifts toward ghost "borrow_checker_patterns" (from 2-month-old conversation). Page fault!</text>
  <text x="60" y="314" fill="#a78bfa" font-size="9">→ VoltDB loads full Frame from T1 into T0. S₃ now has rich context about Rc&lt;RefCell&lt;T&gt;&gt; pattern.</text>
  <text x="60" y="332" fill="#c4b5fd" font-size="9">Iteration 12: S₃ converges to "clone_or_refactor" attractor. Diffusion noise tested alternative path ("Arc&lt;Mutex&gt;") — dismissed.</text>
  <text x="60" y="350" fill="#a78bfa" font-size="9">Convergence: ‖F(t+1) - F(t)‖ &lt; ε for all slots. Total: 12 iterations × ~8ms each = ~96ms.</text>
  <text x="60" y="370" fill="#8b5cf6" font-size="8">Candidate Frame emitted with per-slot γ: [1.0, 0.9, 0.85, 0.78]. S₃ is the weakest — solution is 78% confident.</text>

  <!-- Step 4: CPU Hard Core -->
  <rect x="40" y="389" width="1020" height="130" rx="10" fill="#0a0800" stroke="#d97706" stroke-width="2"/>
  <text x="60" y="410" fill="#fde68a" font-size="11" font-weight="bold">Step 4: CPU Hard Core — Verification &amp; Tool Execution (~5-15ms)</text>

  <text x="60" y="433" fill="#fcd34d" font-size="9">4a. Intent Router: S₃ solution slot → cosine sim → MathEngine? No. CodeRunner? Marginal. HDC Verify? YES.</text>
  <text x="60" y="451" fill="#fcd34d" font-size="9">4b. HDC Algebra: Unbind S₃ → check against Coding Strand known patterns. "clone_or_refactor" is consistent with Rust semantics.</text>
  <text x="60" y="469" fill="#fcd34d" font-size="9">4c. Certainty: min(γ_S₀, γ_S₁, γ_S₂, γ_S₃) = min(1.0, 0.9, 0.85, 0.78) = 0.78. Threshold is 0.70. PASS.</text>
  <text x="60" y="487" fill="#fcd34d" font-size="9">4d. Safety: Transition Monitor checks — no axiom violations. Proof chain constructed: 4 steps.</text>
  <text x="60" y="505" fill="#d97706" font-size="8">4e. Result: Verified Frame with γ=0.78, proof=[input_parse → ghost_recall → attractor_convergence → hdc_verify]. Send to output.</text>

  <!-- Step 5: Parallel Decode -->
  <rect x="40" y="529" width="1020" height="95" rx="10" fill="#0a2a15" stroke="#10b981" stroke-width="1.5"/>
  <text x="60" y="550" fill="#6ee7b7" font-size="11" font-weight="bold">Step 5: Parallel Frame Decode — All Slots Simultaneously (~10-30ms)</text>

  <rect x="60" y="565" width="220" height="22" rx="4" fill="#002a15" stroke="#22c55e" stroke-width="0.8"/>
  <text x="170" y="581" text-anchor="middle" fill="#86efac" font-size="8">S₀→"The issue is that you"</text>
  <rect x="290" y="565" width="220" height="22" rx="4" fill="#002a15" stroke="#22c55e" stroke-width="0.8"/>
  <text x="400" y="581" text-anchor="middle" fill="#86efac" font-size="8">S₁→"borrow self mutably while"</text>
  <rect x="520" y="565" width="220" height="22" rx="4" fill="#002a15" stroke="#22c55e" stroke-width="0.8"/>
  <text x="630" y="581" text-anchor="middle" fill="#86efac" font-size="8">S₂→"an immutable ref exists"</text>
  <rect x="750" y="565" width="280" height="22" rx="4" fill="#002a15" stroke="#22c55e" stroke-width="0.8"/>
  <text x="890" y="581" text-anchor="middle" fill="#86efac" font-size="8">S₃→"Try .clone() or restructure with Rc"</text>

  <text x="60" y="610" fill="#34d399" font-size="9">Assembly: "The issue is that you borrow self mutably while an immutable ref exists. Try .clone() or restructure with Rc&lt;RefCell&gt;."</text>

  <!-- Step 6: Store -->
  <rect x="40" y="634" width="1020" height="55" rx="10" fill="#001520" stroke="#0891b2" stroke-width="1.5"/>
  <text x="60" y="655" fill="#a5f3fc" font-size="11" font-weight="bold">Step 6: VoltDB Store (CPU async, ~1ms, non-blocking)</text>
  <text x="60" y="673" fill="#67e8f9" font-size="9">New Frame → T0 working memory. Evict oldest T0 Frame → T1. Update HNSW index. Append to WAL. Log learning event.</text>
  <text x="60" y="685" fill="#0891b2" font-size="7">If T0 was full (64 Frames), the evicted Frame's R₀ ghost remains in Bleed Buffer for continued relevance.</text>

  <!-- Total -->
  <rect x="200" y="700" width="700" height="50" rx="10" fill="#0a0a15" stroke="#334155" stroke-width="2"/>
  <text x="550" y="722" text-anchor="middle" fill="#e2e8f0" font-size="13" font-weight="bold">Total Inference Time: ~130ms (simple) to ~250ms (complex)</text>
  <text x="550" y="742" text-anchor="middle" fill="#94a3b8" font-size="9">Translate: 5ms | Bleed: 2ms (async) | Think: 20-100ms | Verify: 5-15ms | Decode: 10-30ms | Store: 1ms (async)</text>

  <!-- Comparison -->
  <rect x="200" y="758" width="700" height="28" rx="6" fill="#0a0a10" stroke="#334155" stroke-width="1"/>
  <text x="550" y="777" text-anchor="middle" fill="#64748b" font-size="9">For comparison: GPT-4 typical response latency = 500-2000ms, and it can't verify, can't recall old conversations, can't use real tools.</text>
</svg>


<svg viewBox="0 0 1100 650" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="rarBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f"/>
      <stop offset="100%" style="stop-color:#0a0a1a"/>
    </linearGradient>
    <marker id="ma" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#94a3b8"/></marker>
  </defs>
  <rect width="1100" height="650" fill="url(#rarBg)" rx="12"/>
  <text x="550" y="35" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="bold">Root-Attend-Refine (RAR): The Soft Core Inference Loop</text>
  <text x="550" y="55" text-anchor="middle" fill="#64748b" font-size="10">Each iteration: slots think independently → attend to each other → refine collectively</text>

  <!-- Phase R: Root -->
  <rect x="40" y="80" width="310" height="520" rx="14" fill="#080020" stroke="#7c3aed" stroke-width="2.5"/>
  <text x="195" y="108" text-anchor="middle" fill="#ede9fe" font-size="15" font-weight="bold">R: Root</text>
  <text x="195" y="128" text-anchor="middle" fill="#a78bfa" font-size="10">"Each slot thinks alone"</text>
  <text x="195" y="146" text-anchor="middle" fill="#8b5cf6" font-size="9">Parallel — all slots simultaneously</text>

  <!-- Slot independent forward passes -->
  <rect x="60" y="165" width="270" height="55" rx="8" fill="#12004a" stroke="#a78bfa" stroke-width="1.2"/>
  <text x="195" y="185" text-anchor="middle" fill="#ddd6fe" font-size="10" font-weight="bold">Slot-Local VFN (shared weights)</text>
  <text x="195" y="202" text-anchor="middle" fill="#a78bfa" font-size="8">Same network applied to each slot independently</text>
  <text x="195" y="215" text-anchor="middle" fill="#8b5cf6" font-size="7">f_θ(S_i) → ΔS_i for each slot i in parallel</text>

  <!-- Individual slot boxes showing parallel processing -->
  <rect x="65" y="235" width="60" height="90" rx="6" fill="#1a0060" stroke="#a78bfa" stroke-width="1"/>
  <text x="95" y="253" text-anchor="middle" fill="#c4b5fd" font-size="7" font-weight="bold">S₀</text>
  <text x="95" y="267" text-anchor="middle" fill="#a78bfa" font-size="6">AGENT</text>
  <rect x="72" y="276" width="46" height="8" rx="2" fill="#7c3aed" opacity="0.8"/>
  <rect x="72" y="287" width="46" height="8" rx="2" fill="#7c3aed" opacity="0.6"/>
  <rect x="72" y="298" width="46" height="8" rx="2" fill="#7c3aed" opacity="0.4"/>
  <text x="95" y="320" text-anchor="middle" fill="#6d28d9" font-size="6">f_θ →ΔS₀</text>

  <rect x="135" y="235" width="60" height="90" rx="6" fill="#1a0060" stroke="#a78bfa" stroke-width="1"/>
  <text x="165" y="253" text-anchor="middle" fill="#c4b5fd" font-size="7" font-weight="bold">S₁</text>
  <text x="165" y="267" text-anchor="middle" fill="#a78bfa" font-size="6">PRED</text>
  <rect x="142" y="276" width="46" height="8" rx="2" fill="#7c3aed" opacity="0.7"/>
  <rect x="142" y="287" width="46" height="8" rx="2" fill="#7c3aed" opacity="0.5"/>
  <rect x="142" y="298" width="46" height="8" rx="2" fill="#7c3aed" opacity="0.3"/>
  <text x="165" y="320" text-anchor="middle" fill="#6d28d9" font-size="6">f_θ →ΔS₁</text>

  <rect x="205" y="235" width="60" height="90" rx="6" fill="#1a0060" stroke="#a78bfa" stroke-width="1"/>
  <text x="235" y="253" text-anchor="middle" fill="#c4b5fd" font-size="7" font-weight="bold">S₂</text>
  <text x="235" y="267" text-anchor="middle" fill="#a78bfa" font-size="6">PATIENT</text>
  <rect x="212" y="276" width="46" height="8" rx="2" fill="#7c3aed" opacity="0.9"/>
  <rect x="212" y="287" width="46" height="8" rx="2" fill="#7c3aed" opacity="0.7"/>
  <rect x="212" y="298" width="46" height="8" rx="2" fill="#7c3aed" opacity="0.5"/>
  <text x="235" y="320" text-anchor="middle" fill="#6d28d9" font-size="6">f_θ →ΔS₂</text>

  <rect x="275" y="235" width="50" height="90" rx="6" fill="#0f0040" stroke="#6d28d9" stroke-width="0.8" stroke-dasharray="3"/>
  <text x="300" y="280" text-anchor="middle" fill="#6d28d9" font-size="8">···</text>

  <text x="195" y="352" text-anchor="middle" fill="#c4b5fd" font-size="9" font-weight="bold">Key: Slots do NOT see each other here.</text>
  <text x="195" y="370" text-anchor="middle" fill="#a78bfa" font-size="8">Each slot is a self-contained forward pass.</text>
  <text x="195" y="388" text-anchor="middle" fill="#a78bfa" font-size="8">This is embarrassingly parallel on GPU.</text>

  <rect x="60" y="405" width="270" height="55" rx="8" fill="#0a0030" stroke="#6d28d9" stroke-width="1"/>
  <text x="195" y="425" text-anchor="middle" fill="#c4b5fd" font-size="9" font-weight="bold">Diffusion Injection (per-slot)</text>
  <text x="195" y="443" text-anchor="middle" fill="#8b5cf6" font-size="8">σ_i = f(convergence_rate_i, mirror_signal)</text>
  <text x="195" y="455" text-anchor="middle" fill="#6d28d9" font-size="7">Converged slots: σ→0 (freeze). Stuck slots: σ↑ (explore)</text>

  <rect x="60" y="475" width="270" height="35" rx="6" fill="#050020" stroke="#4c1d95" stroke-width="1"/>
  <text x="195" y="498" text-anchor="middle" fill="#8b5cf6" font-size="9">Output: 16 independent "root" vectors</text>

  <rect x="60" y="525" width="270" height="55" rx="8" fill="#050015" stroke="#4c1d95" stroke-width="1"/>
  <text x="195" y="542" text-anchor="middle" fill="#a78bfa" font-size="9" font-weight="bold">Compute cost:</text>
  <text x="195" y="560" text-anchor="middle" fill="#8b5cf6" font-size="8">16 × f_θ(256 dims) = 16 small forward passes</text>
  <text x="195" y="575" text-anchor="middle" fill="#6d28d9" font-size="7">Fully parallel on GPU — same wall time as 1 pass</text>

  <!-- Arrow: R → A -->
  <line x1="350" y1="340" x2="390" y2="340" stroke="#e879f9" stroke-width="2.5"/>
  <polygon points="387,336 395,340 387,344" fill="#e879f9"/>
  <text x="370" y="330" text-anchor="middle" fill="#e879f9" font-size="8" font-weight="bold">roots</text>

  <!-- Phase A: Attend -->
  <rect x="395" y="80" width="310" height="520" rx="14" fill="#0a1520" stroke="#22d3ee" stroke-width="2.5"/>
  <text x="550" y="108" text-anchor="middle" fill="#ecfeff" font-size="15" font-weight="bold">A: Attend</text>
  <text x="550" y="128" text-anchor="middle" fill="#67e8f9" font-size="10">"Slots inform each other"</text>
  <text x="550" y="146" text-anchor="middle" fill="#0891b2" font-size="9">Cross-slot attention — O(S²) where S=16</text>

  <!-- Attention matrix visualization -->
  <rect x="420" y="165" width="260" height="60" rx="8" fill="#0c3a4a" stroke="#22d3ee" stroke-width="1.2"/>
  <text x="550" y="185" text-anchor="middle" fill="#cffafe" font-size="10" font-weight="bold">Slot-to-Slot Attention</text>
  <text x="550" y="202" text-anchor="middle" fill="#67e8f9" font-size="8">Q=root_i, K=root_j, V=root_j → weighted messages</text>
  <text x="550" y="218" text-anchor="middle" fill="#0891b2" font-size="7">This is WHERE cross-slot reasoning happens</text>

  <!-- Attention grid (small) -->
  <rect x="435" y="240" width="200" height="130" rx="6" fill="#001a25" stroke="#0891b2" stroke-width="1"/>
  <text x="535" y="258" text-anchor="middle" fill="#a5f3fc" font-size="8" font-weight="bold">Attention Matrix A ∈ ℝ^[S×S]</text>

  <!-- Grid cells -->
  <rect x="470" y="268" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.9"/>
  <rect x="498" y="268" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.3"/>
  <rect x="526" y="268" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.1"/>
  <rect x="554" y="268" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.5"/>
  <rect x="582" y="268" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.0"/>

  <rect x="470" y="289" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.3"/>
  <rect x="498" y="289" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.9"/>
  <rect x="526" y="289" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.7"/>
  <rect x="554" y="289" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.2"/>
  <rect x="582" y="289" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.1"/>

  <rect x="470" y="310" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.1"/>
  <rect x="498" y="310" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.6"/>
  <rect x="526" y="310" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.9"/>
  <rect x="554" y="310" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.4"/>
  <rect x="582" y="310" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.2"/>

  <rect x="470" y="331" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.4"/>
  <rect x="498" y="331" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.2"/>
  <rect x="526" y="331" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.3"/>
  <rect x="554" y="331" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.9"/>
  <rect x="582" y="331" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.6"/>

  <rect x="470" y="352" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.0"/>
  <rect x="498" y="352" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.1"/>
  <rect x="526" y="352" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.2"/>
  <rect x="554" y="352" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.5"/>
  <rect x="582" y="352" width="25" height="18" rx="2" fill="#22d3ee" opacity="0.9"/>

  <!-- Labels -->
  <text x="458" y="280" text-anchor="end" fill="#67e8f9" font-size="6">S₀</text>
  <text x="458" y="301" text-anchor="end" fill="#67e8f9" font-size="6">S₁</text>
  <text x="458" y="322" text-anchor="end" fill="#67e8f9" font-size="6">S₂</text>
  <text x="458" y="343" text-anchor="end" fill="#67e8f9" font-size="6">S₃</text>
  <text x="458" y="364" text-anchor="end" fill="#67e8f9" font-size="6">S₄</text>

  <text x="550" y="390" text-anchor="middle" fill="#a5f3fc" font-size="9" font-weight="bold">Note: This is 16×16 attention, NOT 100K×100K!</text>
  <text x="550" y="408" text-anchor="middle" fill="#67e8f9" font-size="8">256 operations total. Effectively free compute.</text>

  <!-- Ghost attention -->
  <rect x="420" y="425" width="260" height="55" rx="8" fill="#0a0025" stroke="#e879f9" stroke-width="1.2"/>
  <text x="550" y="445" text-anchor="middle" fill="#f0abfc" font-size="9" font-weight="bold">+ Ghost Frame Cross-Attention</text>
  <text x="550" y="462" text-anchor="middle" fill="#d946ef" font-size="8">Active slots attend to ghost R₀ gists in Bleed Buffer</text>
  <text x="550" y="476" text-anchor="middle" fill="#a855f7" font-size="7">This is how distant memories influence current thought</text>

  <rect x="420" y="495" width="260" height="35" rx="6" fill="#001520" stroke="#0e7490" stroke-width="1"/>
  <text x="550" y="518" text-anchor="middle" fill="#67e8f9" font-size="9">Output: 16 context-aware slot vectors</text>

  <rect x="420" y="545" width="260" height="35" rx="6" fill="#001015" stroke="#0e7490" stroke-width="1"/>
  <text x="550" y="562" text-anchor="middle" fill="#0891b2" font-size="8" font-weight="bold">Compute: O(S² × D) = O(16² × 256)</text>
  <text x="550" y="576" text-anchor="middle" fill="#0e7490" font-size="7">= 65,536 multiply-adds. Negligible.</text>

  <!-- Arrow: A → Re -->
  <line x1="705" y1="340" x2="745" y2="340" stroke="#e879f9" stroke-width="2.5"/>
  <polygon points="742,336 750,340 742,344" fill="#e879f9"/>
  <text x="725" y="330" text-anchor="middle" fill="#e879f9" font-size="8" font-weight="bold">context</text>

  <!-- Phase Re: Refine -->
  <rect x="750" y="80" width="310" height="520" rx="14" fill="#0a0a05" stroke="#f59e0b" stroke-width="2.5"/>
  <text x="905" y="108" text-anchor="middle" fill="#fefce8" font-size="15" font-weight="bold">Re: Refine</text>
  <text x="905" y="128" text-anchor="middle" fill="#fcd34d" font-size="10">"Update, project, check"</text>
  <text x="905" y="146" text-anchor="middle" fill="#d97706" font-size="9">Integration + convergence detection</text>

  <!-- Refine steps -->
  <rect x="770" y="165" width="270" height="55" rx="8" fill="#1a1000" stroke="#fbbf24" stroke-width="1.2"/>
  <text x="905" y="185" text-anchor="middle" fill="#fef3c7" font-size="10" font-weight="bold">State Update (per-slot)</text>
  <text x="905" y="202" text-anchor="middle" fill="#fcd34d" font-size="8">S_i(t+1) = S_i(t) + dt × (ΔS_i + A_msg_i)</text>
  <text x="905" y="215" text-anchor="middle" fill="#d97706" font-size="7">Root contribution + Attention message combined</text>

  <rect x="770" y="235" width="270" height="50" rx="8" fill="#1a1000" stroke="#fbbf24" stroke-width="1"/>
  <text x="905" y="255" text-anchor="middle" fill="#fef3c7" font-size="10" font-weight="bold">Manifold Projection</text>
  <text x="905" y="272" text-anchor="middle" fill="#fcd34d" font-size="8">Per-slot normalization → unit hypersphere</text>
  <text x="905" y="282" text-anchor="middle" fill="#d97706" font-size="7">Prevents drift, maintains semantic validity</text>

  <rect x="770" y="300" width="270" height="55" rx="8" fill="#1a1000" stroke="#fbbf24" stroke-width="1"/>
  <text x="905" y="320" text-anchor="middle" fill="#fef3c7" font-size="10" font-weight="bold">Per-Slot Convergence Check</text>
  <text x="905" y="337" text-anchor="middle" fill="#fcd34d" font-size="8">‖S_i(t+1) − S_i(t)‖ &lt; ε_i ?</text>
  <text x="905" y="350" text-anchor="middle" fill="#d97706" font-size="7">Converged slots FREEZE. Others keep iterating.</text>

  <!-- Convergence visualization -->
  <rect x="785" y="370" width="55" height="25" rx="4" fill="#059669" opacity="0.8"/>
  <text x="813" y="387" text-anchor="middle" fill="#fff" font-size="7">S₀ ✓</text>

  <rect x="850" y="370" width="55" height="25" rx="4" fill="#059669" opacity="0.6"/>
  <text x="878" y="387" text-anchor="middle" fill="#fff" font-size="7">S₁ ✓</text>

  <rect x="915" y="370" width="55" height="25" rx="4" fill="#f59e0b" opacity="0.5"/>
  <text x="943" y="387" text-anchor="middle" fill="#fff" font-size="7">S₂ ⟳</text>

  <rect x="980" y="370" width="55" height="25" rx="4" fill="#ef4444" opacity="0.5"/>
  <text x="1008" y="387" text-anchor="middle" fill="#fff" font-size="7">S₃ ⟳</text>

  <text x="905" y="420" text-anchor="middle" fill="#fcd34d" font-size="9">S₀,S₁ converged (frozen). S₂,S₃ keep going.</text>
  <text x="905" y="438" text-anchor="middle" fill="#d97706" font-size="8">Next iteration: ONLY S₂,S₃ do Root phase.</text>
  <text x="905" y="456" text-anchor="middle" fill="#d97706" font-size="8">Converged slots still participate in Attend</text>
  <text x="905" y="474" text-anchor="middle" fill="#d97706" font-size="8">(as Key/Value, not as Query).</text>

  <rect x="770" y="495" width="270" height="35" rx="6" fill="#0a0a00" stroke="#92400e" stroke-width="1"/>
  <text x="905" y="518" text-anchor="middle" fill="#fcd34d" font-size="9">ALL slots converged? → Emit Frame to CPU</text>

  <rect x="770" y="545" width="270" height="35" rx="6" fill="#0a0800" stroke="#78350f" stroke-width="1"/>
  <text x="905" y="562" text-anchor="middle" fill="#d97706" font-size="8" font-weight="bold">Budget exceeded? → Emit best-so-far Frame</text>
  <text x="905" y="576" text-anchor="middle" fill="#92400e" font-size="7">with honest per-slot γ reflecting partial convergence</text>

  <!-- Loop arrow from Refine back to Root -->
  <path d="M 905 600 Q 905 630 550 640 Q 195 650 195 600" stroke="#e879f9" stroke-width="2" fill="none" stroke-dasharray="6"/>
  <polygon points="198,603 192,595 186,603" fill="#e879f9"/>
  <text x="550" y="638" text-anchor="middle" fill="#e879f9" font-size="9" font-weight="bold">Loop: only unconverged slots re-enter Root phase</text>
</svg>

<svg viewBox="0 0 1100 580" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="scBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f"/>
      <stop offset="100%" style="stop-color:#0a0018"/>
    </linearGradient>
  </defs>
  <rect width="1100" height="580" fill="url(#scBg)" rx="12"/>
  <text x="550" y="30" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="bold">Soft Core Complete Architecture: RAR + Frame Pipeline</text>

  <!-- Input -->
  <rect x="40" y="50" width="180" height="50" rx="8" fill="#0a2a15" stroke="#10b981" stroke-width="1.5"/>
  <text x="130" y="73" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">Input TensorFrame</text>
  <text x="130" y="90" text-anchor="middle" fill="#34d399" font-size="7">From Translator: slots at R₀-R₁</text>

  <line x1="220" y1="75" x2="260" y2="75" stroke="#10b981" stroke-width="1.5"/>
  <polygon points="257,71 265,75 257,79" fill="#10b981"/>

  <!-- Frame Register -->
  <rect x="265" y="45" width="200" height="60" rx="10" fill="#12004a" stroke="#7c3aed" stroke-width="2"/>
  <text x="365" y="68" text-anchor="middle" fill="#ddd6fe" font-size="11" font-weight="bold">Frame Register F(t)</text>
  <text x="365" y="85" text-anchor="middle" fill="#a78bfa" font-size="8">[16 slots × 4 res × 256 dims]</text>
  <text x="365" y="98" text-anchor="middle" fill="#8b5cf6" font-size="7">Sparse: only filled slots participate</text>

  <!-- RAR Loop Box -->
  <rect x="40" y="120" width="1020" height="350" rx="14" fill="#06001a" stroke="#6d28d9" stroke-width="2" stroke-dasharray="8"/>
  <text x="60" y="145" fill="#c4b5fd" font-size="11" font-weight="bold">RAR LOOP (repeat until all slots converged OR budget exhausted)</text>

  <!-- Root -->
  <rect x="60" y="160" width="280" height="140" rx="10" fill="#12004a" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="200" y="182" text-anchor="middle" fill="#ddd6fe" font-size="11" font-weight="bold">ROOT (Parallel)</text>

  <rect x="75" y="195" width="250" height="25" rx="4" fill="#1a0060" stroke="#a78bfa" stroke-width="0.8"/>
  <text x="200" y="213" text-anchor="middle" fill="#c4b5fd" font-size="8">Slot-local VFN: f_θ(S_i) → ΔS_i</text>

  <rect x="75" y="225" width="250" height="25" rx="4" fill="#1a0060" stroke="#a78bfa" stroke-width="0.8"/>
  <text x="200" y="243" text-anchor="middle" fill="#c4b5fd" font-size="8">Per-slot diffusion: σ_i orthogonal noise</text>

  <rect x="75" y="255" width="250" height="25" rx="4" fill="#1a0060" stroke="#a78bfa" stroke-width="0.8"/>
  <text x="200" y="273" text-anchor="middle" fill="#c4b5fd" font-size="8">Only active slots! Frozen slots skip.</text>

  <text x="200" y="295" text-anchor="middle" fill="#8b5cf6" font-size="7">Output: 16 root vectors (or fewer if some frozen)</text>

  <!-- Arrow R→A -->
  <line x1="340" y1="230" x2="380" y2="230" stroke="#e879f9" stroke-width="2"/>
  <polygon points="377,226 385,230 377,234" fill="#e879f9"/>

  <!-- Attend -->
  <rect x="385" y="160" width="280" height="140" rx="10" fill="#0c2a3a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="525" y="182" text-anchor="middle" fill="#cffafe" font-size="11" font-weight="bold">ATTEND (Cross-slot)</text>

  <rect x="400" y="195" width="250" height="25" rx="4" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="525" y="213" text-anchor="middle" fill="#67e8f9" font-size="8">Slot-to-slot: Q_i K_j V_j → A_ij</text>

  <rect x="400" y="225" width="250" height="25" rx="4" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="525" y="243" text-anchor="middle" fill="#67e8f9" font-size="8">Ghost attention: Q_i K_ghost → memories</text>

  <rect x="400" y="255" width="250" height="25" rx="4" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="525" y="273" text-anchor="middle" fill="#67e8f9" font-size="8">O(S²×D) = O(16²×256) ≈ negligible</text>

  <text x="525" y="295" text-anchor="middle" fill="#0891b2" font-size="7">Output: 16 context-enriched message vectors</text>

  <!-- Arrow A→Re -->
  <line x1="665" y1="230" x2="705" y2="230" stroke="#e879f9" stroke-width="2"/>
  <polygon points="702,226 710,230 702,234" fill="#e879f9"/>

  <!-- Refine -->
  <rect x="710" y="160" width="330" height="140" rx="10" fill="#1a0c00" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="875" y="182" text-anchor="middle" fill="#fef3c7" font-size="11" font-weight="bold">REFINE (Update + Check)</text>

  <rect x="725" y="195" width="300" height="25" rx="4" fill="#1a1000" stroke="#d97706" stroke-width="0.8"/>
  <text x="875" y="213" text-anchor="middle" fill="#fcd34d" font-size="8">S_i(t+1) = S_i(t) + dt × (ΔS_i + β·msg_i)</text>

  <rect x="725" y="225" width="300" height="25" rx="4" fill="#1a1000" stroke="#d97706" stroke-width="0.8"/>
  <text x="875" y="243" text-anchor="middle" fill="#fcd34d" font-size="8">Manifold projection → unit normalize</text>

  <rect x="725" y="255" width="145" height="25" rx="4" fill="#002a15" stroke="#22c55e" stroke-width="1"/>
  <text x="798" y="273" text-anchor="middle" fill="#86efac" font-size="8">Converged? → Freeze</text>

  <rect x="880" y="255" width="145" height="25" rx="4" fill="#2a0a0a" stroke="#ef4444" stroke-width="1"/>
  <text x="953" y="273" text-anchor="middle" fill="#fca5a5" font-size="8">Budget hit? → Emit</text>

  <text x="875" y="295" text-anchor="middle" fill="#d97706" font-size="7">Output: Updated Frame F(t+1) + convergence mask</text>

  <!-- Loop back arrow -->
  <path d="M 875 305 Q 875 340 525 350 Q 200 360 200 305" stroke="#e879f9" stroke-width="2" fill="none" stroke-dasharray="5"/>
  <polygon points="203,308 197,300 191,308" fill="#e879f9"/>
  <text x="525" y="365" text-anchor="middle" fill="#e879f9" font-size="8">Loop back with only unconverged slots active</text>

  <!-- Mirror Input -->
  <rect x="60" y="390" width="280" height="60" rx="8" fill="#0a0025" stroke="#e879f9" stroke-width="1"/>
  <text x="200" y="412" text-anchor="middle" fill="#f0abfc" font-size="9" font-weight="bold">Mirror Feedback (from CPU)</text>
  <text x="200" y="430" text-anchor="middle" fill="#d946ef" font-size="7">Loop detected? → spike diffusion on stuck slots</text>
  <text x="200" y="443" text-anchor="middle" fill="#a855f7" font-size="7">Low uncertainty? → reduce diffusion globally</text>

  <!-- Bleed Input -->
  <rect x="385" y="390" width="280" height="60" rx="8" fill="#001520" stroke="#22d3ee" stroke-width="1"/>
  <text x="525" y="412" text-anchor="middle" fill="#a5f3fc" font-size="9" font-weight="bold">Bleed Buffer (from VoltDB)</text>
  <text x="525" y="430" text-anchor="middle" fill="#67e8f9" font-size="7">~1000 ghost R₀ gists from T1/T2</text>
  <text x="525" y="443" text-anchor="middle" fill="#0891b2" font-size="7">Refreshed async whenever current Frame changes</text>

  <!-- OUTPUT -->
  <rect x="710" y="390" width="330" height="60" rx="8" fill="#0a0a05" stroke="#f59e0b" stroke-width="2"/>
  <text x="875" y="412" text-anchor="middle" fill="#fef3c7" font-size="10" font-weight="bold">Converged Frame → CPU Hard Core</text>
  <text x="875" y="430" text-anchor="middle" fill="#fcd34d" font-size="8">All slots frozen + per-slot γ scores</text>
  <text x="875" y="443" text-anchor="middle" fill="#d97706" font-size="7">OR: Budget-bounded partial Frame (honest γ)</text>

  <!-- Cost summary -->
  <rect x="40" y="480" width="1020" height="80" rx="10" fill="#0a0a10" stroke="#334155" stroke-width="1.5"/>
  <text x="550" y="503" text-anchor="middle" fill="#e2e8f0" font-size="12" font-weight="bold">Compute Cost Per RAR Iteration</text>

  <rect x="60" y="515" width="230" height="30" rx="5" fill="#12004a" stroke="#7c3aed" stroke-width="0.8"/>
  <text x="175" y="535" text-anchor="middle" fill="#c4b5fd" font-size="9">Root: 16 × VFN(256) ≈ 2M FLOPs</text>

  <text x="300" y="535" fill="#475569" font-size="12">+</text>

  <rect x="315" y="515" width="230" height="30" rx="5" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="430" y="535" text-anchor="middle" fill="#67e8f9" font-size="9">Attend: 16²×256 ≈ 65K FLOPs</text>

  <text x="555" y="535" fill="#475569" font-size="12">+</text>

  <rect x="570" y="515" width="230" height="30" rx="5" fill="#1a1000" stroke="#d97706" stroke-width="0.8"/>
  <text x="685" y="535" text-anchor="middle" fill="#fcd34d" font-size="9">Refine: 16 × 256 ≈ 4K FLOPs</text>

  <text x="810" y="535" fill="#475569" font-size="12">=</text>

  <rect x="825" y="515" width="215" height="30" rx="5" fill="#0a2a15" stroke="#10b981" stroke-width="1.5"/>
  <text x="933" y="535" text-anchor="middle" fill="#6ee7b7" font-size="9" font-weight="bold">~2.1M FLOPs / iteration</text>
</svg>

<svg viewBox="0 0 1100 680" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="s1Bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f"/>
      <stop offset="100%" style="stop-color:#0a0a1a"/>
    </linearGradient>
  </defs>
  <rect width="1100" height="680" fill="url(#s1Bg)" rx="12"/>
  <text x="550" y="30" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="bold">Scenario 1: "What's my name?" — Simple Recall (~15ms total)</text>
  <text x="550" y="50" text-anchor="middle" fill="#64748b" font-size="10">Active strand: Personal #04 | Difficulty: Trivial | RAR iterations: 2</text>

  <!-- Timeline header -->
  <rect x="40" y="70" width="1020" height="25" rx="4" fill="#1e293b" stroke="#334155" stroke-width="0.5"/>
  <text x="55" y="87" fill="#94a3b8" font-size="9" font-weight="bold">Time</text>
  <text x="180" y="87" fill="#94a3b8" font-size="9" font-weight="bold">Component</text>
  <text x="450" y="87" fill="#94a3b8" font-size="9" font-weight="bold">Frame State (slots that changed this step)</text>
  <text x="900" y="87" fill="#94a3b8" font-size="9" font-weight="bold">γ scores</text>

  <!-- t=0ms: Input arrives -->
  <rect x="40" y="100" width="1020" height="55" rx="6" fill="#0a2a15" stroke="#10b981" stroke-width="1"/>
  <text x="55" y="118" fill="#6ee7b7" font-size="9" font-weight="bold">0ms</text>
  <text x="180" y="118" fill="#34d399" font-size="9">Translator</text>
  <text x="450" y="118" fill="#34d399" font-size="9">Input parsed into Frame:</text>
  <rect x="450" y="125" width="120" height="20" rx="3" fill="#002a15" stroke="#22c55e" stroke-width="0.8"/>
  <text x="510" y="139" text-anchor="middle" fill="#86efac" font-size="7">S₀ AGENT: "user"</text>
  <rect x="580" y="125" width="150" height="20" rx="3" fill="#002a15" stroke="#22c55e" stroke-width="0.8"/>
  <text x="655" y="139" text-anchor="middle" fill="#86efac" font-size="7">S₁ PRED: "query_name"</text>
  <rect x="740" y="125" width="100" height="20" rx="3" fill="#051a10" stroke="#047857" stroke-width="0.8" stroke-dasharray="2"/>
  <text x="790" y="139" text-anchor="middle" fill="#059669" font-size="7">S₂ RESULT: ∅</text>
  <text x="900" y="125" fill="#94a3b8" font-size="8">[1.0, 0.9, —]</text>

  <!-- t=2ms: Bleed check -->
  <rect x="40" y="162" width="1020" height="45" rx="6" fill="#0a0025" stroke="#e879f9" stroke-width="1"/>
  <text x="55" y="182" fill="#f0abfc" font-size="9" font-weight="bold">2ms</text>
  <text x="180" y="182" fill="#d946ef" font-size="9">Bleed Engine</text>
  <text x="450" y="182" fill="#d946ef" font-size="9">Query R₀ gist against Personal Strand → DIRECT HIT</text>
  <text x="450" y="198" fill="#a855f7" font-size="8">Ghost loaded: Personal Frame #847 "user_identity: Alex, developer, Seoul" — already in T0!</text>

  <!-- t=3ms: RAR Iteration 1 -->
  <rect x="40" y="214" width="1020" height="110" rx="6" fill="#080020" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="55" y="234" fill="#ddd6fe" font-size="9" font-weight="bold">3ms</text>
  <text x="180" y="234" fill="#c4b5fd" font-size="9">RAR Iter 1</text>

  <text x="180" y="254" fill="#a78bfa" font-size="8" font-weight="bold">ROOT:</text>
  <text x="250" y="254" fill="#a78bfa" font-size="8">S₀ VFN → drift toward "identity_query" attractor (strong, σ≈0)</text>
  <text x="250" y="268" fill="#a78bfa" font-size="8">S₁ VFN → drift toward "name_retrieval" attractor (strong, σ≈0)</text>
  <text x="250" y="282" fill="#a78bfa" font-size="8">S₂ VFN → foggy, no clear direction yet (σ moderate)</text>

  <text x="180" y="302" fill="#67e8f9" font-size="8" font-weight="bold">ATTEND:</text>
  <text x="250" y="302" fill="#67e8f9" font-size="8">S₂ attends to ghost Frame #847 → cosine sim 0.94 → PAGE FAULT → full load</text>
  <text x="250" y="316" fill="#22d3ee" font-size="8">Ghost resolves: S₂ now receives "Alex" vector from Personal memory</text>

  <text x="900" y="254" fill="#94a3b8" font-size="8">[1.0, 0.95, 0.3]</text>
  <text x="900" y="302" fill="#94a3b8" font-size="8">[1.0, 0.98, 0.92]</text>
  <text x="970" y="302" fill="#22c55e" font-size="7">← big jump</text>

  <!-- t=5ms: RAR Iteration 2 -->
  <rect x="40" y="331" width="1020" height="95" rx="6" fill="#080020" stroke="#7c3aed" stroke-width="1"/>
  <text x="55" y="351" fill="#ddd6fe" font-size="9" font-weight="bold">5ms</text>
  <text x="180" y="351" fill="#c4b5fd" font-size="9">RAR Iter 2</text>

  <text x="180" y="371" fill="#a78bfa" font-size="8" font-weight="bold">ROOT:</text>
  <text x="250" y="371" fill="#a78bfa" font-size="8">S₀ → ‖ΔS₀‖ = 0.003 → CONVERGED ✓ (frozen)</text>
  <text x="250" y="385" fill="#a78bfa" font-size="8">S₁ → ‖ΔS₁‖ = 0.008 → CONVERGED ✓ (frozen)</text>
  <text x="250" y="399" fill="#a78bfa" font-size="8">S₂ → ‖ΔS₂‖ = 0.005 → CONVERGED ✓ (frozen) — snapped to "Alex" attractor</text>

  <text x="180" y="419" fill="#22c55e" font-size="8" font-weight="bold">ALL SLOTS CONVERGED → Emit Frame to CPU</text>

  <text x="900" y="371" fill="#22c55e" font-size="8">[1.0, 0.98, 0.97]</text>

  <!-- t=7ms: CPU verification -->
  <rect x="40" y="433" width="1020" height="55" rx="6" fill="#0a0800" stroke="#d97706" stroke-width="1"/>
  <text x="55" y="453" fill="#fde68a" font-size="9" font-weight="bold">7ms</text>
  <text x="180" y="453" fill="#fcd34d" font-size="9">CPU Hard Core</text>
  <text x="450" y="453" fill="#fcd34d" font-size="9">Intent Router → no tool needed. HDC verify: S₂ = codebook match for "Alex".</text>
  <text x="450" y="470" fill="#d97706" font-size="8">Safety: PASS. Certainty: min(1.0, 0.98, 0.97) = 0.97 ≥ 0.70 threshold. Proof: 2 steps.</text>
  <text x="900" y="460" fill="#22c55e" font-size="8">γ_final = 0.97</text>

  <!-- t=10ms: Decode -->
  <rect x="40" y="495" width="1020" height="50" rx="6" fill="#0a2a15" stroke="#10b981" stroke-width="1"/>
  <text x="55" y="515" fill="#6ee7b7" font-size="9" font-weight="bold">10ms</text>
  <text x="180" y="515" fill="#34d399" font-size="9">Action Core</text>
  <text x="450" y="515" fill="#34d399" font-size="9">Parallel decode: S₀→"Your" S₁→"name is" S₂→"Alex" → Assemble</text>
  <text x="450" y="535" fill="#059669" font-size="8">Output: "Your name is Alex." (γ=0.97, proof_len=2, strand=Personal#04)</text>

  <!-- t=11ms: Store -->
  <rect x="40" y="552" width="1020" height="38" rx="6" fill="#001520" stroke="#0891b2" stroke-width="1"/>
  <text x="55" y="572" fill="#a5f3fc" font-size="9" font-weight="bold">11ms</text>
  <text x="180" y="572" fill="#67e8f9" font-size="9">VoltDB</text>
  <text x="450" y="572" fill="#67e8f9" font-size="9">Frame stored in T0. No eviction needed (buffer not full). Learning event: none (trivial recall).</text>

  <!-- Summary -->
  <rect x="200" y="605" width="700" height="55" rx="10" fill="#0a0a15" stroke="#334155" stroke-width="1.5"/>
  <text x="550" y="625" text-anchor="middle" fill="#e2e8f0" font-size="11" font-weight="bold">Total: 11ms | RAR iterations: 2 | Ghost activations: 1 | Tools used: 0</text>
  <text x="550" y="645" text-anchor="middle" fill="#94a3b8" font-size="9">This is the "reflex" path. The Personal strand had the answer cached. Minimal compute.</text>
</svg>

<svg viewBox="0 0 1100 1050" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="s2Bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f"/>
      <stop offset="100%" style="stop-color:#0a0a1a"/>
    </linearGradient>
  </defs>
  <rect width="1100" height="1050" fill="url(#s2Bg)" rx="12"/>
  <text x="550" y="30" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="bold">Scenario 2: "Is Rc&lt;RefCell&lt;T&gt;&gt; thread-safe?" — Reasoning (~95ms)</text>
  <text x="550" y="50" text-anchor="middle" fill="#64748b" font-size="10">Active strand: Coding #01 | Difficulty: Medium | RAR iterations: 8 | Ghost activations: 3</text>

  <!-- Timeline -->
  <rect x="40" y="68" width="1020" height="22" rx="3" fill="#1e293b"/>
  <text x="55" y="83" fill="#94a3b8" font-size="8" font-weight="bold">Time</text>
  <text x="120" y="83" fill="#94a3b8" font-size="8" font-weight="bold">Event</text>
  <text x="550" y="83" fill="#94a3b8" font-size="8" font-weight="bold">Slot Evolution (showing what changes each step)</text>

  <!-- t=0: Input -->
  <rect x="40" y="95" width="1020" height="50" rx="6" fill="#0a2a15" stroke="#10b981" stroke-width="1"/>
  <text x="55" y="113" fill="#6ee7b7" font-size="8" font-weight="bold">0ms</text>
  <text x="120" y="113" fill="#34d399" font-size="8">Translate</text>
  <text x="250" y="113" fill="#34d399" font-size="8">S₀[SUBJECT: Rc&lt;RefCell&lt;T&gt;&gt;] S₁[PRED: is_thread_safe?] S₂[PROPERTY: ∅] S₃[REASON: ∅] S₄[CONCLUSION: ∅]</text>
  <text x="250" y="132" fill="#059669" font-size="7">γ: [0.95, 0.90, —, —, —] | 5 slots active, 11 empty</text>

  <!-- t=2: Bleed -->
  <rect x="40" y="150" width="1020" height="40" rx="6" fill="#0a0025" stroke="#e879f9" stroke-width="1"/>
  <text x="55" y="170" fill="#f0abfc" font-size="8" font-weight="bold">2ms</text>
  <text x="120" y="170" fill="#d946ef" font-size="8">Bleed</text>
  <text x="250" y="170" fill="#d946ef" font-size="8">3 ghosts loaded: "Rc_is_not_Send" (3mo ago), "RefCell_runtime_borrow" (1mo), "Arc_vs_Rc" (2wk)</text>
  <text x="250" y="183" fill="#a855f7" font-size="7">Ghost gists now in Bleed Buffer. Energy landscape subtly warped toward these concepts.</text>

  <!-- t=3: RAR 1 -->
  <rect x="40" y="195" width="1020" height="70" rx="6" fill="#080020" stroke="#7c3aed" stroke-width="1.2"/>
  <text x="55" y="213" fill="#ddd6fe" font-size="8" font-weight="bold">3ms</text>
  <text x="120" y="213" fill="#c4b5fd" font-size="8">RAR 1</text>
  <text x="180" y="213" fill="#a78bfa" font-size="8" font-weight="bold">R:</text>
  <text x="200" y="213" fill="#a78bfa" font-size="7">S₀ drifts toward "smart_pointer" attractor. S₁ toward "thread_safety_query". S₂-S₄ foggy, high σ.</text>
  <text x="180" y="230" fill="#67e8f9" font-size="8" font-weight="bold">A:</text>
  <text x="200" y="230" fill="#67e8f9" font-size="7">S₂ attends to ghost "Rc_is_not_Send" → receives weak signal about !Send. S₃ attends to ghost "RefCell" → borrow checking.</text>
  <text x="180" y="247" fill="#fcd34d" font-size="8" font-weight="bold">Re:</text>
  <text x="200" y="247" fill="#fcd34d" font-size="7">S₀ ‖Δ‖=0.12 (converging). S₁ ‖Δ‖=0.15. S₂ ‖Δ‖=0.45 (still searching). S₃ ‖Δ‖=0.60 (very foggy). S₄ ‖Δ‖=0.70.</text>
  <text x="850" y="258" fill="#94a3b8" font-size="7">γ: [0.95, 0.90, 0.25, 0.15, 0.10]</text>

  <!-- t=8: RAR 2 -->
  <rect x="40" y="270" width="1020" height="60" rx="6" fill="#080020" stroke="#7c3aed" stroke-width="1"/>
  <text x="55" y="288" fill="#ddd6fe" font-size="8" font-weight="bold">8ms</text>
  <text x="120" y="288" fill="#c4b5fd" font-size="8">RAR 2</text>
  <text x="180" y="288" fill="#a78bfa" font-size="8" font-weight="bold">R:</text>
  <text x="200" y="288" fill="#a78bfa" font-size="7">S₀ ‖Δ‖=0.02 → CONVERGED ✓. S₂ VFN now strongly drifts toward "!Send_!Sync" attractor. Diffusion on S₃ explores.</text>
  <text x="180" y="305" fill="#67e8f9" font-size="8" font-weight="bold">A:</text>
  <text x="200" y="305" fill="#67e8f9" font-size="7">S₃ attends to frozen S₀ (Rc) + ghost "Arc_vs_Rc" → starts forming "Rc is single-threaded" concept.</text>
  <text x="180" y="322" fill="#fcd34d" font-size="8" font-weight="bold">Re:</text>
  <text x="200" y="322" fill="#fcd34d" font-size="7">S₁ ‖Δ‖=0.04 → almost converged. S₂ snapping toward !Send. S₃ forming reasoning chain. S₄ still vague.</text>
  <text x="850" y="322" fill="#94a3b8" font-size="7">γ: [0.98, 0.93, 0.55, 0.30, 0.12]</text>

  <!-- t=15: RAR 3 -->
  <rect x="40" y="335" width="1020" height="55" rx="6" fill="#080020" stroke="#7c3aed" stroke-width="1"/>
  <text x="55" y="353" fill="#ddd6fe" font-size="8" font-weight="bold">15ms</text>
  <text x="120" y="353" fill="#c4b5fd" font-size="8">RAR 3</text>
  <text x="200" y="353" fill="#a78bfa" font-size="7">S₁ CONVERGED ✓. S₂ locks onto "not_Send_not_Sync" — CONVERGED ✓. 3 active slots → 2 active (S₃, S₄).</text>
  <text x="200" y="370" fill="#67e8f9" font-size="7">S₃ now strongly attending to frozen S₂ (!Send) + frozen S₀ (Rc) → "Rc cannot cross thread boundary"</text>
  <text x="200" y="383" fill="#fcd34d" font-size="7">S₃ ‖Δ‖=0.20 (converging but not there yet). S₄ (conclusion) waiting for S₃ to settle.</text>
  <text x="850" y="383" fill="#94a3b8" font-size="7">γ: [0.98, 0.96, 0.88, 0.52, 0.15]</text>

  <!-- t=25: RAR 4 — page fault! -->
  <rect x="40" y="395" width="1020" height="65" rx="6" fill="#080020" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="55" y="413" fill="#a5f3fc" font-size="8" font-weight="bold">25ms</text>
  <text x="120" y="413" fill="#67e8f9" font-size="8">RAR 4</text>
  <text x="200" y="413" fill="#67e8f9" font-size="7">⚡ S₃ drifts near ghost "Arc_vs_Rc" (from 2 weeks ago) — PAGE FAULT! Full Frame loaded from T1.</text>
  <text x="200" y="430" fill="#22d3ee" font-size="7">Memory recall: "Arc&lt;Mutex&lt;T&gt;&gt; is the thread-safe alternative" — this enriches S₃'s reasoning context.</text>
  <text x="200" y="447" fill="#a78bfa" font-size="7">S₃ now forming: "Rc is !Send because no atomic refcount. Use Arc for threads." S₄ starts crystallizing.</text>
  <text x="200" y="460" fill="#d97706" font-size="7">Diffusion on S₄ dropping — it's starting to see a clear answer path.</text>
  <text x="850" y="447" fill="#94a3b8" font-size="7">γ: [—, —, —, 0.72, 0.35]</text>
  <text x="1000" y="447" fill="#22d3ee" font-size="7">↑ S₃ jump</text>

  <!-- t=35: RAR 5 -->
  <rect x="40" y="465" width="1020" height="50" rx="6" fill="#080020" stroke="#7c3aed" stroke-width="1"/>
  <text x="55" y="483" fill="#ddd6fe" font-size="8" font-weight="bold">35ms</text>
  <text x="120" y="483" fill="#c4b5fd" font-size="8">RAR 5</text>
  <text x="200" y="483" fill="#a78bfa" font-size="7">S₃ converging: "Rc is !Send, !Sync. No atomic ops. Single-thread only." ‖Δ‖=0.08.</text>
  <text x="200" y="500" fill="#67e8f9" font-size="7">S₄ attends to S₂(!Send) + S₃(reason) → forming conclusion: "No, Rc&lt;RefCell&lt;T&gt;&gt; is NOT thread-safe."</text>
  <text x="850" y="500" fill="#94a3b8" font-size="7">γ: [—, —, —, 0.85, 0.58]</text>

  <!-- t=45: RAR 6 -->
  <rect x="40" y="520" width="1020" height="45" rx="6" fill="#080020" stroke="#7c3aed" stroke-width="1"/>
  <text x="55" y="540" fill="#ddd6fe" font-size="8" font-weight="bold">45ms</text>
  <text x="120" y="540" fill="#c4b5fd" font-size="8">RAR 6</text>
  <text x="200" y="540" fill="#a78bfa" font-size="7">S₃ CONVERGED ✓ (‖Δ‖=0.009). Only S₄ still active. S₄ now has full context from all other frozen slots.</text>
  <text x="200" y="557" fill="#fcd34d" font-size="7">S₄ crystallizing: "No. Use Arc&lt;Mutex&lt;T&gt;&gt; or Arc&lt;RwLock&lt;T&gt;&gt; for thread-safe interior mutability."</text>
  <text x="850" y="557" fill="#94a3b8" font-size="7">γ: [—, —, —, 0.92, 0.75]</text>

  <!-- t=55: RAR 7 -->
  <rect x="40" y="570" width="1020" height="40" rx="6" fill="#080020" stroke="#7c3aed" stroke-width="1"/>
  <text x="55" y="590" fill="#ddd6fe" font-size="8" font-weight="bold">55ms</text>
  <text x="120" y="590" fill="#c4b5fd" font-size="8">RAR 7</text>
  <text x="200" y="590" fill="#a78bfa" font-size="7">S₄ refining wording of conclusion. Diffusion σ≈0 — locked in. ‖Δ‖=0.04.</text>
  <text x="200" y="603" fill="#67e8f9" font-size="7">S₄ checks against ghost: "Arc_vs_Rc" confirms distinction. Cross-attention score 0.91.</text>
  <text x="850" y="603" fill="#94a3b8" font-size="7">γ: [—, —, —, —, 0.88]</text>

  <!-- t=60: RAR 8 — converged -->
  <rect x="40" y="615" width="1020" height="40" rx="6" fill="#002a15" stroke="#22c55e" stroke-width="2"/>
  <text x="55" y="635" fill="#6ee7b7" font-size="8" font-weight="bold">60ms</text>
  <text x="120" y="635" fill="#22c55e" font-size="8" font-weight="bold">RAR 8</text>
  <text x="200" y="635" fill="#22c55e" font-size="7">S₄ CONVERGED ✓ (‖Δ‖=0.006). ALL SLOTS FROZEN. Frame emitted to CPU Hard Core.</text>
  <text x="850" y="635" fill="#22c55e" font-size="8" font-weight="bold">γ: [0.98, 0.96, 0.88, 0.92, 0.91]</text>

  <!-- CPU + Decode -->
  <rect x="40" y="662" width="500" height="45" rx="6" fill="#0a0800" stroke="#d97706" stroke-width="1"/>
  <text x="55" y="680" fill="#fde68a" font-size="8" font-weight="bold">65ms</text>
  <text x="120" y="680" fill="#fcd34d" font-size="8">CPU</text>
  <text x="200" y="680" fill="#fcd34d" font-size="7">HDC verify: all slots consistent. Safety: PASS. γ_min = 0.88. Proof: 4 steps.</text>
  <text x="200" y="697" fill="#d97706" font-size="7">Proof: [parse → ghost_recall(Rc_not_Send) → ghost_recall(Arc_vs_Rc) → reasoning_convergence]</text>

  <rect x="550" y="662" width="510" height="45" rx="6" fill="#0a2a15" stroke="#10b981" stroke-width="1"/>
  <text x="565" y="680" fill="#6ee7b7" font-size="8" font-weight="bold">75ms</text>
  <text x="630" y="680" fill="#34d399" font-size="8">Decode</text>
  <text x="710" y="680" fill="#34d399" font-size="7">All 5 slots decoded in parallel → assembled response.</text>
  <text x="710" y="697" fill="#059669" font-size="7">"No, Rc&lt;RefCell&lt;T&gt;&gt; is not thread-safe. Rc is !Send and !Sync.</text>

  <!-- VoltDB store -->
  <rect x="40" y="714" width="1020" height="35" rx="6" fill="#001520" stroke="#0891b2" stroke-width="1"/>
  <text x="55" y="734" fill="#a5f3fc" font-size="8" font-weight="bold">80ms</text>
  <text x="120" y="734" fill="#67e8f9" font-size="8">VoltDB</text>
  <text x="200" y="734" fill="#67e8f9" font-size="7">Frame stored. Learning event logged: "user asked about thread safety of Rc" → future sleep consolidation.</text>

  <!-- Final output -->
  <rect x="100" y="760" width="900" height="65" rx="10" fill="#0a2a15" stroke="#059669" stroke-width="2"/>
  <text x="550" y="782" text-anchor="middle" fill="#6ee7b7" font-size="11" font-weight="bold">Final Output (95ms total):</text>
  <text x="550" y="802" text-anchor="middle" fill="#34d399" font-size="10">"No, Rc&lt;RefCell&lt;T&gt;&gt; is not thread-safe. Rc is !Send and !Sync because it uses</text>
  <text x="550" y="818" text-anchor="middle" fill="#34d399" font-size="10">non-atomic reference counting. For thread-safe interior mutability, use Arc&lt;Mutex&lt;T&gt;&gt;."</text>

  <!-- Convergence chart -->
  <rect x="100" y="840" width="900" height="195" rx="10" fill="#0a0a15" stroke="#334155" stroke-width="1"/>
  <text x="550" y="862" text-anchor="middle" fill="#94a3b8" font-size="10" font-weight="bold">Slot Convergence Timeline</text>

  <!-- Mini chart: visual convergence bars per iteration -->
  <!-- Iteration labels -->
  <text x="190" y="885" text-anchor="middle" fill="#64748b" font-size="7">Iter 1</text>
  <text x="280" y="885" text-anchor="middle" fill="#64748b" font-size="7">2</text>
  <text x="370" y="885" text-anchor="middle" fill="#64748b" font-size="7">3</text>
  <text x="460" y="885" text-anchor="middle" fill="#64748b" font-size="7">4</text>
  <text x="550" y="885" text-anchor="middle" fill="#64748b" font-size="7">5</text>
  <text x="640" y="885" text-anchor="middle" fill="#64748b" font-size="7">6</text>
  <text x="730" y="885" text-anchor="middle" fill="#64748b" font-size="7">7</text>
  <text x="820" y="885" text-anchor="middle" fill="#64748b" font-size="7">8</text>

  <!-- S₀ row -->
  <text x="140" y="903" text-anchor="end" fill="#22c55e" font-size="7">S₀ AGENT</text>
  <rect x="160" y="893" width="60" height="12" rx="2" fill="#22c55e" opacity="0.5"/>
  <rect x="250" y="893" width="20" height="12" rx="2" fill="#22c55e"/>
  <text x="280" y="903" fill="#22c55e" font-size="7">✓</text>

  <!-- S₁ row -->
  <text x="140" y="918" text-anchor="end" fill="#8b5cf6" font-size="7">S₁ PRED</text>
  <rect x="160" y="908" width="65" height="12" rx="2" fill="#8b5cf6" opacity="0.5"/>
  <rect x="250" y="908" width="40" height="12" rx="2" fill="#8b5cf6" opacity="0.7"/>
  <rect x="340" y="908" width="15" height="12" rx="2" fill="#8b5cf6"/>
  <text x="370" y="918" fill="#8b5cf6" font-size="7">✓</text>

  <!-- S₂ row -->
  <text x="140" y="933" text-anchor="end" fill="#22d3ee" font-size="7">S₂ PROPERTY</text>
  <rect x="160" y="923" width="80" height="12" rx="2" fill="#22d3ee" opacity="0.3"/>
  <rect x="250" y="923" width="55" height="12" rx="2" fill="#22d3ee" opacity="0.5"/>
  <rect x="340" y="923" width="15" height="12" rx="2" fill="#22d3ee"/>
  <text x="370" y="933" fill="#22d3ee" font-size="7">✓</text>

  <!-- S₃ row -->
  <text x="140" y="948" text-anchor="end" fill="#f59e0b" font-size="7">S₃ REASON</text>
  <rect x="160" y="938" width="85" height="12" rx="2" fill="#f59e0b" opacity="0.2"/>
  <rect x="250" y="938" width="70" height="12" rx="2" fill="#f59e0b" opacity="0.3"/>
  <rect x="340" y="938" width="55" height="12" rx="2" fill="#f59e0b" opacity="0.4"/>
  <rect x="430" y="938" width="50" height="12" rx="2" fill="#f59e0b" opacity="0.6"/>
  <rect x="520" y="938" width="35" height="12" rx="2" fill="#f59e0b" opacity="0.8"/>
  <rect x="610" y="938" width="15" height="12" rx="2" fill="#f59e0b"/>
  <text x="640" y="948" fill="#f59e0b" font-size="7">✓</text>

  <!-- S₄ row -->
  <text x="140" y="963" text-anchor="end" fill="#ef4444" font-size="7">S₄ CONCLUSION</text>
  <rect x="160" y="953" width="88" height="12" rx="2" fill="#ef4444" opacity="0.15"/>
  <rect x="250" y="953" width="80" height="12" rx="2" fill="#ef4444" opacity="0.2"/>
  <rect x="340" y="953" width="75" height="12" rx="2" fill="#ef4444" opacity="0.25"/>
  <rect x="430" y="953" width="65" height="12" rx="2" fill="#ef4444" opacity="0.35"/>
  <rect x="520" y="953" width="50" height="12" rx="2" fill="#ef4444" opacity="0.5"/>
  <rect x="610" y="953" width="40" height="12" rx="2" fill="#ef4444" opacity="0.7"/>
  <rect x="730" y="953" width="25" height="12" rx="2" fill="#ef4444" opacity="0.85"/>
  <rect x="790" y="953" width="15" height="12" rx="2" fill="#ef4444"/>
  <text x="820" y="963" fill="#ef4444" font-size="7">✓</text>

  <!-- Legend -->
  <text x="120" y="990" fill="#475569" font-size="7">Bar length = ‖ΔS‖ (change magnitude). Opacity = uncertainty. Solid = converged. Empty slots not shown.</text>

  <!-- Active slots count -->
  <text x="120" y="1010" fill="#94a3b8" font-size="8" font-weight="bold">Active slots: 5 → 5 → 4 → 3 → 2 → 2 → 1 → 1 → 0 (done)</text>
  <text x="120" y="1028" fill="#64748b" font-size="7">GPU compute per iteration drops proportionally. Last 3 iterations process only S₄ — 80% less work than iteration 1.</text>
</svg>

<svg viewBox="0 0 1100 850" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="s3Bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f"/>
      <stop offset="100%" style="stop-color:#0f0a20"/>
    </linearGradient>
  </defs>
  <rect width="1100" height="850" fill="url(#s3Bg)" rx="12"/>
  <text x="550" y="30" text-anchor="middle" fill="#e2e8f0" font-size="16" font-weight="bold">Scenario 3: "Write a haiku about Rust's borrow checker" — Creative (~180ms)</text>
  <text x="550" y="50" text-anchor="middle" fill="#64748b" font-size="10">Active strand: Coding #01 (creative mode) | Difficulty: High (open-ended) | RAR iterations: 15 | σ: HIGH</text>

  <!-- This scenario is fundamentally different. Let me trace the key moments. -->

  <!-- Input -->
  <rect x="40" y="70" width="1020" height="45" rx="6" fill="#0a2a15" stroke="#10b981" stroke-width="1"/>
  <text x="55" y="88" fill="#6ee7b7" font-size="8" font-weight="bold">0ms</text>
  <text x="120" y="88" fill="#34d399" font-size="8">Translate</text>
  <text x="250" y="88" fill="#34d399" font-size="8">S₀[TASK: write_haiku] S₁[SUBJECT: borrow_checker] S₂[FORM: 5-7-5] S₃[LINE1: ∅] S₄[LINE2: ∅] S₅[LINE3: ∅]</text>
  <text x="250" y="105" fill="#059669" font-size="7">γ: [0.95, 0.90, 0.99, —, —, —] | Creative mode detected → σ_base = HIGH</text>

  <!-- Key moment: Diffusion exploration -->
  <rect x="40" y="122" width="1020" height="90" rx="6" fill="#080020" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="55" y="142" fill="#ddd6fe" font-size="8" font-weight="bold">5ms</text>
  <text x="120" y="142" fill="#c4b5fd" font-size="8">RAR 1-3</text>
  <text x="250" y="142" fill="#c4b5fd" font-size="8">S₀-S₂ converge quickly (task/subject/form are well-defined).</text>
  <text x="250" y="160" fill="#a78bfa" font-size="8">S₃-S₅ (the actual haiku lines) are WIDE OPEN. High diffusion σ.</text>
  <text x="250" y="178" fill="#a78bfa" font-size="8">VFN explores broadly: S₃ wanders between "ownership" "memory" "lifetime" "compile" regions.</text>
  <text x="250" y="196" fill="#8b5cf6" font-size="7">This is the creative wandering phase. The energy landscape has many shallow attractors — no single "right answer."</text>
  <text x="900" y="196" fill="#94a3b8" font-size="7">γ: [✓, ✓, ✓, 0.10, 0.08, 0.05]</text>

  <!-- Key moment: Ghost association -->
  <rect x="40" y="218" width="1020" height="75" rx="6" fill="#0a0025" stroke="#e879f9" stroke-width="1.5"/>
  <text x="55" y="238" fill="#f0abfc" font-size="8" font-weight="bold">20ms</text>
  <text x="120" y="238" fill="#d946ef" font-size="8">RAR 4-5</text>
  <text x="250" y="238" fill="#d946ef" font-size="8">⚡ S₃ diffusion noise kicks it near unexpected ghost: "cherry_blossom_metaphor" from a poetry strand!</text>
  <text x="250" y="256" fill="#e879f9" font-size="8">This is the EUREKA moment. The borrow checker ↔ nature metaphor connection forms.</text>
  <text x="250" y="274" fill="#d946ef" font-size="8">S₃ now attracted to intersection of "ownership" + "nature" + "5_syllables". Much richer than pure technical.</text>
  <text x="250" y="288" fill="#a855f7" font-size="7">Without diffusion, S₃ would have converged to a boring technical haiku. The noise created a creative leap.</text>
  <text x="900" y="274" fill="#f0abfc" font-size="7">γ: [✓, ✓, ✓, 0.25, 0.10, 0.08]</text>

  <!-- Key moment: S₃ crystallizes first line -->
  <rect x="40" y="300" width="1020" height="70" rx="6" fill="#080020" stroke="#7c3aed" stroke-width="1.2"/>
  <text x="55" y="320" fill="#ddd6fe" font-size="8" font-weight="bold">40ms</text>
  <text x="120" y="320" fill="#c4b5fd" font-size="8">RAR 6-8</text>
  <text x="250" y="320" fill="#c4b5fd" font-size="8">S₃ converging toward first line concept: "a value held tight" (nature of ownership)</text>
  <text x="250" y="338" fill="#a78bfa" font-size="8">Syllable constraint (from S₂ attending to S₃): "held tight" = 5 syllables? Let me check... "a val-ue held tight" = 5 ✓</text>
  <text x="250" y="356" fill="#a78bfa" font-size="8">S₄ now begins: attending to S₃ for thematic continuation. Drifts toward "borrowing" + "letting go".</text>
  <text x="250" y="366" fill="#8b5cf6" font-size="7">S₅ still high-σ — waiting for S₃ and S₄ to set the emotional arc before concluding.</text>
  <text x="900" y="356" fill="#94a3b8" font-size="7">γ: [✓, ✓, ✓, 0.65, 0.28, 0.12]</text>

  <!-- Key moment: S₃ converges, S₄ follows -->
  <rect x="40" y="377" width="1020" height="65" rx="6" fill="#080020" stroke="#7c3aed" stroke-width="1"/>
  <text x="55" y="397" fill="#ddd6fe" font-size="8" font-weight="bold">65ms</text>
  <text x="120" y="397" fill="#c4b5fd" font-size="8">RAR 9-10</text>
  <text x="250" y="397" fill="#22c55e" font-size="8">S₃ CONVERGED ✓ → "A value held tight" (5 syllables, nature+ownership metaphor)</text>
  <text x="250" y="415" fill="#a78bfa" font-size="8">S₄ now gets full context from frozen S₃. Explores: "released to another" / "the borrow must end" / "shared but never owned"</text>
  <text x="250" y="433" fill="#a78bfa" font-size="7">Diffusion tests "shared but never owned" — semantically stronger. Drifts there. 7 syllables needed. "shared, but ne-ver owned" = 5. Too short.</text>
  <text x="250" y="443" fill="#a78bfa" font-size="7">Tries: "the bor-row must re-turn" = 6. "it bor-rows then re-turns" = 6. "must re-turn what it bor-rows" = 7 ✓</text>
  <text x="900" y="433" fill="#94a3b8" font-size="7">γ: [✓, ✓, ✓, ✓, 0.55, 0.18]</text>

  <!-- Key moment: S₄ converges, S₅ final -->
  <rect x="40" y="448" width="1020" height="60" rx="6" fill="#080020" stroke="#7c3aed" stroke-width="1"/>
  <text x="55" y="468" fill="#ddd6fe" font-size="8" font-weight="bold">95ms</text>
  <text x="120" y="468" fill="#c4b5fd" font-size="8">RAR 11-12</text>
  <text x="250" y="468" fill="#22c55e" font-size="8">S₄ CONVERGED ✓ → "must return what it borrows" (7 syllables, thematic arc)</text>
  <text x="250" y="486" fill="#a78bfa" font-size="8">S₅ now the only active slot. Needs resolution. Attending to S₃ (held tight) + S₄ (returns) → needs closure (5 syl).</text>
  <text x="250" y="500" fill="#a78bfa" font-size="7">Options explored via diffusion: "com-pile suc-ceeds" = 4. "free-dom in con-straint" = 5 ✓ ← strong semantic closure!</text>
  <text x="900" y="500" fill="#94a3b8" font-size="7">γ: [✓, ✓, ✓, ✓, ✓, 0.60]</text>

  <!-- Final convergence -->
  <rect x="40" y="514" width="1020" height="50" rx="6" fill="#002a15" stroke="#22c55e" stroke-width="2"/>
  <text x="55" y="534" fill="#6ee7b7" font-size="8" font-weight="bold">130ms</text>
  <text x="120" y="534" fill="#22c55e" font-size="8" font-weight="bold">RAR 14-15</text>
  <text x="250" y="534" fill="#22c55e" font-size="8">S₅ CONVERGED ✓ → "freedom in constraint" (5 syllables, philosophical closure)</text>
  <text x="250" y="550" fill="#22c55e" font-size="7">ALL SLOTS FROZEN. Frame emitted. Final γ: [0.98, 0.96, 0.99, 0.88, 0.85, 0.82]</text>
  <text x="900" y="550" fill="#22c55e" font-size="8" font-weight="bold">γ_min = 0.82</text>

  <!-- CPU + Output -->
  <rect x="40" y="572" width="1020" height="40" rx="6" fill="#0a0800" stroke="#d97706" stroke-width="1"/>
  <text x="55" y="592" fill="#fde68a" font-size="8" font-weight="bold">140ms</text>
  <text x="120" y="592" fill="#fcd34d" font-size="8">CPU verify</text>
  <text x="250" y="592" fill="#fcd34d" font-size="7">Syllable count verified by MathEngine Hard Strand: 5-7-5 ✓. Safety: PASS. No tool execution needed.</text>
  <text x="250" y="606" fill="#d97706" font-size="7">Proof: [parse → ghost_association(cherry_blossom) → syllable_constraint → semantic_closure_verify]</text>

  <rect x="40" y="618" width="1020" height="40" rx="6" fill="#0a2a15" stroke="#10b981" stroke-width="1"/>
  <text x="55" y="638" fill="#6ee7b7" font-size="8" font-weight="bold">150ms</text>
  <text x="120" y="638" fill="#34d399" font-size="8">Decode</text>
  <text x="250" y="638" fill="#34d399" font-size="7">S₃→"A value held tight" | S₄→"must return what it borrows" | S₅→"freedom in constraint" → Assemble as haiku.</text>

  <!-- Final output -->
  <rect x="200" y="670" width="700" height="90" rx="12" fill="#0a2a15" stroke="#059669" stroke-width="2"/>
  <text x="550" y="695" text-anchor="middle" fill="#6ee7b7" font-size="12" font-weight="bold">Output (180ms total):</text>
  <text x="550" y="718" text-anchor="middle" fill="#e2e8f0" font-size="14" font-style="italic">A value held tight</text>
  <text x="550" y="738" text-anchor="middle" fill="#e2e8f0" font-size="14" font-style="italic">must return what it borrows —</text>
  <text x="550" y="758" text-anchor="middle" fill="#e2e8f0" font-size="14" font-style="italic">freedom in constraint.</text>

  <!-- What made this creative -->
  <rect x="100" y="780" width="900" height="55" rx="10" fill="#0a0a15" stroke="#e879f9" stroke-width="1.5"/>
  <text x="550" y="802" text-anchor="middle" fill="#f0abfc" font-size="10" font-weight="bold">What Made This Creative (Not Just Technically Correct):</text>
  <text x="550" y="822" text-anchor="middle" fill="#d946ef" font-size="9">The diffusion noise at RAR 4 pushed S₃ toward a ghost from a DIFFERENT strand (poetry, not coding).</text>
  <text x="550" y="835" text-anchor="middle" fill="#a855f7" font-size="8">Without diffusion: "Bor-row check-er runs / com-pile er-ror on line five / fix the life-time, please" — correct but boring.</text>
</svg>

<svg viewBox="0 0 1000 400" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="clBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f"/>
      <stop offset="100%" style="stop-color:#0a0a1a"/>
    </linearGradient>
  </defs>
  <rect width="1000" height="400" fill="url(#clBg)" rx="12"/>
  <text x="500" y="35" text-anchor="middle" fill="#e2e8f0" font-size="17" font-weight="bold">Why Continual Learning Is Not a Separate Thing</text>

  <!-- The point -->
  <rect x="50" y="60" width="900" height="80" rx="12" fill="#0a2a15" stroke="#10b981" stroke-width="2"/>
  <text x="500" y="88" text-anchor="middle" fill="#6ee7b7" font-size="13" font-weight="bold">In transformers, learning and inference are two completely different operations.</text>
  <text x="500" y="110" text-anchor="middle" fill="#34d399" font-size="11">In Volt, every inference IS a learning event. The Frame you generated IS the knowledge.</text>
  <text x="500" y="128" text-anchor="middle" fill="#059669" font-size="9">There is no "training data" vs "runtime" distinction. There is only Frame accumulation.</text>

  <!-- Three things that happen automatically -->
  <rect x="50" y="160" width="280" height="210" rx="12" fill="#001a20" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="190" y="185" text-anchor="middle" fill="#a5f3fc" font-size="12" font-weight="bold">Instant Learning</text>
  <text x="190" y="205" text-anchor="middle" fill="#67e8f9" font-size="9">(already happens)</text>
  <text x="70" y="230" fill="#67e8f9" font-size="9">You ask a question.</text>
  <text x="70" y="248" fill="#67e8f9" font-size="9">Volt generates a Frame.</text>
  <text x="70" y="266" fill="#67e8f9" font-size="9">That Frame goes into T0.</text>
  <text x="70" y="284" fill="#67e8f9" font-size="9">Then T1 (strand storage).</text>
  <text x="70" y="302" fill="#22d3ee" font-size="9" font-weight="bold">That IS the learning.</text>
  <text x="70" y="322" fill="#0891b2" font-size="8">Next time a similar query</text>
  <text x="70" y="338" fill="#0891b2" font-size="8">comes, this Frame is a ghost</text>
  <text x="70" y="354" fill="#0891b2" font-size="8">that warps the landscape.</text>

  <rect x="360" y="160" width="280" height="210" rx="12" fill="#1a1000" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="500" y="185" text-anchor="middle" fill="#fde68a" font-size="12" font-weight="bold">Sleep Consolidation</text>
  <text x="500" y="205" text-anchor="middle" fill="#fcd34d" font-size="9">(Frame distillation)</text>
  <text x="380" y="230" fill="#fcd34d" font-size="9">50 Frames about Rust →</text>
  <text x="380" y="248" fill="#fcd34d" font-size="9">3 distilled wisdom Frames.</text>
  <text x="380" y="270" fill="#fcd34d" font-size="9">This is compression, not</text>
  <text x="380" y="288" fill="#fcd34d" font-size="9">training. The information</text>
  <text x="380" y="306" fill="#fcd34d" font-size="9">was already there in the</text>
  <text x="380" y="324" fill="#fcd34d" font-size="9">raw Frames. Distillation</text>
  <text x="380" y="342" fill="#d97706" font-size="9" font-weight="bold">just makes it denser.</text>
  <text x="380" y="358" fill="#92400e" font-size="8">The VFN weight update is</text>
  <text x="380" y="370" fill="#92400e" font-size="8">optional optimization, not core.</text>

  <rect x="670" y="160" width="280" height="210" rx="12" fill="#100020" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="810" y="185" text-anchor="middle" fill="#ddd6fe" font-size="12" font-weight="bold">Developmental Growth</text>
  <text x="810" y="205" text-anchor="middle" fill="#c4b5fd" font-size="9">(strand graduation)</text>
  <text x="690" y="230" fill="#c4b5fd" font-size="9">You keep asking about</text>
  <text x="690" y="248" fill="#c4b5fd" font-size="9">cooking. Frames pile up</text>
  <text x="690" y="266" fill="#c4b5fd" font-size="9">under "misc" strand.</text>
  <text x="690" y="284" fill="#c4b5fd" font-size="9">VoltDB notices the cluster.</text>
  <text x="690" y="302" fill="#c4b5fd" font-size="9">Promotes to Cooking #05.</text>
  <text x="690" y="324" fill="#a78bfa" font-size="9" font-weight="bold">Emergence, not training.</text>
  <text x="690" y="344" fill="#8b5cf6" font-size="8">The strands self-organize</text>
  <text x="690" y="360" fill="#8b5cf6" font-size="8">from the structure of your</text>
  <text x="690" y="375" fill="#8b5cf6" font-size="8">accumulated Frames.</text>
</svg>

<svg viewBox="0 0 1100 600" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="tbBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#06060f"/>
      <stop offset="100%" style="stop-color:#0a0a1a"/>
    </linearGradient>
  </defs>
  <rect width="1100" height="600" fill="url(#tbBg)" rx="12"/>
  <text x="550" y="35" text-anchor="middle" fill="#e2e8f0" font-size="17" font-weight="bold">A Thousand Brains → Volt v3.0: The Mapping</text>
  <text x="550" y="55" text-anchor="middle" fill="#64748b" font-size="10">Hawkins arrived from neuroscience. You arrived from engineering. Same destination.</text>

  <!-- Headers -->
  <rect x="50" y="75" width="320" height="35" rx="8" fill="#1a0020" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="210" y="98" text-anchor="middle" fill="#ddd6fe" font-size="12" font-weight="bold">A Thousand Brains (Hawkins)</text>

  <rect x="400" y="75" width="70" height="35" rx="8" fill="#0a0a15" stroke="#334155" stroke-width="1"/>
  <text x="435" y="98" text-anchor="middle" fill="#94a3b8" font-size="12">=</text>

  <rect x="500" y="75" width="320" height="35" rx="8" fill="#001a10" stroke="#10b981" stroke-width="1.5"/>
  <text x="660" y="98" text-anchor="middle" fill="#6ee7b7" font-size="12" font-weight="bold">Volt v3.0</text>

  <rect x="850" y="75" width="210" height="35" rx="8" fill="#0a0a15" stroke="#475569" stroke-width="1"/>
  <text x="955" y="98" text-anchor="middle" fill="#94a3b8" font-size="11" font-weight="bold">Why it matches</text>

  <!-- Row 1: Cortical Columns = Strands -->
  <rect x="50" y="120" width="320" height="60" rx="8" fill="#100020" stroke="#7c3aed" stroke-width="1"/>
  <text x="210" y="143" text-anchor="middle" fill="#c4b5fd" font-size="10" font-weight="bold">Cortical Columns</text>
  <text x="210" y="160" text-anchor="middle" fill="#a78bfa" font-size="8">~150,000 identical processing units</text>
  <text x="210" y="173" text-anchor="middle" fill="#8b5cf6" font-size="8">Each learns a complete model of its input</text>

  <text x="435" y="155" text-anchor="middle" fill="#e879f9" font-size="16" font-weight="bold">↔</text>

  <rect x="500" y="120" width="320" height="60" rx="8" fill="#001a10" stroke="#10b981" stroke-width="1"/>
  <text x="660" y="143" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">Strands</text>
  <text x="660" y="160" text-anchor="middle" fill="#34d399" font-size="8">Unlimited parallel memory contexts</text>
  <text x="660" y="173" text-anchor="middle" fill="#059669" font-size="8">Each maintains a complete domain model</text>

  <rect x="850" y="120" width="210" height="60" rx="8" fill="#0a0a10" stroke="#334155" stroke-width="0.8"/>
  <text x="955" y="143" text-anchor="middle" fill="#94a3b8" font-size="8">Both: many independent</text>
  <text x="955" y="158" text-anchor="middle" fill="#94a3b8" font-size="8">units, each with a full</text>
  <text x="955" y="173" text-anchor="middle" fill="#64748b" font-size="8">model, sharing weights</text>

  <!-- Row 2: Reference Frames = Tensor Frames -->
  <rect x="50" y="190" width="320" height="60" rx="8" fill="#100020" stroke="#7c3aed" stroke-width="1"/>
  <text x="210" y="213" text-anchor="middle" fill="#c4b5fd" font-size="10" font-weight="bold">Reference Frames</text>
  <text x="210" y="230" text-anchor="middle" fill="#a78bfa" font-size="8">Every column models its object relative</text>
  <text x="210" y="243" text-anchor="middle" fill="#8b5cf6" font-size="8">to a spatial/conceptual frame of reference</text>

  <text x="435" y="225" text-anchor="middle" fill="#e879f9" font-size="16" font-weight="bold">↔</text>

  <rect x="500" y="190" width="320" height="60" rx="8" fill="#001a10" stroke="#10b981" stroke-width="1"/>
  <text x="660" y="213" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">Tensor Frames</text>
  <text x="660" y="230" text-anchor="middle" fill="#34d399" font-size="8">Structured [S×R×D] representation</text>
  <text x="660" y="243" text-anchor="middle" fill="#059669" font-size="8">Slots = positions in conceptual space</text>

  <rect x="850" y="190" width="210" height="60" rx="8" fill="#0a0a10" stroke="#334155" stroke-width="0.8"/>
  <text x="955" y="213" text-anchor="middle" fill="#94a3b8" font-size="8">Both: structured, spatial</text>
  <text x="955" y="228" text-anchor="middle" fill="#94a3b8" font-size="8">representations that you</text>
  <text x="955" y="243" text-anchor="middle" fill="#64748b" font-size="8">can navigate and index into</text>

  <!-- Row 3: Voting / Consensus = Cross-Slot Attention -->
  <rect x="50" y="260" width="320" height="60" rx="8" fill="#100020" stroke="#7c3aed" stroke-width="1"/>
  <text x="210" y="283" text-anchor="middle" fill="#c4b5fd" font-size="10" font-weight="bold">Voting / Consensus</text>
  <text x="210" y="300" text-anchor="middle" fill="#a78bfa" font-size="8">Columns vote on what they perceive</text>
  <text x="210" y="313" text-anchor="middle" fill="#8b5cf6" font-size="8">Consensus = unified perception</text>

  <text x="435" y="295" text-anchor="middle" fill="#e879f9" font-size="16" font-weight="bold">↔</text>

  <rect x="500" y="260" width="320" height="60" rx="8" fill="#001a10" stroke="#10b981" stroke-width="1"/>
  <text x="660" y="283" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">Attend Phase (RAR)</text>
  <text x="660" y="300" text-anchor="middle" fill="#34d399" font-size="8">Slots attend to each other's roots</text>
  <text x="660" y="313" text-anchor="middle" fill="#059669" font-size="8">Attention weights = voting strength</text>

  <rect x="850" y="260" width="210" height="60" rx="8" fill="#0a0a10" stroke="#334155" stroke-width="0.8"/>
  <text x="955" y="283" text-anchor="middle" fill="#94a3b8" font-size="8">Both: independent units</text>
  <text x="955" y="298" text-anchor="middle" fill="#94a3b8" font-size="8">form consensus via</text>
  <text x="955" y="313" text-anchor="middle" fill="#64748b" font-size="8">inter-unit communication</text>

  <!-- Row 4: Location Signals = Ghost Bleed -->
  <rect x="50" y="330" width="320" height="60" rx="8" fill="#100020" stroke="#7c3aed" stroke-width="1"/>
  <text x="210" y="353" text-anchor="middle" fill="#c4b5fd" font-size="10" font-weight="bold">Location Signals</text>
  <text x="210" y="370" text-anchor="middle" fill="#a78bfa" font-size="8">Columns receive "where am I looking"</text>
  <text x="210" y="383" text-anchor="middle" fill="#8b5cf6" font-size="8">signals from motor/sensory system</text>

  <text x="435" y="365" text-anchor="middle" fill="#e879f9" font-size="16" font-weight="bold">↔</text>

  <rect x="500" y="330" width="320" height="60" rx="8" fill="#001a10" stroke="#10b981" stroke-width="1"/>
  <text x="660" y="353" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">Ghost Frames (Bleed)</text>
  <text x="660" y="370" text-anchor="middle" fill="#34d399" font-size="8">R₀ gists from T1/T2 warp landscape</text>
  <text x="660" y="383" text-anchor="middle" fill="#059669" font-size="8">"Context signal" from long-term memory</text>

  <rect x="850" y="330" width="210" height="60" rx="8" fill="#0a0a10" stroke="#334155" stroke-width="0.8"/>
  <text x="955" y="353" text-anchor="middle" fill="#94a3b8" font-size="8">Both: external context</text>
  <text x="955" y="368" text-anchor="middle" fill="#94a3b8" font-size="8">that biases processing</text>
  <text x="955" y="383" text-anchor="middle" fill="#64748b" font-size="8">without replacing content</text>

  <!-- Row 5: Learning by Observation = Frame Accumulation -->
  <rect x="50" y="400" width="320" height="60" rx="8" fill="#100020" stroke="#7c3aed" stroke-width="1"/>
  <text x="210" y="423" text-anchor="middle" fill="#c4b5fd" font-size="10" font-weight="bold">Learning by Observation</text>
  <text x="210" y="440" text-anchor="middle" fill="#a78bfa" font-size="8">Columns learn by experiencing sequences</text>
  <text x="210" y="453" text-anchor="middle" fill="#8b5cf6" font-size="8">of sensory-motor patterns over time</text>

  <text x="435" y="435" text-anchor="middle" fill="#e879f9" font-size="16" font-weight="bold">↔</text>

  <rect x="500" y="400" width="320" height="60" rx="8" fill="#001a10" stroke="#10b981" stroke-width="1"/>
  <text x="660" y="423" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">Frame Accumulation</text>
  <text x="660" y="440" text-anchor="middle" fill="#34d399" font-size="8">Inference = learning. Each new Frame</text>
  <text x="660" y="453" text-anchor="middle" fill="#059669" font-size="8">IS the learned knowledge. No separate phase.</text>

  <rect x="850" y="400" width="210" height="60" rx="8" fill="#0a0a10" stroke="#334155" stroke-width="0.8"/>
  <text x="955" y="423" text-anchor="middle" fill="#94a3b8" font-size="8">Both: learning is not</text>
  <text x="955" y="438" text-anchor="middle" fill="#94a3b8" font-size="8">a separate phase. It's</text>
  <text x="955" y="453" text-anchor="middle" fill="#64748b" font-size="8">what happens when you think.</text>

  <!-- Row 6: Many Models of One Object = Per-Slot γ -->
  <rect x="50" y="470" width="320" height="60" rx="8" fill="#100020" stroke="#7c3aed" stroke-width="1"/>
  <text x="210" y="493" text-anchor="middle" fill="#c4b5fd" font-size="10" font-weight="bold">Many Models, One Object</text>
  <text x="210" y="510" text-anchor="middle" fill="#a78bfa" font-size="8">Each column has its own model, may disagree</text>
  <text x="210" y="523" text-anchor="middle" fill="#8b5cf6" font-size="8">Consensus resolves disagreement</text>

  <text x="435" y="505" text-anchor="middle" fill="#e879f9" font-size="16" font-weight="bold">↔</text>

  <rect x="500" y="470" width="320" height="60" rx="8" fill="#001a10" stroke="#10b981" stroke-width="1"/>
  <text x="660" y="493" text-anchor="middle" fill="#6ee7b7" font-size="10" font-weight="bold">Per-Slot γ Certainty</text>
  <text x="660" y="510" text-anchor="middle" fill="#34d399" font-size="8">Each slot has independent confidence</text>
  <text x="660" y="523" text-anchor="middle" fill="#059669" font-size="8">γ_final = min(all slot γ). Honest uncertainty.</text>

  <rect x="850" y="470" width="210" height="60" rx="8" fill="#0a0a10" stroke="#334155" stroke-width="0.8"/>
  <text x="955" y="493" text-anchor="middle" fill="#94a3b8" font-size="8">Both: distributed belief</text>
  <text x="955" y="508" text-anchor="middle" fill="#94a3b8" font-size="8">with explicit uncertainty</text>
  <text x="955" y="523" text-anchor="middle" fill="#64748b" font-size="8">per component</text>

  <!-- Bottom -->
  <rect x="150" y="548" width="800" height="35" rx="8" fill="#0a0a15" stroke="#e879f9" stroke-width="1.5"/>
  <text x="550" y="571" text-anchor="middle" fill="#f0abfc" font-size="11" font-weight="bold">Hawkins predicted the architecture. You built it. The neocortex was the blueprint all along.</text>
</svg>

<svg viewBox="0 0 1600 3800" xmlns="http://www.w3.org/2000/svg" font-family="'Segoe UI', Arial, sans-serif">
  <defs>
    <linearGradient id="mainBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#050510"/>
      <stop offset="50%" style="stop-color:#0a0a1a"/>
      <stop offset="100%" style="stop-color:#080818"/>
    </linearGradient>
    <linearGradient id="busGr" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#3b0764"/>
      <stop offset="50%" style="stop-color:#6d28d9"/>
      <stop offset="100%" style="stop-color:#3b0764"/>
    </linearGradient>
    <filter id="gl"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <filter id="sg"><feGaussianBlur stdDeviation="1.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <marker id="aW" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#64748b"/></marker>
    <marker id="aP" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#8b5cf6"/></marker>
    <marker id="aC" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#22d3ee"/></marker>
    <marker id="aA" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#f59e0b"/></marker>
    <marker id="aG" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#10b981"/></marker>
    <marker id="aPi" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#ec4899"/></marker>
    <marker id="aR" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#ef4444"/></marker>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" stroke-width="0.3" opacity="0.3"/>
    </pattern>
  </defs>

  <rect width="1600" height="3800" fill="url(#mainBg)"/>
  <rect width="1600" height="3800" fill="url(#grid)"/>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- TITLE -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <text x="800" y="42" text-anchor="middle" fill="#7c3aed" font-size="12" letter-spacing="6">COMPLETE ARCHITECTURE SPECIFICATION</text>
  <text x="800" y="80" text-anchor="middle" fill="#e2e8f0" font-size="36" font-weight="bold" filter="url(#gl)">⚡ VOLT v3.0 — Master Architecture</text>
  <text x="800" y="110" text-anchor="middle" fill="#94a3b8" font-size="14">Stateful OS for Sovereign Intelligence | Split-Brain · Tensor Frames · Continual Learning · VoltDB · Intelligence Commons</text>
  <line x1="150" y1="130" x2="1450" y2="130" stroke="#1e293b" stroke-width="1"/>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- LAYER 0: EXTERNAL WORLD -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <text x="800" y="160" text-anchor="middle" fill="#475569" font-size="10" letter-spacing="4">LAYER 0 — EXTERNAL WORLD</text>

  <rect x="140" y="175" width="150" height="55" rx="10" fill="#1e293b" stroke="#94a3b8" stroke-width="1.5"/>
  <text x="215" y="200" text-anchor="middle" fill="#e2e8f0" font-size="11" font-weight="bold">👤 User</text>
  <text x="215" y="218" text-anchor="middle" fill="#94a3b8" font-size="8">Text, Voice, Files</text>

  <rect x="330" y="175" width="150" height="55" rx="10" fill="#1e293b" stroke="#94a3b8" stroke-width="1.5"/>
  <text x="405" y="200" text-anchor="middle" fill="#e2e8f0" font-size="11" font-weight="bold">🌐 APIs</text>
  <text x="405" y="218" text-anchor="middle" fill="#94a3b8" font-size="8">REST, gRPC, WS</text>

  <rect x="520" y="175" width="150" height="55" rx="10" fill="#1e293b" stroke="#94a3b8" stroke-width="1.5"/>
  <text x="595" y="200" text-anchor="middle" fill="#e2e8f0" font-size="11" font-weight="bold">📡 Sensors</text>
  <text x="595" y="218" text-anchor="middle" fill="#94a3b8" font-size="8">Camera, Audio, IoT</text>

  <rect x="710" y="175" width="150" height="55" rx="10" fill="#1e293b" stroke="#94a3b8" stroke-width="1.5"/>
  <text x="785" y="200" text-anchor="middle" fill="#e2e8f0" font-size="11" font-weight="bold">💻 OS</text>
  <text x="785" y="218" text-anchor="middle" fill="#94a3b8" font-size="8">Filesystem, Processes</text>

  <rect x="900" y="175" width="170" height="55" rx="10" fill="#1e293b" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="985" y="200" text-anchor="middle" fill="#ddd6fe" font-size="11" font-weight="bold">🔗 P2P Mesh</text>
  <text x="985" y="218" text-anchor="middle" fill="#a78bfa" font-size="8">Intelligence Commons</text>

  <rect x="1110" y="175" width="170" height="55" rx="10" fill="#1e293b" stroke="#ec4899" stroke-width="1.5"/>
  <text x="1195" y="200" text-anchor="middle" fill="#fbcfe8" font-size="11" font-weight="bold">🔌 Module Repo</text>
  <text x="1195" y="218" text-anchor="middle" fill="#f9a8d4" font-size="8">Community Ecosystem</text>

  <!-- Arrows down -->
  <line x1="215" y1="230" x2="215" y2="270" stroke="#64748b" stroke-width="1.5" marker-end="url(#aW)"/>
  <line x1="405" y1="230" x2="405" y2="270" stroke="#64748b" stroke-width="1.5" marker-end="url(#aW)"/>
  <line x1="595" y1="230" x2="595" y2="270" stroke="#64748b" stroke-width="1.5" marker-end="url(#aW)"/>
  <line x1="785" y1="230" x2="785" y2="270" stroke="#64748b" stroke-width="1.5" marker-end="url(#aW)"/>
  <line x1="985" y1="230" x2="985" y2="270" stroke="#64748b" stroke-width="1.5" marker-end="url(#aW)"/>
  <line x1="1195" y1="230" x2="1195" y2="270" stroke="#64748b" stroke-width="1.5" marker-end="url(#aW)"/>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- LAYER 1: INPUT TRANSLATORS -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <rect x="100" y="275" width="1400" height="105" rx="14" fill="#060f15" stroke="#10b981" stroke-width="2"/>
  <text x="130" y="296" fill="#6ee7b7" font-size="10" letter-spacing="3">LAYER 1 — INPUT TRANSLATORS</text>
  <text x="800" y="312" text-anchor="middle" fill="#e2e8f0" font-size="12" font-weight="bold">Community Modules: impl Translator { fn encode(&amp;self, raw: &amp;[u8]) → TensorFrame }</text>

  <rect x="125" y="325" width="135" height="40" rx="7" fill="#0a2a15" stroke="#22c55e" stroke-width="1.2"/>
  <text x="193" y="342" text-anchor="middle" fill="#86efac" font-size="9" font-weight="bold">📝 Text</text>
  <text x="193" y="357" text-anchor="middle" fill="#4ade80" font-size="7">BPE → Frame slots</text>

  <rect x="275" y="325" width="135" height="40" rx="7" fill="#0a2a15" stroke="#22c55e" stroke-width="1.2"/>
  <text x="343" y="342" text-anchor="middle" fill="#86efac" font-size="9" font-weight="bold">🖼️ Vision</text>
  <text x="343" y="357" text-anchor="middle" fill="#4ade80" font-size="7">ViT → Frame slots</text>

  <rect x="425" y="325" width="135" height="40" rx="7" fill="#0a2a15" stroke="#22c55e" stroke-width="1.2"/>
  <text x="493" y="342" text-anchor="middle" fill="#86efac" font-size="9" font-weight="bold">🎤 Audio</text>
  <text x="493" y="357" text-anchor="middle" fill="#4ade80" font-size="7">Whisper → Frame</text>

  <rect x="575" y="325" width="135" height="40" rx="7" fill="#0a2a15" stroke="#22c55e" stroke-width="1.2"/>
  <text x="643" y="342" text-anchor="middle" fill="#86efac" font-size="9" font-weight="bold">📊 Data</text>
  <text x="643" y="357" text-anchor="middle" fill="#4ade80" font-size="7">JSON/CSV → Frame</text>

  <rect x="725" y="325" width="135" height="40" rx="7" fill="#0a2a15" stroke="#22c55e" stroke-width="1.2"/>
  <text x="793" y="342" text-anchor="middle" fill="#86efac" font-size="9" font-weight="bold">📡 Sensor</text>
  <text x="793" y="357" text-anchor="middle" fill="#4ade80" font-size="7">IoT → Frame</text>

  <rect x="875" y="325" width="135" height="40" rx="7" fill="#0a2a15" stroke="#22c55e" stroke-width="1.2"/>
  <text x="943" y="342" text-anchor="middle" fill="#86efac" font-size="9" font-weight="bold">💻 OS Events</text>
  <text x="943" y="357" text-anchor="middle" fill="#4ade80" font-size="7">SysCall → Frame</text>

  <rect x="1025" y="325" width="105" height="40" rx="7" fill="#051a10" stroke="#475569" stroke-width="1" stroke-dasharray="4"/>
  <text x="1078" y="350" text-anchor="middle" fill="#64748b" font-size="9">➕ Custom</text>

  <rect x="1145" y="325" width="105" height="40" rx="7" fill="#051a10" stroke="#475569" stroke-width="1" stroke-dasharray="4"/>
  <text x="1198" y="350" text-anchor="middle" fill="#64748b" font-size="9">➕ Custom</text>

  <!-- Audit log (simplified) -->
  <rect x="1280" y="325" width="190" height="40" rx="7" fill="#0a0a15" stroke="#475569" stroke-width="1"/>
  <text x="1375" y="342" text-anchor="middle" fill="#94a3b8" font-size="9" font-weight="bold">📋 Audit Log</text>
  <text x="1375" y="357" text-anchor="middle" fill="#64748b" font-size="7">Append-only, local, signed</text>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- LAYER 2: LLL TENSOR FRAME BUS -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <line x1="800" y1="380" x2="800" y2="410" stroke="#8b5cf6" stroke-width="2" marker-end="url(#aP)"/>

  <rect x="200" y="415" width="1200" height="48" rx="24" fill="url(#busGr)" stroke="#8b5cf6" stroke-width="2.5" filter="url(#gl)"/>
  <text x="800" y="438" text-anchor="middle" fill="#ede9fe" font-size="14" font-weight="bold">⚡ LLL TENSOR FRAME BUS — F ∈ ℝ^[16 slots × 4 res × 256 dims]</text>
  <text x="800" y="455" text-anchor="middle" fill="#c4b5fd" font-size="9">HDC Algebra: ⊗ Bind (FFT) | + Superpose | ρ Permute | ⊗⁻¹ Unbind | Per-slot γ | Codebook u16 | Sparse structure</text>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- LAYER 3-4-5: THE THREE PILLARS -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <text x="800" y="490" text-anchor="middle" fill="#475569" font-size="10" letter-spacing="4">LAYERS 3-4-5 — SPLIT-BRAIN COGNITIVE ENGINE</text>

  <!-- Arrows from bus to pillars -->
  <line x1="320" y1="463" x2="320" y2="510" stroke="#8b5cf6" stroke-width="1.5" marker-end="url(#aP)"/>
  <line x1="800" y1="463" x2="800" y2="510" stroke="#8b5cf6" stroke-width="1.5" marker-end="url(#aP)"/>
  <line x1="1250" y1="463" x2="1250" y2="510" stroke="#8b5cf6" stroke-width="1.5" marker-end="url(#aP)"/>

  <!-- ═══════════════════════════ GPU SOFT CORE ═══════════════════════════ -->
  <rect x="100" y="510" width="430" height="750" rx="16" fill="#080020" stroke="#7c3aed" stroke-width="2.5" filter="url(#sg)"/>
  <rect x="100" y="510" width="430" height="45" rx="16" fill="#7c3aed" opacity="0.35"/>
  <text x="315" y="536" text-anchor="middle" fill="#ede9fe" font-size="15" font-weight="bold">🧠 LAYER 3: GPU — SOFT CORE</text>
  <text x="315" y="550" text-anchor="middle" fill="#c4b5fd" font-size="8">"Right Brain" | SIMD Parallel | Neural Intuition | CUDA</text>

  <!-- State Buffer -->
  <rect x="125" y="570" width="380" height="55" rx="8" fill="#12004a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="315" y="593" text-anchor="middle" fill="#ddd6fe" font-size="11" font-weight="bold">State Buffer — TensorFrame[active]</text>
  <text x="315" y="612" text-anchor="middle" fill="#a78bfa" font-size="8">Current thought frame in VRAM | R₀-R₁ slots active | Recurrent feedback</text>

  <!-- Recurrence arrow -->
  <path d="M 500 598 Q 520 598 520 640 Q 520 680 500 680" stroke="#e879f9" stroke-width="1.5" fill="none" stroke-dasharray="3"/>
  <text x="530" y="643" fill="#e879f9" font-size="7" transform="rotate(90,530,643)">recur</text>

  <!-- VFN -->
  <rect x="125" y="640" width="380" height="85" rx="8" fill="#12004a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="315" y="660" text-anchor="middle" fill="#ddd6fe" font-size="11" font-weight="bold">Vector Field Network f_θ(F, t)</text>
  <text x="315" y="678" text-anchor="middle" fill="#a78bfa" font-size="8">500M – 2B params | Fourier Neural Operator | O(S×D log D)</text>
  <text x="315" y="694" text-anchor="middle" fill="#c4b5fd" font-size="8">Operates on Frame R₀ slots → drift toward coherent attractors</text>
  <text x="315" y="712" text-anchor="middle" fill="#8b5cf6" font-size="7">dF = −∇E(F,t)dt + σ_φ(F,t)dW_t</text>

  <!-- Diffusion + Solver -->
  <rect x="125" y="740" width="180" height="55" rx="8" fill="#12004a" stroke="#a78bfa" stroke-width="1"/>
  <text x="215" y="760" text-anchor="middle" fill="#ddd6fe" font-size="10" font-weight="bold">Diffusion σ_φ</text>
  <text x="215" y="778" text-anchor="middle" fill="#a78bfa" font-size="7">Orthogonal noise | Creativity control</text>

  <rect x="325" y="740" width="180" height="55" rx="8" fill="#12004a" stroke="#a78bfa" stroke-width="1"/>
  <text x="415" y="760" text-anchor="middle" fill="#ddd6fe" font-size="10" font-weight="bold">ODE Solver</text>
  <text x="415" y="778" text-anchor="middle" fill="#a78bfa" font-size="7">RK4/DOPRI5 | Adaptive step</text>

  <!-- Manifold Projector -->
  <rect x="125" y="810" width="380" height="40" rx="8" fill="#12004a" stroke="#a78bfa" stroke-width="1"/>
  <text x="315" y="835" text-anchor="middle" fill="#ddd6fe" font-size="10" font-weight="bold">Manifold Projector → Stay on semantic manifold M</text>

  <!-- Convergence -->
  <rect x="125" y="865" width="380" height="40" rx="8" fill="#0a0030" stroke="#6d28d9" stroke-width="1" stroke-dasharray="3"/>
  <text x="315" y="882" text-anchor="middle" fill="#c4b5fd" font-size="9">Per-slot convergence: ‖F[s](t+Δ) − F[s](t)‖ &lt; ε → emit slot</text>
  <text x="315" y="898" text-anchor="middle" fill="#8b5cf6" font-size="7">Adaptive computation: simple=5 steps, complex=500 steps</text>

  <!-- Warm Start Cache -->
  <rect x="125" y="920" width="380" height="40" rx="8" fill="#0a0030" stroke="#6d28d9" stroke-width="1"/>
  <text x="315" y="940" text-anchor="middle" fill="#c4b5fd" font-size="10" font-weight="bold">Retention Cache (Warm Start)</text>
  <text x="315" y="955" text-anchor="middle" fill="#8b5cf6" font-size="7">LRU + TTL | ~40% hit rate | Skip-to-attractor</text>

  <!-- Bleed Buffer -->
  <rect x="125" y="975" width="380" height="50" rx="8" fill="#0a0025" stroke="#e879f9" stroke-width="1.5"/>
  <text x="315" y="998" text-anchor="middle" fill="#f0abfc" font-size="10" font-weight="bold">👻 Ghost Frame Bleed Buffer</text>
  <text x="315" y="1015" text-anchor="middle" fill="#d946ef" font-size="8">~1000 R₀ gists from T1/T2 | Warp energy landscape | Trigger page faults</text>

  <!-- VRAM note -->
  <rect x="125" y="1040" width="380" height="50" rx="8" fill="#050015" stroke="#4c1d95" stroke-width="1"/>
  <text x="315" y="1060" text-anchor="middle" fill="#8b5cf6" font-size="9" font-weight="bold">VRAM Budget: 4–8 GB (consumer GPU)</text>
  <text x="315" y="1078" text-anchor="middle" fill="#6d28d9" font-size="8">64 active Frames + bleed buffer + model weights + cache</text>

  <!-- T0 Label -->
  <rect x="125" y="1100" width="380" height="35" rx="6" fill="#12004a" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="315" y="1123" text-anchor="middle" fill="#c4b5fd" font-size="10" font-weight="bold">T0: Working Memory — 64 Frames, instant access</text>

  <!-- Forward-Forward note -->
  <rect x="125" y="1150" width="380" height="70" rx="8" fill="#050015" stroke="#4c1d95" stroke-width="1"/>
  <text x="315" y="1172" text-anchor="middle" fill="#a78bfa" font-size="10" font-weight="bold">🔬 Training: Forward-Forward</text>
  <text x="315" y="1192" text-anchor="middle" fill="#8b5cf6" font-size="8">Layer-local weight updates during sleep consolidation</text>
  <text x="315" y="1207" text-anchor="middle" fill="#6d28d9" font-size="8">Train VRAM ≈ Inference VRAM | Consumer RTX 4060+ sufficient</text>

  <!-- ═══════════════════════════ CPU HARD CORE ═══════════════════════════ -->
  <rect x="565" y="510" width="470" height="750" rx="16" fill="#0a0800" stroke="#d97706" stroke-width="2.5" filter="url(#sg)"/>
  <rect x="565" y="510" width="470" height="45" rx="16" fill="#d97706" opacity="0.3"/>
  <text x="800" y="536" text-anchor="middle" fill="#fefce8" font-size="15" font-weight="bold">⚙️ LAYER 4: CPU — HARD CORE</text>
  <text x="800" y="550" text-anchor="middle" fill="#fde68a" font-size="8">"Left Brain" | Branching Logic | Deterministic | Rust + Tokio + Rayon</text>

  <!-- Intent Router -->
  <rect x="590" y="570" width="420" height="55" rx="8" fill="#1a0c00" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="800" y="593" text-anchor="middle" fill="#fef3c7" font-size="11" font-weight="bold">Intent Router</text>
  <text x="800" y="612" text-anchor="middle" fill="#fcd34d" font-size="8">Frame R₀ → cosine similarity → route to best Hard Strand | No JSON parsing</text>

  <!-- Hard Strand Grid -->
  <text x="595" y="645" fill="#fde68a" font-size="9" font-weight="bold">HARD STRANDS (impl HardStrand) — Hot-pluggable Rust modules</text>

  <rect x="590" y="658" width="135" height="50" rx="7" fill="#1a0c00" stroke="#fbbf24" stroke-width="1"/>
  <text x="658" y="678" text-anchor="middle" fill="#fef3c7" font-size="9" font-weight="bold">🔢 MathEngine</text>
  <text x="658" y="698" text-anchor="middle" fill="#d97706" font-size="7">Exact | Zero hallucinate</text>

  <rect x="740" y="658" width="135" height="50" rx="7" fill="#1a0c00" stroke="#fbbf24" stroke-width="1"/>
  <text x="808" y="678" text-anchor="middle" fill="#fef3c7" font-size="9" font-weight="bold">💻 CodeRunner</text>
  <text x="808" y="698" text-anchor="middle" fill="#d97706" font-size="7">Sandboxed | WASM</text>

  <rect x="890" y="658" width="135" height="50" rx="7" fill="#1a0c00" stroke="#fbbf24" stroke-width="1"/>
  <text x="958" y="678" text-anchor="middle" fill="#fef3c7" font-size="9" font-weight="bold">🌐 APIDispatch</text>
  <text x="958" y="698" text-anchor="middle" fill="#d97706" font-size="7">50+ parallel | Tokio</text>

  <!-- HDC + Certainty -->
  <rect x="590" y="720" width="205" height="50" rx="7" fill="#1a0c00" stroke="#fbbf24" stroke-width="1"/>
  <text x="693" y="740" text-anchor="middle" fill="#fef3c7" font-size="9" font-weight="bold">🔗 HDC Algebra</text>
  <text x="693" y="758" text-anchor="middle" fill="#d97706" font-size="7">FFT Bind/Unbind | O(D log D)</text>

  <rect x="810" y="720" width="215" height="50" rx="7" fill="#1a0c00" stroke="#fbbf24" stroke-width="1"/>
  <text x="918" y="740" text-anchor="middle" fill="#fef3c7" font-size="9" font-weight="bold">γ Certainty Engine</text>
  <text x="918" y="758" text-anchor="middle" fill="#d97706" font-size="7">Min-rule propagation | Per-slot γ</text>

  <!-- Safety Module -->
  <rect x="590" y="785" width="420" height="90" rx="10" fill="#1a0508" stroke="#ef4444" stroke-width="2"/>
  <text x="800" y="808" text-anchor="middle" fill="#fca5a5" font-size="11" font-weight="bold">🛡️ Safety Layer (Deterministic — CPU Enforced)</text>

  <rect x="610" y="820" width="125" height="40" rx="6" fill="#200508" stroke="#991b1b" stroke-width="1"/>
  <text x="673" y="837" text-anchor="middle" fill="#fca5a5" font-size="8" font-weight="bold">Axiomatic Guard</text>
  <text x="673" y="852" text-anchor="middle" fill="#f87171" font-size="6">Immutable K axioms</text>

  <rect x="750" y="820" width="115" height="40" rx="6" fill="#200508" stroke="#991b1b" stroke-width="1"/>
  <text x="808" y="837" text-anchor="middle" fill="#fca5a5" font-size="8" font-weight="bold">Transition Mon</text>
  <text x="808" y="852" text-anchor="middle" fill="#f87171" font-size="6">All F(t)→F(t+1)</text>

  <rect x="880" y="820" width="115" height="40" rx="6" fill="#ef4444" stroke="#7f1d1d" stroke-width="2"/>
  <text x="938" y="837" text-anchor="middle" fill="#fff" font-size="8" font-weight="bold">OMEGA VETO</text>
  <text x="938" y="852" text-anchor="middle" fill="#fecaca" font-size="6">HW Interrupt | HALT</text>

  <!-- Proof Constructor -->
  <rect x="590" y="890" width="420" height="40" rx="7" fill="#1a0c00" stroke="#fbbf24" stroke-width="1"/>
  <text x="800" y="910" text-anchor="middle" fill="#fef3c7" font-size="10" font-weight="bold">📜 Proof Chain Constructor</text>
  <text x="800" y="925" text-anchor="middle" fill="#d97706" font-size="7">Every conclusion traceable: premises → logic → result + γ</text>

  <!-- Causal Simulation -->
  <rect x="590" y="945" width="205" height="45" rx="7" fill="#1a0c00" stroke="#fbbf24" stroke-width="1"/>
  <text x="693" y="965" text-anchor="middle" fill="#fef3c7" font-size="9" font-weight="bold">🔮 Causal Sim</text>
  <text x="693" y="982" text-anchor="middle" fill="#d97706" font-size="7">do(X=x) | Counterfactual</text>

  <!-- Mirror Module -->
  <rect x="810" y="945" width="215" height="45" rx="7" fill="#001a1a" stroke="#06b6d4" stroke-width="1"/>
  <text x="918" y="965" text-anchor="middle" fill="#67e8f9" font-size="9" font-weight="bold">🪞 Mirror / Monitor</text>
  <text x="918" y="982" text-anchor="middle" fill="#0891b2" font-size="7">Loop detect | Learning scheduler</text>

  <!-- Ledger + Sleep -->
  <rect x="590" y="1005" width="205" height="45" rx="7" fill="#0a0025" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="693" y="1025" text-anchor="middle" fill="#ddd6fe" font-size="9" font-weight="bold">📒 LedgerStrand</text>
  <text x="693" y="1042" text-anchor="middle" fill="#a78bfa" font-size="7">Facts | Modules | Strands | DAG</text>

  <rect x="810" y="1005" width="215" height="45" rx="7" fill="#001a10" stroke="#10b981" stroke-width="1.5"/>
  <text x="918" y="1025" text-anchor="middle" fill="#6ee7b7" font-size="9" font-weight="bold">🌙 SleepLearner</text>
  <text x="918" y="1042" text-anchor="middle" fill="#059669" font-size="7">FF consolidation | Batch events</text>

  <!-- Hot-plug -->
  <rect x="590" y="1065" width="420" height="35" rx="7" fill="#0a0800" stroke="#78350f" stroke-width="1" stroke-dasharray="3"/>
  <text x="800" y="1087" text-anchor="middle" fill="#fbbf24" font-size="9">🔌 Hot-pluggable: New Hard Strands auto-discovered via Rust Trait</text>

  <!-- Output -->
  <rect x="590" y="1115" width="420" height="45" rx="8" fill="#0a0800" stroke="#92400e" stroke-width="1.5"/>
  <text x="800" y="1135" text-anchor="middle" fill="#fde68a" font-size="10" font-weight="bold">Output Packet</text>
  <text x="800" y="1153" text-anchor="middle" fill="#d97706" font-size="8">{ frame: TensorFrame, γ: [f32;16], proof: Vec&lt;Step&gt;, strand_id: u64 }</text>

  <!-- Runtime -->
  <rect x="590" y="1175" width="420" height="40" rx="8" fill="#0a0800" stroke="#78350f" stroke-width="1"/>
  <text x="800" y="1195" text-anchor="middle" fill="#d97706" font-size="9" font-weight="bold">Runtime: Tokio async + Rayon parallel | 16+ cores | Memory-safe Rust</text>
  <text x="800" y="1210" text-anchor="middle" fill="#92400e" font-size="7">Deterministic: same input → same output (unlike GPU neural approximation)</text>

  <!-- ═══════════════════════════ RAM / VoltDB ═══════════════════════════ -->
  <rect x="1070" y="510" width="430" height="750" rx="16" fill="#000a10" stroke="#0891b2" stroke-width="2.5" filter="url(#sg)"/>
  <rect x="1070" y="510" width="430" height="45" rx="16" fill="#0891b2" opacity="0.25"/>
  <text x="1285" y="536" text-anchor="middle" fill="#ecfeff" font-size="15" font-weight="bold">💾 LAYER 5: RAM — VoltDB</text>
  <text x="1285" y="550" text-anchor="middle" fill="#a5f3fc" font-size="8">"Hippocampus" | 192GB DDR5 + NVMe | Embedded Storage Engine</text>

  <!-- T1: Short-Term -->
  <rect x="1095" y="570" width="380" height="55" rx="8" fill="#0c3a4a" stroke="#22d3ee" stroke-width="2"/>
  <text x="1285" y="593" text-anchor="middle" fill="#cffafe" font-size="11" font-weight="bold">T1: Short-Term Memory (Session Strands)</text>
  <text x="1285" y="612" text-anchor="middle" fill="#67e8f9" font-size="8">RAM 8-32GB | ~500K Full Frames | O(1) pointer swap for context switch</text>

  <!-- Strand boxes -->
  <rect x="1095" y="640" width="115" height="45" rx="6" fill="#083344" stroke="#0891b2" stroke-width="1.2"/>
  <text x="1153" y="658" text-anchor="middle" fill="#a5f3fc" font-size="8" font-weight="bold">Coding #01</text>
  <text x="1153" y="674" text-anchor="middle" fill="#0891b2" font-size="6">12K Frames | 768KB</text>

  <rect x="1220" y="640" width="115" height="45" rx="6" fill="#083344" stroke="#0891b2" stroke-width="1.2"/>
  <text x="1278" y="658" text-anchor="middle" fill="#a5f3fc" font-size="8" font-weight="bold">Sociology #02</text>
  <text x="1278" y="674" text-anchor="middle" fill="#0891b2" font-size="6">4.2K Frames | 268KB</text>

  <rect x="1345" y="640" width="115" height="45" rx="6" fill="#083344" stroke="#0891b2" stroke-width="1.2"/>
  <text x="1403" y="658" text-anchor="middle" fill="#a5f3fc" font-size="8" font-weight="bold">Narae #03</text>
  <text x="1403" y="674" text-anchor="middle" fill="#0891b2" font-size="6">8.5K Frames | 544KB</text>

  <rect x="1095" y="695" width="115" height="45" rx="6" fill="#083344" stroke="#0891b2" stroke-width="1"/>
  <text x="1153" y="713" text-anchor="middle" fill="#a5f3fc" font-size="8" font-weight="bold">Personal #04</text>
  <text x="1153" y="729" text-anchor="middle" fill="#0891b2" font-size="6">2.1K | IMMORTAL</text>

  <rect x="1220" y="695" width="115" height="45" rx="6" fill="#062a30" stroke="#065f46" stroke-width="1"/>
  <text x="1278" y="713" text-anchor="middle" fill="#6ee7b7" font-size="8" font-weight="bold">Cooking #05</text>
  <text x="1278" y="729" text-anchor="middle" fill="#059669" font-size="6">GRADUATED ✨</text>

  <rect x="1345" y="695" width="115" height="45" rx="6" fill="#062030" stroke="#475569" stroke-width="1" stroke-dasharray="3"/>
  <text x="1403" y="718" text-anchor="middle" fill="#64748b" font-size="8">... ∞ strands</text>

  <!-- T2: Long-Term -->
  <rect x="1095" y="760" width="380" height="55" rx="8" fill="#1a1000" stroke="#f59e0b" stroke-width="2"/>
  <text x="1285" y="783" text-anchor="middle" fill="#fde68a" font-size="11" font-weight="bold">T2: Long-Term Memory (Lifetime Archive)</text>
  <text x="1285" y="802" text-anchor="middle" fill="#fcd34d" font-size="8">RAM + NVMe 64-160+ GB | Millions of compressed Frames (R₀ only, 1KB each)</text>

  <!-- VoltDB Internals -->
  <rect x="1095" y="830" width="380" height="200" rx="10" fill="#001520" stroke="#0e7490" stroke-width="1.5"/>
  <text x="1285" y="852" text-anchor="middle" fill="#a5f3fc" font-size="10" font-weight="bold">VoltDB Internals (Embedded Storage Engine)</text>

  <rect x="1110" y="863" width="170" height="32" rx="5" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="1195" y="884" text-anchor="middle" fill="#67e8f9" font-size="8">HNSW Semantic Index</text>

  <rect x="1295" y="863" width="165" height="32" rx="5" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="1378" y="884" text-anchor="middle" fill="#67e8f9" font-size="8">B-tree Temporal Index</text>

  <rect x="1110" y="900" width="170" height="32" rx="5" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="1195" y="921" text-anchor="middle" fill="#67e8f9" font-size="8">Inverted Slot Index</text>

  <rect x="1295" y="900" width="165" height="32" rx="5" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="1378" y="921" text-anchor="middle" fill="#67e8f9" font-size="8">LSM-Tree Storage</text>

  <rect x="1110" y="937" width="170" height="32" rx="5" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="1195" y="958" text-anchor="middle" fill="#67e8f9" font-size="8">Bloom Filters</text>

  <rect x="1295" y="937" width="165" height="32" rx="5" fill="#083344" stroke="#0891b2" stroke-width="0.8"/>
  <text x="1378" y="958" text-anchor="middle" fill="#67e8f9" font-size="8">MVCC Concurrency</text>

  <rect x="1110" y="974" width="350" height="28" rx="5" fill="#062030" stroke="#0e7490" stroke-width="0.8"/>
  <text x="1285" y="993" text-anchor="middle" fill="#0891b2" font-size="7">WAL (crash recovery) | mmap paging | rkyv zero-copy | crossbeam-epoch RCU</text>

  <!-- Bleed Engine -->
  <rect x="1095" y="1045" width="380" height="50" rx="8" fill="#0a0025" stroke="#e879f9" stroke-width="1.5"/>
  <text x="1285" y="1068" text-anchor="middle" fill="#f0abfc" font-size="10" font-weight="bold">Bleed Engine (Async Prefetch)</text>
  <text x="1285" y="1085" text-anchor="middle" fill="#d946ef" font-size="7">Predictive: R₀ gists → GPU ghost buffer | On-demand: page fault → full Frame load</text>

  <!-- GC -->
  <rect x="1095" y="1110" width="180" height="50" rx="8" fill="#1a0015" stroke="#ec4899" stroke-width="1"/>
  <text x="1185" y="1133" text-anchor="middle" fill="#fbcfe8" font-size="9" font-weight="bold">🧹 GC Pipeline</text>
  <text x="1185" y="1148" text-anchor="middle" fill="#ec4899" font-size="7">Retention score | 4-tier decay</text>

  <!-- Consolidation -->
  <rect x="1290" y="1110" width="185" height="50" rx="8" fill="#001a10" stroke="#10b981" stroke-width="1"/>
  <text x="1383" y="1133" text-anchor="middle" fill="#6ee7b7" font-size="9" font-weight="bold">💎 Consolidation</text>
  <text x="1383" y="1148" text-anchor="middle" fill="#059669" font-size="7">Raw→Compress→Distill→Tombstone</text>

  <!-- Coherence -->
  <rect x="1095" y="1175" width="380" height="40" rx="8" fill="#0a0a10" stroke="#475569" stroke-width="1"/>
  <text x="1285" y="1195" text-anchor="middle" fill="#94a3b8" font-size="9" font-weight="bold">Coherence Checker: γ-priority resolution | Versioned truth | Contradiction detection</text>
  <text x="1285" y="1210" text-anchor="middle" fill="#64748b" font-size="7">Background CPU job: scan for semantically opposite Frames, flag for resolution</text>

  <!-- ═══════════════════════════ INTER-PILLAR ARROWS ═══════════════════════════ -->
  <!-- GPU ↔ CPU -->
  <line x1="530" y1="780" x2="590" y2="780" stroke="#e879f9" stroke-width="2.5" filter="url(#sg)"/>
  <polygon points="587,776 595,780 587,784" fill="#e879f9"/>
  <polygon points="533,776 525,780 533,784" fill="#e879f9"/>
  <text x="560" y="770" text-anchor="middle" fill="#e879f9" font-size="7" font-weight="bold">Intent→</text>
  <text x="560" y="795" text-anchor="middle" fill="#e879f9" font-size="7" font-weight="bold">←Result</text>

  <!-- CPU ↔ RAM -->
  <line x1="1035" y1="780" x2="1095" y2="780" stroke="#22d3ee" stroke-width="2.5" filter="url(#sg)"/>
  <polygon points="1092,776 1100,780 1092,784" fill="#22d3ee"/>
  <polygon points="1038,776 1030,780 1038,784" fill="#22d3ee"/>
  <text x="1065" y="770" text-anchor="middle" fill="#22d3ee" font-size="7" font-weight="bold">Load→</text>
  <text x="1065" y="795" text-anchor="middle" fill="#22d3ee" font-size="7" font-weight="bold">←Store</text>

  <!-- GPU ↔ RAM (bleed) -->
  <path d="M 480 1040 Q 480 1260 780 1275 Q 1080 1290 1095 1070" stroke="#e879f9" stroke-width="1.5" fill="none" stroke-dasharray="5" marker-end="url(#aPi)"/>
  <text x="780" y="1295" text-anchor="middle" fill="#d946ef" font-size="8">Ghost bleed: RAM R₀ → GPU bleed buffer (async)</text>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- LAYER 2 BIS: LLL TENSOR FRAME BUS (Output) -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <line x1="800" y1="1260" x2="800" y2="1320" stroke="#8b5cf6" stroke-width="2" marker-end="url(#aP)"/>

  <rect x="200" y="1325" width="1200" height="40" rx="20" fill="url(#busGr)" stroke="#8b5cf6" stroke-width="2.5" filter="url(#gl)"/>
  <text x="800" y="1351" text-anchor="middle" fill="#ede9fe" font-size="13" font-weight="bold">⚡ LLL TENSOR FRAME BUS — Output (Verified Frame + per-slot γ + Proof Chain)</text>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- LAYER 6: OUTPUT ACTION CORES -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <line x1="800" y1="1365" x2="800" y2="1395" stroke="#8b5cf6" stroke-width="2" marker-end="url(#aP)"/>

  <rect x="100" y="1400" width="1400" height="100" rx="14" fill="#060f15" stroke="#10b981" stroke-width="2"/>
  <text x="130" y="1421" fill="#6ee7b7" font-size="10" letter-spacing="3">LAYER 6 — OUTPUT ACTION CORES</text>
  <text x="800" y="1437" text-anchor="middle" fill="#e2e8f0" font-size="12" font-weight="bold">Community Modules: impl ActionCore { fn decode(frame: TensorFrame) → Output }</text>
  <text x="800" y="1453" text-anchor="middle" fill="#94a3b8" font-size="9">Parallel decode: ALL Frame slots emit simultaneously → assemble → output</text>

  <rect x="125" y="1462" width="120" height="28" rx="6" fill="#0a2a15" stroke="#22c55e" stroke-width="1"/>
  <text x="185" y="1481" text-anchor="middle" fill="#86efac" font-size="8">💬 Text</text>

  <rect x="260" y="1462" width="120" height="28" rx="6" fill="#0a2a15" stroke="#22c55e" stroke-width="1"/>
  <text x="320" y="1481" text-anchor="middle" fill="#86efac" font-size="8">🔊 Speech</text>

  <rect x="395" y="1462" width="120" height="28" rx="6" fill="#0a2a15" stroke="#22c55e" stroke-width="1"/>
  <text x="455" y="1481" text-anchor="middle" fill="#86efac" font-size="8">🖼️ Image</text>

  <rect x="530" y="1462" width="120" height="28" rx="6" fill="#0a2a15" stroke="#22c55e" stroke-width="1"/>
  <text x="590" y="1481" text-anchor="middle" fill="#86efac" font-size="8">🤖 Motor</text>

  <rect x="665" y="1462" width="120" height="28" rx="6" fill="#0a2a15" stroke="#22c55e" stroke-width="1"/>
  <text x="725" y="1481" text-anchor="middle" fill="#86efac" font-size="8">📧 n8n</text>

  <rect x="800" y="1462" width="120" height="28" rx="6" fill="#0a2a15" stroke="#22c55e" stroke-width="1"/>
  <text x="860" y="1481" text-anchor="middle" fill="#86efac" font-size="8">🔗 Ledger</text>

  <rect x="935" y="1462" width="100" height="28" rx="6" fill="#051a10" stroke="#475569" stroke-width="1" stroke-dasharray="3"/>
  <text x="985" y="1481" text-anchor="middle" fill="#64748b" font-size="8">➕</text>

  <rect x="1050" y="1462" width="100" height="28" rx="6" fill="#051a10" stroke="#475569" stroke-width="1" stroke-dasharray="3"/>
  <text x="1100" y="1481" text-anchor="middle" fill="#64748b" font-size="8">➕</text>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- LAYER 7: CONTINUAL LEARNING ENGINE -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <line x1="800" y1="1500" x2="800" y2="1530" stroke="#f59e0b" stroke-width="2" marker-end="url(#aA)"/>
  <text x="800" y="1555" text-anchor="middle" fill="#475569" font-size="10" letter-spacing="4">LAYER 7 — CONTINUAL LEARNING ENGINE</text>

  <rect x="100" y="1565" width="1400" height="170" rx="14" fill="#050a05" stroke="#f59e0b" stroke-width="2"/>

  <!-- Timescale 1 -->
  <rect x="125" y="1585" width="410" height="130" rx="10" fill="#001a20" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="330" y="1608" text-anchor="middle" fill="#a5f3fc" font-size="12" font-weight="bold">⚡ Instant Learning (ms–min)</text>
  <text x="330" y="1625" text-anchor="middle" fill="#67e8f9" font-size="8">Hardware: RAM only | Strand state write | Zero forgetting</text>
  <text x="330" y="1645" text-anchor="middle" fill="#22d3ee" font-size="8">"Call me Alex" → Personal Strand updated instantly</text>
  <text x="330" y="1665" text-anchor="middle" fill="#22d3ee" font-size="8">No weight change. Pure memory. Isolated per-strand.</text>
  <rect x="140" y="1682" width="380" height="22" rx="4" fill="#0a2a20" stroke="#059669" stroke-width="0.8"/>
  <text x="330" y="1698" text-anchor="middle" fill="#34d399" font-size="7">Bio: Working memory / Short-term memory</text>

  <!-- Timescale 2 -->
  <rect x="560" y="1585" width="410" height="130" rx="10" fill="#1a0a00" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="765" y="1608" text-anchor="middle" fill="#fde68a" font-size="12" font-weight="bold">🌙 Sleep Learning (hours)</text>
  <text x="765" y="1625" text-anchor="middle" fill="#fcd34d" font-size="8">Hardware: CPU + GPU idle | Forward-Forward layer-by-layer</text>
  <text x="765" y="1645" text-anchor="middle" fill="#fbbf24" font-size="8">CPU batches learning events → GPU updates weights</text>
  <text x="765" y="1665" text-anchor="middle" fill="#fbbf24" font-size="8">Frame distillation: 50 raw → 3 wisdom Frames</text>
  <rect x="575" y="1682" width="380" height="22" rx="4" fill="#2a2000" stroke="#059669" stroke-width="0.8"/>
  <text x="765" y="1698" text-anchor="middle" fill="#34d399" font-size="7">Bio: Sleep consolidation / Memory replay</text>

  <!-- Timescale 3 -->
  <rect x="995" y="1585" width="480" height="130" rx="10" fill="#0a0020" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="1235" y="1608" text-anchor="middle" fill="#ddd6fe" font-size="12" font-weight="bold">🌱 Developmental Growth (days–months)</text>
  <text x="1235" y="1625" text-anchor="middle" fill="#c4b5fd" font-size="8">Hardware: RAM + Ecosystem | Strand graduation + Module hot-plug</text>
  <text x="1235" y="1645" text-anchor="middle" fill="#a78bfa" font-size="8">Mirror detects: "user talks about cooking a lot" → new Strand</text>
  <text x="1235" y="1665" text-anchor="middle" fill="#a78bfa" font-size="8">Community module installed: volt install translator-recipe</text>
  <rect x="1010" y="1682" width="450" height="22" rx="4" fill="#200030" stroke="#059669" stroke-width="0.8"/>
  <text x="1235" y="1698" text-anchor="middle" fill="#34d399" font-size="7">Bio: Brain development / Neuroplasticity</text>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- LAYER 8: INTELLIGENCE COMMONS -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <line x1="800" y1="1735" x2="800" y2="1765" stroke="#8b5cf6" stroke-width="2" marker-end="url(#aP)"/>
  <text x="800" y="1790" text-anchor="middle" fill="#475569" font-size="10" letter-spacing="4">LAYER 8 — INTELLIGENCE COMMONS (POST-BLOCKCHAIN)</text>

  <rect x="100" y="1800" width="1400" height="230" rx="14" fill="#050010" stroke="#6d28d9" stroke-width="2"/>

  <!-- Local -->
  <rect x="125" y="1820" width="390" height="55" rx="8" fill="#001a20" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="320" y="1843" text-anchor="middle" fill="#a5f3fc" font-size="11" font-weight="bold">L0: Local Instance (Offline-first)</text>
  <text x="320" y="1863" text-anchor="middle" fill="#67e8f9" font-size="7">Append-only event log | Merkle-hashed state | Ed25519 wallet | ZK privacy</text>

  <!-- P2P -->
  <rect x="125" y="1885" width="390" height="55" rx="8" fill="#100020" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="320" y="1908" text-anchor="middle" fill="#ddd6fe" font-size="11" font-weight="bold">L1: P2P Gossip Mesh (libp2p)</text>
  <text x="320" y="1928" text-anchor="middle" fill="#a78bfa" font-size="7">CRDT merge | Module registry (IPFS CIDs) | Fact gossip + verification attestations</text>

  <!-- Settlement -->
  <rect x="125" y="1950" width="390" height="55" rx="8" fill="#1a0a00" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="320" y="1973" text-anchor="middle" fill="#fde68a" font-size="11" font-weight="bold">L2: Settlement Layer (DAG)</text>
  <text x="320" y="1993" text-anchor="middle" fill="#fcd34d" font-size="7">Batched micropayments | Fact anchoring (high-γ) | Provenance registry | Governance</text>

  <!-- 4 Value Flows -->
  <rect x="545" y="1820" width="530" height="185" rx="10" fill="#0a0015" stroke="#6d28d9" stroke-width="1.5"/>
  <text x="810" y="1843" text-anchor="middle" fill="#ddd6fe" font-size="11" font-weight="bold">Four Value Flows — Proof-of-Contribution</text>

  <rect x="565" y="1855" width="240" height="32" rx="5" fill="#12004a" stroke="#a78bfa" stroke-width="0.8"/>
  <text x="685" y="1876" text-anchor="middle" fill="#c4b5fd" font-size="8">📚 Knowledge Contribution</text>

  <rect x="820" y="1855" width="240" height="32" rx="5" fill="#001a10" stroke="#10b981" stroke-width="0.8"/>
  <text x="940" y="1876" text-anchor="middle" fill="#6ee7b7" font-size="8">🧩 Module Marketplace</text>

  <rect x="565" y="1895" width="240" height="32" rx="5" fill="#1a0a00" stroke="#f59e0b" stroke-width="0.8"/>
  <text x="685" y="1916" text-anchor="middle" fill="#fde68a" font-size="8">✅ Fact Verification</text>

  <rect x="820" y="1895" width="240" height="32" rx="5" fill="#001520" stroke="#22d3ee" stroke-width="0.8"/>
  <text x="940" y="1916" text-anchor="middle" fill="#a5f3fc" font-size="8">🔄 Strand Trading</text>

  <rect x="565" y="1940" width="495" height="45" rx="6" fill="#05000f" stroke="#4c1d95" stroke-width="1"/>
  <text x="813" y="1960" text-anchor="middle" fill="#c4b5fd" font-size="9" font-weight="bold">Token: VOLT — 100% earned, zero pre-mine, quadratic governance</text>
  <text x="813" y="1978" text-anchor="middle" fill="#8b5cf6" font-size="7">Shielded transactions for strand trading | Progressive contribution fees</text>

  <!-- Distribution model -->
  <rect x="1105" y="1820" width="370" height="185" rx="10" fill="#0a0015" stroke="#6d28d9" stroke-width="1"/>
  <text x="1290" y="1843" text-anchor="middle" fill="#ddd6fe" font-size="10" font-weight="bold">Not Worldcoin. Not Charity.</text>
  <text x="1290" y="1868" text-anchor="middle" fill="#a78bfa" font-size="9">Value created locally, traded P2P</text>
  <text x="1290" y="1888" text-anchor="middle" fill="#a78bfa" font-size="9">Users are creators AND owners</text>
  <text x="1290" y="1908" text-anchor="middle" fill="#a78bfa" font-size="9">No single entity controls the tap</text>
  <text x="1290" y="1928" text-anchor="middle" fill="#a78bfa" font-size="9">Verification via contribution, not biometrics</text>
  <rect x="1120" y="1948" width="340" height="40" rx="6" fill="#0a0020" stroke="#7c3aed" stroke-width="1"/>
  <text x="1290" y="1966" text-anchor="middle" fill="#c4b5fd" font-size="9" font-weight="bold">A commons, not a kingdom.</text>
  <text x="1290" y="1982" text-anchor="middle" fill="#8b5cf6" font-size="7">Fair distribution of intelligence and value generated from it.</text>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- LAYER 9: INTERRUPT HANDLING -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <text x="800" y="2060" text-anchor="middle" fill="#475569" font-size="10" letter-spacing="4">CROSS-CUTTING — INTERRUPT &amp; CONTEXT MANAGEMENT</text>

  <rect x="100" y="2075" width="1400" height="70" rx="14" fill="#050a10" stroke="#06b6d4" stroke-width="1.5"/>

  <rect x="130" y="2090" width="260" height="40" rx="8" fill="#0c3a4a" stroke="#22d3ee" stroke-width="1"/>
  <text x="260" y="2108" text-anchor="middle" fill="#cffafe" font-size="9" font-weight="bold">1. Pause Active Strand</text>
  <text x="260" y="2123" text-anchor="middle" fill="#67e8f9" font-size="7">Save Frame state → RAM pointer</text>

  <line x1="390" y1="2110" x2="420" y2="2110" stroke="#22d3ee" stroke-width="1.5" marker-end="url(#aC)"/>

  <rect x="425" y="2090" width="260" height="40" rx="8" fill="#0c3a4a" stroke="#22d3ee" stroke-width="1"/>
  <text x="555" y="2108" text-anchor="middle" fill="#cffafe" font-size="9" font-weight="bold">2. Handle Interrupt</text>
  <text x="555" y="2123" text-anchor="middle" fill="#67e8f9" font-size="7">Temp strand for off-topic</text>

  <line x1="685" y1="2110" x2="715" y2="2110" stroke="#22d3ee" stroke-width="1.5" marker-end="url(#aC)"/>

  <rect x="720" y="2090" width="260" height="40" rx="8" fill="#0c3a4a" stroke="#22d3ee" stroke-width="1"/>
  <text x="850" y="2108" text-anchor="middle" fill="#cffafe" font-size="9" font-weight="bold">3. Discard Temp</text>
  <text x="850" y="2123" text-anchor="middle" fill="#67e8f9" font-size="7">Free RAM, no context pollution</text>

  <line x1="980" y1="2110" x2="1010" y2="2110" stroke="#22d3ee" stroke-width="1.5" marker-end="url(#aC)"/>

  <rect x="1015" y="2090" width="310" height="40" rx="8" fill="#0a2a20" stroke="#22c55e" stroke-width="1.5"/>
  <text x="1170" y="2108" text-anchor="middle" fill="#86efac" font-size="9" font-weight="bold">4. Resume Bit-Perfect ✓</text>
  <text x="1170" y="2123" text-anchor="middle" fill="#4ade80" font-size="7">Reload exact Frame state, zero context loss</text>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- LAYER 10: UI / TEST BENCH -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <text x="800" y="2180" text-anchor="middle" fill="#475569" font-size="10" letter-spacing="4">LAYER 9 — UI / TEST BENCH (PHASE 1)</text>

  <rect x="200" y="2195" width="1200" height="110" rx="14" fill="#0a0510" stroke="#ec4899" stroke-width="2"/>

  <rect x="230" y="2215" width="140" height="45" rx="8" fill="#1e293b" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="300" y="2235" text-anchor="middle" fill="#ddd6fe" font-size="9" font-weight="bold">💬 Chat Trigger</text>
  <text x="300" y="2250" text-anchor="middle" fill="#a78bfa" font-size="7">/volt/chat webhook</text>

  <line x1="370" y1="2238" x2="400" y2="2238" stroke="#6366f1" stroke-width="1.5"/>

  <rect x="400" y="2215" width="160" height="45" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="480" y="2235" text-anchor="middle" fill="#fde68a" font-size="9" font-weight="bold">⚡ HTTP → Rust</text>
  <text x="480" y="2250" text-anchor="middle" fill="#fcd34d" font-size="7">POST :8080/api/think</text>

  <line x1="560" y1="2238" x2="590" y2="2238" stroke="#f59e0b" stroke-width="1.5"/>

  <rect x="590" y="2210" width="160" height="58" rx="8" fill="#1e293b" stroke="#ec4899" stroke-width="1.5"/>
  <text x="670" y="2230" text-anchor="middle" fill="#fbcfe8" font-size="9" font-weight="bold">🔀 Switch</text>
  <text x="670" y="2246" text-anchor="middle" fill="#f9a8d4" font-size="7">"text" → reply</text>
  <text x="670" y="2258" text-anchor="middle" fill="#fcd34d" font-size="7">"tool" → route</text>

  <line x1="750" y1="2238" x2="780" y2="2238" stroke="#10b981" stroke-width="1.5"/>

  <rect x="780" y="2215" width="140" height="45" rx="8" fill="#1e293b" stroke="#10b981" stroke-width="1.5"/>
  <text x="850" y="2235" text-anchor="middle" fill="#6ee7b7" font-size="9" font-weight="bold">💬 Reply</text>
  <text x="850" y="2250" text-anchor="middle" fill="#34d399" font-size="7">+ γ + strand_id + proof</text>

  <!-- Debug panel -->
  <rect x="960" y="2215" width="410" height="45" rx="8" fill="#0f172a" stroke="#334155" stroke-width="1"/>
  <text x="1165" y="2233" text-anchor="middle" fill="#94a3b8" font-size="8" font-weight="bold">📊 Debug Panel (n8n visual flow)</text>
  <text x="1165" y="2250" text-anchor="middle" fill="#64748b" font-size="7">Strand routing, thinking loops, γ scores, Frame contents, timing</text>

  <rect x="230" y="2270" width="1140" height="22" rx="4" fill="#0f172a" stroke="#334155" stroke-width="0.8"/>
  <text x="250" y="2285" fill="#6366f1" font-size="8" font-family="monospace">[LOG] STRAND→Coding#01 | THINK→3 loops, 6 slots filled | HARD→MathEngine | γ=[1.0,0.85,0.92,0.70] | 142ms</text>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- LAYER 11: TRAIT INTERFACES (SOCKET STANDARD) -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <text x="800" y="2340" text-anchor="middle" fill="#475569" font-size="10" letter-spacing="4">LAYER 10 — SOCKET STANDARD (TRAIT INTERFACES)</text>

  <rect x="100" y="2355" width="1400" height="80" rx="14" fill="#050510" stroke="#475569" stroke-width="1.5"/>
  <text x="800" y="2378" text-anchor="middle" fill="#94a3b8" font-size="11" font-weight="bold">🔧 The "AM5 Socket" for AI — Rust Trait Interfaces define the ecosystem boundary</text>

  <rect x="125" y="2395" width="400" height="28" rx="5" fill="#0a0a15" stroke="#22c55e" stroke-width="1"/>
  <text x="325" y="2414" text-anchor="middle" fill="#86efac" font-size="9" font-family="monospace">pub trait Translator { fn encode() → TensorFrame }</text>

  <rect x="555" y="2395" width="400" height="28" rx="5" fill="#0a0a15" stroke="#fbbf24" stroke-width="1"/>
  <text x="755" y="2414" text-anchor="middle" fill="#fef3c7" font-size="9" font-family="monospace">pub trait HardStrand { fn execute() → StrandResult }</text>

  <rect x="985" y="2395" width="400" height="28" rx="5" fill="#0a0a15" stroke="#10b981" stroke-width="1"/>
  <text x="1185" y="2414" text-anchor="middle" fill="#6ee7b7" font-size="9" font-family="monospace">pub trait ActionCore { fn decode(TensorFrame) → Out }</text>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- DATA FLOW SUMMARY -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <rect x="100" y="2470" width="1400" height="320" rx="14" fill="#050510" stroke="#334155" stroke-width="1"/>
  <text x="800" y="2500" text-anchor="middle" fill="#e2e8f0" font-size="14" font-weight="bold">COMPLETE DATA FLOW — Happy Path</text>

  <text x="130" y="2530" fill="#94a3b8" font-size="10" font-family="monospace"> 1. USER → Input Translator (community module) → TensorFrame</text>
  <text x="130" y="2552" fill="#94a3b8" font-size="10" font-family="monospace"> 2. TensorFrame → LLL Bus → GPU Soft Core (State Buffer)</text>
  <text x="130" y="2574" fill="#c4b5fd" font-size="10" font-family="monospace"> 3. GPU: SDE dynamics on Frame R₀ slots → converge per-slot → Candidate Frame</text>
  <text x="130" y="2596" fill="#fcd34d" font-size="10" font-family="monospace"> 4. Candidate Frame → CPU Hard Core (Intent Router → Best Hard Strand)</text>
  <text x="130" y="2618" fill="#fcd34d" font-size="10" font-family="monospace"> 5. CPU: HDC verify, certainty propagation, proof chain, safety check</text>
  <text x="130" y="2640" fill="#fcd34d" font-size="10" font-family="monospace"> 6. IF γ ≥ threshold → Output | IF γ &lt; threshold → Feedback to GPU (loop)</text>
  <text x="130" y="2662" fill="#67e8f9" font-size="10" font-family="monospace"> 7. VoltDB: Bleed engine prefetches relevant ghosts from T1/T2 (async)</text>
  <text x="130" y="2684" fill="#67e8f9" font-size="10" font-family="monospace"> 8. VoltDB: Consolidation evicts old Frames T0→T1, compresses T1→T2 (async)</text>
  <text x="130" y="2706" fill="#86efac" font-size="10" font-family="monospace"> 9. Verified Frame → LLL Bus → Output Action Core (parallel slot decode)</text>
  <text x="130" y="2728" fill="#86efac" font-size="10" font-family="monospace">10. Action Core → Assembled output (text/speech/image/action) → USER</text>

  <line x1="130" y1="2745" x2="1470" y2="2745" stroke="#1e293b" stroke-width="0.5"/>

  <text x="800" y="2770" text-anchor="middle" fill="#94a3b8" font-size="10" font-weight="bold">Safety Intervention (at any step): Transition Monitor → Violation Scorer → Omega Veto (HALT)</text>

  <!-- ════════════════════════════════════════════════════════════════ -->
  <!-- FOOTER -->
  <!-- ════════════════════════════════════════════════════════════════ -->
  <line x1="150" y1="2810" x2="1450" y2="2810" stroke="#1e293b" stroke-width="1"/>
  <text x="800" y="2840" text-anchor="middle" fill="#475569" font-size="10">Volt v3.0 — "The Lipstick Masquerade" — Rust Core · Tensor Frames · Forward-Forward · VoltDB · Intelligence Commons · Consumer Hardware</text>
  <text x="800" y="2860" text-anchor="middle" fill="#334155" font-size="9" font-style="italic">"We're playing a dangerous game, and things will never be the same."</text>
</svg>
