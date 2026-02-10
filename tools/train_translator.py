#!/usr/bin/env python3
"""
Train the Frame Projection Head for Volt X Milestone 2.2.

Downloads Qwen3-0.6B, pre-computes hidden states on PropBank SRL data,
trains a 3-layer MLP projection head (role classifier + embedding projector),
and exports weights as safetensors for Rust inference.

Architecture:
    Input: hidden_states [seq_len, 1024] (from frozen Qwen3-0.6B)
    MLP: 1024 → 4096 (GELU) → 4096 (GELU) → 4096 (GELU)
    Role Head: 4096 → 16 (cross-entropy loss)
    Embed Head: 4096 → 256 (codebook commitment loss)
    Reconstruction Decoder: 256 → 4096 (GELU) → hidden_dim (MSE reconstruction loss)

Weight naming convention (matches Rust loader):
    proj.mlp.0.weight [4096, 1024]   proj.mlp.0.bias [4096]
    proj.mlp.1.weight [4096, 4096]   proj.mlp.1.bias [4096]
    proj.mlp.2.weight [4096, 4096]   proj.mlp.2.bias [4096]
    proj.role_head.weight [16, 4096] proj.role_head.bias [16]
    proj.embed_head.weight [256, 4096] proj.embed_head.bias [256]

Requirements:
    pip install torch transformers datasets safetensors tqdm

Usage:
    # Full training pipeline
    python tools/train_translator.py --output projection.safetensors

    # Resume from checkpoint
    python tools/train_translator.py --output projection.safetensors --resume checkpoint.pt

    # Quick test run (tiny dataset, 2 epochs)
    python tools/train_translator.py --output projection.safetensors --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ─── Constants ──────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-0.6B"
HIDDEN_DIM = 1024       # Qwen3-0.6B hidden dimension
MLP_DIM = 4096          # Projection head MLP dimension
NUM_ROLES = 16          # Number of slot roles
SLOT_DIM = 256          # Embedding dimension per slot

# PropBank label → role index (must match roles.rs exactly)
PROPBANK_TO_ROLE = {
    "ARG0": 0, "A0": 0,
    "V": 1, "REL": 1,
    "ARG1": 2, "A1": 2,
    "ARGM-LOC": 3, "AM-LOC": 3,
    "ARGM-TMP": 4, "AM-TMP": 4,
    "ARGM-MNR": 5, "AM-MNR": 5,
    "ARG2": 6, "A2": 6,
    "ARGM-CAU": 7, "AM-CAU": 7,
    "ARG3": 8, "A3": 8,
    "ARG4": 8, "A4": 8,
    "ARGM-DIR": 9, "AM-DIR": 9,
    "ARGM-ADV": 10, "AM-ADV": 10,
    "ARGM-PNC": 11, "AM-PNC": 11,
    "ARGM-DIS": 12, "AM-DIS": 12,
    "ARGM-NEG": 13, "AM-NEG": 13,
    "ARGM-MOD": 14, "AM-MOD": 14,
    "O": 15, "_": 15, "NONE": 15,
}


# ─── Model ──────────────────────────────────────────────────────────

class FrameProjectionHead(nn.Module):
    """3-layer MLP with role classifier and embedding projector heads."""

    def __init__(self, hidden_dim=HIDDEN_DIM, mlp_dim=MLP_DIM,
                 num_roles=NUM_ROLES, slot_dim=SLOT_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.GELU(),
        )
        self.role_head = nn.Linear(mlp_dim, num_roles)
        self.embed_head = nn.Linear(mlp_dim, slot_dim)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            role_logits: [batch, seq_len, num_roles]
            token_embeds: [batch, seq_len, slot_dim]
        """
        x = self.mlp(hidden_states)
        role_logits = self.role_head(x)
        token_embeds = self.embed_head(x)
        return role_logits, token_embeds


class ReconstructionDecoder(nn.Module):
    """Lightweight decoder: slot embeddings -> reconstructed hidden states.

    Maps predicted 256-dim slot embeddings back to the original hidden_dim
    space. Used for round-trip reconstruction loss (Milestone 2.2 spec).
    This is a training-only component; weights are NOT exported for Rust inference.
    """

    def __init__(self, slot_dim=SLOT_DIM, intermediate_dim=MLP_DIM,
                 hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )

    def forward(self, token_embeds):
        """
        Args:
            token_embeds: [batch, seq_len, slot_dim]

        Returns:
            reconstructed: [batch, seq_len, hidden_dim]
        """
        return self.decoder(token_embeds)


# ─── Dataset ────────────────────────────────────────────────────────

class SRLDataset(Dataset):
    """Dataset of (hidden_states, role_labels) pairs.

    Pre-computes hidden states from the frozen backbone and caches
    them to disk for fast re-use across training runs.
    """

    def __init__(self, hidden_states_list, role_labels_list):
        self.hidden_states = hidden_states_list
        self.role_labels = role_labels_list

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return self.hidden_states[idx], self.role_labels[idx]


def collate_fn(batch):
    """Pad sequences to the same length within a batch."""
    hidden_states, role_labels = zip(*batch)
    max_len = max(h.size(0) for h in hidden_states)

    padded_hidden = torch.zeros(len(batch), max_len, hidden_states[0].size(-1))
    padded_labels = torch.full((len(batch), max_len), NUM_ROLES - 1, dtype=torch.long)  # NoRole
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, (h, l) in enumerate(zip(hidden_states, role_labels)):
        seq_len = h.size(0)
        padded_hidden[i, :seq_len] = h
        padded_labels[i, :seq_len] = l
        mask[i, :seq_len] = True

    return padded_hidden, padded_labels, mask


# ─── Data Loading ───────────────────────────────────────────────────

def load_propbank_data(quick=False):
    """Load PropBank SRL data from HuggingFace datasets.

    Tries multiple dataset sources, falls back to rich synthetic data.
    Returns list of (words, role_labels) tuples.
    """
    from datasets import load_dataset
    from tqdm import tqdm

    # Try multiple CoNLL-2012 sources in order
    dataset_attempts = [
        ("ontonotes/conll2012_ontonotesv5", "english_v12"),
        ("ontonotes/conll2012_ontonotesv5", "english_v4"),
        ("conll2012_ontonotesv5", "english_v4"),
    ]

    dataset = None
    for repo, config in dataset_attempts:
        print(f"Trying {repo} ({config})...")
        try:
            dataset = load_dataset(repo, config, split="train",
                                   trust_remote_code=True)
            print(f"  Loaded successfully.")
            break
        except TypeError:
            # datasets 4.x removed trust_remote_code — try without
            try:
                dataset = load_dataset(repo, config, split="train")
                print(f"  Loaded successfully.")
                break
            except Exception as e:
                print(f"  Failed: {e}")
        except Exception as e:
            print(f"  Failed: {e}")

    if dataset is None:
        print("No CoNLL-2012 dataset available. Using synthetic data.")
        return generate_synthetic_data(100 if quick else 10000)

    sentences = []
    for doc in tqdm(dataset, desc="Parsing SRL data", unit="doc"):
        for sent in doc.get("sentences", []):
            words = sent.get("words", [])
            srl = sent.get("srl_frames", [])
            if not words or not srl:
                continue
            # Take the first SRL frame
            frame = srl[0]
            tags = frame.get("frames", [])
            if len(tags) != len(words):
                continue
            role_indices = []
            for tag in tags:
                tag_upper = tag.upper().replace("B-", "").replace("I-", "")
                role_indices.append(PROPBANK_TO_ROLE.get(tag_upper, NUM_ROLES - 1))
            sentences.append((words, role_indices))

        if quick and len(sentences) >= 200:
            break

    if not sentences:
        print("No valid SRL data found. Using synthetic data.")
        return generate_synthetic_data(100 if quick else 10000)

    print(f"Loaded {len(sentences)} annotated sentences.")
    return sentences


def generate_synthetic_data(n=10000):
    """Generate diverse synthetic SRL training data.

    Uses vocabulary variation on structural templates to produce
    varied sentences covering all 16 slot roles.

    Roles: 0=Agent, 1=Predicate, 2=Patient, 3=Location, 4=Time,
           5=Manner, 6=Instrument, 7=Cause, 8=Result, 9=Direction,
           10=Adverbial, 11=Purpose, 12=Discourse, 13=Negation,
           14=Modal, 15=NoRole
    """
    import random
    random.seed(42)

    # Vocabulary pools for substitution
    agents = [
        "The cat", "The dog", "The teacher", "A scientist", "The manager",
        "She", "He", "The engineer", "A student", "The doctor",
        "The pilot", "A farmer", "The artist", "A child", "The chef",
        "My friend", "The officer", "A nurse", "The driver", "The worker",
    ]
    predicates = [
        "wrote", "built", "opened", "carried", "designed", "fixed",
        "threw", "painted", "cooked", "discovered", "explained", "sold",
        "cleaned", "moved", "broke", "created", "studied", "analyzed",
        "delivered", "tested", "repaired", "assembled", "launched",
    ]
    patients = [
        "a letter", "the report", "a bridge", "the door", "the package",
        "a painting", "the machine", "a program", "the device", "a message",
        "the system", "a solution", "the engine", "a prototype", "the plan",
    ]
    locations = [
        "in the office", "at the park", "on the table", "near the river",
        "at home", "in the lab", "on the roof", "at the station",
        "in the kitchen", "at the factory", "in the garden", "at school",
    ]
    times = [
        "yesterday", "last week", "in the morning", "on Monday",
        "before noon", "after lunch", "during the meeting", "at dawn",
        "tonight", "last summer", "in December", "two days ago",
    ]
    manners = [
        "quickly", "carefully", "silently", "eagerly", "precisely",
        "slowly", "gracefully", "efficiently", "hastily", "thoroughly",
    ]
    instruments = [
        "with a hammer", "with a pen", "using a computer", "with pliers",
        "using a wrench", "with a brush", "using a scanner", "with scissors",
    ]
    causes = [
        "because of the storm", "due to the delay", "because of the error",
        "due to high demand", "because of the failure", "due to budget cuts",
    ]
    directions = [
        "toward the exit", "into the room", "out of the building",
        "across the field", "along the corridor", "through the tunnel",
    ]
    purposes = [
        "to save time", "to improve safety", "in order to reduce costs",
        "to meet the deadline", "to fix the problem", "for the presentation",
    ]

    def _split(phrase):
        return phrase.split()

    # Template generators: each returns (words, roles)
    def _agent_pred_patient():
        a, p, o = random.choice(agents), random.choice(predicates), random.choice(patients)
        aw, pw, ow = _split(a), [p], _split(o)
        return aw + pw + ow, [0]*len(aw) + [1]*len(pw) + [2]*len(ow)

    def _agent_pred_patient_loc():
        a, p, o, l = random.choice(agents), random.choice(predicates), random.choice(patients), random.choice(locations)
        aw, pw, ow, lw = _split(a), [p], _split(o), _split(l)
        return aw + pw + ow + lw, [0]*len(aw) + [1]*len(pw) + [2]*len(ow) + [3]*len(lw)

    def _agent_pred_patient_time():
        a, p, o, t = random.choice(agents), random.choice(predicates), random.choice(patients), random.choice(times)
        aw, pw, ow, tw = _split(a), [p], _split(o), _split(t)
        return aw + pw + ow + tw, [0]*len(aw) + [1]*len(pw) + [2]*len(ow) + [4]*len(tw)

    def _agent_manner_pred_patient():
        a, m, p, o = random.choice(agents), random.choice(manners), random.choice(predicates), random.choice(patients)
        aw, mw, pw, ow = _split(a), [m], [p], _split(o)
        return aw + mw + pw + ow, [0]*len(aw) + [5]*len(mw) + [1]*len(pw) + [2]*len(ow)

    def _agent_pred_patient_instr():
        a, p, o, i = random.choice(agents), random.choice(predicates), random.choice(patients), random.choice(instruments)
        aw, pw, ow, iw = _split(a), [p], _split(o), _split(i)
        return aw + pw + ow + iw, [0]*len(aw) + [1]*len(pw) + [2]*len(ow) + [6]*len(iw)

    def _agent_pred_patient_cause():
        a, p, o, c = random.choice(agents), random.choice(predicates), random.choice(patients), random.choice(causes)
        aw, pw, ow, cw = _split(a), [p], _split(o), _split(c)
        return aw + pw + ow + cw, [0]*len(aw) + [1]*len(pw) + [2]*len(ow) + [7]*len(cw)

    def _agent_pred_patient_dir():
        a, p, o, d = random.choice(agents), random.choice(predicates), random.choice(patients), random.choice(directions)
        aw, pw, ow, dw = _split(a), [p], _split(o), _split(d)
        return aw + pw + ow + dw, [0]*len(aw) + [1]*len(pw) + [2]*len(ow) + [9]*len(dw)

    def _agent_pred_patient_purpose():
        a, p, o, pp = random.choice(agents), random.choice(predicates), random.choice(patients), random.choice(purposes)
        aw, pw, ow, ppw = _split(a), [p], _split(o), _split(pp)
        return aw + pw + ow + ppw, [0]*len(aw) + [1]*len(pw) + [2]*len(ow) + [11]*len(ppw)

    def _neg_agent_pred_patient():
        a, p, o = random.choice(agents), random.choice(predicates), random.choice(patients)
        aw, pw, ow = _split(a), [p], _split(o)
        return aw + ["did", "not"] + pw + ow, [0]*len(aw) + [14, 13] + [1]*len(pw) + [2]*len(ow)

    def _discourse_agent_pred():
        disc = random.choice(["However", "Therefore", "Meanwhile", "Furthermore", "Nevertheless", "Moreover"])
        a, p, o = random.choice(agents), random.choice(predicates), random.choice(patients)
        aw, pw, ow = _split(a), [p], _split(o)
        return [disc, ","] + aw + pw + ow, [12, 15] + [0]*len(aw) + [1]*len(pw) + [2]*len(ow)

    def _modal_agent_pred_patient():
        modal = random.choice(["could", "should", "would", "might", "must", "can"])
        a, p, o = random.choice(agents), random.choice(predicates), random.choice(patients)
        aw, pw, ow = _split(a), [p], _split(o)
        return aw + [modal] + pw + ow, [0]*len(aw) + [14] + [1]*len(pw) + [2]*len(ow)

    def _full_complex():
        disc = random.choice(["However", "Also", "Indeed"])
        a, p, o = random.choice(agents), random.choice(predicates), random.choice(patients)
        m, l, t = random.choice(manners), random.choice(locations), random.choice(times)
        aw, pw, ow = _split(a), [p], _split(o)
        mw, lw, tw = [m], _split(l), _split(t)
        return ([disc, ","] + aw + mw + pw + ow + lw + tw,
                [12, 15] + [0]*len(aw) + [5]*len(mw) + [1]*len(pw) + [2]*len(ow) + [3]*len(lw) + [4]*len(tw))

    def _adverbial_agent_pred():
        adv = random.choice([
            "Surprisingly", "Fortunately", "Apparently", "Obviously",
            "Remarkably", "Interestingly", "Importantly", "Significantly",
        ])
        a, p, o = random.choice(agents), random.choice(predicates), random.choice(patients)
        aw, pw, ow = _split(a), [p], _split(o)
        return [adv, ","] + aw + pw + ow, [10, 15] + [0]*len(aw) + [1]*len(pw) + [2]*len(ow)

    generators = [
        _agent_pred_patient,
        _agent_pred_patient_loc,
        _agent_pred_patient_time,
        _agent_manner_pred_patient,
        _agent_pred_patient_instr,
        _agent_pred_patient_cause,
        _agent_pred_patient_dir,
        _agent_pred_patient_purpose,
        _neg_agent_pred_patient,
        _discourse_agent_pred,
        _modal_agent_pred_patient,
        _full_complex,
        _adverbial_agent_pred,
    ]

    data = []
    for _ in range(n):
        gen = random.choice(generators)
        words, roles = gen()
        assert len(words) == len(roles), f"length mismatch: {words} vs {roles}"
        data.append((words, roles))

    # Report role coverage
    role_counts = [0] * NUM_ROLES
    for _, roles in data:
        for r in roles:
            role_counts[r] += 1
    covered = sum(1 for c in role_counts if c > 0)
    print(f"Generated {n} synthetic training examples ({covered}/{NUM_ROLES} roles covered).")
    return data


def precompute_hidden_states(sentences, cache_path=None, quick=False, model_name=MODEL_NAME):
    """Extract hidden states from a frozen LLM backbone.

    Caches results to disk for fast re-use.
    """
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached hidden states from {cache_path}...")
        cached = torch.load(cache_path, weights_only=True)
        return cached["hidden_states"], cached["role_labels"]

    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    actual_hidden = model.config.hidden_size
    print(f"  Hidden dim: {actual_hidden}")
    if actual_hidden != HIDDEN_DIM:
        print(f"  NOTE: Model hidden dim ({actual_hidden}) differs from default ({HIDDEN_DIM})")
        print(f"        Projection head will use {actual_hidden} as input dim.")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    hidden_states_list = []
    role_labels_list = []

    subset = sentences[:10000] if not quick else sentences[:200]
    print(f"Pre-computing hidden states for {len(subset)} sentences...")
    for words, roles in tqdm(subset, desc="Hidden states", unit="sent"):
        text = " ".join(words)
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Extract word_ids from BatchEncoding BEFORE converting to plain dict
        try:
            word_ids = encoding.word_ids(batch_index=0)  # list of Optional[int]
        except Exception:
            word_ids = None

        inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state[0].cpu()  # [seq_len, hidden_dim]

        # Align BPE tokens to word-level roles
        if word_ids is not None:
            aligned_roles = []
            for wid in word_ids:
                if wid is not None and wid < len(roles):
                    aligned_roles.append(roles[wid])
                else:
                    aligned_roles.append(NUM_ROLES - 1)  # NoRole for special tokens
        else:
            # Fallback: assign roles sequentially, pad with NoRole
            aligned_roles = list(roles) + [NUM_ROLES - 1] * (hidden.size(0) - len(roles))
            aligned_roles = aligned_roles[:hidden.size(0)]

        hidden_states_list.append(hidden)
        role_labels_list.append(torch.tensor(aligned_roles, dtype=torch.long))

    if cache_path:
        print(f"Caching hidden states to {cache_path}...")
        torch.save({
            "hidden_states": hidden_states_list,
            "role_labels": role_labels_list,
        }, cache_path)

    return hidden_states_list, role_labels_list


# ─── Training ───────────────────────────────────────────────────────

def train(args):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = getattr(args, "model_name", MODEL_NAME)

    print(f"\nModel: {model_name}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load data
    sentences = load_propbank_data(quick=args.quick)

    # Pre-compute hidden states
    cache_path = "hidden_states_cache.pt" if not args.quick else None
    hidden_states, role_labels = precompute_hidden_states(
        sentences, cache_path=cache_path, quick=args.quick, model_name=model_name,
    )

    # Create dataset and dataloader
    dataset = SRLDataset(hidden_states, role_labels)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Detect actual hidden dim from data
    actual_hidden_dim = hidden_states[0].size(-1)

    # Create model
    head = FrameProjectionHead(
        hidden_dim=actual_hidden_dim,
        mlp_dim=MLP_DIM,
        num_roles=NUM_ROLES,
        slot_dim=SLOT_DIM,
    ).to(device)

    # Reconstruction decoder (training-only, not exported to Rust)
    decoder = ReconstructionDecoder(
        slot_dim=SLOT_DIM,
        intermediate_dim=MLP_DIM,
        hidden_dim=actual_hidden_dim,
    ).to(device)

    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, weights_only=True)
        head.load_state_dict(checkpoint["model_state_dict"])
        if "decoder_state_dict" in checkpoint:
            decoder.load_state_dict(checkpoint["decoder_state_dict"])

    # Codebook for utilization tracking (optional)
    codebook = None
    if getattr(args, "codebook", None):
        codebook = load_codebook(args.codebook)
        print(f"Loaded codebook: {codebook.shape[0]} entries, dim={codebook.shape[1]}")

    # Optimizer (includes both head and decoder parameters)
    all_params = list(head.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    from tqdm import tqdm

    num_epochs = args.epochs if not args.quick else 2
    num_batches = len(dataloader)
    total_steps = num_epochs * num_batches
    best_loss = float("inf")

    head_params = sum(p.numel() for p in head.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    total_params = head_params + decoder_params
    print(f"\n{'='*60}")
    print(f"  Training Configuration")
    print(f"{'='*60}")
    print(f"  Device:           {device}")
    print(f"  Examples:         {len(dataset):,}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Batches/epoch:    {num_batches}")
    print(f"  Epochs:           {num_epochs}")
    print(f"  Total steps:      {total_steps:,}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  MLP:              {actual_hidden_dim} -> {MLP_DIM} -> {MLP_DIM} -> {MLP_DIM}")
    print(f"  Role Head:        {MLP_DIM} -> {NUM_ROLES}")
    print(f"  Embed Head:       {MLP_DIM} -> {SLOT_DIM}")
    print(f"  Recon Decoder:    {SLOT_DIM} -> {MLP_DIM} -> {actual_hidden_dim}")
    print(f"  Head parameters:  {head_params:,}")
    print(f"  Decoder params:   {decoder_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"{'='*60}\n")

    global_step = 0
    epoch_times = []
    training_start = time.time()

    for epoch in range(num_epochs):
        head.train()
        decoder.train()
        epoch_loss = 0.0
        epoch_role_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start = time.time()

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
            bar_format=(
                "{l_bar}{bar:30}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
            leave=True,
        )

        for batch_hidden, batch_labels, batch_mask in pbar:
            batch_hidden = batch_hidden.to(device)
            batch_labels = batch_labels.to(device)
            batch_mask = batch_mask.to(device)

            optimizer.zero_grad()

            role_logits, token_embeds = head(batch_hidden)

            # Role classification loss (cross-entropy, masked)
            role_logits_flat = role_logits[batch_mask]  # [num_valid_tokens, num_roles]
            labels_flat = batch_labels[batch_mask]  # [num_valid_tokens]
            role_loss = F.cross_entropy(role_logits_flat, labels_flat)

            # Embedding commitment loss: encourage embeddings to be unit-norm
            embeds_flat = token_embeds[batch_mask]  # [num_valid_tokens, slot_dim]
            embed_norms = embeds_flat.norm(dim=-1)
            commitment_loss = ((embed_norms - 1.0) ** 2).mean()

            # Reconstruction loss: decode embeddings back to hidden-state space
            reconstructed = decoder(token_embeds)  # [batch, seq_len, hidden_dim]
            recon_flat = reconstructed[batch_mask]  # [num_valid_tokens, hidden_dim]
            hidden_flat = batch_hidden[batch_mask]  # [num_valid_tokens, hidden_dim]
            reconstruction_loss = F.mse_loss(recon_flat, hidden_flat)

            loss = role_loss + 0.1 * commitment_loss + 0.1 * reconstruction_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_role_loss += role_loss.item()
            epoch_recon_loss += reconstruction_loss.item()
            global_step += 1

            # Accuracy
            preds = role_logits_flat.argmax(dim=-1)
            batch_correct = (preds == labels_flat).sum().item()
            batch_total = labels_flat.numel()
            epoch_correct += batch_correct
            epoch_total += batch_total

            # Update progress bar with live metrics
            running_loss = epoch_loss / (pbar.n + 1)
            running_acc = epoch_correct / max(epoch_total, 1)
            running_recon = epoch_recon_loss / (pbar.n + 1)
            pbar.set_postfix(
                loss=f"{running_loss:.4f}",
                recon=f"{running_recon:.4f}",
                acc=f"{running_acc:.1%}",
                step=f"{global_step}/{total_steps}",
            )

        pbar.close()

        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        avg_loss = epoch_loss / num_batches
        avg_role_loss = epoch_role_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        accuracy = epoch_correct / max(epoch_total, 1)

        # Time estimation
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        total_elapsed = time.time() - training_start

        def _fmt_time(s):
            if s >= 3600:
                return f"{s/3600:.1f}h"
            elif s >= 60:
                return f"{s/60:.1f}m"
            else:
                return f"{s:.1f}s"

        print(
            f"  -> loss={avg_loss:.4f}  role_loss={avg_role_loss:.4f}  "
            f"recon_loss={avg_recon_loss:.4f}  "
            f"acc={accuracy:.1%}  "
            f"epoch={_fmt_time(epoch_elapsed)}  "
            f"elapsed={_fmt_time(total_elapsed)}  "
            f"ETA={_fmt_time(eta_seconds)}"
        )

        if remaining_epochs > 0 and epoch == 0:
            est_total = avg_epoch_time * num_epochs
            print(
                f"\n  ** Estimated total training time: {_fmt_time(est_total)} "
                f"({avg_epoch_time:.1f}s/epoch x {num_epochs} epochs) **\n"
            )

        # Save best checkpoint
        # Run evaluation at end of epoch
        evaluate(head, dataloader, device, codebook=codebook)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": head.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
                "accuracy": accuracy,
            }, str(Path(args.output).parent / "best_checkpoint.pt"))

    total_training_time = time.time() - training_start
    print(f"\n  Total training time: {_fmt_time(total_training_time)}")

    # Export as safetensors
    export_safetensors(head, args.output)

    # BLEU evaluation (optional)
    if getattr(args, "evaluate_bleu", False):
        print("\nRunning BLEU-4 evaluation...")
        bleu = evaluate_bleu(head, decoder, dataloader, device)
        print(f"  Approximate round-trip BLEU-4: {bleu:.4f}")

    # Print evaluation summary
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Output: {args.output}")
    print(f"\nTo use in Rust:")
    print(f"  1. Download Qwen3-0.6B to a local directory")
    print(f"  2. Copy {args.output} alongside the model")
    print(f"  3. Configure LlmTranslatorConfig with the paths")
    print(f"  4. Run: cargo test -p volt-translate --features llm -- --ignored")


def export_safetensors(head, output_path):
    """Export model weights as safetensors with correct key naming.

    Key convention (must match Rust loader in projection.rs):
        proj.mlp.0.weight, proj.mlp.0.bias
        proj.mlp.2.weight, proj.mlp.2.bias  (nn.Sequential skips indices for GELU)
        proj.mlp.4.weight, proj.mlp.4.bias
        proj.role_head.weight, proj.role_head.bias
        proj.embed_head.weight, proj.embed_head.bias
    """
    from safetensors.torch import save_file

    state_dict = head.state_dict()

    # Remap nn.Sequential indices to match candle's VarBuilder prefix scheme:
    # PyTorch nn.Sequential: mlp.0 (Linear), mlp.1 (GELU), mlp.2 (Linear), mlp.3 (GELU), mlp.4 (Linear), mlp.5 (GELU)
    # Rust VarBuilder:       mlp.0, mlp.1, mlp.2 (three Linear layers)
    remap = {
        "mlp.0.weight": "proj.mlp.0.weight",
        "mlp.0.bias": "proj.mlp.0.bias",
        "mlp.2.weight": "proj.mlp.1.weight",
        "mlp.2.bias": "proj.mlp.1.bias",
        "mlp.4.weight": "proj.mlp.2.weight",
        "mlp.4.bias": "proj.mlp.2.bias",
        "role_head.weight": "proj.role_head.weight",
        "role_head.bias": "proj.role_head.bias",
        "embed_head.weight": "proj.embed_head.weight",
        "embed_head.bias": "proj.embed_head.bias",
    }

    tensors = {}
    for old_key, new_key in remap.items():
        if old_key in state_dict:
            tensors[new_key] = state_dict[old_key].contiguous().cpu()
        else:
            print(f"WARNING: expected key '{old_key}' not found in state_dict")

    save_file(tensors, output_path)
    print(f"Exported {len(tensors)} tensors to {output_path}")

    # Print weight shapes for verification
    for key, tensor in sorted(tensors.items()):
        print(f"  {key}: {list(tensor.shape)}")


# ─── Codebook ──────────────────────────────────────────────────────

def load_codebook(path):
    """Load Volt X codebook binary file.

    Format: 4-byte magic 'VXCB' + u32 version + u32 entry_count + u32 dim
            + entry_count * dim * 4 bytes of f32 LE data (row-major).

    Returns:
        torch.Tensor of shape [entry_count, dim].
    """
    import struct
    import numpy as np

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"VXCB":
            raise ValueError(f"Invalid codebook magic: {magic!r}, expected b'VXCB'")

        version = struct.unpack("<I", f.read(4))[0]
        entry_count = struct.unpack("<I", f.read(4))[0]
        dim = struct.unpack("<I", f.read(4))[0]

        if dim != SLOT_DIM:
            raise ValueError(f"Codebook dim {dim} != expected {SLOT_DIM}")

        data = np.frombuffer(f.read(entry_count * dim * 4), dtype=np.float32)
        data = data.reshape(entry_count, dim)

    return torch.from_numpy(data.copy())


def compute_codebook_utilization(head, codebook, dataloader, device):
    """Compute codebook utilization: fraction of unique entries used.

    Quantizes all predicted embeddings via nearest-neighbor cosine lookup
    against the codebook. Reports unique_ids / total_entries.

    Args:
        head: Trained FrameProjectionHead.
        codebook: [entry_count, dim] tensor of codebook vectors.
        dataloader: Validation data.
        device: Torch device.

    Returns:
        Utilization ratio in [0.0, 1.0].
    """
    head.eval()
    codebook_gpu = codebook.to(device)
    codebook_norm = F.normalize(codebook_gpu, dim=-1)
    used_ids = set()

    with torch.no_grad():
        for batch_hidden, batch_labels, batch_mask in dataloader:
            batch_hidden = batch_hidden.to(device)
            batch_mask = batch_mask.to(device)

            _, token_embeds = head(batch_hidden)
            embeds_flat = token_embeds[batch_mask]  # [N, slot_dim]
            if embeds_flat.numel() == 0:
                continue
            embeds_norm = F.normalize(embeds_flat, dim=-1)

            # Cosine similarity: [N, entry_count]
            sims = embeds_norm @ codebook_norm.T
            nearest_ids = sims.argmax(dim=-1)  # [N]
            used_ids.update(nearest_ids.cpu().tolist())

    total_entries = codebook.size(0)
    return len(used_ids) / total_entries


# ─── Evaluate ───────────────────────────────────────────────────────

def evaluate(head, dataloader, device, codebook=None):
    """Evaluate role classification accuracy per role, plus codebook utilization."""
    head.eval()
    per_role_correct = torch.zeros(NUM_ROLES, dtype=torch.long)
    per_role_total = torch.zeros(NUM_ROLES, dtype=torch.long)

    with torch.no_grad():
        for batch_hidden, batch_labels, batch_mask in dataloader:
            batch_hidden = batch_hidden.to(device)
            batch_labels = batch_labels.to(device)
            batch_mask = batch_mask.to(device)

            role_logits, _ = head(batch_hidden)
            preds = role_logits[batch_mask].argmax(dim=-1)
            labels = batch_labels[batch_mask]

            for r in range(NUM_ROLES):
                mask_r = labels == r
                per_role_correct[r] += (preds[mask_r] == r).sum()
                per_role_total[r] += mask_r.sum()

    role_names = [
        "Agent", "Predicate", "Patient", "Location", "Time",
        "Manner", "Instrument", "Cause", "Result",
        "Free(0)", "Free(1)", "Free(2)", "Free(3)", "Free(4)", "Free(5)", "Free(6)",
    ]

    print("\nPer-role accuracy:")
    for r in range(NUM_ROLES):
        total = per_role_total[r].item()
        correct = per_role_correct[r].item()
        acc = correct / max(total, 1)
        print(f"  {role_names[r]:12s}: {correct:5d}/{total:5d} = {acc:.3f}")

    total_correct = per_role_correct.sum().item()
    total_total = per_role_total.sum().item()
    overall_acc = total_correct / max(total_total, 1)
    print(f"  {'Overall':12s}: {total_correct:5d}/{total_total:5d} = {overall_acc:.3f}")

    # Codebook utilization (if codebook provided)
    if codebook is not None:
        utilization = compute_codebook_utilization(head, codebook, dataloader, device)
        used = int(utilization * codebook.size(0))
        total = codebook.size(0)
        print(f"\n  Codebook utilization: {utilization:.1%} ({used}/{total} entries used)")
        return overall_acc, utilization

    return overall_acc


# ─── BLEU ──────────────────────────────────────────────────────────

def compute_bleu4(references, hypotheses):
    """Compute corpus-level BLEU-4 score.

    Self-contained implementation (no external dependencies).

    Args:
        references: List of reference token lists (list[list[str]]).
        hypotheses: List of hypothesis token lists (list[list[str]]).

    Returns:
        BLEU-4 score in [0.0, 1.0].
    """
    from collections import Counter
    import math

    if not references or not hypotheses:
        return 0.0

    clipped_counts = [0] * 4
    total_counts = [0] * 4
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)

        for n in range(1, 5):
            ref_ngrams = Counter(
                tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)
            )
            hyp_ngrams = Counter(
                tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1)
            )
            for ngram, count in hyp_ngrams.items():
                clipped_counts[n - 1] += min(count, ref_ngrams.get(ngram, 0))
            total_counts[n - 1] += sum(hyp_ngrams.values())

    # Modified precisions
    precisions = []
    for n in range(4):
        if total_counts[n] == 0:
            return 0.0
        precisions.append(clipped_counts[n] / total_counts[n])

    # Geometric mean of log precisions
    log_avg = sum(
        math.log(p) if p > 0 else float("-inf") for p in precisions
    ) / 4.0
    if log_avg == float("-inf"):
        return 0.0

    # Brevity penalty
    if hyp_len == 0:
        return 0.0
    bp = math.exp(1 - ref_len / hyp_len) if hyp_len < ref_len else 1.0

    return bp * math.exp(log_avg)


def evaluate_bleu(head, decoder, dataloader, device, max_samples=500):
    """Evaluate approximate round-trip BLEU-4 score.

    Pipeline: hidden_states -> head -> token_embeds -> decoder ->
    reconstructed hidden_states -> cosine-nearest original token.

    Since we don't have access to the LLM embedding matrix at training
    time, we approximate by comparing the reconstructed hidden states
    token-by-token to the original hidden states. Each "token" is
    represented by its nearest neighbor in the input batch.

    For true BLEU this would require a full text decoder; this
    approximation measures reconstruction fidelity at the hidden-state
    level and converts it to a token-matching score.

    Args:
        head: Trained FrameProjectionHead.
        decoder: Trained ReconstructionDecoder.
        dataloader: Validation data.
        device: Torch device.
        max_samples: Maximum number of sentences to evaluate.

    Returns:
        Approximate BLEU-4 score.
    """
    head.eval()
    decoder.eval()

    references = []
    hypotheses = []
    samples_seen = 0

    with torch.no_grad():
        for batch_hidden, batch_labels, batch_mask in dataloader:
            if samples_seen >= max_samples:
                break

            batch_hidden = batch_hidden.to(device)
            batch_mask = batch_mask.to(device)

            _, token_embeds = head(batch_hidden)
            reconstructed = decoder(token_embeds)  # [batch, seq_len, hidden_dim]

            batch_size = batch_hidden.size(0)
            for i in range(batch_size):
                if samples_seen >= max_samples:
                    break

                mask_i = batch_mask[i]  # [seq_len]
                if mask_i.sum() == 0:
                    continue

                orig = batch_hidden[i][mask_i]  # [valid_len, hidden_dim]
                recon = reconstructed[i][mask_i]  # [valid_len, hidden_dim]

                # Normalize for cosine comparison
                orig_norm = F.normalize(orig, dim=-1)
                recon_norm = F.normalize(recon, dim=-1)

                # For each reconstructed token, find its nearest original token
                sims = recon_norm @ orig_norm.T  # [valid_len, valid_len]
                nearest = sims.argmax(dim=-1)  # [valid_len]

                # Reference: original token positions 0, 1, 2, ...
                ref_tokens = [str(j) for j in range(orig.size(0))]
                # Hypothesis: the nearest-original-token indices
                hyp_tokens = [str(idx.item()) for idx in nearest]

                references.append(ref_tokens)
                hypotheses.append(hyp_tokens)
                samples_seen += 1

    if not references:
        return 0.0

    bleu = compute_bleu4(references, hypotheses)
    return bleu


# ─── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train the Frame Projection Head for Volt X Milestone 2.2"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"HuggingFace model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for safetensors + checkpoints (default: current dir)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="projection.safetensors",
        help="Output safetensors filename (default: projection.safetensors)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a PyTorch checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run: tiny dataset, 2 epochs",
    )
    parser.add_argument(
        "--codebook",
        type=str,
        default=None,
        help="Path to codebook.bin for utilization tracking during evaluation",
    )
    parser.add_argument(
        "--evaluate-bleu",
        action="store_true",
        help="Run BLEU-4 evaluation after training (approximate round-trip)",
    )

    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / args.output)

    train(args)


if __name__ == "__main__":
    main()
