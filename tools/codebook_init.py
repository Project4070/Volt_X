#!/usr/bin/env python3
"""
Codebook initializer for Volt X Milestone 2.1.

Downloads pretrained word embeddings (GloVe 6B 300d), projects them to 256
dimensions via PCA, then clusters into 65,536 centroids using Mini-Batch
K-Means. The resulting codebook is saved in the binary format expected by
volt-bus's `Codebook::load`.

Binary format:
    [4 bytes: magic "VXCB"]
    [4 bytes: version u32 LE = 1]
    [4 bytes: entry_count u32 LE]
    [4 bytes: dim u32 LE = 256]
    [entry_count * 256 * 4 bytes: f32 LE data, row-major]

Requirements:
    pip install numpy scikit-learn

Usage:
    # Download GloVe, PCA to 256d, K-Means with K=65536
    python tools/codebook_init.py --embeddings glove.6B.300d.txt --output codebook.bin

    # Use fewer clusters for faster testing
    python tools/codebook_init.py --embeddings glove.6B.300d.txt --output codebook.bin -k 4096

    # Skip PCA if embeddings are already 256d
    python tools/codebook_init.py --embeddings my_256d.txt --output codebook.bin --no-pca
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np


MAGIC = b"VXCB"
FORMAT_VERSION = 1
TARGET_DIM = 256
DEFAULT_K = 65536


def load_embeddings(path: str, max_vocab: int | None = None) -> np.ndarray:
    """Load word embeddings from a text file (GloVe/FastText format).

    Each line: word dim1 dim2 ... dimN
    Returns a float32 numpy array of shape (vocab_size, embed_dim).
    """
    print(f"Loading embeddings from {path}...")
    vectors = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if max_vocab is not None and i >= max_vocab:
                break
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue  # skip header lines (FastText format)
            try:
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                vectors.append(vec)
            except ValueError:
                continue  # skip malformed lines

    data = np.stack(vectors, axis=0)
    print(f"  Loaded {data.shape[0]} vectors of dimension {data.shape[1]}")
    return data


def pca_reduce(data: np.ndarray, target_dim: int) -> np.ndarray:
    """Reduce dimensionality via PCA."""
    from sklearn.decomposition import PCA

    if data.shape[1] == target_dim:
        print(f"  Embeddings already {target_dim}d, skipping PCA")
        return data

    print(f"  PCA: {data.shape[1]}d -> {target_dim}d...")
    t0 = time.time()
    pca = PCA(n_components=target_dim, random_state=42)
    reduced = pca.fit_transform(data).astype(np.float32)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA done in {time.time() - t0:.1f}s (explained variance: {explained:.3f})")
    return reduced


def l2_normalize(data: np.ndarray) -> np.ndarray:
    """L2-normalize each row to unit length."""
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    return (data / norms).astype(np.float32)


def _progress_bar(current: int, total: int, width: int = 40, prefix: str = "", suffix: str = "") -> None:
    """Print an in-place progress bar to stderr."""
    frac = current / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r  {prefix}|{bar}| {current}/{total} {suffix}", end="", flush=True)


def kmeans_cluster(data: np.ndarray, k: int, batch_size: int = 4096, max_iter: int = 300) -> np.ndarray:
    """Run Mini-Batch K-Means with progress bar and return centroids."""
    from sklearn.cluster import MiniBatchKMeans

    print(f"  Mini-Batch K-Means: {data.shape[0]} vectors -> {k} clusters...")
    print(f"  (batch_size={batch_size}, max_iter={max_iter}, init=random)")
    t0 = time.time()

    # Use random init — k-means++ is O(n*k) and infeasible for k=65536.
    # Random init is standard practice for large k and converges fine
    # with enough iterations.
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=42,
        max_iter=max_iter,
        n_init=1,
        init="random",
        verbose=0,
    )

    # Manual partial_fit loop with progress bar
    n = data.shape[0]
    rng = np.random.RandomState(42)
    batches_per_epoch = max(1, n // batch_size)
    prev_inertia = float("inf")

    # First partial_fit (needs >= k samples)
    print(f"  Initializing {k} random centroids...", end="", flush=True)
    init_size = max(k, batch_size)
    indices = rng.choice(n, size=init_size, replace=False)
    kmeans.partial_fit(data[indices])
    print(f" done ({time.time() - t0:.1f}s)")

    for epoch in range(max_iter):
        perm = rng.permutation(n)
        for b in range(batches_per_epoch):
            start = b * batch_size
            end = min(start + batch_size, n)
            kmeans.partial_fit(data[perm[start:end]])

            # Update progress bar every 10 batches
            if b % 10 == 0:
                done_batches = epoch * batches_per_epoch + b + 1
                total_batches = max_iter * batches_per_epoch
                elapsed = time.time() - t0
                rate = done_batches / elapsed if elapsed > 0 else 0
                eta = (total_batches - done_batches) / rate if rate > 0 else 0
                _progress_bar(
                    epoch + 1, max_iter,
                    prefix=f"Training ",
                    suffix=f"| {elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                )

        # Check convergence every 10 epochs
        if (epoch + 1) % 10 == 0:
            inertia = kmeans.inertia_
            delta = ((prev_inertia - inertia) / prev_inertia * 100) if prev_inertia != float("inf") else 0
            if 0 < delta < 0.01 and epoch > 50:
                print(f"\n  Converged at epoch {epoch+1} (inertia change < 0.01%)")
                break
            prev_inertia = inertia

    elapsed = time.time() - t0
    print(f"\n  K-Means done in {elapsed:.1f}s (final inertia: {kmeans.inertia_:.2f})")
    return kmeans.cluster_centers_.astype(np.float32)


def save_codebook(centroids: np.ndarray, path: str) -> None:
    """Save codebook in Volt X binary format."""
    entry_count, dim = centroids.shape
    assert dim == TARGET_DIM, f"Expected dim={TARGET_DIM}, got {dim}"

    with open(path, "wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("<I", FORMAT_VERSION))
        f.write(struct.pack("<I", entry_count))
        f.write(struct.pack("<I", dim))
        # Data (row-major, f32 little-endian)
        f.write(centroids.astype("<f4").tobytes())

    size_mb = Path(path).stat().st_size / (1024 * 1024)
    print(f"  Saved codebook: {entry_count} entries, {dim}d, {size_mb:.1f} MB -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize Volt X VQ-VAE codebook from pretrained embeddings"
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Path to pretrained embeddings (GloVe/FastText text format)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the codebook binary file",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of codebook entries (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=None,
        help="Max number of embedding vectors to load (default: all)",
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Skip PCA (use if embeddings are already 256d)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Mini-Batch K-Means batch size (default: 4096)",
    )
    args = parser.parse_args()

    if args.k > 65536:
        print(f"Error: k={args.k} exceeds maximum codebook capacity of 65536", file=sys.stderr)
        sys.exit(1)

    # Step 1: Load embeddings
    data = load_embeddings(args.embeddings, max_vocab=args.max_vocab)

    if data.shape[0] < args.k:
        print(
            f"Error: only {data.shape[0]} vectors loaded, need at least k={args.k}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 2: PCA to 256d (if needed)
    if not args.no_pca:
        data = pca_reduce(data, TARGET_DIM)
    elif data.shape[1] != TARGET_DIM:
        print(
            f"Error: --no-pca but embeddings are {data.shape[1]}d, expected {TARGET_DIM}d",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 3: L2-normalize
    print("  L2-normalizing vectors...")
    data = l2_normalize(data)

    # Step 4: K-Means clustering
    centroids = kmeans_cluster(data, args.k, batch_size=args.batch_size)

    # Step 5: L2-normalize centroids (match Codebook::from_entries behavior)
    centroids = l2_normalize(centroids)

    # Step 6: Save
    save_codebook(centroids, args.output)

    # Summary
    print("\nDone! To use in Volt X:")
    print(f'  let cb = Codebook::load(Path::new("{args.output}")).unwrap();')


if __name__ == "__main__":
    main()
