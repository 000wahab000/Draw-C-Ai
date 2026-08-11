"""
test_knn_embedding.py
=====================
Compares raw-pixel KNN vs ResNet-embedding KNN on the augmented dataset.
Uses leave-one-out cross validation: for each sample, train on all others,
predict on that one. Reports per-class and overall accuracy for both methods.

Run from project root:
    .venv/Scripts/python.exe KNN/test_knn_embedding.py
"""

import os
import glob
import json
import sys
import tempfile
import numpy as np
from PIL import Image

# Make KNN module importable from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from KNN.knn_pixel import load_dataset, predict
from KNN.knn_embedding import get_embedding

DATASET_DIR = os.path.join(os.path.dirname(__file__), "data")


def vector_to_pil(vector: list) -> Image.Image:
    """Convert a 1024-element 0/1 pixel vector back to a 32x32 RGB PIL image."""
    arr = np.array(vector, dtype=np.float32).reshape(32, 32)
    # 0 = background (white), 1 = drawn line (black)
    img_array = ((1.0 - arr) * 255).astype(np.uint8)
    return Image.fromarray(img_array, mode="L").convert("RGB")


def leave_one_out(X, y, use_embeddings=False, k=5):
    """
    For every sample i: train on all others, predict on sample i.
    Returns list of (true_label, predicted_label) tuples.
    """
    results = []
    n = len(y)

    for i in range(n):
        # Split: everything except i
        X_train = np.concatenate([X[:i], X[i+1:]], axis=0)
        y_train = y[:i] + y[i+1:]
        query = X[i]

        # predict() returns [{label, score, distance}, ...] sorted by distance
        top = predict(query, X_train, y_train, k=k)
        # Majority vote from top-k
        votes = {}
        for item in top:
            votes[item["label"]] = votes.get(item["label"], 0) + 1
        predicted = max(votes, key=votes.get)

        results.append((y[i], predicted))

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] done", flush=True)

    return results


def accuracy_report(results, label=""):
    correct = sum(1 for t, p in results if t == p)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Overall accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"{'='*50}")

    # Per-class breakdown
    classes = sorted(set(t for t, _ in results))
    print(f"  {'Class':<20} {'Correct':<10} {'Total':<10} {'Acc'}")
    print(f"  {'-'*50}")
    for c in classes:
        class_results = [(t, p) for t, p in results if t == c]
        c_correct = sum(1 for t, p in class_results if t == p)
        c_total = len(class_results)
        print(f"  {c:<20} {c_correct:<10} {c_total:<10} {c_correct/c_total*100:.0f}%")


def main():
    print("Loading pixel dataset...")
    X_pixel, y = load_dataset(DATASET_DIR)
    print(f"  Loaded {len(y)} samples, {X_pixel.shape[1]}-dim pixel vectors")

    # ── Pixel KNN ────────────────────────────────────────────────────────────
    print("\nRunning leave-one-out with PIXEL vectors (k=5)...")
    pixel_results = leave_one_out(X_pixel, list(y), k=5)
    accuracy_report(pixel_results, "PIXEL KNN (1024-dim)")

    # ── Embedding KNN ─────────────────────────────────────────────────────────
    print("\nBuilding ResNet-18 embeddings from pixel vectors...")
    print("  (Converting each JSON vector -> PIL image -> ResNet embedding)")
    embeddings = []
    for i, vec in enumerate(X_pixel):
        img = vector_to_pil(vec.tolist())
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            img.save(tmp_path)
            emb = get_embedding(tmp_path)
            embeddings.append(emb)
        finally:
            os.unlink(tmp_path)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(X_pixel)}] embeddings built", flush=True)

    X_emb = np.array(embeddings, dtype=np.float32)
    print(f"  Done. Embedding shape: {X_emb.shape}")

    print("\nRunning leave-one-out with EMBEDDING vectors (k=5)...")
    emb_results = leave_one_out(X_emb, list(y), k=5)
    accuracy_report(emb_results, "EMBEDDING KNN (512-dim, ResNet-18)")

    # ── Side-by-side summary ─────────────────────────────────────────────────
    pixel_acc = sum(1 for t, p in pixel_results if t == p) / len(pixel_results) * 100
    emb_acc   = sum(1 for t, p in emb_results   if t == p) / len(emb_results)   * 100
    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"  Pixel KNN:     {pixel_acc:.1f}%")
    print(f"  Embedding KNN: {emb_acc:.1f}%")
    delta = emb_acc - pixel_acc
    sign = "+" if delta >= 0 else ""
    print(f"  Delta:         {sign}{delta:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
