import os
import sys
import glob
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.classifiers.knn_pixel import load_dataset, predict

def get_latest_dataset():
    datasets = glob.glob(os.path.join(PROJECT_ROOT, "dataset_v*"))
    if not datasets:
        return None
    return sorted(datasets)[-1]

if __name__ == "__main__":
    # Load training data
    dataset_dir = get_latest_dataset()
    if not dataset_dir:
        print("No dataset found.")
        sys.exit(1)

    print(f"Loading training data from {dataset_dir}...")
    X, y = load_dataset(dataset_dir)
    print(f"Loaded {len(y)} training samples.")

    # Load test set (sacred, never trained on)
    test_dir = os.path.join(PROJECT_ROOT, "test_set")
    test_files = glob.glob(os.path.join(test_dir, "*.json"))
    if not test_files:
        print("No test files found in test_set/!")
        sys.exit(1)

    print(f"Evaluating on {len(test_files)} test samples...\n")

    correct = 0
    total = 0
    failures = []  # (true_label, predicted_label, distance_to_wrong, distance_to_correct)

    for filepath in sorted(test_files):
        with open(filepath) as f:
            data = json.load(f)
        true_label = data["label"]
        query_vector = data["vector"]

        top_k = predict(query_vector, X, y, k=5)
        predicted = top_k[0]["label"]
        total += 1

        if predicted == true_label:
            correct += 1
            status = "OK"
        else:
            status = "WRONG"
            # Find nearest correct neighbor in top_k
            nearest_correct = next((r for r in top_k if r["label"] == true_label), None)
            failures.append({
                "file": os.path.basename(filepath),
                "true": true_label,
                "predicted": predicted,
                "dist_to_wrong": top_k[0]["distance"],
                "dist_to_correct": nearest_correct["distance"] if nearest_correct else "not in top-5",
            })

        print(f"  [{status}] {os.path.basename(filepath)}: true={true_label}, pred={predicted} (dist={top_k[0]['distance']})")

    acc = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} = {acc:.1f}%")

    # Write failure log
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "phase2_failures.md")

    with open(log_path, "w") as f:
        f.write("# Phase 2 — Pixel KNN Failure Analysis\n\n")
        f.write(f"**Dataset**: {os.path.basename(dataset_dir)} ({len(y)} training samples)\n")
        f.write(f"**Test set**: {total} samples (2 per class)\n")
        f.write(f"**Accuracy**: {correct}/{total} = {acc:.1f}%\n\n")

        if failures:
            f.write("## Misclassifications\n\n")
            f.write("| File | True Label | Predicted | Dist to Wrong | Dist to Correct |\n")
            f.write("|---|---|---|---|---|\n")
            for fail in failures:
                f.write(f"| {fail['file']} | {fail['true']} | {fail['predicted']} | {fail['dist_to_wrong']} | {fail['dist_to_correct']} |\n")
        else:
            f.write("## No misclassifications\n\n")
            f.write("All test samples correctly classified. This is expected with only 88 total samples ")
            f.write("where augmented variants are very close to the originals.\n")

        f.write("\n## Notes\n\n")
        f.write("With only 8 raw icons and 10x augmentation, the augmented samples are highly correlated. ")
        f.write("High accuracy here does NOT mean the model generalizes to real hand-drawn input. ")
        f.write("Revisit after growing the dataset (Phase 1.3: generate_stylized.py).\n")

    print(f"Failure log written to {log_path}")
