import csv
from pathlib import Path
import argparse

from PIL import Image

from src.model import load_model
from src.robustness_test import TRANSFORMS

def evaluate_robustness(test_dir, checkpoint_path, out_csv, max_per_class=None):
    model = load_model(checkpoint_path)
    test_dir = Path(test_dir)
    results = {name: {"correct": 0, "total": 0} for name in TRANSFORMS}
    for cls, true_label in [("REAL", 0.0), ("FAKE", 1.0)]:
        images = sorted((test_dir / cls).glob("*"))
        if max_per_class:
            images = images[:max_per_class]
        for img_path in images:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                for name, fn in TRANSFORMS.items():
                    transformed = fn(img.copy())
                    pred = model.predict(transformed)
                    predicted_label = 1.0 if pred > 0.5 else 0.0
                    results[name]["total"] += 1
                    if predicted_label == true_label:
                        results[name]["correct"] += 1

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["transform", "accuracy", "n_images"])
        for name, stats in results.items():
            acc = stats["correct"] / stats["total"] if stats["total"] else 0
            writer.writerow([name, f"{acc:.4f}", stats["total"]])

    print(f"Saved robustness table to {out_csv}")
    for name, stats in results.items():
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f" {name:18s} acc={acc:.4f} (n={stats['total']})")

def parse_args():
    parser = argparse.ArgumentParser(description="Robustness evaluation")
    parser.add_argument("--test_dir", default="data/cifake/test")
    parser.add_argument("--checkpoint", default="checkpoints/cnn_cifake.pt")
    parser.add_argument("--out_csv", default="outputs/robustness_table.csv")
    parser.add_argument("--max_per_class", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_robustness(args.test_dir, args.checkpoint, args.out_csv, args.max_per_class)