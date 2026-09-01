"""
Per-transform ACCURACY evaluation (threshold the sigmoid output at 0.5).

Companion to `src/eval_auc.py`, which reports threshold-independent AUC for
the same transform set. This script writes `transform,accuracy,n_images`,
the format `src/robustness_summary.py` consumes, so several runs (e.g.
augmented vs non-augmented baseline) can be charted side by side:

    python -m src.eval_robustness --checkpoint checkpoints/cnn_baseline.pt  --test_dir <dirs> --out_csv outputs/rob_baseline.csv
    python -m src.eval_robustness --checkpoint checkpoints/cnn_augmented.pt --test_dir <dirs> --out_csv outputs/rob_augmented.csv
    python -m src.robustness_summary --csv outputs/rob_baseline.csv:Baseline outputs/rob_augmented.csv:Augmented
"""

import argparse
import csv
from pathlib import Path

from PIL import Image

from src.model import load_model
from src.robustness_test import TRANSFORMS


def evaluate_accuracy(test_dirs, checkpoint_path, out_csv, threshold: float = 0.5, calibration=None):
    model = load_model(checkpoint_path, calibration=calibration)
    results = {name: {"correct": 0, "total": 0} for name in TRANSFORMS}

    for test_dir in test_dirs:
        test_dir = Path(test_dir)
        for cls, true_label in [("REAL", 0), ("FAKE", 1)]:
            cls_dir = test_dir / cls
            if not cls_dir.exists():
                print(f"WARNING: {cls_dir} not found, skipping")
                continue
            for img_path in cls_dir.glob("*"):
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    for name, fn in TRANSFORMS.items():
                        pred = model.predict(fn(img.copy()))  # P(FAKE)
                        pred_label = 1 if pred >= threshold else 0
                        results[name]["correct"] += int(pred_label == true_label)
                        results[name]["total"] += 1

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["transform", "accuracy", "n_images"])
        for name, d in results.items():
            if d["total"] == 0:
                print(f"WARNING: {name} had no images, skipping")
                continue
            acc = d["correct"] / d["total"]
            writer.writerow([name, f"{acc:.4f}", d["total"]])
            print(f"{name:18s} acc={acc:.4f}  (n={d['total']})")

    print(f"Saved accuracy table to {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Per-transform accuracy evaluation")
    parser.add_argument("--test_dir", nargs="+", default=["data/cifake/test"])
    parser.add_argument("--checkpoint", default="checkpoints/cnn_merged.pt")
    parser.add_argument("--out_csv", default="outputs/robustness_table.csv")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="decision threshold on P(FAKE)")
    parser.add_argument("--calibration", default=None,
                         help="src/calibrate.py JSON (temperature scaling)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_accuracy(args.test_dir, args.checkpoint, args.out_csv, args.threshold, args.calibration)
