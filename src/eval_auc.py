import argparse
import csv
from pathlib import Path

from PIL import Image
from sklearn.metrics import roc_auc_score

from src.model import load_model
from src.robustness_test import TRANSFORMS


def evaluate_auc(test_dirs, checkpoint_path, out_csv):
    model = load_model(checkpoint_path)
    results = {name: {"y_true": [], "y_score": []} for name in TRANSFORMS}

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
                        pred = model.predict(fn(img.copy()))
                        results[name]["y_true"].append(true_label)
                        results[name]["y_score"].append(pred)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["transform", "auc", "n_images"])
        for name, data in results.items():
            if len(set(data["y_true"])) < 2:
                print(f"WARNING: {name} has only one class present, skipping AUC")
                continue
            auc = roc_auc_score(data["y_true"], data["y_score"])
            writer.writerow([name, f"{auc:.4f}", len(data["y_true"])])
            print(f"{name:18s} AUC={auc:.4f}  (n={len(data['y_true'])})")

    print(f"Saved AUC table to {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="AUC evaluation across transforms")
    parser.add_argument("--test_dir", nargs="+", default=["data/cifake/test"])
    parser.add_argument("--checkpoint", default="checkpoints/cnn_merged.pt")
    parser.add_argument("--out_csv", default="outputs/auc_table.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_auc(args.test_dir, args.checkpoint, args.out_csv)