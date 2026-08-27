"""
Robustness evaluation harness.

Evaluates a trained checkpoint on:
  1. The clean test set
  2. Each transform folder produced by build_test_transforms.py

Outputs a single CSV table: one row per condition (clean + each transform),
with accuracy / precision / recall / F1 — this is the "Robustness Evaluation
Summary" deliverable (Section 5.5.4 of the problem statement).

Usage:
    python evaluate.py --checkpoint ../checkpoints/resnet50_augmented.pt \
                        --data_dir ../data --transformed_dir ../data/test_transformed \
                        --out ../outputs/robustness_table.csv
"""

import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import AIGCDataset, default_transform
from model import load_checkpoint


@torch.no_grad()
def run_eval(model, dataset, device, batch_size=32, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(FAKE)
        preds = logits.argmax(dim=1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    return all_labels, all_preds, all_probs


def compute_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "n_samples": len(labels),
    }


def evaluate_all_conditions(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, backbone=args.backbone, device=device)
    transform = default_transform(args.image_size)

    results = []
    predictions_dump = {}  # condition -> list of (path, label, pred, prob), for error analysis

    # --- 1. Clean test set ---
    print("Evaluating: clean")
    clean_ds = AIGCDataset(root_dir=args.data_dir, split="test",
                            use_augmentation=False, transform=transform)
    labels, preds, probs = run_eval(model, clean_ds, device, args.batch_size, args.num_workers)
    metrics = compute_metrics(labels, preds)
    metrics["condition"] = "clean"
    results.append(metrics)
    predictions_dump["clean"] = list(zip(
        [p for p, _ in clean_ds.samples], labels, preds, probs
    ))
    print(f"  acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} (n={metrics['n_samples']})")

    # --- 2. Each transform condition ---
    if os.path.isdir(args.transformed_dir):
        transform_names = sorted(os.listdir(args.transformed_dir))
        for t_name in transform_names:
            t_root = args.transformed_dir  # AIGCDataset expects root/split/{REAL,FAKE}
            t_split = t_name  # each transform is its own "split" subfolder
            print(f"Evaluating: {t_name}")
            try:
                t_ds = AIGCDataset(root_dir=t_root, split=t_split,
                                    use_augmentation=False, transform=transform)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"  SKIPPED ({e})")
                continue

            labels, preds, probs = run_eval(model, t_ds, device, args.batch_size, args.num_workers)
            metrics = compute_metrics(labels, preds)
            metrics["condition"] = t_name
            results.append(metrics)
            predictions_dump[t_name] = list(zip(
                [p for p, _ in t_ds.samples], labels, preds, probs
            ))
            print(f"  acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} (n={metrics['n_samples']})")
    else:
        print(f"WARNING: transformed_dir {args.transformed_dir} not found — "
              f"run build_test_transforms.py first. Reporting clean-only results.")

    # --- Save robustness table ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_csv(results, args.out)
    print(f"\nSaved robustness table to {args.out}")

    # --- Save raw predictions for error analysis (gradcam.py consumes this) ---
    preds_path = args.out.replace(".csv", "_predictions.json")
    with open(preds_path, "w") as f:
        json.dump(predictions_dump, f, indent=2)
    print(f"Saved raw predictions to {preds_path}")

    return results


def write_csv(results, out_path):
    fieldnames = ["condition", "accuracy", "precision", "recall", "f1", "n_samples"]
    lines = [",".join(fieldnames)]
    for r in results:
        lines.append(",".join(str(r[k]) for k in fieldnames))
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--transformed_dir", default="../data/test_transformed")
    parser.add_argument("--out", default="../outputs/robustness_table.csv")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    evaluate_all_conditions(args)
