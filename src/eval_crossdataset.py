"""
Per-dataset evaluation, to expose dataset shortcuts / generalisation gaps.

Pooled metrics hide the case where a detector is ~perfect on one dataset
(e.g. CIFAKE) and near-chance on another (e.g. an unseen generator family).
This script scores each dataset *separately*:

  - clean accuracy (threshold 0.5, or --threshold)
  - clean AUC (threshold-independent)
  - mean AUC across the full transform grid (robustness)

Typical shortcut probe: train two checkpoints on different source mixes, then
run this on all datasets and compare.

    # trained on CIFAKE + SID only:
    python -m src.train --data_dir data/CIFAKE/train data/sid_subset --out checkpoints/cnn_no_wildfake.pt
    python -m src.eval_crossdataset --checkpoint checkpoints/cnn_no_wildfake.pt \
        --dataset CIFAKE=data/CIFAKE/test SID=data/sid_test_holdout WildFake=data/wildfake_test_holdout

A large clean-AUC drop on the held-out dataset == the model leaned on a
dataset-specific cue, not a generator artifact.
"""

import argparse
import csv
from pathlib import Path

from PIL import Image
from sklearn.metrics import roc_auc_score

from src.model import load_model
from src.robustness_test import TRANSFORMS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def score_dataset(model, root: Path, threshold: float):
    y_true, clean_score = [], []
    transf_true = {k: [] for k in TRANSFORMS}
    transf_score = {k: [] for k in TRANSFORMS}

    for cls, y in [("REAL", 0), ("FAKE", 1)]:
        cls_dir = root / cls
        if not cls_dir.exists():
            print(f"WARNING: {cls_dir} not found, skipping")
            continue
        for p in cls_dir.iterdir():
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    for name, fn in TRANSFORMS.items():
                        s = model.predict(fn(im.copy()))
                        transf_true[name].append(y)
                        transf_score[name].append(s)
                        if name == "clean":
                            y_true.append(y)
                            clean_score.append(s)
            except Exception as e:
                print(f"  skip {p}: {e}")

    n = len(y_true)
    if n == 0 or len(set(y_true)) < 2:
        return None
    clean_acc = sum(int((s >= threshold) == t) for s, t in zip(clean_score, y_true)) / n
    clean_auc = roc_auc_score(y_true, clean_score)
    aucs = [roc_auc_score(transf_true[k], transf_score[k])
            for k in TRANSFORMS if len(set(transf_true[k])) == 2]
    mean_transf_auc = sum(aucs) / len(aucs)
    return {"n": n, "clean_acc": clean_acc, "clean_auc": clean_auc,
            "mean_transformed_auc": mean_transf_auc}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", nargs="+", required=True, metavar="NAME=PATH",
                    help="one or more NAME=path/to/ImageFolder entries")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--calibration", default=None, help="src/calibrate.py JSON")
    ap.add_argument("--out_csv", default="outputs/crossdataset.csv")
    args = ap.parse_args()

    model = load_model(args.checkpoint, calibration=args.calibration)

    rows = []
    for entry in args.dataset:
        if "=" not in entry:
            raise SystemExit(f"--dataset entries must be NAME=path, got {entry!r}")
        name, path = entry.split("=", 1)
        print(f"\n=== {name} ({path}) ===")
        r = score_dataset(model, Path(path), args.threshold)
        if r is None:
            print("  (skipped — need both REAL and FAKE images)")
            continue
        print(f"  n={r['n']}  clean_acc={r['clean_acc']:.4f}  clean_auc={r['clean_auc']:.4f}  "
              f"mean_transformed_auc={r['mean_transformed_auc']:.4f}")
        rows.append((name, r))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n_images", "clean_accuracy", "clean_auc", "mean_transformed_auc"])
        for name, r in rows:
            w.writerow([name, r["n"], f"{r['clean_acc']:.4f}", f"{r['clean_auc']:.4f}",
                        f"{r['mean_transformed_auc']:.4f}"])

    if len(rows) > 1:
        spread = max(r["clean_auc"] for _, r in rows) - min(r["clean_auc"] for _, r in rows)
        print(f"\nClean-AUC spread across datasets: {spread:.4f}  "
              f"({'likely dataset shortcut' if spread > 0.10 else 'reasonably consistent'})")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
