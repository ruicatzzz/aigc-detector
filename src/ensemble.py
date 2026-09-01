"""
Average several trained checkpoints into one detector.

Members that look at *different signal* fail on different inputs, so the
average is flatter across the transform grid than any single model (brief
slide 1: "combine high-level semantics + low-level frequency patches").
Good pairing: a spatial model (`small_cnn` / `efficientnet_b0`) + the
spectrum model (`freq_cnn`).

Inference (JSON out, same format as src/infer.py):
    python -m src.ensemble --checkpoint checkpoints/cnn_merged.pt checkpoints/freq_cnn.pt \
        --input_dir data/CIFAKE/test --output_json outputs/preds_ensemble.json

Per-transform accuracy eval (CSV for src/robustness_summary.py):
    python -m src.ensemble --checkpoint checkpoints/cnn_merged.pt checkpoints/freq_cnn.pt \
        --test_dir data/CIFAKE/test data/sid_test_holdout data/wildfake_test_holdout \
        --out_csv outputs/rob_ensemble.csv
"""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.model import load_model
from src.robustness_test import TRANSFORMS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class EnsembleModel:
    def __init__(self, checkpoints, weights=None, calibrations=None):
        self.members = [
            load_model(c, calibration=(calibrations[i] if calibrations else None))
            for i, c in enumerate(checkpoints)
        ]
        n = len(self.members)
        self.weights = [w / sum(weights) for w in weights] if weights else [1.0 / n] * n
        self.backbone = "ensemble(" + "+".join(getattr(m, "backbone", "?") for m in self.members) + ")"
        self.temperature = 1.0  # members are calibrated individually

    def predict(self, image) -> float:
        return round(sum(w * m.predict(image) for w, m in zip(self.weights, self.members)), 4)


def find_images(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def run_inference(model, input_dirs, output_json):
    paths = []
    for d in input_dirs:
        paths.extend(find_images(Path(d)))
    if not paths:
        raise FileNotFoundError(f"No images under {input_dirs}")
    results = []
    for p in tqdm(paths, desc="ensemble inference"):
        try:
            with Image.open(p) as img:
                results.append({"image_path": str(p), "pred": model.predict(img.convert("RGB"))})
        except Exception as e:
            print(f"WARNING: {p}: {e}")
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} predictions to {out}")


def run_eval(model, test_dirs, out_csv, threshold):
    per = {name: {"correct": 0, "total": 0, "yt": [], "ys": []} for name in TRANSFORMS}
    for td in test_dirs:
        td = Path(td)
        for cls, y in [("REAL", 0), ("FAKE", 1)]:
            cd = td / cls
            if not cd.exists():
                print(f"WARNING: {cd} not found, skipping")
                continue
            for p in tqdm(list(cd.iterdir()), desc=f"{td.name}/{cls}"):
                if p.suffix.lower() not in IMAGE_EXTS:
                    continue
                with Image.open(p) as img:
                    img = img.convert("RGB")
                    for name, fn in TRANSFORMS.items():
                        s = model.predict(fn(img.copy()))
                        per[name]["yt"].append(y)
                        per[name]["ys"].append(s)
                        per[name]["correct"] += int((s >= threshold) == y)
                        per[name]["total"] += 1

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["transform", "accuracy", "auc", "n_images"])
        for name, d in per.items():
            if d["total"] == 0:
                continue
            acc = d["correct"] / d["total"]
            auc = roc_auc_score(d["yt"], d["ys"]) if len(set(d["yt"])) == 2 else float("nan")
            w.writerow([name, f"{acc:.4f}", f"{auc:.4f}", d["total"]])
            print(f"{name:18s} acc={acc:.4f}  auc={auc:.4f}  (n={d['total']})")
    print(f"Saved {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="+", required=True, help="two or more checkpoints to average")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="per-member weights (default: equal)")
    ap.add_argument("--calibration", nargs="+", default=None,
                    help="per-member calibrate.py JSONs (same order as --checkpoint)")
    ap.add_argument("--threshold", type=float, default=0.5)
    # inference mode
    ap.add_argument("--input_dir", nargs="+")
    ap.add_argument("--output_json", default="outputs/preds_ensemble.json")
    # eval mode
    ap.add_argument("--test_dir", nargs="+")
    ap.add_argument("--out_csv", default="outputs/rob_ensemble.csv")
    args = ap.parse_args()

    if args.weights and len(args.weights) != len(args.checkpoint):
        raise SystemExit("--weights must match number of --checkpoint")
    if args.calibration and len(args.calibration) != len(args.checkpoint):
        raise SystemExit("--calibration must match number of --checkpoint")

    model = EnsembleModel(args.checkpoint, args.weights, args.calibration)
    print(f"Ensemble: {model.backbone}  weights={[round(w, 3) for w in model.weights]}")

    if args.test_dir:
        run_eval(model, args.test_dir, args.out_csv, args.threshold)
    elif args.input_dir:
        run_inference(model, args.input_dir, args.output_json)
    else:
        raise SystemExit("pass --input_dir (inference) or --test_dir (eval)")


if __name__ == "__main__":
    main()
