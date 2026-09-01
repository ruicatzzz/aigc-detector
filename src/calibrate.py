"""
Confidence calibration + threshold selection for a trained checkpoint.

Slide 2 of the brief: "output a calibrated probability, not just a label".
This script:

  1. runs the model over a held-out (clean) set to collect raw logits + labels
  2. fits a single temperature T (logit -> logit / T) by minimising NLL
  3. picks decision thresholds that hit target false-positive rates
     (FPR = real images wrongly flagged as AI)
  4. writes outputs/calibration.json  ({"temperature", "thresholds", ...})
     and prints an operating-point table for the error-analysis / robustness
     write-ups.

Usage:
    python -m src.calibrate --checkpoint checkpoints/cnn_merged.pt \
        --val_dir data/CIFAKE/test data/sid_test_holdout data/wildfake_test_holdout

Then apply it:
    python -m src.infer  ... --calibration outputs/calibration.json
    python -m src.eval_robustness ... --calibration outputs/calibration.json --threshold <t@FPR5%>
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.model import RealModel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_FPRS = [0.01, 0.05, 0.10]


def collect_logits(model: RealModel, val_dirs):
    logits, labels = [], []
    for d in val_dirs:
        d = Path(d)
        for cls, y in [("REAL", 0), ("FAKE", 1)]:
            cls_dir = d / cls
            if not cls_dir.exists():
                print(f"WARNING: {cls_dir} not found, skipping")
                continue
            paths = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            for p in tqdm(paths, desc=f"{d.name}/{cls}"):
                try:
                    with Image.open(p) as img:
                        logits.append(model.logit(img.convert("RGB")))
                        labels.append(y)
                except Exception as e:
                    print(f"  skip {p}: {e}")
    return torch.tensor(logits), torch.tensor(labels, dtype=torch.float32)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T so T stays > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def metrics_at(probs, labels, thr):
    pred = (probs >= thr).float()
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0            # recall on FAKE
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    acc = (tp + tn) / len(labels)
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
    return {"threshold": thr, "fpr": fpr, "recall": tpr, "precision": prec, "f1": f1, "accuracy": acc}


def threshold_for_fpr(probs, labels, target_fpr):
    real = probs[labels == 0]
    if real.numel() == 0:
        return 0.5
    # smallest threshold whose FPR <= target: the (1 - target) quantile of REAL scores
    return float(torch.quantile(real, 1.0 - target_fpr).item())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val_dir", nargs="+", required=True,
                    help="held-out ImageFolder dirs (REAL/ FAKE/), clean images")
    ap.add_argument("--out_json", default="outputs/calibration.json")
    ap.add_argument("--out_md", default="outputs/calibration_note.md")
    args = ap.parse_args()

    model = RealModel(args.checkpoint)
    logits, labels = collect_logits(model, args.val_dir)
    if labels.unique().numel() < 2:
        raise SystemExit("Need both REAL and FAKE images in --val_dir to calibrate.")

    T = fit_temperature(logits, labels)
    cal_probs = torch.sigmoid(logits / T)
    raw_probs = torch.sigmoid(logits)

    rows = [metrics_at(cal_probs, labels, 0.5)]
    thresholds = {"0.5": 0.5}
    for fpr in TARGET_FPRS:
        thr = threshold_for_fpr(cal_probs, labels, fpr)
        thresholds[f"fpr_{fpr:.2f}"] = thr
        rows.append(metrics_at(cal_probs, labels, thr))

    def nll(p):
        p = p.clamp(1e-6, 1 - 1e-6)
        return float(-(labels * p.log() + (1 - labels) * (1 - p).log()).mean())

    out = {
        "checkpoint": args.checkpoint,
        "backbone": model.backbone,
        "n_val": int(labels.numel()),
        "temperature": T,
        "nll_before": nll(raw_probs),
        "nll_after": nll(cal_probs),
        "thresholds": thresholds,
        "operating_points": rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))

    lines = [
        "# Calibration Note", "",
        f"- Checkpoint: `{args.checkpoint}` (backbone: {model.backbone})",
        f"- Calibration set: {int(labels.numel())} images "
        f"({int((labels == 0).sum())} REAL / {int((labels == 1).sum())} FAKE)",
        f"- Fitted temperature **T = {T:.3f}**  "
        f"(NLL {out['nll_before']:.4f} -> {out['nll_after']:.4f})", "",
        "## Operating points (after temperature scaling)", "",
        "| Threshold | FPR | Recall (FAKE) | Precision | F1 | Accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['threshold']:.3f} | {r['fpr']:.1%} | {r['recall']:.1%} | "
            f"{r['precision']:.1%} | {r['f1']:.3f} | {r['accuracy']:.1%} |"
        )
    lines += ["", f"Apply with `--calibration {args.out_json}`; for a fixed-FPR "
                  f"operating point also pass `--threshold <value from the table>`."]
    Path(args.out_md).write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
