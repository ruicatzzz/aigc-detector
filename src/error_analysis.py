"""
Error analysis: surface representative false positives / false negatives and
the images that stay wrong across many transforms.

Produces (deliverable 5, "Error Analysis Note"):
  outputs/error_analysis_examples/false_positives/*   worst REAL-called-AI
  outputs/error_analysis_examples/false_negatives/*   worst AI-called-REAL
  outputs/error_analysis_note.md                      counts, tables, template

Usage:
    python -m src.error_analysis --checkpoint checkpoints/cnn_merged.pt \
        --test_dir data/CIFAKE/test data/sid_test_holdout data/wildfake_test_holdout
    # optional: --calibration outputs/calibration.json --threshold 0.5 --k 12
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.model import load_model
from src.robustness_test import TRANSFORMS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_clean(model, test_dirs, threshold):
    """[(path, dataset, true_label, prob, correct), ...] on clean images."""
    recs = []
    for d in test_dirs:
        d = Path(d)
        for cls, y in [("REAL", 0), ("FAKE", 1)]:
            cls_dir = d / cls
            if not cls_dir.exists():
                print(f"WARNING: {cls_dir} not found, skipping")
                continue
            paths = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            for p in tqdm(paths, desc=f"{d.name}/{cls}"):
                try:
                    with Image.open(p) as im:
                        prob = model.predict(im.convert("RGB"))
                except Exception as e:
                    print(f"  skip {p}: {e}")
                    continue
                pred = 1 if prob >= threshold else 0
                recs.append((p, d.name, y, prob, pred == y))
    return recs


def hard_scan(model, candidates, threshold):
    """For each (path, ...), count how many transforms it is misclassified under."""
    out = []
    for p, dataset, y, prob, _ in tqdm(candidates, desc="transform scan"):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                fails = sum(
                    1 for _, fn in TRANSFORMS.items()
                    if (1 if model.predict(fn(im.copy())) >= threshold else 0) != y
                )
        except Exception:
            continue
        out.append((p, dataset, y, prob, fails))
    out.sort(key=lambda r: r[4], reverse=True)
    return out


def _save(examples, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for i, (p, dataset, _y, prob, *_rest) in enumerate(examples):
        shutil.copy(p, dest / f"{i:02d}_{dataset}_p{prob:.3f}_{p.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_dir", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--k", type=int, default=12, help="examples to save per error type")
    ap.add_argument("--hard_scan_n", type=int, default=300,
                    help="how many near-boundary images to re-test under every transform (0 disables)")
    ap.add_argument("--out_dir", default="outputs/error_analysis_examples")
    ap.add_argument("--out_md", default="outputs/error_analysis_note.md")
    args = ap.parse_args()

    model = load_model(args.checkpoint, calibration=args.calibration)
    recs = collect_clean(model, args.test_dir, args.threshold)
    if not recs:
        raise SystemExit("No images scored.")

    reals = [r for r in recs if r[2] == 0]
    fakes = [r for r in recs if r[2] == 1]
    fp = sorted((r for r in reals if not r[4]), key=lambda r: r[3], reverse=True)  # high prob
    fn = sorted((r for r in fakes if not r[4]), key=lambda r: r[3])                # low prob

    _save(fp[:args.k], Path(args.out_dir) / "false_positives")
    _save(fn[:args.k], Path(args.out_dir) / "false_negatives")

    fpr = len(fp) / len(reals) if reals else 0.0
    fnr = len(fn) / len(fakes) if fakes else 0.0

    lines = [
        "# Error Analysis Note", "",
        f"- Checkpoint: `{args.checkpoint}` (backbone: {model.backbone}, "
        f"temperature: {model.temperature:.3f})",
        f"- Test set: {len(recs)} images ({len(reals)} REAL / {len(fakes)} FAKE) "
        f"from {', '.join(Path(d).name for d in args.test_dir)}",
        f"- Decision threshold: {args.threshold}",
        f"- **False-positive rate** (REAL flagged as AI): {fpr:.1%}  ({len(fp)}/{len(reals)})",
        f"- **False-negative rate** (AI passed as REAL): {fnr:.1%}  ({len(fn)}/{len(fakes)})",
        "",
        "## Worst false positives (REAL, highest P(AI))", "",
        "| rank | dataset | P(AI) | file |", "|---|---|---|---|",
    ]
    for i, (p, ds, _y, prob, _c) in enumerate(fp[:args.k]):
        lines.append(f"| {i} | {ds} | {prob:.3f} | `{p.name}` |")
    lines += ["", "## Worst false negatives (AI, lowest P(AI))", "",
              "| rank | dataset | P(AI) | file |", "|---|---|---|---|"]
    for i, (p, ds, _y, prob, _c) in enumerate(fn[:args.k]):
        lines.append(f"| {i} | {ds} | {prob:.3f} | `{p.name}` |")

    if args.hard_scan_n > 0:
        near = sorted(recs, key=lambda r: abs(r[3] - args.threshold))[:args.hard_scan_n]
        hard = hard_scan(model, near, args.threshold)
        lines += ["", f"## Consistently hard images "
                      f"(misclassified under the most of {len(TRANSFORMS)} transforms)", "",
                  "| dataset | true | clean P(AI) | # transforms wrong | file |",
                  "|---|---|---|---|---|"]
        for p, ds, y, prob, fails in hard[:args.k]:
            lines.append(f"| {ds} | {'REAL' if y == 0 else 'AI'} | {prob:.3f} | "
                         f"{fails}/{len(TRANSFORMS)} | `{p.name}` |")

    lines += [
        "", "## Trade-offs & observations", "",
        "<!-- Fill in after eyeballing outputs/error_analysis_examples/: -->",
        "- Do the false positives share a trait (low resolution? heavy texture? a "
        "particular source dataset)?",
        "- Do the false negatives come from one generator family?",
        "- Are the consistently-hard images the same few across transforms (a small "
        "fragile subset) or spread out?",
        "- Threshold trade-off: raising it cuts false positives but raises the "
        "false-negative rate — see `outputs/calibration_note.md` for operating points.",
        "",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {args.out_md} and examples under {args.out_dir}/")


if __name__ == "__main__":
    main()
