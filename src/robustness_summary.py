import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Accepts CSVs from either src/eval_robustness.py (transform,accuracy,...) or
# src/eval_auc.py (transform,auc,...). The metric column is auto-detected.
_METRIC_COLS = ("accuracy", "auc")


def load_csv(path):
    rows = {}
    metric = None
    with open(path) as f:
        reader = csv.DictReader(f)
        for col in _METRIC_COLS:
            if col in (reader.fieldnames or []):
                metric = col
                break
        if metric is None:
            raise ValueError(f"{path}: expected one of {_METRIC_COLS} as a column, got {reader.fieldnames}")
        for row in reader:
            rows[row["transform"]] = float(row[metric])
    return rows, metric


def summarize(scores: dict) -> dict:
    """clean vs transformed 3-number summary for one series."""
    clean = scores.get("clean")
    transformed = [v for k, v in scores.items() if k != "clean"]
    return {
        "clean": clean,
        "mean_transformed": sum(transformed) / len(transformed) if transformed else None,
        "worst_transformed": min(transformed) if transformed else None,
    }


def make_summary(csv_specs, out_md, out_png):
    # csv_specs: list of (path, label) tuples
    all_data, metrics = {}, set()
    for path, label in csv_specs:
        scores, metric = load_csv(path)
        all_data[label] = scores
        metrics.add(metric)
    metric_name = "/".join(sorted(metrics))

    transforms = list(next(iter(all_data.values())).keys())

    # --- per-transform table ---
    lines = ["| Transform | " + " | ".join(all_data.keys()) + " |",
             "|---|" + "---|" * len(all_data)]
    for t in transforms:
        row = [t] + [f"{all_data[label][t]:.1%}" for label in all_data]
        lines.append("| " + " | ".join(row) + " |")

    # --- clean / mean-transformed / worst-transformed summary ---
    sums = {label: summarize(scores) for label, scores in all_data.items()}
    summary_lines = [
        "",
        "## Clean vs transformed summary",
        "",
        "| Model | Clean | Mean (transformed) | Worst (transformed) |",
        "|---|---|---|---|",
    ]
    for label, s in sums.items():
        def fmt(v):
            return "n/a" if v is None else f"{v:.1%}"
        summary_lines.append(f"| {label} | {fmt(s['clean'])} | {fmt(s['mean_transformed'])} | {fmt(s['worst_transformed'])} |")

    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        f"# Robustness Evaluation Summary ({metric_name})\n\n"
        + "\n".join(lines) + "\n" + "\n".join(summary_lines) + "\n"
    )
    print(f"Saved table to {out_md}")
    print("\n".join(summary_lines[1:]))

    # --- chart: per-transform bars + a small clean/mean/worst panel ---
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})

    x = range(len(transforms))
    width = 0.8 / len(all_data)
    for i, (label, data) in enumerate(all_data.items()):
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, [data[t] for t in transforms], width, label=label)
    ax.set_xticks([xi + width * (len(all_data) - 1) / 2 for xi in x])
    ax.set_xticklabels(transforms, rotation=45, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, 1)
    ax.set_title(f"{metric_name}: clean vs each transform")
    ax.legend()

    groups = ["clean", "mean_transformed", "worst_transformed"]
    gx = range(len(groups))
    for i, (label, s) in enumerate(sums.items()):
        offsets = [xi + i * width for xi in gx]
        ax2.bar(offsets, [s[g] or 0 for g in groups], width, label=label)
    ax2.set_xticks([xi + width * (len(all_data) - 1) / 2 for xi in gx])
    ax2.set_xticklabels(["clean", "mean\ntransf.", "worst\ntransf."])
    ax2.set_ylim(0, 1)
    ax2.set_title("summary")

    plt.tight_layout()
    out_png = Path(out_png)
    plt.savefig(out_png, dpi=150)
    print(f"Saved chart to {out_png}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True,
                         help="path:label pairs, e.g. outputs/table.csv:MyModel")
    parser.add_argument("--out_md", default="outputs/robustness_summary.md")
    parser.add_argument("--out_png", default="outputs/robustness_chart.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    specs = [tuple(s.rsplit(":", 1)) for s in args.csv]
    make_summary(specs, args.out_md, args.out_png)
