import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path):
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["transform"]] = float(row["accuracy"])
    return rows


def make_summary(csv_specs, out_md, out_png):
    # csv_specs: list of (path, label) tuples
    all_data = {}
    for path, label in csv_specs:
        all_data[label] = load_csv(path)

    transforms = list(next(iter(all_data.values())).keys())

    # --- Markdown table ---
    lines = ["| Transform | " + " | ".join(all_data.keys()) + " |",
             "|---|" + "---|" * len(all_data)]
    for t in transforms:
        row = [t] + [f"{all_data[label][t]:.1%}" for label in all_data]
        lines.append("| " + " | ".join(row) + " |")

    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("# Robustness Evaluation Summary\n\n" + "\n".join(lines) + "\n")
    print(f"Saved table to {out_md}")

    # --- Chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(transforms))
    width = 0.8 / len(all_data)
    for i, (label, data) in enumerate(all_data.items()):
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, [data[t] for t in transforms], width, label=label)
    ax.set_xticks([xi + width * (len(all_data) - 1) / 2 for xi in x])
    ax.set_xticklabels(transforms, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy: clean vs transformed images")
    ax.legend()
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
    specs = [tuple(s.split(":", 1)) for s in args.csv]
    make_summary(specs, args.out_md, args.out_png)