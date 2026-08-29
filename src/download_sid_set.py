"""
Downloads a manageable subset of saberzl/SID_Set from Hugging Face and saves
it into the same root_dir/split/{REAL,FAKE} layout used by dataset.py.

SID_Set is 240K images / 140GB total (parquet, streamed) — do NOT try to
download it in full for a hackathon. This script streams the dataset and
saves only `--n_per_class` images per class per split, which is enough for
training/eval without blowing up disk space or download time.

Label mapping (SID_Set has 3 classes, we collapse to binary):
    0 (real)          -> REAL
    1 (full_synthetic) -> FAKE
    2 (tampered)        -> FAKE   (partially AI-edited; still "not authentic")

If you want tampered images kept as their own analysis category later
(the dataset also ships masks for these), the raw label is preserved
alongside the image via a metadata CSV for optional deeper analysis.

Usage:
    pip install datasets pillow --break-system-packages
    python download_sid_set.py --out_dir ../data_sid --n_per_class 1000 --split train
    python download_sid_set.py --out_dir ../data_sid --n_per_class 300 --split validation
"""

import os
import argparse
import csv
from datasets import load_dataset

LABEL_MAP = {
    0: "REAL",
    1: "FAKE",   # full synthetic
    2: "FAKE",   # tampered
}
RAW_LABEL_NAME = {0: "real", 1: "full_synthetic", 2: "tampered"}


def download_subset(out_dir: str, n_per_class: int, split: str):
    # split arg here is SID_Set's own split name ("train" or "validation"),
    # NOT your project's train/test naming — see note below on mapping.
    print(f"Streaming saberzl/SID_Set split='{split}' ...")
    ds = load_dataset("saberzl/SID_Set", split=split, streaming=True)

    counts = {"REAL": 0, "FAKE": 0}
    target = n_per_class

    os.makedirs(os.path.join(out_dir, "REAL"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "FAKE"), exist_ok=True)

    metadata_path = os.path.join(out_dir, f"{split}_metadata.csv")
    with open(metadata_path, "w", newline="") as meta_f:
        writer = csv.writer(meta_f)
        writer.writerow(["img_id", "binary_label", "raw_label", "saved_path"])

        for example in ds:
            raw_label = example["label"]
            binary_label = LABEL_MAP[raw_label]

            if counts[binary_label] >= target:
                # Stop once both classes hit target
                if all(c >= target for c in counts.values()):
                    break
                continue

            img = example["image"].convert("RGB")
            img_id = example["img_id"]
            fname = f"{img_id}.jpg"
            save_path = os.path.join(out_dir, binary_label, fname)
            img.save(save_path, format="JPEG", quality=95)

            writer.writerow([img_id, binary_label, RAW_LABEL_NAME[raw_label], save_path])
            counts[binary_label] += 1

            if sum(counts.values()) % 200 == 0:
                print(f"  progress: REAL={counts['REAL']} FAKE={counts['FAKE']}")

    print(f"Done. Saved REAL={counts['REAL']} FAKE={counts['FAKE']} to {out_dir}/")
    print(f"Metadata (incl. real/full_synthetic/tampered breakdown) written to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="../data_sid/train",
                         help="Where to save REAL/FAKE folders")
    parser.add_argument("--n_per_class", type=int, default=1000,
                         help="How many images per class to download (keep small for a hackathon)")
    parser.add_argument("--split", default="train", choices=["train", "validation"],
                         help="SID_Set's own split name")
    args = parser.parse_args()

    download_subset(args.out_dir, args.n_per_class, args.split)
