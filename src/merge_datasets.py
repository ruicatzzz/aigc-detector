"""
Merges CIFAKE, a downloaded SID_Set subset, and (optionally) a normalized
WildFake subset into one combined dataset, in the same root_dir/split/{REAL,FAKE}
layout that dataset.py expects.

Assumes you've already run:
    - CIFAKE unzipped to some folder with train/REAL, train/FAKE, test/REAL, test/FAKE
    - download_sid_set.py, saved to some folder with REAL/FAKE subfolders
    - (optional) WildFake normalized into REAL/FAKE folders — see the
      "Download WildFake" notebook section, since WildFake's raw structure
      is hierarchical by generator type, not simple REAL/FAKE, so it needs
      manual normalization first (unlike CIFAKE/SID_Set which this script
      handles automatically).

This script copies (not moves) images from both sources into a new combined
folder, renaming files with a source prefix to avoid filename collisions
(CIFAKE and SID_Set may reuse filenames like "0001.jpg").

Usage:
    python merge_datasets.py \
        --cifake_dir ../data \
        --sid_train_dir ../data_sid/train \
        --sid_test_dir ../data_sid/test \
        --out_dir ../data_combined
"""

import os
import shutil
import argparse

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")


def copy_class_folder(src_dir: str, dst_dir: str, prefix: str) -> int:
    """Copy all valid images from src_dir into dst_dir, prefixing filenames
    with `prefix` to avoid collisions between sources. Returns count copied."""
    if not os.path.isdir(src_dir):
        print(f"  WARNING: source folder not found, skipping: {src_dir}")
        return 0

    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(VALID_EXTENSIONS):
            continue
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, f"{prefix}_{fname}")
        shutil.copyfile(src_path, dst_path)
        count += 1
    return count


def merge(args):
    combined_train = os.path.join(args.out_dir, "train")
    combined_test = os.path.join(args.out_dir, "test")

    print("Merging TRAIN split...")
    for cls in ["REAL", "FAKE"]:
        dst = os.path.join(combined_train, cls)
        n_cifake = copy_class_folder(
            os.path.join(args.cifake_dir, "train", cls), dst, prefix="cifake"
        )
        n_sid = copy_class_folder(
            os.path.join(args.sid_train_dir, cls), dst, prefix="sid"
        )
        n_wildfake = 0
        if args.wildfake_train_dir:
            n_wildfake = copy_class_folder(
                os.path.join(args.wildfake_train_dir, cls), dst, prefix="wildfake"
            )
        total = n_cifake + n_sid + n_wildfake
        print(f"  {cls}: {n_cifake} from CIFAKE + {n_sid} from SID_Set "
              f"+ {n_wildfake} from WildFake = {total} total")

    print("Merging TEST split...")
    for cls in ["REAL", "FAKE"]:
        dst = os.path.join(combined_test, cls)
        n_cifake = copy_class_folder(
            os.path.join(args.cifake_dir, "test", cls), dst, prefix="cifake"
        )
        n_sid = copy_class_folder(
            os.path.join(args.sid_test_dir, cls), dst, prefix="sid"
        )
        n_wildfake = 0
        if args.wildfake_test_dir:
            n_wildfake = copy_class_folder(
                os.path.join(args.wildfake_test_dir, cls), dst, prefix="wildfake"
            )
        total = n_cifake + n_sid + n_wildfake
        print(f"  {cls}: {n_cifake} from CIFAKE + {n_sid} from SID_Set "
              f"+ {n_wildfake} from WildFake = {total} total")

    print(f"\nDone. Combined dataset saved to {args.out_dir}/")
    print("Point train.py / evaluate.py / build_test_transforms.py at this folder "
          "with --data_dir to use the combined dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifake_dir", default="../data",
                         help="Path to unzipped CIFAKE (expects train/{REAL,FAKE}, test/{REAL,FAKE})")
    parser.add_argument("--sid_train_dir", default="../data_sid/train",
                         help="Path to SID_Set train subset from download_sid_set.py (expects REAL/FAKE)")
    parser.add_argument("--sid_test_dir", default="../data_sid/test",
                         help="Path to SID_Set validation subset from download_sid_set.py (expects REAL/FAKE)")
    parser.add_argument("--wildfake_train_dir", default=None,
                         help="Optional: path to normalized WildFake train subset (expects REAL/FAKE). "
                              "See the WildFake normalization notebook cell — raw WildFake is NOT "
                              "in this layout by default and must be normalized first.")
    parser.add_argument("--wildfake_test_dir", default=None,
                         help="Optional: path to normalized WildFake test subset (expects REAL/FAKE)")
    parser.add_argument("--out_dir", default="../data_combined",
                         help="Where to write the combined train/{REAL,FAKE}, test/{REAL,FAKE} dataset")
    args = parser.parse_args()

    merge(args)
