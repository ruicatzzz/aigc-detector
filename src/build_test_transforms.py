"""
Builds a FIXED transformed test set — deterministic, applied once, saved to disk.

This is different from the random_transform() used during training augmentation:
- Training augmentation: random transform chosen on-the-fly each epoch, for robustness.
- This script: EVERY transform applied to EVERY test image, saved as static files,
  so the whole team evaluates against the exact same held-out images
  (required for an honest robustness table — see evaluate.py).

Usage:
    python build_test_transforms.py --test_dir ../data/test --out_dir ../data/test_transformed
    python build_test_transforms.py --limit 200   # use a subset per class for fast iteration
"""

import os
import argparse
from PIL import Image
from augmentations import TRANSFORMS
from dataset import CLASSES


def build_transformed_test_set(test_dir: str, out_dir: str, limit: int = None):
    os.makedirs(out_dir, exist_ok=True)
    total_saved = 0

    for cls in CLASSES:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Expected folder not found: {cls_dir}")

        fnames = sorted(
            f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if limit is not None:
            fnames = fnames[:limit]

        print(f"[{cls}] processing {len(fnames)} images...")

        for fname in fnames:
            img_path = os.path.join(cls_dir, fname)
            img = Image.open(img_path).convert("RGB")

            for t_name, t_fn in TRANSFORMS.items():
                out_subdir = os.path.join(out_dir, t_name, cls)
                os.makedirs(out_subdir, exist_ok=True)
                out_path = os.path.join(out_subdir, fname)
                try:
                    t_fn(img).save(out_path)
                    total_saved += 1
                except Exception as e:
                    print(f"  WARNING: failed to apply {t_name} to {fname}: {e}")

    print(f"Done. Saved {total_saved} transformed images to {out_dir}/")
    print(f"Transform types: {list(TRANSFORMS.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="../data/test",
                         help="Path to clean test set (root_dir/test/{REAL,FAKE})")
    parser.add_argument("--out_dir", default="../data/test_transformed",
                         help="Where to save transformed images, one subfolder per transform")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on images per class, for fast iteration")
    args = parser.parse_args()

    build_transformed_test_set(args.test_dir, args.out_dir, args.limit)
