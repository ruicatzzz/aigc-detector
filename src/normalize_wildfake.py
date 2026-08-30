"""
Normalizes a cloned WildFake repo into root_dir/split/{REAL,FAKE} layout.

WildFake is NOT organized as simple REAL/FAKE folders like CIFAKE or
SID_Set — it has a hierarchical structure by generator type (GAN, diffusion,
etc.), and folder names may be in Chinese. Because of this, auto-detection
(like build_test_transforms.py does for CIFAKE) is unreliable here — you
need to look at the actual cloned repo structure first and tell this script
which subfolders are real vs. fake.

IMPORTANT — reserved benchmark exclusion:
The hackathon problem statement reserves a specific WildFake subset (COCO
val2017 for non-AIGC, DALL-E Advanced for AIGC) as a shared reference
benchmark, and explicitly says: "Do not use the following data during
training." This script automatically excludes any path containing "coco",
"val2017", "dalle", "dall-e", or "dall_e" (case-insensitive), regardless of
what you pass in --real_dirs/--fake_dirs, as a safety net against
accidentally training on the reserved data.

Workflow:
    1. Clone the repo (see the notebook "Download WildFake" section)
    2. Run print_tree() on the cloned folder to see the actual structure
    3. Identify which folder(s) contain REAL images and which contain FAKE
       (generator) images
    4. Fill in REAL_DIRS and FAKE_DIRS below, or pass them via --real_dirs /
       --fake_dirs
    5. Run this script to copy images into a normalized REAL/FAKE layout

Usage:
    # First, just inspect:
    python normalize_wildfake.py --raw_dir ../WildFake --inspect_only

    # Then, once you've identified the real/fake subfolders:
    python normalize_wildfake.py --raw_dir ../WildFake \\
        --real_dirs "real" \\
        --fake_dirs "gan/stylegan2,diffusion/stable_diffusion" \\
        --out_dir ../data_wildfake/train --limit 1000
"""

import os
import shutil
import argparse

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def print_tree(path: str, prefix: str = "", max_depth: int = 4, depth: int = 0):
    """Prints folder structure so you can identify real vs. fake subfolders."""
    if depth > max_depth or not os.path.isdir(path):
        return
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return
    for item in entries:
        full = os.path.join(path, item)
        if os.path.isdir(full):
            n_items = len(os.listdir(full))
            print(f"{prefix}{item}/  ({n_items} items)")
            print_tree(full, prefix + "  ", max_depth, depth + 1)
        elif depth <= max_depth:
            # Only show a sample of files, not every single one
            pass


def collect_images(folder: str, limit: int = None, exclude_patterns: list = None):
    """Recursively collect image file paths under a folder, skipping any path
    that matches an exclude pattern (case-insensitive substring match)."""
    exclude_patterns = exclude_patterns or []
    found = []
    skipped = 0
    for dirpath, _, filenames in os.walk(folder):
        lower_path = dirpath.lower()
        if any(pat.lower() in lower_path for pat in exclude_patterns):
            skipped += len(filenames)
            continue
        for fname in filenames:
            if fname.lower().endswith(VALID_EXTENSIONS):
                found.append(os.path.join(dirpath, fname))
                if limit and len(found) >= limit:
                    if skipped:
                        print(f"  (skipped {skipped} images under excluded paths)")
                    return found
    if skipped:
        print(f"  (skipped {skipped} images under excluded paths)")
    return found


# These subsets are reserved by the organizers as a shared reference/demo
# benchmark and must NOT be used during training — see the problem statement:
# "Do not use the following data during training." Any folder path containing
# these substrings (case-insensitive) is automatically excluded below, as a
# safety net even if you point --real_dirs / --fake_dirs at a broader parent
# folder that happens to contain them.
RESERVED_BENCHMARK_PATTERNS = [
    "coco",           # COCO val2017 — reserved non-AIGC reference subset
    "val2017",
    "dalle",          # DALL-E Advanced — reserved AIGC reference subset
    "dall-e",
    "dall_e",
]


def normalize(args):
    os.makedirs(os.path.join(args.out_dir, "REAL"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "FAKE"), exist_ok=True)

    real_dirs = [d.strip() for d in args.real_dirs.split(",")] if args.real_dirs else []
    fake_dirs = [d.strip() for d in args.fake_dirs.split(",")] if args.fake_dirs else []

    user_excludes = [p.strip() for p in args.exclude_patterns.split(",")] if args.exclude_patterns else []
    exclude_patterns = RESERVED_BENCHMARK_PATTERNS + user_excludes
    print(f"Excluding any path matching: {exclude_patterns}\n")

    total_real, total_fake = 0, 0

    for rd in real_dirs:
        full_path = os.path.join(args.raw_dir, rd)
        images = collect_images(full_path, limit=args.limit, exclude_patterns=exclude_patterns)
        for i, img_path in enumerate(images):
            ext = os.path.splitext(img_path)[1]
            dst = os.path.join(args.out_dir, "REAL", f"wildfake_real_{total_real + i}{ext}")
            shutil.copyfile(img_path, dst)
        total_real += len(images)
        print(f"REAL <- {full_path}: {len(images)} images")

    for fd in fake_dirs:
        full_path = os.path.join(args.raw_dir, fd)
        images = collect_images(full_path, limit=args.limit, exclude_patterns=exclude_patterns)
        for i, img_path in enumerate(images):
            ext = os.path.splitext(img_path)[1]
            safe_name = fd.replace("/", "_").replace(" ", "_")
            dst = os.path.join(args.out_dir, "FAKE", f"wildfake_{safe_name}_{i}{ext}")
            shutil.copyfile(img_path, dst)
        total_fake += len(images)
        print(f"FAKE <- {full_path}: {len(images)} images")

    print(f"\nDone. REAL={total_real}, FAKE={total_fake} saved to {args.out_dir}/")
    print("Reserved benchmark data (COCO val2017 / DALL-E Advanced) was excluded "
          "if present under any scanned folder — see 'skipped' counts above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True,
                         help="Path to the cloned WildFake repo")
    parser.add_argument("--inspect_only", action="store_true",
                         help="Just print the folder tree and exit, don't copy anything")
    parser.add_argument("--real_dirs", default=None,
                         help="Comma-separated subfolder(s) under raw_dir containing REAL images")
    parser.add_argument("--fake_dirs", default=None,
                         help="Comma-separated subfolder(s) under raw_dir containing FAKE images "
                              "(you can list multiple generator folders here, e.g. "
                              "'gan/stylegan2,diffusion/stable_diffusion')")
    parser.add_argument("--out_dir", default="../data_wildfake/train",
                         help="Where to save the normalized REAL/FAKE folders")
    parser.add_argument("--limit", type=int, default=1000,
                         help="Max images to copy per source folder (WildFake is very large — "
                              "keep this small for a hackathon)")
    parser.add_argument("--exclude_patterns", default=None,
                         help="Comma-separated extra substrings to exclude from paths, on top of "
                              "the built-in reserved-benchmark exclusions (COCO val2017, DALL-E Advanced)")
    args = parser.parse_args()

    if args.inspect_only:
        print(f"Folder structure under {args.raw_dir}:\n")
        print_tree(args.raw_dir)
    else:
        if not args.real_dirs or not args.fake_dirs:
            raise ValueError(
                "Both --real_dirs and --fake_dirs are required unless --inspect_only is set. "
                "Run with --inspect_only first to see the folder structure."
            )
        normalize(args)