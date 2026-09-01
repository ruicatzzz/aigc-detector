"""
Download the WildFake demo / validation benchmark -- COCO real photos +
DALL-E generated images -- for EVALUATION AND DEMO OUTPUT ONLY.

Per the brief, this subset is "for demonstration purposes only ... Do not
use this data during training." It is written to ``data/demo_benchmark/``,
which is deliberately separate from every training directory, and
``src/download_wildfake.py`` still hard-blocks these same two archives so
they can never leak into a training run.

Mechanics are the same streaming approach as ``download_wildfake.py``
(ModelScope file API + on-the-fly unzip), reusing its helpers unchanged.

Layout (ImageFolder-style, drops straight into the eval scripts):
    data/demo_benchmark/REAL/    <- COCO photos      (from Images/Real/coco.zip)
    data/demo_benchmark/FAKE/    <- DALL-E images    (from Images/Diffusion_based/DALLE.zip)

Usage
-----
    python -m src.download_demo_benchmark
    python -m src.download_demo_benchmark --n_real 4998 --n_fake 4000
    python -m src.download_demo_benchmark --dalle_subset advanced --max_mb_fake 9000

Then score it / produce demo output:
    python -m src.eval_auc        --checkpoint <ckpt> --test_dir data/demo_benchmark --out_csv outputs/demo_auc.csv
    python -m src.eval_robustness --checkpoint <ckpt> --test_dir data/demo_benchmark --out_csv outputs/demo_robustness.csv
    python -m src.eval_crossdataset --checkpoint <ckpt> --dataset DemoBenchmark=data/demo_benchmark
    python -m src.infer           --checkpoint <ckpt> --input_dir data/demo_benchmark --output_json outputs/demo_preds.json
"""

import argparse
from pathlib import Path

import requests
from stream_unzip import stream_unzip

# Reused unchanged from the training-subset downloader.
from src.download_wildfake import _url, _http_chunks, _save_image, _IMAGE_EXTS

COCO_ZIP = "Images/Real/coco.zip"
DALLE_ZIP = "Images/Diffusion_based/DALLE.zip"


def harvest(repo_path: str, out_dir: Path, n: int, max_mb: int, revision: str,
            path_filter: str | None = None, tag: str = "img") -> int:
    """Stream one archive, save up to `n` decoded images into `out_dir`.

    `path_filter` (lowercase substring, e.g. "/advanced/") keeps only entries
    whose path matches -- note the skipped entries are still read off the wire,
    so a deep filter can require a large `max_mb`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = _url(repo_path, revision)
    kept = 0
    print(f"  {repo_path} -> {out_dir}/  (target {n}, byte cap {max_mb} MB"
          + (f", filter '{path_filter}'" if path_filter else "") + ")")
    try:
        for name, _size, chunks in stream_unzip(_http_chunks(url, max_mb * (1 << 20))):
            nm = name.decode("utf-8", "replace") if isinstance(name, bytes) else name
            low = nm.lower()
            if nm.endswith("/") or Path(nm).suffix.lower() not in _IMAGE_EXTS \
                    or (path_filter and path_filter not in low):
                for _ in chunks:  # must drain before advancing the stream
                    pass
                continue
            raw = b"".join(chunks)
            dest = out_dir / f"{tag}__{Path(nm).stem[:100]}.jpg"
            if _save_image(raw, dest):
                kept += 1
                if kept % 500 == 0:
                    print(f"    ... {kept}")
            if kept >= n:
                break
    except requests.HTTPError as exc:
        print(f"    HTTP {exc.response.status_code} -- archive unavailable")
    except Exception as exc:
        print(f"    stopped ({type(exc).__name__}: {exc})")
    print(f"    saved {kept} images")
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="data/demo_benchmark",
                    help="REAL/ and FAKE/ are created inside (kept separate from training dirs)")
    ap.add_argument("--n_real", type=int, default=4998, help="COCO images to pull (brief: 4998)")
    ap.add_argument("--n_fake", type=int, default=8843, help="DALL-E images to pull (brief: 8843)")
    ap.add_argument("--dalle_subset", choices=["front", "advanced", "typical"], default="front",
                    help="'front' = first images in the zip (fast; DALLE2/Typical). "
                         "'advanced'/'typical' filter by folder but the zip is ~26 GB and "
                         "'advanced' sits behind all of 'typical', so raise --max_mb_fake a lot.")
    ap.add_argument("--max_mb_real", type=int, default=2500)
    ap.add_argument("--max_mb_fake", type=int, default=3000)
    ap.add_argument("--revision", default="master")
    args = ap.parse_args()

    root = Path(args.out_dir)
    print("=== REAL (COCO) ===")
    n_real = harvest(COCO_ZIP, root / "REAL", args.n_real, args.max_mb_real, args.revision, tag="coco")

    print("=== FAKE (DALL-E) ===")
    path_filter = None if args.dalle_subset == "front" else f"/{args.dalle_subset}/"
    if path_filter:
        print(f"  filtering to '{args.dalle_subset}'; if this saves 0, raise --max_mb_fake")
    n_fake = harvest(DALLE_ZIP, root / "FAKE", args.n_fake, args.max_mb_fake, args.revision,
                     path_filter=path_filter, tag="dalle")

    print(f"\nDemo benchmark ready: {n_real} REAL + {n_fake} FAKE  ->  {root}/")
    print("EVALUATION / DEMO ONLY -- do NOT pass this directory to `src.train --data_dir`.")


if __name__ == "__main__":
    main()
