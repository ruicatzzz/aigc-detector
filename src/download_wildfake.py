"""
Download a bounded subset of WildFake (ModelScope: hy2628982280/WildFake)
for training, WITHOUT pulling the full ~1.3 TB dataset.

Why this script exists
----------------------
WildFake ships as a handful of very large ``.zip`` archives, one per
generator family (some split into ~50 GB parts). There is no
Hugging-Face-style streaming loader for it. However:

* ModelScope's file endpoint serves each archive over HTTP with
  ``Accept-Ranges: bytes``, and
* every archive stores its entries sequentially (front-to-back).

So we can open an HTTP stream for each zip, decode it on the fly with
``stream-unzip`` (which never needs the central directory at the end of
the file), and stop after the first N images. In practice that pulls a
few hundred MB per archive instead of tens of GB.

Held-out validation data is EXCLUDED here, per the hackathon rules
("Do not use the following data during training"):

* ``Images/Real/coco.zip``             -> COCO val2017  (Non-AIGC benchmark)
* ``Images/Diffusion_based/DALLE.zip`` -> "DALL-E Advanced" (AIGC benchmark)

Both are hard-blocked below (:data:`BLOCKED_SOURCES`) and never fetched.
Skipping the two archives wholesale is the safe move: it guarantees zero
overlap with the demo benchmark without needing the organisers' exact
image id list.

Output layout (ImageFolder-style -- matches ``src/train.py`` and the
eval scripts, which expect ``REAL`` / ``FAKE`` subfolders):

    data/wildfake_subset/{REAL,FAKE}         <- training subset
    data/wildfake_test_holdout/{REAL,FAKE}   <- non-overlapping local test set

The train and holdout sets are drawn from the SAME stream, holdout taken
strictly after the training slice, so they never share an image.

Usage
-----
    pip install stream-unzip requests pillow
    python -m src.download_wildfake                         # sensible defaults
    python -m src.download_wildfake --fake_per_source 1500 --real_per_source 1500
    python -m src.download_wildfake --only afhq celebahq    # just a couple of sources

Then train with WildFake merged in alongside the other datasets:
    python -m src.train --data_dir data/cifake/train data/sid_subset data/wildfake_subset \
        --epochs 10 --out checkpoints/cnn_merged.pt
"""

import argparse
import io
import urllib.parse
from pathlib import Path

import requests
from stream_unzip import stream_unzip

# ModelScope dataset file API. Returns a 302 to a CDN URL that supports
# range requests / plain streaming.
_DATASET = "hy2628982280/WildFake"
_FILE_API = "https://modelscope.cn/api/v1/datasets/{ds}/repo?Revision={rev}&FilePath={path}"

# Archives that contain the reserved demo/validation benchmark. NEVER download.
BLOCKED_SOURCES = {
    "Images/Real/coco.zip",                 # COCO val2017  (Non-AIGC benchmark, 4998 imgs)
    "Images/Diffusion_based/DALLE.zip",      # "DALL-E Advanced" (AIGC benchmark, 8843 imgs)
}

# AI-generated archives to sample from (diverse generator families; DALLE
# deliberately omitted -- see BLOCKED_SOURCES). Each entry is
# "<zip path in repo>": "<short tag used in output filenames>".
FAKE_SOURCES = {
    "Images/Diffusion_based/ADM.zip": "adm",
    "Images/Diffusion_based/DDIM.zip": "ddim",
    "Images/Diffusion_based/DDPM.zip": "ddpm",
    "Images/Diffusion_based/Imagen.zip": "imagen",
    "Images/Diffusion_based/VQDM.zip": "vqdm",
    "Images/Diffusion_based/Midjourney/Typical/part_1.zip": "midjourney_typical",
    "Images/Diffusion_based/SD/personalizedSD.zip": "sd_personalized",
    "Images/GAN_based.zip": "gan",
    "Images/Other_based.zip": "other",
}

# Real-image archives to sample from (coco deliberately omitted).
REAL_SOURCES = {
    "Images/Real/afhq.zip": "afhq",
    "Images/Real/celebahq.zip": "celebahq",
    "Images/Real/church.zip": "church",
    "Images/Real/ffhq.zip": "ffhq",
    "Images/Real/imagenet.zip": "imagenet",
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_HEADERS = {"User-Agent": "aigc-detector-hackathon/1.0 (wildfake subset fetch)"}


def _url(repo_path: str, revision: str) -> str:
    return _FILE_API.format(
        ds=_DATASET, rev=revision, path=urllib.parse.quote(repo_path, safe="")
    )


def _http_chunks(url: str, max_bytes: int, chunk_size: int = 1 << 16):
    """Yield raw bytes from an HTTP stream, stopping once max_bytes is hit.

    The cap is a safety net so a source whose images are unexpectedly large
    (or whose first entries aren't images at all) can't blow up the
    download. Under normal conditions we break out well before this.
    """
    seen = 0
    with requests.get(url, stream=True, timeout=120, headers=_HEADERS) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            seen += len(chunk)
            yield chunk
            if seen >= max_bytes:
                print(f"    [hit {max_bytes // (1 << 20)} MB byte cap for this source]")
                return


def _looks_like_reserved(entry_name: str) -> bool:
    """Defence-in-depth: skip any entry that still smells like the reserved
    COCO / DALL-E Advanced data even if it turns up inside another archive."""
    low = entry_name.lower()
    return "coco" in low or "dalle" in low or "dall-e" in low or "dall_e" in low


def _save_image(raw: bytes, dest: Path) -> bool:
    """Re-encode to RGB JPEG (consistent with the SID/CIFAKE pipeline).
    Returns False if the bytes don't decode as an image."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - clear guidance instead of a stack trace
        raise SystemExit("Pillow is required: pip install pillow") from exc
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.convert("RGB").save(dest, format="JPEG", quality=95)
        return True
    except Exception:
        return False


def harvest_source(
    repo_path: str,
    tag: str,
    train_dir: Path,
    test_dir: Path,
    n_train: int,
    n_test: int,
    max_mb: int,
    revision: str,
) -> tuple[int, int]:
    """Stream one archive; write the first ``n_train`` images into
    ``train_dir`` and the next ``n_test`` into ``test_dir``.

    Returns (train_written, test_written).
    """
    if repo_path in BLOCKED_SOURCES:
        raise ValueError(f"Refusing to download reserved benchmark archive: {repo_path}")

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    url = _url(repo_path, revision)
    need = n_train + n_test
    kept = 0
    train_written = test_written = 0
    print(f"  {repo_path}")

    try:
        stream = stream_unzip(_http_chunks(url, max_mb * (1 << 20)))
        for entry_name, _size, chunks in stream:
            name = entry_name.decode("utf-8", "replace") if isinstance(entry_name, bytes) else entry_name

            if name.endswith("/") or Path(name).suffix.lower() not in _IMAGE_EXTS or _looks_like_reserved(name):
                for _ in chunks:  # must drain before advancing the stream
                    pass
                continue

            raw = b"".join(chunks)
            is_train = kept < n_train
            out_root = train_dir if is_train else test_dir
            stem = Path(name).stem[:100]  # some SD prompts are used verbatim as filenames
            dest = out_root / f"{tag}__{stem}.jpg"

            if _save_image(raw, dest):
                kept += 1
                if is_train:
                    train_written += 1
                else:
                    test_written += 1

            if kept >= need:
                break
    except requests.HTTPError as exc:
        print(f"    SKIPPED ({exc.response.status_code}) -- source not found or unavailable")
    except Exception as exc:  # keep going with the other sources
        print(f"    SKIPPED ({type(exc).__name__}: {exc})")

    print(f"    -> {train_written} train, {test_written} holdout")
    return train_written, test_written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_train", default="data/wildfake_subset",
                    help="ImageFolder root for the training subset (REAL/ FAKE/ created inside)")
    ap.add_argument("--out_test", default="data/wildfake_test_holdout",
                    help="ImageFolder root for the non-overlapping local test set")
    ap.add_argument("--fake_per_source", type=int, default=1000,
                    help="training images to pull from each AI-generated archive")
    ap.add_argument("--real_per_source", type=int, default=1800,
                    help="training images to pull from each real archive "
                         "(higher default: there are fewer real archives than fake ones)")
    ap.add_argument("--fake_test_per_source", type=int, default=200)
    ap.add_argument("--real_test_per_source", type=int, default=350)
    ap.add_argument("--max_mb_per_source", type=int, default=1200,
                    help="hard byte cap per archive stream, in MB (safety net)")
    ap.add_argument("--revision", default="master")
    ap.add_argument("--only", nargs="+", metavar="TAG",
                    help="restrict to these source tags (e.g. afhq ddim gan)")
    args = ap.parse_args()

    train_root = Path(args.out_train)
    test_root = Path(args.out_test)

    plan = [
        ("FAKE", FAKE_SOURCES, args.fake_per_source, args.fake_test_per_source),
        ("REAL", REAL_SOURCES, args.real_per_source, args.real_test_per_source),
    ]

    totals = {"REAL": [0, 0], "FAKE": [0, 0]}
    for label, sources, n_train, n_test in plan:
        print(f"\n=== {label} ===")
        for repo_path, tag in sources.items():
            if args.only and tag not in args.only:
                continue
            tr, te = harvest_source(
                repo_path, tag,
                train_root / label, test_root / label,
                n_train, n_test, args.max_mb_per_source, args.revision,
            )
            totals[label][0] += tr
            totals[label][1] += te

    print("\n=== done ===")
    print(f"train: {totals['REAL'][0]} REAL + {totals['FAKE'][0]} FAKE  -> {train_root}")
    print(f"holdout: {totals['REAL'][1]} REAL + {totals['FAKE'][1]} FAKE  -> {test_root}")
    print("\nExcluded from download (reserved demo benchmark):")
    for b in sorted(BLOCKED_SOURCES):
        print(f"  - {b}")


if __name__ == "__main__":
    main()
