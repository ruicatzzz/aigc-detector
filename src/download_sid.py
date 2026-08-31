"""
Download a bounded subset of SID_Set (Hugging Face: ``saberzl/SID_Set``)
into ImageFolder-style ``REAL`` / ``FAKE`` directories, without pulling the
full ~140 GB.

SID_Set ships a Hugging Face streaming loader, so this is simple: iterate
the stream once, write the first ``--n_train`` examples as the training
subset and the next ``--n_test`` as a non-overlapping held-out test set
(the iterator is shared, so the two never share an image).

SID_Set labels are 3-way: ``0`` = real, ``1`` = fully synthetic,
``2`` = tampered. Per the task, tampered images count as FAKE, so anything
``!= 0`` is written under ``FAKE/``.

Output layout (matches ``src/train.py`` and the eval scripts):

    data/sid_subset/{REAL,FAKE}         <- training subset
    data/sid_test_holdout/{REAL,FAKE}   <- non-overlapping local test set

Usage
-----
    pip install datasets pillow
    python -m src.download_sid                       # 10k train / 2k holdout
    python -m src.download_sid --n_train 4000 --n_test 1000

This reproduces (and extends) the SID cell in ``notebooks/training.ipynb``:
that cell only builds the holdout set; this script builds both slices so a
local ``src.train`` run has ``data/sid_subset`` to point at.
"""

import argparse
from itertools import islice
from pathlib import Path


def save_split(examples, out_dir: Path, start_index: int = 0) -> tuple[int, int]:
    """Write an iterable of SID_Set examples into ``out_dir/{REAL,FAKE}``.

    ``start_index`` only seeds the fallback filename when an example has no
    ``img_id`` field; it does not affect which examples are consumed.
    """
    out_dir = Path(out_dir)
    (out_dir / "REAL").mkdir(parents=True, exist_ok=True)
    (out_dir / "FAKE").mkdir(parents=True, exist_ok=True)

    real = fake = 0
    for i, ex in enumerate(examples):
        label = ex["label"]
        img = ex["image"]
        img_id = ex.get("img_id") or f"sid{start_index + i:07d}"
        cls = "REAL" if label == 0 else "FAKE"
        try:
            img.convert("RGB").save(out_dir / cls / f"{img_id}.jpg", format="JPEG", quality=95)
        except Exception as exc:
            print(f"    skipped {img_id}: {exc}")
            continue
        if label == 0:
            real += 1
        else:
            fake += 1

    print(f"  {out_dir}: {real} REAL, {fake} FAKE")
    return real, fake


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out_train", default="data/sid_subset")
    ap.add_argument("--out_test", default="data/sid_test_holdout")
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--split", default="train")
    ap.add_argument("--hf_id", default="saberzl/SID_Set")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("`datasets` is required: pip install datasets") from exc

    print(f"Streaming {args.hf_id} (split={args.split})...")
    stream = iter(load_dataset(args.hf_id, split=args.split, streaming=True))

    print("training subset:")
    save_split(islice(stream, args.n_train), Path(args.out_train), start_index=0)
    # `stream` is now positioned past the training slice, so the holdout
    # below is drawn from images the training subset never saw.
    print("held-out test set:")
    save_split(islice(stream, args.n_test), Path(args.out_test), start_index=args.n_train)


if __name__ == "__main__":
    main()
