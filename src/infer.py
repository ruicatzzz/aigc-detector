"""
Owned by Person D.

Usage:
    python -m src.infer --input_dir data/test_images --output_json outputs/preds.json
    python -m src.infer --input_dir data/test_images --output_json outputs/preds.json --checkpoint checkpoints/model.pt

Output JSON format (list of records), matching the deliverable spec:
    [
      {"image_path": "data/test_images/img001.jpg", "pred": 0.8421},
      {"image_path": "data/test_images/img002.jpg", "pred": 0.0732},
      ...
    ]
"""

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.model import load_model

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(input_dir: Path):
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def run_inference(input_dir: str, output_json: str, checkpoint: str | None = None):
    input_dir = Path(input_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {input_dir}")

    model = load_model(checkpoint)

    results = []
    for path in tqdm(image_paths, desc="Running inference"):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                pred = model.predict(img)
        except Exception as e:
            print(f"WARNING: failed on {path}: {e}")
            continue
        results.append({"image_path": str(path), "pred": pred})

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} predictions to {output_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="AIGC detector inference")
    parser.add_argument("--input_dir", required=True, help="Directory of images to score")
    parser.add_argument("--output_json", required=True, help="Path to write predictions JSON")
    parser.add_argument("--checkpoint", default=None, help="Path to trained model checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.input_dir, args.output_json, args.checkpoint)
