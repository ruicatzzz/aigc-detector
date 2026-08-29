"""
Grad-CAM explainability for error analysis.

Reads the predictions JSON produced by evaluate.py, finds a few representative
false positives / false negatives, and saves Grad-CAM heatmap overlays showing
what the model attended to — used for the "Error Analysis Note" deliverable
(Section 5.5.5).

Usage:
    python gradcam.py --checkpoint ../checkpoints/resnet50_augmented.pt \
                       --predictions ../outputs/robustness_table_predictions.json \
                       --condition clean --num_examples 3
"""

import os
import argparse
import json
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import load_checkpoint
from dataset import default_transform
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def get_target_layer(model, backbone: str):
    # Works for resnet-family backbones; adjust if you swap architectures.
    if "resnet" in backbone:
        return [model.layer4[-1]]
    raise ValueError(
        f"No target layer configured for backbone '{backbone}'. "
        f"Add a case in get_target_layer() for your architecture."
    )


def find_examples(predictions_path: str, condition: str, num_examples: int):
    with open(predictions_path) as f:
        data = json.load(f)

    if condition not in data:
        raise ValueError(f"Condition '{condition}' not found. Available: {list(data.keys())}")

    entries = data[condition]  # list of [path, label, pred, prob]
    false_positives = [e for e in entries if e[1] == 0 and e[2] == 1]  # REAL flagged as FAKE
    false_negatives = [e for e in entries if e[1] == 1 and e[2] == 0]  # FAKE flagged as REAL
    correct = [e for e in entries if e[1] == e[2]]

    examples = {
        "false_positive": false_positives[:num_examples],
        "false_negative": false_negatives[:num_examples],
        "correct": correct[:num_examples],
    }
    for k, v in examples.items():
        print(f"{k}: found {len(v)} examples")
    return examples


def run_gradcam(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, backbone=args.backbone, device=device)
    target_layers = get_target_layer(model, args.backbone)
    cam = GradCAM(model=model, target_layers=target_layers)

    transform = default_transform(args.image_size)
    examples = find_examples(args.predictions, args.condition, args.num_examples)

    os.makedirs(args.out_dir, exist_ok=True)

    for category, entries in examples.items():
        for i, (path, label, pred, prob) in enumerate(entries):
            img = Image.open(path).convert("RGB")
            img_resized = img.resize((args.image_size, args.image_size))
            rgb_float = np.array(img_resized).astype(np.float32) / 255.0

            input_tensor = transform(img).unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(pred)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  
            overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

            out_name = f"{category}_{i}_label{label}_pred{pred}_prob{prob:.2f}.jpg"
            out_path = os.path.join(args.out_dir, out_name)
            Image.fromarray(overlay).save(out_path)
            print(f"Saved {out_path}")

    print(f"\nDone. Grad-CAM examples saved to {args.out_dir}/")
    print("Use these in the Error Analysis Note — describe what visual regions "
          "the model relies on for correct vs. incorrect predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--predictions", default="../outputs/robustness_table_predictions.json")
    parser.add_argument("--condition", default="clean",
                         help="Which condition from evaluate.py's predictions JSON to draw examples from")
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--out_dir", default="../outputs/error_analysis_examples")
    args = parser.parse_args()

    run_gradcam(args)
