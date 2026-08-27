"""
Training script for the AIGC image detector.

Usage:
    python train.py --data_dir ../data --epochs 5 --augment
    python train.py --data_dir ../data --epochs 3          # baseline, no robustness augmentation

Saves a checkpoint to ../checkpoints/<run_name>.pt
"""

import os
import argparse
import time
import torch
from torch.utils.data import DataLoader

from dataset import AIGCDataset
from model import build_model, count_parameters


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = AIGCDataset(
        root_dir=args.data_dir, split="train",
        use_augmentation=args.augment, image_size=args.image_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    print(f"Loaded {len(train_ds)} training samples (augment={args.augment})")

    model = build_model(backbone=args.backbone, pretrained=True).to(device)
    print(f"Model: {args.backbone} ({count_parameters(model) / 1e6:.1f}M params)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        start = time.time()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        acc = correct / total
        elapsed = time.time() - start
        print(f"Epoch {epoch + 1}/{args.epochs} | loss={avg_loss:.4f} | "
              f"train_acc={acc:.4f} | {elapsed:.1f}s")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    run_name = args.run_name or (f"{args.backbone}_{'augmented' if args.augment else 'baseline'}")
    ckpt_path = os.path.join(args.checkpoint_dir, f"{run_name}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--checkpoint_dir", default="../checkpoints")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--augment", action="store_true",
                         help="Enable robustness augmentation during training")
    parser.add_argument("--run_name", default=None,
                         help="Checkpoint filename (without .pt); auto-generated if omitted")
    args = parser.parse_args()

    train(args)
