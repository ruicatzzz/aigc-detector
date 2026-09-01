import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from src.model import build_model, get_transform, get_train_transform, pick_device, BACKBONES
from src.augment_transform import RandomRobustnessAugment


def _pick_source_indices(info, cap, balance_classes, rng):
    """Choose which images to keep from one ImageFolder source.

    - groups indices by class and shuffles each group
    - if balance_classes: undersamples the majority class to the minority count
    - interleaves classes round-robin so a later `cap` truncation stays balanced
    - if cap: keeps only the first `cap` after interleaving
    """
    by_class = {}
    for i, t in enumerate(info.targets):
        by_class.setdefault(t, []).append(i)
    for idxs in by_class.values():
        rng.shuffle(idxs)

    if balance_classes:
        k = min(len(v) for v in by_class.values())
        by_class = {c: v[:k] for c, v in by_class.items()}

    pools = list(by_class.values())
    interleaved = []
    for j in range(max(len(v) for v in pools)):
        for v in pools:
            if j < len(v):
                interleaved.append(v[j])

    if cap is not None:
        interleaved = interleaved[:cap]
    return interleaved


def get_dataloaders(data_dirs: list[str], batch_size: int, val_split: float = 0.1, seed: int = 42,
                    augment: bool = True, backbone: str = "small_cnn",
                    cap_per_source: int | None = None, max_total: int | None = None,
                    balance_classes: bool = False):
    # Build one ImageFolder per source dataset, so we can trace which images
    # came from where and verify each uses the same REAL/FAKE label convention.
    class_to_idx = None
    train_folders, val_folders = [], []

    # Validation always uses the deterministic base transform. Training adds
    # RandomRobustnessAugment (redistribution sim) + a crop-based train
    # transform on top; --no_augment falls back to the plain base transform.
    base_transform = get_transform(backbone)
    train_transform = (
        transforms.Compose([RandomRobustnessAugment(p=0.7), get_train_transform(backbone)])
        if augment else base_transform
    )
    print(f"Backbone: {backbone}  |  Train-time augmentation: {'ON' if augment else 'OFF (baseline)'}")

    # Turn a total budget into an even per-source cap, so no single dataset
    # (e.g. CIFAKE's 100k) dominates the merged set.
    if max_total is not None:
        even = max_total // len(data_dirs)
        cap_per_source = even if cap_per_source is None else min(cap_per_source, even)
    if cap_per_source is not None or balance_classes:
        print(f"Sampling: cap_per_source={cap_per_source}  balance_classes={balance_classes}")

    rng = random.Random(seed)
    for data_dir in data_dirs:
        info = datasets.ImageFolder(data_dir)

        if class_to_idx is None:
            class_to_idx = info.class_to_idx
        elif info.class_to_idx != class_to_idx:
            raise ValueError(
                f"Class mapping mismatch: {data_dir} has {info.class_to_idx}, "
                f"expected {class_to_idx}. Every --data_dir needs matching "
                f"REAL/FAKE subfolder names so labels line up after merging."
            )

        sel = _pick_source_indices(info, cap_per_source, balance_classes, rng)
        n_val = int(len(sel) * val_split)
        val_idx, train_idx = sel[:n_val], sel[n_val:]
        print(f"{data_dir}: {len(info)} available -> using {len(sel)} "
              f"(train {len(train_idx)}, val {len(val_idx)}), classes={info.class_to_idx}")

        train_ds = datasets.ImageFolder(data_dir, transform=train_transform)
        val_ds = datasets.ImageFolder(data_dir, transform=base_transform)

        train_folders.append(Subset(train_ds, train_idx))
        val_folders.append(Subset(val_ds, val_idx))

    train_dataset = ConcatDataset(train_folders)
    val_dataset = ConcatDataset(val_folders)
    print(f"Merged: {len(train_dataset)} train, {len(val_dataset)} val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, class_to_idx


def remap_labels(labels: torch.Tensor, class_to_idx: dict) -> torch.Tensor:
    fake_idx = class_to_idx.get("FAKE", class_to_idx.get("fake"))
    if fake_idx is None:
        raise ValueError(f"Expected a 'FAKE' class folder, got: {class_to_idx}")
    return (labels == fake_idx).float()


def evaluate(model, loader, class_to_idx, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = remap_labels(labels, class_to_idx).to(device)
            logits = model(images).squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def train(data_dirs: list[str], epochs: int, batch_size: int, lr: float, weight_decay: float, out_path: str,
          augment: bool = True, backbone: str = "small_cnn", freeze_backbone: bool = False,
          cap_per_source: int | None = None, max_total: int | None = None,
          balance_classes: bool = False):
    device = pick_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_to_idx = get_dataloaders(
        data_dirs, batch_size, augment=augment, backbone=backbone,
        cap_per_source=cap_per_source, max_total=max_total, balance_classes=balance_classes)

    model = build_model(backbone).to(device)

    if freeze_backbone:
        if backbone == "small_cnn":
            print("--freeze_backbone ignored: small_cnn is trained from scratch")
        else:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.get_classifier().parameters():
                p.requires_grad = True
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Backbone frozen — training {n} head params only")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images = images.to(device)
            labels = remap_labels(labels, class_to_idx).to(device)

            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, class_to_idx, device)

        is_best = val_acc > best_val_acc
        marker = "  <- new best, saving" if is_best else ""
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_acc={val_acc:.4f}{marker}")

        if is_best:
            best_val_acc = val_acc
            torch.save({"backbone": backbone, "state_dict": model.state_dict()}, out_path)

    print(f"Training done. Best val_acc={best_val_acc:.4f}, checkpoint saved at {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an AIGC detector (SmallCNN or EfficientNet)")
    parser.add_argument("--data_dir", nargs="+", required=True,
                         help="One or more ImageFolder-style dirs, e.g. data/cifake/train data/sid_subset")
    parser.add_argument("--backbone", default="small_cnn", choices=list(BACKBONES),
                         help="model architecture (default: small_cnn, 32x32 from scratch)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128,
                         help="use ~32 for efficientnet_b0 (224x224 inputs)")
    parser.add_argument("--lr", type=float, default=1e-3,
                         help="use ~1e-4 when fine-tuning a pretrained backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--out", default="checkpoints/cnn_merged.pt")
    parser.add_argument("--no_augment", action="store_true",
                         help="disable train-time robustness augmentation "
                              "(non-augmented baseline for the A/B comparison)")
    parser.add_argument("--freeze_backbone", action="store_true",
                         help="train only the classifier head (pretrained backbones only) — much faster")
    parser.add_argument("--cap_per_source", type=int, default=None,
                         help="max images to take from EACH --data_dir (keeps one dataset "
                              "from dominating, e.g. CIFAKE's 100k)")
    parser.add_argument("--max_total", type=int, default=None,
                         help="overall image budget; split evenly across sources "
                              "(e.g. 150000 over 3 dirs -> 50000 each)")
    parser.add_argument("--balance_classes", action="store_true",
                         help="within each source, undersample the majority class so "
                              "REAL and FAKE counts match")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.weight_decay, args.out,
          augment=not args.no_augment, backbone=args.backbone, freeze_backbone=args.freeze_backbone,
          cap_per_source=args.cap_per_source, max_total=args.max_total,
          balance_classes=args.balance_classes)