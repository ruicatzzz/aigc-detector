import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from tqdm import tqdm

from src.model import SmallCNN, _TRANSFORM

def get_dataloaders(data_dir: str, batch_size: int, val_split: float = 0.1):
    dataset = datasets.ImageFolder(data_dir, transform = _TRANSFORM)
    print(f"Classes found: {dataset.class_to_idx}  "
          f"(expect something like FAKE=0, REAL=1 — see docstring)")
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, dataset.class_to_idx

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

def train(data_dir: str, epochs: int, batch_size: int, lr: float, out_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, class_to_idx = get_dataloaders(data_dir, batch_size)
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, epochs+1):
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
        train_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, class_to_idx, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved checkpoint to {out_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train SmallCNN AIGC detector")
    parser.add_argument("--data_dir", required=True, help="ImageFolder-style dir, e.g. data/cifake/train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_sizze", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="checkpoints/cnn_cifake.pt")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.epochs, args.batch_sizze, args.lr, args.out)