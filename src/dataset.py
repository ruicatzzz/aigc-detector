"""
Dataset class for AIGC detection.

Expects a folder layout like:
    data/
        train/REAL/*.jpg
        train/FAKE/*.jpg
        test/REAL/*.jpg
        test/FAKE/*.jpg

This matches the CIFAKE Kaggle dataset layout. Label convention: 0 = REAL, 1 = FAKE.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from augmentations import random_transform

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASSES = ["REAL", "FAKE"]  # index 0, 1


def default_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class AIGCDataset(Dataset):
    """
    root_dir: path to the dataset root (e.g. "data")
    split: "train" or "test" — expects root_dir/split/REAL and root_dir/split/FAKE
    use_augmentation: if True, applies random_transform (robustness augmentation)
        BEFORE the standard resize/normalize pipeline. Only use this for training.
    """

    def __init__(self, root_dir: str, split: str = "train",
                 use_augmentation: bool = False, image_size: int = 224,
                 transform=None):
        self.samples = []
        self.use_augmentation = use_augmentation
        self.transform = transform or default_transform(image_size)

        for label, cls in enumerate(CLASSES):
            cls_dir = os.path.join(root_dir, split, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(
                    f"Expected folder not found: {cls_dir}. "
                    f"Check your data/ layout matches root_dir/split/{{REAL,FAKE}}."
                )
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root_dir}/{split}. Did the dataset download correctly?")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.use_augmentation:
            img = random_transform(img, p=0.5)

        img = self.transform(img)
        return img, label


if __name__ == "__main__":
    # Quick smoke test — run from src/ with data/ present two levels up, e.g.:
    #   cd src && python dataset.py
    ds = AIGCDataset(root_dir="../data", split="train", use_augmentation=True)
    print(f"Loaded {len(ds)} training samples")
    x, y = ds[0]
    print(f"Sample 0: tensor shape {x.shape}, label {y}")
