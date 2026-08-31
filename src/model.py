import hashlib
import random

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class DummyModel:
    """Placeholder that returns a deterministic pseudo-random score per
    image so demos/tests are reproducible before a real model exists."""

    def predict(self, image: Image.Image) -> float:
        # Hash image bytes for a stable-but-fake confidence score.
        digest = hashlib.md5(image.tobytes()[:4096]).hexdigest()
        rng = random.Random(digest)
        return round(rng.uniform(0.0, 1.0), 4)

class SmallCNN(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

def pick_device():
    """Best available torch device: CUDA > Apple Silicon MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


IMAGE_SIZE = 32
_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ImageNet-pretrained timm backbones want a bigger input and ImageNet stats.
EFFICIENTNET_IMAGE_SIZE = 224
_TRANSFORM_EFFICIENTNET = transforms.Compose([
    transforms.Resize((EFFICIENTNET_IMAGE_SIZE, EFFICIENTNET_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BACKBONES = ("small_cnn", "efficientnet_b0")


def get_transform(backbone: str = "small_cnn"):
    """Preprocessing pipeline matching the given backbone."""
    return _TRANSFORM_EFFICIENTNET if backbone == "efficientnet_b0" else _TRANSFORM


def build_model(backbone: str = "small_cnn", pretrained: bool = True) -> nn.Module:
    """Untrained backbone with a single-logit head (pairs with BCEWithLogitsLoss).

    `pretrained` only matters for timm backbones and only at training time; for
    inference we load our own weights, so it is passed as False there.
    """
    if backbone == "small_cnn":
        return SmallCNN()
    if backbone == "efficientnet_b0":
        try:
            import timm
        except ImportError as exc:
            raise SystemExit("efficientnet_b0 needs timm  ->  pip install timm") from exc
        return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=1)
    raise ValueError(f"unknown backbone {backbone!r}; choose from {BACKBONES}")


class RealModel:
    def __init__(self, checkpoint_path: str):
        self.device = pick_device()
        obj = torch.load(checkpoint_path, map_location=self.device)
        # New checkpoints are {"backbone", "state_dict"}; legacy ones are a bare
        # SmallCNN state_dict.
        if isinstance(obj, dict) and "backbone" in obj and "state_dict" in obj:
            self.backbone, state_dict = obj["backbone"], obj["state_dict"]
        else:
            self.backbone, state_dict = "small_cnn", obj
        self.transform = get_transform(self.backbone)
        self.net = build_model(self.backbone, pretrained=False).to(self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        x = self.transform(image).unsqueeze(0).to(self.device)
        logit = self.net(x)
        prob = torch.sigmoid(logit).item()
        return round(prob, 4)

def load_model(checkpoint_path: str | None = None):
    if checkpoint_path is None:
        return DummyModel()
    return RealModel(checkpoint_path)


FRIEND_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # placeholder — CONFIRM
FRIEND_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)   # placeholder — CONFIRM
MY_MEAN, MY_STD = 0.5, 0.5


class SmallCNNAdapter(nn.Module):
    def __init__(self, small_cnn: SmallCNN):
        super().__init__()
        self.net = small_cnn

    def forward(self, x):
        import torch.nn.functional as F
        device = x.device
        x = x * FRIEND_STD.to(device) + FRIEND_MEAN.to(device)  # undo her normalization -> [0,1]
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        x = (x - MY_MEAN) / MY_STD  # apply your model's own normalization
        logit = self.net(x)  # [N, 1] raw sigmoid logit
        zeros = torch.zeros_like(logit)
        return torch.cat([-logit, zeros], dim=1)  # [N, 2], matches her softmax(...)[:,1] convention


def load_checkpoint(checkpoint_path, backbone=None, device="cpu"):
    net = SmallCNN()
    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict)
    model = SmallCNNAdapter(net).to(device)
    model.eval()
    return model