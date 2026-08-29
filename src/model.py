"""
Owned by Person B.

CONTRACT (do not change the signatures below — infer.py, evaluate.py, and
gradcam.py are all written against this interface):

    load_model(checkpoint_path: str | None) -> model
    model.predict(image: PIL.Image.Image) -> float   # in [0, 1], P(AIGC)

Until a real checkpoint exists, DummyModel below lets everyone else's code
run end-to-end today. Person B: replace DummyModel's internals (or add a
new class) but keep load_model()'s return object exposing .predict().
"""

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
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),  n.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2s(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x - torch.flatten(x, 1)
        return self.classifier(x)

IMAGE_SIZE = 32
_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

class RealModel:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = SmallCNN().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        x = _TRANSFORM(image).unsqueeze(0).to(self.device)
        logit = self.net(x)
        prob = torch.sigmoid(logit).item()
        return round(prob, 4)

def load_model(checkpoint_path: str | None = None):
    """
    Person B: replace this to actually load a trained checkpoint
    (e.g. CLIP/DINOv2 backbone + classification head) and return an
    object with a .predict(PIL.Image) -> float method.
    """
    if checkpoint_path is None:
        return DummyModel()
    return RealModel(checkpoint_path)
