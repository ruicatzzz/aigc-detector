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