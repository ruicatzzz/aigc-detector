"""
Model definition / loading helper.

Keeps backbone choice in one place so train.py, evaluate.py, infer.py, and
gradcam.py all build the identical architecture and never drift out of sync.
"""

import torch
import timm

# Any backbone here must stay under the hackathon's <2B parameter limit.
# resnet50 (~25M params) and efficientnet_b0 (~5M params) are safe defaults.
DEFAULT_BACKBONE = "resnet50"
NUM_CLASSES = 2  # 0 = REAL, 1 = FAKE


def build_model(backbone: str = DEFAULT_BACKBONE, pretrained: bool = True,
                 num_classes: int = NUM_CLASSES):
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    return model


def load_checkpoint(checkpoint_path: str, backbone: str = DEFAULT_BACKBONE,
                     device: str = "cpu"):
    model = build_model(backbone=backbone, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = build_model()
    n_params = count_parameters(m)
    print(f"Backbone: {DEFAULT_BACKBONE}")
    print(f"Total parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    assert n_params < 2_000_000_000, "Model exceeds the 2B parameter hackathon limit!"
