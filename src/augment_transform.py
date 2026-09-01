"""
Standalone train-time augmentation wrapper — applies a random subset of the
problem statement's transforms during training so the model sees degraded
images, not just clean ones. This mirrors what Person A's augmentations.py
will formalize later; feel free to swap this out once that's ready.
"""
import io
import random

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms as T


def _jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def _blur(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))

def _resize_down_up(img, scale):
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return small.resize((w, h))

def _noise(img, sigma):
    arr = np.array(img).astype(np.float32) / 255.0
    noisy = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 1) * 255
    return Image.fromarray(noisy.astype(np.uint8))

def _color_jitter(img):
    return T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(img)

def _center_crop_80(img):
    w, h = img.size
    new_w, new_h = int(w * 0.8), int(h * 0.8)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h)).resize((w, h))

def _rotate(img):
    # small rotations kill orientation/semantic shortcuts (SAFE, KDD 2025)
    return img.rotate(random.uniform(-12, 12), resample=Image.BILINEAR, fillcolor=(0, 0, 0))

def _hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def _random_crop_80(img):
    # like center_crop_80 but off-centre — a random 80% window
    w, h = img.size
    new_w, new_h = int(w * 0.8), int(h * 0.8)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    return img.crop((left, top, left + new_w, top + new_h)).resize((w, h))

BLUR_SIGMAS = [0.5, 1.0, 2.0]
BLUR_WEIGHTS = [0.2, 0.3, 0.5]

RESIZE_SCALES = [0.5, 0.25]
RESIZE_WEIGHTS = [0.4, 0.6]

NOISE_SIGMAS = [0.02, 0.05, 0.10]
JPEG_QUALITIES = [30, 50, 70, 90]


AUGMENTATIONS = [
    lambda img: _jpeg(img, random.choice(JPEG_QUALITIES)),
    lambda img: _blur(img, random.choices(BLUR_SIGMAS, weights=BLUR_WEIGHTS, k=1)[0]),
    lambda img: _resize_down_up(img, random.choices(RESIZE_SCALES, weights=RESIZE_WEIGHTS, k=1)[0]),
    lambda img: _noise(img, random.choice(NOISE_SIGMAS)),
    lambda img: _color_jitter(img),
    lambda img: _center_crop_80(img),
    lambda img: _random_crop_80(img),
    lambda img: _rotate(img),
    lambda img: _hflip(img),
]


class RandomRobustnessAugment:
    """Applies 0..max_ops random transforms per call.

    Real redistribution stacks operations (compress -> resize -> re-compress),
    so with probability `p` we apply a chain of 1..max_ops transforms rather
    than just one. `max_ops=1` reproduces the original single-transform
    behaviour.
    """

    def __init__(self, p=0.7, max_ops=2):
        self.p = p
        self.max_ops = max_ops

    def __call__(self, img):
        if random.random() < self.p:
            k = random.randint(1, self.max_ops)
            for fn in random.sample(AUGMENTATIONS, k):
                img = fn(img)
        return img