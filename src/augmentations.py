"""
Augmentation / transform functions used both for:
1) Training-time robustness augmentation (applied randomly during training)
2) Building the fixed transformed test set (applied deterministically for eval)

Each function takes a PIL.Image and returns a PIL.Image.
Parameters match the ranges given in the problem statement (Section 5.2).
"""

import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    """quality in {90, 70, 50, 30} — lower quality = more compression artifacts."""
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """sigma in {0.5, 1.0, 2.0} — simulates out-of-focus capture."""
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_resize(img: Image.Image, scale: float) -> Image.Image:
    """scale in {0.5, 0.25} — downscale then upscale back, simulating thumbnailing."""
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def apply_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """sigma in {0.02, 0.05, 0.10} — relative to [0,1] pixel range, simulates sensor noise."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def apply_color_jitter(img: Image.Image, brightness=0.2, contrast=0.2, saturation=0.2) -> Image.Image:
    """+/- fraction jitter on brightness/contrast/saturation, simulates filter apps."""
    out = img.convert("RGB")
    out = ImageEnhance.Brightness(out).enhance(1 + np.random.uniform(-brightness, brightness))
    out = ImageEnhance.Contrast(out).enhance(1 + np.random.uniform(-contrast, contrast))
    out = ImageEnhance.Color(out).enhance(1 + np.random.uniform(-saturation, saturation))
    return out


def apply_center_crop(img: Image.Image, crop_fraction: float = 0.8) -> Image.Image:
    """Crop crop_fraction of width/height from the center, then resize back to original size."""
    w, h = img.size
    new_w, new_h = int(w * crop_fraction), int(h * crop_fraction)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)


# Registry so train.py / evaluate.py / build_test_transforms.py can loop generically.
# Keys are used as folder / column names downstream — keep them stable.
TRANSFORMS = {
    "jpeg_90": lambda img: apply_jpeg_compression(img, 90),
    "jpeg_70": lambda img: apply_jpeg_compression(img, 70),
    "jpeg_50": lambda img: apply_jpeg_compression(img, 50),
    "jpeg_30": lambda img: apply_jpeg_compression(img, 30),
    "blur_0.5": lambda img: apply_gaussian_blur(img, 0.5),
    "blur_1.0": lambda img: apply_gaussian_blur(img, 1.0),
    "blur_2.0": lambda img: apply_gaussian_blur(img, 2.0),
    "resize_0.5": lambda img: apply_resize(img, 0.5),
    "resize_0.25": lambda img: apply_resize(img, 0.25),
    "noise_0.02": lambda img: apply_gaussian_noise(img, 0.02),
    "noise_0.05": lambda img: apply_gaussian_noise(img, 0.05),
    "noise_0.10": lambda img: apply_gaussian_noise(img, 0.10),
    "color_jitter": lambda img: apply_color_jitter(img),
    "center_crop": lambda img: apply_center_crop(img, 0.8),
}


def random_transform(img: Image.Image, p: float = 0.5) -> Image.Image:
    """
    Used during TRAINING as an augmentation step: with probability p, apply one
    randomly chosen transform from TRANSFORMS; otherwise return the image unchanged.
    This is different from build_test_transforms.py, which applies every transform
    deterministically to build a fixed evaluation set.
    """
    if np.random.rand() > p:
        return img
    name = np.random.choice(list(TRANSFORMS.keys()))
    try:
        return TRANSFORMS[name](img)
    except Exception:
        # Fail safe — never let a bad augmentation crash a training step
        return img


if __name__ == "__main__":
    # Quick manual sanity check: point this at any single test image.
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python augmentations.py <path_to_image> [out_dir]")
        sys.exit(0)

    img_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "test_outputs"
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    for name, fn in TRANSFORMS.items():
        out = fn(img)
        out.save(os.path.join(out_dir, f"{name}.jpg"))
        print(f"{name}: size={out.size}")
    print(f"Saved {len(TRANSFORMS)} transformed images to {out_dir}/")
