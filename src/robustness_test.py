import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms as T

from src.model import load_model

def jpeg_compress(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def gaussian_blur(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))

def resize_down_up(img, scale):
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return small.resize((w, h))

def gaussian_noise(img, sigma):
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 1) * 255
    return Image.fromarray(noisy.astype(np.uint8))

def color_jitter(img, strength=0.2):
    jitter = T.ColorJitter(brightness=strength, contrast=strength, saturation=strength)
    return jitter(img)

def center_crop_80(img):
    w, h = img.size
    new_w, new_h = int(w * 0.8), int(h * 0.8)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


TRANSFORMS = {
    "clean":            lambda img: img,
    "jpeg_q30":         lambda img: jpeg_compress(img, 30),
    "jpeg_q70":         lambda img: jpeg_compress(img, 70),
    "blur_sigma1.0":    lambda img: gaussian_blur(img, 1.0),
    "blur_sigma2.0":    lambda img: gaussian_blur(img, 2.0),
    "resize_0.5x":      lambda img: resize_down_up(img, 0.5),
    "resize_0.25x":     lambda img: resize_down_up(img, 0.25),
    "noise_sigma0.05":  lambda img: gaussian_noise(img, 0.05),
    "color_jitter":     lambda img: color_jitter(img),
    "center_crop_80":   lambda img: center_crop_80(img),
}


def run_quick_test(test_dir, checkpoint_path, n_per_class=5):
    model = load_model(checkpoint_path)
    test_dir = Path(test_dir)

    for cls in ["REAL", "FAKE"]:
        cls_dir = test_dir / cls
        if not cls_dir.exists():
            print(f"Skipping {cls}, folder not found at {cls_dir}")
            continue
        images = list(cls_dir.glob("*"))
        sample = random.sample(images, min(n_per_class, len(images)))

        print(f"\n=== {cls} images ===")
        for img_path in sample:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                print(f"\n{img_path.name}:")
                for name, fn in TRANSFORMS.items():
                    transformed = fn(img.copy())
                    pred = model.predict(transformed)
                    print(f"  {name:18s} pred={pred:.4f}")


if __name__ == "__main__":
    run_quick_test("data/cifake/test", "checkpoints/cnn_merged.pt", n_per_class=20)