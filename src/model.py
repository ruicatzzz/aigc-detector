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

from PIL import Image


class DummyModel:
    """Placeholder that returns a deterministic pseudo-random score per
    image so demos/tests are reproducible before a real model exists."""

    def predict(self, image: Image.Image) -> float:
        # Hash image bytes for a stable-but-fake confidence score.
        digest = hashlib.md5(image.tobytes()[:4096]).hexdigest()
        rng = random.Random(digest)
        return round(rng.uniform(0.0, 1.0), 4)


def load_model(checkpoint_path: str | None = None):
    """
    Person B: replace this to actually load a trained checkpoint
    (e.g. CLIP/DINOv2 backbone + classification head) and return an
    object with a .predict(PIL.Image) -> float method.
    """
    if checkpoint_path is None:
        return DummyModel()
    # TODO(Person B): real checkpoint loading goes here.
    raise NotImplementedError(
        f"No real model loader yet — checkpoint_path={checkpoint_path} "
        "was passed but load_model() only supports the dummy model so far."
    )
