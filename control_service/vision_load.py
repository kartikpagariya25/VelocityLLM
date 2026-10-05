import random
from functools import lru_cache

from PIL import Image

from scheduler_engine import vision

from .config import ROOT

SAMPLES_DIR = ROOT / "assets" / "vision_samples"
IMAGE_MIXES = {
    "small": {"label": "Small", "sizes": ((224, 1.0),), "description": "Thumbnails, about 64 vision tokens each"},
    "mixed": {"label": "Mixed", "sizes": ((224, 0.5), (448, 0.35), (896, 0.15)), "description": "Mostly small images with some large ones"},
    "large": {"label": "Large", "sizes": ((896, 1.0),), "description": "Full-size images, about 1024 vision tokens each"},
}
PROMPTS = (
    "Describe this image in detail.",
    "What is the main subject of this image?",
    "Summarise what this image shows in two sentences.",
    "List the key elements you can see.",
)


@lru_cache(maxsize=1)
def sample_images():
    paths = sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        raise FileNotFoundError(f"No sample images found in {SAMPLES_DIR}")
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return tuple(images)


@lru_cache(maxsize=64)
def encoded(index, side):
    return vision.to_base64(sample_images()[index], size=(side, side))


def attach_images(plan, mix, seed):
    spec = IMAGE_MIXES[mix]
    sides = [side for side, _ in spec["sizes"]]
    weights = [w for _, w in spec["sizes"]]
    rng = random.Random(seed)
    count = len(sample_images())
    for item in plan:
        side = rng.choices(sides, weights)[0]
        item["image"] = encoded(rng.randrange(count), side)
        item["image_side"] = side
        item["image_tokens"] = vision.tokens_for_size(side, side)
        item["prompt"] = rng.choice(PROMPTS)
    return plan
