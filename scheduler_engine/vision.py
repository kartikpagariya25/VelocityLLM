"""
VelocityLLM - Vision input helpers
Decodes request images and estimates how many vision tokens a model will spend on them.
"""

import base64
import binascii
import io
import math
from typing import Optional, Tuple

PATCH = 28
MIN_PIXELS = 4 * PATCH * PATCH
MAX_PIXELS = 896 * 896
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _strip_data_uri(data: str) -> str:
    if data.startswith("data:") and "," in data:
        return data.split(",", 1)[1]
    return data


def decode_bytes(data: str) -> bytes:
    try:
        raw = base64.b64decode(_strip_data_uri(data.strip()), validate=True)
    except (binascii.Error, ValueError) as err:
        raise ValueError("image must be valid base64 data") from err
    if not raw:
        raise ValueError("image is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    return raw


def image_size(data: str) -> Tuple[int, int]:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(decode_bytes(data))) as img:
            return img.size
    except ValueError:
        raise
    except Exception as err:
        raise ValueError("image could not be read") from err


def tokens_for_size(width: int, height: int, max_pixels: int = MAX_PIXELS) -> int:
    """Vision tokens after the processor resizes to 28 px multiples within the pixel budget."""
    area = max(1, width * height)
    scale = 1.0
    if area > max_pixels:
        scale = math.sqrt(max_pixels / area)
    elif area < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / area)
    cols = max(1, round(width * scale / PATCH))
    rows = max(1, round(height * scale / PATCH))
    return cols * rows


def estimate_tokens(data: str, max_pixels: int = MAX_PIXELS) -> int:
    width, height = image_size(data)
    return tokens_for_size(width, height, max_pixels)


def load_image(data: str):
    from PIL import Image

    img = Image.open(io.BytesIO(decode_bytes(data)))
    img.load()
    return img.convert("RGB")


def to_base64(img, quality: int = 85, size: Optional[Tuple[int, int]] = None) -> str:
    buf = io.BytesIO()
    out = img.convert("RGB")
    if size:
        out = out.resize(size)
    out.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
