import base64
import io

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from scheduler_engine import vision
from scheduler_engine.server import app, lifespan, set_config
from scheduler_engine.types import InferenceRequest, ServerConfig


def make_image(width, height):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 40, 200)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("size,expected", [((224, 224), 64), ((448, 448), 256), ((896, 896), 1024), ((2000, 2000), 1024)])
def test_token_estimate_follows_image_size(size, expected):
    assert vision.tokens_for_size(*size) == expected


def test_request_derives_image_tokens():
    req = InferenceRequest(prompt="Describe it", image=make_image(448, 448))
    assert req.image_tokens == 256
    assert InferenceRequest(prompt="Describe it").image_tokens == 0


def test_data_uri_is_accepted():
    uri = "data:image/png;base64," + make_image(224, 224)
    assert InferenceRequest(prompt="x", image=uri).image_tokens == 64


@pytest.mark.parametrize("bad", ["not base64 !!", base64.b64encode(b"plain text").decode()])
def test_invalid_image_is_rejected(bad):
    with pytest.raises(ValueError):
        InferenceRequest(prompt="x", image=bad)


async def _client(vision_enabled):
    set_config(ServerConfig(policy="dynamic", use_mock_backend=True, vision=vision_enabled, target_sla_ms=5000.0))
    return lifespan(app)


async def test_vision_server_serves_images():
    async with await _client(True):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post("/generate", json={"prompt": "Describe", "max_tokens": 8, "image": make_image(448, 448)})
            assert res.status_code == 200
            assert res.json()["image_tokens"] == 256
            assert (await client.get("/health")).json()["vision"] is True


async def test_text_only_server_rejects_images():
    async with await _client(False):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post("/generate", json={"prompt": "Describe", "max_tokens": 8, "image": make_image(224, 224)})
            assert res.status_code == 422
            ok = await client.post("/generate", json={"prompt": "Hi", "max_tokens": 8})
            assert ok.status_code == 200
            assert ok.json()["image_tokens"] == 0
