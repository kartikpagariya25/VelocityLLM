"""Smart must report the vision tokens of image requests the same way static and dynamic do.

Found on a mock vision server: static and dynamic returned image_tokens 64/256/1024 for 224/448/896 px images,
smart returned 0 and /stats showed total_image_tokens 0, so image cost was invisible in smart runs.
"""
import base64
import io

import pytest
from PIL import Image

from scheduler_engine import vision
from scheduler_engine.backend import MockBackend
from scheduler_engine.policy import DynamicBatchPolicy, StaticBatchPolicy
from scheduler_engine.smart_policy import SmartBatchPolicy
from scheduler_engine.types import InferenceRequest, ServerConfig


def picture(side):
    buf = io.BytesIO()
    Image.new("RGB", (side, side), "red").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def cfg(policy):
    return ServerConfig(policy=policy, use_mock_backend=True, vision=True, initial_concurrency=4, min_concurrency=2,
                        max_concurrency=8, target_sla_ms=8000.0, smart={"warmup_probe": False})


@pytest.mark.parametrize("make,name", [(StaticBatchPolicy, "static"), (DynamicBatchPolicy, "dynamic"), (SmartBatchPolicy, "smart")])
async def test_image_tokens_are_reported_by_every_policy(make, name):
    policy = make(MockBackend(tokens_per_second=200.0), cfg(name))
    await policy.initialize()
    try:
        expected = 0
        for side in (224, 448, 896):
            want = vision.tokens_for_size(side, side)
            response = await policy.schedule(InferenceRequest(prompt="Describe it.", max_tokens=8, image=picture(side)))
            assert response.image_tokens == want, f"{name}: response reported {response.image_tokens}, expected {want}"
            expected += want
        stats = policy.get_stats()
        assert stats.total_image_tokens == expected
        assert stats.active_image_tokens == 0  # nothing is running any more
    finally:
        await policy.shutdown()


async def test_active_image_tokens_are_counted_while_a_smart_request_runs():
    import asyncio

    policy = SmartBatchPolicy(MockBackend(tokens_per_second=20.0), cfg("smart"))
    await policy.initialize()
    try:
        task = asyncio.create_task(policy.schedule(InferenceRequest(prompt="Describe it.", max_tokens=20, image=picture(448))))
        seen = 0
        for _ in range(40):
            await asyncio.sleep(0.05)
            seen = max(seen, policy.get_stats().active_image_tokens)
            if task.done():
                break
        await task
        assert seen == vision.tokens_for_size(448, 448)
        assert policy.get_stats().active_image_tokens == 0
    finally:
        await policy.shutdown()
