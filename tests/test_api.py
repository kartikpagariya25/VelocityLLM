"""
API Integration Tests - FastAPI Service Endpoints
"""

import pytest
from httpx import ASGITransport, AsyncClient

from scheduler_engine.server import app, set_config
from scheduler_engine.types import ServerConfig


@pytest.fixture(autouse=True)
async def configure_test_server():
    cfg = ServerConfig(
        policy="dynamic",
        use_mock_backend=True,
        target_sla_ms=5000.0,
        initial_concurrency=4,
    )
    set_config(cfg)
    from scheduler_engine.server import lifespan
    async with lifespan(app):
        yield


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["policy"] == "dynamic"


@pytest.mark.asyncio
async def test_generate_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {"prompt": "What is 2+2?", "max_tokens": 10, "priority": 1}
        res = await client.post("/generate", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "response" in data
        assert data["prompt"] == "What is 2+2?"
        assert data["latency_seconds"] > 0
        assert data["tokens_generated"] > 0


@pytest.mark.asyncio
async def test_v1_completions_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {"prompt": "Tell me a joke", "max_tokens": 10, "priority": "high"}
        res = await client.post("/v1/completions", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) > 0
        assert "usage" in data


@pytest.mark.asyncio
async def test_stats_and_metrics_endpoints():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Check /stats
        stats_res = await client.get("/stats")
        assert stats_res.status_code == 200
        stats_data = stats_res.json()
        assert "policy_name" in stats_data
        assert "effective_concurrency_limit" in stats_data
        assert "uptime_seconds" in stats_data

        # Check /metrics
        metrics_res = await client.get("/metrics")
        assert metrics_res.status_code == 200
        assert "velocityllm_active_requests" in metrics_res.text
        assert "velocityllm_total_completed" in metrics_res.text
