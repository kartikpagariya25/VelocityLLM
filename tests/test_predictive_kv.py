import asyncio
import json

import pytest

from scheduler_engine.backend import MockBackend
from scheduler_engine.estimators import KVEstimator, OutputLengthPredictor
from scheduler_engine.kv_model import derive_footprint, kv_bytes_from_config
from scheduler_engine.smart_config import SmartConfig
from scheduler_engine.smart_policy import SmartBatchPolicy
from scheduler_engine.types import InferenceRequest, ServerConfig


def test_kv_bytes_from_config_llama_and_nested_vision():
    llama = {"num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 2048}
    assert kv_bytes_from_config(llama) == 2 * 16 * 8 * 64 * 2
    vl = {"text_config": {"num_hidden_layers": 36, "num_attention_heads": 16, "num_key_value_heads": 2, "hidden_size": 2048}}
    assert kv_bytes_from_config(vl) == 2 * 36 * 2 * 128 * 2
    assert kv_bytes_from_config({}) is None


def test_derive_footprint_reads_config_and_weight_files(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(
        {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 2, "hidden_size": 1536}))
    (tmp_path / "model.safetensors").write_bytes(b"0" * (10 * 1024 * 1024))
    fp = derive_footprint(str(tmp_path), overhead_mb=100)
    assert fp.kv_bytes_per_token == 2 * 28 * 2 * 96 * 2 and fp.weights_mb == 110
    assert derive_footprint(str(tmp_path / "missing")) is None


def test_engine_capacity_beats_formula():
    cfg = ServerConfig(policy="smart", use_mock_backend=True)
    smart = SmartConfig()
    est = KVEstimator(cfg, smart, OutputLengthPredictor())
    formula = est.capacity_tokens(8192)
    est.engine_capacity_tokens = 12345
    assert est.capacity_tokens(8192) == 12345 != formula
    smart.kv_capacity_tokens_override = 999
    assert est.capacity_tokens(8192) == 999


class CountingBackend(MockBackend):
    def pop_token_count(self, request_id):
        return 37

    def kv_capacity_tokens(self):
        return 50000


def _cfg(**smart):
    return ServerConfig(policy="smart", use_mock_backend=True, initial_concurrency=2, min_concurrency=2,
                        max_concurrency=16, target_sla_ms=5000.0, max_queue_size=200, smart=smart)


@pytest.mark.asyncio
async def test_real_token_count_feeds_predictor_and_engine_capacity_is_used():
    policy = SmartBatchPolicy(CountingBackend(tokens_per_second=400, simulated_ttft_seconds=0.01), _cfg())
    await policy.initialize()
    assert policy.kv_estimator.source == "engine" and policy.kv_estimator.capacity_tokens(8192) == 50000
    try:
        res = await policy.schedule(InferenceRequest(prompt="hello", max_tokens=8))
        assert res.tokens_generated == 37
    finally:
        await policy.shutdown()


@pytest.mark.asyncio
async def test_predictive_ramp_reaches_selected_capacity_faster_than_aimd():
    async def peak(predictive: bool) -> int:
        policy = SmartBatchPolicy(MockBackend(tokens_per_second=300, simulated_ttft_seconds=0.02),
                                  _cfg(predictive_ramp=predictive, aimd_ramp_step=1))
        await policy.initialize()
        try:
            tasks = [asyncio.create_task(policy.schedule(InferenceRequest(prompt="x " * 20, max_tokens=60)))
                     for _ in range(40)]
            top = 0
            for _ in range(60):
                await asyncio.sleep(0.02)
                top = max(top, len(policy._active_requests))
            await asyncio.gather(*tasks, return_exceptions=True)
            return top
        finally:
            await policy.shutdown()

    assert await peak(True) > await peak(False)
