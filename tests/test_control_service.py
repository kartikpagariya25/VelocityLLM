import pytest
from fastapi.testclient import TestClient

from control_service.app import create_app
from control_service.config import Settings
from control_service.logparse import classify, is_noise
from control_service.scoring import diagnose, median_of, score


def row(status, arrival, end, tokens=10):
    return {"status": status, "arrival_s": arrival, "end_s": end, "tokens": tokens}


def test_classify_admission_and_controller():
    _, cat, rid, _ = classify("2026-10-02 10:00:00 [INFO] [cid=-] velocityllm.admission: Request r-3 rejected: Queue full (100/100). Retry after 1.00s")
    assert cat == "admission" and rid == "r-3"
    _, cat, _, _ = classify("2026-10-02 10:00:00 [INFO] [cid=-] velocityllm.controller: Adaptive Batch Controller: concurrency 8 -> 6 (x)")
    assert cat == "controller"
    assert classify("INFO:     Finished server process [1]")[1] == "system"
    assert classify("[ERROR] boom")[1] == "error"


def test_noise_filter():
    assert is_noise('127.0.0.1:1 - "GET /stats HTTP/1.1" 200 OK')
    assert not is_noise('127.0.0.1:1 - "POST /generate HTTP/1.1" 200 OK')


def test_score_counts_and_sla():
    rows = [row("served", 0, 1), row("served", 0, 9), row("rejected", 0, 0.1, 0), row("error", 0, 2, 0)]
    r = score(rows, 8.0, 10.0)
    assert (r["served"], r["offered"], r["rejected"], r["errors"]) == (2, 4, 1, 1)
    assert r["within_sla_served"] == 0.5 and r["within_sla_offered"] == 0.25
    assert r["tokens_per_s"] == pytest.approx(2.0)
    assert r["goodput_tokens_per_s"] == pytest.approx(1.0)


def test_median_of_repeats():
    a, b, c = (score([row("served", 0, v)], 8.0, 1.0) for v in (1, 5, 3))
    assert median_of([a, b, c])["p99_s"] == 3


@pytest.fixture
def client(tmp_path):
    s = Settings(mock=True, results_dir=tmp_path, frontend_dir=tmp_path / "none", settle_seconds=0.1)
    with TestClient(create_app(s)) as c:
        yield c


def test_validation_errors(client):
    assert client.post("/api/runs", json={"model": "nope"}).status_code == 422
    assert client.post("/api/runs", json={"model": "mock-model", "requests": 0}).status_code == 422
    assert client.post("/api/runs", json={"model": "mock-model", "mode": "side_by_side"}).status_code == 422
    assert client.post("/api/runs", json={"model": "mock-model", "prompt_preset": "custom"}).status_code == 422
    assert client.get("/api/runs/missing").status_code == 404


def test_models_and_preflight(client):
    assert client.get("/api/models").json()[0]["id"] == "mock-model"
    assert client.get("/api/preflight").json()["ok"] is True


def test_levels_validation(client):
    base = {"model": "mock-model"}
    assert client.post("/api/runs", json={**base, "levels": [0, 10]}).status_code == 422
    assert client.post("/api/runs", json={**base, "levels": [10, 500]}).status_code == 422
    assert client.post("/api/runs", json={**base, "levels": list(range(1, 11))}).status_code == 422


def test_recording_and_export_need_completed_run(client):
    assert client.get("/api/benchmarks/export").status_code == 404
    assert client.get("/api/runs/missing/recording").status_code == 404


def test_diagnose_flags_invalid_results():
    from control_service.scoring import diagnose

    base = {"offered": 100, "served": 100, "rejected": 0, "errors": 0, "zero_token_share": 0.0}
    assert diagnose(base) == []
    assert "No request was answered" in diagnose({**base, "served": 0, "rejected": 100})[0]
    assert "mis-calibrated" in diagnose({**base, "served": 5, "rejected": 95})[0]
    assert "failed" in diagnose({**base, "errors": 40})[0]
    assert "zero tokens" in diagnose({**base, "zero_token_share": 0.9})[0]


def test_score_reports_total_tokens_and_requests_use_greedy_sampling():
    from control_service.scoring import score

    rows = [
        {"status": "served", "arrival_s": 0.0, "end_s": 1.0, "tokens": 10},
        {"status": "served", "arrival_s": 0.0, "end_s": 2.0, "tokens": 30},
    ]
    assert score(rows, 8.0, 2.0)["tokens"] == 40
    import inspect
    from control_service import loadgen

    assert '"temperature": 0.0' in inspect.getsource(loadgen)


def test_frontend_cache_headers(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<html></html>")
    (tmp_path / "assets" / "app.js").write_text("x")
    c = TestClient(create_app(Settings(mock=True, frontend_dir=tmp_path)))
    assert c.get("/").headers["cache-control"] == "no-cache"
    assert "immutable" in c.get("/assets/app.js").headers["cache-control"]


def test_live_adapter_forwards_sweep_events():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "frontend" / "src" / "lab" / "live.js").read_text()
    assert "'level_result'" in src


def test_memory_declines_are_counted_and_explained():
    rows = [row("served", 0, 1)] + [
        {"status": "rejected", "arrival_s": 0, "end_s": 0, "tokens": 0, "reject_reason": "memory"} for _ in range(4)
    ]
    result = score(rows, 5.0, 2.0)
    assert result["memory_rejected"] == 4
    assert any("GPU memory" in note for note in diagnose(result))


# ---- Smart policy in the Arena -------------------------------------------------------------
def test_smart_is_one_of_the_compared_policies():
    from control_service.runner import POLICIES

    assert POLICIES == ("static", "dynamic", "smart")


def test_smart_settings_come_from_the_model_folder(tmp_path):
    import json

    from control_service.smartcfg import OVERHEAD_MB, smart_settings, write_smart_config

    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2, "hidden_size": 896}))
    (tmp_path / "model.safetensors").write_bytes(b"\0" * (50 * 1024 * 1024))
    st = smart_settings(str(tmp_path))
    assert st == {"kv_bytes_per_token": 2 * 24 * 2 * 64 * 2, "model_weights_mb": 50 + OVERHEAD_MB}
    path, written = write_smart_config(str(tmp_path), tmp_path)
    assert written == st and "kv_bytes_per_token: 12288" in path.read_text()
    assert smart_settings(str(tmp_path / "missing")) is None


def test_smart_settings_handle_nested_and_explicit_head_dim(tmp_path):
    import json

    from control_service.smartcfg import kv_bytes_per_token

    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "hidden_size": 1024, "head_dim": 128}))
    assert kv_bytes_per_token(tmp_path) == 2 * 28 * 8 * 128 * 2
    (tmp_path / "config.json").write_text("not json")
    assert kv_bytes_per_token(tmp_path) is None


def test_engine_passes_the_smart_config_only_when_given(tmp_path):
    from control_service.engine import Engine

    s = Settings(mock=True, results_dir=tmp_path)
    plain = Engine(s, "dynamic", "mock", 8000, 8100, True, None).command()
    assert "--config" not in plain
    smart = Engine(s, "smart", "mock", 8000, 8100, True, None, config_path=tmp_path / "smart_config.yaml").command()
    assert smart[smart.index("--config") + 1].endswith("smart_config.yaml") and "smart" in smart


def test_reject_reasons_use_the_smart_reason_code():
    from control_service.loadgen import reject_reason

    assert reject_reason({"reason_code": "memory_risk", "reason": "Predicted future KV demand 120%"}) == "memory"
    assert reject_reason({"reason_code": "sla_risk", "reason": "x"}) == "sla_impossible"
    assert reject_reason({"reason_code": "policy_limit", "reason": "x"}) == "policy_limit"
    assert reject_reason({"reason_code": "queue_overload", "reason": "Queue full (100/100)."}) == "queue_full"
    assert reject_reason({"reason": "Predicted latency (9s) exceeds target SLA"}) == "sla_impossible"  # Static/Dynamic bodies unchanged


def test_smart_trace_events_become_console_lines():
    from control_service.runner import trace_line

    level, cat, rid, msg = trace_line({"event": "reject", "request_id": "r-4", "reason_code": "sla_risk", "reason": "Predicted 9s exceeds SLA 8s."})
    assert (level, cat, rid) == ("WARNING", "admission", "r-4") and "[sla_risk]" in msg and "rejected" in msg
    assert trace_line({"event": "capacity_change", "reason": "Capacity 8 -> 10"})[1] == "controller"
    assert trace_line({"event": "complete", "request_id": "r-1", "reason": "Within SLA.", "details": {"actual_latency_ms": 1500, "tokens": 40}})[3].startswith("Request r-1 completed in 1.50s")
    assert is_noise('127.0.0.1:1 - "GET /smart/trace?limit=1000&since_seq=3 HTTP/1.1" 200 OK')


def test_models_report_their_weight_size(tmp_path):
    from control_service.models import describe

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"\0" * (3 * 1024 * 1024))
    assert describe("m", str(tmp_path), False)["size_mb"] == 3
    assert describe("mock-model", "mock", True)["size_mb"] is None
