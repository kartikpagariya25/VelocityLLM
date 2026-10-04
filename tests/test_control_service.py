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
