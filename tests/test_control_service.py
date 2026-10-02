import pytest
from fastapi.testclient import TestClient

from control_service.app import create_app
from control_service.config import Settings
from control_service.logparse import classify, is_noise
from control_service.scoring import median_of, score


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
