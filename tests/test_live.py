import time

import pytest
from fastapi.testclient import TestClient

from control_service.app import create_app
from control_service.config import Settings
from control_service.live import LiveHub, live_plan, rows_by_device


@pytest.fixture
def client(tmp_path):
    s = Settings(mock=True, results_dir=tmp_path, frontend_dir=tmp_path / "none", settle_seconds=0.1)
    with TestClient(create_app(s)) as c:
        yield c


def join(c, name, **spec):
    did = c.post("/api/live/join", json={"name": name}).json()["device_id"]
    body = {"requests": 6, "scenario": "flood", "prompt_preset": "short", **spec}
    assert c.post("/api/live/spec", json={"device_id": did, "spec": body}).status_code == 200
    return did


def test_join_gives_unique_names_and_validates(client):
    a = client.post("/api/live/join", json={"name": "Kartik"}).json()
    b = client.post("/api/live/join", json={"name": "kartik"}).json()
    assert a["name"] != b["name"]
    assert client.post("/api/live/join", json={"name": "   "}).status_code in (409, 422)
    assert client.post("/api/live/spec", json={"device_id": "nope", "spec": {}}).status_code == 404
    state = client.get("/api/live/state", params={"device_id": a["device_id"]}).json()
    assert state["joined"] is True and len(state["devices"]) == 2
    assert client.get("/api/live/state", params={"device_id": "nope"}).json()["joined"] is False


def test_spec_validation(client):
    did = client.post("/api/live/join", json={"name": "A"}).json()["device_id"]
    bad = [{"requests": 0}, {"requests": 101}, {"scenario": "weird"}, {"prompt_preset": "custom"}]
    for spec in bad:
        assert client.post("/api/live/spec", json={"device_id": did, "spec": spec}).status_code == 422


def test_start_needs_ready_devices_and_config_is_validated(client):
    assert client.post("/api/live/start").status_code == 422
    join(client, "A")
    assert client.post("/api/live/config", json={"model": "nope"}).status_code == 422
    assert client.post("/api/live/config", json={"policies": ["bogus"]}).status_code == 422
    assert client.post("/api/live/config", json={"policies": ["static", "dynamic", "smart"], "window_s": 0.5}).status_code == 200
    assert client.get("/api/live/state").json()["config"]["policies"] == ["static", "dynamic", "smart"]


def test_device_limit():
    hub = LiveHub()
    for i in range(6):
        hub.join(f"d{i}")
    with pytest.raises(OverflowError):
        hub.join("extra")


def test_live_plan_tags_devices_and_is_reproducible():
    devices = [
        {"name": "A", "spec": {"requests": 5, "scenario": "flood", "prompt_preset": "short"}},
        {"name": "B", "spec": {"requests": 7, "scenario": "mixed", "prompt_preset": "long"}},
    ]
    plan = live_plan(devices, 7)
    assert len(plan) == 12
    assert {i["device"] for i in plan} == {"A", "B"}
    assert [i["arrival_time"] for i in plan] == sorted(i["arrival_time"] for i in plan)
    assert plan == live_plan(devices, 7)
    assert {k: len(v) for k, v in rows_by_device([{"device": i["device"]} for i in plan]).items()} == {"A": 5, "B": 7}


def test_live_run_end_to_end_runs_static_then_dynamic(client):
    join(client, "Kartik", requests=8)
    join(client, "Aditya", requests=6, scenario="mixed", prompt_preset="long")
    res = client.post("/api/live/start")
    assert res.status_code == 202, res.text
    run_id = res.json()["run_id"]
    assert res.json()["total"] == 14
    assert client.post("/api/live/start").status_code == 409
    for _ in range(160):
        state = client.get(f"/api/runs/{run_id}").json()
        if state["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert state["status"] == "completed", state
    results = state["results"]
    assert list(results["results"]) == ["static", "dynamic"]
    assert set(results["per_device"]["static"]) == {"Kartik", "Aditya"}
    assert results["per_device"]["dynamic"]["Kartik"]["offered"] == 8
    assert results["results"]["static"]["offered"] == 14
    assert run_id in [r["id"] for r in client.get("/api/runs").json()]
    assert run_id in [r["id"] for r in client.get("/api/live/runs").json()]
    report = client.get(f"/api/live/runs/{run_id}/report.csv").text.splitlines()
    assert report[0].startswith("policy,device") and any(line.startswith("dynamic,Aditya") for line in report)
    live_state = client.get("/api/live/state").json()
    assert live_state["run"]["id"] == run_id and live_state["busy"] is False
    events = client.get(f"/api/runs/{run_id}/events").text
    assert '"device": "Kartik"' in events and "event: device_result" in events
    assert client.post("/api/live/reset").status_code == 200
    assert all(d["ready"] is False for d in client.get("/api/live/state").json()["devices"])


def test_send_auto_starts_after_window_and_late_phone_joins_same_run(client):
    assert client.post("/api/live/config", json={"window_s": 1.0}).status_code == 200
    a = client.post("/api/live/join", json={"name": "Kartik"}).json()["device_id"]
    b = client.post("/api/live/join", json={"name": "Aditya"}).json()["device_id"]
    spec = {"requests": 4, "scenario": "flood", "prompt_preset": "short"}
    first = client.post("/api/live/send", json={"device_id": a, "spec": spec}).json()
    assert first["armed"] is True and client.get("/api/live/state").json()["countdown_s"] is not None
    client.post("/api/live/send", json={"device_id": b, "spec": spec})
    run_id = None
    for _ in range(40):
        run = client.get("/api/live/state").json()["run"]
        if run:
            run_id = run["id"]
            break
        time.sleep(0.25)
    assert run_id and client.get("/api/live/state").json()["run"]["config"]["requests"] == 8
    assert client.post("/api/live/send", json={"device_id": a, "spec": spec}).status_code == 409
    for _ in range(120):
        if client.get(f"/api/runs/{run_id}").json()["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "completed"


def test_manual_mode_does_not_auto_start(client):
    client.post("/api/live/config", json={"auto_start": False})
    did = client.post("/api/live/join", json={"name": "A"}).json()["device_id"]
    res = client.post("/api/live/send", json={"device_id": did, "spec": {"requests": 3}}).json()
    assert res["armed"] is False
    time.sleep(0.5)
    assert client.get("/api/live/state").json()["run"] is None
