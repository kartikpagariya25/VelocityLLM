"""The vision benchmark tool (tools/vision_bench): plans, image costs and an end-to-end run on the simulated engine."""
import base64
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

from scheduler_engine import vision

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "vision_bench"))
import bench_core as core  # noqa: E402


def test_plan_is_reproducible_and_uses_the_image_mix():
    a, b = core.make_plan(20, 20, seed=3), core.make_plan(20, 20, seed=3)
    assert [(p["t"], p["side"], p["prompt"]) for p in a] == [(p["t"], p["side"], p["prompt"]) for p in b]
    assert len(a) > 200 and {p["side"] for p in a} == {224, 448, 896}
    share = {s: sum(p["side"] == s for p in a) / len(a) for s in (224, 448, 896)}
    assert 0.40 < share[224] < 0.60 and 0.25 < share[448] < 0.45 and 0.07 < share[896] < 0.25


def test_every_request_has_its_own_picture_of_the_stated_size():
    plan = core.make_plan(10, 10, seed=5)
    assert len({p["image"] for p in plan}) == len(plan)  # no two identical pictures: caches cannot hide the cost
    for p in plan[:12]:
        assert vision.image_size(p["image"]) == (p["side"], p["side"])
        assert p["image_tokens"] == vision.tokens_for_size(p["side"], p["side"]) == vision.estimate_tokens(p["image"])


def test_bigger_pictures_cost_more_vision_tokens():
    assert [vision.tokens_for_size(s, s) for s in (224, 448, 896)] == [64, 256, 1024]


def test_summary_flags_a_server_that_miscounts_vision_tokens():
    rows = [dict(side=224, image_tokens=64, reported_tokens=0, status="served", latency=1.0, on_time=True, outcome="on time", why="")]
    assert core.summarize(rows, 1.0, {})["tokens_ok"] is False
    rows[0]["reported_tokens"] = 64
    assert core.summarize(rows, 1.0, {})["tokens_ok"] is True


@pytest.mark.parametrize("policy", core.POLICIES)
def test_end_to_end_on_the_simulated_vision_engine(policy, monkeypatch):
    monkeypatch.chdir(ROOT)  # `python -m velocityllm` is started from the repository
    result = core.run_policy(policy, rate=6, duration=3, seed=1, warmup_s=0)
    s = result["summary"]
    assert s["offered"] > 5 and s["answered"] > 0
    assert s["tokens_ok"], f"{policy} reported {s['image_tokens_reported']} vision tokens for {s['image_tokens_sent']} sent"
    assert s["server_image_tokens"] >= s["image_tokens_sent"]


def test_server_command_works_with_and_without_the_pip_package(monkeypatch):
    real = core.importlib.util.find_spec

    monkeypatch.setattr(core.importlib.util, "find_spec", lambda name, *a: object() if name == "velocityllm.cli" else real(name, *a))
    assert core.server_command()[1:] == ["-m", "velocityllm", "serve"]

    monkeypatch.setattr(core.importlib.util, "find_spec", lambda name, *a: None if name == "velocityllm.cli" else real(name, *a))
    assert core.server_command()[1:] == ["-m", "scheduler_engine.server"]

    def missing(name, *a):
        if name == "velocityllm.cli":
            raise ModuleNotFoundError("no velocityllm")
        return real(name, *a)

    monkeypatch.setattr(core.importlib.util, "find_spec", missing)
    assert core.server_command()[1:] == ["-m", "scheduler_engine.server"]
