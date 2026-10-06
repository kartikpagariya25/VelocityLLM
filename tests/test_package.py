import json
import socket

import pytest

import velocityllm
from velocityllm import bench, cli


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_models(root, names):
    for n in names:
        d = root / n
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"max_position_embeddings": 2048}))


def test_version_and_commands(capsys):
    assert velocityllm.__version__ == "0.4.1"
    assert cli.main(["version"]) == 0
    assert "0.4.1" in capsys.readouterr().out
    assert "bench" in cli.COMMANDS and cli.main(["nope"]) == 2


def test_find_models_filters_and_validates(tmp_path):
    make_models(tmp_path, ["b-model", "a-model", "c-model"])
    (tmp_path / "notes").mkdir()
    assert list(bench.find_models(tmp_path)) == ["a-model", "b-model", "c-model"]
    assert list(bench.find_models(tmp_path, only=["c-model"])) == ["c-model"]
    assert len(bench.find_models(tmp_path, limit=2)) == 2
    with pytest.raises(SystemExit):
        bench.find_models(tmp_path, only=["missing"])


def test_score_percentiles_and_rejections(tmp_path):
    path = tmp_path / "run.csv"
    lines = ["status,client_latency_seconds,tokens_generated"]
    lines += [f"200,{i / 10},10" for i in range(1, 11)]
    lines += ["429,0.1,"]
    path.write_text("\n".join(lines))
    r = bench.score(path, sla_s=0.5)
    assert r["offered"] == 11 and r["served"] == 10 and r["rejected"] == 1
    assert r["p50"] == pytest.approx(0.6) and r["p99"] == pytest.approx(1.0)
    assert r["within_sla_of_served"] == 50 and r["within_sla_of_offered"] == pytest.approx(100 * 5 / 11)


def test_unknown_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        velocityllm.benchmark({"m": str(tmp_path)}, policies=("static", "fast"), out_dir=tmp_path / "o")


def test_mock_benchmark_end_to_end(tmp_path, capsys):
    make_models(tmp_path / "models", ["alpha", "beta"])
    out = tmp_path / "out"
    rc = bench.main([
        "--models-dir", str(tmp_path / "models"), "--requests", "6", "--repeats", "1", "--settle", "0.2",
        "--port", str(free_port()), "--mock", "--out-dir", str(out), "--startup-timeout", "60",
    ])
    assert rc == 0
    rows = json.loads((out / "summary.json").read_text())["rows"]
    assert {(r["model"], r["policy"]) for r in rows} == {
        ("alpha", "static"), ("alpha", "dynamic"), ("beta", "static"), ("beta", "dynamic")}
    assert all(r["served"] > 0 and r["p50"] > 0 for r in rows)
    shown = capsys.readouterr().out
    assert "p95 s" in shown and "vs static" in shown and (out / "summary.csv").exists()


def test_relative_out_dir_is_resolved(tmp_path, monkeypatch):
    make_models(tmp_path / "models", ["alpha"])
    monkeypatch.chdir(tmp_path)
    rows = velocityllm.benchmark({"alpha": str(tmp_path / "models" / "alpha")}, policies=("dynamic",), requests=4,
                                 repeats=1, settle=0.2, port=free_port(), mock=True, quiet=True, out_dir="rel_out")
    assert rows and (tmp_path / "rel_out" / "summary.json").exists()


def test_resolve_models_accepts_names_paths_and_dict(tmp_path):
    make_models(tmp_path, ["alpha", "beta"])
    by_names = bench.resolve_models(["alpha"], str(tmp_path))
    assert list(by_names) == ["alpha"]
    assert list(bench.resolve_models(None, str(tmp_path))) == ["alpha", "beta"]
    assert bench.resolve_models([str(tmp_path / "beta")], None) == {"beta": str(tmp_path / "beta")}
    assert bench.resolve_models({"x": "/p"}, None) == {"x": "/p"}
    with pytest.raises(SystemExit):
        bench.resolve_models(["missing"], str(tmp_path))


def test_benchmark_with_model_names(tmp_path):
    make_models(tmp_path / "models", ["alpha", "beta"])
    rows = velocityllm.benchmark(["alpha"], models_dir=str(tmp_path / "models"), policies=("dynamic",), requests=4,
                                 repeats=1, settle=0.2, port=free_port(), mock=True, quiet=True, out_dir=str(tmp_path / "o"))
    assert {r["model"] for r in rows} == {"alpha"}
