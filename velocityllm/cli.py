"""The `velocityllm` command. A thin wrapper: it hands the arguments to the existing programs unchanged.

    velocityllm serve ...     scheduler_engine.server   (the API server and its policies)
    velocityllm arena ...     control_service           (the benchmark service behind the Arena page)
    velocityllm loadgen ...   load_generator.generate_load (send test traffic to a running server)
    velocityllm bench ...     velocityllm.bench (compare policies on one or more models, p50/p95/p99 and tokens/s)
"""
import importlib.util
import os
import sys
from pathlib import Path

from . import __version__

USAGE = f"""velocityllm {__version__}

usage: velocityllm <command> [options]

commands:
  chargeVelocity  Start VelocityLLM: detects your GPU and model automatically, then serves
  serve           Run the scheduler server (static, dynamic ...); add --mock to run without a GPU
  detect          Show the GPU and model that VelocityLLM would use
  arena           Run the benchmark service used by the Arena dashboard
  loadgen         Send test traffic to a running server
  bench           Compare static and dynamic scheduling on your models (p50, p95, p99, tokens/s)
  version         Show the version

Run 'velocityllm <command> --help' for the options of each command.
Real models need an NVIDIA GPU and: pip install "velocityllm[gpu]"
"""


def _package_dir(name):
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"velocityllm: the '{name}' package is missing from this install")
    return Path(list(spec.submodule_search_locations)[0])


def _has_option(args, option):
    return any(a == option or a.startswith(option + "=") for a in args)


def arena_args(args, cwd=None):
    """Models and results default to the folder you run the command in, not the install folder."""
    here = Path(cwd) if cwd else Path(os.getcwd())
    out = list(args)
    if not _has_option(out, "--models-dir"):
        out += ["--models-dir", str(here / "models")]
    if not _has_option(out, "--results-dir"):
        out += ["--results-dir", str(here / "velocity_runs")]
    return out


def _serve():
    from scheduler_engine.server import main

    return main()


def _chargeVelocity():
    from scheduler_engine.hardware import detect_gpu
    gpu = detect_gpu()
    print("=== chargeVelocity ===")
    print(f"GPU: {gpu['name']} ({gpu['memory_mb']} MB)" if gpu else "GPU: none found, simulated engine will be used")
    return _serve()


def _detect():
    from scheduler_engine.hardware import detect_gpu, find_local_model, default_search_dirs, MODEL_ENV
    gpu = detect_gpu()
    print(f"GPU:   {gpu['name']} ({gpu['memory_mb']} MB)" if gpu else "GPU:   none found (simulated engine)")
    env = os.environ.get(MODEL_ENV)
    found = find_local_model(default_search_dirs())
    print(f"Model: {env or found or 'not found (pass --model-path, set ' + MODEL_ENV + ', or use the terminal prompt)'}")
    return 0


def _arena():
    from control_service.__main__ import main

    return main()


def _loadgen():
    # the load generator was written as flat scripts, so its own folders must be importable
    for name in ("load_generator", "scheduler_engine"):
        sys.path.insert(0, str(_package_dir(name)))
    from generate_load import main

    return main()


def _bench():
    from velocityllm.bench import main

    return main()


COMMANDS = {
    "chargeVelocity": _chargeVelocity, "serve": _serve, "detect": _detect, "arena": _arena,
    "loadgen": _loadgen, "bench": _bench,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if argv[0] in ("-V", "--version", "version"):
        print(f"velocityllm {__version__}")
        return 0
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"velocityllm: unknown command '{command}'\n\n{USAGE}", file=sys.stderr)
        return 2
    sys.argv = [f"velocityllm {command}", *(arena_args(rest) if command == "arena" else rest)]
    try:
        COMMANDS[command]()
    except SystemExit as exc:
        code = exc.code
        return 0 if code is None else code if isinstance(code, int) else 1
    return 0
