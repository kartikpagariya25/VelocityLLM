import argparse
import sys
from pathlib import Path

import uvicorn

from .app import create_app
from .config import Settings


def parse_models(items):
    out = {}
    for item in items:
        name, _, path = item.partition("=")
        if not name or not path:
            sys.exit(f"--model expects NAME=PATH, got: {item}")
        out[name] = path
    return out


def main():
    p = argparse.ArgumentParser(description="VelocityLLM control service: runs real benchmarks for the Arena page")
    p.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to reach the service from another machine")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--model", action="append", default=[], help="NAME=PATH, repeat for each model")
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--engine-port", type=int, default=8100)
    p.add_argument("--mock", action="store_true", help="Use the simulated engine (no GPU), to test the whole pipeline")
    p.add_argument("--startup-timeout", type=float, default=600.0)
    p.add_argument("--settle", type=float, default=5.0, help="Seconds to rest between engine runs")
    p.add_argument("--cors-origin", action="append", default=[], help="Extra allowed browser origin, or * for any")
    p.add_argument("--api-key", default=None, help="Require this key in the X-API-Key header")
    p.add_argument("--no-frontend", action="store_true", help="Do not serve frontend/dist")
    a = p.parse_args()

    s = Settings(host=a.host, port=a.port, mock=a.mock, engine_base_port=a.engine_port,
                 startup_timeout=a.startup_timeout, settle_seconds=a.settle, cors_origins=a.cors_origin,
                 explicit_models=parse_models(a.model), api_key=a.api_key)
    if a.models_dir:
        s.models_dir = a.models_dir
    if a.results_dir:
        s.results_dir = a.results_dir
    if a.no_frontend:
        s.frontend_dir = Path("/nonexistent")
    where = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}"
    print(f"VelocityLLM control service on {where}  ({'mock engine' if a.mock else 'real engine'})")
    if s.frontend_dir.is_dir():
        print(f"Arena: {where}/#/lab")
    else:
        print("Frontend build not found (run npm run build in frontend/); API only")
    uvicorn.run(create_app(s), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
