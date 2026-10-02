import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_USERS = 200
DEFAULT_MAX_MODEL_LEN = 4096


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 9000
    models_dir: Path = ROOT / "models"
    explicit_models: dict = field(default_factory=dict)
    results_dir: Path = ROOT / "results" / "runs"
    engine_base_port: int = 8100
    mock: bool = False
    startup_timeout: float = 600.0
    request_timeout: float = 300.0
    settle_seconds: float = 5.0
    cors_origins: list = field(default_factory=list)
    frontend_dir: Path = ROOT / "frontend" / "dist"
    api_key: str | None = None


def native_context(path: str) -> int | None:
    cfg = Path(path)
    if not cfg.is_absolute():
        cfg = ROOT / cfg
    try:
        return int(json.loads((cfg / "config.json").read_text())["max_position_embeddings"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def context_limit(path: str, requested: int = DEFAULT_MAX_MODEL_LEN) -> int:
    native = native_context(path)
    return min(requested, native) if native else requested
