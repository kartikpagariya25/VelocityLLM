from pathlib import Path

from .config import ROOT, Settings, native_context


def discover(settings: Settings) -> dict:
    found = {}
    if settings.models_dir.is_dir():
        for child in sorted(settings.models_dir.iterdir()):
            if child.is_dir() and (child / "config.json").exists():
                found[child.name] = str(child)
    found.update(settings.explicit_models)
    if settings.mock and not found:
        found["mock-model"] = "mock"
    return found


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def describe(name: str, path: str, mock: bool) -> dict:
    available = mock or resolve(path).is_dir()
    return {
        "id": name,
        "name": name,
        "path": path,
        "available": available,
        "max_position_embeddings": native_context(path),
        "note": None if available else "Model folder not found on this machine",
    }
