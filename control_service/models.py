from pathlib import Path

from .config import ROOT, Settings, is_vision, native_context


def discover(settings: Settings) -> dict:
    if settings.mock:
        return {"mock-model": "mock", "mock-vision": "mock-vision"}
    found = {}
    if settings.models_dir.is_dir():
        for child in sorted(settings.models_dir.iterdir()):
            if child.is_dir() and (child / "config.json").exists():
                found[child.name] = str(child)
    found.update(settings.explicit_models)
    return found


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def weights_mb(folder: Path):
    files = list(folder.glob("*.safetensors")) or list(folder.glob("*.bin"))
    try:
        total = sum(f.stat().st_size for f in files)
    except OSError:
        return None
    return int(total / (1024 * 1024)) or None


def describe(name: str, path: str, mock: bool) -> dict:
    available = mock or resolve(path).is_dir()
    return {
        "id": name,
        "name": name,
        "path": path,
        "available": available,
        "vision": name == "mock-vision" if mock else is_vision(path),
        "max_position_embeddings": native_context(path),
        "size_mb": None if mock or not available else weights_mb(resolve(path)),
        "note": None if available else "Model folder not found on this machine",
    }
