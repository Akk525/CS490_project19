from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
