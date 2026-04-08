from pathlib import Path
from typing import Dict


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    root = project_root.resolve()
    if resolved.is_relative_to(root):
        return str(resolved.relative_to(root))
    return str(resolved)


def index_frame_paths(frames_dir: Path, project_root: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not frames_dir.exists():
        return index

    for path in sorted(frames_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        candidates = [path.stem]
        if path.parent != frames_dir:
            candidates.insert(0, path.parent.name)

        portable = _portable_path(path, project_root)
        for key in candidates:
            index.setdefault(key, portable)
    return index
