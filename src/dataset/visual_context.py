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


def index_step_frame_paths(frames_dir: Path, project_root: Path) -> Dict[str, Dict[int, str]]:
    index: Dict[str, Dict[int, str]] = {}
    if not frames_dir.exists():
        return index

    for path in sorted(frames_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        video_id = path.parent.name if path.parent != frames_dir else ''
        step_index = _step_index_from_stem(path.stem)
        if not video_id or step_index is None:
            continue

        index.setdefault(video_id, {}).setdefault(step_index, _portable_path(path, project_root))
    return index


def _step_index_from_stem(stem: str) -> int | None:
    for prefix in ('step_', 'segment_', 'frame_'):
        if stem.startswith(prefix):
            raw = stem[len(prefix):]
            if raw.isdigit():
                return int(raw)
    return None
