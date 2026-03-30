from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
from dotenv import load_dotenv


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


def _load_single_config(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    extends = cfg.pop('extends', None)
    if not extends:
        return cfg
    parent_paths: List[Union[str, Path]] = extends if isinstance(extends, list) else [extends]
    merged: Dict[str, Any] = {}
    for p in parent_paths:
        candidate = Path(str(p))
        if candidate.is_absolute():
            parent = candidate
        else:
            from_current = (path.parent / candidate).resolve()
            from_cwd = (Path.cwd() / candidate).resolve()
            if from_current.exists():
                parent = from_current
            elif from_cwd.exists():
                parent = from_cwd
            else:
                raise FileNotFoundError(
                    f"Could not resolve extended config '{p}' from '{path}'."
                )
        merged = _merge_dicts(merged, _load_single_config(parent))
    return _merge_dicts(merged, cfg)


def load_config(config_path: str) -> Dict[str, Any]:
    load_dotenv()
    path = Path(config_path).resolve()
    cfg = _load_single_config(path)
    cfg['_config_path'] = str(path)
    return cfg
