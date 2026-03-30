from pathlib import Path
from typing import Dict

from src.utils.io import append_jsonl, write_json


def register_run(manifest: Dict, manifests_path: Path) -> None:
    append_jsonl(manifests_path, manifest)


def save_manifest(manifest: Dict, run_dir: Path) -> None:
    write_json(run_dir / 'manifest.json', manifest)
