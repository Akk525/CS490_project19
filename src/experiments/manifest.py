import subprocess
from pathlib import Path
from typing import Dict

from src.utils.timestamps import utc_now_iso


def git_commit_hash_or_none(project_dir: Path):
    try:
        out = subprocess.check_output(['git', '-C', str(project_dir), 'rev-parse', 'HEAD'], text=True)
        return out.strip()
    except Exception:
        return None


def build_manifest(run_id: str, cfg: Dict, project_dir: Path) -> Dict:
    return {
        'run_id': run_id,
        'timestamp': utc_now_iso(),
        'config_path': cfg.get('_config_path'),
        'git_commit_hash': git_commit_hash_or_none(project_dir),
        'generation_model': cfg['models']['generation_model'],
        'embedding_model': cfg['models']['embedding_model'],
        'reranker_model': cfg['models']['reranker_model'],
        'prompt_type': cfg['prompting']['type'],
        'retrieval': cfg['retrieval'],
        'split': cfg.get('experiment', {}).get('split', 'test'),
    }
