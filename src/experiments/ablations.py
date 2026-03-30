from copy import deepcopy
from pathlib import Path
from typing import Dict, List

from src.experiments.runner import run_experiment


def run_ablation_suite(cfg: Dict, project_dir: Path) -> List[str]:
    runs: List[str] = []
    ab = cfg.get('ablations', {})

    if 'retrieval_k' in ab:
        for k in ab['retrieval_k']:
            for use_r in ab.get('use_reranker', [cfg['retrieval']['use_reranker']]):
                c = deepcopy(cfg)
                c['retrieval']['k'] = k
                c['retrieval']['use_reranker'] = use_r
                c['experiment']['name'] = f"{cfg['experiment']['name']}_k{k}_rerank{int(use_r)}"
                runs.append(run_experiment(c, project_dir))

    if 'retrieval_strategies' in ab:
        for s in ab['retrieval_strategies']:
            c = deepcopy(cfg)
            c['retrieval']['strategy'] = s
            c['experiment']['name'] = f"{cfg['experiment']['name']}_{s}"
            runs.append(run_experiment(c, project_dir))

    if 'models' in ab:
        for m in ab['models']:
            c = deepcopy(cfg)
            c['models']['generation_model'] = m
            c['experiment']['name'] = f"{cfg['experiment']['name']}_{m.replace(':', '_')}"
            runs.append(run_experiment(c, project_dir))

    return runs
