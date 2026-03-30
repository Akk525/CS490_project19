from typing import Dict, List

import numpy as np
import pandas as pd


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> Dict[str, float]:
    if values.size == 0:
        return {'mean': 0.0, 'ci_low': 0.0, 'ci_high': 0.0}
    if values.size == 1:
        v = float(values[0])
        return {'mean': v, 'ci_low': v, 'ci_high': v}
    rng = np.random.default_rng(42)
    boots: List[float] = []
    n = values.size
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots.append(float(np.mean(sample)))
    alpha = (1.0 - ci) / 2.0
    return {
        'mean': float(np.mean(values)),
        'ci_low': float(np.quantile(boots, alpha)),
        'ci_high': float(np.quantile(boots, 1.0 - alpha)),
    }


def aggregate_eval(df: pd.DataFrame) -> Dict:
    metrics = [
        'constraint_violation',
        'feasibility_score',
        'adaptation_quality_score',
        'helpfulness_score',
        'semantic_similarity_score',
        'llm_judge_score',
    ]
    out = {}
    for m in metrics:
        if m in df.columns:
            arr = df[m].dropna().astype(float).to_numpy()
            if arr.size > 0:
                stats = _bootstrap_ci(arr)
                out[m] = stats['mean']
                out[f'{m}_ci_low'] = stats['ci_low']
                out[f'{m}_ci_high'] = stats['ci_high']
    out['n_examples'] = int(len(df))
    return out
