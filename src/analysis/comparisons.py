from typing import Dict, List

import numpy as np
import pandas as pd


def compare_methods(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(['method', 'model']).mean(numeric_only=True).reset_index()


def paired_permutation_tests(
    df: pd.DataFrame,
    baseline_method: str = 'vanilla',
    metrics: List[str] = None,
    n_permutations: int = 5000,
) -> pd.DataFrame:
    metrics = metrics or ['feasibility_score', 'adaptation_quality_score', 'helpfulness_score', 'semantic_similarity_score']
    if 'example_id' not in df.columns or 'method' not in df.columns or 'model' not in df.columns:
        return pd.DataFrame()

    rng = np.random.default_rng(42)
    rows: List[Dict] = []
    for model_name, model_df in df.groupby('model'):
        methods = sorted(model_df['method'].dropna().unique().tolist())
        for method in methods:
            if method == baseline_method:
                continue
            for metric in metrics:
                if metric not in model_df.columns:
                    continue
                pivot = model_df.pivot_table(index='example_id', columns='method', values=metric, aggfunc='mean').dropna()
                if baseline_method not in pivot.columns or method not in pivot.columns or pivot.empty:
                    continue
                diffs = (pivot[method] - pivot[baseline_method]).astype(float).to_numpy()
                observed = float(np.mean(diffs))
                # Two-sided paired permutation using sign flips.
                signs = rng.choice([-1.0, 1.0], size=(n_permutations, diffs.size), replace=True)
                perm_means = (signs * diffs).mean(axis=1)
                p_value = float((np.abs(perm_means) >= abs(observed)).mean())
                rows.append(
                    {
                        'model': model_name,
                        'metric': metric,
                        'baseline_method': baseline_method,
                        'comparison_method': method,
                        'n_pairs': int(diffs.size),
                        'mean_delta': observed,
                        'p_value': p_value,
                    }
                )
    return pd.DataFrame(rows)
