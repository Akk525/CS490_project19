import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.comparisons import compare_methods, paired_permutation_tests
from src.analysis.error_analysis import add_failure_taxonomy, hardest_examples
from src.analysis.export import export_dataframe
from src.utils.config import load_config
from src.utils.io import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    _cfg = load_config(args.config)
    root = ROOT
    registry_path = root / 'outputs' / 'manifests' / 'run_registry.jsonl'
    manifests = read_jsonl(registry_path) if registry_path.exists() else []

    all_eval = []
    for m in manifests:
        run_id = m['run_id']
        eval_path = root / 'outputs' / 'evaluations' / run_id / 'evaluation_results.jsonl'
        if eval_path.exists():
            all_eval.extend(read_jsonl(eval_path))

    if not all_eval:
        raise RuntimeError('No evaluation results available to summarize.')

    df = pd.DataFrame(all_eval)
    cmp_df = compare_methods(df)
    taxonomy_df = add_failure_taxonomy(df)
    hard_df = hardest_examples(taxonomy_df)
    failure_counts = taxonomy_df.groupby(['failure_type', 'method', 'model']).size().reset_index(name='count')
    sig_df = paired_permutation_tests(df, baseline_method='vanilla')

    export_dataframe(cmp_df, root / 'outputs' / 'summaries' / 'model_comparison.csv')
    export_dataframe(cmp_df, root / 'outputs' / 'summaries' / 'retrieval_strategy.csv')
    export_dataframe(cmp_df, root / 'outputs' / 'summaries' / 'ablation_k.csv')
    export_dataframe(hard_df, root / 'outputs' / 'analysis' / 'hardest_examples.csv')
    export_dataframe(failure_counts, root / 'outputs' / 'analysis' / 'failure_taxonomy_counts.csv')
    export_dataframe(taxonomy_df, root / 'outputs' / 'analysis' / 'failure_taxonomy_examples.csv')
    if not sig_df.empty:
        export_dataframe(sig_df, root / 'outputs' / 'summaries' / 'paired_significance_tests.csv')

    print('Global summaries written to outputs/summaries and outputs/analysis.')


if __name__ == '__main__':
    main()
