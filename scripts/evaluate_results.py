import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.tables import save_summary_table
from src.evaluation.aggregator import aggregate_eval
from src.evaluation.breakdowns import breakdown_tables
from src.utils.config import load_config
from src.utils.io import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    _cfg = load_config(args.config)
    root = ROOT
    eval_dir = root / 'outputs' / 'evaluations' / args.run_id
    rows = read_jsonl(eval_dir / 'evaluation_results.jsonl')
    df = pd.DataFrame(rows)
    summary = aggregate_eval(df)
    write_json(eval_dir / 'summary.json', summary)
    df.to_csv(eval_dir / 'summary.csv', index=False)
    pd.DataFrame([summary]).to_csv(eval_dir / 'summary_metrics_ci.csv', index=False)

    by_method, by_model, by_disruption, by_source = breakdown_tables(df)
    save_summary_table(by_method, root / 'outputs' / 'summaries' / f'{args.run_id}_by_method.csv', root / 'outputs' / 'summaries' / f'{args.run_id}_by_method.json')
    save_summary_table(by_model, root / 'outputs' / 'summaries' / f'{args.run_id}_by_model.csv', root / 'outputs' / 'summaries' / f'{args.run_id}_by_model.json')
    if not by_disruption.empty:
        by_disruption.to_csv(root / 'outputs' / 'summaries' / f'{args.run_id}_by_disruption.csv', index=False)
    if not by_source.empty:
        by_source.to_csv(root / 'outputs' / 'summaries' / f'{args.run_id}_by_source.csv', index=False)

    print(f'Evaluation summary written for run {args.run_id}.')


if __name__ == '__main__':
    main()
