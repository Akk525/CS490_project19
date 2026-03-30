from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.dataset.normalizer import normalize_steps
from src.utils.io import read_json, read_jsonl


def _normalize_row(item: Dict, idx: int) -> Dict:
    title = item.get('title') or item.get('goal') or item.get('task')
    steps = item.get('steps') or item.get('procedure') or item.get('methods')
    if not title or not isinstance(steps, list):
        raise ValueError(f'WikiHow record {idx} missing title or steps list.')
    return {
        'source_dataset': 'wikihow',
        'source_item_id': str(item.get('id', idx)),
        'domain': item.get('domain', 'general_diy'),
        'goal': str(title).strip(),
        'steps': normalize_steps([str(s) for s in steps]),
        'metadata': {'categories': item.get('categories', [])},
    }


def load_wikihow(raw_dir: Path, min_steps: int = 3) -> List[Dict]:
    jsonl_path = raw_dir / 'procedures.jsonl'
    json_path = raw_dir / 'procedures.json'
    csv_path = raw_dir / 'wikihow.csv'

    data: List[Dict]
    if jsonl_path.exists():
        data = read_jsonl(jsonl_path)
    elif json_path.exists():
        loaded = read_json(json_path)
        data = loaded if isinstance(loaded, list) else loaded.get('data', [])
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        if not {'title', 'steps'}.issubset(set(df.columns)):
            raise ValueError('wikihow.csv requires title and steps columns, with steps serialized as ||-joined strings.')
        data = []
        for i, row in df.iterrows():
            data.append(
                {
                    'id': row.get('id', i),
                    'title': row['title'],
                    'steps': [s.strip() for s in str(row['steps']).split('||') if s.strip()],
                    'domain': row.get('domain', 'general_diy'),
                }
            )
    else:
        raise FileNotFoundError(
            f'No supported WikiHow source found in {raw_dir}. Expected procedures.jsonl, procedures.json, or wikihow.csv.'
        )

    out: List[Dict] = []
    for i, item in enumerate(data):
        row = _normalize_row(item, i)
        if len(row['steps']) >= min_steps:
            out.append(row)
    return out
