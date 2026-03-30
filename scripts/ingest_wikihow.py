import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.wikihow_loader import load_wikihow
from src.utils.config import load_config
from src.utils.io import write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = ROOT

    rows = load_wikihow(root / cfg['paths']['raw_wikihow'], min_steps=cfg['dataset']['min_steps'])
    out_path = root / cfg['paths']['interim'] / 'wikihow_examples.jsonl'
    write_jsonl(out_path, rows)
    print(f'Wrote {len(rows)} WikiHow rows -> {out_path}')


if __name__ == '__main__':
    main()
