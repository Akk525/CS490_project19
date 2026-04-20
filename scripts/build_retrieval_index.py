import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.index_builder import build_index
from src.retrieval.factory import build_embedder
from src.utils.config import load_config
from src.utils.io import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = ROOT

    split_dir = root / cfg['paths']['splits_dir']
    library_path = split_dir / 'retrieval_library.jsonl'
    if library_path.exists():
        rows = read_jsonl(library_path)
    else:
        rows = read_jsonl(split_dir / 'train.jsonl')

    embedder = build_embedder(cfg['backends']['embedding'], cfg['models']['embedding_model'])
    build_index(
        rows=rows,
        embedder=embedder,
        out_index=root / cfg['paths']['retrieval_index'],
        out_library=root / cfg['paths']['retrieval_library'],
    )
    print(f'Index built with {len(rows)} rows.')


if __name__ == '__main__':
    main()
