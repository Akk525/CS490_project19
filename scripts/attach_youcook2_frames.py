import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.visual_context import index_frame_paths
from src.utils.io import read_jsonl, write_jsonl


def _default_inputs() -> list[Path]:
    return [
        ROOT / 'data' / 'interim' / 'youcook2_examples.jsonl',
        ROOT / 'data' / 'processed' / 'benchmark.jsonl',
        ROOT / 'data' / 'splits' / 'train.jsonl',
        ROOT / 'data' / 'splits' / 'dev.jsonl',
        ROOT / 'data' / 'splits' / 'test.jsonl',
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description='Attach YouCook2 frame paths to existing JSONL artifacts.')
    parser.add_argument('--frames-dir', default='data/raw/youcook2/frames')
    parser.add_argument('--input', dest='inputs', action='append', help='Optional JSONL file to update. Can be passed multiple times.')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing image_path values.')
    args = parser.parse_args()

    frames_dir = (ROOT / args.frames_dir).resolve() if not Path(args.frames_dir).is_absolute() else Path(args.frames_dir).resolve()
    frame_index = index_frame_paths(frames_dir, ROOT)
    if not frame_index:
        raise RuntimeError(f'No frames found under {frames_dir}')

    inputs = [Path(p).resolve() for p in args.inputs] if args.inputs else _default_inputs()
    for input_path in inputs:
        if not input_path.exists():
            continue

        rows = read_jsonl(input_path)
        updated = 0
        for row in rows:
            if row.get('source_dataset') != 'youcook2':
                continue
            if row.get('image_path') and not args.overwrite:
                continue
            image_path = frame_index.get(str(row.get('source_item_id', '')).strip())
            if image_path:
                row['image_path'] = image_path
                updated += 1

        write_jsonl(input_path, rows)
        print(f'Updated {updated} rows in {input_path}')


if __name__ == '__main__':
    main()
