import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.ablations import run_ablation_suite
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = ROOT
    run_ids = run_ablation_suite(cfg, root)
    print('Completed runs:')
    for r in run_ids:
        print(r)


if __name__ == '__main__':
    main()
