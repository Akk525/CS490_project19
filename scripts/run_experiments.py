import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.runner import run_experiment
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume-run-id')
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = ROOT
    run_id = run_experiment(cfg, root, resume_run_id=args.resume_run_id)
    print(f'Completed run: {run_id}')


if __name__ == '__main__':
    main()
