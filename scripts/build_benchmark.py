import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.benchmark_builder import build_benchmark
from src.dataset.split_builder import build_splits_by_source, check_leakage
from src.utils.config import load_config
from src.utils.io import read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = ROOT

    yc_path = root / cfg["paths"]["interim"] / "youcook2_examples.jsonl"
    wh_path = root / cfg["paths"]["interim"] / "wikihow_examples.jsonl"

    yc_rows = []
    wh_rows = []

    if yc_path.exists():
        yc_rows = read_jsonl(yc_path)
        print(f"Loaded {len(yc_rows)} YouCook2 interim rows")

    if wh_path.exists():
        wh_rows = read_jsonl(wh_path)
        print(f"Loaded {len(wh_rows)} WikiHow interim rows")

    if not yc_rows and not wh_rows:
        raise RuntimeError(
            "No interim rows found. Run ingest_youcook2.py and ingest_wikihow.py first."
        )

    seed = cfg["project"]["seed"]
    rng = random.Random(seed)

    rng.shuffle(yc_rows)
    rng.shuffle(wh_rows)

    # Per-source caps
    yc_cap = cfg.get("project", {}).get("youcook2_cap")
    wh_cap = cfg.get("project", {}).get("wikihow_cap")

    if yc_cap:
        yc_rows = yc_rows[:yc_cap]
    if wh_cap:
        wh_rows = wh_rows[:wh_cap]

    print(f"Using {len(yc_rows)} YouCook2 rows and {len(wh_rows)} WikiHow rows")

    rows = yc_rows + wh_rows
    rng.shuffle(rows)

    # Optional global cap after balancing
    benchmark_cap = cfg.get("project", {}).get("benchmark_cap")
    if benchmark_cap:
        rows = rows[:benchmark_cap]
        print(f"Using final capped subset of {len(rows)} rows for benchmark build")

    print("Building benchmark...")
    benchmark = build_benchmark(rows, cfg["disruptions"]["enabled_types"], cfg.get("disruptions", {}))
    print(f"Built benchmark with {len(benchmark)} examples")

    benchmark_rows = [x.model_dump() for x in benchmark]
    write_jsonl(root / cfg["paths"]["benchmark"], benchmark_rows)

    ids = [x["example_id"] for x in benchmark_rows]
    splits = build_splits_by_source(benchmark_rows, seed=seed)
    check_leakage(benchmark_rows, splits)

    split_dir = root / cfg["paths"]["splits_dir"]
    split_dir.mkdir(parents=True, exist_ok=True)

    print("Materializing split rows...")
    benchmark_by_id = {x["example_id"]: x for x in benchmark_rows}

    for split_name, split_ids in splits.items():
        split_rows = [benchmark_by_id[i] for i in split_ids if i in benchmark_by_id]
        write_jsonl(split_dir / f"{split_name}.jsonl", split_rows)
        print(f"Wrote {len(split_rows)} rows to {split_name}.jsonl")

    write_json(split_dir / "split_index.json", splits)
    print(f"Benchmark size: {len(benchmark_rows)}")


if __name__ == "__main__":
    main()
