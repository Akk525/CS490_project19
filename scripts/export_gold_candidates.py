"""
Export a balanced, review-ready gold candidate subset from a benchmark or split JSONL.

Writes:
  <output_prefix>.jsonl — full records + default review fields
  <output_prefix>.csv   — spreadsheet-friendly columns for manual review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.gold_subset import export_candidates, sample_gold_candidates
from src.utils.io import read_jsonl, write_json


def _default_test_split() -> Path:
    return ROOT / "data" / "splits" / "test.jsonl"


def _normalize_output_prefix(raw: str) -> Path:
    p = Path(raw)
    if p.suffix.lower() in {".jsonl", ".csv"}:
        return p.with_suffix("")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Export balanced gold review candidates from a JSONL split/benchmark.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(_default_test_split()),
        help="Path to benchmark or split JSONL (default: data/splits/test.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "data" / "processed" / "gold_candidates"),
        help="Output path prefix without extension (writes .jsonl and .csv)",
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        default="youcook2",
        help="Filter to this source_dataset (default: youcook2). Use empty string for no filter.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="cooking",
        help="Filter to this domain (default: cooking). Use empty string for no filter.",
    )
    parser.add_argument("--n-per-type", type=int, default=25, help="Max examples per disruption_type (default: 25)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible sampling (default: 42)")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    examples = read_jsonl(input_path)
    source_ds = args.source_dataset.strip() or None
    domain = args.domain.strip() or None

    selected, stats = sample_gold_candidates(
        examples,
        source_dataset=source_ds,
        domain=domain,
        n_per_type=args.n_per_type,
        seed=args.seed,
    )

    out_prefix = _normalize_output_prefix(args.output)
    if not out_prefix.is_absolute():
        out_prefix = (ROOT / out_prefix).resolve()

    jsonl_path, csv_path = export_candidates(selected, out_prefix)

    stats_path = out_prefix.parent / f"{out_prefix.name}_export_stats.json"
    write_json(stats_path, {**stats, "jsonl_path": str(jsonl_path), "csv_path": str(csv_path), "input_path": str(input_path)})

    print(f"Wrote {len(selected)} candidate rows")
    print(f"  JSONL: {jsonl_path}")
    print(f"  CSV:   {csv_path}")
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
