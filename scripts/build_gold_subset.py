"""
Build the final gold subset from a manually reviewed CSV.

Reads review decisions, merges full records from source JSONL, validates rules,
and writes gold JSONL, audit CSV, and summary JSON/CSV under outputs/summaries/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.gold_subset import (
    build_gold_subset,
    load_examples_by_id,
    load_reviewed_csv,
    write_gold_subset_reviewed_csv,
    write_summary_outputs,
)
from src.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final gold subset from reviewed CSV.")
    parser.add_argument(
        "--reviewed-csv",
        type=str,
        default=str(ROOT / "data" / "processed" / "gold_candidates.csv"),
        help="Path to reviewed CSV (default: data/processed/gold_candidates.csv)",
    )
    parser.add_argument(
        "--source-jsonl",
        type=str,
        default=str(ROOT / "data" / "splits" / "test.jsonl"),
        help="Source benchmark/split JSONL for full records (default: data/splits/test.jsonl)",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=str(ROOT / "data" / "processed" / "gold_subset.jsonl"),
        help="Output path for final gold subset JSONL (default: data/processed/gold_subset.jsonl)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(ROOT / "data" / "processed" / "gold_subset_reviewed.csv"),
        help="Audit CSV path (default: data/processed/gold_subset_reviewed.csv)",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default=str(ROOT / "outputs" / "summaries"),
        help="Directory for gold_subset_summary.json and .csv (default: outputs/summaries)",
    )
    args = parser.parse_args()

    reviewed_path = Path(args.reviewed_csv).resolve()
    source_path = Path(args.source_jsonl).resolve()
    out_jsonl = Path(args.output_jsonl).resolve()
    out_csv = Path(args.output_csv).resolve()
    summary_dir = Path(args.summary_dir).resolve()

    if not reviewed_path.is_file():
        raise FileNotFoundError(f"Reviewed CSV not found: {reviewed_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source JSONL not found: {source_path}")

    reviewed_rows = load_reviewed_csv(reviewed_path)
    source_by_id = load_examples_by_id(source_path)

    gold, audit_rows, summary = build_gold_subset(reviewed_rows, source_by_id)

    write_jsonl(out_jsonl, gold)
    write_gold_subset_reviewed_csv(audit_rows, out_csv)
    write_summary_outputs(summary, summary_dir)

    print(f"Final gold subset size: {len(gold)}")
    print(f"  Gold JSONL:    {out_jsonl}")
    print(f"  Audit CSV:     {out_csv}")
    print(f"  Summary JSON:  {summary_dir / 'gold_subset_summary.json'}")
    print(f"  Summary CSV:   {summary_dir / 'gold_subset_summary.csv'}")
    nw = len(summary.get("warnings", []))
    if nw:
        print(f"Warnings: {nw} (see summary JSON 'warnings' list)")


if __name__ == "__main__":
    main()
