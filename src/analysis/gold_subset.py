"""
Gold subset export, review, and build workflow.

Supports balanced sampling by disruption_type, CSV/JSONL round-trip for manual curation,
and production of a final gold evaluation JSONL with summary reports.
"""

from __future__ import annotations

import csv
import json
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.utils.io import read_jsonl, write_json, write_jsonl

VALID_REVIEW_STATUSES = frozenset({"accept", "reject", "edit", "pending"})


def load_examples_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load a JSONL benchmark or split file keyed by example_id.

    Raises if duplicate example_id appears (first wins with warning).
    """
    rows = read_jsonl(path)
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        eid = row.get("example_id")
        if not eid:
            continue
        if eid in by_id:
            warnings.warn(f"Duplicate example_id in {path}: {eid} (keeping first)")
            continue
        by_id[eid] = row
    return by_id


def _filter_examples(
    examples: Sequence[Dict[str, Any]],
    source_dataset: Optional[str],
    domain: Optional[str],
) -> List[Dict[str, Any]]:
    out = []
    for ex in examples:
        if source_dataset and ex.get("source_dataset") != source_dataset:
            continue
        if domain and ex.get("domain") != domain:
            continue
        out.append(ex)
    return out


def _sample_diverse_by_source_item(
    candidates: List[Dict[str, Any]],
    n: int,
    rng: random.Random,
    used_example_ids: Set[str],
) -> List[Dict[str, Any]]:
    """
    Sample up to n examples, preferring unseen source_item_id when possible.
    Skips example_id already in used_example_ids.
    """
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    picked: List[Dict[str, Any]] = []
    picked_ids: Set[str] = set()
    used_sources: Set[str] = set()

    def try_take(row: Dict[str, Any], require_new_source: bool) -> bool:
        eid = row.get("example_id")
        if not eid or eid in used_example_ids or eid in picked_ids:
            return False
        sid = str(row.get("source_item_id", ""))
        if require_new_source and sid in used_sources:
            return False
        picked.append(row)
        picked_ids.add(eid)
        used_sources.add(sid)
        return True

    for row in shuffled:
        if len(picked) >= n:
            break
        try_take(row, require_new_source=True)

    if len(picked) < n:
        for row in shuffled:
            if len(picked) >= n:
                break
            try_take(row, require_new_source=False)

    return picked


def sample_gold_candidates(
    examples: Sequence[Dict[str, Any]],
    *,
    source_dataset: Optional[str] = None,
    domain: Optional[str] = None,
    n_per_type: int = 25,
    seed: int = 42,
    disruption_types: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Balanced sample: up to ``n_per_type`` examples per disruption_type.

    Returns (selected_rows, stats_dict).
    """
    filtered = _filter_examples(examples, source_dataset, domain)
    rng = random.Random(seed)

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in filtered:
        dt = ex.get("disruption_type") or "unknown"
        if disruption_types is not None and dt not in disruption_types:
            continue
        by_type[dt].append(ex)

    used_ids: Set[str] = set()
    selected: List[Dict[str, Any]] = []
    per_type_counts: Dict[str, Dict[str, int]] = {}

    for dt in sorted(by_type.keys()):
        pool = by_type[dt]
        take = _sample_diverse_by_source_item(pool, n_per_type, rng, used_ids)
        for row in take:
            used_ids.add(row["example_id"])
            selected.append(row)
        per_type_counts[dt] = {"available": len(pool), "sampled": len(take)}

    stats = {
        "seed": seed,
        "n_per_type": n_per_type,
        "source_dataset_filter": source_dataset,
        "domain_filter": domain,
        "total_filtered": len(filtered),
        "total_selected": len(selected),
        "per_disruption_type": per_type_counts,
    }
    return selected, stats


def _row_for_export(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Full benchmark row plus default review columns for JSONL export."""
    row = dict(ex)
    row.setdefault("review_status", "pending")
    row.setdefault("reviewed_target_adaptation", "")
    row.setdefault("review_notes", "")
    row.setdefault("quality_tier", "")
    return row


CSV_EXPORT_FIELDS = [
    "example_id",
    "source_item_id",
    "source_dataset",
    "domain",
    "goal",
    "disruption_type",
    "disruption_description",
    "disrupted_step_text",
    "disrupted_step_index",
    "target_adaptation",
    "current_state",
    "full_procedure_json",
    "review_status",
    "reviewed_target_adaptation",
    "review_notes",
    "quality_tier",
]


def _full_procedure_json(ex: Dict[str, Any]) -> str:
    fp = ex.get("full_procedure", [])
    if isinstance(fp, list):
        return json.dumps(fp, ensure_ascii=False)
    return json.dumps(fp, ensure_ascii=False)


def export_candidates(
    candidates: Sequence[Dict[str, Any]],
    output_prefix: Path,
) -> Tuple[Path, Path]:
    """
    Write JSONL (full records + review defaults) and CSV (review-friendly).

    ``output_prefix`` should be a path without extension, e.g.
    ``data/processed/gold_candidates`` -> ``gold_candidates.jsonl`` and ``gold_candidates.csv``.
    """
    output_prefix = output_prefix.resolve()
    jsonl_path = output_prefix.parent / f"{output_prefix.name}.jsonl"
    csv_path = output_prefix.parent / f"{output_prefix.name}.csv"

    jsonl_rows = [_row_for_export(ex) for ex in candidates]
    write_jsonl(jsonl_path, jsonl_rows)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_EXPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for ex in candidates:
            writer.writerow(
                {
                    "example_id": ex.get("example_id", ""),
                    "source_item_id": ex.get("source_item_id", ""),
                    "source_dataset": ex.get("source_dataset", ""),
                    "domain": ex.get("domain", ""),
                    "goal": ex.get("goal", ""),
                    "disruption_type": ex.get("disruption_type", ""),
                    "disruption_description": ex.get("disruption_description", ""),
                    "disrupted_step_text": ex.get("disrupted_step_text", ""),
                    "disrupted_step_index": ex.get("disrupted_step_index", ""),
                    "target_adaptation": ex.get("target_adaptation") or "",
                    "current_state": ex.get("current_state", ""),
                    "full_procedure_json": _full_procedure_json(ex),
                    "review_status": "pending",
                    "reviewed_target_adaptation": "",
                    "review_notes": "",
                    "quality_tier": "",
                }
            )

    return jsonl_path, csv_path


def load_reviewed_csv(path: Path) -> List[Dict[str, Any]]:
    """Load reviewed CSV rows as dicts (UTF-8, strip keys)."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
            rows.append(row)
    return rows


def _resolve_target_for_row(
    review_status: str,
    reviewed_target: str,
    source_target: Optional[str],
) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    Returns (final_target, reviewed_target_adaptation_used, error_message).
    """
    if review_status == "accept":
        if reviewed_target:
            return reviewed_target, True, None
        if source_target:
            return source_target, False, None
        return None, False, "accept: missing target_adaptation in source and empty reviewed_target_adaptation"
    if review_status == "edit":
        if not reviewed_target:
            return None, False, "edit: reviewed_target_adaptation is required and must be non-empty"
        return reviewed_target, True, None
    return None, False, None


def build_gold_subset(
    reviewed_rows: Sequence[Dict[str, Any]],
    source_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply review rules; return (gold_records, normalized_review_rows_for_csv, summary).

    ``normalized_review_rows_for_csv`` includes one row per input CSV row with
    resolution columns for audit.
    """
    summary_warnings: List[str] = []
    gold: List[Dict[str, Any]] = []
    seen_gold_ids: Set[str] = set()
    audit_rows: List[Dict[str, Any]] = []

    counts = Counter()
    counts["total_csv_rows"] = len(reviewed_rows)

    by_dt = Counter()
    by_ds = Counter()
    by_dom = Counter()

    for csv_row in reviewed_rows:
        eid = (csv_row.get("example_id") or "").strip()
        status_raw = (csv_row.get("review_status") or "pending").strip().lower()
        reviewed_tgt = (csv_row.get("reviewed_target_adaptation") or "").strip()
        notes = (csv_row.get("review_notes") or "").strip()

        audit = dict(csv_row)
        audit["resolution"] = ""
        audit["resolution_detail"] = ""

        if not eid:
            summary_warnings.append("Row missing example_id; skipped")
            counts["invalid_missing_id"] += 1
            audit["resolution"] = "skipped"
            audit["resolution_detail"] = "missing example_id"
            audit_rows.append(audit)
            continue

        if status_raw not in VALID_REVIEW_STATUSES:
            summary_warnings.append(f"Invalid review_status for {eid!r}: {status_raw!r}")
            counts["invalid_status"] += 1
            audit["resolution"] = "skipped"
            audit["resolution_detail"] = f"invalid review_status: {status_raw}"
            audit_rows.append(audit)
            continue

        counts[f"status_{status_raw}"] += 1

        if status_raw == "reject":
            audit["resolution"] = "rejected"
            audit_rows.append(audit)
            continue

        if status_raw == "pending":
            summary_warnings.append(f"example_id {eid!r} still pending; excluded from gold")
            audit["resolution"] = "pending_excluded"
            audit_rows.append(audit)
            continue

        if eid not in source_by_id:
            summary_warnings.append(f"example_id {eid!r} not found in source JSONL")
            counts["unknown_id"] += 1
            audit["resolution"] = "skipped"
            audit["resolution_detail"] = "not in source"
            audit_rows.append(audit)
            continue

        source = dict(source_by_id[eid])
        src_target = source.get("target_adaptation")
        if isinstance(src_target, str):
            src_target = src_target.strip() or None

        final_target, used_reviewed, err = _resolve_target_for_row(
            status_raw, reviewed_tgt, src_target
        )
        if err:
            summary_warnings.append(f"{eid}: {err}")
            counts["resolve_error"] += 1
            audit["resolution"] = "skipped"
            audit["resolution_detail"] = err
            audit_rows.append(audit)
            continue

        if eid in seen_gold_ids:
            summary_warnings.append(f"Duplicate example_id in gold output: {eid} (keeping first)")
            counts["duplicate_id_skipped"] += 1
            audit["resolution"] = "skipped"
            audit["resolution_detail"] = "duplicate example_id"
            audit_rows.append(audit)
            continue

        seen_gold_ids.add(eid)
        record = dict(source)
        record["target_adaptation"] = final_target
        record["quality_tier"] = "gold"
        record["manually_reviewed"] = True
        record["review_notes"] = notes
        record["reviewed_target_adaptation_used"] = used_reviewed
        record["gold_review_status"] = status_raw

        gold.append(record)
        counts["accepted_gold"] += 1
        if status_raw == "edit":
            counts["edited"] += 1
        elif status_raw == "accept" and used_reviewed:
            counts["accept_with_override"] += 1
        elif status_raw == "accept":
            counts["accept_unchanged"] += 1

        by_dt[record.get("disruption_type", "unknown")] += 1
        by_ds[record.get("source_dataset", "unknown")] += 1
        by_dom[record.get("domain", "unknown")] += 1

        audit["resolution"] = "gold"
        audit["final_target_adaptation"] = final_target or ""
        audit["reviewed_target_adaptation_used"] = str(used_reviewed).lower()
        audit_rows.append(audit)

    summary: Dict[str, Any] = {
        "counts": dict(counts),
        "final_gold_subset_size": len(gold),
        "by_disruption_type": dict(by_dt),
        "by_source_dataset": dict(by_ds),
        "by_domain": dict(by_dom),
        "review_status_counts": {
            "accept": int(counts.get("status_accept", 0)),
            "reject": int(counts.get("status_reject", 0)),
            "edit": int(counts.get("status_edit", 0)),
            "pending": int(counts.get("status_pending", 0)),
        },
        "warnings": summary_warnings,
    }
    return gold, audit_rows, summary


def summarize_gold_subset_to_tables(summary: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Flatten summary for CSV export (one row per metric group where useful).
    """
    flat_rows: List[Dict[str, Any]] = []
    c = summary.get("counts", {})
    flat_rows.append({"metric_group": "overall", "key": "total_csv_rows", "value": c.get("total_csv_rows", 0)})
    flat_rows.append({"metric_group": "overall", "key": "final_gold_subset_size", "value": summary.get("final_gold_subset_size", 0)})
    for k in ["status_accept", "status_reject", "status_edit", "status_pending", "invalid_missing_id", "invalid_status", "unknown_id", "resolve_error", "duplicate_id_skipped"]:
        if k in c:
            flat_rows.append({"metric_group": "status", "key": k, "value": c[k]})
    for k in ["accepted_gold", "edited", "accept_with_override", "accept_unchanged"]:
        if k in c:
            flat_rows.append({"metric_group": "resolution", "key": k, "value": c[k]})

    for k, v in summary.get("review_status_counts", {}).items():
        flat_rows.append({"metric_group": "review_status", "key": k, "value": v})

    for dt, n in summary.get("by_disruption_type", {}).items():
        flat_rows.append({"metric_group": "by_disruption_type", "key": dt, "value": n})
    for ds, n in summary.get("by_source_dataset", {}).items():
        flat_rows.append({"metric_group": "by_source_dataset", "key": ds, "value": n})
    for d, n in summary.get("by_domain", {}).items():
        flat_rows.append({"metric_group": "by_domain", "key": d, "value": n})

    return summary, flat_rows


def write_summary_outputs(
    summary: Dict[str, Any],
    summary_dir: Path,
) -> Tuple[Path, Path]:
    """Write gold_subset_summary.json and gold_subset_summary.csv."""
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "gold_subset_summary.json"
    csv_path = summary_dir / "gold_subset_summary.csv"
    _, flat = summarize_gold_subset_to_tables(summary)
    write_json(json_path, summary)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric_group", "key", "value"])
        writer.writeheader()
        for row in flat:
            writer.writerow(row)
    return json_path, csv_path


def write_gold_subset_reviewed_csv(audit_rows: Sequence[Dict[str, Any]], path: Path) -> None:
    """Write audit CSV with dynamic field union."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not audit_rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["example_id", "resolution"])
            writer.writeheader()
        return
    keys: List[str] = []
    seen: Set[str] = set()
    for row in audit_rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in audit_rows:
            writer.writerow({k: row.get(k, "") for k in keys})
