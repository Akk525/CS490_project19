from typing import Dict, Iterable, List

from src.dataset.schema import ProceduralExample
from src.disruptions.generator import generate_disruptions_for_example
from src.disruptions.validator import validate_disrupted_example
from src.utils.hashing import stable_hash


MAX_STEP_CHARS = 500
MIN_STEPS = 3

COOKING_KEYWORDS = {
    "cook", "cooking", "recipe", "bake", "baking", "food", "kitchen",
    "meal", "ingredient", "ingredients", "oven", "stove", "pan", "boil",
    "fry", "saute", "roast", "grill", "whisk", "mix", "chop"
}

GENERAL_PROCEDURAL_KEYWORDS = {
    "repair", "clean", "install", "replace", "remove", "assemble",
    "build", "make", "prepare", "fix", "wash", "paint", "sew"
}


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _looks_too_long(step: str) -> bool:
    return len(_normalize_text(step)) > MAX_STEP_CHARS


def _is_actionable_step(step: str) -> bool:
    step_l = _normalize_text(step).lower()
    if not step_l:
        return False
    # crude but useful: procedural steps usually start with an action verb
    starter_verbs = (
        "add", "mix", "stir", "heat", "cook", "bake", "chop", "slice", "pour",
        "remove", "install", "clean", "wash", "apply", "attach", "replace",
        "cut", "boil", "fry", "whisk", "preheat", "combine", "place", "use",
        "check", "turn", "let", "set", "dry", "measure", "open", "close"
    )
    return step_l.startswith(starter_verbs)


def _row_categories(row: Dict) -> List[str]:
    metadata = row.get("metadata", {}) or {}
    cats = metadata.get("categories", []) or []
    return [str(c).lower() for c in cats]


def _row_text_blob(row: Dict) -> str:
    parts = [
        row.get("goal", ""),
        " ".join(row.get("steps", []) or []),
        " ".join(_row_categories(row)),
    ]
    return " ".join(parts).lower()


def _is_relevant_row(row: Dict) -> bool:
    goal = _normalize_text(row.get("goal", ""))
    steps = row.get("steps", []) or []

    if not goal or len(steps) < MIN_STEPS:
        return False

    # reject giant article-like steps
    if any(_looks_too_long(step) for step in steps):
        return False

    # reject procedures with too few actionable steps
    actionable_count = sum(_is_actionable_step(step) for step in steps)
    if actionable_count < max(2, len(steps) // 2):
        return False

    source = row.get("source_dataset", "").lower()
    blob = _row_text_blob(row)

    # Always keep YouCook2 unless step quality is terrible
    if source == "youcook2":
        return True

    # For WikiHow, prefer cooking first, then practical procedural tasks
    if source == "wikihow":
        if any(k in blob for k in COOKING_KEYWORDS):
            return True
        if any(k in blob for k in GENERAL_PROCEDURAL_KEYWORDS):
            return True
        return False

    return True


def _disruption_fits_example(row: Dict, disruption: Dict) -> bool:
    disruption_type = disruption.get("disruption_type", "")
    step_text = _normalize_text(disruption.get("disrupted_step_text", "")).lower()

    if not step_text:
        return False

    ingredients = " ".join((row.get("ingredients", []) or [])).lower()
    requirements = " ".join((row.get("requirements", []) or [])).lower()
    goal_blob = _row_text_blob(row)

    if disruption_type == "missing_ingredient":
        return any(tok in goal_blob for tok in ["ingredient", "ingredients", "mix", "cook", "bake", "recipe"]) or bool(ingredients)

    if disruption_type == "missing_tool":
        return any(tok in goal_blob for tok in ["tool", "knife", "pan", "oven", "stove", "bowl", "whisk", "machine", "clean", "repair", "install"]) or bool(requirements)

    if disruption_type == "incorrect_object":
        return any(tok in step_text for tok in ["use", "add", "apply", "attach", "replace", "pour", "mix", "install"])

    if disruption_type == "step_failure":
        return _is_actionable_step(step_text)

    if disruption_type == "environmental_constraint":
        return True

    return True


def _clean_target_adaptation(target: str, original_step: str) -> str:
    target = _normalize_text(target)
    original_step = _normalize_text(original_step)

    # reject trivial "prefix + original step" targets
    if target.endswith(original_step):
        return ""

    if len(target) > MAX_STEP_CHARS:
        return ""

    return target


def build_benchmark(rows: Iterable[Dict], disruption_types: List[str]) -> List[ProceduralExample]:
    benchmark: List[ProceduralExample] = []

    for row in rows:
        if not _is_relevant_row(row):
            continue

        goal = _normalize_text(row["goal"])
        steps = [_normalize_text(s) for s in row["steps"] if _normalize_text(s)]

        if len(steps) < MIN_STEPS:
            continue

        base_id = stable_hash(f"{row['source_dataset']}::{row['source_item_id']}::{goal}")
        disrupted = generate_disruptions_for_example({**row, "goal": goal, "steps": steps}, disruption_types)

        for idx, d in enumerate(disrupted):
            if not _disruption_fits_example(row, d):
                continue

            target_adaptation = d.get("target_adaptation")
            if target_adaptation:
                target_adaptation = _clean_target_adaptation(
                    target_adaptation,
                    d.get("disrupted_step_text", "")
                )

            ex = ProceduralExample(
                example_id=f"{base_id}_{idx}",
                source_dataset=row["source_dataset"],
                source_item_id=row["source_item_id"],
                domain=row.get("domain", "general"),
                image_path=row.get("image_path"),
                goal=goal,
                full_procedure=steps,
                current_state=_normalize_text(d["current_state"]),
                disrupted_step_index=d["disrupted_step_index"],
                disrupted_step_text=_normalize_text(d["disrupted_step_text"]),
                disruption_type=d["disruption_type"],
                disruption_description=_normalize_text(d["disruption_description"]),
                target_adaptation=target_adaptation,
                target_provenance=d.get("target_provenance"),
                available_context={
                    "metadata": row.get("metadata", {}),
                    "source_procedure_length": len(steps),
                },
                metadata=d.get("metadata", {}),
            )

            if validate_disrupted_example(ex):
                benchmark.append(ex)

    return benchmark
