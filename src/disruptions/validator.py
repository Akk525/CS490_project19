from src.dataset.schema import ProceduralExample


WEAK_STEP_WORDS = {
    "serve", "enjoy", "repeat", "continue", "finish", "done",
    "set aside", "garnish", "as needed", "if desired"
}

GENERIC_TARGET_PHRASES = {
    "continue with the workflow",
    "preserve the intent",
    "adjust as needed",
    "correct the failed result as simply as possible"
}


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _is_actionable(step: str) -> bool:
    step = _normalize(step)
    return len(step.split()) >= 3


def _is_weak_step(step: str) -> bool:
    step = _normalize(step)
    return any(word in step for word in WEAK_STEP_WORDS)


def _target_is_meaningful(target: str, step: str) -> bool:
    target = _normalize(target)
    step = _normalize(step)

    if not target:
        return False

    # reject trivial "prefix + original step"
    if target.endswith(step):
        return False

    # reject overly short targets
    if len(target.split()) < 5:
        return False

    # reject generic fluff
    if any(p in target for p in GENERIC_TARGET_PHRASES):
        return False

    return True


def _disruption_matches_step(ex: ProceduralExample) -> bool:
    step = _normalize(ex.disrupted_step_text)
    d_type = ex.disruption_type

    # missing ingredient should involve adding/using something
    if d_type == "missing_ingredient":
        return any(v in step for v in ["add", "mix", "combine", "pour", "season"])

    # missing tool should involve manipulation
    if d_type == "missing_tool":
        return any(v in step for v in ["chop", "slice", "cut", "mix", "whisk", "grill", "bake"])

    # incorrect object should involve a choice
    if d_type == "incorrect_object":
        return any(v in step for v in ["add", "use", "place", "spread", "apply"])

    # step failure should be actionable
    if d_type == "step_failure":
        return _is_actionable(step)

    # environmental constraint is broad but still should be actionable
    if d_type == "environmental_constraint":
        return _is_actionable(step)

    return True


def validate_disrupted_example(ex: ProceduralExample) -> bool:
    # ---- Basic checks ----
    if len(ex.full_procedure) < 3:
        return False

    if ex.disrupted_step_index <= 0 or ex.disrupted_step_index >= len(ex.full_procedure):
        return False

    if not ex.goal or not ex.disruption_description:
        return False

    step = ex.disrupted_step_text

    # ---- Step quality ----
    if not _is_actionable(step):
        return False

    if _is_weak_step(step):
        return False

    # ---- Disruption fit ----
    if not _disruption_matches_step(ex):
        return False

    # ---- Target quality ----
    if not _target_is_meaningful(ex.target_adaptation or "", step):
        return False

    return True