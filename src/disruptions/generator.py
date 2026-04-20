import re
from typing import Dict, List

from src.disruptions.templates import COOKING_TEMPLATES, DIY_TEMPLATES


COOKING_INGREDIENT_WORDS = {
    "salt", "pepper", "sugar", "butter", "milk", "cream", "oil", "olive oil",
    "vinegar", "lemon", "lemon juice", "lime", "lime juice", "egg", "eggs",
    "flour", "water", "cheese", "chicken", "beef", "garlic", "onion", "celery",
    "apple", "apples", "raisins", "cranberries", "nuts", "bread", "breadcrumbs",
    "parmesan", "sauce", "hot sauce", "mustard", "mayo", "mayonnaise"
}

COOKING_TOOL_WORDS = {
    "knife", "pan", "pot", "bowl", "oven", "stove", "skillet", "grill",
    "whisk", "spoon", "fork", "board", "cutting board", "blender", "mixer",
    "plate", "tray"
}

ACTION_WORDS = {
    "add", "mix", "stir", "combine", "chop", "slice", "cut", "pour", "heat",
    "cook", "bake", "fry", "boil", "grill", "whisk", "season", "spread",
    "place", "flip", "press", "remove", "serve"
}


def _domain_templates(domain: str):
    return COOKING_TEMPLATES if "cook" in domain.lower() else DIY_TEMPLATES


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def _contains_any(text: str, vocab: set[str]) -> bool:
    text_l = text.lower()
    return any(term in text_l for term in vocab)


def _looks_actionable(step: str) -> bool:
    step_l = _normalize(step).lower()
    return any(step_l.startswith(v + " ") or step_l == v for v in ACTION_WORDS)


def _has_ingredient_signal(step: str) -> bool:
    return _contains_any(step, COOKING_INGREDIENT_WORDS)


def _has_tool_signal(step: str) -> bool:
    return _contains_any(step, COOKING_TOOL_WORDS)


def _extract_first_matching_term(text: str, vocab: set[str]) -> str | None:
    text_l = text.lower()
    # prefer longer matches first
    for term in sorted(vocab, key=len, reverse=True):
        if term in text_l:
            return term
    return None


def _step_mentions_heat(step: str) -> bool:
    return _contains_any(step, {"bake", "fry", "boil", "grill", "cook", "heat", "roast"})


def _safe_substitute_for(term: str | None) -> str:
    substitutes = {
        "lemon juice": "lime juice or a small amount of vinegar",
        "lemon": "lime",
        "lime juice": "lemon juice",
        "vinegar": "lemon juice or lime juice",
        "salt": "a small amount of soy sauce or omit it if needed",
        "pepper": "paprika, chili flakes, or omit it",
        "sugar": "honey, maple syrup, or omit it",
        "butter": "oil",
        "milk": "water or plant milk",
        "cream": "milk",
        "oil": "melted butter or ghee",
        "olive oil": "another neutral cooking oil, melted butter, or ghee",
        "water": "broth, stock, or another cooking liquid",
        "egg": "a flax egg or other binder",
        "eggs": "a flax egg or other binder",
        "flour": "cornstarch, rice flour, or another binder",
        "breadcrumbs": "crushed crackers",
        "parmesan": "another hard cheese",
        "celery": "another crunchy vegetable such as apple or fennel",
        "raisins": "dried cranberries",
        "cranberries": "raisins",
        "nuts": "seeds or omit them",
        "bread": "another bread or toast base",
        "cheese": "another melty cheese",
        "chicken": "tofu, mushrooms, chickpeas, paneer, or another protein",
        "beef": "mushrooms, lentils, tofu, or another protein",
        "garlic": "garlic powder or omit it",
        "onion": "shallot or onion powder",
        "sauce": "a similar sauce or a quick sauce made from available seasonings",
        "hot sauce": "chili flakes, cayenne, or another spicy sauce",
        "mustard": "a small amount of vinegar plus seasoning, or omit it",
        "mayo": "yogurt, sour cream, or another creamy spread",
        "mayonnaise": "yogurt, sour cream, or another creamy spread",
    }
    if not term:
        return "a similar available ingredient"
    return substitutes.get(term, "a similar available ingredient")


def _target_missing_ingredient(step: str) -> str:
    ingredient = _extract_first_matching_term(step, COOKING_INGREDIENT_WORDS)
    sub = _safe_substitute_for(ingredient)
    if ingredient:
        return f"If {ingredient} is unavailable, replace it with {sub} and continue."
    return "Use a similar available ingredient that preserves the purpose of the step and continue."


def _missing_ingredient_for_step(step: str) -> str | None:
    return _extract_first_matching_term(step, COOKING_INGREDIENT_WORDS)


def _target_missing_tool(step: str) -> str:
    tool = _extract_first_matching_term(step, COOKING_TOOL_WORDS)
    if tool == "knife" or "chop" in step.lower() or "slice" in step.lower():
        return "Use the sharpest safe available utensil and cut more slowly and carefully."
    if tool in {"bowl", "plate"}:
        return "Use any clean container that can safely hold and mix the ingredients."
    if tool in {"oven", "stove", "grill", "pan", "pot", "skillet"} or _step_mentions_heat(step):
        return "Use another available heating method that achieves the same cooking effect, then continue."
    if tool:
        return f"Use a safe alternative to the {tool} that can perform the same function."
    return "Use a safe alternative tool or method that achieves the same result."

def _target_incorrect_object(step: str) -> str:
    ingredient = _extract_first_matching_term(step, COOKING_INGREDIENT_WORDS)
    if ingredient:
        return f"If the wrong ingredient was used, remove or correct it if possible, then continue using {ingredient} or a suitable substitute."
    return "Undo or correct the incorrect item choice if possible, then repeat the step with the correct item."

def _target_step_failure(step: str) -> str:
    if "chop" in step.lower() or "slice" in step.lower():
        return "Redo the cut so the pieces are even and suitable for the next step."
    if "mix" in step.lower() or "combine" in step.lower() or "stir" in step.lower():
        return "Adjust the texture by remixing or adding a small amount of the needed ingredient, then continue."
    if _step_mentions_heat(step):
        return "Reduce or adjust heat, correct the cooking state if possible, and continue once the food is back on track."
    return "Correct the failed result as simply as possible, then continue with the workflow."

def _target_environmental_constraint(step: str) -> str:
    if _step_mentions_heat(step):
        return "Adapt the step to the available cooking environment and use a different heat source or cooking method if needed."
    if "chop" in step.lower() or "slice" in step.lower():
        return "Use any clean, stable surface available and proceed carefully."
    return "Adjust the step to fit the current environment while preserving the intended outcome."

def _rule_based_target(step: str, disruption_type: str) -> str:
    step = _normalize(step)

    if disruption_type == "missing_tool":
        return _target_missing_tool(step)
    if disruption_type == "missing_ingredient":
        return _target_missing_ingredient(step)
    if disruption_type == "incorrect_object":
        return _target_incorrect_object(step)
    if disruption_type == "step_failure":
        return _target_step_failure(step)
    if disruption_type == "environmental_constraint":
        return _target_environmental_constraint(step)
    return ""


def _missing_ingredient_modalities(row: Dict) -> List[str]:
    configured = row.get("missing_ingredient_modalities")
    if configured:
        return [str(x).lower() for x in configured]
    return ["text"]


def _modality_description(base_description: str, modality: str) -> str:
    if modality == "vision":
        return (
            "The provided image of the current state indicates that a key ingredient "
            "needed for this step is missing. Adapt the recipe while preserving the intended dish."
        )
    return base_description


def _allowed_disruption_types(step: str, domain: str) -> List[str]:
    step = _normalize(step)
    domain_l = domain.lower()

    allowed = []

    if not _looks_actionable(step):
        return allowed

    if "cook" in domain_l:
        if _has_ingredient_signal(step):
            allowed.append("missing_ingredient")
            allowed.append("incorrect_object")

        if _has_tool_signal(step) or any(w in step.lower() for w in ["chop", "slice", "mix", "whisk", "grill", "bake", "cook"]):
            allowed.append("missing_tool")

        allowed.append("step_failure")
        allowed.append("environmental_constraint")
    else:
        # more conservative for non-cooking
        if _has_tool_signal(step):
            allowed.append("missing_tool")
        allowed.append("step_failure")
        allowed.append("environmental_constraint")

    # preserve order, remove duplicates
    seen = set()
    deduped = []
    for a in allowed:
        if a not in seen:
            seen.add(a)
            deduped.append(a)
    return deduped


def generate_disruptions_for_example(row: Dict, disruption_types: List[str]) -> List[Dict]:
    steps = row["steps"]
    domain = row.get("domain", "general")
    templates = _domain_templates(domain)

    out = []

    # Skip the first step less aggressively, but still allow later steps only by default
    for i in range(1, len(steps)):
        step = _normalize(steps[i])
        if not step:
            continue

        allowed = _allowed_disruption_types(step, domain)
        selected_types = [d for d in disruption_types if d in allowed and d in templates]

        # Cap the number of disruptions per step to reduce noise
        selected_types = selected_types[:3]

        for d_type in selected_types:
            target = _rule_based_target(step, d_type)
            modalities = _missing_ingredient_modalities(row) if d_type == "missing_ingredient" else ["text"]
            missing_ingredient = _missing_ingredient_for_step(step) if d_type == "missing_ingredient" else None

            for modality in modalities:
                if modality == "vision" and not row.get("image_path"):
                    continue

                out.append(
                    {
                        "current_state": " ".join(_normalize(s) for s in steps[:i]),
                        "disrupted_step_index": i,
                        "disrupted_step_text": step,
                        "disruption_type": d_type,
                        "disruption_description": _modality_description(templates[d_type], modality),
                        "target_adaptation": target,
                        "target_provenance": "rule_based" if target else None,
                        "metadata": {
                            "domain_templates": "cooking" if templates is COOKING_TEMPLATES else "diy",
                            "allowed_disruption_types": allowed,
                            "disruption_modality": modality,
                            "missing_ingredient": missing_ingredient,
                            "suggested_substitute": _safe_substitute_for(missing_ingredient) if missing_ingredient else None,
                        },
                    }
                )

    return out
