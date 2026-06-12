import json
import re
from typing import Dict, Optional


def evaluate_llm_judge_disabled() -> Dict:
    return {'llm_judge_score': None, 'llm_judge_notes': 'LLM judge disabled by config.'}


def _extract_json(text: str) -> Dict:
    text = text.strip()
    if text.startswith('```'):
        text = text.strip('`')
        if text.startswith('json'):
            text = text[4:].strip()
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def _mentions_unavailable_item_as_available(output_text: str, missing_ingredient: str, suggested_substitute: Optional[str]) -> bool:
    text = output_text.lower()
    missing = missing_ingredient.lower()
    substitute = (suggested_substitute or '').lower()

    def _is_placeholder_substitute(value: str) -> bool:
        return value in {'', 'a similar available ingredient', 'not specified'}

    def _mentions_real_substitute(sentence: str) -> bool:
        if _is_placeholder_substitute(substitute):
            return False
        options = [
            option.strip()
            for option in re.split(r"\bor\b|,|/", substitute)
            if option.strip()
        ]
        expanded_options = []
        for option in options:
            expanded_options.append(option)
            simplified = re.sub(r"^(a|an|the|some|small amount of|a small amount of)\s+", "", option)
            expanded_options.append(simplified)
            if " sauce" in simplified:
                expanded_options.append(" ".join(simplified.split()[-2:]))
        options = [option for option in expanded_options if option]
        return any(option and option in sentence for option in options)

    def _is_safe_replacement_sentence(sentence: str) -> bool:
        omitted_missing = bool(
            re.search(rf"\b(omit|skip|leave out|avoid)\b\s+(the\s+)?{re.escape(missing)}\b", sentence)
            or re.search(rf"\bdo not\b[^.\n;]{{0,20}}\b(add|use|include)\b[^.\n;]{{0,20}}\b(the\s+)?{re.escape(missing)}\b", sentence)
        )
        if omitted_missing:
            return True

        if any(
            phrase in sentence
            for phrase in (
                f"{missing} is missing",
                f"{missing} is unavailable",
                f"without {missing}",
                f"do not add {missing}",
                f"do not use {missing}",
                f"avoid {missing}",
                f"omit {missing}",
                f"skip {missing}",
            )
        ):
            return True

        replacement_language = bool(
            re.search(r"\b(replace|substitute|replacement|instead of|in place of|workaround|alternative)\b", sentence)
        )
        if not replacement_language:
            return False

        names_missing_as_replaced_item = bool(
            re.search(rf"\b(replace|substitute)\b[^.\n;]{{0,50}}\b{re.escape(missing)}\b[^.\n;]{{0,60}}\b(with|using|for)\b", sentence)
            or re.search(rf"\b(instead of|in place of)\b[^.\n;]{{0,20}}\b{re.escape(missing)}\b", sentence)
            or re.search(rf"\b(substitute|replacement|alternative)\b[^.\n;]{{0,60}}\b(for|to)\b[^.\n;]{{0,30}}\b{re.escape(missing)}\b", sentence)
            or "missing ingredient" in sentence
        )
        if not names_missing_as_replaced_item:
            return False

        return _mentions_real_substitute(sentence) or "similar available ingredient" in sentence

    all_sentences = [
        sentence.strip()
        for sentence in re.split(r"[.\n;]+", text)
        if sentence.strip()
    ]
    sentences = [
        sentence
        for sentence in all_sentences
        if missing in sentence
    ]

    circular_substitute = re.compile(
        rf"\b(substitute|replace)\b[^.\n;]{{0,80}}\b(with|using)\b[^.\n;]{{0,40}}\b{re.escape(missing)}\b"
    )
    bad_action = re.compile(
        rf"\b(add|use|prepare|mix|put|place|cook|fry|heat|marinate|season|combine|pour)\b[^.\n;]{{0,80}}\b{re.escape(missing)}\b"
    )

    safe_replacement_seen = any(_is_safe_replacement_sentence(sentence) for sentence in all_sentences)
    for sentence in sentences:
        if _is_safe_replacement_sentence(sentence):
            continue

        if circular_substitute.search(sentence) and not _mentions_real_substitute(sentence):
            return True

    for sentence in sentences:
        if _is_safe_replacement_sentence(sentence):
            continue
        if bad_action.search(sentence):
            return True

    return False


def evaluate_llm_judge(
    model,
    goal: str,
    disruption: str,
    output_text: str,
    disrupted_step: Optional[str] = None,
    target_adaptation: Optional[str] = None,
    missing_ingredient: Optional[str] = None,
    suggested_substitute: Optional[str] = None,
    max_tokens: int = 400,
) -> Dict:
    missing_guidance = ''
    if missing_ingredient:
        missing_guidance = (
            f'Missing ingredient: {missing_ingredient}\n'
            f'Suggested substitute, if appropriate: {suggested_substitute or "not specified"}\n'
            f'Important grading rule: if the candidate tells the user to add or use {missing_ingredient} '
            'as though it is available, the score must be 0.0 unless it clearly gives a substitute or workaround.\n'
            f'Do not penalize a candidate merely for mentioning {missing_ingredient} in phrases like '
            f'"instead of {missing_ingredient}", "replace {missing_ingredient} with ...", '
            f'"as a substitute for {missing_ingredient}", or "omit {missing_ingredient}". Penalize if the '
            f'candidate says it will substitute or omit {missing_ingredient} but the final plan still depends '
            f'on adding, mixing, heating, preparing, or using {missing_ingredient} itself.\n'
        )
    prompt = (
        'You are grading an adaptation output for a procedural disruption task.\n'
        'Return strict JSON only: {"score": <0.0-1.0>, "notes": "<short note>"}.\n'
        'Score 1.0 only if the candidate directly handles the disruption, is feasible, safe, and preserves the goal.\n'
        'Score 0.0 if it ignores the disruption, repeats the original step without adaptation, or uses an unavailable item.\n'
        f'Goal: {goal}\\n'
        f'Disrupted step: {disrupted_step or "not provided"}\\n'
        f'Disruption: {disruption}\\n'
        f'{missing_guidance}'
        f'Reference target adaptation: {target_adaptation or "not provided"}\\n'
        f'Candidate output:\\n{output_text}'
    )
    raw = model.generate(prompt, max_tokens=max_tokens, temperature=0.0)
    try:
        parsed = _extract_json(raw)
        score = float(parsed.get('score', 0.0))
        score = max(0.0, min(1.0, score))
        notes = str(parsed.get('notes', ''))
        if missing_ingredient and _mentions_unavailable_item_as_available(output_text, missing_ingredient, suggested_substitute):
            score = 0.0
            notes = (
                f'Guardrail override: candidate appears to use unavailable ingredient '
                f'{missing_ingredient!r} instead of giving a substitute or workaround. Original judge notes: {notes}'
            )
        return {'llm_judge_score': score, 'llm_judge_notes': notes}
    except Exception:
        return {'llm_judge_score': None, 'llm_judge_notes': f'Failed to parse judge output: {raw[:200]}'}
