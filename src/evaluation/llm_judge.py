import json
from typing import Dict


def evaluate_llm_judge_disabled() -> Dict:
    return {'llm_judge_score': None, 'llm_judge_notes': 'LLM judge disabled by config.'}


def evaluate_llm_judge(model, goal: str, disruption: str, output_text: str) -> Dict:
    prompt = (
        'You are grading an adaptation output for a procedural disruption task. '
        'Return strict JSON: {"score": <0.0-1.0>, "notes": "<short note>"}.\\n'
        f'Goal: {goal}\\n'
        f'Disruption: {disruption}\\n'
        f'Candidate output:\\n{output_text}'
    )
    raw = model.generate(prompt, max_tokens=200, temperature=0.0)
    try:
        parsed = json.loads(raw.strip())
        score = float(parsed.get('score', 0.0))
        score = max(0.0, min(1.0, score))
        return {'llm_judge_score': score, 'llm_judge_notes': str(parsed.get('notes', ''))}
    except Exception:
        return {'llm_judge_score': None, 'llm_judge_notes': f'Failed to parse judge output: {raw[:200]}'}
