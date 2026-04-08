from typing import Dict, List

from src.dataset.schema import ProceduralExample


def build_multimodal_retrieval_augmented_prompt(ex: ProceduralExample, retrieved: List[Dict]) -> str:
    cases = []
    for i, r in enumerate(retrieved, start=1):
        cases.append(
            f"Case {i}: goal={r['goal']} | disruption={r['disruption_type']} | adapted={r.get('target_adaptation', '')}"
        )
    return (
        'Task: Adapt a disrupted procedure using the provided image of the current state and analogous retrieved cases.\n'
        'Return JSON with keys: diagnosis, adaptation_steps, justification, final_plan.\n'
        'The image shows the current workspace or procedural state and should be used when it clarifies missing tools, ingredients, or step progress.\n'
        f'Goal: {ex.goal}\n'
        f'Current state: {ex.current_state}\n'
        f'Disrupted step: {ex.disrupted_step_text}\n'
        f'Disruption: {ex.disruption_type} - {ex.disruption_description}\n'
        f'Retrieved cases:\n' + '\n'.join(cases)
    )
