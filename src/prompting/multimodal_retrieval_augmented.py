from typing import Dict, List

from src.dataset.schema import ProceduralExample


def build_multimodal_retrieval_augmented_prompt(ex: ProceduralExample, retrieved: List[Dict]) -> str:
    disruption_modality = ex.metadata.get('disruption_modality', 'text')
    missing_ingredient = ex.metadata.get('missing_ingredient')
    suggested_substitute = ex.metadata.get('suggested_substitute')
    missing_line = ''
    if missing_ingredient:
        missing_line = f'Missing ingredient to adapt around: {missing_ingredient}\n'
        if suggested_substitute:
            missing_line += f'Known safe substitute option: {suggested_substitute}\n'
    cases = []
    for i, r in enumerate(retrieved, start=1):
        cases.append(
            f"Case {i}: goal={r['goal']} | disruption={r['disruption_type']} | adapted={r.get('target_adaptation', '')}"
        )
    return (
        'Task: Adapt a disrupted procedure using the provided image of the current state and analogous retrieved cases.\n'
        'Return JSON with keys: diagnosis, adaptation_steps, justification, final_plan.\n'
        'The image shows the current workspace or procedural state and should be used when it clarifies missing tools, ingredients, or step progress.\n'
        'If an ingredient is missing, do not instruct the user to add that unavailable ingredient. Use a practical substitute or adapt the step.\n'
        f'Disruption modality: {disruption_modality}\n'
        f'{missing_line}'
        f'Goal: {ex.goal}\n'
        f'Current state: {ex.current_state}\n'
        f'Disrupted step: {ex.disrupted_step_text}\n'
        f'Disruption: {ex.disruption_type} - {ex.disruption_description}\n'
        f'Retrieved cases:\n' + '\n'.join(cases)
    )
