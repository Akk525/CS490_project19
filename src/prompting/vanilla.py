from src.dataset.schema import ProceduralExample


def build_vanilla_prompt(ex: ProceduralExample) -> str:
    return (
        'You are an expert procedural assistant.\n'
        f'Goal: {ex.goal}\n'
        f'Current state: {ex.current_state}\n'
        f'Disrupted step: {ex.disrupted_step_text}\n'
        f'Disruption: {ex.disruption_type} - {ex.disruption_description}\n'
        'Provide an adapted continuation with concise rationale.'
    )
