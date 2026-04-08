from src.dataset.schema import ProceduralExample


def build_multimodal_structured_prompt(ex: ProceduralExample) -> str:
    return (
        'Task: Adapt a disrupted procedure using the provided image of the current state.\n'
        'Return JSON with keys: diagnosis, adaptation_steps, safety_notes, final_plan.\n'
        'Use the image when it clarifies tool availability, ingredients, or what step has already been completed.\n'
        f'Goal: {ex.goal}\n'
        f'Procedure: {ex.full_procedure}\n'
        f'Current state: {ex.current_state}\n'
        f'Disrupted step index: {ex.disrupted_step_index}\n'
        f'Disrupted step text: {ex.disrupted_step_text}\n'
        f'Disruption details: {ex.disruption_description}'
    )
