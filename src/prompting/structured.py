from src.dataset.schema import ProceduralExample


def build_structured_prompt(ex: ProceduralExample) -> str:
    missing_ingredient = ex.metadata.get('missing_ingredient')
    suggested_substitute = ex.metadata.get('suggested_substitute')
    missing_line = ''
    if missing_ingredient:
        missing_line = f'Missing ingredient to adapt around: {missing_ingredient}\n'
        if suggested_substitute:
            missing_line += f'Known safe substitute option: {suggested_substitute}\n'
    return (
        'Task: Adapt a disrupted procedure.\n'
        'Return JSON with keys: diagnosis, adaptation_steps, safety_notes, final_plan.\n'
        'If an ingredient is missing, do not instruct the user to add that unavailable ingredient. Use a practical substitute or adapt the step.\n'
        f'{missing_line}'
        f'Goal: {ex.goal}\n'
        f'Procedure: {ex.full_procedure}\n'
        f'Current state: {ex.current_state}\n'
        f'Disrupted step index: {ex.disrupted_step_index}\n'
        f'Disrupted step text: {ex.disrupted_step_text}\n'
        f'Disruption details: {ex.disruption_description}'
    )
