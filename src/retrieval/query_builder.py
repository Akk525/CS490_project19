from src.dataset.schema import ProceduralExample


def build_query(example: ProceduralExample, strategy: str) -> str:
    if strategy == 'disruption_only':
        return f"Goal: {example.goal}\nDisruption: {example.disruption_type}. {example.disruption_description}\nStep: {example.disrupted_step_text}"
    if strategy == 'full_context':
        return f"Goal: {example.goal}\nProcedure: {' | '.join(example.full_procedure)}\nCurrent state: {example.current_state}\nDisruption: {example.disruption_description}"
    return (
        f"Goal: {example.goal}\nCurrent state: {example.current_state}\n"
        f"Disrupted step: {example.disrupted_step_text}\nDisruption: {example.disruption_type} - {example.disruption_description}"
    )
