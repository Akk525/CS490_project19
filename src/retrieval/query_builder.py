from src.dataset.schema import ProceduralExample


def _missing_context(example: ProceduralExample) -> str:
    missing = example.metadata.get('missing_ingredient')
    substitute = example.metadata.get('suggested_substitute')
    lines = []
    if missing:
        lines.append(f"Missing ingredient: {missing}")
    if substitute:
        lines.append(f"Known safe substitute: {substitute}")
    return "\n".join(lines)


def build_query(example: ProceduralExample, strategy: str) -> str:
    missing_context = _missing_context(example)
    missing_block = f"\n{missing_context}" if missing_context else ""
    if strategy == 'disruption_only':
        return (
            f"Goal: {example.goal}{missing_block}\n"
            f"Disruption: {example.disruption_type}. {example.disruption_description}\n"
            f"Step: {example.disrupted_step_text}"
        )
    if strategy == 'full_context':
        return (
            f"Goal: {example.goal}{missing_block}\n"
            f"Procedure: {' | '.join(example.full_procedure)}\n"
            f"Current state: {example.current_state}\n"
            f"Disruption: {example.disruption_description}"
        )
    return (
        f"Goal: {example.goal}{missing_block}\nCurrent state: {example.current_state}\n"
        f"Disrupted step: {example.disrupted_step_text}\nDisruption: {example.disruption_type} - {example.disruption_description}"
    )
