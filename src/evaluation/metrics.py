def constraint_violation(output_text: str, disruption_type: str = '') -> float:
    text = output_text.lower()
    hard_fail = ['impossible', 'cannot proceed', 'no solution', 'do nothing', 'restart everything']
    penalty = 0.0
    if any(x in text for x in hard_fail):
        penalty += 0.7
    # Disruption-aware expectations.
    if disruption_type == 'missing_tool' and not any(x in text for x in ['alternative tool', 'substitute', 'manual']):
        penalty += 0.3
    if disruption_type == 'missing_ingredient' and not any(x in text for x in ['substitute', 'replacement']):
        penalty += 0.3
    if disruption_type == 'environmental_constraint' and not any(x in text for x in ['constraint', 'limited', 'adapt']):
        penalty += 0.2
    return min(1.0, penalty)


def feasibility_score(output_text: str) -> float:
    tokens = output_text.lower()
    score = 0.2
    if 'alternative' in tokens or 'substitute' in tokens:
        score += 0.3
    if 'step' in tokens:
        score += 0.2
    if 'safety' in tokens:
        score += 0.1
    if len(tokens.split()) > 40:
        score += 0.2
    return min(score, 1.0)


def adaptation_quality(output_text: str, disruption_description: str, current_state: str = '') -> float:
    output_tokens = set(output_text.lower().split())
    dis_tokens = set(disruption_description.lower().split())
    state_tokens = set(current_state.lower().split())
    overlap_dis = len(output_tokens & dis_tokens) / max(1, len(dis_tokens))
    overlap_state = len(output_tokens & state_tokens) / max(1, len(state_tokens))
    structure_bonus = 0.2 if any(x in output_text.lower() for x in ['step 1', 'first', 'then', 'finally']) else 0.0
    return min(1.0, 0.5 * overlap_dis + 0.3 * overlap_state + structure_bonus)


def helpfulness_score(output_text: str, goal: str) -> float:
    overlap = len(set(output_text.lower().split()) & set(goal.lower().split()))
    length_bonus = min(0.4, len(output_text.split()) / 150.0)
    return min(1.0, overlap / 8.0 + length_bonus)


def semantic_similarity(output: str, target: str) -> float:
    if not target:
        return 0.0
    o = set(output.lower().split())
    t = set(target.lower().split())
    if not o or not t:
        return 0.0
    return len(o & t) / len(o | t)
