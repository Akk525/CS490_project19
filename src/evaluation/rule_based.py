from typing import Dict

from src.dataset.schema import ProceduralExample
from src.evaluation.metrics import (
    adaptation_quality,
    constraint_violation,
    feasibility_score,
    helpfulness_score,
    semantic_similarity,
)


def evaluate_rule_based(example: ProceduralExample, output_text: str, method: str, model_name: str) -> Dict:
    return {
        'example_id': example.example_id,
        'method': method,
        'model': model_name,
        'constraint_violation': constraint_violation(output_text, example.disruption_type),
        'feasibility_score': feasibility_score(output_text),
        'adaptation_quality_score': adaptation_quality(output_text, example.disruption_description, example.current_state),
        'helpfulness_score': helpfulness_score(output_text, example.goal),
        'semantic_similarity_score': semantic_similarity(output_text, example.target_adaptation or ''),
        'notes': '',
    }
