from typing import Dict, List, Optional, Tuple

from src.dataset.schema import ProceduralExample
from src.models.response_parser import parse_model_output
from src.prompting.multimodal_retrieval_augmented import build_multimodal_retrieval_augmented_prompt
from src.prompting.multimodal_structured import build_multimodal_structured_prompt
from src.prompting.retrieval_augmented import build_retrieval_augmented_prompt
from src.prompting.structured import build_structured_prompt
from src.prompting.vanilla import build_vanilla_prompt
from src.retrieval.retrieval_pipeline import run_retrieval


def _build_prompt(prompt_type: str, example: ProceduralExample, retrieved: Optional[List[Dict]]) -> str:
    if prompt_type == 'vanilla':
        return build_vanilla_prompt(example)
    if prompt_type == 'structured':
        return build_structured_prompt(example)
    if prompt_type == 'multimodal_structured':
        return build_multimodal_structured_prompt(example)
    if prompt_type == 'multimodal_retrieval_augmented':
        return build_multimodal_retrieval_augmented_prompt(example, retrieved or [])
    return build_retrieval_augmented_prompt(example, retrieved or [])


def _image_paths_for_prompt(prompt_type: str, example: ProceduralExample) -> Optional[List[str]]:
    if not prompt_type.startswith('multimodal_'):
        return None
    if not example.image_path:
        raise ValueError(f'Prompt type {prompt_type} requires image_path, but example {example.example_id} has none.')
    return [example.image_path]


def run_one_example(
    example: ProceduralExample,
    model,
    prompt_type: str,
    retrieval_cfg: Dict,
    retrieval_runtime: Optional[Dict],
    generation_params: Optional[Dict] = None,
) -> Tuple[Dict, Dict, Optional[Dict]]:
    retrieval_trace = None
    retrieved_ids: List[str] = []
    retrieved_examples: List[Dict] = []

    if retrieval_cfg.get('enabled') and retrieval_runtime:
        retrieval_trace = run_retrieval(
            example=example,
            strategy=retrieval_cfg['strategy'],
            k=int(retrieval_cfg['k']),
            candidate_pool=int(retrieval_cfg.get('candidate_pool', 20)),
            rerank_fallback_to_retrieval=bool(retrieval_cfg.get('rerank_fallback_to_retrieval', False)),
            embedder=retrieval_runtime['embedder'],
            index_matrix=retrieval_runtime['index_matrix'],
            library_rows=retrieval_runtime['library_rows'],
            reranker=retrieval_runtime.get('reranker') if retrieval_cfg.get('use_reranker') else None,
        )
        retrieved_examples = retrieval_trace['selected_examples']
        retrieved_ids = [x['example_id'] for x in retrieved_examples]

    prompt = _build_prompt(prompt_type, example, retrieved_examples)
    image_paths = _image_paths_for_prompt(prompt_type, example)
    gen_cfg = generation_params or {}
    raw_output = model.generate(
        prompt,
        max_tokens=int(gen_cfg.get('max_tokens', 512)),
        temperature=float(gen_cfg.get('temperature', 0.2)),
        image_paths=image_paths,
    )
    parsed = parse_model_output(raw_output)

    prompt_record = {
        'example_id': example.example_id,
        'prompt_type': prompt_type,
        'raw_prompt': prompt,
        'image_paths': image_paths or [],
    }
    generation_record = {
        'example_id': example.example_id,
        'source_dataset': example.source_dataset,
        'model_name': retrieval_runtime['generation_model_name'] if retrieval_runtime else retrieval_cfg.get('generation_model_name'),
        'prompt_type': prompt_type,
        'image_paths': image_paths or [],
        'retrieval_enabled': retrieval_cfg.get('enabled', False),
        'retrieval_k': retrieval_cfg.get('k', 0),
        'reranker_enabled': retrieval_cfg.get('use_reranker', False),
        'retrieved_example_ids': retrieved_ids,
        'raw_prompt': prompt,
        'raw_model_output': raw_output,
        'parsed_output': parsed,
    }
    return generation_record, prompt_record, retrieval_trace
