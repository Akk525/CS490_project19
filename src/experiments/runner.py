from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
from tqdm import tqdm

from src.dataset.schema import ProceduralExample
from src.evaluation.llm_judge import evaluate_llm_judge, evaluate_llm_judge_disabled
from src.evaluation.rule_based import evaluate_rule_based
from src.experiments.generation_pipeline import run_one_example
from src.experiments.manifest import build_manifest
from src.experiments.tracking import register_run, save_manifest
from src.models.model_registry import validate_model_names
from src.models.school_server_generation import SchoolServerGeneration
from src.retrieval.factory import build_embedder, build_reranker
from src.utils.io import append_jsonl, read_jsonl, write_json
from src.utils.timestamps import utc_now_iso


def _run_id(cfg: Dict) -> str:
    exp_name = cfg.get('experiment', {}).get('name', 'run')
    return f"{exp_name}_{utc_now_iso().replace(':', '-')}"


def _filter_examples(rows: list[Dict], cfg: Dict) -> list[Dict]:
    filters = cfg.get('experiment', {}).get('filters', {}) or {}
    if not filters:
        return rows

    out = rows
    source_dataset = filters.get('source_dataset')
    if source_dataset:
        out = [r for r in out if r.get('source_dataset') == source_dataset]

    disruption_type = filters.get('disruption_type')
    if disruption_type:
        out = [r for r in out if r.get('disruption_type') == disruption_type]

    disruption_modality = filters.get('disruption_modality')
    if disruption_modality:
        out = [r for r in out if (r.get('metadata') or {}).get('disruption_modality') == disruption_modality]

    if filters.get('require_image'):
        out = [r for r in out if r.get('image_path')]

    max_examples = filters.get('max_examples')
    if max_examples:
        out = out[: int(max_examples)]

    if not out:
        raise ValueError(f'Experiment filters removed all examples: {filters}')
    return out


def run_experiment(cfg: Dict, project_dir: Path, resume_run_id: Optional[str] = None) -> str:
    validate_model_names(
        cfg['models']['generation_model'],
        cfg['models']['embedding_model'],
        cfg['models']['reranker_model'],
    )

    benchmark_path = project_dir / cfg['paths']['benchmark']
    split_name = cfg.get('experiment', {}).get('split', 'test')
    split_path = project_dir / cfg['paths']['splits_dir'] / f'{split_name}.jsonl'
    examples_raw = read_jsonl(split_path if split_path.exists() else benchmark_path)
    examples_raw = _filter_examples(examples_raw, cfg)
    examples = [ProceduralExample(**x) for x in examples_raw]

    run_id = resume_run_id or _run_id(cfg)
    run_dir = project_dir / 'outputs' / 'generations' / run_id
    eval_dir = project_dir / 'outputs' / 'evaluations' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not resume_run_id:
        with (run_dir / 'config_snapshot.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        manifest = build_manifest(run_id, cfg, project_dir)
        save_manifest(manifest, run_dir)
        register_run(manifest, project_dir / 'outputs' / 'manifests' / 'run_registry.jsonl')

    generation_model = SchoolServerGeneration(cfg['backends']['generation'], cfg['models']['generation_model'])
    llm_judge_model = None
    if cfg.get('evaluation', {}).get('enable_llm_judge'):
        llm_judge_model = SchoolServerGeneration(cfg['backends']['generation'], cfg['evaluation']['llm_judge_model'])

    retrieval_runtime = {'generation_model_name': cfg['models']['generation_model']}
    if cfg['retrieval']['enabled']:
        library_rows = read_jsonl(project_dir / cfg['paths']['retrieval_library'])
        index = np.load(project_dir / cfg['paths']['retrieval_index'])
        embedder = build_embedder(cfg['backends']['embedding'], cfg['models']['embedding_model'])
        reranker = build_reranker(
            cfg['backends']['reranker'],
            cfg['models']['reranker_model'],
        ) if cfg['retrieval']['use_reranker'] else None
        retrieval_runtime.update(
            {
                'embedder': embedder,
                'index_matrix': index['embeddings'],
                'library_rows': library_rows,
                'reranker': reranker,
            }
        )

    existing_generations = {}
    generations_path = run_dir / 'generations.jsonl'
    if generations_path.exists():
        existing_generations = {row['example_id']: row for row in read_jsonl(generations_path)}

    completed_eval_ids = set()
    evaluation_path = eval_dir / 'evaluation_results.jsonl'
    if evaluation_path.exists():
        completed_eval_ids = {row['example_id'] for row in read_jsonl(evaluation_path)}

    for ex in tqdm(examples, desc=f'Running {run_id}'):
        if ex.example_id in completed_eval_ids:
            continue

        if ex.example_id in existing_generations:
            gen_rec = existing_generations[ex.example_id]
            prompt_rec = None
            retrieval_trace = None
        else:
            gen_rec, prompt_rec, retrieval_trace = run_one_example(
                example=ex,
                model=generation_model,
                prompt_type=cfg['prompting']['type'],
                retrieval_cfg=cfg['retrieval'],
                retrieval_runtime=retrieval_runtime,
                generation_params=cfg.get('prompting', {}),
            )
            gen_rec.update({'run_id': run_id, 'timestamp': utc_now_iso()})
            append_jsonl(run_dir / 'generations.jsonl', gen_rec)
            append_jsonl(run_dir / 'prompts.jsonl', {'run_id': run_id, **prompt_rec})
            if retrieval_trace:
                append_jsonl(run_dir / 'retrieval_traces.jsonl', {'run_id': run_id, 'example_id': ex.example_id, **retrieval_trace})

        eval_rec = evaluate_rule_based(ex, gen_rec['raw_model_output'], cfg['prompting']['type'], cfg['models']['generation_model'])
        if llm_judge_model:
            eval_rec.update(
                evaluate_llm_judge(
                    llm_judge_model,
                    goal=ex.goal,
                    disruption=ex.disruption_description,
                    output_text=gen_rec['raw_model_output'],
                    disrupted_step=ex.disrupted_step_text,
                    target_adaptation=ex.target_adaptation,
                    missing_ingredient=ex.metadata.get('missing_ingredient'),
                    suggested_substitute=ex.metadata.get('suggested_substitute'),
                    max_tokens=int(cfg.get('evaluation', {}).get('llm_judge_max_tokens', 400)),
                )
            )
        else:
            eval_rec.update(evaluate_llm_judge_disabled())
        eval_rec.update({'run_id': run_id, 'source_dataset': ex.source_dataset, 'disruption_type': ex.disruption_type, 'domain': ex.domain})
        append_jsonl(eval_dir / 'evaluation_results.jsonl', eval_rec)

    write_json(eval_dir / 'summary.json', {'run_id': run_id, 'status': 'complete'})
    return run_id
