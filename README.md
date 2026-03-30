# Improving Improvisational Reasoning in Language Models for Procedural Adaptation

Research-grade local codebase for building a real procedural adaptation benchmark (YouCook2 + WikiHow), running retrieval-augmented adaptation experiments against school-server-hosted models, and persisting all run artifacts.

## Project Motivation
This project evaluates whether retrieval-augmented prompting improves procedural adaptation under disruptions (missing tools/ingredients, step failures, environmental constraints, etc.).

## Architecture
- `src/dataset`: ingestion + normalization for YouCook2 and WikiHow
- `src/disruptions`: disruption taxonomy, templates, generation, validation
- `src/retrieval`: embedding, indexing, similarity search, optional reranking
- `src/models`: generation adapters for school server model endpoints
  - robust response normalization for heterogeneous backend payloads
- `src/prompting`: deterministic prompt builders (vanilla, structured, retrieval-augmented)
- `src/experiments`: run orchestration, manifest tracking, persistence
- `src/evaluation`: per-example metrics and aggregated summaries
- `src/analysis`: comparison and error-analysis table exports

## Supported Datasets
Primary: YouCook2  
Secondary: WikiHow

### Expected Raw Data Layout
```text
improv_procedural_adaptation/
  data/raw/
    youcook2/
      annotations/
        youcookii_annotations_trainval.json
        youcook2_annotations_trainval.json               # accepted alternate name
        youcookii_annotations_test_segments_only.json   # optional
        label_foodtype.csv                                # recipe_type id → goal label (recommended)
      metadata/
        youcookii_videos_trainval.json                  # optional
    wikihow/
      procedures.jsonl                                  # preferred
      procedures.json                                   # supported
      wikihow.csv                                       # supported
```

## Ingest Datasets
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Copy raw files into `data/raw/youcook2` and `data/raw/wikihow`.
   - YouCook2 loader auto-detects common annotation filename variants in both `data/raw/youcook2/annotations/` and `data/raw/youcook2/`.
3. Run ingestion:
```bash
python scripts/ingest_youcook2.py --config configs/local.yaml
python scripts/ingest_wikihow.py --config configs/local.yaml
```

## Build Benchmark and Retrieval Library
```bash
python scripts/build_benchmark.py --config configs/experiments/main.yaml
python scripts/build_retrieval_index.py --config configs/experiments/main.yaml
```

## Configure School Server Access
1. Copy `.env.example` to `.env`.
2. Set endpoint and auth values for your backend mode.
   - for Purdue AIFORGE-style proxy: `SCHOOL_OPENAI_BASE_URL=http://aiforge.cs.purdue.edu:8002/v1`
   - set `SCHOOL_OPENAI_API_KEY` to your issued API key
3. Edit `configs/server.yaml` (or override fields in experiment configs):
   - generation adapter backend (`openai_compatible`, `http`, `cli`)
   - embedding backend
   - reranker backend
   - model names
   - optional LLM judge model (`evaluation.enable_llm_judge`, `evaluation.llm_judge_model`)

Quick connection check:
```bash
python scripts/test_openai_proxy.py --model gemma3:4b-it-q8_0 --prompt "Hello!"
```

## Run Experiments
Main run:
```bash
python scripts/run_experiments.py --config configs/experiments/main.yaml
```

Ablation (k, reranker, strategies):
```bash
python scripts/run_ablation.py --config configs/experiments/ablation_k.yaml
python scripts/run_ablation.py --config configs/experiments/retrieval_strategies.yaml
```

## Evaluate and Summarize
```bash
python scripts/evaluate_results.py --run-id <run_id> --config configs/experiments/main.yaml
python scripts/summarize_results.py --config configs/experiments/model_comparison.yaml
```

When `evaluation.enable_llm_judge=true`, per-example evaluation rows include `llm_judge_score` and `llm_judge_notes`.

Additional research-grade analysis outputs:
- bootstrap confidence intervals in per-run summaries (`summary_metrics_ci.csv`)
- paired permutation significance tests vs vanilla (`outputs/summaries/paired_significance_tests.csv`)
- failure taxonomy exports (`outputs/analysis/failure_taxonomy_counts.csv`, `outputs/analysis/failure_taxonomy_examples.csv`)

## Gold Subset Workflow

Use this to export a **balanced, review-friendly candidate set** from a split or full benchmark, curate it in a spreadsheet, and rebuild a **final gold JSONL** for trustworthy final evaluation.

### 1. Export candidates

Defaults: **test** split, `source_dataset=youcook2`, `domain=cooking`, **25** examples per `disruption_type`, reproducible `--seed`.

```bash
python scripts/export_gold_candidates.py \
  --input data/splits/test.jsonl \
  --output data/processed/gold_candidates \
  --source-dataset youcook2 \
  --domain cooking \
  --n-per-type 25 \
  --seed 42
```

This writes:

- `data/processed/gold_candidates.jsonl` — full benchmark rows plus default review fields  
- `data/processed/gold_candidates.csv` — columns for manual review (open in Excel/Sheets)  
- `data/processed/gold_candidates_export_stats.json` — sampling stats per disruption type  

Use `--input path/to/benchmark.jsonl` or another split (`train.jsonl`, `dev.jsonl`) if needed. Pass `--source-dataset ""` or `--domain ""` to disable those filters.

### 2. Review the CSV

Edit **`review_status`** for each row:

| `review_status` | Meaning |
|-----------------|--------|
| `pending` | Not reviewed — **excluded** from gold (build warns) |
| `accept` | Include — keep `target_adaptation` from source unless you fill `reviewed_target_adaptation` |
| `edit` | Include — **must** set non-empty `reviewed_target_adaptation` (replaces gold target) |
| `reject` | Exclude from gold |

Optional: `review_notes`, `quality_tier` (informational in export; final gold sets `quality_tier` to `"gold"`).

### 3. Build the final gold subset

Point `--source-jsonl` at the **same** file the candidates came from (so full fields merge correctly), and `--reviewed-csv` at your edited sheet.

```bash
python scripts/build_gold_subset.py \
  --reviewed-csv data/processed/gold_candidates.csv \
  --source-jsonl data/splits/test.jsonl
```

Outputs:

- `data/processed/gold_subset.jsonl` — final gold benchmark records (`quality_tier: gold`, `manually_reviewed: true`, …)  
- `data/processed/gold_subset_reviewed.csv` — per-row audit (resolution, final target, etc.)  
- `outputs/summaries/gold_subset_summary.json` — counts, per-type breakdowns, warnings  
- `outputs/summaries/gold_subset_summary.csv` — flat metrics table  

Override paths with `--output-jsonl`, `--output-csv`, `--summary-dir` if needed.

### 4. Using gold for evaluation

Point your experiment split or a dedicated config at `data/processed/gold_subset.jsonl` (copy/rename or add a small script that loads this file as the example list) for final model runs and qualitative analysis.

## Output Layout
```text
outputs/
  generations/<run_id>/
    manifest.json
    config_snapshot.yaml
    generations.jsonl
    prompts.jsonl
    retrieval_traces.jsonl
  evaluations/<run_id>/
    evaluation_results.jsonl
    summary.json
    summary.csv
  summaries/
    model_comparison.csv
    ablation_k.csv
    retrieval_strategy.csv
  logs/<run_id>.log
  manifests/run_registry.jsonl
data/processed/
  gold_candidates.jsonl
  gold_candidates.csv
  gold_candidates_export_stats.json
  gold_subset.jsonl
  gold_subset_reviewed.csv
```

## Migration to School Server
1. Copy project root with configs, scripts, and `src/`.
2. Place raw datasets on server in matching `data/raw/...` paths or update config paths.
3. Set server environment variables in `.env`.
4. Run ingestion/build steps once on the server.
5. Launch experiments with `configs/server.yaml` merged into your experiment config.

## First Config Files to Edit
1. `configs/server.yaml`
2. `configs/experiments/main.yaml`
3. `.env`
