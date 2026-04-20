#!/usr/bin/env bash
#SBATCH --job-name=ipa-vllm-vlm
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=ai-forge
#SBATCH --partition=a10
#SBATCH --mem=165G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a10
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
CONFIG="${CONFIG:-configs/experiments/gilbreth_qwen2vl_2b_smoke_with_judge.yaml}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-}"
REBUILD_BENCHMARK="${REBUILD_BENCHMARK:-0}"
CREATE_VISUAL_DISRUPTIONS="${CREATE_VISUAL_DISRUPTIONS:-1}"
DOWNLOAD_YOUCOOK2_FRAMES="${DOWNLOAD_YOUCOOK2_FRAMES:-0}"
YOUCOOK2_FRAME_LIMIT="${YOUCOOK2_FRAME_LIMIT:-}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_OPENAI_BASE_URL="${VLLM_OPENAI_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
VLLM_OPENAI_API_KEY="${VLLM_OPENAI_API_KEY:-EMPTY}"
VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-900}"
HF_HOME="${HF_HOME:-${CLUSTER_SCRATCH:-${PROJECT_DIR}}/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CLUSTER_SCRATCH:-/scratch/gilbreth/${USER}}/envs/ipa-vllm-py311}"

module load external || true
module load conda/2025.09 || true
module load ffmpeg/6.1.1 || true
module load cuda/12.6.0 || true

cd "${PROJECT_DIR}"
mkdir -p outputs/slurm outputs/vllm

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ -d "${CONDA_ENV_PATH}" ]]; then
    conda activate "${CONDA_ENV_PATH}"
    export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
  fi
fi

if [[ -f "${PROJECT_DIR}/.venv/bin/activate" ]]; then
  source "${PROJECT_DIR}/.venv/bin/activate"
fi

export VLLM_MODEL
export VLLM_HOST
export VLLM_PORT
export VLLM_OPENAI_BASE_URL
export VLLM_OPENAI_API_KEY
export HF_HOME
export HUGGINGFACE_HUB_CACHE
export TRANSFORMERS_CACHE

if [[ "${DOWNLOAD_YOUCOOK2_FRAMES}" == "1" ]]; then
  frame_args=(--raw-dir data/raw/youcook2 --output-dir data/raw/youcook2/frames)
  if [[ -n "${YOUCOOK2_FRAME_LIMIT}" ]]; then
    frame_args+=(--limit "${YOUCOOK2_FRAME_LIMIT}")
  fi
  python scripts/download_youcook2_frames.py "${frame_args[@]}"
  python scripts/ingest_youcook2.py --config configs/experiments/vllm_benchmark_missing_ingredient_modalities.yaml
fi

VLLM_LOG="outputs/vllm/${SLURM_JOB_ID}.log"
scripts/serve_vllm_vlm.sh "${VLLM_MODEL}" > "${VLLM_LOG}" 2>&1 &
VLLM_PID="$!"

cleanup() {
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
until python scripts/test_openai_proxy.py --model "${VLLM_MODEL}" --prompt "health check"; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM process exited before the health check succeeded."
    echo "Last 120 lines from ${VLLM_LOG}:"
    tail -120 "${VLLM_LOG}" || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT_SECONDS} seconds."
    echo "Last 120 lines from ${VLLM_LOG}:"
    tail -120 "${VLLM_LOG}" || true
    exit 1
  fi
  sleep 15
done

if [[ "${REBUILD_BENCHMARK}" == "1" ]]; then
  if [[ -z "${BENCHMARK_CONFIG}" ]]; then
    echo "REBUILD_BENCHMARK=1 requires BENCHMARK_CONFIG."
    exit 1
  fi
  python scripts/build_benchmark.py --config "${BENCHMARK_CONFIG}"
fi

if [[ "${CREATE_VISUAL_DISRUPTIONS}" == "1" ]]; then
  python scripts/create_visual_disruption_images.py
fi

python scripts/run_experiments.py --config "${CONFIG}"
