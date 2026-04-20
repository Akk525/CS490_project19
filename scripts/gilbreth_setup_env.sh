#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
HF_HOME="${HF_HOME:-${CLUSTER_SCRATCH:-${PROJECT_DIR}}/huggingface}"
PYTHON_BIN="${PYTHON_BIN:-}"

cd "${PROJECT_DIR}"
module load ffmpeg/6.1.1 || true
module load cuda/12.6.0 || true

if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3.11 python3.10 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No Python interpreter found."
  exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"vLLM requires Python >= 3.10 for this project setup; found {sys.version.split()[0]}. "
        "Load a newer Python module or rerun with PYTHON_BIN=/path/to/python3.10."
    )
PY

echo "Using Python ${PYTHON_VERSION}: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p "${HF_HOME}/hub" "${HF_HOME}/transformers"

cat <<EOF
Environment ready.

Activate it with:
  cd ${PROJECT_DIR}
  source .venv/bin/activate

Recommended cache exports:
  export HF_HOME=${HF_HOME}
  export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
  export TRANSFORMERS_CACHE=${HF_HOME}/transformers
EOF
