#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
PYTORCH_DAS_SPEC=${PYTORCH_DAS_SPEC:-torch}
CI_TEST_TIMEOUT=${CI_TEST_TIMEOUT:-60m}
HCUPROBE_TEST_TIMEOUT=${HCUPROBE_TEST_TIMEOUT:-300}
DTK_ENV_FILE=${DTK_ENV_FILE:-/opt/dtk/env.sh}

cd "${PROJECT_ROOT}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: HCU CI must run on Linux." >&2
    exit 1
fi

if [[ ! -f "${DTK_ENV_FILE}" ]]; then
    echo "ERROR: DTK environment file not found: ${DTK_ENV_FILE}" >&2
    exit 1
fi
set +u
# shellcheck disable=SC1090
source "${DTK_ENV_FILE}"
set -u

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"ERROR: Python 3.10 is required, got {sys.version.split()[0]}")
print(f"Python: {sys.version.split()[0]}")
PY

"${PYTHON_BIN}" -m pip install --disable-pip-version-check --upgrade pip

if ! "${PYTHON_BIN}" -c \
    "import torch; assert hasattr(torch, '__hcu_version__')" >/dev/null 2>&1; then
    if [[ -z "${PYTORCH_DAS_INDEX_URL:-}" ]]; then
        echo "ERROR: pytorch-das is not installed and PYTORCH_DAS_INDEX_URL is empty." >&2
        exit 1
    fi
    pip_args=(--index-url "${PYTORCH_DAS_INDEX_URL}")
    if [[ -n "${PYTORCH_DAS_TRUSTED_HOST:-}" ]]; then
        pip_args+=(--trusted-host "${PYTORCH_DAS_TRUSTED_HOST}")
    fi
    "${PYTHON_BIN}" -m pip install --disable-pip-version-check --upgrade --force-reinstall \
        "${pip_args[@]}" "${PYTORCH_DAS_SPEC}"
fi

"${PYTHON_BIN}" - <<'PY'
import torch

if not hasattr(torch, "__hcu_version__"):
    raise SystemExit("ERROR: installed torch is not a pytorch-das HCU build")
device_count = torch.cuda.device_count()
if device_count < 1:
    raise SystemExit("ERROR: pytorch-das cannot see an HCU device")
print(f"pytorch-das: {torch.__version__}")
print(f"HCU runtime: {torch.__hcu_version__}")
print(f"HCU devices visible: {device_count}")
PY

"${PYTHON_BIN}" -m pip install --disable-pip-version-check \
    build onnxruntime protobuf pytest pytest-cov pytest-timeout
"${PYTHON_BIN}" -m pip install --disable-pip-version-check .
"${PYTHON_BIN}" -m pip check

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import hcuprobe

print(f"hcu-probe: {importlib.metadata.version('hcu-probe')}")
print(f"hcuprobe import: {hcuprobe.__file__}")
PY

rm -rf dist
(
    cd /tmp
    "${PYTHON_BIN}" -m build --wheel \
        --outdir "${PROJECT_ROOT}/dist" "${PROJECT_ROOT}"
)

export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} --timeout=${HCUPROBE_TEST_TIMEOUT} --durations=20"
cd "${PROJECT_ROOT}/test/hcuprobe_test"
timeout --signal=TERM --kill-after=5m "${CI_TEST_TIMEOUT}" \
    "${PYTHON_BIN}" run_ut.py
