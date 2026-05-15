#!/usr/bin/env bash
set -euo pipefail

log() {
    echo "[art-vllm-runtime-setup] $*"
}

fail() {
    echo "[art-vllm-runtime-setup] $*" >&2
    exit 1
}

detect_cuda_home() {
    if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
        echo "${CUDA_HOME}"
        return
    fi
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        echo /usr/local/cuda
        return
    fi

    local candidate latest
    latest=""
    for candidate in /usr/local/cuda-*; do
        if [ -x "${candidate}/bin/nvcc" ]; then
            latest="${candidate}"
        fi
    done
    if [ -n "${latest}" ]; then
        echo "${latest}"
        return
    fi
    fail "Could not find CUDA nvcc. Set CUDA_HOME to the CUDA toolkit path."
}

detect_cuda_major() {
    local cuda_home="$1"
    local major
    major="$("${cuda_home}/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1)"
    if [ -z "${major}" ]; then
        fail "Could not parse CUDA major version from ${cuda_home}/bin/nvcc --version."
    fi
    echo "${major}"
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

cuda_home="$(detect_cuda_home)"
cuda_major="$(detect_cuda_major "${cuda_home}")"
export CUDA_HOME="${cuda_home}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

case "${cuda_major}" in
    12)
        runtime_extra="cuda12"
        torch_backend="cu128"
        ;;
    13)
        runtime_extra="cuda13"
        torch_backend="cu130"
        ;;
    *)
        fail "Unsupported CUDA major version ${cuda_major}; expected CUDA 12 or 13."
        ;;
esac

uv_bin="uv"
if [ -x "${HOME}/.local/bin/uv" ]; then
    uv_bin="${HOME}/.local/bin/uv"
fi

log "CUDA_HOME=${CUDA_HOME}"
log "Using uv extra: ${runtime_extra}"
log "Using torch backend: ${torch_backend}"
UV_TORCH_BACKEND="${torch_backend}" "${uv_bin}" sync --extra "${runtime_extra}" --frozen

".venv/bin/python" - <<'PY'
import torch
import vllm
import art_vllm_runtime

print(f"[art-vllm-runtime-setup] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[art-vllm-runtime-setup] vllm={vllm.__version__}")
print(f"[art-vllm-runtime-setup] runtime={art_vllm_runtime.__name__}")
PY
