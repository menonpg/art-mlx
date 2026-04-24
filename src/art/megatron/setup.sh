#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
# install missing cudnn headers, DeepEP RDMA headers, and ninja build tools
if ! dpkg-query -W libcudnn9-headers-cuda-12 libibverbs-dev ninja-build >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then
        apt-get update
        apt-get install -y libcudnn9-headers-cuda-12 libibverbs-dev ninja-build
    elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y libcudnn9-headers-cuda-12 libibverbs-dev ninja-build
    else
        echo "Missing required system packages; install libcudnn9-headers-cuda-12, libibverbs-dev, and ninja-build or rerun with passwordless sudo." >&2
        exit 1
    fi
fi

# Python dependencies are declared in pyproject.toml extras.
# Keep backend + megatron together so setup does not prune runtime deps (e.g. vllm).
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd "${repo_root}"
uv_bin="${HOME}/.local/bin/uv"
if [[ -x "${uv_bin}" ]]; then
    "${uv_bin}" sync --extra backend --extra megatron --frozen --active
else
    uv sync --extra backend --extra megatron --frozen --active
fi
