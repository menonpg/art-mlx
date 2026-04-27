#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
# install missing cudnn headers, DeepEP RDMA headers, and ninja build tools
apt-get update
apt-get install -y libcudnn9-headers-cuda-12 libibverbs-dev ninja-build

# Python dependencies are declared in pyproject.toml extras.
# Megatron setup still needs the shared backend extras, but the vLLM runtime now
# lives in its own project and venv under vllm_runtime/.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd "${repo_root}"
uv sync --extra backend --extra megatron --frozen --active
