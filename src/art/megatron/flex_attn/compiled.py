"""Compiled flex attention entrypoints."""

import math
from typing import Any, TypeAlias, cast

import torch
from torch.nn.attention.flex_attention import (
    AuxRequest,
    FlexKernelOptions,
    flex_attention,
)

from art.megatron.flex_attn.flash_dlse_patch import apply_flash_flex_dlse_patch

apply_flash_flex_dlse_patch()


# Tests may patch the backend; SM100/SM110 use TRITON until FA4 coverage is sufficient.
_FORCED_FLEX_BACKEND = "FLASH"
_FLASH_LSE_RESCALE = math.log(2.0)
SparseBlockSize: TypeAlias = int | tuple[int, int]
_FORCED_FLEX_KERNEL_OPTIONS = cast(FlexKernelOptions, {"BACKEND": _FORCED_FLEX_BACKEND})


def normalize_flex_lse(lse: torch.Tensor) -> torch.Tensor:
    if _runtime_flex_backend(lse.device) != "FLASH":
        return lse
    return lse / _FLASH_LSE_RESCALE


def _runtime_flex_backend(device: torch.device) -> str:
    if _FORCED_FLEX_BACKEND != "FLASH" or device.type != "cuda":
        return _FORCED_FLEX_BACKEND
    with torch.cuda.device(device):
        major, _minor = torch.cuda.get_device_capability(device)
    return "TRITON" if major in {10, 11} else _FORCED_FLEX_BACKEND


def _runtime_flex_kernel_options(device: torch.device) -> FlexKernelOptions:
    backend = _runtime_flex_backend(device)
    if backend == _FORCED_FLEX_BACKEND:
        return _FORCED_FLEX_KERNEL_OPTIONS
    return cast(FlexKernelOptions, {"BACKEND": backend})


def normalize_sparse_block_size(block_size: SparseBlockSize) -> tuple[int, int]:
    if isinstance(block_size, tuple):
        if len(block_size) != 2:
            raise RuntimeError(f"Expected 2D sparse block size, got {block_size!r}")
        return int(block_size[0]), int(block_size[1])
    value = int(block_size)
    return value, value


def flash_sparse_block_size_for_head_dim(
    *,
    head_dim: int,
    head_dim_v: int,
    device: torch.device,
) -> tuple[int, int]:
    if _runtime_flex_backend(device) != "FLASH":
        return (128, 128)
    with torch.cuda.device(device):
        major, _minor = torch.cuda.get_device_capability(device)
    if major != 9:
        return (128, 128)
    if int(head_dim) <= 128:
        return (128, 128)
    if int(head_dim) <= 192:
        return (128, 96)
    return (128, 64)


def _forced_flex_attention_dense(
    q,
    k,
    v,
    *,
    block_mask,
    scale,
    enable_gqa,
    return_aux: AuxRequest | None = None,
):
    return flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        kernel_options=_runtime_flex_kernel_options(q.device),
        return_aux=return_aux,
    )


def _forced_flex_attention_sparse(
    q,
    k,
    v,
    *,
    block_mask,
    scale,
    enable_gqa,
    return_aux: AuxRequest | None = None,
):
    return flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        kernel_options=_runtime_flex_kernel_options(q.device),
        return_aux=return_aux,
    )


def select_sparse_execution_family(
    *,
    is_local_stage: bool,
    q_len: int,
    k_len: int,
    block_size: SparseBlockSize,
) -> tuple[int, int, str]:
    del is_local_stage
    q_block, k_block = normalize_sparse_block_size(block_size)
    target_q_len = (
        0 if int(q_len) <= 0 else ((int(q_len) + q_block - 1) // q_block) * q_block
    )
    target_k_len = (
        0 if int(k_len) <= 0 else ((int(k_len) + k_block - 1) // k_block) * k_block
    )
    return int(target_q_len), int(target_k_len), "sparse"


def get_sparse_compiled_flex_attention(*, family_key: str) -> Any:
    del family_key
    return sparse_compiled_flex_attention


dense_compiled_flex_attention = torch.compile(
    _forced_flex_attention_dense,
)

sparse_compiled_flex_attention = torch.compile(
    _forced_flex_attention_sparse,
)
