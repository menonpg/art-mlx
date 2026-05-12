from __future__ import annotations

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask

from art.megatron.compiled_flex_attention import normalize_sparse_block_size

from .types import AttnMaskKind, ExactMaskMetadata, FlexMaskSpec

_INVALID_Q_GROUP = -(1 << 63)
_INVALID_Q_PARENT = _INVALID_Q_GROUP + 1
_INVALID_K_GROUP = _INVALID_Q_GROUP + 2


def _index_select_with_invalid(
    values: torch.Tensor,
    indices: torch.Tensor,
    *,
    invalid_value: int,
) -> torch.Tensor:
    selected = torch.full_like(indices, invalid_value)
    valid = indices >= 0
    if bool(valid.any()):
        selected[valid] = values.index_select(0, indices[valid])
    return selected


def _build_exact_mask_mod(
    metadata: ExactMaskMetadata,
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    device: torch.device,
):
    q_abs = metadata.q_token_indices.to(device=device, dtype=torch.int64)
    k_abs = metadata.k_token_indices.to(device=device, dtype=torch.int64)
    flat_group_ids = group_ids.to(device=device, dtype=torch.int64).reshape(-1)
    flat_parent_ids = parent_ids.to(device=device, dtype=torch.int64).reshape(-1)
    q_group = _index_select_with_invalid(
        flat_group_ids,
        q_abs,
        invalid_value=_INVALID_Q_GROUP,
    )
    q_parent = _index_select_with_invalid(
        flat_parent_ids,
        q_abs,
        invalid_value=_INVALID_Q_PARENT,
    )
    k_group = _index_select_with_invalid(
        flat_group_ids,
        k_abs,
        invalid_value=_INVALID_K_GROUP,
    )

    def mask_mod(
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        query_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del batch_idx, head_idx
        q_abs_local = q_abs[query_idx]
        k_abs_local = k_abs[kv_idx]
        same_group = q_group[query_idx] == k_group[kv_idx]
        parent_prefix = q_parent[query_idx] == k_group[kv_idx]
        return (q_abs_local >= k_abs_local) & (same_group | parent_prefix)

    return mask_mod


def _dense_blocks_to_ordered(
    blocks: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.from_numpy(blocks.sum(axis=-1).astype(np.int32))
    indices = torch.from_numpy(
        np.argsort(-blocks.astype(np.int32), axis=-1, kind="stable").astype(np.int32)
    )
    return (
        counts.view(1, 1, -1).to(device=device),
        indices.view(1, 1, blocks.shape[0], blocks.shape[1]).to(device=device),
    )


def _select_with_invalid_cpu(
    values: torch.Tensor,
    indices: torch.Tensor,
    *,
    invalid_value: int,
) -> torch.Tensor:
    selected = torch.full_like(indices, invalid_value)
    valid = indices >= 0
    if bool(valid.any()):
        selected[valid] = values.index_select(0, indices[valid])
    return selected


def _exact_block_state(
    *,
    q_abs: torch.Tensor,
    k_abs: torch.Tensor,
    flat_group_ids: torch.Tensor,
    flat_parent_ids: torch.Tensor,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
) -> tuple[bool, bool]:
    q = q_abs[q_start:q_end]
    k = k_abs[k_start:k_end]
    if int(q.numel()) == 0 or int(k.numel()) == 0:
        return False, False
    q_group = _select_with_invalid_cpu(
        flat_group_ids,
        q,
        invalid_value=_INVALID_Q_GROUP,
    )
    q_parent = _select_with_invalid_cpu(
        flat_parent_ids,
        q,
        invalid_value=_INVALID_Q_PARENT,
    )
    k_group = _select_with_invalid_cpu(
        flat_group_ids,
        k,
        invalid_value=_INVALID_K_GROUP,
    )
    allowed = (q[:, None] >= k[None, :]) & (
        (q_group[:, None] == k_group[None, :]) | (q_parent[:, None] == k_group[None, :])
    )
    return bool(allowed.any()), bool(allowed.all())


def _build_sparse_block_mask(
    spec: FlexMaskSpec,
    *,
    device: torch.device,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    mask_mod,
    block_size: tuple[int, int],
) -> BlockMask:
    q_block, k_block = block_size
    q_blocks = (int(spec.q_len) + q_block - 1) // q_block
    k_blocks = (int(spec.k_len) + k_block - 1) // k_block
    partial_blocks = np.zeros((q_blocks, k_blocks), dtype=bool)
    full_blocks = np.zeros((q_blocks, k_blocks), dtype=bool)
    touch_counts = np.zeros((q_blocks, k_blocks), dtype=np.int16)
    q_abs_tensor = spec.exact_mask.q_token_indices.detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    k_abs_tensor = spec.exact_mask.k_token_indices.detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    q_abs = q_abs_tensor.numpy()
    k_abs = k_abs_tensor.numpy()
    flat_group_ids = group_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    flat_parent_ids = (
        parent_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    )
    if not spec.slices:
        raise RuntimeError(
            "Cannot build a CP attention block mask without stage slices"
        )

    for slice_ in spec.slices:
        q_start = max(0, int(slice_.q_range.start))
        q_end = min(int(spec.q_len), int(slice_.q_range.end))
        k_start = max(0, int(slice_.k_range.start))
        k_end = min(int(spec.k_len), int(slice_.k_range.end))
        q_block_indices = np.arange(
            q_start // q_block,
            (q_end + q_block - 1) // q_block,
            dtype=np.int64,
        )
        k_block_indices = np.arange(
            k_start // k_block,
            (k_end + k_block - 1) // k_block,
            dtype=np.int64,
        )
        if int(q_block_indices.size) == 0 or int(k_block_indices.size) == 0:
            continue
        q_block_start = q_block_indices * q_block
        q_block_end = np.minimum(
            (q_block_indices + 1) * q_block,
            int(spec.q_len),
        )
        k_block_start = k_block_indices * k_block
        k_block_end = np.minimum(
            (k_block_indices + 1) * k_block,
            int(spec.k_len),
        )
        q_overlap_start = np.maximum(
            q_block_start,
            q_start,
        )
        q_overlap_end = np.minimum(
            q_block_end,
            q_end,
        )
        k_overlap_start = np.maximum(
            k_block_start,
            k_start,
        )
        k_overlap_end = np.minimum(
            k_block_end,
            k_end,
        )
        q_min = q_abs[q_overlap_start]
        q_max = q_abs[q_overlap_end - 1]
        k_min = k_abs[k_overlap_start]
        k_max = k_abs[k_overlap_end - 1]
        q_is_full = (q_overlap_start == q_block_start) & (q_overlap_end == q_block_end)
        k_is_full = (k_overlap_start == k_block_start) & (k_overlap_end == k_block_end)
        covers_block = q_is_full[:, None] & k_is_full[None, :]
        if slice_.mask_kind == AttnMaskKind.FULL:
            has_any = np.ones(
                (int(q_block_indices.size), int(k_block_indices.size)), dtype=bool
            )
            is_full = covers_block
        else:
            has_any = q_max[:, None] >= k_min[None, :]
            is_full = covers_block & (q_min[:, None] >= k_max[None, :])

        q_slice = slice(int(q_block_indices[0]), int(q_block_indices[-1]) + 1)
        k_slice = slice(int(k_block_indices[0]), int(k_block_indices[-1]) + 1)
        touch_counts[q_slice, k_slice] += has_any.astype(np.int16)
        partial_blocks[q_slice, k_slice] |= has_any
        full_blocks[q_slice, k_slice] |= is_full

    ambiguous = (touch_counts > 1) & partial_blocks & ~full_blocks
    for q_idx, k_idx in np.argwhere(ambiguous):
        q_start = int(q_idx) * q_block
        q_end = min((int(q_idx) + 1) * q_block, int(spec.q_len))
        k_start = int(k_idx) * k_block
        k_end = min((int(k_idx) + 1) * k_block, int(spec.k_len))
        has_any, is_full = _exact_block_state(
            q_abs=q_abs_tensor,
            k_abs=k_abs_tensor,
            flat_group_ids=flat_group_ids,
            flat_parent_ids=flat_parent_ids,
            q_start=q_start,
            q_end=q_end,
            k_start=k_start,
            k_end=k_end,
        )
        partial_blocks[q_idx, k_idx] = False
        full_blocks[q_idx, k_idx] = False
        if is_full:
            full_blocks[q_idx, k_idx] = True
        elif has_any:
            partial_blocks[q_idx, k_idx] = True

    partial_blocks &= ~full_blocks
    kv_num_blocks, kv_indices = _dense_blocks_to_ordered(
        partial_blocks,
        device=device,
    )
    full_kv_num_blocks, full_kv_indices = _dense_blocks_to_ordered(
        full_blocks,
        device=device,
    )
    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(int(spec.q_len), int(spec.k_len)),
    )


def build_block_mask(
    spec: FlexMaskSpec,
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    device: torch.device,
) -> BlockMask | None:
    if spec.q_len <= 0 or spec.k_len <= 0:
        return None
    if int(spec.exact_mask.q_token_indices.numel()) != int(spec.q_len):
        raise RuntimeError(
            "Exact stage q-token metadata length mismatch: "
            f"{int(spec.exact_mask.q_token_indices.numel())} != {int(spec.q_len)}"
        )
    if int(spec.exact_mask.k_token_indices.numel()) != int(spec.k_len):
        raise RuntimeError(
            "Exact stage k-token metadata length mismatch: "
            f"{int(spec.exact_mask.k_token_indices.numel())} != {int(spec.k_len)}"
        )
    mask_mod = _build_exact_mask_mod(
        spec.exact_mask,
        group_ids=group_ids,
        parent_ids=parent_ids,
        device=device,
    )
    block_size = normalize_sparse_block_size(spec.block_size)
    return _build_sparse_block_mask(
        spec,
        device=device,
        group_ids=group_ids,
        parent_ids=parent_ids,
        mask_mod=mask_mod,
        block_size=block_size,
    )
