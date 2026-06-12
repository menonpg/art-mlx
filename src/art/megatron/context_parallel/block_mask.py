from __future__ import annotations

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask

from art.megatron.flex_attn.compiled import normalize_sparse_block_size

from .types import AttnMaskKind, FlexMaskSpec

_INVALID_Q_GROUP = -(1 << 63)
_INVALID_Q_PARENT = _INVALID_Q_GROUP + 1
_INVALID_K_GROUP = _INVALID_Q_GROUP + 2


def _build_exact_mask_mod(
    *,
    q_abs: np.ndarray,
    k_abs: np.ndarray,
    q_group: np.ndarray,
    q_parent: np.ndarray,
    k_group: np.ndarray,
    device: torch.device,
):
    q_abs_tensor = torch.as_tensor(q_abs, device=device, dtype=torch.int64)
    k_abs_tensor = torch.as_tensor(k_abs, device=device, dtype=torch.int64)
    q_group_tensor = torch.as_tensor(q_group, device=device, dtype=torch.int64)
    q_parent_tensor = torch.as_tensor(q_parent, device=device, dtype=torch.int64)
    k_group_tensor = torch.as_tensor(k_group, device=device, dtype=torch.int64)

    def mask_mod(
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        query_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del batch_idx, head_idx
        q_abs_local = q_abs_tensor[query_idx]
        k_abs_local = k_abs_tensor[kv_idx]
        same_group = q_group_tensor[query_idx] == k_group_tensor[kv_idx]
        parent_prefix = q_parent_tensor[query_idx] == k_group_tensor[kv_idx]
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


def _select_with_invalid_np(
    values: np.ndarray,
    indices: np.ndarray,
    *,
    invalid_value: int,
) -> np.ndarray:
    selected = np.full(indices.shape, invalid_value, dtype=np.int64)
    valid = indices >= 0
    if bool(valid.any()):
        selected[valid] = values[indices[valid]]
    return selected


def _build_q_block_group_state(
    *,
    q_abs: np.ndarray,
    q_group: np.ndarray,
    q_parent: np.ndarray,
    q_block: int,
    block_idx: int,
) -> tuple[int, dict[int, int], frozenset[int]]:
    start = int(block_idx) * q_block
    end = min((int(block_idx) + 1) * q_block, int(q_abs.size))
    q = q_abs[start:end]
    q_group_block = q_group[start:end]
    q_parent_block = q_parent[start:end]
    q_min = int(q.min()) if int(q.size) else 0
    max_by_group: dict[int, int] = {}
    all_groups: list[int] = []
    for group_value in np.unique(np.concatenate((q_group_block, q_parent_block))):
        allowed = (q_group_block == group_value) | (q_parent_block == group_value)
        if bool(allowed.any()):
            max_by_group[int(group_value)] = int(q[allowed].max())
        if bool(allowed.all()):
            all_groups.append(int(group_value))
    return q_min, max_by_group, frozenset(all_groups)


def _build_k_block_group_state(
    *,
    k_abs: np.ndarray,
    k_group: np.ndarray,
    k_block: int,
    block_idx: int,
) -> tuple[int, dict[int, int], tuple[int, ...]]:
    start = int(block_idx) * k_block
    end = min((int(block_idx) + 1) * k_block, int(k_abs.size))
    k = k_abs[start:end]
    k_group_block = k_group[start:end]
    k_max = int(k.max()) if int(k.size) else 0
    min_by_group: dict[int, int] = {}
    for group_value in np.unique(k_group_block):
        min_by_group[int(group_value)] = int(k[k_group_block == group_value].min())
    return k_max, min_by_group, tuple(min_by_group)


def _exact_block_state(
    *,
    q_state: tuple[int, dict[int, int], frozenset[int]],
    k_state: tuple[int, dict[int, int], tuple[int, ...]],
) -> tuple[bool, bool]:
    q_min, q_allowed_max, q_all_allowed = q_state
    k_max, k_min, k_groups = k_state
    if not any(
        q_allowed_max.get(k_group_value, _INVALID_Q_GROUP) >= min_k
        for k_group_value, min_k in k_min.items()
    ):
        return False, False
    if int(q_min) < int(k_max):
        return True, False
    return True, all(k_group_value in q_all_allowed for k_group_value in k_groups)


def _build_sparse_block_mask(
    spec: FlexMaskSpec,
    *,
    device: torch.device,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
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
    flat_group_ids_np = flat_group_ids.numpy()
    flat_parent_ids_np = flat_parent_ids.numpy()
    q_group = _select_with_invalid_np(
        flat_group_ids_np,
        q_abs,
        invalid_value=_INVALID_Q_GROUP,
    )
    q_parent = _select_with_invalid_np(
        flat_parent_ids_np,
        q_abs,
        invalid_value=_INVALID_Q_PARENT,
    )
    k_group = _select_with_invalid_np(
        flat_group_ids_np,
        k_abs,
        invalid_value=_INVALID_K_GROUP,
    )
    mask_mod = _build_exact_mask_mod(
        q_abs=q_abs,
        k_abs=k_abs,
        q_group=q_group,
        q_parent=q_parent,
        k_group=k_group,
        device=device,
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
    q_state_cache: dict[int, tuple[int, dict[int, int], frozenset[int]]] = {}
    k_state_cache: dict[int, tuple[int, dict[int, int], tuple[int, ...]]] = {}
    for q_idx, k_idx in np.argwhere(ambiguous):
        q_state = q_state_cache.get(int(q_idx))
        if q_state is None:
            q_state = _build_q_block_group_state(
                q_abs=q_abs,
                q_group=q_group,
                q_parent=q_parent,
                q_block=q_block,
                block_idx=int(q_idx),
            )
            q_state_cache[int(q_idx)] = q_state
        k_state = k_state_cache.get(int(k_idx))
        if k_state is None:
            k_state = _build_k_block_group_state(
                k_abs=k_abs,
                k_group=k_group,
                k_block=k_block,
                block_idx=int(k_idx),
            )
            k_state_cache[int(k_idx)] = k_state
        has_any, is_full = _exact_block_state(
            q_state=q_state,
            k_state=k_state,
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
    q_num_blocks, q_indices = _dense_blocks_to_ordered(
        partial_blocks.T,
        device=device,
    )
    full_q_num_blocks, full_q_indices = _dense_blocks_to_ordered(
        full_blocks.T,
        device=device,
    )
    return BlockMask(
        seq_lengths=(int(spec.q_len), int(spec.k_len)),
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        full_q_num_blocks=full_q_num_blocks,
        full_q_indices=full_q_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
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
    block_size = normalize_sparse_block_size(spec.block_size)
    return _build_sparse_block_mask(
        spec,
        device=device,
        group_ids=group_ids,
        parent_ids=parent_ids,
        block_size=block_size,
    )
