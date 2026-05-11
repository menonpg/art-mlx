from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from art.megatron.compiled_flex_attention import normalize_sparse_block_size

from .types import ExactMaskMetadata, FlexMaskSpec

_INVALID_Q_GROUP = -(1 << 63)
_INVALID_Q_PARENT = _INVALID_Q_GROUP + 1
_INVALID_K_GROUP = _INVALID_Q_GROUP + 2
_COMPILED_CREATE_BLOCK_MASK = torch.compile(
    create_block_mask,
    backend="aot_eager",
)


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
    return _COMPILED_CREATE_BLOCK_MASK(
        mask_mod,
        1,
        None,
        int(spec.q_len),
        int(spec.k_len),
        device=device,
        BLOCK_SIZE=block_size,
    )
