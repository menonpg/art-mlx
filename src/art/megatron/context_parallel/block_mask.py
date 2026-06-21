from __future__ import annotations

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask

from art.megatron.flex_attn.compiled import normalize_sparse_block_size

from .types import AttnMaskKind, FlexMaskSpec

_INVALID_GROUP_INDEX = 0


def _build_exact_mask_mod(
    *,
    q_abs: np.ndarray,
    k_abs: np.ndarray,
    q_group_index: np.ndarray,
    k_group_index: np.ndarray,
    group_can_attend: np.ndarray,
    device: torch.device,
):
    q_abs_tensor = torch.as_tensor(q_abs, device=device, dtype=torch.int64)
    k_abs_tensor = torch.as_tensor(k_abs, device=device, dtype=torch.int64)
    q_group_tensor = torch.as_tensor(q_group_index, device=device, dtype=torch.int32)
    k_group_tensor = torch.as_tensor(k_group_index, device=device, dtype=torch.int32)
    group_can_attend_tensor = torch.as_tensor(
        group_can_attend,
        device=device,
        dtype=torch.bool,
    )

    def mask_mod(
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        query_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del batch_idx, head_idx
        q_abs_local = q_abs_tensor[query_idx]
        k_abs_local = k_abs_tensor[kv_idx]
        allowed_group = group_can_attend_tensor[
            q_group_tensor[query_idx],
            k_group_tensor[kv_idx],
        ]
        return (q_abs_local >= k_abs_local) & allowed_group

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


def _is_strictly_increasing(values: np.ndarray) -> bool:
    return int(values.size) <= 1 or bool(np.all(values[1:] > values[:-1]))


def _block_min_max(
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mins = np.empty(starts.shape, dtype=values.dtype)
    maxes = np.empty(starts.shape, dtype=values.dtype)
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        block = values[int(start) : int(end)]
        mins[index] = block.min()
        maxes[index] = block.max()
    return mins, maxes


def _build_group_can_attend(
    *,
    group_ids: np.ndarray,
    parent_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = group_ids >= 0
    sorted_group_ids = np.unique(group_ids[valid]).astype(np.int64, copy=False)
    group_to_index = {
        int(group_id): index + 1 for index, group_id in enumerate(sorted_group_ids)
    }
    group_can_attend = np.zeros(
        (int(sorted_group_ids.size) + 1, int(sorted_group_ids.size) + 1),
        dtype=bool,
    )
    parent_by_group: dict[int, int | None] = {}
    for group_id in sorted_group_ids.tolist():
        positions = np.flatnonzero(group_ids == int(group_id))
        parent_id = int(parent_ids[int(positions[0])])
        parent_by_group[int(group_id)] = (
            None if parent_id < 0 or parent_id == int(group_id) else parent_id
        )

    for group_id in sorted_group_ids.tolist():
        query_index = group_to_index[int(group_id)]
        cursor = int(group_id)
        seen: set[int] = set()
        while cursor in group_to_index:
            group_can_attend[query_index, group_to_index[cursor]] = True
            parent_id = parent_by_group.get(cursor)
            if parent_id is None or parent_id in seen:
                break
            seen.add(cursor)
            cursor = parent_id
    return sorted_group_ids, group_can_attend


def _remap_group_values(
    values: np.ndarray,
    *,
    sorted_group_ids: np.ndarray,
) -> np.ndarray:
    remapped = np.full(values.shape, _INVALID_GROUP_INDEX, dtype=np.int32)
    if int(sorted_group_ids.size) == 0:
        return remapped
    positions = np.searchsorted(sorted_group_ids, values)
    in_bounds = positions < int(sorted_group_ids.size)
    matched = np.zeros(values.shape, dtype=bool)
    matched[in_bounds] = sorted_group_ids[positions[in_bounds]] == values[in_bounds]
    remapped[matched] = positions[matched].astype(np.int32, copy=False) + 1
    return remapped


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
    q_abs_sorted = _is_strictly_increasing(q_abs[q_abs >= 0])
    k_abs_sorted = _is_strictly_increasing(k_abs[k_abs >= 0])
    flat_group_ids = group_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    flat_parent_ids = (
        parent_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    )
    flat_group_ids_np = flat_group_ids.numpy()
    flat_parent_ids_np = flat_parent_ids.numpy()
    q_group = _select_with_invalid_np(
        flat_group_ids_np,
        q_abs,
        invalid_value=-1,
    )
    k_group = _select_with_invalid_np(
        flat_group_ids_np,
        k_abs,
        invalid_value=-1,
    )
    sorted_group_ids, group_can_attend = _build_group_can_attend(
        group_ids=flat_group_ids_np,
        parent_ids=flat_parent_ids_np,
    )
    q_group_index = _remap_group_values(
        q_group,
        sorted_group_ids=sorted_group_ids,
    )
    k_group_index = _remap_group_values(
        k_group,
        sorted_group_ids=sorted_group_ids,
    )
    mask_mod = _build_exact_mask_mod(
        q_abs=q_abs,
        k_abs=k_abs,
        q_group_index=q_group_index,
        k_group_index=k_group_index,
        group_can_attend=group_can_attend,
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
        q_min, q_max = (
            (q_abs[q_overlap_start], q_abs[q_overlap_end - 1])
            if q_abs_sorted
            else _block_min_max(q_abs, q_overlap_start, q_overlap_end)
        )
        k_min, k_max = (
            (k_abs[k_overlap_start], k_abs[k_overlap_end - 1])
            if k_abs_sorted
            else _block_min_max(k_abs, k_overlap_start, k_overlap_end)
        )
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
        partial_blocks[q_slice, k_slice] |= has_any
        full_blocks[q_slice, k_slice] |= is_full

    # Overlapping tree slices are left as partial blocks. The block-level program
    # only decides which blocks to visit; `mask_mod` above is the exact authority.

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


def _valid_prefix(indices: torch.Tensor, *, name: str) -> torch.Tensor:
    if indices.ndim != 1:
        raise RuntimeError(f"{name} exact token indices must be rank 1.")
    if indices.dtype != torch.int64:
        raise RuntimeError(f"{name} exact token indices must be int64.")
    indices_cpu = indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
    invalid = indices_cpu < 0
    if bool(invalid.any().item()):
        first_invalid = int(torch.nonzero(invalid, as_tuple=False)[0].item())
        if bool((indices_cpu[first_invalid:] >= 0).any().item()):
            raise RuntimeError(
                f"{name} exact token indices must use only contiguous tail padding."
            )
        return indices_cpu[:first_invalid]
    return indices_cpu


def _validate_exact_indices(
    indices: torch.Tensor,
    *,
    name: str,
    source_len: int,
) -> int:
    valid = _valid_prefix(indices, name=name)
    if int(valid.numel()) == 0:
        return 0
    if int(valid.unique().numel()) != int(valid.numel()):
        raise RuntimeError(f"{name} exact token indices must not contain duplicates.")
    max_index = int(valid.max().item())
    if max_index >= int(source_len):
        raise RuntimeError(
            f"{name} exact token index {max_index} exceeds source metadata length {int(source_len)}."
        )
    return int(valid.numel())


def _validate_supported_mask_spec(
    spec: FlexMaskSpec,
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> None:
    if group_ids.ndim != 1 or parent_ids.ndim != 1:
        raise RuntimeError(
            "Shared-prefix sparse block masks require rank-1 group_ids and parent_ids."
        )
    if int(group_ids.numel()) != int(parent_ids.numel()):
        raise RuntimeError(
            "Shared-prefix sparse block masks require equal group_ids and parent_ids lengths."
        )
    q_valid_len = _validate_exact_indices(
        spec.exact_mask.q_token_indices,
        name="q",
        source_len=int(group_ids.numel()),
    )
    k_valid_len = _validate_exact_indices(
        spec.exact_mask.k_token_indices,
        name="k",
        source_len=int(group_ids.numel()),
    )
    for slice_ in spec.slices:
        if int(slice_.row_index) != 0:
            raise RuntimeError(
                "Shared-prefix sparse block masks support exactly one packed row."
            )
        if slice_.mask_kind not in {AttnMaskKind.FULL, AttnMaskKind.CAUSAL}:
            raise RuntimeError(f"Unsupported attention mask kind: {slice_.mask_kind}")
        if (
            slice_.q_range.start < 0
            or slice_.q_range.end > int(spec.q_len)
            or slice_.k_range.start < 0
            or slice_.k_range.end > int(spec.k_len)
            or slice_.q_range.end < slice_.q_range.start
            or slice_.k_range.end < slice_.k_range.start
        ):
            raise RuntimeError(f"Attention slice is outside mask bounds: {slice_}")
        if slice_.q_range.end > q_valid_len or slice_.k_range.end > k_valid_len:
            raise RuntimeError(
                "Attention slices may not cover exact-index tail padding: "
                f"slice={slice_}, q_valid_len={q_valid_len}, k_valid_len={k_valid_len}"
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
    _validate_supported_mask_spec(spec, group_ids=group_ids, parent_ids=parent_ids)
    block_size = normalize_sparse_block_size(spec.block_size)
    return _build_sparse_block_mask(
        spec,
        device=device,
        group_ids=group_ids,
        parent_ids=parent_ids,
        block_size=block_size,
    )
