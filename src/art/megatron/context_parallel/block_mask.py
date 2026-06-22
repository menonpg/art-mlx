from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask

from art.megatron.flex_attn.compiled import normalize_sparse_block_size
from art.megatron.shared_prefix_tree import parse_shared_prefix_row

from .types import AttnMaskKind, FlexMaskSpec

_INVALID_GROUP_INDEX = 0


@dataclass(frozen=True, slots=True)
class PreparedBlockMaskContext:
    group_ids: torch.Tensor
    parent_ids: torch.Tensor
    group_ids_np: np.ndarray
    sorted_group_ids: np.ndarray
    group_can_attend: np.ndarray
    max_depth: int


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
    row_indices, column_indices = np.nonzero(blocks)
    counts_np = np.bincount(row_indices, minlength=blocks.shape[0]).astype(np.int32)
    indices_np = np.zeros(blocks.shape, dtype=np.int32)
    if int(row_indices.size) > 0:
        starts = np.concatenate(([0], np.cumsum(counts_np[:-1], dtype=np.int64)))
        active_rows = np.flatnonzero(counts_np)
        for row_index in active_rows:
            start = int(starts[row_index])
            end = start + int(counts_np[row_index])
            indices_np[row_index, : end - start] = column_indices[start:end]
    counts = torch.from_numpy(counts_np)
    indices = torch.from_numpy(indices_np)
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


def _promote_exact_full_blocks(
    *,
    partial_blocks: np.ndarray,
    full_blocks: np.ndarray,
    q_abs: np.ndarray,
    k_abs: np.ndarray,
    q_group_index: np.ndarray,
    k_group_index: np.ndarray,
    group_can_attend: np.ndarray,
    q_block: int,
    k_block: int,
    q_len: int,
    k_len: int,
) -> None:
    for q_block_index, k_block_index in np.argwhere(partial_blocks):
        q_start = int(q_block_index) * q_block
        k_start = int(k_block_index) * k_block
        q_end = q_start + q_block
        k_end = k_start + k_block
        if q_end > q_len or k_end > k_len:
            continue

        q_slice = slice(q_start, q_end)
        k_slice = slice(k_start, k_end)
        can_attend = group_can_attend[
            q_group_index[q_slice, None],
            k_group_index[None, k_slice],
        ]
        causal = q_abs[q_slice, None] >= k_abs[None, k_slice]
        if bool(np.all(causal & can_attend)):
            partial_blocks[q_block_index, k_block_index] = False
            full_blocks[q_block_index, k_block_index] = True


def _build_sparse_block_mask(
    spec: FlexMaskSpec,
    *,
    device: torch.device,
    context: PreparedBlockMaskContext,
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
    q_group = _select_with_invalid_np(
        context.group_ids_np,
        q_abs,
        invalid_value=-1,
    )
    k_group = _select_with_invalid_np(
        context.group_ids_np,
        k_abs,
        invalid_value=-1,
    )
    q_group_index = _remap_group_values(
        q_group,
        sorted_group_ids=context.sorted_group_ids,
    )
    k_group_index = _remap_group_values(
        k_group,
        sorted_group_ids=context.sorted_group_ids,
    )
    mask_mod = _build_exact_mask_mod(
        q_abs=q_abs,
        k_abs=k_abs,
        q_group_index=q_group_index,
        k_group_index=k_group_index,
        group_can_attend=context.group_can_attend,
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
        q_block_end_raw = (q_block_indices + 1) * q_block
        q_block_end = np.minimum(q_block_end_raw, int(spec.q_len))
        k_block_start = k_block_indices * k_block
        k_block_end_raw = (k_block_indices + 1) * k_block
        k_block_end = np.minimum(k_block_end_raw, int(spec.k_len))
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
        q_is_full = (q_overlap_start == q_block_start) & (
            q_overlap_end == q_block_end_raw
        )
        k_is_full = (k_overlap_start == k_block_start) & (
            k_overlap_end == k_block_end_raw
        )
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

    partial_blocks &= ~full_blocks
    if context.max_depth > 1:
        _promote_exact_full_blocks(
            partial_blocks=partial_blocks,
            full_blocks=full_blocks,
            q_abs=q_abs,
            k_abs=k_abs,
            q_group_index=q_group_index,
            k_group_index=k_group_index,
            group_can_attend=context.group_can_attend,
            q_block=q_block,
            k_block=k_block,
            q_len=int(spec.q_len),
            k_len=int(spec.k_len),
        )
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


def prepare_block_mask_context(
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> PreparedBlockMaskContext:
    if group_ids.ndim != 1 or parent_ids.ndim != 1:
        raise RuntimeError(
            "Shared-prefix sparse block masks require rank-1 group_ids and parent_ids."
        )
    if int(group_ids.numel()) != int(parent_ids.numel()):
        raise RuntimeError(
            "Shared-prefix sparse block masks require equal group_ids and parent_ids lengths."
        )
    flat_group_ids = group_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    flat_parent_ids = (
        parent_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    )
    row_tree = parse_shared_prefix_row(
        group_ids=flat_group_ids,
        parent_ids=flat_parent_ids,
    )
    group_ids_for_matrix, group_can_attend_values = row_tree.group_can_attend_matrix()
    return PreparedBlockMaskContext(
        group_ids=flat_group_ids,
        parent_ids=flat_parent_ids,
        group_ids_np=flat_group_ids.numpy(),
        sorted_group_ids=np.asarray(group_ids_for_matrix, dtype=np.int64),
        group_can_attend=np.asarray(group_can_attend_values, dtype=bool),
        max_depth=int(row_tree.max_depth),
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
    return build_block_mask_from_context(
        spec,
        context=prepare_block_mask_context(
            group_ids=group_ids,
            parent_ids=parent_ids,
        ),
        device=device,
    )


def build_block_mask_from_context(
    spec: FlexMaskSpec,
    *,
    context: PreparedBlockMaskContext,
    device: torch.device,
    validate: bool = True,
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
    if validate:
        _validate_supported_mask_spec(
            spec,
            group_ids=context.group_ids,
            parent_ids=context.parent_ids,
        )
    block_size = normalize_sparse_block_size(spec.block_size)
    return _build_sparse_block_mask(
        spec,
        device=device,
        context=context,
        block_size=block_size,
    )
