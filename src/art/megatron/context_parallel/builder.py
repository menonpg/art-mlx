from __future__ import annotations

import torch

from .types import (
    AttnMaskKind,
    AttnSlice,
    PackedBatchAttentionSpec,
    PackedRowAttentionSpec,
    SharedPrefixBuilderConfig,
    TokenRange,
)


def _valid_length(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    *,
    ignore_padding_group_id: int,
) -> int:
    valid_mask = group_ids != ignore_padding_group_id
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return 0
    if not bool(valid_mask[:valid_count].all().item()):
        raise RuntimeError("Padding tokens must be a contiguous tail")
    return _infer_terminal_padding_length(
        group_ids[:valid_count],
        parent_ids[:valid_count],
    )


def _infer_terminal_padding_length(
    group_row: torch.Tensor,
    parent_row: torch.Tensor,
) -> int:
    if group_row.numel() == 0:
        return 0
    runs = _scan_runs(group_row, parent_row)
    if len(runs) < 2:
        return int(group_row.numel())
    last_start, _last_end, last_group_id, last_parent_id = runs[-1]
    if last_parent_id >= 0:
        return int(group_row.numel())
    terminal_pair = (last_group_id, last_parent_id)
    if any(
        (group_id, parent_id) == terminal_pair
        for _start, _end, group_id, parent_id in runs[:-1]
    ):
        return last_start
    return int(group_row.numel())


def _scan_runs(
    group_row: torch.Tensor,
    parent_row: torch.Tensor,
) -> list[tuple[int, int, int, int]]:
    length = int(group_row.numel())
    if length == 0:
        return []

    group_changes = group_row[1:] != group_row[:-1]
    parent_changes = parent_row[1:] != parent_row[:-1]
    inconsistent_parent = torch.nonzero(
        torch.logical_not(group_changes) & parent_changes,
        as_tuple=False,
    ).flatten()
    if int(inconsistent_parent.numel()) > 0:
        mismatch_index = int(inconsistent_parent[0].item()) + 1
        prior_boundaries = torch.nonzero(
            group_changes[: mismatch_index - 1],
            as_tuple=False,
        ).flatten()
        start = (
            0
            if int(prior_boundaries.numel()) == 0
            else int(prior_boundaries[-1].item()) + 1
        )
        group_id = int(group_row[start].item())
        raise RuntimeError(
            "Found one group run with inconsistent parent ids: "
            f"group_id={group_id}, start={start}, end={mismatch_index}"
        )

    run_starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=group_row.device),
            torch.nonzero(group_changes, as_tuple=False).flatten() + 1,
        )
    )
    run_ends = torch.cat(
        (
            run_starts[1:],
            torch.tensor([length], dtype=torch.int64, device=group_row.device),
        )
    )
    starts = run_starts.to(device="cpu").tolist()
    ends = run_ends.to(device="cpu").tolist()
    group_ids = group_row.index_select(0, run_starts).to(device="cpu").tolist()
    parent_ids = parent_row.index_select(0, run_starts).to(device="cpu").tolist()
    return [
        (int(start), int(end), int(group_id), int(parent_id))
        for start, end, group_id, parent_id in zip(
            starts, ends, group_ids, parent_ids, strict=True
        )
    ]


def _sort_and_dedupe_slices(slices: list[AttnSlice]) -> tuple[AttnSlice, ...]:
    sorted_slices = sorted(
        slices,
        key=lambda slice_: (
            int(slice_.row_index),
            int(slice_.q_range.start),
            int(slice_.q_range.end),
            int(slice_.k_range.start),
            int(slice_.k_range.end),
            str(slice_.mask_kind),
            -1 if slice_.family_index is None else int(slice_.family_index),
        ),
    )
    deduped: list[AttnSlice] = []
    last_key: tuple[int, int, int, int, int, str, int] | None = None
    for slice_ in sorted_slices:
        key = (
            int(slice_.row_index),
            int(slice_.q_range.start),
            int(slice_.q_range.end),
            int(slice_.k_range.start),
            int(slice_.k_range.end),
            str(slice_.mask_kind),
            -1 if slice_.family_index is None else int(slice_.family_index),
        )
        if key == last_key:
            continue
        deduped.append(slice_)
        last_key = key
    return tuple(deduped)


def _is_prompt_run(
    *,
    start: int,
    group_id: int,
    parent_id: int,
    ignore_padding_group_id: int,
) -> bool:
    return group_id == parent_id or (
        start == 0 and parent_id == ignore_padding_group_id
    )


def build_shared_prefix_attention_spec(
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    config: SharedPrefixBuilderConfig = SharedPrefixBuilderConfig(),
) -> PackedBatchAttentionSpec:
    if group_ids.shape != parent_ids.shape:
        raise RuntimeError(
            "group_ids and parent_ids must share shape, got "
            f"{tuple(group_ids.shape)} vs {tuple(parent_ids.shape)}"
        )
    if group_ids.ndim != 2:
        raise RuntimeError(
            "group_ids and parent_ids must be rank-2 packed tensors, got "
            f"{group_ids.ndim}"
        )
    if int(group_ids.shape[0]) != 1:
        raise RuntimeError(
            "ART shared-prefix attention spec currently supports exactly one packed sequence, "
            f"got batch={int(group_ids.shape[0])}."
        )

    rows: list[PackedRowAttentionSpec] = []
    for row_index in range(group_ids.shape[0]):
        group_row = group_ids[row_index]
        parent_row = parent_ids[row_index]
        valid_tokens = _valid_length(
            group_row,
            parent_row,
            ignore_padding_group_id=config.ignore_padding_group_id,
        )
        if valid_tokens == 0:
            rows.append(
                PackedRowAttentionSpec(row_index=row_index, valid_tokens=0, slices=())
            )
            continue

        group_row = group_row[:valid_tokens]
        parent_row = parent_row[:valid_tokens]
        runs = _scan_runs(group_row, parent_row)

        group_run_count: dict[int, int] = {}
        prompt_by_group_id: dict[int, tuple[tuple[int, int], int]] = {}
        completion_ranges_by_prompt: dict[int, list[tuple[int, int]]] = {}

        for start, end, group_id, parent_id in runs:
            group_run_count[group_id] = group_run_count.get(group_id, 0) + 1
            if _is_prompt_run(
                start=start,
                group_id=group_id,
                parent_id=parent_id,
                ignore_padding_group_id=config.ignore_padding_group_id,
            ):
                if group_id in prompt_by_group_id:
                    raise RuntimeError(
                        f"Prompt group_id {group_id} appears more than once in row {row_index}"
                    )
                family_index = len(prompt_by_group_id)
                prompt_by_group_id[group_id] = (
                    (start, end),
                    family_index,
                )
                completion_ranges_by_prompt[group_id] = []

        if config.require_contiguous_group_runs:
            repeated_groups = {
                group_id: count
                for group_id, count in group_run_count.items()
                if count > 1 and group_id != config.ignore_padding_group_id
            }
            if repeated_groups:
                raise RuntimeError(
                    "Shared-prefix builder requires contiguous group runs per row, "
                    f"found repeats in row {row_index}: {repeated_groups}"
                )

        for start, end, group_id, parent_id in runs:
            if _is_prompt_run(
                start=start,
                group_id=group_id,
                parent_id=parent_id,
                ignore_padding_group_id=config.ignore_padding_group_id,
            ):
                continue
            prompt_entry = prompt_by_group_id.get(parent_id)
            if prompt_entry is None:
                raise RuntimeError(
                    "Completion run points to a missing prompt run: "
                    f"row={row_index}, group_id={group_id}, parent_id={parent_id}"
                )
            completion_ranges_by_prompt[parent_id].append((start, end))

        row_slices: list[AttnSlice] = []
        for prompt_group_id, (
            (prompt_start, prompt_end),
            family_index,
        ) in prompt_by_group_id.items():
            prompt_range = TokenRange(start=prompt_start, end=prompt_end)
            row_slices.append(
                AttnSlice(
                    q_range=prompt_range,
                    k_range=prompt_range,
                    mask_kind=AttnMaskKind.CAUSAL,
                    row_index=row_index,
                    family_index=family_index,
                )
            )
            for completion_start, completion_end in completion_ranges_by_prompt[
                prompt_group_id
            ]:
                completion_range = TokenRange(
                    start=completion_start,
                    end=completion_end,
                )
                row_slices.append(
                    AttnSlice(
                        q_range=completion_range,
                        k_range=prompt_range,
                        mask_kind=AttnMaskKind.FULL,
                        row_index=row_index,
                        family_index=family_index,
                    )
                )
                row_slices.append(
                    AttnSlice(
                        q_range=completion_range,
                        k_range=completion_range,
                        mask_kind=AttnMaskKind.CAUSAL,
                        row_index=row_index,
                        family_index=family_index,
                    )
                )

        rows.append(
            PackedRowAttentionSpec(
                row_index=row_index,
                valid_tokens=valid_tokens,
                slices=_sort_and_dedupe_slices(row_slices),
            )
        )

    return PackedBatchAttentionSpec(rows=tuple(rows))


def build_dense_reference_mask(
    *,
    row_spec: PackedRowAttentionSpec,
) -> torch.Tensor:
    dense = torch.zeros(
        (row_spec.valid_tokens, row_spec.valid_tokens),
        dtype=torch.bool,
    )
    for slice_ in row_spec.slices:
        q = slice_.q_range
        k = slice_.k_range
        if slice_.mask_kind is AttnMaskKind.FULL:
            dense[q.start : q.end, k.start : k.end] = True
            continue
        for q_idx in range(q.start, q.end):
            rel_q = q_idx - q.start
            max_k = k.start + rel_q
            if max_k < k.start:
                continue
            dense[q_idx, k.start : min(k.end, max_k + 1)] = True
    return dense
