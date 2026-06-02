from __future__ import annotations

from typing import Any

from cases import Dsv4WorkloadCase, dsv4_family_token_count, dsv4_row_token_count
from pydantic import BaseModel, ConfigDict, Field
import torch


class Dsv4CaseSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    total_tokens: int
    family_count: int
    completion_count: int
    max_segment_length: int
    min_completion_length: int
    max_completion_length: int
    unique_completion_lengths: tuple[int, ...]
    completion_lengths_vary: bool
    cp_boundary_prefix: bool
    cp_boundary_completion: bool
    family_boundary_at_partition: bool
    empty_trailing_rank: bool
    csa_ratio_boundary: bool
    hca_ratio_boundary: bool
    swa_boundary: bool
    topk_tie_or_near_tie: bool
    no_stage_keys: bool
    valid_lengths: tuple[int, ...]
    tags: tuple[str, ...] = Field(default_factory=tuple)


def build_dsv4_packed_tensors(case: Dsv4WorkloadCase) -> dict[str, Any]:
    shape = (len(case.rows), case.sequence_length)
    generator = torch.Generator().manual_seed(case.seed)
    tokens = torch.zeros(shape, dtype=torch.long)
    group_ids = torch.full(shape, -1, dtype=torch.long)
    parent_ids = torch.full(shape, -1, dtype=torch.long)
    input_pos = torch.zeros(shape, dtype=torch.long)
    assistant_mask = torch.zeros(shape, dtype=torch.bool)
    logprobs = torch.full(shape, float("nan"), dtype=torch.float32)
    advantages = torch.zeros(shape, dtype=torch.float32)
    weights = torch.zeros(shape, dtype=torch.float32)

    for row_index, row in enumerate(case.rows):
        cursor = 0
        next_group_id = row_index * 100_000
        for family in row.families:
            required = dsv4_family_token_count(family)
            if cursor + required > case.sequence_length:
                raise ValueError(
                    f"case {case.name} row {row_index}: family requires {required} "
                    f"tokens with only {case.sequence_length - cursor} remaining"
                )
            prefix_group_id = next_group_id
            next_group_id += 1
            prefix_end = cursor + family.prefix_length
            _write_tokens(tokens, row_index, cursor, prefix_end, generator)
            group_ids[row_index, cursor:prefix_end] = prefix_group_id
            parent_ids[row_index, cursor:prefix_end] = prefix_group_id
            input_pos[row_index, cursor:prefix_end] = torch.arange(
                family.prefix_length, dtype=torch.long
            )
            cursor = prefix_end

            for completion_length in family.completion_lengths:
                completion_group_id = next_group_id
                next_group_id += 1
                completion_end = cursor + completion_length
                _write_tokens(tokens, row_index, cursor, completion_end, generator)
                group_ids[row_index, cursor:completion_end] = completion_group_id
                parent_ids[row_index, cursor:completion_end] = prefix_group_id
                input_pos[row_index, cursor:completion_end] = torch.arange(
                    family.prefix_length,
                    family.prefix_length + completion_length,
                    dtype=torch.long,
                )
                trainable_start = cursor + 1
                assistant_mask[row_index, trainable_start:completion_end] = True
                logprobs[row_index, trainable_start:completion_end] = _sample_logprobs(
                    completion_length - 1, generator
                )
                advantages[row_index, trainable_start:completion_end] = (
                    _sample_advantage(generator)
                )
                weights[row_index, trainable_start:completion_end] = 1.0 / (
                    completion_length - 1
                )
                cursor = completion_end

    return {
        "tokens": tokens,
        "group_ids": group_ids,
        "parent_ids": parent_ids,
        "input_pos": input_pos,
        "assistant_mask": assistant_mask,
        "logprobs": logprobs,
        "advantages": advantages,
        "weights": weights,
        "pixel_values": [None] * len(case.rows),
        "image_grid_thw": [None] * len(case.rows),
    }


def build_dsv4_group_parent_tensors(case: Dsv4WorkloadCase) -> dict[str, torch.Tensor]:
    shape = (len(case.rows), case.sequence_length)
    group_ids = torch.full(shape, -1, dtype=torch.long)
    parent_ids = torch.full(shape, -1, dtype=torch.long)
    for row_index, row in enumerate(case.rows):
        cursor = 0
        next_group_id = row_index * 100_000
        for family in row.families:
            required = dsv4_family_token_count(family)
            if cursor + required > case.sequence_length:
                raise ValueError(
                    f"case {case.name} row {row_index}: family requires {required} "
                    f"tokens with only {case.sequence_length - cursor} remaining"
                )
            prefix_group_id = next_group_id
            next_group_id += 1
            prefix_end = cursor + family.prefix_length
            group_ids[row_index, cursor:prefix_end] = prefix_group_id
            parent_ids[row_index, cursor:prefix_end] = prefix_group_id
            cursor = prefix_end
            for completion_length in family.completion_lengths:
                completion_group_id = next_group_id
                next_group_id += 1
                completion_end = cursor + completion_length
                group_ids[row_index, cursor:completion_end] = completion_group_id
                parent_ids[row_index, cursor:completion_end] = prefix_group_id
                cursor = completion_end
    return {"group_ids": group_ids, "parent_ids": parent_ids}


def summarize_case(
    case: Dsv4WorkloadCase,
    *,
    cp_sizes: tuple[int, ...] = (2, 4, 8),
    csa_ratio: int = 4,
    hca_ratio: int = 128,
    swa_window: int = 128,
) -> Dsv4CaseSummary:
    completion_lengths = tuple(
        int(length)
        for row in case.rows
        for family in row.families
        for length in family.completion_lengths
    )
    valid_lengths = tuple(dsv4_row_token_count(row) for row in case.rows)
    boundary = _boundary_flags(case, cp_sizes)
    tags = set(case.tags)
    starts_and_ends = tuple(_segment_starts_and_ends(case))
    return Dsv4CaseSummary(
        name=case.name,
        total_tokens=sum(valid_lengths),
        family_count=sum(len(row.families) for row in case.rows),
        completion_count=len(completion_lengths),
        max_segment_length=max(
            (
                max((family.prefix_length, *family.completion_lengths))
                for row in case.rows
                for family in row.families
            ),
            default=0,
        ),
        min_completion_length=min(completion_lengths, default=0),
        max_completion_length=max(completion_lengths, default=0),
        unique_completion_lengths=tuple(sorted(set(completion_lengths))),
        completion_lengths_vary=(
            "randomized_completions" in tags or len(set(completion_lengths)) > 1
        ),
        cp_boundary_prefix=boundary["cp_boundary_prefix"],
        cp_boundary_completion=boundary["cp_boundary_completion"],
        family_boundary_at_partition=boundary["family_boundary_at_partition"],
        empty_trailing_rank=boundary["empty_trailing_rank"],
        csa_ratio_boundary=(
            "csa_ratio_boundary" in tags
            or any(position % csa_ratio != 0 for position in starts_and_ends)
        ),
        hca_ratio_boundary=(
            "hca_ratio_boundary" in tags
            or any(position % hca_ratio != 0 for position in starts_and_ends)
        ),
        swa_boundary=(
            "swa_boundary" in tags
            or any(
                length in (swa_window - 1, swa_window, swa_window + 1)
                for length in completion_lengths
            )
        ),
        topk_tie_or_near_tie="topk_tie_or_near_tie" in tags,
        no_stage_keys="no_stage_keys" in tags,
        valid_lengths=valid_lengths,
        tags=case.tags,
    )


def format_case_summary(summary: Dsv4CaseSummary) -> str:
    flags = [
        name
        for name in (
            "completion_lengths_vary",
            "cp_boundary_prefix",
            "cp_boundary_completion",
            "family_boundary_at_partition",
            "empty_trailing_rank",
            "csa_ratio_boundary",
            "hca_ratio_boundary",
            "swa_boundary",
            "topk_tie_or_near_tie",
            "no_stage_keys",
        )
        if getattr(summary, name)
    ]
    return (
        f"{summary.name}: tokens={summary.total_tokens} "
        f"families={summary.family_count} completions={summary.completion_count} "
        f"completion_lengths={summary.min_completion_length}"
        f"..{summary.max_completion_length} "
        f"unique={len(summary.unique_completion_lengths)} "
        f"max_segment={summary.max_segment_length} "
        f"valid_lengths={summary.valid_lengths} flags={','.join(flags) or 'none'}"
    )


def _write_tokens(
    tokens: torch.Tensor,
    row_index: int,
    start: int,
    end: int,
    generator: torch.Generator,
) -> None:
    tokens[row_index, start:end] = torch.randint(
        low=10, high=8192, size=(end - start,), dtype=torch.long, generator=generator
    )


def _sample_logprobs(length: int, generator: torch.Generator) -> torch.Tensor:
    return (
        torch.randn((length,), generator=generator, dtype=torch.float32) * 0.25 - 1.75
    )


def _sample_advantage(generator: torch.Generator) -> float:
    return float(
        (torch.randn((1,), generator=generator, dtype=torch.float32) * 0.5).item()
    )


def _boundary_flags(
    case: Dsv4WorkloadCase, cp_sizes: tuple[int, ...]
) -> dict[str, bool]:
    flags = {
        "cp_boundary_prefix": False,
        "cp_boundary_completion": False,
        "family_boundary_at_partition": False,
        "empty_trailing_rank": False,
    }
    total_tokens = sum(dsv4_row_token_count(row) for row in case.rows)
    if total_tokens == 0:
        return flags
    for cp_size in cp_sizes:
        shard = (total_tokens + cp_size - 1) // cp_size
        boundaries = {shard * rank for rank in range(1, cp_size)}
        if shard * (cp_size - 1) >= total_tokens:
            flags["empty_trailing_rank"] = True
        for segment in _segments(case):
            start = int(segment["global_start"])
            end = int(segment["global_end"])
            if segment["kind"] == "family" and (
                start in boundaries or end in boundaries
            ):
                flags["family_boundary_at_partition"] = True
            if any(start < boundary < end for boundary in boundaries):
                if segment["kind"] == "prefix":
                    flags["cp_boundary_prefix"] = True
                if segment["kind"] == "completion":
                    flags["cp_boundary_completion"] = True
    return flags


def _segment_starts_and_ends(case: Dsv4WorkloadCase) -> list[int]:
    positions: list[int] = []
    for segment in _segments(case):
        if segment["kind"] in {"prefix", "completion"}:
            positions.extend((int(segment["global_start"]), int(segment["global_end"])))
    return positions


def _segments(case: Dsv4WorkloadCase) -> list[dict[str, int | str]]:
    segments: list[dict[str, int | str]] = []
    global_cursor = 0
    for row_index, row in enumerate(case.rows):
        row_cursor = 0
        for family_index, family in enumerate(row.families):
            family_global_start = global_cursor
            prefix_start = row_cursor
            prefix_global_start = global_cursor
            row_cursor += family.prefix_length
            global_cursor += family.prefix_length
            segments.append(
                {
                    "kind": "prefix",
                    "row": row_index,
                    "family": family_index,
                    "start": prefix_start,
                    "end": row_cursor,
                    "global_start": prefix_global_start,
                    "global_end": global_cursor,
                }
            )
            for completion_index, completion_length in enumerate(
                family.completion_lengths
            ):
                completion_start = row_cursor
                completion_global_start = global_cursor
                row_cursor += completion_length
                global_cursor += completion_length
                segments.append(
                    {
                        "kind": "completion",
                        "row": row_index,
                        "family": family_index,
                        "completion": completion_index,
                        "start": completion_start,
                        "end": row_cursor,
                        "global_start": completion_global_start,
                        "global_end": global_cursor,
                    }
                )
            segments.append(
                {
                    "kind": "family",
                    "row": row_index,
                    "family": family_index,
                    "start": 0,
                    "end": row_cursor,
                    "global_start": family_global_start,
                    "global_end": global_cursor,
                }
            )
    return segments
