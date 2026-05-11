from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.multiprocessing as mp

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.gdn.layout import (
    build_gdn_cp_layout_plan,
    exchange_rank_tensor_all_to_all,
)

from .cases import (
    GdnFamilyShape,
    GdnPackedRowShape,
    GdnPhase0Case,
    default_phase0_cases,
)
from .metrics import GDN_CORRECTNESS_DTYPE
from .packed_layout import build_phase0_packed_tensors


@pytest.mark.parametrize("cp_size", (2, 4, 8))
def test_distributed_gdn_cp_layout_all_to_all_roundtrips(
    cp_size: int, tmp_path: Path
) -> None:
    init_path = tmp_path / f"gdn_cp_layout_gloo_{cp_size}"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_layout_worker,
        args=(cp_size, str(init_path), "ragged_family_mix", True),
        nprocs=cp_size,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def test_distributed_gdn_cp_layout_handles_empty_ranks(tmp_path: Path) -> None:
    cp_size = 8
    init_path = tmp_path / "gdn_cp_layout_gloo_empty"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_layout_worker,
        args=(cp_size, str(init_path), "tiny_empty_rank", False),
        nprocs=cp_size,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def _distributed_layout_worker(
    rank: int,
    world_size: int,
    init_path: str,
    case_name: str,
    reverse_attention: bool,
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        case = _case_by_name(case_name)
        tensors = build_phase0_packed_tensors(case)
        real_indices = _real_token_indices(tensors["group_ids"])
        attention_order = (
            tuple(reversed(real_indices)) if reverse_attention else real_indices
        )
        plan = build_gdn_cp_layout_plan(
            group_ids=tensors["group_ids"],
            parent_ids=tensors["parent_ids"],
            cp_size=world_size,
            attention_token_layout_index=_layout_from_tokens_by_rank(
                _striped_rank_indices(attention_order, cp_size=world_size)
            ),
        )

        flat = torch.arange(
            int(tensors["group_ids"].numel()) * 6,
            dtype=GDN_CORRECTNESS_DTYPE,
        ).reshape(-1, 2, 3)
        local_source = flat.index_select(
            0,
            torch.tensor(
                _tokens_by_rank_from_ranges(plan.attention_token_ranges_by_rank)[rank],
                dtype=torch.long,
            ),
        )
        local_source = local_source.detach().clone().requires_grad_(True)

        gdn_local = exchange_rank_tensor_all_to_all(
            local_source,
            plan.attention_to_gdn,
            rank=rank,
            backward_plan=plan.gdn_to_attention,
        )
        expected_gdn = flat.index_select(
            0,
            torch.tensor(
                _tokens_by_rank_from_ranges(plan.gdn_token_ranges_by_rank)[rank],
                dtype=torch.long,
            ),
        )
        torch.testing.assert_close(gdn_local, expected_gdn, rtol=0, atol=0)

        restored = exchange_rank_tensor_all_to_all(
            gdn_local,
            plan.gdn_to_attention,
            rank=rank,
            backward_plan=plan.attention_to_gdn,
        )
        torch.testing.assert_close(restored, local_source, rtol=0, atol=0)

        weight = torch.arange(
            restored.numel(),
            dtype=restored.dtype,
            device=restored.device,
        ).reshape_as(restored)
        (restored * weight).sum().backward()
        assert local_source.grad is not None
        torch.testing.assert_close(local_source.grad, weight, rtol=0, atol=0)
    finally:
        destroy_process_group()


def _case_by_name(case_name: str) -> GdnPhase0Case:
    if case_name == "tiny_empty_rank":
        return GdnPhase0Case(
            name="tiny_empty_rank",
            sequence_length=8,
            rows=(
                GdnPackedRowShape(
                    families=(GdnFamilyShape(prefix_length=2, suffix_lengths=(1,)),)
                ),
            ),
        )
    return next(
        case for case in default_phase0_cases(conv_width=4) if case.name == case_name
    )


def _real_token_indices(group_ids: torch.Tensor) -> tuple[int, ...]:
    sequence_length = int(group_ids.shape[1])
    return tuple(
        row * sequence_length + position
        for row in range(int(group_ids.shape[0]))
        for position in torch.nonzero(group_ids[row] != -1, as_tuple=False)
        .flatten()
        .tolist()
    )


def _striped_rank_indices(
    token_indices: tuple[int, ...],
    *,
    cp_size: int,
) -> tuple[tuple[int, ...], ...]:
    ranks: list[list[int]] = [[] for _ in range(cp_size)]
    for offset, token_index in enumerate(token_indices):
        ranks[offset % cp_size].append(token_index)
    return tuple(tuple(rank_indices) for rank_indices in ranks)


def _layout_from_tokens_by_rank(
    tokens_by_rank: tuple[tuple[int, ...], ...],
) -> TokenLayoutIndex:
    return TokenLayoutIndex(
        ownership_ranges_by_rank=tuple(
            _rank_ranges_from_tokens(tokens) for tokens in tokens_by_rank
        ),
        token_counts_by_rank=tuple(len(tokens) for tokens in tokens_by_rank),
    )


def _rank_ranges_from_tokens(
    tokens: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    if not tokens:
        return ()
    ranges = []
    start = tokens[0]
    end = start + 1
    position = 0
    for local_position, token in enumerate(tokens[1:], start=1):
        if token == end:
            end += 1
            continue
        ranges.append((start, end, position))
        start = token
        end = token + 1
        position = local_position
    ranges.append((start, end, position))
    return tuple(ranges)


def _tokens_by_rank_from_ranges(
    ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(token for start, end, _ in ranges for token in range(start, end))
        for ranges in ranges_by_rank
    )
