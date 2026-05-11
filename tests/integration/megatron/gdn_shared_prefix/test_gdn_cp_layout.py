from __future__ import annotations

import pytest
import torch

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.gdn.layout import (
    GdnCpExchangePlan,
    build_cp_exchange_plan_from_rank_ranges,
    build_gdn_cp_layout_plan,
    recv_split_sizes_for_rank,
    send_split_sizes_for_rank,
    simulate_all_to_all_single,
    split_gdn_families_by_rank,
)

from .cases import (
    GdnFamilyShape,
    GdnPackedRowShape,
    GdnPhase0Case,
    default_phase0_cases,
)
from .metrics import GDN_CORRECTNESS_DTYPE
from .packed_layout import build_phase0_packed_tensors
from .parser_import import parse_gdn_shared_prefix_segments


@pytest.mark.parametrize("cp_size", (2, 4, 8))
def test_gdn_cp_layout_roundtrips_generated_cases(cp_size: int) -> None:
    for case in default_phase0_cases(conv_width=4):
        tensors = build_phase0_packed_tensors(case)
        real_indices = _real_token_indices(tensors["group_ids"])
        attention_indices = _striped_rank_indices(real_indices, cp_size=cp_size)
        plan = build_gdn_cp_layout_plan(
            group_ids=tensors["group_ids"],
            parent_ids=tensors["parent_ids"],
            cp_size=cp_size,
            attention_token_layout_index=_layout_from_tokens_by_rank(attention_indices),
        )

        assert set(_tokens_from_rank_ranges(plan.gdn_token_ranges_by_rank)) == set(
            real_indices
        )
        assert any(
            len(rank_ranges) == 0 for rank_ranges in plan.gdn_token_ranges_by_rank
        ) == (len(real_indices) < cp_size)
        if len(real_indices) > cp_size:
            assert plan.attention_to_gdn.cross_rank_token_count > 0

        flat = torch.arange(
            int(tensors["group_ids"].numel()) * 3,
            dtype=GDN_CORRECTNESS_DTYPE,
        ).reshape(-1, 3)
        source = _rank_tensors(
            flat, _tokens_by_rank_from_ranges(plan.attention_token_ranges_by_rank)
        )
        _assert_split_sizes_are_consistent(plan.attention_to_gdn)
        _assert_split_sizes_are_consistent(plan.gdn_to_attention)
        gdn_order = simulate_all_to_all_single(source, plan.attention_to_gdn)
        restored = simulate_all_to_all_single(gdn_order, plan.gdn_to_attention)

        assert len(restored) == cp_size
        for rank, restored_rank in enumerate(restored):
            assert torch.equal(restored_rank, source[rank])


@pytest.mark.parametrize("cp_size", (2, 4, 8))
def test_gdn_cp_layout_roundtrip_preserves_gradients(cp_size: int) -> None:
    tensors = build_phase0_packed_tensors(
        next(
            case
            for case in default_phase0_cases(conv_width=4)
            if case.name == "ragged_family_mix"
        )
    )
    real_indices = _real_token_indices(tensors["group_ids"])
    plan = build_gdn_cp_layout_plan(
        group_ids=tensors["group_ids"],
        parent_ids=tensors["parent_ids"],
        cp_size=cp_size,
        attention_token_layout_index=_layout_from_tokens_by_rank(
            _striped_rank_indices(
                tuple(reversed(real_indices)),
                cp_size=cp_size,
            )
        ),
    )

    flat = torch.randn(
        int(tensors["group_ids"].numel()),
        2,
        3,
        generator=torch.Generator().manual_seed(1234),
        requires_grad=True,
    )
    attention_tokens_by_rank = _tokens_by_rank_from_ranges(
        plan.attention_token_ranges_by_rank
    )
    source = _rank_tensors(flat, attention_tokens_by_rank)
    gdn_order = simulate_all_to_all_single(source, plan.attention_to_gdn)
    restored = simulate_all_to_all_single(gdn_order, plan.gdn_to_attention)

    expected_grad = torch.zeros_like(flat)
    loss = flat.new_zeros(())
    for rank, restored_rank in enumerate(restored):
        weight = torch.arange(
            restored_rank.numel(),
            device=restored_rank.device,
            dtype=restored_rank.dtype,
        ).reshape_as(restored_rank)
        loss = loss + (restored_rank * weight).sum()
        for local_pos, token_index in enumerate(attention_tokens_by_rank[rank]):
            expected_grad[token_index] = weight[local_pos]
    loss.backward()

    assert flat.grad is not None
    assert torch.equal(flat.grad, expected_grad)


def test_gdn_cp_layout_handles_empty_ranks() -> None:
    case = GdnPhase0Case(
        name="tiny_empty_rank",
        sequence_length=8,
        rows=(
            GdnPackedRowShape(
                families=(GdnFamilyShape(prefix_length=2, suffix_lengths=(1,)),)
            ),
        ),
    )
    tensors = build_phase0_packed_tensors(case)
    cp_size = 8
    plan = build_gdn_cp_layout_plan(
        group_ids=tensors["group_ids"],
        parent_ids=tensors["parent_ids"],
        cp_size=cp_size,
        attention_token_layout_index=_layout_from_tokens_by_rank(
            ((), (), (), (0,), (), (), (1, 2), ())
        ),
    )

    assert sum(len(rank) == 0 for rank in plan.gdn_token_ranges_by_rank) == 5
    flat = torch.arange(8 * 4, dtype=GDN_CORRECTNESS_DTYPE).reshape(8, 4)
    source = _rank_tensors(
        flat, _tokens_by_rank_from_ranges(plan.attention_token_ranges_by_rank)
    )
    gdn_order = simulate_all_to_all_single(source, plan.attention_to_gdn)
    restored = simulate_all_to_all_single(gdn_order, plan.gdn_to_attention)

    for rank, restored_rank in enumerate(restored):
        assert torch.equal(restored_rank, source[rank])


@pytest.mark.parametrize("cp_size", (2, 4, 8))
def test_gdn_cp_family_split_keeps_whole_families_on_one_rank(cp_size: int) -> None:
    tensors = build_phase0_packed_tensors(
        next(
            case
            for case in default_phase0_cases(conv_width=4)
            if case.name == "ragged_family_mix"
        )
    )
    spec = parse_gdn_shared_prefix_segments(
        tensors["group_ids"], tensors["parent_ids"], min_completions_per_family=0
    )
    gdn_indices = split_gdn_families_by_rank(spec, cp_size=cp_size)
    token_to_rank = {
        token_index: rank
        for rank, rank_tokens in enumerate(gdn_indices)
        for token_index in rank_tokens
    }

    for family in spec.families:
        family_ranks = {
            token_to_rank[token_index]
            for segment in (family.prefix, *family.completions)
            for token_index in segment.linear_indices(spec.sequence_length)
        }
        assert len(family_ranks) == 1

    plan = build_gdn_cp_layout_plan(
        group_ids=tensors["group_ids"],
        parent_ids=tensors["parent_ids"],
        cp_size=cp_size,
        gdn_token_ranges_by_rank=_rank_ranges_from_tokens_by_rank(gdn_indices),
    )
    assert _tokens_by_rank_from_ranges(plan.gdn_token_ranges_by_rank) == gdn_indices


def test_gdn_cp_layout_rejects_duplicate_or_missing_attention_tokens() -> None:
    tensors = build_phase0_packed_tensors(default_phase0_cases(conv_width=4)[0])
    real_indices = _real_token_indices(tensors["group_ids"])
    valid_source = _striped_rank_indices(real_indices, cp_size=2)
    duplicated = (valid_source[0] + (valid_source[0][0],), valid_source[1])
    with pytest.raises(ValueError, match="cover the same tokens"):
        build_gdn_cp_layout_plan(
            group_ids=tensors["group_ids"],
            parent_ids=tensors["parent_ids"],
            cp_size=2,
            attention_token_layout_index=_layout_from_tokens_by_rank(duplicated),
        )

    missing = (valid_source[0][:-1], valid_source[1])
    with pytest.raises(ValueError, match="cover the same tokens"):
        build_gdn_cp_layout_plan(
            group_ids=tensors["group_ids"],
            parent_ids=tensors["parent_ids"],
            cp_size=2,
            attention_token_layout_index=_layout_from_tokens_by_rank(missing),
        )


@pytest.mark.parametrize(
    ("source_indices", "dest_indices"),
    (
        (
            ((0, 2, 1, 3), (4, 6, 5, 7)),
            ((0, 1, 2, 3), (4, 5, 6, 7)),
        ),
        (
            ((0, 3, 4), (1, 2, 5)),
            ((0, 1, 2), (3, 4, 5)),
        ),
    ),
)
def test_cp_exchange_plan_does_not_trust_dense_layout_endpoints(
    source_indices: tuple[tuple[int, ...], ...],
    dest_indices: tuple[tuple[int, ...], ...],
) -> None:
    plan = build_cp_exchange_plan_from_rank_ranges(
        source_ranges_by_rank=_rank_ranges_from_tokens_by_rank(source_indices),
        dest_ranges_by_rank=_rank_ranges_from_tokens_by_rank(dest_indices),
        device="cpu",
        validate=False,
    )

    flat = torch.arange(
        sum(len(indices) for indices in source_indices),
        dtype=GDN_CORRECTNESS_DTYPE,
    ).unsqueeze(-1)
    source = _rank_tensors(flat, source_indices)
    actual = simulate_all_to_all_single(source, plan)
    expected = _rank_tensors(flat, dest_indices)

    for actual_rank, expected_rank in zip(actual, expected, strict=True):
        assert torch.equal(actual_rank, expected_rank)


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
        ownership_ranges_by_rank=_rank_ranges_from_tokens_by_rank(tokens_by_rank),
        token_counts_by_rank=tuple(len(tokens) for tokens in tokens_by_rank),
    )


def _rank_ranges_from_tokens_by_rank(
    tokens_by_rank: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    return tuple(_rank_ranges_from_tokens(tokens) for tokens in tokens_by_rank)


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


def _tokens_from_rank_ranges(
    ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...],
) -> tuple[int, ...]:
    return tuple(
        token
        for rank_ranges in ranges_by_rank
        for start, end, _ in rank_ranges
        for token in range(start, end)
    )


def _rank_tensors(
    flat: torch.Tensor,
    indices_by_rank: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, ...]:
    return tuple(
        flat.index_select(
            0,
            torch.tensor(indices, device=flat.device, dtype=torch.long),
        )
        for indices in indices_by_rank
    )


def _assert_split_sizes_are_consistent(plan: GdnCpExchangePlan) -> None:
    cp_size = int(getattr(plan, "cp_size"))
    for rank in range(cp_size):
        assert (
            sum(send_split_sizes_for_rank(plan, rank))
            == plan.source_token_counts_by_rank[rank]
        )
        assert (
            sum(recv_split_sizes_for_rank(plan, rank))
            == plan.dest_token_counts_by_rank[rank]
        )
