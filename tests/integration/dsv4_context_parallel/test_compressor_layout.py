from __future__ import annotations

from oracles import branch_view_tokens
from pydantic import BaseModel, ConfigDict
import pytest
import torch

from art.megatron.dsv4 import (
    Dsv4BranchView,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    build_dsv4_compressed_layout,
)


class _LayoutIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    ownership_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    token_counts_by_rank: tuple[int, ...]


def test_csa_layout_reuses_shared_prefix_and_adds_branch_entries() -> None:
    layout = build_dsv4_compressed_layout(
        group_ids=_group_ids(),
        parent_ids=_parent_ids(),
        token_layout_index=_two_rank_layout(),
        spec=Dsv4CompressionSpec(kind=Dsv4CompressionKind.CSA, ratio=4),
    )

    assert [stream.stream_id for stream in layout.streams] == [0, 1, 2]
    assert [stream.kind.value for stream in layout.streams] == [
        "prefix",
        "completion",
        "completion",
    ]
    assert [view.branch_stream_id for view in layout.branch_views] == [0, 1, 2]
    assert _packed_tokens(layout.branch_views[1]) == tuple(range(8)) + tuple(
        range(8, 13)
    )
    assert _packed_tokens(layout.branch_views[2]) == tuple(range(8)) + tuple(
        range(13, 18)
    )

    assert layout.entry_ids_by_owner_rank == ((0, 1), (2, 3))
    assert layout.compressed_entry_owner_ranks == (0, 0, 1, 1)
    assert layout.entry_branch_stream_ids == (0, 0, 1, 2)
    assert layout.entry_prefix_stream_ids == (0, 0, 0, 0)
    assert layout.entry_closure_view_positions == (3, 7, 11, 11)
    assert layout.entry_shared_prefix_flags == (True, True, False, False)
    assert layout.entry_dependency_start_view_positions == (0, 4, 8, 8)
    assert layout.dependency_token_ids_by_owner_rank == (
        tuple(range(8)),
        tuple(range(4, 12)) + tuple(range(13, 17)),
    )
    assert layout.closure_token_ids == (3, 7, 11, 16)
    assert layout.closure_entry_ids == (0, 1, 2, 3)

    assert len(layout.halo_transfers) == 1
    halo = layout.halo_transfers[0]
    assert halo.source_rank == 0
    assert halo.target_rank == 1
    assert halo.token_ids == (4, 5, 6, 7)
    assert halo.entry_ids == (2, 3)


def test_hca_layout_drops_incomplete_tails_and_reuses_prefix_entries() -> None:
    layout = build_dsv4_compressed_layout(
        group_ids=_group_ids(),
        parent_ids=_parent_ids(),
        token_layout_index=_two_rank_layout(),
        spec=Dsv4CompressionSpec(kind=Dsv4CompressionKind.HCA, ratio=4),
    )

    assert layout.entry_count() == 4
    assert layout.compressed_entry_owner_ranks == (0, 0, 1, 1)
    assert layout.entry_branch_stream_ids == (0, 0, 1, 2)
    assert layout.entry_closure_view_positions == (3, 7, 11, 11)
    assert layout.entry_shared_prefix_flags == (True, True, False, False)
    assert layout.entry_dependency_start_view_positions == (0, 4, 8, 8)
    assert layout.dependency_token_ids_by_owner_rank == (
        tuple(range(8)),
        tuple(range(8, 12)) + tuple(range(13, 17)),
    )
    assert layout.closure_token_ids == (3, 7, 11, 16)
    assert layout.closure_entry_ids == (0, 1, 2, 3)
    assert layout.halo_transfers == ()


def test_branch_view_position_maps_only_visible_tokens() -> None:
    layout = build_dsv4_compressed_layout(
        group_ids=_group_ids(),
        parent_ids=_parent_ids(),
        token_layout_index=_two_rank_layout(),
        spec=Dsv4CompressionSpec(kind=Dsv4CompressionKind.CSA, ratio=4),
    )
    branch_one = layout.branch_views[1]

    assert branch_one.position_of_token(6) == 6
    assert branch_one.position_of_token(10) == 10
    assert branch_one.position_of_token(14) is None


def test_layout_rejects_non_tail_padding() -> None:
    group_ids = torch.tensor([[0, -1, 0]], dtype=torch.long)
    parent_ids = torch.tensor([[0, -1, 0]], dtype=torch.long)

    with pytest.raises(RuntimeError, match="contiguous tail"):
        build_dsv4_compressed_layout(
            group_ids=group_ids,
            parent_ids=parent_ids,
            token_layout_index=_LayoutIndex(
                ownership_ranges_by_rank=(((0, 3, 0),),),
                token_counts_by_rank=(3,),
            ),
            spec=Dsv4CompressionSpec(kind=Dsv4CompressionKind.CSA, ratio=4),
        )


def _group_ids() -> torch.Tensor:
    return torch.tensor([[0] * 8 + [1] * 5 + [2] * 5 + [-1] * 2], dtype=torch.long)


def _parent_ids() -> torch.Tensor:
    return torch.tensor([[0] * 8 + [0] * 5 + [0] * 5 + [-1] * 2], dtype=torch.long)


def _two_rank_layout() -> _LayoutIndex:
    return _LayoutIndex(
        ownership_ranges_by_rank=(
            ((0, 8, 0),),
            ((8, 18, 0),),
        ),
        token_counts_by_rank=(8, 10),
    )


def _packed_tokens(branch_view: Dsv4BranchView) -> tuple[int, ...]:
    return tuple(token_id for token_id, _ in branch_view_tokens(branch_view))
