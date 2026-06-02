from __future__ import annotations

from pydantic import BaseModel, ConfigDict
import pytest
import torch

from art.megatron.dsv4 import (
    Dsv4BranchView,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    build_dsv4_compressed_layout,
)
from art.megatron.dsv4.compressor import position_in_query_view


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

    assert layout.entry_ids_by_branch_stream == {
        0: (0, 1),
        1: (0, 1, 2),
        2: (0, 1, 3),
    }
    assert layout.entry_ids_by_owner_rank == ((0, 1), (2, 3))

    assert len(layout.entries) == 4
    assert layout.entries[0].dependency_token_ids == (0, 1, 2, 3)
    assert layout.entries[0].closure_token_id == 3
    assert layout.entries[0].shared_prefix_entry
    assert layout.entries[0].owner_rank == 0
    assert layout.entries[0].remote_dependency_token_ids == ()

    assert layout.entries[1].dependency_token_ids == (0, 1, 2, 3, 4, 5, 6, 7)
    assert layout.entries[1].closure_token_id == 7
    assert layout.entries[1].shared_prefix_entry
    assert layout.entries[1].owner_rank == 0
    assert layout.entries[1].remote_dependency_token_ids == ()

    assert layout.entries[2].dependency_token_ids == (4, 5, 6, 7, 8, 9, 10, 11)
    assert layout.entries[2].closure_token_id == 11
    assert not layout.entries[2].shared_prefix_entry
    assert layout.entries[2].owner_rank == 1
    assert layout.entries[2].remote_dependency_token_ids == (4, 5, 6, 7)

    assert layout.entries[3].dependency_token_ids == (4, 5, 6, 7, 13, 14, 15, 16)
    assert layout.entries[3].closure_token_id == 16
    assert not layout.entries[3].shared_prefix_entry
    assert layout.entries[3].owner_rank == 1
    assert layout.entries[3].remote_dependency_token_ids == (4, 5, 6, 7)

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

    assert layout.entry_ids_by_branch_stream == {
        0: (0, 1),
        1: (0, 1, 2),
        2: (0, 1, 3),
    }
    assert [entry.dependency_token_ids for entry in layout.entries] == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (13, 14, 15, 16),
    ]
    assert [entry.closure_token_id for entry in layout.entries] == [3, 7, 11, 16]
    assert [entry.shared_prefix_entry for entry in layout.entries] == [
        True,
        True,
        False,
        False,
    ]
    assert layout.entries[2].remote_dependency_token_ids == ()
    assert layout.entries[3].remote_dependency_token_ids == ()
    assert layout.halo_transfers == ()


def test_branch_view_position_maps_only_visible_tokens() -> None:
    layout = build_dsv4_compressed_layout(
        group_ids=_group_ids(),
        parent_ids=_parent_ids(),
        token_layout_index=_two_rank_layout(),
        spec=Dsv4CompressionSpec(kind=Dsv4CompressionKind.CSA, ratio=4),
    )
    branch_one = layout.branch_views[1]

    assert position_in_query_view(branch_view=branch_one, candidate_token_id=6) == 6
    assert position_in_query_view(branch_view=branch_one, candidate_token_id=10) == 10
    assert position_in_query_view(branch_view=branch_one, candidate_token_id=14) is None


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
    return tuple(int(token.packed_token_id) for token in branch_view.tokens)
