from __future__ import annotations

from oracles import dense_dsv4_packed_attention_oracle
from pydantic import BaseModel, ConfigDict
import torch

from art.megatron.dsv4 import (
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    build_dsv4_compressed_layout,
)


class _LayoutIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    ownership_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    token_counts_by_rank: tuple[int, ...]


def test_dense_oracle_matches_unpacked_branch_views_and_shared_prefix() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    torch.manual_seed(101)
    query = torch.randn(18, 3, 5, dtype=torch.float64)
    raw_kv = torch.randn(18, 5, dtype=torch.float64)
    compressed_kv = torch.randn(len(layout.entries), 5, dtype=torch.float64)
    attn_sink = torch.randn(3, dtype=torch.float64)
    topk = _all_visible_topk(layout)

    result = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=query,
        raw_kv=raw_kv,
        compressed_kv=compressed_kv,
        attn_sink=attn_sink,
        topk_by_query=topk,
        window_size=128,
        scale=0.7,
    )

    assert result.query_token_ids == tuple(range(18))
    assert result.out.shape == (1, 18, 3, 5)
    assert result.lse.shape == (1, 18, 3)
    completion_branch_ids = {
        int(branch.branch_stream_id)
        for branch in layout.branch_views
        if branch.suffix_stream_id is not None
    }
    prefix_results = [
        branch.out[:, :8]
        for branch in result.branches
        if int(branch.branch_stream_id) in completion_branch_ids
    ]
    assert len(prefix_results) == 2
    torch.testing.assert_close(prefix_results[0], prefix_results[1])
    torch.testing.assert_close(result.out[:, :8], prefix_results[0])
    assert torch.isfinite(result.out).all()
    assert torch.isfinite(result.lse).all()


def test_dense_oracle_rejects_sibling_compressed_leakage_from_bad_topk() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    torch.manual_seed(103)
    query = torch.randn(18, 2, 4, dtype=torch.float64)
    raw_kv = torch.randn(18, 4, dtype=torch.float64)
    compressed_kv = torch.randn(len(layout.entries), 4, dtype=torch.float64)
    attn_sink = torch.randn(2, dtype=torch.float64)
    topk = torch.full((18, 3), -1, dtype=torch.long)
    query_token_id = 12
    sibling_entry_id = next(
        entry.entry_id
        for entry in layout.entries
        if entry.branch_stream_id == 2 and not entry.shared_prefix_entry
    )
    visible_entry_id = next(
        entry.entry_id
        for entry in layout.entries
        if entry.branch_stream_id == 1 and not entry.shared_prefix_entry
    )
    topk[query_token_id] = torch.tensor([sibling_entry_id, visible_entry_id, -1])
    without_sibling = topk.clone()
    without_sibling[query_token_id] = torch.tensor([visible_entry_id, -1, -1])

    with_bad_topk = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=query,
        raw_kv=raw_kv,
        compressed_kv=compressed_kv,
        attn_sink=attn_sink,
        topk_by_query=topk,
        window_size=128,
        scale=1.0,
    )
    expected = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=query,
        raw_kv=raw_kv,
        compressed_kv=compressed_kv,
        attn_sink=attn_sink,
        topk_by_query=without_sibling,
        window_size=128,
        scale=1.0,
    )
    torch.testing.assert_close(
        with_bad_topk.out[:, query_token_id],
        expected.out[:, query_token_id],
    )
    torch.testing.assert_close(
        with_bad_topk.lse[:, query_token_id],
        expected.lse[:, query_token_id],
    )


def _all_visible_topk(layout: Dsv4CompressedLayout) -> torch.Tensor:
    max_visible = max(
        sum(
            1
            for entry in layout.entries
            if (
                int(entry.branch_stream_id) == int(branch.branch_stream_id)
                or (
                    entry.shared_prefix_entry
                    and int(entry.prefix_stream_id) == int(branch.prefix_stream_id)
                )
            )
            and int(entry.closure_view_pos) <= int(token.view_pos)
        )
        for branch in layout.branch_views
        for token in branch.tokens
    )
    topk = torch.full((18, max_visible), -1, dtype=torch.long)
    for branch in layout.branch_views:
        for token in branch.tokens:
            visible = [
                int(entry.entry_id)
                for entry in layout.entries
                if (
                    int(entry.branch_stream_id) == int(branch.branch_stream_id)
                    or (
                        entry.shared_prefix_entry
                        and int(entry.prefix_stream_id) == int(branch.prefix_stream_id)
                    )
                )
                and int(entry.closure_view_pos) <= int(token.view_pos)
            ]
            topk[int(token.packed_token_id), : len(visible)] = torch.tensor(visible)
    return topk


def _layout(kind: Dsv4CompressionKind) -> Dsv4CompressedLayout:
    return build_dsv4_compressed_layout(
        group_ids=torch.tensor([[0] * 8 + [1] * 5 + [2] * 5 + [-1] * 2]),
        parent_ids=torch.tensor([[0] * 8 + [0] * 5 + [0] * 5 + [-1] * 2]),
        token_layout_index=_LayoutIndex(
            ownership_ranges_by_rank=(
                ((0, 8, 0),),
                ((8, 18, 0),),
            ),
            token_counts_by_rank=(8, 10),
        ),
        spec=Dsv4CompressionSpec(kind=kind, ratio=4),
    )
