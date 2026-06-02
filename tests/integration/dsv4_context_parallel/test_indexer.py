from __future__ import annotations

from pydantic import BaseModel, ConfigDict
import torch

from art.megatron.dsv4 import (
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    Dsv4TopkResult,
    build_dsv4_compressed_layout,
    build_indexer_visibility_mask,
    compute_indexer_scores,
    compute_indexer_topk,
    merge_indexer_topk_results,
    stable_topk_by_score_and_id,
    stage_candidate_entry_ids,
    visible_entry_ids_for_query,
)


class _LayoutIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    ownership_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    token_counts_by_rank: tuple[int, ...]


class _Range(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int
    end: int


def test_indexer_visibility_respects_shared_prefix_and_sibling_boundaries() -> None:
    layout = _layout()

    assert visible_entry_ids_for_query(layout=layout, query_token_id=2) == ()
    assert visible_entry_ids_for_query(layout=layout, query_token_id=3) == (0,)
    assert visible_entry_ids_for_query(layout=layout, query_token_id=7) == (0, 1)
    assert visible_entry_ids_for_query(layout=layout, query_token_id=8) == (0, 1)
    assert visible_entry_ids_for_query(layout=layout, query_token_id=11) == (0, 1, 2)
    assert visible_entry_ids_for_query(layout=layout, query_token_id=14) == (0, 1)
    assert visible_entry_ids_for_query(layout=layout, query_token_id=16) == (0, 1, 3)

    assert stage_candidate_entry_ids(
        layout=layout,
        global_k_ranges=(_Range(start=0, end=8),),
    ) == (0, 1)
    assert stage_candidate_entry_ids(
        layout=layout,
        global_k_ranges=(_Range(start=8, end=13),),
    ) == (2,)
    assert stage_candidate_entry_ids(
        layout=layout,
        global_k_ranges=(_Range(start=13, end=18),),
    ) == (3,)


def test_indexer_scores_match_reference_formula_and_visibility() -> None:
    layout = _layout()
    torch.manual_seed(5)
    query_token_ids = (3, 8, 11, 16)
    candidate_entry_ids = (0, 1, 2, 3)
    q = torch.randn(len(query_token_ids), 3, 5)
    kv = torch.randn(len(candidate_entry_ids), 5)
    weights = torch.randn(len(query_token_ids), 3)
    scale = 0.25

    mask = build_indexer_visibility_mask(
        layout=layout,
        query_token_ids=query_token_ids,
        candidate_entry_ids=candidate_entry_ids,
    )
    actual = compute_indexer_scores(
        indexer_q=q,
        indexer_kv=kv,
        indexer_weights=weights,
        score_scale=scale,
        visibility_mask=mask,
    )
    expected = _slow_indexer_scores(
        q=q,
        kv=kv,
        weights=weights,
        visibility_mask=mask,
        score_scale=scale,
    ).unsqueeze(0)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert torch.isneginf(actual[0, 0, 1:]).all()
    assert torch.isneginf(actual[0, 1, 2:]).all()
    assert torch.isneginf(actual[0, 2, 3]).all()
    assert torch.isfinite(actual[0, 3, [0, 1, 3]]).all()


def test_compute_indexer_topk_returns_global_ids_with_stable_ties_and_padding() -> None:
    layout = _layout()
    q = torch.zeros(1, 2, 4)
    kv = torch.zeros(2, 4)
    weights = torch.ones(1, 2)

    result = compute_indexer_topk(
        indexer_q=q,
        indexer_kv=kv,
        indexer_weights=weights,
        candidate_entry_ids=(1, 0),
        topk=4,
        query_token_ids=(7,),
        layout=layout,
    )

    assert result.indices.shape == (1, 1, 4)
    assert result.scores.shape == (1, 1, 4)
    assert result.indices[0, 0].tolist() == [0, 1, -1, -1]
    assert result.scores[0, 0, :2].tolist() == [0.0, 0.0]
    assert torch.isneginf(result.scores[0, 0, 2:]).all()
    assert not result.indices.requires_grad
    assert not result.scores.requires_grad


def test_compute_indexer_topk_handles_no_visible_candidates() -> None:
    layout = _layout()
    q = torch.randn(1, 2, 4)
    kv = torch.randn(2, 4)
    weights = torch.randn(1, 2)

    result = compute_indexer_topk(
        indexer_q=q,
        indexer_kv=kv,
        indexer_weights=weights,
        candidate_entry_ids=(0, 1),
        topk=3,
        query_token_ids=(2,),
        layout=layout,
    )

    assert result.indices[0, 0].tolist() == [-1, -1, -1]
    assert torch.isneginf(result.scores[0, 0]).all()


def test_stable_topk_repairs_boundary_ties_by_global_id() -> None:
    scores = torch.tensor([[[1.0, 1.0, 2.0, float("-inf")]]])
    candidate_ids = torch.tensor([5, 3, 4, 1])

    top_scores, top_ids = stable_topk_by_score_and_id(
        scores=scores,
        candidate_ids=candidate_ids,
        topk=4,
    )

    assert top_scores[0, 0].tolist() == [2.0, 1.0, 1.0, float("-inf")]
    assert top_ids[0, 0].tolist() == [4, 3, 5, -1]


def test_merge_indexer_topk_results_uses_score_then_global_id() -> None:
    result = merge_indexer_topk_results(
        results=(
            Dsv4TopkResult(
                scores=torch.tensor([[[5.0, 1.0]]]),
                indices=torch.tensor([[[10, 2]]]),
            ),
            Dsv4TopkResult(
                scores=torch.tensor([[[5.0, 2.0]]]),
                indices=torch.tensor([[[3, 4]]]),
            ),
        ),
        topk=3,
    )

    assert result.scores[0, 0].tolist() == [5.0, 5.0, 2.0]
    assert result.indices[0, 0].tolist() == [3, 10, 4]


def _slow_indexer_scores(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    visibility_mask: torch.Tensor,
    score_scale: float,
) -> torch.Tensor:
    scores = torch.einsum("qhd,cd->qhc", q.float(), kv.float())
    scores = torch.relu(scores)
    scores = scores.mul(weights.float().unsqueeze(-1)).sum(dim=1)
    scores = scores * float(score_scale)
    return scores.masked_fill(~visibility_mask, float("-inf"))


def _layout() -> Dsv4CompressedLayout:
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
        spec=Dsv4CompressionSpec(kind=Dsv4CompressionKind.CSA, ratio=4),
    )
