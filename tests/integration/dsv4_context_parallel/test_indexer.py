from __future__ import annotations

from pydantic import BaseModel, ConfigDict
import pytest
import torch

from art.megatron.dsv4 import (
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    Dsv4TopkResult,
    build_dsv4_compressed_layout,
    build_dsv4_indexer_kv_exchange_peer_plans,
    build_dsv4_indexer_stage_plan_from_stage_plans,
    build_indexer_visibility_mask,
    compute_indexer_scores,
    compute_indexer_stage_topk,
    compute_indexer_topk,
    launch_exchanged_dsv4_indexer_topk,
    launch_planned_dsv4_indexer_kv_exchange,
    merge_indexer_topk_results,
    stable_topk_by_score_and_id,
    stage_candidate_entry_ids,
    visible_entry_ids_for_query,
)
import art.megatron.dsv4.indexer as indexer_module


class _LayoutIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    ownership_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    token_counts_by_rank: tuple[int, ...]


class _Range(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int
    end: int


class _StagePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage_index: int
    global_q_ranges: tuple[_Range, ...]
    global_k_ranges: tuple[_Range, ...]


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


def test_indexer_stage_topks_merge_to_global_with_explicit_id_maps() -> None:
    layout = _layout()
    torch.manual_seed(17)
    query_token_ids = (11, 16)
    q = torch.randn(len(query_token_ids), 3, 5, requires_grad=True)
    weights = torch.randn(len(query_token_ids), 3, requires_grad=True)
    kv_by_entry = {
        entry_id: torch.randn(5, requires_grad=True) for entry_id in (0, 1, 2, 3)
    }
    fetched_ids = (3, 1, 0, 2)
    fetched_kv = torch.stack([kv_by_entry[entry_id] for entry_id in fetched_ids])
    topk = 3

    stage_prefix = compute_indexer_stage_topk(
        layout=layout,
        query_token_ids=query_token_ids,
        candidate_entry_ids=(0, 1),
        indexer_q=q,
        indexer_weights=weights,
        indexer_kv=fetched_kv,
        indexer_kv_entry_ids=fetched_ids,
        topk=topk,
        score_scale=0.25,
    )
    stage_suffix = compute_indexer_stage_topk(
        layout=layout,
        query_token_ids=query_token_ids,
        candidate_entry_ids=(2, 3),
        indexer_q=q,
        indexer_weights=weights,
        indexer_kv=fetched_kv,
        indexer_kv_entry_ids=fetched_ids,
        topk=topk,
        score_scale=0.25,
    )
    merged = merge_indexer_topk_results(
        results=(stage_prefix, stage_suffix),
        topk=topk,
    )
    expected = compute_indexer_topk(
        indexer_q=q,
        indexer_kv=torch.stack([kv_by_entry[entry_id] for entry_id in (0, 1, 2, 3)]),
        indexer_weights=weights,
        candidate_entry_ids=(0, 1, 2, 3),
        topk=topk,
        query_token_ids=query_token_ids,
        layout=layout,
        score_scale=0.25,
    )

    torch.testing.assert_close(merged.indices, expected.indices)
    torch.testing.assert_close(merged.scores, expected.scores, rtol=1e-6, atol=1e-6)
    assert not merged.indices.requires_grad
    assert not merged.scores.requires_grad


def test_indexer_kv_exchange_peer_plan_uses_compressed_ownership() -> None:
    layout = _layout()
    plans = build_dsv4_indexer_kv_exchange_peer_plans(
        layout=layout,
        candidate_entry_ids_by_rank=((0, 1, 2), (1, 3)),
    )

    assert plans[0].recv_entry_ids_by_peer == ((0, 1), (2,))
    assert plans[1].recv_entry_ids_by_peer == ((1,), (3,))
    assert plans[0].send_entry_ids_by_peer == ((0, 1), (1,))
    assert plans[1].send_entry_ids_by_peer == ((2,), (3,))

    with pytest.raises(RuntimeError, match="duplicate id"):
        build_dsv4_indexer_kv_exchange_peer_plans(
            layout=layout,
            candidate_entry_ids_by_rank=((0, 0), (1,)),
        )
    with pytest.raises(RuntimeError, match="outside layout"):
        build_dsv4_indexer_kv_exchange_peer_plans(
            layout=layout,
            candidate_entry_ids_by_rank=((9,), (1,)),
        )


def test_planned_indexer_exchange_uses_prepared_peer_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout()
    candidates_by_rank = ((0, 1, 2), (1, 3))
    prepared = build_dsv4_indexer_kv_exchange_peer_plans(
        layout=layout,
        candidate_entry_ids_by_rank=candidates_by_rank,
    )
    captured: dict[str, object] = {}

    def fail_rebuild(**_kwargs: object) -> object:
        raise AssertionError("indexer peer plan should be prepared")

    def fake_launch(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        indexer_module,
        "build_dsv4_indexer_kv_exchange_peer_plans",
        fail_rebuild,
    )
    monkeypatch.setattr(indexer_module, "launch_dsv4_indexer_kv_exchange", fake_launch)

    result = launch_planned_dsv4_indexer_kv_exchange(
        layout=layout,
        rank=1,
        candidate_entry_ids_by_rank=candidates_by_rank,
        query_token_ids=(11,),
        indexer_q=torch.empty(1, 2, 4),
        indexer_weights=torch.empty(1, 2),
        indexer_kv=torch.empty(0, 4),
        indexer_kv_entry_ids=(),
        topk=2,
        group=None,
        async_op=False,
        peer_plans=prepared,
    )

    assert result is not None
    assert captured["send_entry_ids_by_peer"] == prepared[1].send_entry_ids_by_peer
    assert captured["recv_entry_ids_by_peer"] == prepared[1].recv_entry_ids_by_peer


def test_indexer_stage_plan_derives_queries_and_candidates_from_art_stage_plan() -> (
    None
):
    layout = _layout()

    plan = build_dsv4_indexer_stage_plan_from_stage_plans(
        layout=layout,
        stage_plans_by_rank=(
            _stage_plan(stage_index=3, q_ranges=((7, 8),), k_ranges=((4, 13),)),
            _stage_plan(
                stage_index=3,
                q_ranges=((11, 12), (16, 17)),
                k_ranges=((8, 18),),
            ),
        ),
    )

    assert plan.stage_index == 3
    assert plan.query_token_ids_by_rank == ((7,), (11, 16))
    assert plan.candidate_entry_ids_by_rank == ((1, 2), (2, 3))


def test_exchanged_indexer_topk_merges_stage_work_results() -> None:
    stage_works = (
        _FakeIndexerStageWork(
            (11,),
            Dsv4TopkResult(
                scores=torch.tensor([[[5.0, 1.0]]]),
                indices=torch.tensor([[[10, 2]]]),
            ),
        ),
        _FakeIndexerStageWork(
            (11,),
            Dsv4TopkResult(
                scores=torch.tensor([[[5.0, 2.0]]]),
                indices=torch.tensor([[[3, 4]]]),
            ),
        ),
    )

    work = launch_exchanged_dsv4_indexer_topk(
        stage_works=stage_works,
        query_token_ids=(11,),
        topk=3,
    )
    work.wait()
    result = work.wait_post_process()

    assert result.scores[0, 0].tolist() == [5.0, 5.0, 2.0]
    assert result.indices[0, 0].tolist() == [3, 10, 4]
    for stage_work in stage_works:
        assert stage_work.wait_count == 1
        assert stage_work.post_process_count == 1


def test_exchanged_indexer_topk_aligns_partial_query_stage_results() -> None:
    stage_works = (
        _FakeIndexerStageWork(
            (10, 11),
            Dsv4TopkResult(
                scores=torch.tensor([[[3.0, 1.0], [2.0, 2.0]]]),
                indices=torch.tensor([[[5, 2], [8, 7]]]),
            ),
        ),
        _FakeIndexerStageWork(
            (11, 12),
            Dsv4TopkResult(
                scores=torch.tensor([[[4.0, 2.0], [9.0, 1.0]]]),
                indices=torch.tensor([[[3, 6], [1, 4]]]),
            ),
        ),
    )

    result = launch_exchanged_dsv4_indexer_topk(
        stage_works=stage_works,
        query_token_ids=(10, 11, 12),
        topk=3,
    ).wait_post_process()

    assert result.scores[0, 0].tolist() == [3.0, 1.0, float("-inf")]
    assert result.indices[0, 0].tolist() == [5, 2, -1]
    assert result.scores[0, 1].tolist() == [4.0, 2.0, 2.0]
    assert result.indices[0, 1].tolist() == [3, 6, 7]
    assert result.scores[0, 2].tolist() == [9.0, 1.0, float("-inf")]
    assert result.indices[0, 2].tolist() == [1, 4, -1]


def test_exchanged_indexer_topk_rejects_bad_stage_work() -> None:
    with pytest.raises(RuntimeError, match="missing wait"):
        launch_exchanged_dsv4_indexer_topk(
            stage_works=(object(),),
            query_token_ids=(11,),
            topk=1,
        )

    work = launch_exchanged_dsv4_indexer_topk(
        stage_works=(_BadIndexerStageWork(),),
        query_token_ids=(11,),
        topk=1,
    )
    with pytest.raises(TypeError, match="expected Dsv4TopkResult"):
        work.wait_post_process()


def test_indexer_stage_topk_handles_empty_candidate_stage() -> None:
    layout = _layout()
    q = torch.randn(2, 3, 5)
    weights = torch.randn(2, 3)
    kv = torch.randn(2, 5)

    result = compute_indexer_stage_topk(
        layout=layout,
        query_token_ids=(11, 16),
        candidate_entry_ids=(),
        indexer_q=q,
        indexer_weights=weights,
        indexer_kv=kv,
        indexer_kv_entry_ids=(0, 1),
        topk=4,
    )

    assert result.indices.shape == (1, 2, 4)
    assert result.scores.shape == (1, 2, 4)
    assert result.indices.tolist() == [[[-1, -1, -1, -1], [-1, -1, -1, -1]]]
    assert torch.isneginf(result.scores).all()


def test_indexer_stage_topk_rejects_bad_entry_id_maps() -> None:
    layout = _layout()
    q = torch.randn(1, 2, 5)
    weights = torch.randn(1, 2)
    kv = torch.randn(2, 5)

    with pytest.raises(RuntimeError, match="missing ids"):
        compute_indexer_stage_topk(
            layout=layout,
            query_token_ids=(11,),
            candidate_entry_ids=(2,),
            indexer_q=q,
            indexer_weights=weights,
            indexer_kv=kv,
            indexer_kv_entry_ids=(0, 1),
            topk=1,
        )
    with pytest.raises(RuntimeError, match="duplicate id"):
        compute_indexer_stage_topk(
            layout=layout,
            query_token_ids=(11,),
            candidate_entry_ids=(1,),
            indexer_q=q,
            indexer_weights=weights,
            indexer_kv=kv,
            indexer_kv_entry_ids=(1, 1),
            topk=1,
        )


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


def _stage_plan(
    *,
    stage_index: int,
    q_ranges: tuple[tuple[int, int], ...],
    k_ranges: tuple[tuple[int, int], ...],
) -> _StagePlan:
    return _StagePlan(
        stage_index=stage_index,
        global_q_ranges=tuple(_Range(start=start, end=end) for start, end in q_ranges),
        global_k_ranges=tuple(_Range(start=start, end=end) for start, end in k_ranges),
    )


class _FakeIndexerStageWork:
    def __init__(
        self,
        query_token_ids: tuple[int, ...],
        result: Dsv4TopkResult,
    ) -> None:
        self.query_token_ids = query_token_ids
        self.result = result
        self.wait_count = 0
        self.post_process_count = 0
        self._wait_complete = False

    def wait(self) -> None:
        if self._wait_complete:
            return
        self.wait_count += 1
        self._wait_complete = True

    def wait_post_process(self) -> Dsv4TopkResult:
        self.post_process_count += 1
        self.wait()
        return self.result


class _BadIndexerStageWork:
    query_token_ids = (11,)

    def wait(self) -> None:
        pass

    def wait_post_process(self) -> object:
        return object()
