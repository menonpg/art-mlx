from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict
import torch

from .comm import Dsv4TensorExchangeWork, launch_dsv4_tensor_exchange
from .types import (
    Dsv4BranchView,
    Dsv4CompressedLayout,
    Dsv4IndexerKvExchangePeerPlan,
    Dsv4IndexerStagePlan,
    Dsv4TensorExchangePlan,
    Dsv4TopkResult,
)

_INVALID_INDEX = -1


class TokenRangeLike(Protocol):
    start: int
    end: int


class Dsv4IndexerKvExchangeWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    layout: Dsv4CompressedLayout
    query_token_ids: tuple[int, ...]
    candidate_entry_ids: tuple[int, ...]
    indexer_q: torch.Tensor
    indexer_weights: torch.Tensor
    topk: int
    score_scale: float
    recv_entry_ids_by_peer: tuple[tuple[int, ...], ...]
    tensor_work: Dsv4TensorExchangeWork

    def wait(self) -> None:
        self.tensor_work.wait()

    def wait_post_process(self) -> Dsv4TopkResult:
        result = self.tensor_work.wait_post_process()
        expected_ids = tuple(
            entry_id
            for peer_ids in self.recv_entry_ids_by_peer
            for entry_id in peer_ids
        )
        if result.ids != expected_ids:
            raise RuntimeError(
                "DSV4 indexer KV exchange received ids "
                f"{result.ids} but expected {expected_ids}"
            )
        return compute_indexer_stage_topk(
            layout=self.layout,
            query_token_ids=self.query_token_ids,
            candidate_entry_ids=self.candidate_entry_ids,
            indexer_q=self.indexer_q,
            indexer_weights=self.indexer_weights,
            indexer_kv=result.tensor,
            indexer_kv_entry_ids=result.ids,
            topk=int(self.topk),
            score_scale=float(self.score_scale),
        )


class Dsv4ExchangedIndexerTopkWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stage_works: tuple[Any, ...]
    query_token_ids: tuple[int, ...]
    topk: int

    def wait(self) -> None:
        for work in self.stage_works:
            work.wait()

    def wait_post_process(self) -> Dsv4TopkResult:
        stage_results = tuple(
            (
                _query_token_ids_from_stage_work(work=work, position=position),
                _indexer_topk_result_from_work(work=work, position=position),
            )
            for position, work in enumerate(self.stage_works)
        )
        return merge_indexer_stage_topk_results(
            stage_results=stage_results,
            query_token_ids=self.query_token_ids,
            topk=int(self.topk),
        )


def stage_candidate_entry_ids(
    *,
    layout: Dsv4CompressedLayout,
    global_k_ranges: Sequence[TokenRangeLike],
) -> tuple[int, ...]:
    if not global_k_ranges:
        return ()
    if layout.entry_count() and (
        not layout.entry_ids_by_closure_token or not layout.closure_token_ids
    ):
        raise RuntimeError(
            "DSV4 compressed layout is missing closure-token entry index"
        )
    candidates: set[int] = set()
    closure_tokens = layout.closure_token_ids
    for range_ in global_k_ranges:
        start = bisect_left(closure_tokens, int(range_.start))
        end = bisect_left(closure_tokens, int(range_.end))
        for token_id in closure_tokens[start:end]:
            candidates.update(layout.entry_ids_by_closure_token.get(token_id, ()))
    return tuple(sorted(candidates))


def build_dsv4_indexer_kv_exchange_peer_plans(
    *,
    layout: Dsv4CompressedLayout,
    candidate_entry_ids_by_rank: Sequence[Sequence[int]],
) -> tuple[Dsv4IndexerKvExchangePeerPlan, ...]:
    rank_count = len(layout.entry_ids_by_owner_rank)
    if len(candidate_entry_ids_by_rank) != rank_count:
        raise RuntimeError(
            "DSV4 indexer exchange planning requires one candidate list per rank, "
            f"got {len(candidate_entry_ids_by_rank)} vs {rank_count}"
        )
    recv_by_rank = tuple(
        _ids_by_owner_rank_from_table(
            ids=candidate_ids,
            rank_count=rank_count,
            owner_ranks=_compressed_owner_rank_table(layout),
            name=f"rank{rank}_candidate_entry_ids",
        )
        for rank, candidate_ids in enumerate(candidate_entry_ids_by_rank)
    )
    send_by_rank = _transpose_peer_ids(recv_by_rank)
    return tuple(
        Dsv4IndexerKvExchangePeerPlan(
            send_entry_ids_by_peer=send_by_rank[rank],
            recv_entry_ids_by_peer=recv_by_rank[rank],
        )
        for rank in range(rank_count)
    )


def build_dsv4_indexer_stage_plan_from_stage_plans(
    *,
    layout: Dsv4CompressedLayout,
    stage_plans_by_rank: Sequence[Any],
) -> Dsv4IndexerStagePlan:
    stage_plans = tuple(stage_plans_by_rank)
    _validate_stage_plan_count(layout=layout, stage_plans_by_rank=stage_plans)
    stage_index = _shared_stage_index(stage_plans)
    query_ids_by_rank = tuple(
        _token_ids_from_ranges(stage_plan.global_q_ranges) for stage_plan in stage_plans
    )
    return Dsv4IndexerStagePlan(
        stage_index=stage_index,
        query_token_ids_by_rank=query_ids_by_rank,
        candidate_entry_ids_by_rank=tuple(
            stage_candidate_entry_ids(
                layout=layout,
                global_k_ranges=stage_plan.global_k_ranges
                if query_ids_by_rank[rank]
                else (),
            )
            for rank, stage_plan in enumerate(stage_plans)
        ),
    )


def visible_entry_ids_for_query(
    *,
    layout: Dsv4CompressedLayout,
    query_token_id: int,
    candidate_entry_ids: Sequence[int] | None = None,
) -> tuple[int, ...]:
    query = _query_visibility(layout=layout, query_token_id=query_token_id)
    if candidate_entry_ids is None:
        candidate_entry_ids = range(layout.entry_count())
    return tuple(
        int(entry_id)
        for entry_id in candidate_entry_ids
        if _entry_id_visible_to_query(
            layout=layout,
            entry_id=int(entry_id),
            query_branch_stream_id=query[0],
            query_prefix_stream_id=query[1],
            query_view_pos=query[2],
        )
    )


def build_indexer_visibility_mask(
    *,
    layout: Dsv4CompressedLayout,
    query_token_ids: Sequence[int],
    candidate_entry_ids: Sequence[int],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    query_branch, query_prefix, query_pos = _query_visibility_tensors(
        layout=layout,
        query_token_ids=query_token_ids,
        device=device,
    )
    entry_branch, entry_prefix, entry_pos, entry_shared = _entry_visibility_tensors(
        layout=layout,
        candidate_entry_ids=candidate_entry_ids,
        device=device,
    )
    same_branch = query_branch.unsqueeze(1) == entry_branch.unsqueeze(0)
    same_prefix = query_prefix.unsqueeze(1) == entry_prefix.unsqueeze(0)
    shared_prefix = entry_shared.unsqueeze(0) & same_prefix
    closed = entry_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
    return (same_branch | shared_prefix) & closed


def compute_indexer_scores(
    *,
    indexer_q: torch.Tensor,
    indexer_kv: torch.Tensor,
    indexer_weights: torch.Tensor,
    score_scale: float = 1.0,
    visibility_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    q = _ensure_batched_q(indexer_q)
    kv = _ensure_batched_kv(indexer_kv, batch_size=int(q.shape[0]))
    weights = _ensure_batched_weights(indexer_weights, batch_size=int(q.shape[0]))
    _validate_indexer_shapes(q=q, kv=kv, weights=weights)
    if kv.shape[0] == 1 and q.shape[0] != 1:
        scores = torch.einsum("bqhd,kd->bqhk", q.float(), kv[0].float())
    else:
        scores = torch.einsum("bqhd,bkd->bqhk", q.float(), kv.float())
    scores = torch.relu(scores)
    scores = scores * weights.float().unsqueeze(-1)
    scores = scores.sum(dim=2) * float(score_scale)
    if visibility_mask is not None:
        mask = visibility_mask.to(device=scores.device, dtype=torch.bool)
        if mask.ndim != 2 or tuple(mask.shape) != tuple(scores.shape[-2:]):
            raise RuntimeError(
                "DSV4 indexer visibility mask must have shape [Q, C], got "
                f"{tuple(mask.shape)} for scores {tuple(scores.shape)}"
            )
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    return scores


@torch.no_grad()
def compute_indexer_topk(
    *,
    indexer_q: torch.Tensor,
    indexer_kv: torch.Tensor,
    indexer_weights: torch.Tensor,
    candidate_entry_ids: Sequence[int],
    topk: int,
    query_token_ids: Sequence[int] | None = None,
    layout: Dsv4CompressedLayout | None = None,
    score_scale: float = 1.0,
    visibility_mask: torch.Tensor | None = None,
) -> Dsv4TopkResult:
    if visibility_mask is None and layout is not None and query_token_ids is not None:
        visibility_mask = build_indexer_visibility_mask(
            layout=layout,
            query_token_ids=query_token_ids,
            candidate_entry_ids=candidate_entry_ids,
            device=indexer_q.device,
        )
    scores = compute_indexer_scores(
        indexer_q=indexer_q,
        indexer_kv=indexer_kv,
        indexer_weights=indexer_weights,
        score_scale=score_scale,
        visibility_mask=visibility_mask,
    )
    candidate_ids = torch.tensor(
        tuple(int(entry_id) for entry_id in candidate_entry_ids),
        device=scores.device,
        dtype=torch.long,
    )
    top_scores, top_ids = stable_topk_by_score_and_id(
        scores=scores,
        candidate_ids=candidate_ids,
        topk=topk,
    )
    return Dsv4TopkResult(indices=top_ids.to(torch.int64), scores=top_scores)


@torch.no_grad()
def compute_indexer_stage_topk(
    *,
    layout: Dsv4CompressedLayout,
    query_token_ids: Sequence[int],
    candidate_entry_ids: Sequence[int],
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_kv: torch.Tensor,
    indexer_kv_entry_ids: Sequence[int],
    topk: int,
    score_scale: float = 1.0,
) -> Dsv4TopkResult:
    candidate_ids = tuple(int(entry_id) for entry_id in candidate_entry_ids)
    candidate_kv = _gather_indexer_kv_by_ids(
        tensor=indexer_kv,
        tensor_ids=indexer_kv_entry_ids,
        selected_ids=candidate_ids,
    )
    return compute_indexer_topk(
        indexer_q=indexer_q,
        indexer_kv=candidate_kv,
        indexer_weights=indexer_weights,
        candidate_entry_ids=candidate_ids,
        topk=topk,
        query_token_ids=query_token_ids,
        layout=layout,
        score_scale=score_scale,
    )


@torch.compiler.disable
def launch_dsv4_indexer_kv_exchange(
    *,
    layout: Dsv4CompressedLayout,
    query_token_ids: Sequence[int],
    candidate_entry_ids: Sequence[int],
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_kv: torch.Tensor,
    indexer_kv_entry_ids: Sequence[int],
    send_entry_ids_by_peer: Sequence[Sequence[int]],
    recv_entry_ids_by_peer: Sequence[Sequence[int]],
    topk: int,
    group: Any,
    async_op: bool,
    score_scale: float = 1.0,
) -> Dsv4IndexerKvExchangeWork:
    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    candidate_ids = tuple(int(entry_id) for entry_id in candidate_entry_ids)
    _row_by_id(candidate_ids, name="candidate_entry_ids")
    local_entry_ids = tuple(int(entry_id) for entry_id in indexer_kv_entry_ids)
    if len(local_entry_ids) != int(indexer_kv.shape[-2]):
        raise RuntimeError(
            "DSV4 indexer_kv_entry_ids length must match KV rows, got "
            f"{len(local_entry_ids)} vs {int(indexer_kv.shape[-2])}"
        )
    _row_by_id(local_entry_ids, name="indexer_kv_entry_ids")
    rank_count = _indexer_peer_count(send_entry_ids_by_peer, recv_entry_ids_by_peer)
    send_ids = _normalize_indexer_peer_ids(
        send_entry_ids_by_peer,
        rank_count=rank_count,
        name="send_entry_ids_by_peer",
    )
    recv_ids = _normalize_indexer_peer_ids(
        recv_entry_ids_by_peer,
        rank_count=rank_count,
        name="recv_entry_ids_by_peer",
    )
    return Dsv4IndexerKvExchangeWork(
        layout=layout,
        query_token_ids=query_ids,
        candidate_entry_ids=candidate_ids,
        indexer_q=indexer_q,
        indexer_weights=indexer_weights,
        topk=int(topk),
        score_scale=float(score_scale),
        recv_entry_ids_by_peer=recv_ids,
        tensor_work=launch_dsv4_tensor_exchange(
            tensor=indexer_kv,
            tensor_ids=local_entry_ids,
            plan=Dsv4TensorExchangePlan(
                send_ids_by_peer=send_ids,
                recv_ids_by_peer=recv_ids,
            ),
            group=group,
            async_op=async_op,
        ),
    )


@torch.compiler.disable
def launch_planned_dsv4_indexer_kv_exchange(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    candidate_entry_ids_by_rank: Sequence[Sequence[int]],
    query_token_ids: Sequence[int],
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_kv: torch.Tensor,
    indexer_kv_entry_ids: Sequence[int],
    topk: int,
    group: Any,
    async_op: bool,
    score_scale: float = 1.0,
    peer_plans: Sequence[Dsv4IndexerKvExchangePeerPlan] | None = None,
) -> Dsv4IndexerKvExchangeWork:
    rank_int = _validate_rank(rank=rank, rank_count=len(layout.entry_ids_by_owner_rank))
    plans = _indexer_kv_peer_plans_or_build(
        layout=layout,
        candidate_entry_ids_by_rank=candidate_entry_ids_by_rank,
        peer_plans=peer_plans,
    )
    plan = plans[rank_int]
    return launch_dsv4_indexer_kv_exchange(
        layout=layout,
        query_token_ids=query_token_ids,
        candidate_entry_ids=tuple(
            int(id_) for id_ in candidate_entry_ids_by_rank[rank_int]
        ),
        indexer_q=indexer_q,
        indexer_weights=indexer_weights,
        indexer_kv=indexer_kv,
        indexer_kv_entry_ids=indexer_kv_entry_ids,
        send_entry_ids_by_peer=plan.send_entry_ids_by_peer,
        recv_entry_ids_by_peer=plan.recv_entry_ids_by_peer,
        topk=topk,
        group=group,
        async_op=async_op,
        score_scale=score_scale,
    )


@torch.compiler.disable
def launch_dsv4_indexer_topk_from_stage_plans(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    indexer_stage_plans: Sequence[Dsv4IndexerStagePlan],
    query_token_ids: Sequence[int],
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_kv: torch.Tensor,
    indexer_kv_entry_ids: Sequence[int],
    topk: int,
    group: Any,
    async_op: bool,
    score_scale: float = 1.0,
    indexer_kv_peer_plans_by_stage: Sequence[Sequence[Dsv4IndexerKvExchangePeerPlan]]
    | None = None,
) -> Dsv4ExchangedIndexerTopkWork:
    rank_int = _validate_rank(rank=rank, rank_count=len(layout.entry_ids_by_owner_rank))
    stage_plans = tuple(indexer_stage_plans)
    if not stage_plans:
        raise RuntimeError("DSV4 indexer topk launch requires at least one stage plan")
    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    _row_by_id(query_ids, name="query_token_ids")
    prepared_peer_plans = _validate_indexer_stage_peer_plans(
        layout=layout,
        indexer_stage_plans=stage_plans,
        indexer_kv_peer_plans_by_stage=indexer_kv_peer_plans_by_stage,
    )

    stage_works = []
    for stage_position, stage_plan in enumerate(stage_plans):
        _validate_indexer_stage_plan_rank_count(
            stage_plan=stage_plan,
            rank_count=len(layout.entry_ids_by_owner_rank),
        )
        stage_query_ids = stage_plan.query_token_ids_by_rank[rank_int]
        stage_works.append(
            launch_planned_dsv4_indexer_kv_exchange(
                layout=layout,
                rank=rank_int,
                candidate_entry_ids_by_rank=stage_plan.candidate_entry_ids_by_rank,
                query_token_ids=stage_query_ids,
                indexer_q=_gather_indexer_query_rows(
                    tensor=indexer_q,
                    tensor_ids=query_ids,
                    selected_ids=stage_query_ids,
                    name="indexer_q",
                ),
                indexer_weights=_gather_indexer_query_rows(
                    tensor=indexer_weights,
                    tensor_ids=query_ids,
                    selected_ids=stage_query_ids,
                    name="indexer_weights",
                ),
                indexer_kv=indexer_kv,
                indexer_kv_entry_ids=indexer_kv_entry_ids,
                topk=topk,
                group=group,
                async_op=async_op,
                score_scale=score_scale,
                peer_plans=prepared_peer_plans[stage_position]
                if prepared_peer_plans is not None
                else None,
            )
        )
    return launch_exchanged_dsv4_indexer_topk(
        stage_works=stage_works,
        query_token_ids=query_ids,
        topk=topk,
    )


@torch.compiler.disable
def launch_exchanged_dsv4_indexer_topk(
    *,
    stage_works: Sequence[Any],
    query_token_ids: Sequence[int],
    topk: int,
) -> Dsv4ExchangedIndexerTopkWork:
    works = tuple(stage_works)
    if not works:
        raise RuntimeError("DSV4 exchanged indexer topk requires at least one stage")
    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    _row_by_id(query_ids, name="query_token_ids")
    if int(topk) < 0:
        raise RuntimeError(f"DSV4 exchanged indexer topk must be non-negative: {topk}")
    for position, work in enumerate(works):
        _validate_indexer_stage_work(work=work, position=position)
    return Dsv4ExchangedIndexerTopkWork(
        stage_works=works,
        query_token_ids=query_ids,
        topk=int(topk),
    )


def _indexer_kv_peer_plans_or_build(
    *,
    layout: Dsv4CompressedLayout,
    candidate_entry_ids_by_rank: Sequence[Sequence[int]],
    peer_plans: Sequence[Dsv4IndexerKvExchangePeerPlan] | None,
) -> tuple[Dsv4IndexerKvExchangePeerPlan, ...]:
    if peer_plans is None:
        return build_dsv4_indexer_kv_exchange_peer_plans(
            layout=layout,
            candidate_entry_ids_by_rank=candidate_entry_ids_by_rank,
        )
    return _validate_indexer_kv_peer_plan_count(layout=layout, peer_plans=peer_plans)


def _validate_indexer_stage_peer_plans(
    *,
    layout: Dsv4CompressedLayout,
    indexer_stage_plans: Sequence[Dsv4IndexerStagePlan],
    indexer_kv_peer_plans_by_stage: Sequence[Sequence[Dsv4IndexerKvExchangePeerPlan]]
    | None,
) -> tuple[tuple[Dsv4IndexerKvExchangePeerPlan, ...], ...] | None:
    if indexer_kv_peer_plans_by_stage is None:
        return None
    stage_peer_plans = tuple(
        _validate_indexer_kv_peer_plan_count(layout=layout, peer_plans=peer_plans)
        for peer_plans in indexer_kv_peer_plans_by_stage
    )
    if len(stage_peer_plans) != len(indexer_stage_plans):
        raise RuntimeError(
            "DSV4 prepared indexer KV peer plan stage count must match indexer "
            f"StagePlans: {len(stage_peer_plans)} vs {len(indexer_stage_plans)}"
        )
    return stage_peer_plans


def _validate_indexer_kv_peer_plan_count(
    *,
    layout: Dsv4CompressedLayout,
    peer_plans: Sequence[Dsv4IndexerKvExchangePeerPlan],
) -> tuple[Dsv4IndexerKvExchangePeerPlan, ...]:
    plans = tuple(peer_plans)
    rank_count = len(layout.entry_ids_by_owner_rank)
    if len(plans) != rank_count:
        raise RuntimeError(
            "DSV4 prepared indexer KV peer plan count must match rank count: "
            f"{len(plans)} vs {rank_count}"
        )
    return plans


def merge_indexer_topk_results(
    *,
    results: Sequence[Dsv4TopkResult],
    topk: int,
) -> Dsv4TopkResult:
    if not results:
        raise RuntimeError("DSV4 indexer topk merge requires at least one result")
    scores = torch.cat([result.scores for result in results], dim=-1)
    indices = torch.cat([result.indices for result in results], dim=-1).to(torch.long)
    merged_scores, merged_ids = stable_select_from_scored_ids(
        scores=scores,
        indices=indices,
        topk=topk,
    )
    return Dsv4TopkResult(indices=merged_ids.to(torch.int64), scores=merged_scores)


def merge_indexer_stage_topk_results(
    *,
    stage_results: Sequence[tuple[Sequence[int], Dsv4TopkResult]],
    query_token_ids: Sequence[int],
    topk: int,
) -> Dsv4TopkResult:
    if not stage_results:
        raise RuntimeError("DSV4 indexer stage topk merge requires at least one result")
    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    query_index = _row_by_id(query_ids, name="query_token_ids")
    score_parts: list[torch.Tensor] = []
    index_parts: list[torch.Tensor] = []
    batch_size: int | None = None
    for position, (stage_query_ids, result) in enumerate(stage_results):
        if result.scores.shape != result.indices.shape or result.scores.ndim != 3:
            raise RuntimeError(
                "DSV4 indexer stage topk result must have matching [B,Q,K] "
                f"scores/indices, got {tuple(result.scores.shape)} and "
                f"{tuple(result.indices.shape)}"
            )
        stage_ids = tuple(int(token_id) for token_id in stage_query_ids)
        _row_by_id(stage_ids, name=f"stage{position}_query_token_ids")
        if len(stage_ids) != int(result.scores.shape[1]):
            raise RuntimeError(
                "DSV4 indexer stage query id count must match result Q dim, got "
                f"{len(stage_ids)} vs {int(result.scores.shape[1])}"
            )
        missing = tuple(
            token_id for token_id in stage_ids if token_id not in query_index
        )
        if missing:
            raise RuntimeError(
                f"DSV4 indexer stage query ids missing from global ids: {missing}"
            )
        if batch_size is None:
            batch_size = int(result.scores.shape[0])
        elif batch_size != int(result.scores.shape[0]):
            raise RuntimeError("DSV4 indexer stage topk batch sizes must match")
        rows = torch.tensor(
            tuple(query_index[token_id] for token_id in stage_ids),
            device=result.scores.device,
            dtype=torch.long,
        )
        scores = result.scores.new_full(
            (int(result.scores.shape[0]), len(query_ids), int(result.scores.shape[2])),
            float("-inf"),
        )
        indices = torch.full(
            scores.shape,
            _INVALID_INDEX,
            device=result.indices.device,
            dtype=torch.long,
        )
        scores.index_copy_(1, rows, result.scores)
        indices.index_copy_(1, rows.to(result.indices.device), result.indices.long())
        score_parts.append(scores)
        index_parts.append(indices)
    merged_scores, merged_ids = stable_select_from_scored_ids(
        scores=torch.cat(score_parts, dim=-1),
        indices=torch.cat(index_parts, dim=-1),
        topk=topk,
    )
    return Dsv4TopkResult(indices=merged_ids.to(torch.int64), scores=merged_scores)


def _validate_indexer_stage_work(*, work: Any, position: int) -> None:
    if not callable(getattr(work, "wait", None)):
        raise RuntimeError(f"DSV4 indexer stage work {position} is missing wait()")
    if not callable(getattr(work, "wait_post_process", None)):
        raise RuntimeError(
            f"DSV4 indexer stage work {position} is missing wait_post_process()"
        )


def _validate_indexer_stage_plan_rank_count(
    *,
    stage_plan: Dsv4IndexerStagePlan,
    rank_count: int,
) -> None:
    if len(stage_plan.query_token_ids_by_rank) != int(rank_count):
        raise RuntimeError(
            "DSV4 indexer stage plan query rank count mismatch: "
            f"{len(stage_plan.query_token_ids_by_rank)} vs {rank_count}"
        )
    if len(stage_plan.candidate_entry_ids_by_rank) != int(rank_count):
        raise RuntimeError(
            "DSV4 indexer stage plan candidate rank count mismatch: "
            f"{len(stage_plan.candidate_entry_ids_by_rank)} vs {rank_count}"
        )


def _gather_indexer_query_rows(
    *,
    tensor: torch.Tensor,
    tensor_ids: Sequence[int],
    selected_ids: Sequence[int],
    name: str,
) -> torch.Tensor:
    token_dim = _indexer_query_token_dim(tensor=tensor, name=name)
    ids = tuple(int(token_id) for token_id in tensor_ids)
    if len(ids) != int(tensor.shape[token_dim]):
        raise RuntimeError(
            f"DSV4 {name} ids must match token rows, got "
            f"{len(ids)} vs {int(tensor.shape[token_dim])}"
        )
    row_by_id = _row_by_id(ids, name=f"{name}_token_ids")
    selected = tuple(int(token_id) for token_id in selected_ids)
    missing = tuple(token_id for token_id in selected if token_id not in row_by_id)
    if missing:
        raise RuntimeError(f"DSV4 {name} is missing query ids: {missing}")
    index = torch.tensor(
        tuple(row_by_id[token_id] for token_id in selected),
        device=tensor.device,
        dtype=torch.long,
    )
    return tensor.index_select(token_dim, index)


def _indexer_query_token_dim(*, tensor: torch.Tensor, name: str) -> int:
    if name == "indexer_q":
        if tensor.ndim == 3:
            return 0
        if tensor.ndim == 4:
            return 1
    if name == "indexer_weights":
        if tensor.ndim == 2:
            return 0
        if tensor.ndim == 3:
            return 1
    raise RuntimeError(
        f"DSV4 {name} must have token dimension in [Q,...] or [B,Q,...], got "
        f"{tuple(tensor.shape)}"
    )


def _validate_stage_plan_count(
    *,
    layout: Dsv4CompressedLayout,
    stage_plans_by_rank: Sequence[Any],
) -> None:
    if len(stage_plans_by_rank) != len(layout.entry_ids_by_owner_rank):
        raise RuntimeError(
            "DSV4 stage-plan metadata requires one ART StagePlan per rank, got "
            f"{len(stage_plans_by_rank)} vs {len(layout.entry_ids_by_owner_rank)}"
        )


def _shared_stage_index(stage_plans: Sequence[Any]) -> int:
    stage_indices = tuple(int(stage_plan.stage_index) for stage_plan in stage_plans)
    if len(set(stage_indices)) != 1:
        raise RuntimeError(
            f"DSV4 stage-plan metadata requires one shared stage index, got {stage_indices}"
        )
    return stage_indices[0]


def _token_ids_from_ranges(ranges: Sequence[TokenRangeLike]) -> tuple[int, ...]:
    token_ids: list[int] = []
    for range_ in ranges:
        token_ids.extend(range(int(range_.start), int(range_.end)))
    _row_by_id(tuple(token_ids), name="stage_plan_token_ranges")
    return tuple(token_ids)


def _query_token_ids_from_stage_work(*, work: Any, position: int) -> tuple[int, ...]:
    ids = getattr(work, "query_token_ids", None)
    if ids is None:
        raise RuntimeError(
            f"DSV4 indexer stage work {position} is missing query_token_ids"
        )
    return tuple(int(token_id) for token_id in ids)


def _indexer_topk_result_from_work(*, work: Any, position: int) -> Dsv4TopkResult:
    result = work.wait_post_process()
    if not isinstance(result, Dsv4TopkResult):
        raise TypeError(
            f"DSV4 indexer stage work {position} returned {type(result)!r}, "
            "expected Dsv4TopkResult"
        )
    return result


def stable_topk_by_score_and_id(
    *,
    scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 3:
        raise RuntimeError(f"DSV4 indexer scores must be [B,Q,C], got {scores.shape}")
    if candidate_ids.ndim != 1 or int(candidate_ids.shape[0]) != int(scores.shape[-1]):
        raise RuntimeError(
            "DSV4 candidate id tensor must be [C] matching scores, got "
            f"{tuple(candidate_ids.shape)} for scores {tuple(scores.shape)}"
        )
    if topk < 0:
        raise RuntimeError(f"DSV4 topk must be non-negative, got {topk}")
    if topk == 0:
        empty_scores = scores.new_empty((*scores.shape[:-1], 0))
        empty_ids = torch.empty(
            (*scores.shape[:-1], 0), device=scores.device, dtype=torch.long
        )
        return empty_scores, empty_ids
    candidate_count = int(scores.shape[-1])
    if candidate_count == 0:
        return _empty_topk(scores=scores, topk=topk)

    actual_topk = min(int(topk), candidate_count)
    top_values = torch.topk(scores, actual_topk, dim=-1).values
    threshold = top_values[..., -1:]
    finite = torch.isfinite(scores)

    gt_scores = torch.where(
        finite & (scores > threshold),
        scores,
        scores.new_full((), float("-inf")),
    )
    gt_values, gt_pos = torch.topk(gt_scores, actual_topk, dim=-1)

    candidate_ids = candidate_ids.to(device=scores.device, dtype=torch.long)
    id_view = _candidate_ids_view(candidate_ids, scores)
    min_int = torch.iinfo(torch.long).min
    tie_priority = torch.where(
        finite & (scores == threshold),
        -id_view,
        torch.full_like(id_view, min_int),
    )
    tie_priority, tie_pos = torch.topk(tie_priority, actual_topk, dim=-1)
    tie_scores = scores.gather(-1, tie_pos)
    tie_scores = torch.where(
        tie_priority != min_int,
        tie_scores,
        scores.new_full((), float("-inf")),
    )

    gt_ids = _ids_for_positions(candidate_ids=candidate_ids, positions=gt_pos)
    tie_ids = _ids_for_positions(candidate_ids=candidate_ids, positions=tie_pos)
    sentinel = _invalid_id_sentinel(scores.device)
    gt_ids = torch.where(torch.isfinite(gt_values), gt_ids, sentinel)
    tie_ids = torch.where(torch.isfinite(tie_scores), tie_ids, sentinel)

    selected_scores = torch.cat([gt_values, tie_scores], dim=-1)
    selected_ids = torch.cat([gt_ids, tie_ids], dim=-1)
    top_scores, top_ids = stable_select_from_scored_ids(
        scores=selected_scores,
        indices=selected_ids,
        topk=actual_topk,
    )
    if actual_topk == int(topk):
        return top_scores, top_ids

    pad = int(topk) - actual_topk
    return (
        torch.cat(
            [top_scores, scores.new_full((*scores.shape[:-1], pad), float("-inf"))],
            dim=-1,
        ),
        torch.cat(
            [
                top_ids,
                torch.full(
                    (*scores.shape[:-1], pad),
                    _INVALID_INDEX,
                    device=scores.device,
                    dtype=torch.long,
                ),
            ],
            dim=-1,
        ),
    )


def stable_select_from_scored_ids(
    *,
    scores: torch.Tensor,
    indices: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.shape != indices.shape:
        raise RuntimeError(
            "DSV4 scored id selection requires matching score/index shapes, got "
            f"{tuple(scores.shape)} vs {tuple(indices.shape)}"
        )
    if scores.ndim != 3:
        raise RuntimeError(f"DSV4 scored ids must be [B,Q,K], got {scores.shape}")
    if topk < 0:
        raise RuntimeError(f"DSV4 topk must be non-negative, got {topk}")
    if topk == 0:
        return (
            scores.new_empty((*scores.shape[:-1], 0)),
            torch.empty(
                (*scores.shape[:-1], 0), device=scores.device, dtype=torch.long
            ),
        )
    if int(scores.shape[-1]) == 0:
        return _empty_topk(scores=scores, topk=topk)

    sentinel = _invalid_id_sentinel(scores.device)
    ids = torch.where(indices.to(torch.long) >= 0, indices.to(torch.long), sentinel)
    clean_scores = torch.where(
        (ids != sentinel) & torch.isfinite(scores),
        scores,
        scores.new_full((), float("-inf")),
    )
    id_order = torch.argsort(ids, dim=-1, stable=True)
    ids = ids.gather(-1, id_order)
    clean_scores = clean_scores.gather(-1, id_order)
    score_order = torch.argsort(clean_scores, dim=-1, descending=True, stable=True)
    ids = ids.gather(-1, score_order)
    clean_scores = clean_scores.gather(-1, score_order)

    actual_topk = min(int(topk), int(scores.shape[-1]))
    ids = ids[..., :actual_topk]
    clean_scores = clean_scores[..., :actual_topk]
    invalid = (ids == sentinel) | ~torch.isfinite(clean_scores)
    ids = torch.where(
        invalid,
        torch.full_like(ids, _INVALID_INDEX),
        ids,
    )
    clean_scores = torch.where(
        invalid,
        clean_scores.new_full((), float("-inf")),
        clean_scores,
    )
    if actual_topk == int(topk):
        return clean_scores, ids

    pad = int(topk) - actual_topk
    return (
        torch.cat(
            [clean_scores, scores.new_full((*scores.shape[:-1], pad), float("-inf"))],
            dim=-1,
        ),
        torch.cat(
            [
                ids,
                torch.full(
                    (*scores.shape[:-1], pad),
                    _INVALID_INDEX,
                    device=scores.device,
                    dtype=torch.long,
                ),
            ],
            dim=-1,
        ),
    )


def _gather_indexer_kv_by_ids(
    *,
    tensor: torch.Tensor,
    tensor_ids: Sequence[int],
    selected_ids: Sequence[int],
) -> torch.Tensor:
    if tensor.ndim not in (2, 3):
        raise RuntimeError(
            "DSV4 indexer KV must have shape [C,D] or [B,C,D], got "
            f"{tuple(tensor.shape)}"
        )
    tensor_ids = tuple(int(entry_id) for entry_id in tensor_ids)
    selected_ids = tuple(int(entry_id) for entry_id in selected_ids)
    if len(tensor_ids) != int(tensor.shape[-2]):
        raise RuntimeError(
            "DSV4 indexer KV id count must match tensor rows, got "
            f"{len(tensor_ids)} vs {int(tensor.shape[-2])}"
        )
    row_by_id = _row_by_id(tensor_ids, name="indexer_kv_entry_ids")
    _row_by_id(selected_ids, name="candidate_entry_ids")
    missing = tuple(entry_id for entry_id in selected_ids if entry_id not in row_by_id)
    if missing:
        raise RuntimeError(f"DSV4 indexer KV tensor is missing ids: {missing}")
    indices = torch.tensor(
        tuple(row_by_id[entry_id] for entry_id in selected_ids),
        device=tensor.device,
        dtype=torch.long,
    )
    return tensor.index_select(0 if tensor.ndim == 2 else 1, indices)


def _indexer_peer_count(*peers: Sequence[Sequence[int]]) -> int:
    counts = {len(peer_ids) for peer_ids in peers}
    if len(counts) != 1:
        raise RuntimeError("DSV4 indexer exchange peer-list counts must match")
    return counts.pop()


def _ids_by_owner_rank_from_table(
    *,
    ids: Sequence[int],
    rank_count: int,
    owner_ranks: Sequence[int],
    name: str,
) -> tuple[tuple[int, ...], ...]:
    by_rank: list[list[int]] = [[] for _ in range(rank_count)]
    seen: set[int] = set()
    for id_ in ids:
        id_int = int(id_)
        if id_int in seen:
            raise RuntimeError(f"DSV4 {name} contains duplicate id {id_int}")
        seen.add(id_int)
        if id_int < 0 or id_int >= len(owner_ranks):
            raise RuntimeError(f"DSV4 {name} id {id_int} is outside layout owner table")
        rank = int(owner_ranks[id_int])
        if rank < 0 or rank >= int(rank_count):
            raise RuntimeError(f"DSV4 {name} id {id_int} has invalid owner rank {rank}")
        by_rank[rank].append(id_int)
    return tuple(tuple(peer_ids) for peer_ids in by_rank)


def _validate_rank(*, rank: int, rank_count: int) -> int:
    rank_int = int(rank)
    if rank_int < 0 or rank_int >= int(rank_count):
        raise RuntimeError(f"DSV4 rank {rank_int} is outside rank count {rank_count}")
    return rank_int


def _transpose_peer_ids(
    recv_by_rank: tuple[tuple[tuple[int, ...], ...], ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    rank_count = len(recv_by_rank)
    return tuple(
        tuple(recv_by_rank[peer][rank] for peer in range(rank_count))
        for rank in range(rank_count)
    )


def _compressed_entry_owner_rank(
    *,
    layout: Dsv4CompressedLayout,
    entry_id: int,
) -> int:
    entry_int = int(entry_id)
    if entry_int < 0 or entry_int >= layout.entry_count():
        raise RuntimeError(f"DSV4 compressed entry {entry_int} is outside layout")
    return int(layout.compressed_entry_owner_ranks[entry_int])


def _compressed_owner_rank_table(layout: Dsv4CompressedLayout) -> tuple[int, ...]:
    return layout.compressed_entry_owner_ranks


def _normalize_indexer_peer_ids(
    ids_by_peer: Sequence[Sequence[int]],
    *,
    rank_count: int,
    name: str,
) -> tuple[tuple[int, ...], ...]:
    if len(ids_by_peer) != int(rank_count):
        raise RuntimeError(
            f"DSV4 {name} peer count {len(ids_by_peer)} does not match {rank_count}"
        )
    normalized: list[tuple[int, ...]] = []
    for peer, ids in enumerate(ids_by_peer):
        peer_ids = tuple(int(entry_id) for entry_id in ids)
        if any(entry_id < 0 for entry_id in peer_ids):
            raise RuntimeError(f"DSV4 {name}[{peer}] ids must be non-negative")
        _row_by_id(peer_ids, name=f"{name}[{peer}]")
        normalized.append(peer_ids)
    return tuple(normalized)


def _row_by_id(ids: Sequence[int], name: str) -> dict[int, int]:
    row_by_id: dict[int, int] = {}
    for row, id_ in enumerate(ids):
        id_int = int(id_)
        if id_int in row_by_id:
            raise RuntimeError(f"DSV4 {name} contains duplicate id {id_int}")
        row_by_id[id_int] = row
    return row_by_id


def _query_visibility(
    *,
    layout: Dsv4CompressedLayout,
    query_token_id: int,
) -> tuple[int, int, int]:
    stream = _stream_for_token(layout=layout, token_id=query_token_id)
    view = _branch_view_by_stream(layout=layout, branch_stream_id=int(stream.stream_id))
    query_pos = view.position_of_token(query_token_id)
    if query_pos is None:
        raise RuntimeError(
            f"DSV4 query token {query_token_id} is not in its branch view"
        )
    return (
        int(view.branch_stream_id),
        int(view.prefix_stream_id),
        int(query_pos),
    )


def _query_visibility_tensors(
    *,
    layout: Dsv4CompressedLayout,
    query_token_ids: Sequence[int],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branch: list[int] = []
    prefix: list[int] = []
    pos: list[int] = []
    for token_id in query_token_ids:
        query = _query_visibility(layout=layout, query_token_id=int(token_id))
        branch.append(int(query[0]))
        prefix.append(int(query[1]))
        pos.append(int(query[2]))
    return (
        torch.tensor(branch, device=device, dtype=torch.long),
        torch.tensor(prefix, device=device, dtype=torch.long),
        torch.tensor(pos, device=device, dtype=torch.long),
    )


def _entry_visibility_tensors(
    *,
    layout: Dsv4CompressedLayout,
    candidate_entry_ids: Sequence[int],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    branch: list[int] = []
    prefix: list[int] = []
    pos: list[int] = []
    shared: list[bool] = []
    for entry_id in candidate_entry_ids:
        entry_int = _validate_entry_id(layout=layout, entry_id=int(entry_id))
        branch.append(int(layout.entry_branch_stream_ids[entry_int]))
        prefix.append(int(layout.entry_prefix_stream_ids[entry_int]))
        pos.append(int(layout.entry_closure_view_positions[entry_int]))
        shared.append(bool(layout.entry_shared_prefix_flags[entry_int]))
    return (
        torch.tensor(branch, device=device, dtype=torch.long),
        torch.tensor(prefix, device=device, dtype=torch.long),
        torch.tensor(pos, device=device, dtype=torch.long),
        torch.tensor(shared, device=device, dtype=torch.bool),
    )


def _entry_id_visible_to_query(
    *,
    layout: Dsv4CompressedLayout,
    entry_id: int,
    query_branch_stream_id: int,
    query_prefix_stream_id: int,
    query_view_pos: int,
) -> bool:
    entry_int = _validate_entry_id(layout=layout, entry_id=entry_id)
    same_branch = int(layout.entry_branch_stream_ids[entry_int]) == int(
        query_branch_stream_id
    )
    shared_prefix = bool(layout.entry_shared_prefix_flags[entry_int]) and int(
        layout.entry_prefix_stream_ids[entry_int]
    ) == int(query_prefix_stream_id)
    return (same_branch or shared_prefix) and int(
        layout.entry_closure_view_positions[entry_int]
    ) <= int(query_view_pos)


def _validate_entry_id(*, layout: Dsv4CompressedLayout, entry_id: int) -> int:
    entry_int = int(entry_id)
    if entry_int < 0 or entry_int >= layout.entry_count():
        raise RuntimeError(f"DSV4 compressed entry {entry_int} is outside layout")
    return entry_int


def _stream_for_token(*, layout: Dsv4CompressedLayout, token_id: int):
    for stream in layout.streams:
        if int(stream.start) <= int(token_id) < int(stream.end):
            return stream
    raise RuntimeError(f"DSV4 token {token_id} does not belong to any stream")


def _branch_view_by_stream(
    *,
    layout: Dsv4CompressedLayout,
    branch_stream_id: int,
) -> Dsv4BranchView:
    for view in layout.branch_views:
        if int(view.branch_stream_id) == int(branch_stream_id):
            return view
    raise RuntimeError(f"DSV4 missing branch view {branch_stream_id}")


def _ensure_batched_q(indexer_q: torch.Tensor) -> torch.Tensor:
    if indexer_q.ndim == 3:
        return indexer_q.unsqueeze(0)
    if indexer_q.ndim == 4:
        return indexer_q
    raise RuntimeError(
        "DSV4 indexer q must have shape [Q,H,D] or [B,Q,H,D], got "
        f"{tuple(indexer_q.shape)}"
    )


def _ensure_batched_kv(indexer_kv: torch.Tensor, batch_size: int) -> torch.Tensor:
    if indexer_kv.ndim == 2:
        return indexer_kv.unsqueeze(0)
    if indexer_kv.ndim == 3:
        if int(indexer_kv.shape[0]) not in (1, batch_size):
            raise RuntimeError(
                "DSV4 batched indexer KV batch must be 1 or match q batch, got "
                f"{int(indexer_kv.shape[0])} vs {batch_size}"
            )
        return indexer_kv
    raise RuntimeError(
        "DSV4 indexer KV must have shape [C,D] or [B,C,D], got "
        f"{tuple(indexer_kv.shape)}"
    )


def _ensure_batched_weights(
    indexer_weights: torch.Tensor, batch_size: int
) -> torch.Tensor:
    if indexer_weights.ndim == 2:
        return indexer_weights.unsqueeze(0)
    if indexer_weights.ndim == 3:
        if int(indexer_weights.shape[0]) not in (1, batch_size):
            raise RuntimeError(
                "DSV4 batched indexer weights batch must be 1 or match q batch, got "
                f"{int(indexer_weights.shape[0])} vs {batch_size}"
            )
        return indexer_weights
    raise RuntimeError(
        "DSV4 indexer weights must have shape [Q,H] or [B,Q,H], got "
        f"{tuple(indexer_weights.shape)}"
    )


def _validate_indexer_shapes(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    if q.device != kv.device or q.device != weights.device:
        raise RuntimeError(
            "DSV4 indexer q, kv, and weights must share device, got "
            f"{q.device}, {kv.device}, {weights.device}"
        )
    if int(weights.shape[-2]) != int(q.shape[-3]) or int(weights.shape[-1]) != int(
        q.shape[-2]
    ):
        raise RuntimeError(
            "DSV4 indexer weights must match q [Q,H], got "
            f"q={tuple(q.shape)}, weights={tuple(weights.shape)}"
        )
    if int(kv.shape[-1]) != int(q.shape[-1]):
        raise RuntimeError(
            "DSV4 indexer KV dim must match q dim, got "
            f"q={tuple(q.shape)}, kv={tuple(kv.shape)}"
        )
    if int(weights.shape[0]) not in (1, int(q.shape[0])):
        raise RuntimeError(
            "DSV4 indexer weights batch must match q batch, got "
            f"{int(weights.shape[0])} vs {int(q.shape[0])}"
        )
    if int(kv.shape[0]) not in (1, int(q.shape[0])):
        raise RuntimeError(
            "DSV4 indexer KV batch must be 1 or match q batch, got "
            f"{int(kv.shape[0])} vs {int(q.shape[0])}"
        )


def _candidate_ids_view(
    candidate_ids: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    return candidate_ids.view(*((1,) * (scores.ndim - 1)), -1).expand_as(scores)


def _ids_for_positions(
    *,
    candidate_ids: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    view = candidate_ids.view(*((1,) * (positions.ndim - 1)), -1).expand(
        *positions.shape[:-1],
        int(candidate_ids.shape[0]),
    )
    return view.gather(-1, positions)


def _empty_topk(
    *, scores: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        scores.new_full((*scores.shape[:-1], int(topk)), float("-inf")),
        torch.full(
            (*scores.shape[:-1], int(topk)),
            _INVALID_INDEX,
            device=scores.device,
            dtype=torch.long,
        ),
    )


def _invalid_id_sentinel(device: torch.device) -> torch.Tensor:
    return torch.tensor(torch.iinfo(torch.long).max, device=device, dtype=torch.long)
