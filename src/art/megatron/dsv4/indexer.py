from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict
import torch

from .comm import Dsv4TensorExchangeWork, launch_dsv4_tensor_exchange
from .types import (
    Dsv4BranchView,
    Dsv4CompressedEntry,
    Dsv4CompressedLayout,
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


class Dsv4IndexerKvExchangePeerPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    send_entry_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_entry_ids_by_peer: tuple[tuple[int, ...], ...]


def stage_candidate_entry_ids(
    *,
    layout: Dsv4CompressedLayout,
    global_k_ranges: Sequence[TokenRangeLike],
) -> tuple[int, ...]:
    """Return compressed ids whose raw closure token is in a CP stage K range."""
    if not global_k_ranges:
        return ()
    return tuple(
        int(entry.entry_id)
        for entry in layout.entries
        if _token_in_ranges(int(entry.closure_token_id), global_k_ranges)
    )


def build_dsv4_indexer_kv_exchange_peer_plans(
    *,
    layout: Dsv4CompressedLayout,
    candidate_entry_ids_by_rank: Sequence[Sequence[int]],
) -> tuple[Dsv4IndexerKvExchangePeerPlan, ...]:
    """Plan indexer-compressed KV exchange peer ids for CSA stage scoring.

    This is host metadata work only. It derives each receiver's candidate
    compressed entries from layout ownership and transposes those requests into
    send lists, without touching live activation tensors or CUDA state.
    """
    rank_count = len(layout.entry_ids_by_owner_rank)
    if len(candidate_entry_ids_by_rank) != rank_count:
        raise RuntimeError(
            "DSV4 indexer exchange planning requires one candidate list per rank, "
            f"got {len(candidate_entry_ids_by_rank)} vs {rank_count}"
        )
    recv_by_rank = tuple(
        _ids_by_owner_rank(
            ids=candidate_ids,
            rank_count=rank_count,
            owner_rank=lambda entry_id: _compressed_entry_owner_rank(
                layout=layout,
                entry_id=entry_id,
            ),
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


def visible_entry_ids_for_query(
    *,
    layout: Dsv4CompressedLayout,
    query_token_id: int,
    candidate_entry_ids: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return candidate compressed ids visible to one packed query token."""
    query = _query_visibility(layout=layout, query_token_id=query_token_id)
    if candidate_entry_ids is None:
        candidate_entry_ids = range(len(layout.entries))
    return tuple(
        int(entry_id)
        for entry_id in candidate_entry_ids
        if _entry_visible_to_query(
            entry=layout.entries[int(entry_id)],
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
    """Build a `[Q, C]` mask for branch-view indexer candidate visibility."""
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
    """Compute DSV4 CSA indexer logits.

    Shapes are `[Q,H,D]` or `[B,Q,H,D]` for `indexer_q`, `[C,D]` or `[B,C,D]`
    for `indexer_kv`, and `[Q,H]` or `[B,Q,H]` for `indexer_weights`. The
    result is always `[B,Q,C]` fp32.
    """
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
    """Compute no-grad global compressed-id topk for DSV4 CSA routing."""
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
    """Compute CSA indexer topk for one CP stage's compressed candidates.

    The caller provides an explicit id space for the fetched indexer-compressed
    KV rows. This keeps the stage path independent of how communication fetched
    those rows, while still rejecting missing or duplicate compressed ids before
    scoring.
    """
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
    """Launch distributed indexer-compressed KV fetch and stage topk scoring.

    This is the DSV4 CSA indexer communication path. The indexer KV projection
    has its own dim and is intentionally separate from main-attention KV
    communication; after fetch, `wait_post_process` runs the existing frozen
    no-grad stage topk implementation over explicit compressed entry ids.
    """
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
            label="dsv4_indexer_kv_exchange",
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
) -> Dsv4IndexerKvExchangeWork:
    """Launch one rank's indexer KV exchange from DSV4 host metadata.

    The peer-id plan is CPU metadata derived from compressed-entry ownership.
    The actual exchange remains the existing eager custom-communication path,
    outside compiled regions.
    """
    rank_int = _validate_rank(rank=rank, rank_count=len(layout.entry_ids_by_owner_rank))
    plans = build_dsv4_indexer_kv_exchange_peer_plans(
        layout=layout,
        candidate_entry_ids_by_rank=candidate_entry_ids_by_rank,
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
def launch_exchanged_dsv4_indexer_topk(
    *,
    stage_works: Sequence[Any],
    query_token_ids: Sequence[int],
    topk: int,
) -> Dsv4ExchangedIndexerTopkWork:
    """Create the eager bridge from exchanged indexer stages to global topk.

    Each stage work owns custom DSV4 indexer KV communication plus stage-local
    topk scoring. ART stages may cover only a subset of this rank's query rows,
    so this wrapper keeps the wait/materialization boundary eager and merges by
    explicit query token id before selecting one global compressed-id topk.
    """
    works = tuple(stage_works)
    if not works:
        raise RuntimeError("DSV4 exchanged indexer topk requires at least one stage")
    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    if not query_ids:
        raise RuntimeError("DSV4 exchanged indexer topk requires query_token_ids")
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


def merge_indexer_topk_results(
    *,
    results: Sequence[Dsv4TopkResult],
    topk: int,
) -> Dsv4TopkResult:
    """Merge stage-local topk results by score desc and global id asc."""
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
    """Select exact stable topk without sorting every candidate in the usual case."""
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


def _ids_by_owner_rank(
    *,
    ids: Sequence[int],
    rank_count: int,
    owner_rank: Callable[[int], int],
    name: str,
) -> tuple[tuple[int, ...], ...]:
    by_rank: list[list[int]] = [[] for _ in range(rank_count)]
    seen: set[int] = set()
    for id_ in ids:
        id_int = int(id_)
        if id_int in seen:
            raise RuntimeError(f"DSV4 {name} contains duplicate id {id_int}")
        seen.add(id_int)
        rank = int(owner_rank(id_int))
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
    if entry_int < 0 or entry_int >= len(layout.entries):
        raise RuntimeError(f"DSV4 compressed entry {entry_int} is outside layout")
    return int(layout.entries[entry_int].owner_rank)


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
    for token in view.tokens:
        if int(token.packed_token_id) == int(query_token_id):
            return (
                int(view.branch_stream_id),
                int(view.prefix_stream_id),
                int(token.view_pos),
            )
    raise RuntimeError(f"DSV4 query token {query_token_id} is not in its branch view")


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
        entry = layout.entries[int(entry_id)]
        branch.append(int(entry.branch_stream_id))
        prefix.append(int(entry.prefix_stream_id))
        pos.append(int(entry.closure_view_pos))
        shared.append(bool(entry.shared_prefix_entry))
    return (
        torch.tensor(branch, device=device, dtype=torch.long),
        torch.tensor(prefix, device=device, dtype=torch.long),
        torch.tensor(pos, device=device, dtype=torch.long),
        torch.tensor(shared, device=device, dtype=torch.bool),
    )


def _entry_visible_to_query(
    *,
    entry: Dsv4CompressedEntry,
    query_branch_stream_id: int,
    query_prefix_stream_id: int,
    query_view_pos: int,
) -> bool:
    same_branch = int(entry.branch_stream_id) == int(query_branch_stream_id)
    shared_prefix = bool(entry.shared_prefix_entry) and int(
        entry.prefix_stream_id
    ) == int(query_prefix_stream_id)
    return (same_branch or shared_prefix) and int(entry.closure_view_pos) <= int(
        query_view_pos
    )


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


def _token_in_ranges(token_id: int, ranges: Sequence[TokenRangeLike]) -> bool:
    return any(
        int(range_.start) <= int(token_id) < int(range_.end) for range_ in ranges
    )


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
