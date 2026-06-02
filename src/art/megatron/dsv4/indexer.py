from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from .types import (
    Dsv4BranchView,
    Dsv4CompressedEntry,
    Dsv4CompressedLayout,
    Dsv4TopkResult,
)

_INVALID_INDEX = -1


class TokenRangeLike(Protocol):
    start: int
    end: int


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
