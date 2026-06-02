from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from .indexer import stage_candidate_entry_ids, visible_entry_ids_for_query
from .types import (
    Dsv4BranchView,
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4StageInputs,
    Dsv4StageKeyKind,
)

_INVALID_INDEX = -1


class TokenRangeLike(Protocol):
    start: int
    end: int


def build_dsv4_stage_inputs(
    *,
    layout: Dsv4CompressedLayout,
    stage_index: int,
    query_token_ids: Sequence[int],
    global_k_ranges: Sequence[TokenRangeLike],
    compression_kind: Dsv4CompressionKind,
    global_topk: torch.Tensor | None = None,
    window_size: int = 128,
    raw_list_size: int | None = None,
    compressed_list_size: int | None = None,
) -> Dsv4StageInputs:
    """Build DSV4 Miles-kernel stage metadata from ART CP stage ranges.

    The returned `topk_stage_local` indexes `raw_token_ids + compressed_entry_ids`.
    It is metadata/index materialization only: it does not fetch Q/KV tensors or
    launch communication.
    """
    if int(window_size) <= 0:
        raise RuntimeError(
            f"DSV4 raw SWA window size must be positive, got {window_size}"
        )
    if raw_list_size is None:
        raw_list_size = int(window_size)
    if int(raw_list_size) < 0:
        raise RuntimeError(
            f"DSV4 raw list size must be non-negative, got {raw_list_size}"
        )

    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    raw_token_ids = _stage_raw_token_ids(global_k_ranges)
    raw_local = {token_id: offset for offset, token_id in enumerate(raw_token_ids)}
    compressed_entry_ids = stage_candidate_entry_ids(
        layout=layout,
        global_k_ranges=global_k_ranges,
    )
    compressed_local = {
        entry_id: len(raw_token_ids) + offset
        for offset, entry_id in enumerate(compressed_entry_ids)
    }

    topk = _normalize_global_topk(global_topk, query_count=len(query_ids))
    if compression_kind == Dsv4CompressionKind.CSA:
        if topk is None:
            raise RuntimeError("DSV4 CSA stage remap requires global_topk")
        batch_size = int(topk.shape[0])
        default_compressed_list_size = int(topk.shape[-1])
    elif compression_kind == Dsv4CompressionKind.HCA:
        if topk is not None:
            raise RuntimeError("DSV4 HCA stage remap does not consume global_topk")
        batch_size = 1
        default_compressed_list_size = _max_visible_compressed_count(
            layout=layout,
            query_token_ids=query_ids,
            candidate_entry_ids=compressed_entry_ids,
        )
    else:
        raise RuntimeError(f"Unsupported DSV4 compression kind: {compression_kind}")

    if compressed_list_size is None:
        compressed_list_size = default_compressed_list_size
    if int(compressed_list_size) < 0:
        raise RuntimeError(
            f"DSV4 compressed list size must be non-negative, got {compressed_list_size}"
        )

    raw_by_query = tuple(
        _visible_raw_swa_token_ids(
            layout=layout,
            query_token_id=query_id,
            candidate_token_ids=raw_token_ids,
            window_size=int(window_size),
        )
        for query_id in query_ids
    )
    compressed_by_query = _compressed_ids_by_query(
        layout=layout,
        query_token_ids=query_ids,
        candidate_entry_ids=compressed_entry_ids,
        compression_kind=compression_kind,
        global_topk=topk,
    )
    local_topk = _build_stage_local_topk(
        raw_by_query=raw_by_query,
        compressed_by_query=compressed_by_query,
        raw_local=raw_local,
        compressed_local=compressed_local,
        batch_size=batch_size,
        raw_list_size=int(raw_list_size),
        compressed_list_size=int(compressed_list_size),
        device=topk.device if topk is not None else torch.device("cpu"),
    )

    return Dsv4StageInputs(
        stage_index=int(stage_index),
        query_token_ids=query_ids,
        raw_token_ids=raw_token_ids,
        compressed_entry_ids=compressed_entry_ids,
        key_kinds=(Dsv4StageKeyKind.RAW,) * len(raw_token_ids)
        + (Dsv4StageKeyKind.COMPRESSED,) * len(compressed_entry_ids),
        key_global_ids=raw_token_ids + compressed_entry_ids,
        raw_token_ids_by_query=raw_by_query,
        compressed_entry_ids_by_query=_compressed_metadata_by_query(
            compressed_by_query=compressed_by_query,
            query_count=len(query_ids),
        ),
        topk_stage_local=local_topk,
    )


def build_stage_local_topk_for_csa(
    *,
    layout: Dsv4CompressedLayout,
    stage_index: int,
    query_token_ids: Sequence[int],
    global_k_ranges: Sequence[TokenRangeLike],
    global_topk: torch.Tensor,
    window_size: int = 128,
    raw_list_size: int | None = None,
    compressed_list_size: int | None = None,
) -> Dsv4StageInputs:
    return build_dsv4_stage_inputs(
        layout=layout,
        stage_index=stage_index,
        query_token_ids=query_token_ids,
        global_k_ranges=global_k_ranges,
        compression_kind=Dsv4CompressionKind.CSA,
        global_topk=global_topk,
        window_size=window_size,
        raw_list_size=raw_list_size,
        compressed_list_size=compressed_list_size,
    )


def build_stage_local_topk_for_hca(
    *,
    layout: Dsv4CompressedLayout,
    stage_index: int,
    query_token_ids: Sequence[int],
    global_k_ranges: Sequence[TokenRangeLike],
    window_size: int = 128,
    raw_list_size: int | None = None,
    compressed_list_size: int | None = None,
) -> Dsv4StageInputs:
    return build_dsv4_stage_inputs(
        layout=layout,
        stage_index=stage_index,
        query_token_ids=query_token_ids,
        global_k_ranges=global_k_ranges,
        compression_kind=Dsv4CompressionKind.HCA,
        window_size=window_size,
        raw_list_size=raw_list_size,
        compressed_list_size=compressed_list_size,
    )


def raw_swa_token_ids_for_query(
    *,
    layout: Dsv4CompressedLayout,
    query_token_id: int,
    candidate_token_ids: Sequence[int],
    window_size: int,
) -> tuple[int, ...]:
    return _visible_raw_swa_token_ids(
        layout=layout,
        query_token_id=query_token_id,
        candidate_token_ids=tuple(int(token_id) for token_id in candidate_token_ids),
        window_size=window_size,
    )


def _stage_raw_token_ids(ranges: Sequence[TokenRangeLike]) -> tuple[int, ...]:
    seen: set[int] = set()
    token_ids: list[int] = []
    for range_ in ranges:
        for token_id in range(int(range_.start), int(range_.end)):
            if token_id not in seen:
                seen.add(token_id)
                token_ids.append(token_id)
    return tuple(token_ids)


def _visible_raw_swa_token_ids(
    *,
    layout: Dsv4CompressedLayout,
    query_token_id: int,
    candidate_token_ids: Sequence[int],
    window_size: int,
) -> tuple[int, ...]:
    view, query_pos = _query_branch_view(layout=layout, query_token_id=query_token_id)
    min_pos = max(0, int(query_pos) - int(window_size) + 1)
    positioned: list[tuple[int, int]] = []
    for token_id in candidate_token_ids:
        candidate_pos = _position_in_view(view=view, token_id=int(token_id))
        if candidate_pos is None:
            continue
        if min_pos <= int(candidate_pos) <= int(query_pos):
            positioned.append((int(candidate_pos), int(token_id)))
    positioned.sort()
    return tuple(token_id for _, token_id in positioned)


def _compressed_ids_by_query(
    *,
    layout: Dsv4CompressedLayout,
    query_token_ids: tuple[int, ...],
    candidate_entry_ids: tuple[int, ...],
    compression_kind: Dsv4CompressionKind,
    global_topk: torch.Tensor | None,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if compression_kind == Dsv4CompressionKind.HCA:
        return (
            tuple(
                visible_entry_ids_for_query(
                    layout=layout,
                    query_token_id=query_id,
                    candidate_entry_ids=candidate_entry_ids,
                )
                for query_id in query_token_ids
            ),
        )

    if global_topk is None:
        raise RuntimeError("DSV4 CSA compressed-id remap requires global_topk")
    candidate_set = set(candidate_entry_ids)
    rows_by_batch: list[tuple[tuple[int, ...], ...]] = []
    for batch in range(int(global_topk.shape[0])):
        batch_rows: list[tuple[int, ...]] = []
        for query_idx, query_id in enumerate(query_token_ids):
            visible = set(
                visible_entry_ids_for_query(
                    layout=layout,
                    query_token_id=query_id,
                    candidate_entry_ids=candidate_entry_ids,
                )
            )
            selected: list[int] = []
            seen: set[int] = set()
            for entry_id in global_topk[batch, query_idx].tolist():
                entry = int(entry_id)
                if entry < 0 or entry in seen:
                    continue
                if entry in candidate_set and entry in visible:
                    selected.append(entry)
                    seen.add(entry)
            batch_rows.append(tuple(selected))
        rows_by_batch.append(tuple(batch_rows))
    return tuple(rows_by_batch)


def _compressed_metadata_by_query(
    *,
    compressed_by_query: tuple[tuple[tuple[int, ...], ...], ...],
    query_count: int,
) -> tuple[tuple[int, ...], ...]:
    by_query: list[tuple[int, ...]] = []
    for query_idx in range(query_count):
        merged: list[int] = []
        seen: set[int] = set()
        for batch_rows in compressed_by_query:
            for entry_id in batch_rows[query_idx]:
                entry = int(entry_id)
                if entry not in seen:
                    merged.append(entry)
                    seen.add(entry)
        by_query.append(tuple(merged))
    return tuple(by_query)


def _build_stage_local_topk(
    *,
    raw_by_query: tuple[tuple[int, ...], ...],
    compressed_by_query: tuple[tuple[tuple[int, ...], ...], ...],
    raw_local: dict[int, int],
    compressed_local: dict[int, int],
    batch_size: int,
    raw_list_size: int,
    compressed_list_size: int,
    device: torch.device,
) -> torch.Tensor:
    query_count = len(raw_by_query)
    local = torch.full(
        (batch_size, query_count, raw_list_size + compressed_list_size),
        _INVALID_INDEX,
        device=device,
        dtype=torch.int64,
    )
    for query_idx, raw_ids in enumerate(raw_by_query):
        selected_raw = raw_ids[-raw_list_size:] if raw_list_size > 0 else ()
        for offset, token_id in enumerate(selected_raw):
            local[:, query_idx, offset] = int(raw_local[int(token_id)])
    for batch in range(batch_size):
        batch_rows = compressed_by_query[batch if len(compressed_by_query) > 1 else 0]
        for query_idx, compressed_ids in enumerate(batch_rows):
            base = raw_list_size
            for offset, entry_id in enumerate(compressed_ids[:compressed_list_size]):
                local[batch, query_idx, base + offset] = int(
                    compressed_local[int(entry_id)]
                )
    return local


def _normalize_global_topk(
    global_topk: torch.Tensor | None,
    *,
    query_count: int,
) -> torch.Tensor | None:
    if global_topk is None:
        return None
    if global_topk.ndim == 2:
        global_topk = global_topk.unsqueeze(0)
    if global_topk.ndim != 3:
        raise RuntimeError(
            "DSV4 global topk must have shape [Q,K] or [B,Q,K], got "
            f"{tuple(global_topk.shape)}"
        )
    if int(global_topk.shape[1]) != int(query_count):
        raise RuntimeError(
            "DSV4 global topk query dimension mismatch: "
            f"topk={int(global_topk.shape[1])}, queries={query_count}"
        )
    return global_topk.to(dtype=torch.int64)


def _max_visible_compressed_count(
    *,
    layout: Dsv4CompressedLayout,
    query_token_ids: tuple[int, ...],
    candidate_entry_ids: tuple[int, ...],
) -> int:
    if not query_token_ids:
        return 0
    return max(
        len(
            visible_entry_ids_for_query(
                layout=layout,
                query_token_id=query_id,
                candidate_entry_ids=candidate_entry_ids,
            )
        )
        for query_id in query_token_ids
    )


def _query_branch_view(
    *,
    layout: Dsv4CompressedLayout,
    query_token_id: int,
) -> tuple[Dsv4BranchView, int]:
    for view in layout.branch_views:
        for token in view.tokens:
            if int(token.packed_token_id) == int(query_token_id):
                return view, int(token.view_pos)
    raise RuntimeError(f"DSV4 query token {query_token_id} is not in any branch view")


def _position_in_view(*, view: Dsv4BranchView, token_id: int) -> int | None:
    for token in view.tokens:
        if int(token.packed_token_id) == int(token_id):
            return int(token.view_pos)
    return None
