from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
import torch

from art.megatron.dsv4 import Dsv4BranchView, Dsv4CompressedLayout


class Dsv4OracleBranchResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    branch_stream_id: int
    token_ids: tuple[int, ...]
    out: torch.Tensor
    lse: torch.Tensor


class Dsv4OracleAttentionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    query_token_ids: tuple[int, ...]
    out: torch.Tensor
    lse: torch.Tensor
    branches: tuple[Dsv4OracleBranchResult, ...]


def dense_dsv4_packed_attention_oracle(
    *,
    layout: Dsv4CompressedLayout,
    query: torch.Tensor,
    raw_kv: torch.Tensor,
    compressed_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    query_token_ids: Sequence[int] | None = None,
    raw_token_ids: Sequence[int] | None = None,
    compressed_entry_ids: Sequence[int] | None = None,
    topk_by_query: torch.Tensor | None = None,
    window_size: int = 128,
    scale: float = 1.0,
) -> Dsv4OracleAttentionResult:
    """Dense CP1 oracle over equivalent unpacked `prefix + completion` views.

    This reference does not use ART StagePlans. It evaluates every branch view
    independently, then verifies that repeated shared-prefix tokens produce the
    same packed output across all completions.
    """
    q, raw, compressed = _normalize_inputs(
        query=query,
        raw_kv=raw_kv,
        compressed_kv=compressed_kv,
    )
    q_ids = _default_ids(query_token_ids, int(q.shape[1]), "query_token_ids")
    raw_ids = _default_ids(raw_token_ids, int(raw.shape[1]), "raw_token_ids")
    entry_ids = _default_ids(
        compressed_entry_ids,
        int(compressed.shape[1]),
        "compressed_entry_ids",
    )
    raw_map = {token_id: offset for offset, token_id in enumerate(raw_ids)}
    compressed_map = {entry_id: offset for offset, entry_id in enumerate(entry_ids)}
    q_map = {token_id: offset for offset, token_id in enumerate(q_ids)}
    topk_rows = _topk_rows_by_query(
        topk_by_query=topk_by_query,
        query_token_ids=q_ids,
    )

    out = torch.full(
        (int(q.shape[0]), len(q_ids), int(q.shape[2]), int(q.shape[3])),
        float("nan"),
        dtype=q.dtype,
        device=q.device,
    )
    lse = torch.full(
        (int(q.shape[0]), len(q_ids), int(q.shape[2])),
        float("nan"),
        dtype=torch.float32,
        device=q.device,
    )
    written: set[int] = set()
    branch_results: list[Dsv4OracleBranchResult] = []
    for branch in layout.branch_views:
        branch_result = dense_dsv4_branch_attention_reference(
            layout=layout,
            branch=branch,
            query=q,
            raw_kv=raw,
            compressed_kv=compressed,
            attn_sink=attn_sink,
            q_map=q_map,
            raw_map=raw_map,
            compressed_map=compressed_map,
            topk_rows=topk_rows,
            window_size=window_size,
            scale=scale,
        )
        branch_results.append(branch_result)
        for branch_offset, token_id in enumerate(branch_result.token_ids):
            if token_id not in q_map:
                continue
            q_offset = q_map[token_id]
            branch_out = branch_result.out[:, branch_offset]
            branch_lse = branch_result.lse[:, branch_offset]
            if q_offset in written:
                torch.testing.assert_close(out[:, q_offset], branch_out)
                torch.testing.assert_close(lse[:, q_offset], branch_lse)
            else:
                out[:, q_offset] = branch_out
                lse[:, q_offset] = branch_lse
                written.add(q_offset)
    if len(written) != len(q_ids):
        missing = tuple(
            token_id for index, token_id in enumerate(q_ids) if index not in written
        )
        raise RuntimeError(f"DSV4 oracle did not write query token ids {missing}")
    return Dsv4OracleAttentionResult(
        query_token_ids=q_ids,
        out=out,
        lse=lse,
        branches=tuple(branch_results),
    )


def dense_dsv4_branch_attention_reference(
    *,
    layout: Dsv4CompressedLayout,
    branch: Dsv4BranchView,
    query: torch.Tensor,
    raw_kv: torch.Tensor,
    compressed_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    q_map: dict[int, int],
    raw_map: dict[int, int],
    compressed_map: dict[int, int],
    topk_rows: dict[int, tuple[int, ...]] | None = None,
    window_size: int = 128,
    scale: float = 1.0,
) -> Dsv4OracleBranchResult:
    branch_token_ids = tuple(int(token.packed_token_id) for token in branch.tokens)
    branch_out: list[torch.Tensor] = []
    branch_lse: list[torch.Tensor] = []
    for token in branch.tokens:
        query_token_id = int(token.packed_token_id)
        if query_token_id not in q_map:
            continue
        raw_ids = _visible_raw_ids(
            branch=branch,
            query_view_pos=int(token.view_pos),
            raw_map=raw_map,
            window_size=window_size,
        )
        compressed_ids = _visible_compressed_ids(
            layout=layout,
            branch=branch,
            query_token_id=query_token_id,
            query_view_pos=int(token.view_pos),
            compressed_map=compressed_map,
            topk_rows=topk_rows,
        )
        q_row = query[:, q_map[query_token_id]]
        raw_rows = _select_kv_rows(raw_kv, raw_ids, raw_map)
        compressed_rows = _select_kv_rows(compressed_kv, compressed_ids, compressed_map)
        kv_rows = torch.cat((raw_rows, compressed_rows), dim=1)
        out, lse = _dense_attention_row(
            q_row=q_row,
            kv_rows=kv_rows,
            attn_sink=attn_sink,
            scale=scale,
        )
        branch_out.append(out)
        branch_lse.append(lse)
    return Dsv4OracleBranchResult(
        branch_stream_id=int(branch.branch_stream_id),
        token_ids=branch_token_ids,
        out=torch.stack(branch_out, dim=1),
        lse=torch.stack(branch_lse, dim=1),
    )


def _normalize_inputs(
    *,
    query: torch.Tensor,
    raw_kv: torch.Tensor,
    compressed_kv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query.ndim == 3:
        query = query.unsqueeze(0)
    if raw_kv.ndim == 2:
        raw_kv = raw_kv.unsqueeze(0)
    if compressed_kv.ndim == 2:
        compressed_kv = compressed_kv.unsqueeze(0)
    if query.ndim != 4:
        raise RuntimeError(
            f"query must be [T,H,D] or [B,T,H,D], got {tuple(query.shape)}"
        )
    if raw_kv.ndim != 3 or compressed_kv.ndim != 3:
        raise RuntimeError(
            "raw_kv and compressed_kv must be [T,D] or [B,T,D], got "
            f"{tuple(raw_kv.shape)} and {tuple(compressed_kv.shape)}"
        )
    if int(query.shape[-1]) != int(raw_kv.shape[-1]):
        raise RuntimeError(
            f"query/raw dim mismatch: {int(query.shape[-1])} vs {int(raw_kv.shape[-1])}"
        )
    if int(raw_kv.shape[-1]) != int(compressed_kv.shape[-1]):
        raise RuntimeError(
            "raw/compressed dim mismatch: "
            f"{int(raw_kv.shape[-1])} vs {int(compressed_kv.shape[-1])}"
        )
    batch_size = max(
        int(query.shape[0]), int(raw_kv.shape[0]), int(compressed_kv.shape[0])
    )
    return (
        _expand_batch(query, batch_size=batch_size, name="query"),
        _expand_batch(raw_kv, batch_size=batch_size, name="raw_kv"),
        _expand_batch(compressed_kv, batch_size=batch_size, name="compressed_kv"),
    )


def _default_ids(
    ids: Sequence[int] | None,
    count: int,
    name: str,
) -> tuple[int, ...]:
    if ids is None:
        return tuple(range(count))
    normalized = tuple(int(value) for value in ids)
    if len(normalized) != count:
        raise RuntimeError(f"{name} length {len(normalized)} does not match {count}")
    if len(set(normalized)) != len(normalized):
        raise RuntimeError(f"{name} contains duplicate ids: {normalized}")
    return normalized


def _topk_rows_by_query(
    *,
    topk_by_query: torch.Tensor | None,
    query_token_ids: tuple[int, ...],
) -> dict[int, tuple[int, ...]] | None:
    if topk_by_query is None:
        return None
    if topk_by_query.ndim != 2 or int(topk_by_query.shape[0]) != len(query_token_ids):
        raise RuntimeError(
            "topk_by_query must be [Q,K] in query_token_ids order, got "
            f"{tuple(topk_by_query.shape)}"
        )
    cpu_topk = topk_by_query.detach().cpu()
    return {
        token_id: tuple(
            int(value) for value in cpu_topk[row].tolist() if int(value) >= 0
        )
        for row, token_id in enumerate(query_token_ids)
    }


def _visible_raw_ids(
    *,
    branch: Dsv4BranchView,
    query_view_pos: int,
    raw_map: dict[int, int],
    window_size: int,
) -> tuple[int, ...]:
    return tuple(
        int(token.packed_token_id)
        for token in branch.tokens
        if int(token.packed_token_id) in raw_map
        and int(token.view_pos) <= int(query_view_pos)
        and int(query_view_pos) - int(token.view_pos) < int(window_size)
    )


def _visible_compressed_ids(
    *,
    layout: Dsv4CompressedLayout,
    branch: Dsv4BranchView,
    query_token_id: int,
    query_view_pos: int,
    compressed_map: dict[int, int],
    topk_rows: dict[int, tuple[int, ...]] | None,
) -> tuple[int, ...]:
    selected = None if topk_rows is None else set(topk_rows.get(query_token_id, ()))
    visible: list[int] = []
    for (
        entry_id,
        branch_stream_id,
        prefix_stream_id,
        closure_view_pos,
        shared,
    ) in _iter_entry_visibility(layout):
        if entry_id not in compressed_map:
            continue
        if selected is not None and entry_id not in selected:
            continue
        same_branch = int(branch_stream_id) == int(branch.branch_stream_id)
        shared_prefix = bool(shared) and int(prefix_stream_id) == int(
            branch.prefix_stream_id
        )
        if (same_branch or shared_prefix) and int(closure_view_pos) <= int(
            query_view_pos
        ):
            visible.append(entry_id)
    return tuple(visible)


def _iter_entry_visibility(
    layout: Dsv4CompressedLayout,
) -> tuple[tuple[int, int, int, int, bool], ...]:
    count = int(layout.entry_count())
    if (
        len(layout.entry_branch_stream_ids) < count
        or len(layout.entry_prefix_stream_ids) < count
        or len(layout.entry_closure_view_positions) < count
        or len(layout.entry_shared_prefix_flags) < count
    ):
        raise RuntimeError("DSV4 compact oracle layout is missing entry metadata")
    return tuple(
        (
            entry_id,
            int(layout.entry_branch_stream_ids[entry_id]),
            int(layout.entry_prefix_stream_ids[entry_id]),
            int(layout.entry_closure_view_positions[entry_id]),
            bool(layout.entry_shared_prefix_flags[entry_id]),
        )
        for entry_id in range(count)
    )


def _select_kv_rows(
    tensor: torch.Tensor,
    ids: tuple[int, ...],
    id_map: dict[int, int],
) -> torch.Tensor:
    if not ids:
        return tensor.new_empty((int(tensor.shape[0]), 0, int(tensor.shape[-1])))
    positions = torch.tensor(
        [id_map[token_id] for token_id in ids],
        device=tensor.device,
        dtype=torch.long,
    )
    return tensor.index_select(1, positions)


def _dense_attention_row(
    *,
    q_row: torch.Tensor,
    kv_rows: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sink = attn_sink.to(device=q_row.device, dtype=torch.float32)
    logits = torch.einsum("bhd,bkd->bhk", q_row.float(), kv_rows.float()) * float(scale)
    sink_logits = sink.view(1, -1, 1).expand(
        int(q_row.shape[0]), int(q_row.shape[1]), 1
    )
    full_logits = torch.cat((logits, sink_logits), dim=-1)
    lse = torch.logsumexp(full_logits, dim=-1)
    if int(kv_rows.shape[1]) == 0:
        return torch.zeros_like(q_row), lse
    real_weights = torch.softmax(full_logits, dim=-1)[..., :-1]
    out = torch.einsum("bhk,bkd->bhd", real_weights.to(kv_rows.dtype), kv_rows)
    return out.to(dtype=q_row.dtype), lse


def _expand_batch(tensor: torch.Tensor, *, batch_size: int, name: str) -> torch.Tensor:
    if int(tensor.shape[0]) == batch_size:
        return tensor
    if int(tensor.shape[0]) == 1:
        return tensor.expand(batch_size, *tensor.shape[1:])
    raise RuntimeError(
        f"{name} batch {int(tensor.shape[0])} cannot expand to {batch_size}"
    )


Dsv4OracleBranchResult.model_rebuild()
Dsv4OracleAttentionResult.model_rebuild()
