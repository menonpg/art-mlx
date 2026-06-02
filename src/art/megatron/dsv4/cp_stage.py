from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict
import torch

from .comm import Dsv4TensorExchangeWork, launch_dsv4_tensor_exchange
from .indexer import stage_candidate_entry_ids, visible_entry_ids_for_query
from .types import (
    Dsv4BranchView,
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4MaterializedStage,
    Dsv4StageInputs,
    Dsv4StageKeyKind,
    Dsv4TensorExchangePlan,
)

_INVALID_INDEX = -1


class TokenRangeLike(Protocol):
    start: int
    end: int


class Dsv4StageKvExchangeWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stage_inputs: Dsv4StageInputs
    query: torch.Tensor
    query_token_ids: tuple[int, ...]
    recv_raw_token_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_compressed_entry_ids_by_peer: tuple[tuple[int, ...], ...]
    tensor_work: Dsv4TensorExchangeWork

    def wait(self) -> None:
        self.tensor_work.wait()

    def wait_post_process(self) -> Dsv4MaterializedStage:
        result = self.tensor_work.wait_post_process()
        expected_wire_ids = _stage_wire_peer_ids_by_peer(
            raw_ids_by_peer=self.recv_raw_token_ids_by_peer,
            compressed_ids_by_peer=self.recv_compressed_entry_ids_by_peer,
        )
        _validate_stage_exchange_ids(
            actual=result.ids,
            expected=expected_wire_ids,
        )
        raw_positions, raw_ids = _positions_for_stage_kind(
            wire_ids=result.ids,
            kind=Dsv4StageKeyKind.RAW,
        )
        compressed_positions, compressed_ids = _positions_for_stage_kind(
            wire_ids=result.ids,
            kind=Dsv4StageKeyKind.COMPRESSED,
        )
        raw_kv = _index_select_stage_token_dim(result.tensor, raw_positions)
        compressed_kv = _index_select_stage_token_dim(
            result.tensor,
            compressed_positions,
        )
        return materialize_dsv4_stage_tensors(
            stage_inputs=self.stage_inputs,
            query=self.query,
            query_token_ids=self.query_token_ids,
            raw_kv=raw_kv,
            raw_token_ids=raw_ids,
            compressed_kv=compressed_kv,
            compressed_entry_ids=compressed_ids,
        )


class Dsv4StageKvExchangePeerPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    send_raw_token_ids_by_peer: tuple[tuple[int, ...], ...]
    send_compressed_entry_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_raw_token_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_compressed_entry_ids_by_peer: tuple[tuple[int, ...], ...]


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
    raw_token_ids = _stage_raw_token_ids(layout=layout, ranges=global_k_ranges)
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


def build_dsv4_stage_kv_exchange_peer_plans(
    *,
    layout: Dsv4CompressedLayout,
    stage_inputs_by_rank: Sequence[Dsv4StageInputs],
) -> tuple[Dsv4StageKvExchangePeerPlan, ...]:
    """Plan fused raw+compressed KV exchange peer ids for DSV4 stages.

    This is host metadata work: it must not inspect live activation tensors or
    synchronize CUDA. Each rank receives the raw/compressed ids needed by its
    stage, grouped by owner rank; send lists are the transpose of those receive
    requests.
    """
    rank_count = len(layout.entry_ids_by_owner_rank)
    if len(stage_inputs_by_rank) != rank_count:
        raise RuntimeError(
            "DSV4 stage exchange planning requires one stage input per rank, got "
            f"{len(stage_inputs_by_rank)} vs {rank_count}"
        )
    recv_raw = tuple(
        _ids_by_owner_rank(
            ids=stage.raw_token_ids,
            rank_count=rank_count,
            owner_rank=lambda token_id: _raw_token_owner_rank(
                layout=layout,
                token_id=token_id,
            ),
            name=f"rank{rank}_raw_token_ids",
        )
        for rank, stage in enumerate(stage_inputs_by_rank)
    )
    recv_compressed = tuple(
        _ids_by_owner_rank(
            ids=stage.compressed_entry_ids,
            rank_count=rank_count,
            owner_rank=lambda entry_id: _compressed_entry_owner_rank(
                layout=layout,
                entry_id=entry_id,
            ),
            name=f"rank{rank}_compressed_entry_ids",
        )
        for rank, stage in enumerate(stage_inputs_by_rank)
    )
    send_raw = _transpose_peer_ids(recv_raw)
    send_compressed = _transpose_peer_ids(recv_compressed)
    return tuple(
        Dsv4StageKvExchangePeerPlan(
            send_raw_token_ids_by_peer=send_raw[rank],
            send_compressed_entry_ids_by_peer=send_compressed[rank],
            recv_raw_token_ids_by_peer=recv_raw[rank],
            recv_compressed_entry_ids_by_peer=recv_compressed[rank],
        )
        for rank in range(rank_count)
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


def materialize_dsv4_stage_tensors(
    *,
    stage_inputs: Dsv4StageInputs,
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    compressed_kv: torch.Tensor,
    compressed_entry_ids: Sequence[int],
) -> Dsv4MaterializedStage:
    """Gather already-available DSV4 stage tensors into sparse-kernel order.

    This function does no communication and makes no physical-id shortcut. The
    caller supplies explicit row-id maps for local/fetched tensors. The returned
    KV rows are exactly `stage.raw_token_ids + stage.compressed_entry_ids`, which
    is the id space used by `stage.topk_stage_local`.
    """
    q_stage = _gather_mapped_rows(
        tensor=query,
        tensor_ids=query_token_ids,
        selected_ids=stage_inputs.query_token_ids,
        name="query",
    )
    raw_stage = _gather_mapped_rows(
        tensor=raw_kv,
        tensor_ids=raw_token_ids,
        selected_ids=stage_inputs.raw_token_ids,
        name="raw_kv",
    )
    compressed_stage = _gather_mapped_rows(
        tensor=compressed_kv,
        tensor_ids=compressed_entry_ids,
        selected_ids=stage_inputs.compressed_entry_ids,
        name="compressed_kv",
    )

    q_stage = _ensure_batched_stage_tensor(q_stage, name="query")
    raw_stage = _ensure_batched_stage_tensor(raw_stage, name="raw_kv")
    compressed_stage = _ensure_batched_stage_tensor(
        compressed_stage,
        name="compressed_kv",
    )
    if q_stage.ndim != 4:
        raise RuntimeError(
            f"DSV4 materialized query must be [B,Q,H,D], got {tuple(q_stage.shape)}"
        )
    if raw_stage.ndim != 3 or compressed_stage.ndim != 3:
        raise RuntimeError(
            "DSV4 materialized KV tensors must be [B,K,D], got "
            f"raw={tuple(raw_stage.shape)}, compressed={tuple(compressed_stage.shape)}"
        )
    if q_stage.shape[-1] != raw_stage.shape[-1]:
        raise RuntimeError(
            "DSV4 query and raw KV dims must match, got "
            f"{int(q_stage.shape[-1])} vs {int(raw_stage.shape[-1])}"
        )
    if raw_stage.shape[-1] != compressed_stage.shape[-1]:
        raise RuntimeError(
            "DSV4 raw and compressed KV dims must match, got "
            f"{int(raw_stage.shape[-1])} vs {int(compressed_stage.shape[-1])}"
        )

    topk = stage_inputs.topk_stage_local.to(device=q_stage.device, dtype=torch.int64)
    batch_size = max(
        int(q_stage.shape[0]),
        int(raw_stage.shape[0]),
        int(compressed_stage.shape[0]),
        int(topk.shape[0]),
    )
    q_stage = _expand_stage_batch(q_stage, batch_size=batch_size, name="query")
    raw_stage = _expand_stage_batch(raw_stage, batch_size=batch_size, name="raw_kv")
    compressed_stage = _expand_stage_batch(
        compressed_stage,
        batch_size=batch_size,
        name="compressed_kv",
    )
    topk = _expand_stage_batch(topk, batch_size=batch_size, name="topk")
    kv_stage = torch.cat((raw_stage, compressed_stage), dim=1).contiguous()

    return Dsv4MaterializedStage(
        stage_index=stage_inputs.stage_index,
        query_token_ids=stage_inputs.query_token_ids,
        q_stage=q_stage.contiguous(),
        kv_stage=kv_stage,
        topk_stage_local=topk.contiguous(),
        raw_count=len(stage_inputs.raw_token_ids),
        compressed_count=len(stage_inputs.compressed_entry_ids),
        key_kinds=stage_inputs.key_kinds,
        key_global_ids=stage_inputs.key_global_ids,
    )


@torch.compiler.disable
def launch_dsv4_stage_kv_exchange(
    *,
    stage_inputs: Dsv4StageInputs,
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    compressed_kv: torch.Tensor,
    compressed_entry_ids: Sequence[int],
    send_raw_token_ids_by_peer: Sequence[Sequence[int]],
    send_compressed_entry_ids_by_peer: Sequence[Sequence[int]],
    recv_raw_token_ids_by_peer: Sequence[Sequence[int]],
    recv_compressed_entry_ids_by_peer: Sequence[Sequence[int]],
    group: Any,
    async_op: bool,
) -> Dsv4StageKvExchangeWork:
    """Launch one fused raw+compressed KV exchange for a DSV4 attention stage.

    Raw and compressed rows share the main attention KV dim, so the production
    stage path exchanges one fused tensor per stage instead of separate raw and
    compressed collectives. Wire ids encode key kind to keep raw token ids and
    compressed entry ids in disjoint spaces.
    """
    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    _validate_stage_kv_pair(
        raw_kv=raw_kv,
        raw_token_ids=raw_token_ids,
        compressed_kv=compressed_kv,
        compressed_entry_ids=compressed_entry_ids,
    )
    rank_count = _peer_count(
        send_raw_token_ids_by_peer,
        send_compressed_entry_ids_by_peer,
        recv_raw_token_ids_by_peer,
        recv_compressed_entry_ids_by_peer,
    )
    send_raw = _normalize_stage_peer_ids(
        send_raw_token_ids_by_peer,
        rank_count=rank_count,
        name="send_raw_token_ids_by_peer",
    )
    send_compressed = _normalize_stage_peer_ids(
        send_compressed_entry_ids_by_peer,
        rank_count=rank_count,
        name="send_compressed_entry_ids_by_peer",
    )
    recv_raw = _normalize_stage_peer_ids(
        recv_raw_token_ids_by_peer,
        rank_count=rank_count,
        name="recv_raw_token_ids_by_peer",
    )
    recv_compressed = _normalize_stage_peer_ids(
        recv_compressed_entry_ids_by_peer,
        rank_count=rank_count,
        name="recv_compressed_entry_ids_by_peer",
    )
    fused_kv, fused_ids = _fuse_local_stage_kv(
        raw_kv=raw_kv,
        raw_token_ids=tuple(int(token_id) for token_id in raw_token_ids),
        compressed_kv=compressed_kv,
        compressed_entry_ids=tuple(int(entry_id) for entry_id in compressed_entry_ids),
    )
    return Dsv4StageKvExchangeWork(
        stage_inputs=stage_inputs,
        query=query,
        query_token_ids=query_ids,
        recv_raw_token_ids_by_peer=recv_raw,
        recv_compressed_entry_ids_by_peer=recv_compressed,
        tensor_work=launch_dsv4_tensor_exchange(
            tensor=fused_kv,
            tensor_ids=fused_ids,
            plan=Dsv4TensorExchangePlan(
                send_ids_by_peer=_stage_wire_peer_ids_by_peer(
                    raw_ids_by_peer=send_raw,
                    compressed_ids_by_peer=send_compressed,
                ),
                recv_ids_by_peer=_stage_wire_peer_ids_by_peer(
                    raw_ids_by_peer=recv_raw,
                    compressed_ids_by_peer=recv_compressed,
                ),
            ),
            group=group,
            async_op=async_op,
            label="dsv4_stage_kv_exchange",
        ),
    )


@torch.compiler.disable
def launch_planned_dsv4_stage_kv_exchange(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    stage_inputs_by_rank: Sequence[Dsv4StageInputs],
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    compressed_kv: torch.Tensor,
    compressed_entry_ids: Sequence[int],
    group: Any,
    async_op: bool,
) -> Dsv4StageKvExchangeWork:
    """Launch one rank's fused stage KV exchange from DSV4 host metadata.

    This wrapper turns all-rank stage metadata into this rank's peer send/recv
    lists, then delegates to the eager raw+compressed KV exchange path. It keeps
    DSV4-specific planning outside the generic Flex CP executor.
    """
    rank_int = _validate_rank(rank=rank, rank_count=len(layout.entry_ids_by_owner_rank))
    peer_plans = build_dsv4_stage_kv_exchange_peer_plans(
        layout=layout,
        stage_inputs_by_rank=stage_inputs_by_rank,
    )
    plan = peer_plans[rank_int]
    return launch_dsv4_stage_kv_exchange(
        stage_inputs=stage_inputs_by_rank[rank_int],
        query=query,
        query_token_ids=query_token_ids,
        raw_kv=raw_kv,
        raw_token_ids=raw_token_ids,
        compressed_kv=compressed_kv,
        compressed_entry_ids=compressed_entry_ids,
        send_raw_token_ids_by_peer=plan.send_raw_token_ids_by_peer,
        send_compressed_entry_ids_by_peer=plan.send_compressed_entry_ids_by_peer,
        recv_raw_token_ids_by_peer=plan.recv_raw_token_ids_by_peer,
        recv_compressed_entry_ids_by_peer=plan.recv_compressed_entry_ids_by_peer,
        group=group,
        async_op=async_op,
    )


def _stage_raw_token_ids(
    *,
    layout: Dsv4CompressedLayout,
    ranges: Sequence[TokenRangeLike],
) -> tuple[int, ...]:
    seen: set[int] = set()
    token_ids: list[int] = []
    for range_ in ranges:
        for token_id in range(int(range_.start), int(range_.end)):
            if token_id not in seen and _token_in_layout(
                layout=layout, token_id=token_id
            ):
                seen.add(token_id)
                token_ids.append(token_id)
    return tuple(token_ids)


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


def _validate_stage_kv_pair(
    *,
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    compressed_kv: torch.Tensor,
    compressed_entry_ids: Sequence[int],
) -> None:
    if raw_kv.ndim not in (2, 3) or compressed_kv.ndim not in (2, 3):
        raise RuntimeError(
            "DSV4 stage KV tensors must be [N,D] or [B,N,D], got "
            f"raw={tuple(raw_kv.shape)}, compressed={tuple(compressed_kv.shape)}"
        )
    if raw_kv.ndim != compressed_kv.ndim:
        raise RuntimeError(
            "DSV4 raw and compressed stage KV ranks must match, got "
            f"{raw_kv.ndim} vs {compressed_kv.ndim}"
        )
    if raw_kv.device != compressed_kv.device or raw_kv.dtype != compressed_kv.dtype:
        raise RuntimeError("DSV4 raw and compressed stage KV device/dtype must match")
    if int(raw_kv.shape[-1]) != int(compressed_kv.shape[-1]):
        raise RuntimeError("DSV4 raw and compressed stage KV dims must match")
    if raw_kv.ndim == 3 and int(raw_kv.shape[0]) != int(compressed_kv.shape[0]):
        raise RuntimeError("DSV4 raw and compressed stage KV batch dims must match")
    if len(raw_token_ids) != int(raw_kv.shape[_stage_token_dim(raw_kv)]):
        raise RuntimeError("DSV4 raw_token_ids length must match raw_kv rows")
    if len(compressed_entry_ids) != int(
        compressed_kv.shape[_stage_token_dim(compressed_kv)]
    ):
        raise RuntimeError(
            "DSV4 compressed_entry_ids length must match compressed_kv rows"
        )
    _row_by_id(tensor_ids=raw_token_ids, name="raw_token_ids")
    _row_by_id(tensor_ids=compressed_entry_ids, name="compressed_entry_ids")


def _peer_count(*peers: Sequence[Sequence[int]]) -> int:
    counts = {len(peer_ids) for peer_ids in peers}
    if len(counts) != 1:
        raise RuntimeError("DSV4 stage exchange peer-list counts must match")
    return counts.pop()


def _normalize_stage_peer_ids(
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
        peer_ids = tuple(int(id_) for id_ in ids)
        if any(id_ < 0 for id_ in peer_ids):
            raise RuntimeError(f"DSV4 {name}[{peer}] ids must be non-negative")
        _row_by_id(tensor_ids=peer_ids, name=f"{name}[{peer}]")
        normalized.append(peer_ids)
    return tuple(normalized)


def _fuse_local_stage_kv(
    *,
    raw_kv: torch.Tensor,
    raw_token_ids: tuple[int, ...],
    compressed_kv: torch.Tensor,
    compressed_entry_ids: tuple[int, ...],
) -> tuple[torch.Tensor, tuple[int, ...]]:
    fused = torch.cat((raw_kv, compressed_kv), dim=_stage_token_dim(raw_kv))
    return fused, tuple(
        _stage_wire_id(Dsv4StageKeyKind.RAW, token_id) for token_id in raw_token_ids
    ) + tuple(
        _stage_wire_id(Dsv4StageKeyKind.COMPRESSED, entry_id)
        for entry_id in compressed_entry_ids
    )


def _stage_wire_peer_ids_by_peer(
    *,
    raw_ids_by_peer: tuple[tuple[int, ...], ...],
    compressed_ids_by_peer: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    if len(raw_ids_by_peer) != len(compressed_ids_by_peer):
        raise RuntimeError("DSV4 stage wire peer counts must match")
    return tuple(
        tuple(_stage_wire_id(Dsv4StageKeyKind.RAW, id_) for id_ in raw_ids)
        + tuple(
            _stage_wire_id(Dsv4StageKeyKind.COMPRESSED, id_) for id_ in compressed_ids
        )
        for raw_ids, compressed_ids in zip(raw_ids_by_peer, compressed_ids_by_peer)
    )


def _stage_wire_id(kind: Dsv4StageKeyKind, id_: int) -> int:
    id_int = int(id_)
    if id_int < 0:
        raise RuntimeError("DSV4 stage key ids must be non-negative")
    return id_int * 2 + (1 if kind is Dsv4StageKeyKind.COMPRESSED else 0)


def _stage_wire_kind_and_id(wire_id: int) -> tuple[Dsv4StageKeyKind, int]:
    wire_int = int(wire_id)
    if wire_int < 0:
        raise RuntimeError("DSV4 stage wire ids must be non-negative")
    kind = Dsv4StageKeyKind.COMPRESSED if wire_int % 2 else Dsv4StageKeyKind.RAW
    return kind, wire_int // 2


def _validate_stage_exchange_ids(
    *,
    actual: tuple[int, ...],
    expected: tuple[tuple[int, ...], ...],
) -> None:
    flat_expected = tuple(id_ for peer_ids in expected for id_ in peer_ids)
    if actual != flat_expected:
        raise RuntimeError(
            f"DSV4 stage exchange received ids {actual} but expected {flat_expected}"
        )


def _positions_for_stage_kind(
    *,
    wire_ids: Sequence[int],
    kind: Dsv4StageKeyKind,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    positions: list[int] = []
    ids: list[int] = []
    for position, wire_id in enumerate(wire_ids):
        row_kind, row_id = _stage_wire_kind_and_id(int(wire_id))
        if row_kind is kind:
            positions.append(position)
            ids.append(row_id)
    return tuple(positions), tuple(ids)


def _index_select_stage_token_dim(
    tensor: torch.Tensor,
    positions: Sequence[int],
) -> torch.Tensor:
    indices = torch.tensor(
        tuple(int(position) for position in positions),
        device=tensor.device,
        dtype=torch.long,
    )
    return tensor.index_select(_stage_token_dim(tensor), indices)


def _stage_token_dim(tensor: torch.Tensor) -> int:
    return 0 if tensor.ndim == 2 else 1


def _token_in_layout(*, layout: Dsv4CompressedLayout, token_id: int) -> bool:
    return any(
        int(stream.start) <= int(token_id) < int(stream.end)
        for stream in layout.streams
    )


def _raw_token_owner_rank(*, layout: Dsv4CompressedLayout, token_id: int) -> int:
    token_int = int(token_id)
    if token_int < 0 or token_int >= len(layout.raw_token_owner_ranks):
        raise RuntimeError(f"DSV4 raw token {token_int} has no CP owner")
    rank = int(layout.raw_token_owner_ranks[token_int])
    if rank < 0:
        raise RuntimeError(f"DSV4 raw token {token_int} has no CP owner")
    return rank


def _compressed_entry_owner_rank(
    *,
    layout: Dsv4CompressedLayout,
    entry_id: int,
) -> int:
    entry_int = int(entry_id)
    if entry_int < 0 or entry_int >= len(layout.entries):
        raise RuntimeError(f"DSV4 compressed entry {entry_int} is outside layout")
    return int(layout.entries[entry_int].owner_rank)


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


def _gather_mapped_rows(
    *,
    tensor: torch.Tensor,
    tensor_ids: Sequence[int],
    selected_ids: Sequence[int],
    name: str,
) -> torch.Tensor:
    if name == "query" and tensor.ndim == 3:
        token_dim = 0
    elif name == "query" and tensor.ndim == 4:
        token_dim = 1
    elif name != "query" and tensor.ndim == 2:
        token_dim = 0
    elif name != "query" and tensor.ndim == 3:
        token_dim = 1
    else:
        raise RuntimeError(f"DSV4 {name} tensor has unsupported rank {tensor.ndim}")
    if len(tensor_ids) != int(tensor.shape[token_dim]):
        raise RuntimeError(
            f"DSV4 {name} id count {len(tensor_ids)} does not match token "
            f"dimension {int(tensor.shape[token_dim])}"
        )

    row_by_id = _row_by_id(tensor_ids=tensor_ids, name=name)
    missing = tuple(int(id_) for id_ in selected_ids if int(id_) not in row_by_id)
    if missing:
        raise RuntimeError(f"DSV4 {name} tensor is missing ids {missing}")
    indices = torch.tensor(
        [row_by_id[int(id_)] for id_ in selected_ids],
        device=tensor.device,
        dtype=torch.long,
    )
    return tensor.index_select(token_dim, indices)


def _row_by_id(*, tensor_ids: Sequence[int], name: str) -> dict[int, int]:
    row_by_id: dict[int, int] = {}
    for row, id_ in enumerate(tensor_ids):
        id_int = int(id_)
        if id_int in row_by_id:
            raise RuntimeError(f"DSV4 {name} ids contain duplicate id {id_int}")
        row_by_id[id_int] = row
    return row_by_id


def _ensure_batched_stage_tensor(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if name == "query":
        if tensor.ndim == 3:
            return tensor.unsqueeze(0)
        return tensor
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    return tensor


def _expand_stage_batch(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    name: str,
) -> torch.Tensor:
    if int(tensor.shape[0]) == int(batch_size):
        return tensor
    if int(tensor.shape[0]) != 1:
        raise RuntimeError(
            f"DSV4 {name} batch cannot expand from {int(tensor.shape[0])} "
            f"to {int(batch_size)}"
        )
    return tensor.expand(batch_size, *tensor.shape[1:])


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
