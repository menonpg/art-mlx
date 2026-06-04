from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from typing import Any, Protocol

import torch

from .comm import Dsv4TensorExchangeWork, launch_dsv4_tensor_exchange
from .indexer import stage_candidate_entry_ids
from .types import (
    Dsv4BranchView,
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4IndexerKvExchangePeerPlan,
    Dsv4MaterializedStage,
    Dsv4StageInputs,
    Dsv4StageKeyKind,
    Dsv4StageKvExchangePeerPlan,
    Dsv4StagePlanSlot,
    Dsv4TensorExchangePlan,
    Dsv4WorkModel,
)

_INVALID_INDEX = -1
_QUERY_TABLE_CACHE_LIMIT = 8
_QUERY_TABLE_CACHE: dict[
    int,
    tuple[
        Dsv4CompressedLayout,
        tuple[int, ...],
        tuple[int, ...],
        dict[int, Dsv4BranchView],
    ],
] = {}


class TokenRangeLike(Protocol):
    start: int
    end: int


class Dsv4StageKvExchangeWork(Dsv4WorkModel):
    stage_inputs: Dsv4StageInputs | None
    query: torch.Tensor
    query_token_ids: tuple[int, ...]
    recv_raw_token_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_compressed_entry_ids_by_peer: tuple[tuple[int, ...], ...]
    tensor_work: Dsv4TensorExchangeWork

    def bind_stage_inputs(
        self,
        stage_inputs: Dsv4StageInputs,
    ) -> Dsv4StageKvExchangeWork:
        if self.stage_inputs is not None:
            raise RuntimeError("DSV4 stage KV exchange already has stage inputs")
        self.stage_inputs = stage_inputs
        return self

    def wait(self) -> None:
        self.tensor_work.wait()

    def wait_post_process(self) -> Dsv4MaterializedStage:
        if self.stage_inputs is None:
            raise RuntimeError(
                "DSV4 stage KV exchange must be bound to stage inputs before "
                "materialization"
            )
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
    stage_k_ranges = tuple(global_k_ranges) if query_ids else ()
    raw_token_ids = _stage_raw_token_ids(layout=layout, ranges=stage_k_ranges)
    compressed_entry_ids = stage_candidate_entry_ids(
        layout=layout,
        global_k_ranges=stage_k_ranges,
    )

    topk = _normalize_global_topk(global_topk, query_count=len(query_ids))
    if compression_kind == Dsv4CompressionKind.CSA:
        if topk is None:
            raise RuntimeError("DSV4 CSA stage remap requires global_topk")
        default_compressed_list_size = int(topk.shape[-1])
    elif compression_kind == Dsv4CompressionKind.HCA:
        if topk is not None:
            raise RuntimeError("DSV4 HCA stage remap does not consume global_topk")
        default_compressed_list_size = None
    else:
        raise RuntimeError(f"Unsupported DSV4 compression kind: {compression_kind}")

    if compressed_list_size is None and default_compressed_list_size is not None:
        compressed_list_size = default_compressed_list_size
    if compressed_list_size is not None and int(compressed_list_size) < 0:
        raise RuntimeError(
            f"DSV4 compressed list size must be non-negative, got {compressed_list_size}"
        )

    raw_topk = _build_raw_swa_stage_local_topk_tensor(
        layout=layout,
        query_token_ids=query_ids,
        candidate_token_ids=raw_token_ids,
        window_size=int(window_size),
        raw_list_size=int(raw_list_size),
    )
    if compression_kind == Dsv4CompressionKind.CSA:
        if topk is None:
            raise RuntimeError("DSV4 CSA stage remap requires global_topk")
        if compressed_list_size is None:
            raise RuntimeError("DSV4 CSA stage remap requires a compressed list size")
        local_topk = _build_csa_stage_local_topk_tensor(
            layout=layout,
            query_token_ids=query_ids,
            raw_topk=raw_topk,
            raw_token_count=len(raw_token_ids),
            candidate_entry_ids=compressed_entry_ids,
            global_topk=topk,
            raw_list_size=int(raw_list_size),
            compressed_list_size=compressed_list_size,
        )
    else:
        local_topk = _build_hca_stage_local_topk_tensor(
            layout=layout,
            query_token_ids=query_ids,
            raw_topk=raw_topk,
            raw_token_count=len(raw_token_ids),
            candidate_entry_ids=compressed_entry_ids,
            raw_list_size=int(raw_list_size),
            compressed_list_size=compressed_list_size,
        )

    return Dsv4StageInputs(
        stage_index=int(stage_index),
        query_token_ids=query_ids,
        raw_token_ids=raw_token_ids,
        compressed_entry_ids=compressed_entry_ids,
        key_kinds=(Dsv4StageKeyKind.RAW,) * len(raw_token_ids)
        + (Dsv4StageKeyKind.COMPRESSED,) * len(compressed_entry_ids),
        key_global_ids=raw_token_ids + compressed_entry_ids,
        topk_stage_local=local_topk,
    )


def build_dsv4_stage_plan_slots(
    *,
    stage_plans_by_rank: Sequence[Sequence[Any]],
) -> tuple[Dsv4StagePlanSlot, ...]:
    plans_by_rank = tuple(tuple(rank_plans) for rank_plans in stage_plans_by_rank)
    if not plans_by_rank:
        raise RuntimeError("DSV4 StagePlan slot grouping requires at least one rank")

    maps_by_rank: list[dict[int, Any]] = []
    stage_order: list[int] = []
    for rank, rank_plans in enumerate(plans_by_rank):
        rank_map: dict[int, Any] = {}
        for stage_plan in rank_plans:
            stage_index = int(stage_plan.stage_index)
            if stage_index in rank_map:
                raise RuntimeError(
                    f"DSV4 rank {rank} has duplicate StagePlan {stage_index}"
                )
            rank_map[stage_index] = stage_plan
            if rank == 0:
                stage_order.append(stage_index)
        maps_by_rank.append(rank_map)

    expected = set(stage_order)
    for rank, rank_map in enumerate(maps_by_rank):
        actual = set(rank_map)
        if actual != expected:
            raise RuntimeError(
                "DSV4 StagePlan slots require identical stage ids on all ranks: "
                f"rank0={tuple(stage_order)}, rank{rank}={tuple(rank_map)}"
            )
    return tuple(
        Dsv4StagePlanSlot(
            stage_index=stage_index,
            stage_plans_by_rank=tuple(
                rank_map[stage_index] for rank_map in maps_by_rank
            ),
        )
        for stage_index in stage_order
    )


def build_dsv4_stage_inputs_from_stage_plan(
    *,
    layout: Dsv4CompressedLayout,
    stage_plan: Any,
    compression_kind: Dsv4CompressionKind,
    global_topk: torch.Tensor | None = None,
    topk_query_token_ids: Sequence[int] | None = None,
    window_size: int = 128,
    raw_list_size: int | None = None,
    compressed_list_size: int | None = None,
) -> Dsv4StageInputs:
    query_ids = _token_ids_from_ranges(stage_plan.global_q_ranges)
    stage_topk = None
    if compression_kind == Dsv4CompressionKind.CSA:
        if global_topk is None or topk_query_token_ids is None:
            raise RuntimeError("DSV4 CSA stage inputs require local topk metadata")
        stage_topk = _select_topk_for_query_ids(
            global_topk=global_topk,
            global_query_token_ids=topk_query_token_ids,
            selected_query_token_ids=query_ids,
        )
    elif global_topk is not None or topk_query_token_ids is not None:
        raise RuntimeError("DSV4 HCA stage inputs do not consume topk metadata")

    return build_dsv4_stage_inputs(
        layout=layout,
        stage_index=int(stage_plan.stage_index),
        query_token_ids=query_ids,
        global_k_ranges=stage_plan.global_k_ranges,
        compression_kind=compression_kind,
        global_topk=stage_topk,
        window_size=window_size,
        raw_list_size=raw_list_size,
        compressed_list_size=compressed_list_size,
    )


def build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans(
    *,
    layout: Dsv4CompressedLayout,
    stage_plans_by_rank: Sequence[Any],
) -> tuple[Dsv4StageKvExchangePeerPlan, ...]:
    return build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans_for_layouts(
        layouts=(layout,),
        stage_plans_by_rank=stage_plans_by_rank,
    )[0]


def build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans_for_layouts(
    *,
    layouts: Sequence[Dsv4CompressedLayout],
    stage_plans_by_rank: Sequence[Any],
    compressed_peer_plans_by_layout: Sequence[
        Sequence[Dsv4IndexerKvExchangePeerPlan] | None
    ]
    | None = None,
) -> tuple[tuple[Dsv4StageKvExchangePeerPlan, ...], ...]:
    layout_tuple = tuple(layouts)
    if not layout_tuple:
        return ()
    compressed_peer_plan_tuple = (
        (None,) * len(layout_tuple)
        if compressed_peer_plans_by_layout is None
        else tuple(compressed_peer_plans_by_layout)
    )
    if len(compressed_peer_plan_tuple) != len(layout_tuple):
        raise RuntimeError(
            "DSV4 stage KV compressed peer-plan override count must match layouts: "
            f"{len(compressed_peer_plan_tuple)} vs {len(layout_tuple)}"
        )
    stage_plans = tuple(stage_plans_by_rank)
    _validate_stage_plan_count(layout=layout_tuple[0], stage_plans_by_rank=stage_plans)
    _shared_stage_index(stage_plans)
    rank_count = len(layout_tuple[0].entry_ids_by_owner_rank)
    for layout in layout_tuple[1:]:
        _validate_stage_plan_count(layout=layout, stage_plans_by_rank=stage_plans)
        if len(layout.entry_ids_by_owner_rank) != rank_count:
            raise RuntimeError(
                "DSV4 stage KV exchange layouts must share rank count, got "
                f"{len(layout.entry_ids_by_owner_rank)} vs {rank_count}"
            )
    has_queries = tuple(
        _ranges_have_tokens(stage_plan.global_q_ranges) for stage_plan in stage_plans
    )
    recv_raw = tuple(
        _stage_raw_token_ids_by_stage_plan_sources(
            layout=layout_tuple[0],
            stage_plan=stage_plan,
            rank=rank,
            rank_count=rank_count,
        )
        if has_queries[rank]
        else tuple(tuple() for _ in range(rank_count))
        for rank, stage_plan in enumerate(stage_plans)
    )
    send_raw = _transpose_peer_ids(recv_raw)
    return tuple(
        _build_stage_kv_exchange_peer_plans_for_layout(
            layout=layout,
            stage_plans=stage_plans,
            has_queries=has_queries,
            send_raw=send_raw,
            recv_raw=recv_raw,
            rank_count=rank_count,
            compressed_peer_plans=compressed_peer_plans,
        )
        for layout, compressed_peer_plans in zip(
            layout_tuple,
            compressed_peer_plan_tuple,
            strict=True,
        )
    )


def _build_stage_kv_exchange_peer_plans_for_layout(
    *,
    layout: Dsv4CompressedLayout,
    stage_plans: tuple[Any, ...],
    has_queries: tuple[bool, ...],
    send_raw: tuple[tuple[tuple[int, ...], ...], ...],
    recv_raw: tuple[tuple[tuple[int, ...], ...], ...],
    rank_count: int,
    compressed_peer_plans: Sequence[Dsv4IndexerKvExchangePeerPlan] | None,
) -> tuple[Dsv4StageKvExchangePeerPlan, ...]:
    if compressed_peer_plans is not None:
        plans = tuple(compressed_peer_plans)
        if len(plans) != int(rank_count):
            raise RuntimeError(
                "DSV4 stage KV compressed peer-plan override rank count must "
                f"match: {len(plans)} vs {rank_count}"
            )
        return tuple(
            Dsv4StageKvExchangePeerPlan.model_construct(
                send_raw_token_ids_by_peer=send_raw[rank],
                send_compressed_entry_ids_by_peer=plans[rank].send_entry_ids_by_peer,
                recv_raw_token_ids_by_peer=recv_raw[rank],
                recv_compressed_entry_ids_by_peer=plans[rank].recv_entry_ids_by_peer,
            )
            for rank in range(rank_count)
        )
    compressed_by_rank = tuple(
        stage_candidate_entry_ids(
            layout=layout,
            global_k_ranges=stage_plan.global_k_ranges if has_queries[rank] else (),
        )
        for rank, stage_plan in enumerate(stage_plans)
    )
    recv_compressed = tuple(
        _ids_by_owner_rank_from_table(
            ids=compressed_entry_ids,
            rank_count=rank_count,
            owner_ranks=_compressed_owner_rank_table(layout),
            name=f"rank{rank}_compressed_entry_ids",
        )
        for rank, compressed_entry_ids in enumerate(compressed_by_rank)
    )
    send_compressed = _transpose_peer_ids(recv_compressed)
    return tuple(
        Dsv4StageKvExchangePeerPlan.model_construct(
            send_raw_token_ids_by_peer=send_raw[rank],
            send_compressed_entry_ids_by_peer=send_compressed[rank],
            recv_raw_token_ids_by_peer=recv_raw[rank],
            recv_compressed_entry_ids_by_peer=recv_compressed[rank],
        )
        for rank in range(rank_count)
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


def _launch_dsv4_stage_kv_exchange_impl(
    *,
    stage_inputs: Dsv4StageInputs | None,
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
        ),
    )


@torch.compiler.disable
def launch_dsv4_stage_kv_exchange_from_stage_plan_slot(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    stage_plan_slot: Dsv4StagePlanSlot,
    local_stage_inputs: Dsv4StageInputs,
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    compressed_kv: torch.Tensor,
    compressed_entry_ids: Sequence[int],
    group: Any,
    async_op: bool,
    peer_plans: Sequence[Dsv4StageKvExchangePeerPlan] | None = None,
) -> Dsv4StageKvExchangeWork:
    rank_int = _validate_rank(rank=rank, rank_count=len(layout.entry_ids_by_owner_rank))
    if int(stage_plan_slot.stage_index) != int(local_stage_inputs.stage_index):
        raise RuntimeError(
            "DSV4 stage slot and local inputs stage_index mismatch: "
            f"{stage_plan_slot.stage_index} vs {local_stage_inputs.stage_index}"
        )
    plans = _stage_kv_peer_plans_or_build_from_slot(
        layout=layout,
        stage_plan_slot=stage_plan_slot,
        peer_plans=peer_plans,
    )
    plan = plans[rank_int]
    return _launch_dsv4_stage_kv_exchange_impl(
        stage_inputs=local_stage_inputs,
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


@torch.compiler.disable
def launch_dsv4_stage_kv_exchange_deferred_from_stage_plan_slot(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    stage_plan_slot: Dsv4StagePlanSlot,
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    compressed_kv: torch.Tensor,
    compressed_entry_ids: Sequence[int],
    group: Any,
    async_op: bool,
    peer_plans: Sequence[Dsv4StageKvExchangePeerPlan] | None = None,
) -> Dsv4StageKvExchangeWork:
    rank_int = _validate_rank(rank=rank, rank_count=len(layout.entry_ids_by_owner_rank))
    plans = _stage_kv_peer_plans_or_build_from_slot(
        layout=layout,
        stage_plan_slot=stage_plan_slot,
        peer_plans=peer_plans,
    )
    plan = plans[rank_int]
    return _launch_dsv4_stage_kv_exchange_impl(
        stage_inputs=None,
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


def _stage_kv_peer_plans_or_build_from_slot(
    *,
    layout: Dsv4CompressedLayout,
    stage_plan_slot: Dsv4StagePlanSlot,
    peer_plans: Sequence[Dsv4StageKvExchangePeerPlan] | None,
) -> tuple[Dsv4StageKvExchangePeerPlan, ...]:
    if peer_plans is None:
        return build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans(
            layout=layout,
            stage_plans_by_rank=stage_plan_slot.stage_plans_by_rank,
        )
    return _validate_stage_kv_peer_plan_count(
        layout=layout,
        peer_plans=peer_plans,
    )


def _validate_stage_kv_peer_plan_count(
    *,
    layout: Dsv4CompressedLayout,
    peer_plans: Sequence[Dsv4StageKvExchangePeerPlan],
) -> tuple[Dsv4StageKvExchangePeerPlan, ...]:
    plans = tuple(peer_plans)
    rank_count = len(layout.entry_ids_by_owner_rank)
    if len(plans) != rank_count:
        raise RuntimeError(
            "DSV4 prepared stage KV peer plan count must match rank count: "
            f"{len(plans)} vs {rank_count}"
        )
    return plans


def _stage_raw_token_ids(
    *,
    layout: Dsv4CompressedLayout,
    ranges: Sequence[TokenRangeLike],
) -> tuple[int, ...]:
    seen: set[int] = set()
    token_ids: list[int] = []
    valid_ranges = _layout_stream_ranges(layout)
    for range_ in ranges:
        range_start = int(range_.start)
        range_end = int(range_.end)
        for stream_start, stream_end in valid_ranges:
            start = max(range_start, stream_start)
            end = min(range_end, stream_end)
            if start >= end:
                continue
            for token_id in range(start, end):
                if token_id not in seen:
                    seen.add(token_id)
                    token_ids.append(token_id)
    return tuple(token_ids)


def _stage_raw_token_ids_by_stage_plan_sources(
    *,
    layout: Dsv4CompressedLayout,
    stage_plan: Any,
    rank: int,
    rank_count: int,
) -> tuple[tuple[int, ...], ...]:
    by_rank: list[list[int]] = [[] for _ in range(int(rank_count))]
    ranges = tuple(stage_plan.global_k_ranges)
    if not ranges:
        return tuple(tuple() for _ in range(int(rank_count)))
    if not hasattr(stage_plan, "is_local_stage") or not hasattr(
        stage_plan,
        "kv_fetch_plan",
    ):
        return _ids_by_owner_rank_from_table(
            ids=_stage_raw_token_ids(layout=layout, ranges=ranges),
            rank_count=rank_count,
            owner_ranks=layout.raw_token_owner_ranks,
            name="stage_plan_raw_token_ids",
        )
    if bool(stage_plan.is_local_stage):
        _extend_token_ids_from_ranges(by_rank[int(rank)], ranges)
        return tuple(tuple(ids) for ids in by_rank)
    fetch_plan = stage_plan.kv_fetch_plan
    if fetch_plan is None:
        raise RuntimeError("DSV4 remote stage raw planning requires a KV fetch plan")
    range_index = 0
    range_offset = 0
    for peer_rank, split in enumerate(fetch_plan.recv_splits):
        remaining = int(split)
        while remaining > 0:
            if range_index >= len(ranges):
                raise RuntimeError("DSV4 stage raw split exceeds global K ranges")
            range_ = ranges[range_index]
            start = int(range_.start) + int(range_offset)
            available = int(range_.end) - start
            take = min(available, remaining)
            if take <= 0:
                raise RuntimeError("DSV4 stage raw split found an empty K range")
            by_rank[peer_rank].extend(range(start, start + take))
            remaining -= take
            if take == available:
                range_index += 1
                range_offset = 0
            else:
                range_offset += take
    if range_index != len(ranges) or range_offset != 0:
        raise RuntimeError("DSV4 stage raw K ranges exceed recv splits")
    return tuple(tuple(ids) for ids in by_rank)


def _extend_token_ids_from_ranges(target: list[int], ranges: Sequence[Any]) -> None:
    for range_ in ranges:
        target.extend(range(int(range_.start), int(range_.end)))


def _merge_stage_id_ranges(
    ranges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    sorted_ranges = sorted(
        (int(start), int(end)) for start, end in ranges if int(start) < int(end)
    )
    if not sorted_ranges:
        return ()
    merged: list[list[int]] = [[sorted_ranges[0][0], sorted_ranges[0][1]]]
    for start, end in sorted_ranges[1:]:
        current = merged[-1]
        if int(start) <= int(current[1]):
            current[1] = max(int(current[1]), int(end))
        else:
            merged.append([int(start), int(end)])
    return tuple((int(start), int(end)) for start, end in merged)


def _validate_stage_plan_count(
    *,
    layout: Dsv4CompressedLayout,
    stage_plans_by_rank: Sequence[Any],
) -> None:
    if len(stage_plans_by_rank) != len(layout.entry_ids_by_owner_rank):
        raise RuntimeError(
            "DSV4 StagePlan conversion requires one ART StagePlan per rank, got "
            f"{len(stage_plans_by_rank)} vs {len(layout.entry_ids_by_owner_rank)}"
        )


def _shared_stage_index(stage_plans: Sequence[Any]) -> int:
    stage_indices = tuple(int(stage_plan.stage_index) for stage_plan in stage_plans)
    if len(set(stage_indices)) != 1:
        raise RuntimeError(
            f"DSV4 StagePlan conversion requires one shared stage index, got {stage_indices}"
        )
    return stage_indices[0]


def _token_ids_from_ranges(ranges: Sequence[TokenRangeLike]) -> tuple[int, ...]:
    token_ids: list[int] = []
    for range_ in ranges:
        token_ids.extend(range(int(range_.start), int(range_.end)))
    _row_by_id(tensor_ids=tuple(token_ids), name="stage_plan_token_ranges")
    return tuple(token_ids)


def _ranges_have_tokens(ranges: Sequence[TokenRangeLike]) -> bool:
    return any(int(range_.start) < int(range_.end) for range_ in ranges)


def _select_topk_for_query_ids(
    *,
    global_topk: torch.Tensor,
    global_query_token_ids: Sequence[int],
    selected_query_token_ids: Sequence[int],
) -> torch.Tensor:
    global_ids = tuple(int(token_id) for token_id in global_query_token_ids)
    selected_ids = tuple(int(token_id) for token_id in selected_query_token_ids)
    if not selected_ids:
        return _empty_query_topk(global_topk=global_topk, query_count=len(global_ids))
    topk = _normalize_global_topk(global_topk, query_count=len(global_ids))
    if topk is None:
        raise RuntimeError("DSV4 global topk is unexpectedly absent")
    row_by_id = _row_by_id(
        tensor_ids=global_ids,
        name="global_topk_query_token_ids",
    )
    missing = tuple(token_id for token_id in selected_ids if token_id not in row_by_id)
    if missing:
        raise RuntimeError(f"DSV4 stage query ids missing from global topk: {missing}")
    index = torch.tensor(
        tuple(row_by_id[token_id] for token_id in selected_ids),
        device=topk.device,
        dtype=torch.long,
    )
    return topk.index_select(1, index)


def _empty_query_topk(*, global_topk: torch.Tensor, query_count: int) -> torch.Tensor:
    if global_topk.ndim == 2:
        if int(global_topk.shape[0]) != int(query_count):
            raise RuntimeError(
                "DSV4 global topk query ids must match Q dim, got "
                f"{query_count} vs {int(global_topk.shape[0])}"
            )
        return global_topk.unsqueeze(0)[:, :0]
    if global_topk.ndim == 3:
        if int(global_topk.shape[1]) != int(query_count):
            raise RuntimeError(
                "DSV4 global topk query ids must match Q dim, got "
                f"{query_count} vs {int(global_topk.shape[1])}"
            )
        return global_topk[:, :0]
    raise RuntimeError(
        "DSV4 global topk must have shape [Q,K] or [B,Q,K], got "
        f"{tuple(global_topk.shape)}"
    )


def _ids_by_owner_rank_from_table(
    *,
    ids: Sequence[int],
    rank_count: int,
    owner_ranks: Sequence[int],
    name: str,
) -> tuple[tuple[int, ...], ...]:
    by_rank: list[list[int]] = [[] for _ in range(rank_count)]
    owner_count = len(owner_ranks)
    previous: int | None = None
    for id_ in ids:
        id_int = int(id_)
        if previous is not None and id_int <= previous:
            return _strict_ids_by_owner_rank_from_table(
                ids=ids,
                rank_count=rank_count,
                owner_ranks=owner_ranks,
                name=name,
            )
        previous = id_int
        if id_int < 0 or id_int >= owner_count:
            raise RuntimeError(f"DSV4 {name} id {id_int} is outside layout owner table")
        rank = int(owner_ranks[id_int])
        if rank < 0 or rank >= int(rank_count):
            raise RuntimeError(f"DSV4 {name} id {id_int} has invalid owner rank {rank}")
        by_rank[rank].append(id_int)
    return tuple(tuple(peer_ids) for peer_ids in by_rank)


def _strict_ids_by_owner_rank_from_table(
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
    dim = _stage_token_dim(tensor)
    if not positions:
        return tensor.narrow(dim, 0, 0)
    start = int(positions[0])
    if all(
        int(position) == start + offset for offset, position in enumerate(positions)
    ):
        return tensor.narrow(dim, start, len(positions))
    indices = torch.tensor(
        tuple(int(position) for position in positions),
        device=tensor.device,
        dtype=torch.long,
    )
    return tensor.index_select(dim, indices)


def _stage_token_dim(tensor: torch.Tensor) -> int:
    return 0 if tensor.ndim == 2 else 1


def _layout_stream_ranges(layout: Dsv4CompressedLayout) -> tuple[tuple[int, int], ...]:
    ranges = sorted((int(stream.start), int(stream.end)) for stream in layout.streams)
    if not ranges:
        return ()
    merged = [list(ranges[0])]
    for start, end in ranges[1:]:
        current = merged[-1]
        if int(start) <= int(current[1]):
            current[1] = max(int(current[1]), int(end))
        else:
            merged.append([int(start), int(end)])
    return tuple((int(start), int(end)) for start, end in merged)


def _compressed_owner_rank_table(layout: Dsv4CompressedLayout) -> tuple[int, ...]:
    return layout.compressed_entry_owner_ranks


def _visible_raw_swa_token_ids(
    *,
    layout: Dsv4CompressedLayout,
    query_token_id: int,
    candidate_token_ids: Sequence[int],
    window_size: int,
) -> tuple[int, ...]:
    view, query_pos = _query_branch_view(layout=layout, query_token_id=query_token_id)
    min_pos = max(0, int(query_pos) - int(window_size) + 1)
    ids: list[int] = []
    prefix_end_pos = min(int(query_pos), int(view.prefix_token_count) - 1)
    if min_pos <= prefix_end_pos:
        ids.extend(
            _candidate_ids_in_token_range(
                candidate_token_ids,
                int(view.prefix_start) + min_pos,
                int(view.prefix_start) + prefix_end_pos + 1,
            )
        )
    if view.suffix_start is not None and view.suffix_end is not None:
        suffix_start_pos = max(min_pos, int(view.prefix_token_count))
        if suffix_start_pos <= int(query_pos):
            ids.extend(
                _candidate_ids_in_token_range(
                    candidate_token_ids,
                    int(view.suffix_start)
                    + suffix_start_pos
                    - int(view.prefix_token_count),
                    int(view.suffix_start)
                    + int(query_pos)
                    - int(view.prefix_token_count)
                    + 1,
                )
            )
    return tuple(ids)


def _build_csa_stage_local_topk_tensor(
    *,
    layout: Dsv4CompressedLayout,
    query_token_ids: tuple[int, ...],
    raw_topk: torch.Tensor,
    raw_token_count: int,
    candidate_entry_ids: tuple[int, ...],
    global_topk: torch.Tensor,
    raw_list_size: int,
    compressed_list_size: int,
) -> torch.Tensor:
    device = global_topk.device
    batch_size, query_count, topk_size = global_topk.shape
    local = torch.full(
        (batch_size, query_count, raw_list_size + compressed_list_size),
        _INVALID_INDEX,
        device=device,
        dtype=torch.int64,
    )
    local[:, :, :raw_list_size] = raw_topk.to(device=device, dtype=local.dtype)
    if compressed_list_size == 0 or topk_size == 0 or not candidate_entry_ids:
        return local

    topk_long = global_topk.to(dtype=torch.long)
    candidate_ids = torch.tensor(candidate_entry_ids, device=device, dtype=torch.long)
    sorted_candidate_ids, sorted_order = candidate_ids.sort()
    positions = torch.searchsorted(sorted_candidate_ids, topk_long.clamp(min=0))
    positions = positions.clamp(max=len(candidate_entry_ids) - 1)
    matched_ids = sorted_candidate_ids[positions]
    local_values = sorted_order[positions].to(torch.long) + int(raw_token_count)
    valid = (
        (topk_long >= 0)
        & (matched_ids == topk_long)
        & _topk_entry_visibility_mask(
            layout=layout,
            query_token_ids=query_token_ids,
            topk=topk_long,
            device=device,
        )
    )
    valid = valid & _first_topk_occurrence_mask(ids=topk_long, valid=valid)
    positions = torch.arange(topk_size, device=device, dtype=torch.long)

    invalid = torch.full_like(local_values, _INVALID_INDEX)
    packed_values = torch.where(valid, local_values, invalid)
    order = torch.where(
        valid,
        positions.view(1, 1, topk_size),
        positions.view(1, 1, topk_size) + topk_size,
    )
    take = min(int(compressed_list_size), int(topk_size))
    selected_positions = order.argsort(dim=-1)[..., :take]
    selected = packed_values.gather(-1, selected_positions)
    if take < int(compressed_list_size):
        selected = torch.cat(
            (
                selected,
                selected.new_full(
                    (batch_size, query_count, int(compressed_list_size) - take),
                    _INVALID_INDEX,
                ),
            ),
            dim=-1,
        )
    local[:, :, raw_list_size : raw_list_size + compressed_list_size] = selected
    return local


def _build_raw_swa_stage_local_topk_tensor(
    *,
    layout: Dsv4CompressedLayout,
    query_token_ids: tuple[int, ...],
    candidate_token_ids: tuple[int, ...],
    window_size: int,
    raw_list_size: int,
) -> torch.Tensor:
    query_count = len(query_token_ids)
    local = _empty_stage_local_topk(
        query_count=query_count,
        raw_list_size=raw_list_size,
        compressed_list_size=0,
    )
    if raw_list_size <= 0 or not query_token_ids or not candidate_token_ids:
        return local
    branch_ids, positions, views = _layout_query_tables(layout)
    query_positions: list[int] = []
    prefix_starts: list[int] = []
    prefix_counts: list[int] = []
    suffix_starts: list[int] = []
    for token_id in query_token_ids:
        token = int(token_id)
        if token < 0 or token >= len(branch_ids) or int(branch_ids[token]) < 0:
            raise RuntimeError(f"DSV4 query token {token} is not in any branch view")
        view = views[int(branch_ids[token])]
        query_positions.append(int(positions[token]))
        prefix_starts.append(int(view.prefix_start))
        prefix_counts.append(int(view.prefix_token_count))
        suffix_starts.append(int(view.suffix_start or 0))
    query_pos = torch.tensor(query_positions, dtype=torch.long).view(-1, 1)
    prefix_start = torch.tensor(prefix_starts, dtype=torch.long).view(-1, 1)
    prefix_count = torch.tensor(prefix_counts, dtype=torch.long).view(-1, 1)
    suffix_start = torch.tensor(suffix_starts, dtype=torch.long).view(-1, 1)
    offsets = torch.arange(-int(raw_list_size) + 1, 1, dtype=torch.long).view(1, -1)
    view_pos = query_pos + offsets
    valid_view = (view_pos >= 0) & (offsets >= 1 - int(window_size))
    token_ids = torch.where(
        view_pos < prefix_count,
        prefix_start + view_pos,
        suffix_start + view_pos - prefix_count,
    )
    raw_ids = torch.tensor(candidate_token_ids, dtype=torch.long)
    token_to_local = torch.full(
        (len(layout.raw_token_owner_ranks),), _INVALID_INDEX, dtype=torch.long
    )
    token_to_local[raw_ids] = torch.arange(len(candidate_token_ids), dtype=torch.long)
    safe_token_ids = token_ids.clamp(min=0, max=max(len(token_to_local) - 1, 0))
    raw_values = torch.where(valid_view, token_to_local[safe_token_ids], local[0])
    valid = raw_values >= 0
    compact_pos = valid.to(torch.int32).cumsum(dim=1, dtype=torch.int32) - 1
    keep = valid & (compact_pos < int(raw_list_size))
    rows = torch.arange(query_count).view(-1, 1).expand_as(compact_pos)[keep]
    local[0, rows, compact_pos[keep].to(torch.long)] = raw_values[keep]
    return local


def _topk_entry_visibility_mask(
    *,
    layout: Dsv4CompressedLayout,
    query_token_ids: tuple[int, ...],
    topk: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    query_parts = tuple(
        _query_branch_view(layout=layout, query_token_id=query_id)
        for query_id in query_token_ids
    )
    query_branch = torch.tensor(
        [int(view.branch_stream_id) for view, _pos in query_parts],
        device=device,
        dtype=torch.long,
    ).view(1, -1, 1)
    query_prefix = torch.tensor(
        [int(view.prefix_stream_id) for view, _pos in query_parts],
        device=device,
        dtype=torch.long,
    ).view(1, -1, 1)
    query_pos = torch.tensor(
        [int(pos) for _view, pos in query_parts], device=device, dtype=torch.long
    ).view(1, -1, 1)
    safe = topk.clamp(min=0, max=max(layout.entry_count() - 1, 0))
    entry_branch = torch.tensor(layout.entry_branch_stream_ids, device=device)[safe]
    entry_prefix = torch.tensor(layout.entry_prefix_stream_ids, device=device)[safe]
    entry_pos = torch.tensor(layout.entry_closure_view_positions, device=device)[safe]
    entry_shared = torch.tensor(layout.entry_shared_prefix_flags, device=device)[safe]
    return (
        (topk >= 0)
        & (
            (entry_branch == query_branch)
            | (entry_shared & (entry_prefix == query_prefix))
        )
        & (entry_pos <= query_pos)
    )


def _first_topk_occurrence_mask(
    *, ids: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    sentinel = torch.full((), torch.iinfo(torch.long).max, device=ids.device)
    safe_ids = torch.where(valid, ids, sentinel)
    order = torch.argsort(safe_ids, dim=-1, stable=True)
    sorted_ids = safe_ids.gather(-1, order)
    sorted_valid = valid.gather(-1, order)
    first = torch.ones_like(sorted_valid)
    first[..., 1:] = sorted_ids[..., 1:] != sorted_ids[..., :-1]
    keep = torch.zeros_like(valid)
    keep.scatter_(-1, order, sorted_valid & first)
    return keep


def _build_hca_stage_local_topk_tensor(
    *,
    layout: Dsv4CompressedLayout,
    query_token_ids: tuple[int, ...],
    raw_topk: torch.Tensor,
    raw_token_count: int,
    candidate_entry_ids: tuple[int, ...],
    raw_list_size: int,
    compressed_list_size: int | None,
) -> torch.Tensor:
    query_count = int(raw_topk.shape[1])
    if not query_token_ids or not candidate_entry_ids:
        compressed_size = int(compressed_list_size or 0)
        local = _empty_stage_local_topk(
            query_count=query_count,
            raw_list_size=raw_list_size,
            compressed_list_size=compressed_size,
        )
        local[:, :, :raw_list_size] = raw_topk.to(dtype=local.dtype)
        return local
    query_branch, query_prefix, query_pos = _query_visibility_tensors_for_stage(
        layout=layout,
        query_token_ids=query_token_ids,
    )
    entry_ids = torch.tensor(candidate_entry_ids, dtype=torch.long)
    entry_branch = torch.tensor(layout.entry_branch_stream_ids, dtype=torch.long)[
        entry_ids
    ]
    entry_prefix = torch.tensor(layout.entry_prefix_stream_ids, dtype=torch.long)[
        entry_ids
    ]
    entry_pos = torch.tensor(layout.entry_closure_view_positions, dtype=torch.long)[
        entry_ids
    ]
    entry_shared = torch.tensor(layout.entry_shared_prefix_flags, dtype=torch.bool)[
        entry_ids
    ]
    visible = (
        (entry_branch.view(1, -1) == query_branch.view(-1, 1))
        | (
            entry_shared.view(1, -1)
            & (entry_prefix.view(1, -1) == query_prefix.view(-1, 1))
        )
    ) & (entry_pos.view(1, -1) <= query_pos.view(-1, 1))
    compressed_size = (
        int(visible.sum(dim=1).max().item())
        if compressed_list_size is None
        else int(compressed_list_size)
    )
    local = _empty_stage_local_topk(
        query_count=query_count,
        raw_list_size=raw_list_size,
        compressed_list_size=compressed_size,
    )
    local[:, :, :raw_list_size] = raw_topk.to(dtype=local.dtype)
    if compressed_size <= 0:
        return local
    compact_pos = visible.to(torch.int32).cumsum(dim=1, dtype=torch.int32) - 1
    keep = visible & (compact_pos < compressed_size)
    rows = torch.arange(query_count).view(-1, 1).expand_as(compact_pos)[keep]
    positions = compact_pos[keep].to(torch.long)
    values = (
        torch.arange(len(candidate_entry_ids), dtype=torch.long).view(1, -1)
        + int(raw_token_count)
    ).expand_as(compact_pos)[keep]
    local[0, rows, raw_list_size + positions] = values
    return local


def _empty_stage_local_topk(
    *,
    query_count: int,
    raw_list_size: int,
    compressed_list_size: int,
) -> torch.Tensor:
    return torch.full(
        (1, query_count, raw_list_size + compressed_list_size),
        _INVALID_INDEX,
        dtype=torch.int64,
    )


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

    tensor_id_tuple = tuple(int(id_) for id_ in tensor_ids)
    selected_id_tuple = tuple(int(id_) for id_ in selected_ids)
    if not selected_id_tuple:
        return tensor.narrow(token_dim, 0, 0)
    if selected_id_tuple == tensor_id_tuple:
        return tensor
    start = bisect_left(tensor_id_tuple, selected_id_tuple[0])
    end = start + len(selected_id_tuple)
    if end <= len(tensor_id_tuple) and tensor_id_tuple[start:end] == selected_id_tuple:
        return tensor.narrow(token_dim, start, len(selected_id_tuple))

    row_by_id = _row_by_id(tensor_ids=tensor_ids, name=name)
    missing = tuple(id_ for id_ in selected_id_tuple if id_ not in row_by_id)
    if missing:
        raise RuntimeError(f"DSV4 {name} tensor is missing ids {missing}")
    indices = torch.tensor(
        [row_by_id[id_] for id_ in selected_id_tuple],
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


def _query_branch_view(
    *,
    layout: Dsv4CompressedLayout,
    query_token_id: int,
) -> tuple[Dsv4BranchView, int]:
    token_id = int(query_token_id)
    branch_ids, positions, views = _layout_query_tables(layout)
    if token_id < 0 or token_id >= len(branch_ids) or int(branch_ids[token_id]) < 0:
        raise RuntimeError(f"DSV4 query token {token_id} is not in any branch view")
    return views[int(branch_ids[token_id])], int(positions[token_id])


def _query_visibility_tensors_for_stage(
    *,
    layout: Dsv4CompressedLayout,
    query_token_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branch_ids, positions, views = _layout_query_tables(layout)
    branches = []
    prefixes = []
    query_positions = []
    for token_id in query_token_ids:
        token = int(token_id)
        if token < 0 or token >= len(branch_ids) or int(branch_ids[token]) < 0:
            raise RuntimeError(f"DSV4 query token {token} is not in any branch view")
        view = views[int(branch_ids[token])]
        branches.append(int(view.branch_stream_id))
        prefixes.append(int(view.prefix_stream_id))
        query_positions.append(int(positions[token]))
    return (
        torch.tensor(branches, dtype=torch.long),
        torch.tensor(prefixes, dtype=torch.long),
        torch.tensor(query_positions, dtype=torch.long),
    )


def _candidate_ids_in_token_range(
    candidate_token_ids: Sequence[int], start: int, end: int
) -> tuple[int, ...]:
    left = bisect_left(candidate_token_ids, int(start))
    right = bisect_left(candidate_token_ids, int(end))
    return tuple(int(token_id) for token_id in candidate_token_ids[left:right])


def _layout_query_tables(
    layout: Dsv4CompressedLayout,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[int, Dsv4BranchView]]:
    key = id(layout)
    cached = _QUERY_TABLE_CACHE.get(key)
    if cached is not None and cached[0] is layout:
        return cached[1], cached[2], cached[3]
    if len(_QUERY_TABLE_CACHE) >= _QUERY_TABLE_CACHE_LIMIT:
        _QUERY_TABLE_CACHE.clear()
    branch_ids = [-1] * len(layout.raw_token_owner_ranks)
    positions = [-1] * len(layout.raw_token_owner_ranks)
    views = {int(view.branch_stream_id): view for view in layout.branch_views}
    for view in layout.branch_views:
        branch_id = int(view.branch_stream_id)
        if branch_id == int(view.prefix_stream_id):
            for token_id in range(int(view.prefix_start), int(view.prefix_end)):
                branch_ids[token_id] = branch_id
                positions[token_id] = token_id - int(view.prefix_start)
        if view.suffix_start is not None and view.suffix_end is not None:
            for token_id in range(int(view.suffix_start), int(view.suffix_end)):
                branch_ids[token_id] = branch_id
                positions[token_id] = (
                    int(view.prefix_token_count) + token_id - int(view.suffix_start)
                )
    _QUERY_TABLE_CACHE[key] = (layout, tuple(branch_ids), tuple(positions), views)
    return _QUERY_TABLE_CACHE[key][1], _QUERY_TABLE_CACHE[key][2], views
