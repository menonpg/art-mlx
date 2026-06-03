from __future__ import annotations

from bisect import bisect_left
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
    Dsv4IndexerKvExchangePeerPlan,
    Dsv4MaterializedStage,
    Dsv4StageInputs,
    Dsv4StageKeyKind,
    Dsv4StageKvExchangePeerPlan,
    Dsv4StagePlanGroup,
    Dsv4StagePlanSlot,
    Dsv4TensorExchangePlan,
)

_INVALID_INDEX = -1


class TokenRangeLike(Protocol):
    start: int
    end: int


class Dsv4StageKvExchangeWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    materialize_compressed_metadata: bool = True,
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
    stage_k_ranges = tuple(global_k_ranges) if query_ids else ()
    raw_token_ids = _stage_raw_token_ids(layout=layout, ranges=stage_k_ranges)
    raw_local = {token_id: offset for offset, token_id in enumerate(raw_token_ids)}
    compressed_entry_ids = stage_candidate_entry_ids(
        layout=layout,
        global_k_ranges=stage_k_ranges,
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
    if compression_kind == Dsv4CompressionKind.CSA:
        if topk is None:
            raise RuntimeError("DSV4 CSA stage remap requires global_topk")
        local_topk = _build_csa_stage_local_topk_tensor(
            layout=layout,
            query_token_ids=query_ids,
            raw_by_query=raw_by_query,
            raw_local=raw_local,
            raw_token_count=len(raw_token_ids),
            candidate_entry_ids=compressed_entry_ids,
            global_topk=topk,
            raw_list_size=int(raw_list_size),
            compressed_list_size=int(compressed_list_size),
        )
        compressed_by_query = (
            _compressed_ids_by_query(
                layout=layout,
                query_token_ids=query_ids,
                candidate_entry_ids=compressed_entry_ids,
                compression_kind=compression_kind,
                global_topk=topk,
            )
            if materialize_compressed_metadata
            else ()
        )
    else:
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
    materialize_compressed_metadata: bool = True,
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
        materialize_compressed_metadata=materialize_compressed_metadata,
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


def build_dsv4_stage_plan_slots(
    *,
    stage_plans_by_rank: Sequence[Sequence[Any]],
) -> tuple[Dsv4StagePlanSlot, ...]:
    """Group ART StagePlans by stage index across all CP ranks.

    This is host-only DSV4 planning metadata. It preserves rank 0's stage order
    and validates that every rank has exactly one StagePlan for each stage id,
    so later DSV4 launchers can derive compatible all-rank exchange requests.
    """
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
    materialize_compressed_metadata: bool = True,
) -> Dsv4StageInputs:
    """Build this rank's DSV4 stage inputs from one ART StagePlan."""
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
        materialize_compressed_metadata=materialize_compressed_metadata,
    )


def build_dsv4_stage_plan_group_from_stage_plans(
    *,
    layout: Dsv4CompressedLayout,
    stage_plans_by_rank: Sequence[Any],
    compression_kind: Dsv4CompressionKind,
    global_topk_indices_by_rank: Sequence[torch.Tensor] | None = None,
    topk_query_token_ids_by_rank: Sequence[Sequence[int]] | None = None,
    window_size: int = 128,
    raw_list_size: int | None = None,
    compressed_list_size: int | None = None,
) -> Dsv4StagePlanGroup:
    """Derive DSV4 main-attention metadata from one ART `StagePlan` slot.

    This is host/index metadata plus optional CSA topk row selection only. It
    uses ART's planned query/K ranges without calling the generic Flex executor.
    """
    stage_plans = tuple(stage_plans_by_rank)
    _validate_stage_plan_count(layout=layout, stage_plans_by_rank=stage_plans)
    stage_index = _shared_stage_index(stage_plans)
    if compression_kind == Dsv4CompressionKind.CSA:
        if global_topk_indices_by_rank is None or topk_query_token_ids_by_rank is None:
            raise RuntimeError("DSV4 CSA StagePlan conversion requires global topk")
        if len(global_topk_indices_by_rank) != len(stage_plans) or len(
            topk_query_token_ids_by_rank
        ) != len(stage_plans):
            raise RuntimeError("DSV4 CSA topk metadata must have one entry per rank")
    elif (
        global_topk_indices_by_rank is not None
        or topk_query_token_ids_by_rank is not None
    ):
        raise RuntimeError("DSV4 HCA StagePlan conversion does not consume topk")

    csa_topk_by_rank = (
        tuple(global_topk_indices_by_rank)
        if global_topk_indices_by_rank is not None
        else None
    )
    csa_topk_ids_by_rank = (
        tuple(topk_query_token_ids_by_rank)
        if topk_query_token_ids_by_rank is not None
        else None
    )
    stage_inputs = []
    for rank, stage_plan in enumerate(stage_plans):
        stage_inputs.append(
            build_dsv4_stage_inputs_from_stage_plan(
                layout=layout,
                compression_kind=compression_kind,
                stage_plan=stage_plan,
                global_topk=csa_topk_by_rank[rank]
                if csa_topk_by_rank is not None
                else None,
                topk_query_token_ids=csa_topk_ids_by_rank[rank]
                if csa_topk_ids_by_rank is not None
                else None,
                window_size=window_size,
                raw_list_size=raw_list_size,
                compressed_list_size=compressed_list_size,
                materialize_compressed_metadata=True,
            )
        )
    return Dsv4StagePlanGroup(
        stage_index=stage_index,
        stage_inputs_by_rank=tuple(stage_inputs),
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
    return _build_stage_kv_exchange_peer_plans_from_ids(
        layout=layout,
        raw_token_ids_by_rank=tuple(
            stage.raw_token_ids for stage in stage_inputs_by_rank
        ),
        compressed_entry_ids_by_rank=tuple(
            stage.compressed_entry_ids for stage in stage_inputs_by_rank
        ),
    )


def _build_stage_kv_exchange_peer_plans_from_ids(
    *,
    layout: Dsv4CompressedLayout,
    raw_token_ids_by_rank: Sequence[Sequence[int]],
    compressed_entry_ids_by_rank: Sequence[Sequence[int]],
) -> tuple[Dsv4StageKvExchangePeerPlan, ...]:
    rank_count = len(layout.entry_ids_by_owner_rank)
    if len(raw_token_ids_by_rank) != rank_count:
        raise RuntimeError(
            "DSV4 stage raw exchange planning requires one id list per rank, got "
            f"{len(raw_token_ids_by_rank)} vs {rank_count}"
        )
    if len(compressed_entry_ids_by_rank) != rank_count:
        raise RuntimeError(
            "DSV4 stage compressed exchange planning requires one id list per rank, "
            f"got {len(compressed_entry_ids_by_rank)} vs {rank_count}"
        )
    recv_raw = tuple(
        _ids_by_owner_rank(
            ids=raw_token_ids,
            rank_count=rank_count,
            owner_rank=lambda token_id: _raw_token_owner_rank(
                layout=layout,
                token_id=token_id,
            ),
            name=f"rank{rank}_raw_token_ids",
        )
        for rank, raw_token_ids in enumerate(raw_token_ids_by_rank)
    )
    recv_compressed = tuple(
        _ids_by_owner_rank(
            ids=compressed_entry_ids,
            rank_count=rank_count,
            owner_rank=lambda entry_id: _compressed_entry_owner_rank(
                layout=layout,
                entry_id=entry_id,
            ),
            name=f"rank{rank}_compressed_entry_ids",
        )
        for rank, compressed_entry_ids in enumerate(compressed_entry_ids_by_rank)
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


def build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans(
    *,
    layout: Dsv4CompressedLayout,
    stage_plans_by_rank: Sequence[Any],
) -> tuple[Dsv4StageKvExchangePeerPlan, ...]:
    """Plan fused stage KV exchange from ART StagePlans without topk metadata.

    The KV exchange fetches every raw token and compressed candidate whose
    closure lies in the stage K ranges. CSA topk only affects the local sparse
    kernel list tensor, not which compressed candidate rows are fetched for the
    stage. This avoids requiring an all-rank topk tensor gather before planning
    compatible all-to-all sends.
    """
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
    """Plan fused stage KV exchange metadata for multiple DSV4 layouts.

    CSA and HCA share the raw SWA token requests for one ART StagePlan slot.
    Building them together avoids expanding raw K ranges and bucketing raw
    owners separately for each compression family.
    """
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
        _stage_raw_token_ids_by_owner_rank(
            layout=layout_tuple[0],
            ranges=stage_plan.global_k_ranges if has_queries[rank] else (),
            rank_count=rank_count,
        )
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
    return _launch_dsv4_stage_kv_exchange_impl(
        stage_inputs=stage_inputs,
        query=query,
        query_token_ids=query_token_ids,
        raw_kv=raw_kv,
        raw_token_ids=raw_token_ids,
        compressed_kv=compressed_kv,
        compressed_entry_ids=compressed_entry_ids,
        send_raw_token_ids_by_peer=send_raw_token_ids_by_peer,
        send_compressed_entry_ids_by_peer=send_compressed_entry_ids_by_peer,
        recv_raw_token_ids_by_peer=recv_raw_token_ids_by_peer,
        recv_compressed_entry_ids_by_peer=recv_compressed_entry_ids_by_peer,
        group=group,
        async_op=async_op,
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
    peer_plans: Sequence[Dsv4StageKvExchangePeerPlan] | None = None,
) -> Dsv4StageKvExchangeWork:
    """Launch one rank's fused stage KV exchange from DSV4 host metadata.

    This wrapper turns all-rank stage metadata into this rank's peer send/recv
    lists, then delegates to the eager raw+compressed KV exchange path. It keeps
    DSV4-specific planning outside the generic Flex CP executor.
    """
    rank_int = _validate_rank(rank=rank, rank_count=len(layout.entry_ids_by_owner_rank))
    plans = _stage_kv_peer_plans_or_build(
        layout=layout,
        stage_inputs_by_rank=stage_inputs_by_rank,
        peer_plans=peer_plans,
    )
    plan = plans[rank_int]
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
    """Launch one rank's fused stage KV exchange from an ART StagePlan slot.

    `local_stage_inputs` contains this rank's real sparse-kernel topk list.
    Peer send/recv lists are derived from all ranks' StagePlan K ranges and are
    deliberately independent of CSA topk, because every compressed candidate in
    a stage K range may be fetched while the local topk list selects the rows
    the Miles kernel actually scores.
    """
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
    return launch_dsv4_stage_kv_exchange(
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
    """Launch a stage KV exchange before CSA topk-derived inputs exist.

    CSA topk controls only the sparse-kernel list tensor. The raw/compressed
    candidate rows fetched for an ART StagePlan slot are determined by the CP
    K ranges, so this topk-independent exchange can be queued while the frozen
    indexer topk is still pending, then bound to `Dsv4StageInputs` before
    materialization.
    """
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


def _stage_kv_peer_plans_or_build(
    *,
    layout: Dsv4CompressedLayout,
    stage_inputs_by_rank: Sequence[Dsv4StageInputs],
    peer_plans: Sequence[Dsv4StageKvExchangePeerPlan] | None,
) -> tuple[Dsv4StageKvExchangePeerPlan, ...]:
    if peer_plans is None:
        return build_dsv4_stage_kv_exchange_peer_plans(
            layout=layout,
            stage_inputs_by_rank=stage_inputs_by_rank,
        )
    return _validate_stage_kv_peer_plan_count(
        layout=layout,
        peer_plans=peer_plans,
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


def _stage_raw_token_ids_by_owner_rank(
    *,
    layout: Dsv4CompressedLayout,
    ranges: Sequence[TokenRangeLike],
    rank_count: int,
) -> tuple[tuple[int, ...], ...]:
    by_rank: list[list[int]] = [[] for _ in range(int(rank_count))]
    intersections: list[tuple[int, int]] = []
    valid_ranges = _layout_stream_ranges(layout)
    owner_ranks = layout.raw_token_owner_ranks
    owner_count = len(owner_ranks)
    owner_changes = layout.raw_token_owner_change_positions
    for range_ in ranges:
        range_start = int(range_.start)
        range_end = int(range_.end)
        for stream_start, stream_end in valid_ranges:
            start = max(range_start, stream_start)
            end = min(range_end, stream_end)
            if start < end:
                intersections.append((start, end))
    for start, end in _merge_stage_id_ranges(intersections):
        current = int(start)
        end_int = int(end)
        if current < 0 or end_int > owner_count:
            raise RuntimeError(
                f"DSV4 stage raw token range {current}:{end_int} is outside owner table"
            )
        while current < end_int:
            rank = int(owner_ranks[current])
            if rank < 0 or rank >= int(rank_count):
                raise RuntimeError(
                    f"DSV4 stage raw token id {current} has invalid owner rank {rank}"
                )
            change_index = bisect_left(owner_changes, current + 1)
            next_change = (
                int(owner_changes[change_index])
                if change_index < len(owner_changes)
                else end_int
            )
            segment_end = min(end_int, next_change)
            by_rank[rank].extend(range(current, segment_end))
            current = segment_end
    return tuple(tuple(ids) for ids in by_rank)


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
    if global_topk.ndim != 3:
        raise RuntimeError(
            f"DSV4 global topk must be [B,Q,K], got {tuple(global_topk.shape)}"
        )
    global_ids = tuple(int(token_id) for token_id in global_query_token_ids)
    if len(global_ids) != int(global_topk.shape[1]):
        raise RuntimeError(
            "DSV4 global topk query ids must match Q dim, got "
            f"{len(global_ids)} vs {int(global_topk.shape[1])}"
        )
    row_by_id = _row_by_id(
        tensor_ids=global_ids,
        name="global_topk_query_token_ids",
    )
    selected_ids = tuple(int(token_id) for token_id in selected_query_token_ids)
    missing = tuple(token_id for token_id in selected_ids if token_id not in row_by_id)
    if missing:
        raise RuntimeError(f"DSV4 stage query ids missing from global topk: {missing}")
    index = torch.tensor(
        tuple(row_by_id[token_id] for token_id in selected_ids),
        device=global_topk.device,
        dtype=torch.long,
    )
    return global_topk.index_select(1, index)


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


def _layout_stream_ranges(layout: Dsv4CompressedLayout) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted((int(stream.start), int(stream.end)) for stream in layout.streams)
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
    if entry_int < 0 or entry_int >= layout.entry_count():
        raise RuntimeError(f"DSV4 compressed entry {entry_int} is outside layout")
    if layout.compressed_entry_owner_ranks:
        return int(layout.compressed_entry_owner_ranks[entry_int])
    return int(layout.entries[entry_int].owner_rank)


def _compressed_owner_rank_table(layout: Dsv4CompressedLayout) -> tuple[int, ...]:
    if layout.compressed_entry_owner_ranks:
        return layout.compressed_entry_owner_ranks
    return tuple(int(entry.owner_rank) for entry in layout.entries)


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


def _build_csa_stage_local_topk_tensor(
    *,
    layout: Dsv4CompressedLayout,
    query_token_ids: tuple[int, ...],
    raw_by_query: tuple[tuple[int, ...], ...],
    raw_local: dict[int, int],
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
    _fill_raw_stage_local_topk(
        local=local,
        raw_by_query=raw_by_query,
        raw_local=raw_local,
        raw_list_size=raw_list_size,
    )
    if compressed_list_size == 0 or topk_size == 0 or not candidate_entry_ids:
        return local

    candidate_ids = torch.tensor(
        candidate_entry_ids,
        device=device,
        dtype=torch.long,
    )
    candidate_local = torch.arange(
        len(candidate_entry_ids), device=device, dtype=torch.long
    ) + int(raw_token_count)
    visible_mask = _compressed_visibility_mask(
        layout=layout,
        query_token_ids=query_token_ids,
        candidate_entry_ids=candidate_entry_ids,
        device=device,
    )
    topk_long = global_topk.to(dtype=torch.long)
    matches = topk_long.unsqueeze(-1) == candidate_ids.view(1, 1, 1, -1)
    visible_matches = matches & visible_mask.view(1, query_count, 1, -1)
    valid = visible_matches.any(dim=-1)

    local_values = torch.where(
        visible_matches,
        candidate_local.view(1, 1, 1, -1),
        torch.zeros((), device=device, dtype=torch.long),
    ).amax(dim=-1)
    positions = torch.arange(topk_size, device=device, dtype=torch.long)
    first_pos_by_candidate = torch.where(
        matches,
        positions.view(1, 1, topk_size, 1),
        torch.full((), topk_size, device=device, dtype=torch.long),
    ).amin(dim=2)
    first_pos_for_topk = torch.where(
        matches,
        first_pos_by_candidate.unsqueeze(2),
        torch.zeros((), device=device, dtype=torch.long),
    ).amax(dim=-1)
    valid = valid & (positions.view(1, 1, topk_size) == first_pos_for_topk)

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


def _fill_raw_stage_local_topk(
    *,
    local: torch.Tensor,
    raw_by_query: tuple[tuple[int, ...], ...],
    raw_local: dict[int, int],
    raw_list_size: int,
) -> None:
    for query_idx, raw_ids in enumerate(raw_by_query):
        selected_raw = raw_ids[-raw_list_size:] if raw_list_size > 0 else ()
        for offset, token_id in enumerate(selected_raw):
            local[:, query_idx, offset] = int(raw_local[int(token_id)])


def _compressed_visibility_mask(
    *,
    layout: Dsv4CompressedLayout,
    query_token_ids: tuple[int, ...],
    candidate_entry_ids: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not query_token_ids:
        return torch.zeros(
            (0, len(candidate_entry_ids)), device=device, dtype=torch.bool
        )
    rows = [
        [
            entry_id
            in visible_entry_ids_for_query(
                layout=layout,
                query_token_id=query_id,
                candidate_entry_ids=candidate_entry_ids,
            )
            for entry_id in candidate_entry_ids
        ]
        for query_id in query_token_ids
    ]
    return torch.tensor(rows, device=device, dtype=torch.bool)


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
    stream_id = _stream_id_for_query_token(layout=layout, token_id=query_token_id)
    for view in layout.branch_views:
        if int(view.branch_stream_id) != int(stream_id):
            continue
        query_pos = view.position_of_token(query_token_id)
        if query_pos is not None:
            return view, int(query_pos)
    raise RuntimeError(f"DSV4 query token {query_token_id} is not in any branch view")


def _position_in_view(*, view: Dsv4BranchView, token_id: int) -> int | None:
    return view.position_of_token(token_id)


def _stream_id_for_query_token(*, layout: Dsv4CompressedLayout, token_id: int) -> int:
    token_int = int(token_id)
    for stream in layout.streams:
        if int(stream.start) <= token_int < int(stream.end):
            return int(stream.stream_id)
    raise RuntimeError(f"DSV4 query token {token_int} is not in any stream")
