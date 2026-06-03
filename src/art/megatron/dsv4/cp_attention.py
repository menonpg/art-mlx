from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, PrivateAttr
import torch
import torch.distributed as dist

from . import sparse_kernel
from .comm import Dsv4TensorExchangeWork, launch_dsv4_tensor_exchange
from .compressor import (
    Dsv4CompressedKvForwardWork,
    launch_dsv4_compressed_kv_backward,
    launch_dsv4_compressed_kv_forward,
)
from .cp_stage import (
    build_dsv4_stage_inputs_from_stage_plan,
    build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans_for_layouts,
    launch_dsv4_stage_kv_exchange_deferred_from_stage_plan_slot,
    launch_dsv4_stage_kv_exchange_from_stage_plan_slot,
)
from .indexer import (
    build_dsv4_indexer_stage_plan_from_stage_plans,
    launch_dsv4_indexer_topk_from_stage_plans,
)
from .types import (
    Dsv4AttentionBackwardPlan,
    Dsv4AttentionBackwardRankPlan,
    Dsv4AttentionBackwardReplayResult,
    Dsv4AttentionForwardResult,
    Dsv4AttentionGradientResult,
    Dsv4CompressedKvForwardResult,
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4ContextParallelState,
    Dsv4GradientOwnerBucket,
    Dsv4IndexerKvExchangePeerPlan,
    Dsv4IndexerStagePlan,
    Dsv4MaterializedStage,
    Dsv4ProjectedAttentionForwardResult,
    Dsv4ProjectedAttentionGradientResult,
    Dsv4StageBackwardRecord,
    Dsv4StageForwardRecord,
    Dsv4StageKeyKind,
    Dsv4StageKvExchangePeerPlan,
    Dsv4StagePlanSlot,
    Dsv4TensorExchangePlan,
)

_DIST = cast(Any, dist)


class Dsv4GradientOwnerExchangeWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rank: int
    rank_count: int
    recv_query_token_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_raw_token_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_compressed_entry_ids_by_peer: tuple[tuple[int, ...], ...]
    query_head_count: int
    query_head_dim: int
    query_work: Dsv4TensorExchangeWork
    raw_work: Dsv4TensorExchangeWork
    compressed_work: Dsv4TensorExchangeWork

    def wait(self) -> None:
        self.query_work.wait()
        self.raw_work.wait()
        self.compressed_work.wait()

    def wait_post_process(self) -> tuple[Dsv4GradientOwnerBucket, ...]:
        query_result = self.query_work.wait_post_process()
        raw_result = self.raw_work.wait_post_process()
        compressed_result = self.compressed_work.wait_post_process()
        _validate_owner_exchange_result_ids(
            actual=query_result.ids,
            expected=_wire_peer_ids_by_peer(
                ids_by_peer=self.recv_query_token_ids_by_peer,
                rank_count=int(self.rank_count),
            ),
            name="query",
        )
        _validate_owner_exchange_result_ids(
            actual=raw_result.ids,
            expected=_wire_peer_ids_by_peer(
                ids_by_peer=self.recv_raw_token_ids_by_peer,
                rank_count=int(self.rank_count),
            ),
            name="raw",
        )
        _validate_owner_exchange_result_ids(
            actual=compressed_result.ids,
            expected=_wire_peer_ids_by_peer(
                ids_by_peer=self.recv_compressed_entry_ids_by_peer,
                rank_count=int(self.rank_count),
            ),
            name="compressed",
        )

        query_tensor = query_result.tensor.reshape(
            int(query_result.tensor.shape[0]),
            int(query_result.tensor.shape[1]),
            int(self.query_head_count),
            int(self.query_head_dim),
        )
        buckets: list[Dsv4GradientOwnerBucket] = []
        q_cursor = 0
        raw_cursor = 0
        compressed_cursor = 0
        for peer in range(int(self.rank_count)):
            query_ids = self.recv_query_token_ids_by_peer[peer]
            raw_ids = self.recv_raw_token_ids_by_peer[peer]
            compressed_ids = self.recv_compressed_entry_ids_by_peer[peer]
            q_count = len(query_ids)
            raw_count = len(raw_ids)
            compressed_count = len(compressed_ids)
            if q_count or raw_count or compressed_count:
                buckets.append(
                    Dsv4GradientOwnerBucket(
                        owner_rank=int(self.rank),
                        query_token_ids=query_ids,
                        raw_token_ids=raw_ids,
                        compressed_entry_ids=compressed_ids,
                        dq=query_tensor[:, q_cursor : q_cursor + q_count],
                        draw_kv=raw_result.tensor[
                            :,
                            raw_cursor : raw_cursor + raw_count,
                        ],
                        dcompressed_kv=compressed_result.tensor[
                            :,
                            compressed_cursor : compressed_cursor + compressed_count,
                        ],
                    )
                )
            q_cursor += q_count
            raw_cursor += raw_count
            compressed_cursor += compressed_count
        return tuple(buckets)


class Dsv4SinkGradientReduceWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    d_attn_sink: torch.Tensor
    handle: Any | None
    _wait_complete: bool = PrivateAttr(default=False)

    def wait(self) -> None:
        if self._wait_complete:
            return
        if self.handle is not None:
            self.handle.wait()
        self._wait_complete = True

    def wait_post_process(self) -> torch.Tensor:
        self.wait()
        return self.d_attn_sink


class Dsv4ExchangedAttentionForwardWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stage_works: tuple[Any, ...]
    query_token_ids: tuple[int, ...]
    attn_sink: torch.Tensor
    scale: float | None = None

    def wait(self) -> None:
        for work in self.stage_works:
            work.wait()

    def wait_post_process(self) -> Dsv4AttentionForwardResult:
        stages = tuple(
            _materialized_stage_from_work(work=work, position=position)
            for position, work in enumerate(self.stage_works)
        )
        return run_materialized_dsv4_attention_forward(
            stages=stages,
            query_token_ids=self.query_token_ids,
            attn_sink=self.attn_sink,
            scale=self.scale,
        )


class Dsv4ExchangedAttentionBackwardWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    local_gradients: Dsv4AttentionGradientResult
    owner_work: Any
    sink_work: Any | None = None
    owned_query_token_ids: tuple[int, ...]
    owned_raw_token_ids: tuple[int, ...]
    owned_compressed_entry_ids: tuple[int, ...]

    def wait(self) -> None:
        self.owner_work.wait()
        if self.sink_work is not None:
            self.sink_work.wait()

    def wait_post_process(self) -> Dsv4AttentionGradientResult:
        buckets = _owner_buckets_from_work(self.owner_work)
        d_attn_sink = _sink_gradient_from_work(
            sink_work=self.sink_work,
            local_d_attn_sink=self.local_gradients.d_attn_sink,
        )
        if buckets:
            return accumulate_dsv4_gradient_owner_buckets(
                buckets=buckets,
                query_token_ids=self.owned_query_token_ids,
                raw_token_ids=self.owned_raw_token_ids,
                compressed_entry_ids=self.owned_compressed_entry_ids,
                d_attn_sink=d_attn_sink,
            )
        return _empty_owned_gradient_result(
            template=self.local_gradients,
            d_attn_sink=d_attn_sink,
            query_token_ids=self.owned_query_token_ids,
            raw_token_ids=self.owned_raw_token_ids,
            compressed_entry_ids=self.owned_compressed_entry_ids,
        )


class Dsv4ProjectedCompressionForwardWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    compression_kind: Dsv4CompressionKind
    layout: Dsv4CompressedLayout
    rank: int
    main_compression_work: Dsv4CompressedKvForwardWork
    indexer_compression_work: Dsv4CompressedKvForwardWork | None = None

    def wait(self) -> None:
        self.main_compression_work.wait()
        if self.indexer_compression_work is not None:
            self.indexer_compression_work.wait()


class Dsv4ProjectedAttentionForwardWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    compression_kind: Dsv4CompressionKind
    layout: Dsv4CompressedLayout
    rank: int
    stage_plan_slots: tuple[Dsv4StagePlanSlot, ...]
    query: torch.Tensor
    query_token_ids: tuple[int, ...]
    raw_kv: torch.Tensor
    raw_token_ids: tuple[int, ...]
    main_compression_work: Dsv4CompressedKvForwardWork
    indexer_compression_work: Dsv4CompressedKvForwardWork | None = None
    indexer_q: torch.Tensor | None = None
    indexer_weights: torch.Tensor | None = None
    indexer_topk: int | None = None
    indexer_stage_plans: tuple[Dsv4IndexerStagePlan, ...] | None = None
    indexer_kv_peer_plans_by_stage: (
        tuple[tuple[Dsv4IndexerKvExchangePeerPlan, ...], ...] | None
    ) = None
    stage_kv_peer_plans_by_slot: (
        tuple[tuple[Dsv4StageKvExchangePeerPlan, ...], ...] | None
    ) = None
    indexer_score_scale: float = 1.0
    attn_sink: torch.Tensor
    group: Any
    async_op: bool
    scale: float | None = None
    window_size: int = 128
    raw_list_size: int | None = None
    compressed_list_size: int | None = None
    _attention_work: Any | None = PrivateAttr(default=None)
    _main_compressed: Dsv4CompressedKvForwardResult | None = PrivateAttr(default=None)
    _indexer_compressed: Dsv4CompressedKvForwardResult | None = PrivateAttr(
        default=None
    )
    _result: Dsv4ProjectedAttentionForwardResult | None = PrivateAttr(default=None)

    def wait(self) -> None:
        self._ensure_attention_work().wait()

    def wait_post_process(self) -> Dsv4ProjectedAttentionForwardResult:
        if self._result is not None:
            return self._result
        attention = self._ensure_attention_work().wait_post_process()
        if self._main_compressed is None:
            raise RuntimeError("DSV4 projected attention missing main compression")
        self._result = Dsv4ProjectedAttentionForwardResult(
            compression_kind=self.compression_kind,
            attention=attention,
            main_compressed=self._main_compressed,
            indexer_compressed=self._indexer_compressed,
        )
        return self._result

    def _ensure_attention_work(self) -> Any:
        if self._attention_work is not None:
            return self._attention_work
        if self.compression_kind == Dsv4CompressionKind.CSA:
            if (
                self.indexer_compression_work is None
                or self.indexer_q is None
                or self.indexer_weights is None
                or self.indexer_topk is None
            ):
                raise RuntimeError("DSV4 CSA projected attention requires indexer data")
            (
                rank_int,
                slots,
                query_ids,
                indexer_stage_plans,
                stage_kv_peer_plans,
            ) = _prepare_csa_stage_plan_slot_launch(
                layout=self.layout,
                rank=int(self.rank),
                stage_plan_slots=self.stage_plan_slots,
                query_token_ids=self.query_token_ids,
                indexer_stage_plans=self.indexer_stage_plans,
                stage_kv_peer_plans_by_slot=self.stage_kv_peer_plans_by_slot,
            )
            with torch.no_grad():
                self._indexer_compressed = (
                    self.indexer_compression_work.wait_post_process()
                )
            topk_work = _launch_dsv4_csa_indexer_topk_from_prepared_slots(
                layout=self.layout,
                rank=rank_int,
                indexer_stage_plans=indexer_stage_plans,
                query_token_ids=query_ids,
                indexer_q=self.indexer_q,
                indexer_weights=self.indexer_weights,
                indexer_kv=self._indexer_compressed.compressed_kv,
                indexer_kv_entry_ids=self._indexer_compressed.compressed_entry_ids,
                indexer_topk=int(self.indexer_topk),
                group=self.group,
                async_op=bool(self.async_op),
                indexer_score_scale=float(self.indexer_score_scale),
                indexer_kv_peer_plans_by_stage=self.indexer_kv_peer_plans_by_stage,
            )
            with torch.no_grad():
                self._main_compressed = self.main_compression_work.wait_post_process()
            self._attention_work = _launch_dsv4_csa_attention_forward_from_topk_work(
                layout=self.layout,
                rank=rank_int,
                slots=slots,
                query=self.query,
                query_token_ids=query_ids,
                raw_kv=self.raw_kv,
                raw_token_ids=self.raw_token_ids,
                compressed_kv=self._main_compressed.compressed_kv,
                compressed_entry_ids=self._main_compressed.compressed_entry_ids,
                topk_work=topk_work,
                stage_kv_peer_plans_by_slot=stage_kv_peer_plans,
                attn_sink=self.attn_sink,
                group=self.group,
                async_op=bool(self.async_op),
                scale=self.scale,
                window_size=int(self.window_size),
                raw_list_size=self.raw_list_size,
                compressed_list_size=self.compressed_list_size,
            )
        elif self.compression_kind == Dsv4CompressionKind.HCA:
            with torch.no_grad():
                self._main_compressed = self.main_compression_work.wait_post_process()
            self._attention_work = (
                launch_dsv4_hca_attention_forward_from_stage_plan_slots(
                    layout=self.layout,
                    rank=int(self.rank),
                    stage_plan_slots=self.stage_plan_slots,
                    query=self.query,
                    query_token_ids=self.query_token_ids,
                    raw_kv=self.raw_kv,
                    raw_token_ids=self.raw_token_ids,
                    compressed_kv=self._main_compressed.compressed_kv,
                    compressed_entry_ids=self._main_compressed.compressed_entry_ids,
                    stage_kv_peer_plans_by_slot=self.stage_kv_peer_plans_by_slot,
                    attn_sink=self.attn_sink,
                    group=self.group,
                    async_op=bool(self.async_op),
                    scale=self.scale,
                    window_size=int(self.window_size),
                    raw_list_size=self.raw_list_size,
                    compressed_list_size=self.compressed_list_size,
                )
            )
        else:
            raise RuntimeError(
                f"Unsupported DSV4 projected attention kind {self.compression_kind}"
            )
        return self._attention_work


class Dsv4ProjectedAttentionBackwardWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    attention_work: Dsv4ExchangedAttentionBackwardWork
    forward_result: Dsv4ProjectedAttentionForwardResult
    group: Any
    async_op: bool
    _compressor_work: Any | None = PrivateAttr(default=None)
    _result: Dsv4ProjectedAttentionGradientResult | None = PrivateAttr(default=None)

    def wait(self) -> None:
        self.wait_post_process()

    def wait_post_process(self) -> Dsv4ProjectedAttentionGradientResult:
        if self._result is not None:
            return self._result
        attention_gradients = self.attention_work.wait_post_process()
        dcompressed = _align_compressed_grad_to_forward(
            forward_result=self.forward_result.main_compressed,
            attention_gradients=attention_gradients,
        )
        compressor_work = launch_dsv4_compressed_kv_backward(
            forward_result=self.forward_result.main_compressed,
            dcompressed_kv=dcompressed,
            group=self.group,
            async_op=bool(self.async_op),
        )
        self._compressor_work = compressor_work
        main_compressor = compressor_work.wait_post_process()
        self._result = Dsv4ProjectedAttentionGradientResult(
            attention=attention_gradients,
            main_compressor=main_compressor,
        )
        return self._result


def _accum_output_dtype(input_dtype: torch.dtype) -> torch.dtype:
    if input_dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return input_dtype


def _safe_logaddexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.logaddexp(a, b)
    both_neg_inf = torch.isneginf(a) & torch.isneginf(b)
    return torch.where(both_neg_inf, torch.full_like(out, float("-inf")), out)


def _safe_exp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a - b
    both_neg_inf = torch.isneginf(a) & torch.isneginf(b)
    diff = torch.where(both_neg_inf, torch.full_like(diff, float("-inf")), diff)
    return torch.exp(diff)


def _zero_invalid_rows(out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isneginf(lse).unsqueeze(-1), torch.zeros_like(out), out)


def merge_two_stage_outputs(
    prev_out: torch.Tensor,
    prev_lse: torch.Tensor,
    stage_out: torch.Tensor,
    stage_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    merged_lse = _safe_logaddexp(prev_lse, stage_lse)
    prev_weight = _safe_exp_diff(prev_lse, merged_lse).unsqueeze(-1)
    stage_weight = _safe_exp_diff(stage_lse, merged_lse).unsqueeze(-1)
    merged_out = prev_weight * prev_out + stage_weight * stage_out
    return _zero_invalid_rows(merged_out, merged_lse), merged_lse


def merge_stage_outputs(
    stage_outputs: Sequence[torch.Tensor],
    stage_lses: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(stage_outputs) == 0:
        raise ValueError("at least one DSV4 stage output is required")
    if len(stage_outputs) != len(stage_lses):
        raise ValueError("stage_outputs and stage_lses must have the same length")

    target_dtype = _accum_output_dtype(stage_outputs[0].dtype)
    accum_out = stage_outputs[0].to(dtype=target_dtype)
    accum_lse = stage_lses[0].to(dtype=target_dtype)
    if accum_out.shape[:-1] != accum_lse.shape:
        raise ValueError("stage output shape must be stage_lse shape plus head dim")
    accum_out = _zero_invalid_rows(accum_out, accum_lse)

    expected_out_shape = accum_out.shape
    expected_lse_shape = accum_lse.shape
    for stage_out, stage_lse in zip(stage_outputs[1:], stage_lses[1:]):
        if (
            stage_out.shape != expected_out_shape
            or stage_lse.shape != expected_lse_shape
        ):
            raise ValueError("all stage outputs and LSEs must share the same shape")
        accum_out, accum_lse = merge_two_stage_outputs(
            accum_out,
            accum_lse,
            stage_out.to(dtype=target_dtype),
            stage_lse.to(dtype=target_dtype),
        )
    return accum_out, accum_lse


def run_materialized_dsv4_attention_forward(
    *,
    stages: Sequence[Dsv4MaterializedStage],
    query_token_ids: Sequence[int],
    attn_sink: torch.Tensor,
    scale: float | None = None,
) -> Dsv4AttentionForwardResult:
    if len(stages) == 0:
        raise ValueError("at least one materialized DSV4 stage is required")
    query_ids = tuple(int(token_id) for token_id in query_token_ids)

    records: list[Dsv4StageForwardRecord] = []
    disabled_sink = sparse_kernel.dsv4_disabled_attn_sink(attn_sink)
    for stage in stages:
        stage_result = sparse_kernel.dsv4_sparse_fwd(
            q=stage.q_stage,
            kv=stage.kv_stage,
            attn_sink=disabled_sink,
            topk=stage.topk_stage_local,
            scale=scale,
        )
        _validate_stage_forward_shapes(
            stage=stage, out=stage_result.out, lse=stage_result.lse
        )
        records.append(
            Dsv4StageForwardRecord(
                materialized_stage=stage,
                out=stage_result.out,
                lse=stage_result.lse,
            )
        )

    real_out, real_lse = merge_materialized_stage_records(
        records=records,
        query_token_ids=query_ids,
    )
    out, lse = merge_single_sink_branch(real_out, real_lse, attn_sink)
    return Dsv4AttentionForwardResult(
        out=out,
        lse=lse,
        real_out=real_out,
        real_lse=real_lse,
        query_token_ids=query_ids,
        attn_sink=attn_sink,
        scale=scale,
        stage_records=tuple(records),
    )


@torch.compiler.disable
def launch_exchanged_dsv4_attention_forward(
    *,
    stage_works: Sequence[Any],
    query_token_ids: Sequence[int],
    attn_sink: torch.Tensor,
    scale: float | None = None,
) -> Dsv4ExchangedAttentionForwardWork:
    works = tuple(stage_works)
    if not works:
        raise ValueError("at least one DSV4 exchanged attention stage is required")
    for position, work in enumerate(works):
        _validate_stage_exchange_work(work=work, position=position)
    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    return Dsv4ExchangedAttentionForwardWork(
        stage_works=works,
        query_token_ids=query_ids,
        attn_sink=attn_sink,
        scale=scale,
    )


def _prepare_csa_stage_plan_slot_launch(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    stage_plan_slots: Sequence[Dsv4StagePlanSlot],
    query_token_ids: Sequence[int],
    indexer_stage_plans: Sequence[Dsv4IndexerStagePlan] | None,
    stage_kv_peer_plans_by_slot: Sequence[Sequence[Dsv4StageKvExchangePeerPlan]] | None,
) -> tuple[
    int,
    tuple[Dsv4StagePlanSlot, ...],
    tuple[int, ...],
    tuple[Dsv4IndexerStagePlan, ...],
    tuple[tuple[Dsv4StageKvExchangePeerPlan, ...], ...] | None,
]:
    rank_int = int(rank)
    _validate_exchange_rank(
        rank=rank_int, rank_count=len(layout.entry_ids_by_owner_rank)
    )
    slots = _validate_stage_plan_slots(
        layout=layout,
        stage_plan_slots=stage_plan_slots,
    )
    query_ids = _normalize_output_ids(query_token_ids, name="query_token_ids")
    prepared_stage_kv_peer_plans = _optional_stage_kv_peer_plans_by_slot(
        slots=slots,
        stage_kv_peer_plans_by_slot=stage_kv_peer_plans_by_slot,
        name="CSA stage KV",
    )
    prepared_indexer_stage_plans = tuple(indexer_stage_plans or ())
    if prepared_indexer_stage_plans:
        if len(prepared_indexer_stage_plans) != len(slots):
            raise RuntimeError(
                "DSV4 prepared CSA indexer stage plan count must match slots: "
                f"{len(prepared_indexer_stage_plans)} vs {len(slots)}"
            )
        prepared_stage_ids = tuple(
            int(plan.stage_index) for plan in prepared_indexer_stage_plans
        )
        slot_stage_ids = tuple(int(slot.stage_index) for slot in slots)
        if prepared_stage_ids != slot_stage_ids:
            raise RuntimeError(
                "DSV4 prepared CSA indexer stage ids must match StagePlan slots"
            )
    else:
        prepared_indexer_stage_plans = tuple(
            build_dsv4_indexer_stage_plan_from_stage_plans(
                layout=layout,
                stage_plans_by_rank=slot.stage_plans_by_rank,
            )
            for slot in slots
        )
    return (
        rank_int,
        slots,
        query_ids,
        prepared_indexer_stage_plans,
        prepared_stage_kv_peer_plans,
    )


@torch.compiler.disable
def _launch_dsv4_csa_indexer_topk_from_prepared_slots(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    indexer_stage_plans: Sequence[Dsv4IndexerStagePlan],
    query_token_ids: Sequence[int],
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_kv: torch.Tensor,
    indexer_kv_entry_ids: Sequence[int],
    indexer_topk: int,
    group: Any,
    async_op: bool,
    indexer_score_scale: float,
    indexer_kv_peer_plans_by_stage: Sequence[Sequence[Dsv4IndexerKvExchangePeerPlan]]
    | None,
) -> Any:
    with torch.no_grad():
        return launch_dsv4_indexer_topk_from_stage_plans(
            layout=layout,
            rank=int(rank),
            indexer_stage_plans=indexer_stage_plans,
            query_token_ids=query_token_ids,
            indexer_q=indexer_q,
            indexer_weights=indexer_weights,
            indexer_kv=indexer_kv,
            indexer_kv_entry_ids=indexer_kv_entry_ids,
            topk=indexer_topk,
            group=group,
            async_op=async_op,
            score_scale=indexer_score_scale,
            indexer_kv_peer_plans_by_stage=indexer_kv_peer_plans_by_stage,
        )


@torch.compiler.disable
def _launch_dsv4_csa_attention_forward_from_topk_work(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    slots: Sequence[Dsv4StagePlanSlot],
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    compressed_kv: torch.Tensor,
    compressed_entry_ids: Sequence[int],
    topk_work: Any,
    stage_kv_peer_plans_by_slot: Sequence[Sequence[Dsv4StageKvExchangePeerPlan]] | None,
    attn_sink: torch.Tensor,
    group: Any,
    async_op: bool,
    scale: float | None,
    window_size: int,
    raw_list_size: int | None,
    compressed_list_size: int | None,
) -> Dsv4ExchangedAttentionForwardWork:
    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    deferred_stage_works = tuple(
        launch_dsv4_stage_kv_exchange_deferred_from_stage_plan_slot(
            layout=layout,
            rank=int(rank),
            stage_plan_slot=slot,
            query=query,
            query_token_ids=query_ids,
            raw_kv=raw_kv,
            raw_token_ids=raw_token_ids,
            compressed_kv=compressed_kv,
            compressed_entry_ids=compressed_entry_ids,
            group=group,
            async_op=async_op,
            peer_plans=stage_kv_peer_plans_by_slot[stage_position]
            if stage_kv_peer_plans_by_slot is not None
            else None,
        )
        for stage_position, slot in enumerate(slots)
    )
    topk_result = topk_work.wait_post_process()
    stage_works = tuple(
        stage_work.bind_stage_inputs(
            build_dsv4_stage_inputs_from_stage_plan(
                layout=layout,
                stage_plan=slot.stage_plans_by_rank[int(rank)],
                compression_kind=Dsv4CompressionKind.CSA,
                global_topk=topk_result.indices,
                topk_query_token_ids=query_ids,
                window_size=window_size,
                raw_list_size=raw_list_size,
                compressed_list_size=compressed_list_size,
                materialize_compressed_metadata=False,
            )
        )
        for stage_work, slot in zip(deferred_stage_works, slots, strict=True)
    )
    return launch_exchanged_dsv4_attention_forward(
        stage_works=stage_works,
        query_token_ids=query_ids,
        attn_sink=attn_sink,
        scale=scale,
    )


@torch.compiler.disable
def launch_dsv4_csa_attention_forward_from_stage_plan_slots(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    stage_plan_slots: Sequence[Dsv4StagePlanSlot],
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    compressed_kv: torch.Tensor,
    compressed_entry_ids: Sequence[int],
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_kv: torch.Tensor,
    indexer_kv_entry_ids: Sequence[int],
    indexer_topk: int,
    attn_sink: torch.Tensor,
    group: Any,
    async_op: bool,
    indexer_stage_plans: Sequence[Dsv4IndexerStagePlan] | None = None,
    indexer_kv_peer_plans_by_stage: Sequence[Sequence[Dsv4IndexerKvExchangePeerPlan]]
    | None = None,
    stage_kv_peer_plans_by_slot: Sequence[Sequence[Dsv4StageKvExchangePeerPlan]]
    | None = None,
    indexer_score_scale: float = 1.0,
    scale: float | None = None,
    window_size: int = 128,
    raw_list_size: int | None = None,
    compressed_list_size: int | None = None,
) -> Dsv4ExchangedAttentionForwardWork:
    (
        rank_int,
        slots,
        query_ids,
        prepared_indexer_stage_plans,
        prepared_stage_kv_peer_plans,
    ) = _prepare_csa_stage_plan_slot_launch(
        layout=layout,
        rank=rank,
        stage_plan_slots=stage_plan_slots,
        query_token_ids=query_token_ids,
        indexer_stage_plans=indexer_stage_plans,
        stage_kv_peer_plans_by_slot=stage_kv_peer_plans_by_slot,
    )
    topk_work = _launch_dsv4_csa_indexer_topk_from_prepared_slots(
        layout=layout,
        rank=rank_int,
        indexer_stage_plans=prepared_indexer_stage_plans,
        query_token_ids=query_ids,
        indexer_q=indexer_q,
        indexer_weights=indexer_weights,
        indexer_kv=indexer_kv,
        indexer_kv_entry_ids=indexer_kv_entry_ids,
        indexer_topk=indexer_topk,
        group=group,
        async_op=async_op,
        indexer_score_scale=indexer_score_scale,
        indexer_kv_peer_plans_by_stage=indexer_kv_peer_plans_by_stage,
    )
    return _launch_dsv4_csa_attention_forward_from_topk_work(
        layout=layout,
        rank=rank_int,
        slots=slots,
        query=query,
        query_token_ids=query_ids,
        raw_kv=raw_kv,
        raw_token_ids=raw_token_ids,
        compressed_kv=compressed_kv,
        compressed_entry_ids=compressed_entry_ids,
        topk_work=topk_work,
        stage_kv_peer_plans_by_slot=prepared_stage_kv_peer_plans,
        attn_sink=attn_sink,
        group=group,
        async_op=async_op,
        scale=scale,
        window_size=window_size,
        raw_list_size=raw_list_size,
        compressed_list_size=compressed_list_size,
    )


@torch.compiler.disable
def launch_dsv4_hca_attention_forward_from_stage_plan_slots(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    stage_plan_slots: Sequence[Dsv4StagePlanSlot],
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    compressed_kv: torch.Tensor,
    compressed_entry_ids: Sequence[int],
    attn_sink: torch.Tensor,
    group: Any,
    async_op: bool,
    stage_kv_peer_plans_by_slot: Sequence[Sequence[Dsv4StageKvExchangePeerPlan]]
    | None = None,
    scale: float | None = None,
    window_size: int = 128,
    raw_list_size: int | None = None,
    compressed_list_size: int | None = None,
) -> Dsv4ExchangedAttentionForwardWork:
    rank_int = int(rank)
    _validate_exchange_rank(
        rank=rank_int, rank_count=len(layout.entry_ids_by_owner_rank)
    )
    slots = _validate_stage_plan_slots(
        layout=layout,
        stage_plan_slots=stage_plan_slots,
    )
    query_ids = _normalize_output_ids(query_token_ids, name="query_token_ids")
    prepared_stage_kv_peer_plans = _optional_stage_kv_peer_plans_by_slot(
        slots=slots,
        stage_kv_peer_plans_by_slot=stage_kv_peer_plans_by_slot,
        name="HCA stage KV",
    )
    stage_works = tuple(
        _launch_dsv4_stage_from_slot(
            stage_kv_peer_plans=prepared_stage_kv_peer_plans[stage_position]
            if prepared_stage_kv_peer_plans is not None
            else None,
            layout=layout,
            rank=rank_int,
            slot=slot,
            compression_kind=Dsv4CompressionKind.HCA,
            query=query,
            query_token_ids=query_ids,
            raw_kv=raw_kv,
            raw_token_ids=raw_token_ids,
            compressed_kv=compressed_kv,
            compressed_entry_ids=compressed_entry_ids,
            group=group,
            async_op=async_op,
            global_topk=None,
            topk_query_token_ids=None,
            window_size=window_size,
            raw_list_size=raw_list_size,
            compressed_list_size=compressed_list_size,
        )
        for stage_position, slot in enumerate(slots)
    )
    return launch_exchanged_dsv4_attention_forward(
        stage_works=stage_works,
        query_token_ids=query_ids,
        attn_sink=attn_sink,
        scale=scale,
    )


@torch.compiler.disable
def launch_dsv4_csa_projected_compression_forward(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    main_projected_kv: torch.Tensor,
    main_projected_gate: torch.Tensor,
    main_positional_bias: torch.Tensor,
    main_token_ids: Sequence[int],
    indexer_projected_kv: torch.Tensor,
    indexer_projected_gate: torch.Tensor,
    indexer_positional_bias: torch.Tensor,
    indexer_token_ids: Sequence[int],
    group: Any,
    async_op: bool,
) -> Dsv4ProjectedCompressionForwardWork:
    if layout.spec.kind != Dsv4CompressionKind.CSA:
        raise RuntimeError(
            f"DSV4 CSA compression launch received {layout.spec.kind} layout"
        )
    rank_int = int(rank)
    _validate_exchange_rank(
        rank=rank_int,
        rank_count=len(layout.entry_ids_by_owner_rank),
    )
    with torch.no_grad():
        main_work = launch_dsv4_compressed_kv_forward(
            layout=layout,
            rank=rank_int,
            projected_kv=main_projected_kv,
            projected_gate=main_projected_gate,
            positional_bias=main_positional_bias,
            token_ids=main_token_ids,
            group=group,
            async_op=async_op,
        )
        indexer_work = launch_dsv4_compressed_kv_forward(
            layout=layout,
            rank=rank_int,
            projected_kv=indexer_projected_kv,
            projected_gate=indexer_projected_gate,
            positional_bias=indexer_positional_bias,
            token_ids=indexer_token_ids,
            group=group,
            async_op=async_op,
        )
    return Dsv4ProjectedCompressionForwardWork(
        compression_kind=Dsv4CompressionKind.CSA,
        layout=layout,
        rank=rank_int,
        main_compression_work=main_work,
        indexer_compression_work=indexer_work,
    )


@torch.compiler.disable
def launch_dsv4_hca_projected_compression_forward(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    positional_bias: torch.Tensor,
    token_ids: Sequence[int],
    group: Any,
    async_op: bool,
) -> Dsv4ProjectedCompressionForwardWork:
    if layout.spec.kind != Dsv4CompressionKind.HCA:
        raise RuntimeError(
            f"DSV4 HCA compression launch received {layout.spec.kind} layout"
        )
    rank_int = int(rank)
    _validate_exchange_rank(
        rank=rank_int,
        rank_count=len(layout.entry_ids_by_owner_rank),
    )
    with torch.no_grad():
        compressed_work = launch_dsv4_compressed_kv_forward(
            layout=layout,
            rank=rank_int,
            projected_kv=projected_kv,
            projected_gate=projected_gate,
            positional_bias=positional_bias,
            token_ids=token_ids,
            group=group,
            async_op=async_op,
        )
    return Dsv4ProjectedCompressionForwardWork(
        compression_kind=Dsv4CompressionKind.HCA,
        layout=layout,
        rank=rank_int,
        main_compression_work=compressed_work,
    )


@torch.compiler.disable
def launch_dsv4_csa_projected_attention_forward_from_compression_work(
    *,
    compression_work: Dsv4ProjectedCompressionForwardWork,
    stage_plan_slots: Sequence[Dsv4StagePlanSlot],
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_topk: int,
    attn_sink: torch.Tensor,
    group: Any,
    async_op: bool,
    indexer_stage_plans: Sequence[Dsv4IndexerStagePlan] | None = None,
    indexer_kv_peer_plans_by_stage: Sequence[Sequence[Dsv4IndexerKvExchangePeerPlan]]
    | None = None,
    stage_kv_peer_plans_by_slot: Sequence[Sequence[Dsv4StageKvExchangePeerPlan]]
    | None = None,
    indexer_score_scale: float = 1.0,
    scale: float | None = None,
    window_size: int = 128,
    raw_list_size: int | None = None,
    compressed_list_size: int | None = None,
) -> Dsv4ProjectedAttentionForwardWork:
    compression = _validate_projected_compression_work(
        compression_work=compression_work,
        kind=Dsv4CompressionKind.CSA,
    )
    slots = _validate_stage_plan_slots(
        layout=compression.layout,
        stage_plan_slots=stage_plan_slots,
    )
    query_ids = _normalize_output_ids(query_token_ids, name="query_token_ids")
    raw_ids = _normalize_output_ids(raw_token_ids, name="raw_token_ids")
    return Dsv4ProjectedAttentionForwardWork(
        compression_kind=Dsv4CompressionKind.CSA,
        layout=compression.layout,
        rank=int(compression.rank),
        stage_plan_slots=slots,
        query=query,
        query_token_ids=query_ids,
        raw_kv=raw_kv,
        raw_token_ids=raw_ids,
        main_compression_work=compression.main_compression_work,
        indexer_compression_work=compression.indexer_compression_work,
        indexer_q=indexer_q,
        indexer_weights=indexer_weights,
        indexer_topk=int(indexer_topk),
        indexer_stage_plans=tuple(indexer_stage_plans)
        if indexer_stage_plans is not None
        else None,
        indexer_kv_peer_plans_by_stage=tuple(
            tuple(peer_plans) for peer_plans in indexer_kv_peer_plans_by_stage
        )
        if indexer_kv_peer_plans_by_stage is not None
        else None,
        stage_kv_peer_plans_by_slot=tuple(
            tuple(peer_plans) for peer_plans in stage_kv_peer_plans_by_slot
        )
        if stage_kv_peer_plans_by_slot is not None
        else None,
        indexer_score_scale=float(indexer_score_scale),
        attn_sink=attn_sink,
        group=group,
        async_op=async_op,
        scale=scale,
        window_size=int(window_size),
        raw_list_size=raw_list_size,
        compressed_list_size=compressed_list_size,
    )


@torch.compiler.disable
def launch_dsv4_hca_projected_attention_forward_from_compression_work(
    *,
    compression_work: Dsv4ProjectedCompressionForwardWork,
    stage_plan_slots: Sequence[Dsv4StagePlanSlot],
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    attn_sink: torch.Tensor,
    group: Any,
    async_op: bool,
    stage_kv_peer_plans_by_slot: Sequence[Sequence[Dsv4StageKvExchangePeerPlan]]
    | None = None,
    scale: float | None = None,
    window_size: int = 128,
    raw_list_size: int | None = None,
    compressed_list_size: int | None = None,
) -> Dsv4ProjectedAttentionForwardWork:
    compression = _validate_projected_compression_work(
        compression_work=compression_work,
        kind=Dsv4CompressionKind.HCA,
    )
    slots = _validate_stage_plan_slots(
        layout=compression.layout,
        stage_plan_slots=stage_plan_slots,
    )
    query_ids = _normalize_output_ids(query_token_ids, name="query_token_ids")
    raw_ids = _normalize_output_ids(raw_token_ids, name="raw_token_ids")
    return Dsv4ProjectedAttentionForwardWork(
        compression_kind=Dsv4CompressionKind.HCA,
        layout=compression.layout,
        rank=int(compression.rank),
        stage_plan_slots=slots,
        query=query,
        query_token_ids=query_ids,
        raw_kv=raw_kv,
        raw_token_ids=raw_ids,
        main_compression_work=compression.main_compression_work,
        stage_kv_peer_plans_by_slot=tuple(
            tuple(peer_plans) for peer_plans in stage_kv_peer_plans_by_slot
        )
        if stage_kv_peer_plans_by_slot is not None
        else None,
        attn_sink=attn_sink,
        group=group,
        async_op=async_op,
        scale=scale,
        window_size=int(window_size),
        raw_list_size=raw_list_size,
        compressed_list_size=compressed_list_size,
    )


@torch.compiler.disable
def launch_dsv4_csa_projected_attention_forward_from_stage_plan_slots(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    stage_plan_slots: Sequence[Dsv4StagePlanSlot],
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    main_projected_kv: torch.Tensor,
    main_projected_gate: torch.Tensor,
    main_positional_bias: torch.Tensor,
    main_token_ids: Sequence[int],
    indexer_projected_kv: torch.Tensor,
    indexer_projected_gate: torch.Tensor,
    indexer_positional_bias: torch.Tensor,
    indexer_token_ids: Sequence[int],
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_topk: int,
    attn_sink: torch.Tensor,
    group: Any,
    async_op: bool,
    indexer_stage_plans: Sequence[Dsv4IndexerStagePlan] | None = None,
    indexer_kv_peer_plans_by_stage: Sequence[Sequence[Dsv4IndexerKvExchangePeerPlan]]
    | None = None,
    stage_kv_peer_plans_by_slot: Sequence[Sequence[Dsv4StageKvExchangePeerPlan]]
    | None = None,
    indexer_score_scale: float = 1.0,
    scale: float | None = None,
    window_size: int = 128,
    raw_list_size: int | None = None,
    compressed_list_size: int | None = None,
) -> Dsv4ProjectedAttentionForwardWork:
    compression_work = launch_dsv4_csa_projected_compression_forward(
        layout=layout,
        rank=rank,
        main_projected_kv=main_projected_kv,
        main_projected_gate=main_projected_gate,
        main_positional_bias=main_positional_bias,
        main_token_ids=main_token_ids,
        indexer_projected_kv=indexer_projected_kv,
        indexer_projected_gate=indexer_projected_gate,
        indexer_positional_bias=indexer_positional_bias,
        indexer_token_ids=indexer_token_ids,
        group=group,
        async_op=async_op,
    )
    return launch_dsv4_csa_projected_attention_forward_from_compression_work(
        compression_work=compression_work,
        stage_plan_slots=stage_plan_slots,
        query=query,
        query_token_ids=query_token_ids,
        raw_kv=raw_kv,
        raw_token_ids=raw_token_ids,
        indexer_q=indexer_q,
        indexer_weights=indexer_weights,
        indexer_topk=indexer_topk,
        attn_sink=attn_sink,
        group=group,
        async_op=async_op,
        indexer_stage_plans=indexer_stage_plans,
        indexer_kv_peer_plans_by_stage=indexer_kv_peer_plans_by_stage,
        stage_kv_peer_plans_by_slot=stage_kv_peer_plans_by_slot,
        indexer_score_scale=indexer_score_scale,
        scale=scale,
        window_size=window_size,
        raw_list_size=raw_list_size,
        compressed_list_size=compressed_list_size,
    )


@torch.compiler.disable
def launch_dsv4_csa_projected_compression_forward_from_context_parallel_state(
    *,
    context_state: Dsv4ContextParallelState,
    main_projected_kv: torch.Tensor,
    main_projected_gate: torch.Tensor,
    main_positional_bias: torch.Tensor,
    main_token_ids: Sequence[int],
    indexer_projected_kv: torch.Tensor,
    indexer_projected_gate: torch.Tensor,
    indexer_positional_bias: torch.Tensor,
    indexer_token_ids: Sequence[int],
    async_op: bool,
) -> Dsv4ProjectedCompressionForwardWork:
    return launch_dsv4_csa_projected_compression_forward(
        layout=_require_prepared_layout(
            context_state=context_state,
            kind=Dsv4CompressionKind.CSA,
        ),
        rank=_prepared_rank(context_state),
        main_projected_kv=main_projected_kv,
        main_projected_gate=main_projected_gate,
        main_positional_bias=main_positional_bias,
        main_token_ids=main_token_ids,
        indexer_projected_kv=indexer_projected_kv,
        indexer_projected_gate=indexer_projected_gate,
        indexer_positional_bias=indexer_positional_bias,
        indexer_token_ids=indexer_token_ids,
        group=context_state.cp_state.cp_group,
        async_op=async_op,
    )


@torch.compiler.disable
def launch_dsv4_csa_projected_attention_forward_from_context_parallel_state_and_compression_work(
    *,
    context_state: Dsv4ContextParallelState,
    compression_work: Dsv4ProjectedCompressionForwardWork,
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_topk: int,
    attn_sink: torch.Tensor,
    async_op: bool,
    indexer_score_scale: float = 1.0,
    scale: float | None = None,
    window_size: int = 128,
    raw_list_size: int | None = None,
    compressed_list_size: int | None = None,
) -> Dsv4ProjectedAttentionForwardWork:
    plan = context_state.dsv4_plan
    layout = _require_prepared_layout(
        context_state=context_state,
        kind=Dsv4CompressionKind.CSA,
    )
    _validate_projected_compression_work(
        compression_work=compression_work,
        kind=Dsv4CompressionKind.CSA,
        layout=layout,
        rank=_prepared_rank(context_state),
    )
    indexer_stage_plans = tuple(plan.csa_indexer_stage_plans)
    if not indexer_stage_plans:
        raise RuntimeError(
            "DSV4 prepared CSA context state is missing indexer StagePlans"
        )
    return launch_dsv4_csa_projected_attention_forward_from_compression_work(
        compression_work=compression_work,
        stage_plan_slots=_require_prepared_stage_slots(context_state),
        query=query,
        query_token_ids=query_token_ids,
        raw_kv=raw_kv,
        raw_token_ids=raw_token_ids,
        indexer_q=indexer_q,
        indexer_weights=indexer_weights,
        indexer_topk=indexer_topk,
        attn_sink=attn_sink,
        group=context_state.cp_state.cp_group,
        async_op=async_op,
        indexer_stage_plans=indexer_stage_plans,
        indexer_kv_peer_plans_by_stage=plan.csa_indexer_kv_peer_plans_by_stage or None,
        stage_kv_peer_plans_by_slot=plan.csa_stage_kv_peer_plans_by_slot or None,
        indexer_score_scale=indexer_score_scale,
        scale=scale,
        window_size=window_size,
        raw_list_size=raw_list_size,
        compressed_list_size=compressed_list_size,
    )


@torch.compiler.disable
def launch_dsv4_hca_projected_attention_forward_from_stage_plan_slots(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    stage_plan_slots: Sequence[Dsv4StagePlanSlot],
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    positional_bias: torch.Tensor,
    token_ids: Sequence[int],
    attn_sink: torch.Tensor,
    group: Any,
    async_op: bool,
    stage_kv_peer_plans_by_slot: Sequence[Sequence[Dsv4StageKvExchangePeerPlan]]
    | None = None,
    scale: float | None = None,
    window_size: int = 128,
    raw_list_size: int | None = None,
    compressed_list_size: int | None = None,
) -> Dsv4ProjectedAttentionForwardWork:
    compression_work = launch_dsv4_hca_projected_compression_forward(
        layout=layout,
        rank=rank,
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        positional_bias=positional_bias,
        token_ids=token_ids,
        group=group,
        async_op=async_op,
    )
    return launch_dsv4_hca_projected_attention_forward_from_compression_work(
        compression_work=compression_work,
        stage_plan_slots=stage_plan_slots,
        query=query,
        query_token_ids=query_token_ids,
        raw_kv=raw_kv,
        raw_token_ids=raw_token_ids,
        attn_sink=attn_sink,
        group=group,
        async_op=async_op,
        stage_kv_peer_plans_by_slot=stage_kv_peer_plans_by_slot,
        scale=scale,
        window_size=window_size,
        raw_list_size=raw_list_size,
        compressed_list_size=compressed_list_size,
    )


@torch.compiler.disable
def launch_dsv4_hca_projected_compression_forward_from_context_parallel_state(
    *,
    context_state: Dsv4ContextParallelState,
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    positional_bias: torch.Tensor,
    token_ids: Sequence[int],
    async_op: bool,
) -> Dsv4ProjectedCompressionForwardWork:
    return launch_dsv4_hca_projected_compression_forward(
        layout=_require_prepared_layout(
            context_state=context_state,
            kind=Dsv4CompressionKind.HCA,
        ),
        rank=_prepared_rank(context_state),
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        positional_bias=positional_bias,
        token_ids=token_ids,
        group=context_state.cp_state.cp_group,
        async_op=async_op,
    )


@torch.compiler.disable
def launch_dsv4_hca_projected_attention_forward_from_context_parallel_state_and_compression_work(
    *,
    context_state: Dsv4ContextParallelState,
    compression_work: Dsv4ProjectedCompressionForwardWork,
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    attn_sink: torch.Tensor,
    async_op: bool,
    scale: float | None = None,
    window_size: int = 128,
    raw_list_size: int | None = None,
    compressed_list_size: int | None = None,
) -> Dsv4ProjectedAttentionForwardWork:
    layout = _require_prepared_layout(
        context_state=context_state,
        kind=Dsv4CompressionKind.HCA,
    )
    _validate_projected_compression_work(
        compression_work=compression_work,
        kind=Dsv4CompressionKind.HCA,
        layout=layout,
        rank=_prepared_rank(context_state),
    )
    return launch_dsv4_hca_projected_attention_forward_from_compression_work(
        compression_work=compression_work,
        stage_plan_slots=_require_prepared_stage_slots(context_state),
        query=query,
        query_token_ids=query_token_ids,
        raw_kv=raw_kv,
        raw_token_ids=raw_token_ids,
        attn_sink=attn_sink,
        group=context_state.cp_state.cp_group,
        async_op=async_op,
        stage_kv_peer_plans_by_slot=context_state.dsv4_plan.hca_stage_kv_peer_plans_by_slot
        or None,
        scale=scale,
        window_size=window_size,
        raw_list_size=raw_list_size,
        compressed_list_size=compressed_list_size,
    )


@torch.compiler.disable
def launch_dsv4_projected_attention_backward_from_stage_plan_slots(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    stage_plan_slots: Sequence[Dsv4StagePlanSlot],
    forward_result: Dsv4ProjectedAttentionForwardResult,
    grad_out: torch.Tensor,
    group: Any,
    async_op: bool,
    owned_query_token_ids: Sequence[int] | None = None,
    owned_raw_token_ids: Sequence[int] | None = None,
    owned_compressed_entry_ids: Sequence[int] | None = None,
    backward_plan: Dsv4AttentionBackwardPlan | None = None,
) -> Dsv4ProjectedAttentionBackwardWork:
    if forward_result.compression_kind != layout.spec.kind:
        raise RuntimeError(
            "DSV4 projected forward kind does not match layout kind: "
            f"{forward_result.compression_kind} vs {layout.spec.kind}"
        )
    attention_work = launch_dsv4_attention_backward_from_stage_plan_slots(
        layout=layout,
        rank=int(rank),
        stage_plan_slots=stage_plan_slots,
        forward_result=forward_result.attention,
        grad_out=grad_out,
        group=group,
        async_op=async_op,
        owned_query_token_ids=owned_query_token_ids,
        owned_raw_token_ids=owned_raw_token_ids,
        owned_compressed_entry_ids=owned_compressed_entry_ids,
        backward_plan=backward_plan,
    )
    return Dsv4ProjectedAttentionBackwardWork(
        attention_work=attention_work,
        forward_result=forward_result,
        group=group,
        async_op=async_op,
    )


@torch.compiler.disable
def launch_dsv4_projected_attention_backward_from_context_parallel_state(
    *,
    context_state: Dsv4ContextParallelState,
    forward_result: Dsv4ProjectedAttentionForwardResult,
    grad_out: torch.Tensor,
    async_op: bool,
    owned_query_token_ids: Sequence[int] | None = None,
    owned_raw_token_ids: Sequence[int] | None = None,
    owned_compressed_entry_ids: Sequence[int] | None = None,
) -> Dsv4ProjectedAttentionBackwardWork:
    return launch_dsv4_projected_attention_backward_from_stage_plan_slots(
        layout=_require_prepared_layout(
            context_state=context_state,
            kind=forward_result.compression_kind,
        ),
        rank=_prepared_rank(context_state),
        stage_plan_slots=_require_prepared_stage_slots(context_state),
        forward_result=forward_result,
        grad_out=grad_out,
        group=context_state.cp_state.cp_group,
        async_op=async_op,
        backward_plan=_prepared_attention_backward_plan(
            context_state=context_state,
            kind=forward_result.compression_kind,
        ),
        owned_query_token_ids=owned_query_token_ids,
        owned_raw_token_ids=owned_raw_token_ids,
        owned_compressed_entry_ids=owned_compressed_entry_ids,
    )


@torch.compiler.disable
def build_dsv4_attention_backward_plan_from_stage_plan_slots(
    *,
    layout: Dsv4CompressedLayout,
    stage_plan_slots: Sequence[Dsv4StagePlanSlot],
    stage_kv_peer_plans_by_slot: Sequence[Sequence[Dsv4StageKvExchangePeerPlan]]
    | None = None,
    local_rank: int | None = None,
) -> Dsv4AttentionBackwardPlan:
    peer_plans_by_layout = (
        None if stage_kv_peer_plans_by_slot is None else (stage_kv_peer_plans_by_slot,)
    )
    return build_dsv4_attention_backward_plans_from_stage_plan_slots(
        layouts=(layout,),
        stage_plan_slots=stage_plan_slots,
        stage_kv_peer_plans_by_layout=peer_plans_by_layout,
        local_rank=local_rank,
    )[0]


@torch.compiler.disable
def build_dsv4_attention_backward_plans_from_stage_plan_slots(
    *,
    layouts: Sequence[Dsv4CompressedLayout],
    stage_plan_slots: Sequence[Dsv4StagePlanSlot],
    stage_kv_peer_plans_by_layout: Sequence[
        Sequence[Sequence[Dsv4StageKvExchangePeerPlan]]
    ]
    | None = None,
    local_rank: int | None = None,
) -> tuple[Dsv4AttentionBackwardPlan, ...]:
    layout_tuple = tuple(layouts)
    if not layout_tuple:
        return ()
    slots = _validate_stage_plan_slots(
        layout=layout_tuple[0],
        stage_plan_slots=stage_plan_slots,
    )
    rank_count = len(layout_tuple[0].entry_ids_by_owner_rank)
    for layout in layout_tuple[1:]:
        _validate_stage_plan_slots(layout=layout, stage_plan_slots=slots)
        if len(layout.entry_ids_by_owner_rank) != rank_count:
            raise RuntimeError(
                "DSV4 backward plan layouts must share rank count, got "
                f"{len(layout.entry_ids_by_owner_rank)} vs {rank_count}"
            )

    if stage_kv_peer_plans_by_layout is None:
        first_slot_plans = (
            build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans_for_layouts(
                layouts=layout_tuple,
                stage_plans_by_rank=slots[0].stage_plans_by_rank,
            )
        )
        stage_kv_peer_plans_by_layout = tuple(
            (layout_plans,) for layout_plans in first_slot_plans
        )
        for slot in slots[1:]:
            slot_plans = (
                build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans_for_layouts(
                    layouts=layout_tuple,
                    stage_plans_by_rank=slot.stage_plans_by_rank,
                )
            )
            stage_kv_peer_plans_by_layout = tuple(
                prior + (current,)
                for prior, current in zip(
                    stage_kv_peer_plans_by_layout,
                    slot_plans,
                    strict=True,
                )
            )
    else:
        stage_kv_peer_plans_by_layout = tuple(
            tuple(tuple(rank_plans) for rank_plans in layout_plans)
            for layout_plans in stage_kv_peer_plans_by_layout
        )
        if len(stage_kv_peer_plans_by_layout) != len(layout_tuple):
            raise RuntimeError(
                "DSV4 backward peer-plan layout count must match layouts: "
                f"{len(stage_kv_peer_plans_by_layout)} vs {len(layout_tuple)}"
            )

    stage_indices = _slot_stage_indices(slots)
    if local_rank is None:
        if rank_count != 1:
            raise RuntimeError("DSV4 backward planning requires local_rank")
        local_rank = 0
    local_rank_int = int(local_rank)
    _validate_exchange_rank(rank=local_rank_int, rank_count=rank_count)
    query_ids, query_owners = _stage_plan_slot_query_id_space_for_rank(
        slots=slots,
        rank=local_rank_int,
    )
    raw_ids, raw_owners, raw_recv, raw_owned = _id_space_from_stage_peer_plans_for_rank(
        peer_plans_by_slot=stage_kv_peer_plans_by_layout[0],
        recv_attr="recv_raw_token_ids_by_peer",
        rank=local_rank_int,
        rank_count=rank_count,
    )
    common_rank_part = _common_backward_rank_part(
        query_token_ids=query_ids,
        query_owner_ranks=query_owners,
        raw_token_ids=raw_ids,
        raw_owner_ranks=raw_owners,
        recv_raw_token_ids_by_peer=raw_recv,
        owned_raw_token_ids=raw_owned,
        rank=local_rank_int,
        rank_count=rank_count,
    )
    return tuple(
        _build_local_attention_backward_plan_for_layout(
            layout=layout,
            stage_indices=stage_indices,
            local_rank=local_rank_int,
            common_rank_part=common_rank_part,
            compressed_peer_plans_by_slot=compressed_peer_plans_by_slot,
            rank_count=rank_count,
        )
        for layout, compressed_peer_plans_by_slot in zip(
            layout_tuple,
            stage_kv_peer_plans_by_layout,
            strict=True,
        )
    )


@torch.compiler.disable
def launch_dsv4_attention_backward_from_stage_plan_slots(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    stage_plan_slots: Sequence[Dsv4StagePlanSlot],
    forward_result: Dsv4AttentionForwardResult,
    grad_out: torch.Tensor,
    group: Any,
    async_op: bool,
    owned_query_token_ids: Sequence[int] | None = None,
    owned_raw_token_ids: Sequence[int] | None = None,
    owned_compressed_entry_ids: Sequence[int] | None = None,
    backward_plan: Dsv4AttentionBackwardPlan | None = None,
) -> Dsv4ExchangedAttentionBackwardWork:
    rank_int = int(rank)
    rank_count = len(layout.entry_ids_by_owner_rank)
    _validate_exchange_rank(rank=rank_int, rank_count=rank_count)
    slots = _validate_stage_plan_slots(
        layout=layout,
        stage_plan_slots=stage_plan_slots,
    )
    plan = _attention_backward_plan_or_build(
        layout=layout,
        slots=slots,
        backward_plan=backward_plan,
        rank=rank_int,
    )
    if plan.local_rank != rank_int:
        raise RuntimeError(
            "DSV4 local backward plan rank does not match launch rank: "
            f"{plan.local_rank} vs {rank_int}"
        )
    rank_plan = plan.local_rank_plan
    if tuple(forward_result.query_token_ids) != rank_plan.query_token_ids:
        raise RuntimeError(
            "DSV4 forward_result query ids do not match StagePlan-slot query ids: "
            f"{tuple(forward_result.query_token_ids)} vs {rank_plan.query_token_ids}"
        )
    return launch_exchanged_dsv4_attention_backward(
        forward_result=forward_result,
        grad_out=grad_out,
        query_token_ids=rank_plan.query_token_ids,
        raw_token_ids=rank_plan.raw_token_ids,
        compressed_entry_ids=rank_plan.compressed_entry_ids,
        query_owner_ranks=rank_plan.query_owner_ranks,
        raw_owner_ranks=rank_plan.raw_owner_ranks,
        compressed_owner_ranks=rank_plan.compressed_owner_ranks,
        recv_query_token_ids_by_peer=rank_plan.recv_query_token_ids_by_peer,
        recv_raw_token_ids_by_peer=rank_plan.recv_raw_token_ids_by_peer,
        recv_compressed_entry_ids_by_peer=rank_plan.recv_compressed_entry_ids_by_peer,
        owned_query_token_ids=owned_query_token_ids
        if owned_query_token_ids is not None
        else rank_plan.owned_query_token_ids,
        owned_raw_token_ids=owned_raw_token_ids
        if owned_raw_token_ids is not None
        else rank_plan.owned_raw_token_ids,
        owned_compressed_entry_ids=owned_compressed_entry_ids
        if owned_compressed_entry_ids is not None
        else rank_plan.owned_compressed_entry_ids,
        rank=rank_int,
        rank_count=rank_count,
        group=group,
        async_op=async_op,
    )


@torch.compiler.disable
def launch_exchanged_dsv4_attention_backward(
    *,
    forward_result: Dsv4AttentionForwardResult,
    grad_out: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_token_ids: Sequence[int],
    compressed_entry_ids: Sequence[int],
    query_owner_ranks: Sequence[int],
    raw_owner_ranks: Sequence[int],
    compressed_owner_ranks: Sequence[int],
    recv_query_token_ids_by_peer: Sequence[Sequence[int]],
    recv_raw_token_ids_by_peer: Sequence[Sequence[int]],
    recv_compressed_entry_ids_by_peer: Sequence[Sequence[int]],
    owned_query_token_ids: Sequence[int],
    owned_raw_token_ids: Sequence[int],
    owned_compressed_entry_ids: Sequence[int],
    rank: int,
    rank_count: int,
    group: Any,
    async_op: bool,
) -> Dsv4ExchangedAttentionBackwardWork:
    owned_query_ids = _normalize_output_ids(
        owned_query_token_ids,
        name="owned_query_token_ids",
    )
    owned_raw_ids = _normalize_output_ids(
        owned_raw_token_ids,
        name="owned_raw_token_ids",
    )
    owned_compressed_ids = _normalize_output_ids(
        owned_compressed_entry_ids,
        name="owned_compressed_entry_ids",
    )
    replay_result = replay_materialized_dsv4_attention_backward(
        forward_result=forward_result,
        grad_out=grad_out,
    )
    local_gradients = accumulate_materialized_dsv4_attention_backward(
        replay_result=replay_result,
        query_token_ids=query_token_ids,
        raw_token_ids=raw_token_ids,
        compressed_entry_ids=compressed_entry_ids,
    )
    owner_work = launch_dsv4_gradient_owner_bucket_exchange(
        gradients=local_gradients,
        query_owner_ranks=query_owner_ranks,
        raw_owner_ranks=raw_owner_ranks,
        compressed_owner_ranks=compressed_owner_ranks,
        recv_query_token_ids_by_peer=recv_query_token_ids_by_peer,
        recv_raw_token_ids_by_peer=recv_raw_token_ids_by_peer,
        recv_compressed_entry_ids_by_peer=recv_compressed_entry_ids_by_peer,
        rank=rank,
        rank_count=rank_count,
        group=group,
        async_op=async_op,
    )
    sink_work = launch_dsv4_attn_sink_gradient_reduce(
        d_attn_sink=local_gradients.d_attn_sink,
        rank=rank,
        rank_count=rank_count,
        group=group,
        async_op=async_op,
    )
    return Dsv4ExchangedAttentionBackwardWork(
        local_gradients=local_gradients,
        owner_work=owner_work,
        sink_work=sink_work,
        owned_query_token_ids=owned_query_ids,
        owned_raw_token_ids=owned_raw_ids,
        owned_compressed_entry_ids=owned_compressed_ids,
    )


@torch.compiler.disable
def launch_dsv4_attn_sink_gradient_reduce(
    *,
    d_attn_sink: torch.Tensor,
    rank: int,
    rank_count: int,
    group: Any,
    async_op: bool,
) -> Dsv4SinkGradientReduceWork:
    rank = int(rank)
    rank_count = int(rank_count)
    _validate_exchange_rank(rank=rank, rank_count=rank_count)
    if d_attn_sink.ndim != 1:
        raise ValueError(
            f"DSV4 d_attn_sink must be a per-head vector, got {d_attn_sink.shape}"
        )
    reduced = d_attn_sink.contiguous().clone()
    handle = None
    if rank_count > 1:
        handle = _DIST.all_reduce(reduced, group=group, async_op=async_op)
    return Dsv4SinkGradientReduceWork(d_attn_sink=reduced, handle=handle)


def _align_compressed_grad_to_forward(
    *,
    forward_result: Dsv4CompressedKvForwardResult,
    attention_gradients: Dsv4AttentionGradientResult,
) -> torch.Tensor:
    target_ids = tuple(
        int(entry_id) for entry_id in forward_result.compressed_entry_ids
    )
    source_ids = tuple(
        int(entry_id) for entry_id in attention_gradients.compressed_entry_ids
    )
    target_index = _row_by_id(ids=target_ids, name="forward compressed_entry_ids")
    source_index = _row_by_id(ids=source_ids, name="attention compressed_entry_ids")
    missing = tuple(entry_id for entry_id in source_ids if entry_id not in target_index)
    if missing:
        raise RuntimeError(
            "DSV4 attention produced compressed gradients for entries not owned "
            f"by compressor forward: {missing}"
        )

    target = forward_result.compressed_kv
    source = attention_gradients.dcompressed_kv
    if target.ndim == source.ndim:
        source_for_target = source
    elif target.ndim == 2 and source.ndim == 3:
        source_for_target = source.sum(dim=0)
    else:
        raise RuntimeError(
            "DSV4 compressed gradient rank mismatch: "
            f"target={tuple(target.shape)}, source={tuple(source.shape)}"
        )
    if int(source_for_target.shape[-2]) != len(source_ids):
        raise RuntimeError(
            "DSV4 attention compressed-gradient row count mismatch: "
            f"{int(source_for_target.shape[-2])} vs {len(source_ids)}"
        )
    if int(source_for_target.shape[-1]) != int(target.shape[-1]):
        raise RuntimeError(
            "DSV4 attention compressed-gradient dim mismatch: "
            f"{int(source_for_target.shape[-1])} vs {int(target.shape[-1])}"
        )

    aligned = source_for_target.new_zeros(tuple(target.shape))
    if not source_ids:
        return aligned
    source_positions = []
    target_positions = []
    for entry_id in source_ids:
        source_positions.append(source_index[entry_id])
        target_positions.append(target_index[entry_id])
    source_index_tensor = torch.tensor(
        source_positions,
        device=source_for_target.device,
        dtype=torch.long,
    )
    target_index_tensor = torch.tensor(
        target_positions,
        device=aligned.device,
        dtype=torch.long,
    )
    token_dim = 0 if aligned.ndim == 2 else 1
    selected = source_for_target.index_select(token_dim, source_index_tensor)
    aligned.index_add_(token_dim, target_index_tensor, selected)
    return aligned


def _validate_stage_exchange_work(*, work: Any, position: int) -> None:
    if not callable(getattr(work, "wait", None)):
        raise ValueError(f"DSV4 stage exchange work {position} is missing wait()")
    if not callable(getattr(work, "wait_post_process", None)):
        raise ValueError(
            f"DSV4 stage exchange work {position} is missing wait_post_process()"
        )


def _materialized_stage_from_work(*, work: Any, position: int) -> Dsv4MaterializedStage:
    stage = work.wait_post_process()
    if not isinstance(stage, Dsv4MaterializedStage):
        raise TypeError(
            f"DSV4 stage exchange work {position} returned {type(stage)!r}, "
            "expected Dsv4MaterializedStage"
        )
    return stage


def _owner_buckets_from_work(work: Any) -> tuple[Dsv4GradientOwnerBucket, ...]:
    if not callable(getattr(work, "wait", None)):
        raise ValueError("DSV4 owner-gradient exchange work is missing wait()")
    if not callable(getattr(work, "wait_post_process", None)):
        raise ValueError(
            "DSV4 owner-gradient exchange work is missing wait_post_process()"
        )
    buckets = tuple(work.wait_post_process())
    for position, bucket in enumerate(buckets):
        if not isinstance(bucket, Dsv4GradientOwnerBucket):
            raise TypeError(
                f"DSV4 owner-gradient exchange returned {type(bucket)!r} "
                f"at position {position}, expected Dsv4GradientOwnerBucket"
            )
    return buckets


def _sink_gradient_from_work(
    *,
    sink_work: Any | None,
    local_d_attn_sink: torch.Tensor,
) -> torch.Tensor:
    if sink_work is None:
        return local_d_attn_sink
    if not callable(getattr(sink_work, "wait", None)):
        raise ValueError("DSV4 sink-gradient reduce work is missing wait()")
    if not callable(getattr(sink_work, "wait_post_process", None)):
        raise ValueError(
            "DSV4 sink-gradient reduce work is missing wait_post_process()"
        )
    result = sink_work.wait_post_process()
    if not isinstance(result, torch.Tensor):
        raise TypeError(
            f"DSV4 sink-gradient reduce returned {type(result)!r}, expected Tensor"
        )
    if result.shape != local_d_attn_sink.shape:
        raise ValueError(
            "DSV4 reduced sink gradient shape mismatch: "
            f"{tuple(result.shape)} vs {tuple(local_d_attn_sink.shape)}"
        )
    return result


def _empty_owned_gradient_result(
    *,
    template: Dsv4AttentionGradientResult,
    d_attn_sink: torch.Tensor,
    query_token_ids: tuple[int, ...],
    raw_token_ids: tuple[int, ...],
    compressed_entry_ids: tuple[int, ...],
) -> Dsv4AttentionGradientResult:
    _validate_gradient_result_shapes(template)
    batch_size = int(template.dq.shape[0])
    head_count = int(template.dq.shape[2])
    head_dim = int(template.dq.shape[3])
    kv_dim = int(template.draw_kv.shape[-1])
    return Dsv4AttentionGradientResult(
        query_token_ids=query_token_ids,
        raw_token_ids=raw_token_ids,
        compressed_entry_ids=compressed_entry_ids,
        dq=template.dq.new_zeros(
            (batch_size, len(query_token_ids), head_count, head_dim)
        ),
        draw_kv=template.draw_kv.new_zeros((batch_size, len(raw_token_ids), kv_dim)),
        dcompressed_kv=template.dcompressed_kv.new_zeros(
            (batch_size, len(compressed_entry_ids), kv_dim)
        ),
        d_attn_sink=d_attn_sink,
    )


def _optional_stage_kv_peer_plans_by_slot(
    *,
    slots: Sequence[Dsv4StagePlanSlot],
    stage_kv_peer_plans_by_slot: Sequence[Sequence[Dsv4StageKvExchangePeerPlan]] | None,
    name: str,
) -> tuple[tuple[Dsv4StageKvExchangePeerPlan, ...], ...] | None:
    if stage_kv_peer_plans_by_slot is None:
        return None
    peer_plans_by_slot = tuple(
        tuple(peer_plans) for peer_plans in stage_kv_peer_plans_by_slot
    )
    if len(peer_plans_by_slot) != len(slots):
        raise RuntimeError(
            f"DSV4 prepared {name} peer plan slot count must match StagePlan "
            f"slots: {len(peer_plans_by_slot)} vs {len(slots)}"
        )
    return peer_plans_by_slot


def _validate_projected_compression_work(
    *,
    compression_work: Dsv4ProjectedCompressionForwardWork,
    kind: Dsv4CompressionKind,
    layout: Dsv4CompressedLayout | None = None,
    rank: int | None = None,
) -> Dsv4ProjectedCompressionForwardWork:
    if compression_work.compression_kind != kind:
        raise RuntimeError(
            "DSV4 projected compression kind mismatch: "
            f"{compression_work.compression_kind} vs {kind}"
        )
    if compression_work.layout.spec.kind != kind:
        raise RuntimeError(
            "DSV4 projected compression layout kind mismatch: "
            f"{compression_work.layout.spec.kind} vs {kind}"
        )
    if layout is not None and compression_work.layout != layout:
        raise RuntimeError("DSV4 projected compression layout does not match context")
    if rank is not None and int(compression_work.rank) != int(rank):
        raise RuntimeError(
            "DSV4 projected compression rank mismatch: "
            f"{int(compression_work.rank)} vs {int(rank)}"
        )
    if compression_work.main_compression_work.layout != compression_work.layout:
        raise RuntimeError("DSV4 main compression work uses a different layout")
    if int(compression_work.main_compression_work.rank) != int(compression_work.rank):
        raise RuntimeError("DSV4 main compression work uses a different rank")
    indexer_work = compression_work.indexer_compression_work
    if kind == Dsv4CompressionKind.CSA:
        if indexer_work is None:
            raise RuntimeError("DSV4 CSA projected compression requires indexer work")
        if indexer_work.layout != compression_work.layout:
            raise RuntimeError("DSV4 indexer compression work uses a different layout")
        if int(indexer_work.rank) != int(compression_work.rank):
            raise RuntimeError("DSV4 indexer compression work uses a different rank")
    elif indexer_work is not None:
        raise RuntimeError(
            "DSV4 HCA projected compression must not include indexer work"
        )
    return compression_work


def _launch_dsv4_stage_from_slot(
    *,
    stage_kv_peer_plans: Sequence[Dsv4StageKvExchangePeerPlan] | None,
    layout: Dsv4CompressedLayout,
    rank: int,
    slot: Dsv4StagePlanSlot,
    compression_kind: Dsv4CompressionKind,
    query: torch.Tensor,
    query_token_ids: Sequence[int],
    raw_kv: torch.Tensor,
    raw_token_ids: Sequence[int],
    compressed_kv: torch.Tensor,
    compressed_entry_ids: Sequence[int],
    group: Any,
    async_op: bool,
    global_topk: torch.Tensor | None,
    topk_query_token_ids: Sequence[int] | None,
    window_size: int,
    raw_list_size: int | None,
    compressed_list_size: int | None,
) -> Any:
    local_stage_inputs = build_dsv4_stage_inputs_from_stage_plan(
        layout=layout,
        stage_plan=slot.stage_plans_by_rank[int(rank)],
        compression_kind=compression_kind,
        global_topk=global_topk,
        topk_query_token_ids=topk_query_token_ids,
        window_size=window_size,
        raw_list_size=raw_list_size,
        compressed_list_size=compressed_list_size,
        materialize_compressed_metadata=compression_kind != Dsv4CompressionKind.CSA,
    )
    return launch_dsv4_stage_kv_exchange_from_stage_plan_slot(
        layout=layout,
        rank=rank,
        stage_plan_slot=slot,
        local_stage_inputs=local_stage_inputs,
        query=query,
        query_token_ids=query_token_ids,
        raw_kv=raw_kv,
        raw_token_ids=raw_token_ids,
        compressed_kv=compressed_kv,
        compressed_entry_ids=compressed_entry_ids,
        group=group,
        async_op=async_op,
        peer_plans=stage_kv_peer_plans,
    )


def _validate_stage_plan_slots(
    *,
    layout: Dsv4CompressedLayout,
    stage_plan_slots: Sequence[Dsv4StagePlanSlot],
) -> tuple[Dsv4StagePlanSlot, ...]:
    slots = tuple(stage_plan_slots)
    if not slots:
        raise ValueError("DSV4 attention forward requires stage_plan_slots")
    rank_count = len(layout.entry_ids_by_owner_rank)
    seen: set[int] = set()
    for slot in slots:
        if len(slot.stage_plans_by_rank) != rank_count:
            raise RuntimeError(
                "DSV4 StagePlan slot rank count mismatch: "
                f"{len(slot.stage_plans_by_rank)} vs {rank_count}"
            )
        stage_index = int(slot.stage_index)
        if stage_index in seen:
            raise RuntimeError(f"DSV4 duplicate StagePlan slot {stage_index}")
        seen.add(stage_index)
    return slots


def _require_prepared_layout(
    *,
    context_state: Dsv4ContextParallelState,
    kind: Dsv4CompressionKind,
) -> Dsv4CompressedLayout:
    if kind == Dsv4CompressionKind.CSA:
        layout = context_state.dsv4_plan.csa_layout
    elif kind == Dsv4CompressionKind.HCA:
        layout = context_state.dsv4_plan.hca_layout
    else:
        raise RuntimeError(f"Unsupported DSV4 compression kind {kind}")
    if layout is None:
        raise RuntimeError(
            f"DSV4 prepared context state is missing {kind.value} layout"
        )
    return layout


def _require_prepared_stage_slots(
    context_state: Dsv4ContextParallelState,
) -> tuple[Dsv4StagePlanSlot, ...]:
    slots = tuple(context_state.dsv4_plan.stage_plan_slots)
    if not slots:
        raise RuntimeError("DSV4 prepared context state is missing StagePlan slots")
    return slots


def _prepared_rank(context_state: Dsv4ContextParallelState) -> int:
    return int(context_state.cp_state.rank_plan.rank)


def _prepared_attention_backward_plan(
    *,
    context_state: Dsv4ContextParallelState,
    kind: Dsv4CompressionKind,
) -> Dsv4AttentionBackwardPlan | None:
    if kind == Dsv4CompressionKind.CSA:
        return context_state.dsv4_plan.csa_attention_backward_plan
    if kind == Dsv4CompressionKind.HCA:
        return context_state.dsv4_plan.hca_attention_backward_plan
    raise RuntimeError(f"Unsupported DSV4 compression kind {kind}")


def _attention_backward_plan_or_build(
    *,
    layout: Dsv4CompressedLayout,
    slots: Sequence[Dsv4StagePlanSlot],
    backward_plan: Dsv4AttentionBackwardPlan | None,
    rank: int,
) -> Dsv4AttentionBackwardPlan:
    if backward_plan is None:
        return build_dsv4_attention_backward_plan_from_stage_plan_slots(
            layout=layout,
            stage_plan_slots=slots,
            local_rank=int(rank),
        )
    return _validate_attention_backward_plan(
        layout=layout,
        slots=slots,
        backward_plan=backward_plan,
    )


def _validate_attention_backward_plan(
    *,
    layout: Dsv4CompressedLayout,
    slots: Sequence[Dsv4StagePlanSlot],
    backward_plan: Dsv4AttentionBackwardPlan,
) -> Dsv4AttentionBackwardPlan:
    if backward_plan.compression_kind != layout.spec.kind:
        raise RuntimeError(
            "DSV4 prepared backward plan kind must match layout kind: "
            f"{backward_plan.compression_kind} vs {layout.spec.kind}"
        )
    if backward_plan.stage_indices != _slot_stage_indices(slots):
        raise RuntimeError(
            "DSV4 prepared backward plan stage ids must match StagePlan slots"
        )
    rank_count = len(layout.entry_ids_by_owner_rank)
    _validate_exchange_rank(rank=int(backward_plan.local_rank), rank_count=rank_count)
    return backward_plan


def _build_local_attention_backward_plan_for_layout(
    *,
    layout: Dsv4CompressedLayout,
    stage_indices: tuple[int, ...],
    local_rank: int,
    common_rank_part: dict[str, Any],
    compressed_peer_plans_by_slot: Sequence[Sequence[Dsv4StageKvExchangePeerPlan]],
    rank_count: int,
) -> Dsv4AttentionBackwardPlan:
    compressed_ids, compressed_owners, recv_compressed, owned_compressed = (
        _id_space_from_stage_peer_plans_for_rank(
            peer_plans_by_slot=compressed_peer_plans_by_slot,
            recv_attr="recv_compressed_entry_ids_by_peer",
            rank=int(local_rank),
            rank_count=rank_count,
        )
    )
    return Dsv4AttentionBackwardPlan.model_construct(
        compression_kind=layout.spec.kind,
        stage_indices=stage_indices,
        local_rank=int(local_rank),
        local_rank_plan=_build_attention_backward_rank_plan_from_parts(
            common=common_rank_part,
            compressed_entry_ids=compressed_ids,
            compressed_owner_ranks=compressed_owners,
            recv_compressed_entry_ids_by_peer=recv_compressed,
            owned_compressed_entry_ids=owned_compressed,
        ),
    )


def _build_attention_backward_rank_plan_from_parts(
    *,
    common: dict[str, Any],
    compressed_entry_ids: tuple[int, ...],
    compressed_owner_ranks: tuple[int, ...],
    recv_compressed_entry_ids_by_peer: tuple[tuple[int, ...], ...],
    owned_compressed_entry_ids: tuple[int, ...],
) -> Dsv4AttentionBackwardRankPlan:
    return Dsv4AttentionBackwardRankPlan.model_construct(
        query_token_ids=common["query_token_ids"],
        raw_token_ids=common["raw_token_ids"],
        compressed_entry_ids=compressed_entry_ids,
        query_owner_ranks=common["query_owner_ranks"],
        raw_owner_ranks=common["raw_owner_ranks"],
        compressed_owner_ranks=compressed_owner_ranks,
        recv_query_token_ids_by_peer=common["recv_query_token_ids_by_peer"],
        recv_raw_token_ids_by_peer=common["recv_raw_token_ids_by_peer"],
        recv_compressed_entry_ids_by_peer=recv_compressed_entry_ids_by_peer,
        owned_query_token_ids=common["owned_query_token_ids"],
        owned_raw_token_ids=common["owned_raw_token_ids"],
        owned_compressed_entry_ids=owned_compressed_entry_ids,
    )


def _slot_stage_indices(slots: Sequence[Dsv4StagePlanSlot]) -> tuple[int, ...]:
    return tuple(int(slot.stage_index) for slot in slots)


def _stage_plan_slot_query_id_space_for_rank(
    *,
    slots: Sequence[Dsv4StagePlanSlot],
    rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    query_ranges: list[tuple[int, int]] = []
    for slot in slots:
        _append_stage_ranges(
            query_ranges,
            slot.stage_plans_by_rank[int(rank)].global_q_ranges,
        )
    query_ids = _token_ids_from_merged_ranges(query_ranges)
    return query_ids, (int(rank),) * len(query_ids)


def _id_space_from_stage_peer_plans_for_rank(
    *,
    peer_plans_by_slot: Sequence[Sequence[Dsv4StageKvExchangePeerPlan]],
    recv_attr: str,
    rank: int,
    rank_count: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
]:
    ids: list[int] = []
    owners: list[int] = []
    recv: list[list[int]] = [[] for _ in range(rank_count)]
    rank_int = int(rank)
    send_attr = _stage_peer_send_attr_for_recv_attr(recv_attr)
    for slot_plans in peer_plans_by_slot:
        if len(slot_plans) != int(rank_count):
            raise RuntimeError("DSV4 backward peer-plan rank count mismatch")
        plan = slot_plans[rank_int]
        ids_by_owner = getattr(plan, recv_attr)
        ids_by_peer_sent = getattr(plan, send_attr)
        if len(ids_by_owner) != int(rank_count) or len(ids_by_peer_sent) != int(
            rank_count
        ):
            raise RuntimeError("DSV4 backward peer-plan peer count mismatch")
        for owner_rank, owner_ids in enumerate(ids_by_owner):
            for id_ in owner_ids:
                ids.append(int(id_))
                owners.append(owner_rank)
        for peer_rank, peer_ids in enumerate(ids_by_peer_sent):
            recv[peer_rank].extend(int(id_) for id_ in peer_ids)
    ids_tuple, owners_tuple = _dedupe_ids_and_owners(ids=ids, owners=owners)
    recv_ids = tuple(tuple(dict.fromkeys(peer_ids).keys()) for peer_ids in recv)
    return (
        ids_tuple,
        owners_tuple,
        recv_ids,
        tuple(dict.fromkeys(chain.from_iterable(recv_ids)).keys()),
    )


def _stage_peer_send_attr_for_recv_attr(recv_attr: str) -> str:
    if recv_attr == "recv_raw_token_ids_by_peer":
        return "send_raw_token_ids_by_peer"
    if recv_attr == "recv_compressed_entry_ids_by_peer":
        return "send_compressed_entry_ids_by_peer"
    raise RuntimeError(f"Unsupported DSV4 stage peer recv attr: {recv_attr}")


def _dedupe_ids_and_owners(
    *,
    ids: Sequence[int],
    owners: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    seen: set[int] = set()
    out_ids: list[int] = []
    out_owners: list[int] = []
    for id_, owner in zip(ids, owners, strict=True):
        if id_ not in seen:
            seen.add(id_)
            out_ids.append(id_)
            out_owners.append(owner)
    return tuple(out_ids), tuple(out_owners)


def _append_stage_ranges(
    target: list[tuple[int, int]],
    ranges: Sequence[Any],
) -> None:
    for range_ in ranges:
        start = int(range_.start)
        end = int(range_.end)
        if start < end:
            target.append((start, end))


def _token_ids_from_merged_ranges(ranges: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    ids: list[int] = []
    for start, end in _merge_token_ranges(ranges):
        if int(start) < 0:
            raise RuntimeError(f"DSV4 query token range {start}:{end} has no CP owner")
        ids.extend(range(int(start), int(end)))
    return tuple(ids)


def _merge_token_ranges(
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
    return tuple((start, end) for start, end in merged)


def _common_backward_rank_part(
    *,
    query_token_ids: tuple[int, ...],
    query_owner_ranks: tuple[int, ...],
    raw_token_ids: tuple[int, ...],
    raw_owner_ranks: tuple[int, ...],
    recv_raw_token_ids_by_peer: tuple[tuple[int, ...], ...],
    owned_raw_token_ids: tuple[int, ...],
    rank: int,
    rank_count: int,
) -> dict[str, Any]:
    recv_query_by_peer, owned_query_ids = _identity_query_recv_and_owned_for_rank(
        query_token_ids=query_token_ids,
        rank=int(rank),
        rank_count=int(rank_count),
    )
    return {
        "query_token_ids": query_token_ids,
        "raw_token_ids": raw_token_ids,
        "query_owner_ranks": query_owner_ranks,
        "raw_owner_ranks": raw_owner_ranks,
        "recv_query_token_ids_by_peer": recv_query_by_peer,
        "recv_raw_token_ids_by_peer": recv_raw_token_ids_by_peer,
        "owned_query_token_ids": owned_query_ids,
        "owned_raw_token_ids": owned_raw_token_ids,
    }


def _identity_query_recv_and_owned_for_rank(
    *,
    query_token_ids: tuple[int, ...],
    rank: int,
    rank_count: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    recv_by_peer: list[tuple[int, ...]] = [() for _ in range(int(rank_count))]
    recv_by_peer[int(rank)] = query_token_ids
    return tuple(recv_by_peer), query_token_ids


def _token_ids_from_ranges(ranges: Sequence[Any]) -> tuple[int, ...]:
    ids: list[int] = []
    for range_ in ranges:
        ids.extend(range(int(range_.start), int(range_.end)))
    _row_by_id(ids=tuple(ids), name="stage_plan_token_ranges")
    return tuple(ids)


def _normalize_output_ids(ids: Sequence[int], *, name: str) -> tuple[int, ...]:
    ids = tuple(int(id_) for id_ in ids)
    _validate_nonnegative_ids(ids, name=name)
    _row_by_id(ids=ids, name=name)
    return ids


def merge_materialized_stage_records(
    *,
    records: Sequence[Dsv4StageForwardRecord],
    query_token_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(records) == 0:
        raise ValueError("at least one DSV4 stage record is required")
    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    query_index = _row_by_id(ids=query_ids, name="query_token_ids")
    first = records[0]
    batch_size, _, head_count, dim = first.out.shape
    target_dtype = _accum_output_dtype(first.out.dtype)
    real_out = torch.zeros(
        (batch_size, len(query_ids), head_count, dim),
        device=first.out.device,
        dtype=target_dtype,
    )
    real_lse = torch.full(
        (batch_size, len(query_ids), head_count),
        float("-inf"),
        device=first.lse.device,
        dtype=target_dtype,
    )

    for record in records:
        _validate_stage_forward_shapes(
            stage=record.materialized_stage,
            out=record.out,
            lse=record.lse,
        )
        if (
            int(record.out.shape[0]) != batch_size
            or int(record.out.shape[2]) != head_count
        ):
            raise ValueError("all DSV4 stage records must share batch and head count")
        if int(record.out.shape[-1]) != dim:
            raise ValueError("all DSV4 stage records must share head dim")
        stage_indices = _query_indices_for_stage(
            stage_query_ids=record.materialized_stage.query_token_ids,
            query_index=query_index,
            device=real_out.device,
        )
        prev_out = real_out.index_select(1, stage_indices)
        prev_lse = real_lse.index_select(1, stage_indices)
        merged_out, merged_lse = merge_two_stage_outputs(
            prev_out,
            prev_lse,
            record.out.to(dtype=target_dtype),
            record.lse.to(dtype=target_dtype),
        )
        real_out.index_copy_(1, stage_indices, merged_out)
        real_lse.index_copy_(1, stage_indices, merged_lse)
    return real_out, real_lse


def merge_single_sink_branch(
    real_out: torch.Tensor,
    real_lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attn_sink.ndim != 1:
        raise ValueError("attn_sink must be a per-head tensor with shape [H]")
    if real_out.shape[:-1] != real_lse.shape:
        raise ValueError("real_out shape must be real_lse shape plus head dim")
    if real_lse.shape[-1] != attn_sink.shape[0]:
        raise ValueError("real_lse head count must match attn_sink")

    target_dtype = _accum_output_dtype(real_out.dtype)
    real_out = real_out.to(dtype=target_dtype)
    real_lse = real_lse.to(dtype=target_dtype)
    sink_lse = attn_sink.to(dtype=target_dtype).view(
        *((1,) * (real_lse.ndim - 1)),
        attn_sink.shape[0],
    )

    global_lse = _safe_logaddexp(real_lse, sink_lse)
    real_weight = _safe_exp_diff(real_lse, global_lse).unsqueeze(-1)
    global_out = real_out * real_weight
    return _zero_invalid_rows(global_out, global_lse), global_lse


def replay_materialized_dsv4_attention_backward(
    *,
    forward_result: Dsv4AttentionForwardResult,
    grad_out: torch.Tensor,
) -> Dsv4AttentionBackwardReplayResult:
    if grad_out.shape != forward_result.out.shape:
        raise ValueError(
            "DSV4 replay grad_out must match forward output, got "
            f"{tuple(grad_out.shape)} vs {tuple(forward_result.out.shape)}"
        )
    query_index = _row_by_id(ids=forward_result.query_token_ids, name="query_token_ids")
    disabled_sink = sparse_kernel.dsv4_disabled_attn_sink(forward_result.attn_sink)
    stage_records: list[Dsv4StageBackwardRecord] = []
    for record in forward_result.stage_records:
        stage_indices = _query_indices_for_stage(
            stage_query_ids=record.materialized_stage.query_token_ids,
            query_index=query_index,
            device=forward_result.out.device,
        )
        stage_global_out = forward_result.out.index_select(1, stage_indices)
        stage_grad_out = grad_out.index_select(1, stage_indices)
        stage_global_lse = forward_result.lse.index_select(1, stage_indices)
        stage_grad = sparse_kernel.dsv4_sparse_bwd(
            q=record.materialized_stage.q_stage,
            kv=record.materialized_stage.kv_stage,
            attn_sink=disabled_sink,
            topk=record.materialized_stage.topk_stage_local,
            global_out=stage_global_out,
            grad_out=stage_grad_out,
            global_lse=stage_global_lse,
            scale=forward_result.scale,
        )
        stage_records.append(
            Dsv4StageBackwardRecord(
                materialized_stage=record.materialized_stage,
                dq_stage=stage_grad.dq,
                dkv_stage=stage_grad.dkv,
            )
        )

    d_attn_sink = compute_single_sink_grad(
        grad_out=grad_out,
        global_out=forward_result.out,
        global_lse=forward_result.lse,
        attn_sink=forward_result.attn_sink,
    )
    return Dsv4AttentionBackwardReplayResult(
        stage_records=tuple(stage_records),
        d_attn_sink=d_attn_sink,
    )


def accumulate_materialized_dsv4_attention_backward(
    *,
    replay_result: Dsv4AttentionBackwardReplayResult,
    query_token_ids: Sequence[int],
    raw_token_ids: Sequence[int],
    compressed_entry_ids: Sequence[int],
) -> Dsv4AttentionGradientResult:
    if len(replay_result.stage_records) == 0:
        raise ValueError("at least one DSV4 replay stage record is required")
    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    raw_ids = tuple(int(token_id) for token_id in raw_token_ids)
    compressed_ids = tuple(int(entry_id) for entry_id in compressed_entry_ids)
    query_index = _row_by_id(ids=query_ids, name="query_token_ids")
    raw_index = _row_by_id(ids=raw_ids, name="raw_token_ids")
    compressed_index = _row_by_id(
        ids=compressed_ids,
        name="compressed_entry_ids",
    )

    first = replay_result.stage_records[0]
    _validate_stage_backward_shapes(first)
    batch_size, _, head_count, dim = first.dq_stage.shape
    kv_dim = int(first.dkv_stage.shape[-1])
    target_dtype = _accum_output_dtype(first.dq_stage.dtype)
    dq = torch.zeros(
        (batch_size, len(query_ids), head_count, dim),
        device=first.dq_stage.device,
        dtype=target_dtype,
    )
    draw_kv = torch.zeros(
        (batch_size, len(raw_ids), kv_dim),
        device=first.dkv_stage.device,
        dtype=target_dtype,
    )
    dcompressed_kv = torch.zeros(
        (batch_size, len(compressed_ids), kv_dim),
        device=first.dkv_stage.device,
        dtype=target_dtype,
    )

    for record in replay_result.stage_records:
        _validate_stage_backward_shapes(record)
        if (
            int(record.dq_stage.shape[0]) != batch_size
            or int(record.dq_stage.shape[2]) != head_count
            or int(record.dq_stage.shape[-1]) != dim
        ):
            raise ValueError("all DSV4 replay dq_stage tensors must share shape")
        if (
            int(record.dkv_stage.shape[0]) != batch_size
            or int(record.dkv_stage.shape[-1]) != kv_dim
        ):
            raise ValueError("all DSV4 replay dkv_stage tensors must share shape")

        q_indices = _query_indices_for_stage(
            stage_query_ids=record.materialized_stage.query_token_ids,
            query_index=query_index,
            device=dq.device,
        )
        dq.index_add_(1, q_indices, record.dq_stage.to(dtype=target_dtype))

        raw_ids_stage, compressed_ids_stage = _stage_raw_and_compressed_key_ids(
            record.materialized_stage,
        )
        raw_count = len(raw_ids_stage)
        compressed_count = len(compressed_ids_stage)
        if raw_count:
            raw_indices = _indices_for_ids(
                ids=raw_ids_stage,
                id_index=raw_index,
                name="raw_token_ids",
                device=draw_kv.device,
            )
            draw_kv.index_add_(
                1,
                raw_indices,
                record.dkv_stage[:, :raw_count].to(dtype=target_dtype),
            )
        if compressed_count:
            compressed_indices = _indices_for_ids(
                ids=compressed_ids_stage,
                id_index=compressed_index,
                name="compressed_entry_ids",
                device=dcompressed_kv.device,
            )
            dcompressed_kv.index_add_(
                1,
                compressed_indices,
                record.dkv_stage[:, raw_count : raw_count + compressed_count].to(
                    dtype=target_dtype
                ),
            )

    return Dsv4AttentionGradientResult(
        query_token_ids=query_ids,
        raw_token_ids=raw_ids,
        compressed_entry_ids=compressed_ids,
        dq=dq,
        draw_kv=draw_kv,
        dcompressed_kv=dcompressed_kv,
        d_attn_sink=replay_result.d_attn_sink,
    )


def pack_dsv4_gradient_owner_buckets(
    *,
    gradients: Dsv4AttentionGradientResult,
    query_owner_ranks: Sequence[int],
    raw_owner_ranks: Sequence[int],
    compressed_owner_ranks: Sequence[int],
) -> tuple[Dsv4GradientOwnerBucket, ...]:
    query_owner_ranks = _validate_owner_ranks(
        ranks=query_owner_ranks,
        expected_count=len(gradients.query_token_ids),
        name="query_owner_ranks",
    )
    raw_owner_ranks = _validate_owner_ranks(
        ranks=raw_owner_ranks,
        expected_count=len(gradients.raw_token_ids),
        name="raw_owner_ranks",
    )
    compressed_owner_ranks = _validate_owner_ranks(
        ranks=compressed_owner_ranks,
        expected_count=len(gradients.compressed_entry_ids),
        name="compressed_owner_ranks",
    )
    _validate_gradient_result_shapes(gradients)

    owner_ranks = sorted(
        set(query_owner_ranks) | set(raw_owner_ranks) | set(compressed_owner_ranks)
    )
    buckets: list[Dsv4GradientOwnerBucket] = []
    for owner_rank in owner_ranks:
        q_positions = _positions_for_owner(query_owner_ranks, owner_rank)
        raw_positions = _positions_for_owner(raw_owner_ranks, owner_rank)
        compressed_positions = _positions_for_owner(compressed_owner_ranks, owner_rank)
        buckets.append(
            Dsv4GradientOwnerBucket(
                owner_rank=owner_rank,
                query_token_ids=_ids_at_positions(
                    gradients.query_token_ids, q_positions
                ),
                raw_token_ids=_ids_at_positions(gradients.raw_token_ids, raw_positions),
                compressed_entry_ids=_ids_at_positions(
                    gradients.compressed_entry_ids,
                    compressed_positions,
                ),
                dq=_index_select_positions(
                    gradients.dq,
                    positions=q_positions,
                    device=gradients.dq.device,
                ),
                draw_kv=_index_select_positions(
                    gradients.draw_kv,
                    positions=raw_positions,
                    device=gradients.draw_kv.device,
                ),
                dcompressed_kv=_index_select_positions(
                    gradients.dcompressed_kv,
                    positions=compressed_positions,
                    device=gradients.dcompressed_kv.device,
                ),
            )
        )
    return tuple(buckets)


@torch.compiler.disable
def launch_dsv4_gradient_owner_bucket_exchange(
    *,
    gradients: Dsv4AttentionGradientResult,
    query_owner_ranks: Sequence[int],
    raw_owner_ranks: Sequence[int],
    compressed_owner_ranks: Sequence[int],
    recv_query_token_ids_by_peer: Sequence[Sequence[int]],
    recv_raw_token_ids_by_peer: Sequence[Sequence[int]],
    recv_compressed_entry_ids_by_peer: Sequence[Sequence[int]],
    rank: int,
    rank_count: int,
    group: Any,
    async_op: bool,
) -> Dsv4GradientOwnerExchangeWork:
    _validate_gradient_result_shapes(gradients)
    rank = int(rank)
    rank_count = int(rank_count)
    _validate_exchange_rank(rank=rank, rank_count=rank_count)
    recv_query = _normalize_peer_ids(
        recv_query_token_ids_by_peer,
        rank_count=rank_count,
        name="recv_query_token_ids_by_peer",
    )
    recv_raw = _normalize_peer_ids(
        recv_raw_token_ids_by_peer,
        rank_count=rank_count,
        name="recv_raw_token_ids_by_peer",
    )
    recv_compressed = _normalize_peer_ids(
        recv_compressed_entry_ids_by_peer,
        rank_count=rank_count,
        name="recv_compressed_entry_ids_by_peer",
    )
    buckets = pack_dsv4_gradient_owner_buckets(
        gradients=gradients,
        query_owner_ranks=query_owner_ranks,
        raw_owner_ranks=raw_owner_ranks,
        compressed_owner_ranks=compressed_owner_ranks,
    )
    query_tensor, query_ids, query_send = _pack_owner_exchange_rows(
        rows_by_owner=tuple(
            (
                bucket.owner_rank,
                bucket.query_token_ids,
                bucket.dq.flatten(2),
            )
            for bucket in buckets
        ),
        empty=gradients.dq.flatten(2).new_empty(
            (
                int(gradients.dq.shape[0]),
                0,
                int(gradients.dq.shape[2]) * int(gradients.dq.shape[3]),
            )
        ),
        rank=rank,
        rank_count=rank_count,
        name="query owner gradients",
    )
    raw_tensor, raw_ids, raw_send = _pack_owner_exchange_rows(
        rows_by_owner=tuple(
            (bucket.owner_rank, bucket.raw_token_ids, bucket.draw_kv)
            for bucket in buckets
        ),
        empty=gradients.draw_kv.new_empty(
            (int(gradients.draw_kv.shape[0]), 0, int(gradients.draw_kv.shape[-1]))
        ),
        rank=rank,
        rank_count=rank_count,
        name="raw owner gradients",
    )
    compressed_tensor, compressed_ids, compressed_send = _pack_owner_exchange_rows(
        rows_by_owner=tuple(
            (
                bucket.owner_rank,
                bucket.compressed_entry_ids,
                bucket.dcompressed_kv,
            )
            for bucket in buckets
        ),
        empty=gradients.dcompressed_kv.new_empty(
            (
                int(gradients.dcompressed_kv.shape[0]),
                0,
                int(gradients.dcompressed_kv.shape[-1]),
            )
        ),
        rank=rank,
        rank_count=rank_count,
        name="compressed owner gradients",
    )
    return Dsv4GradientOwnerExchangeWork(
        rank=rank,
        rank_count=rank_count,
        recv_query_token_ids_by_peer=recv_query,
        recv_raw_token_ids_by_peer=recv_raw,
        recv_compressed_entry_ids_by_peer=recv_compressed,
        query_head_count=int(gradients.dq.shape[2]),
        query_head_dim=int(gradients.dq.shape[3]),
        query_work=launch_dsv4_tensor_exchange(
            tensor=query_tensor,
            tensor_ids=query_ids,
            plan=Dsv4TensorExchangePlan(
                send_ids_by_peer=query_send,
                recv_ids_by_peer=_wire_peer_ids_by_peer(
                    ids_by_peer=recv_query,
                    rank_count=rank_count,
                ),
            ),
            group=group,
            async_op=async_op,
        ),
        raw_work=launch_dsv4_tensor_exchange(
            tensor=raw_tensor,
            tensor_ids=raw_ids,
            plan=Dsv4TensorExchangePlan(
                send_ids_by_peer=raw_send,
                recv_ids_by_peer=_wire_peer_ids_by_peer(
                    ids_by_peer=recv_raw,
                    rank_count=rank_count,
                ),
            ),
            group=group,
            async_op=async_op,
        ),
        compressed_work=launch_dsv4_tensor_exchange(
            tensor=compressed_tensor,
            tensor_ids=compressed_ids,
            plan=Dsv4TensorExchangePlan(
                send_ids_by_peer=compressed_send,
                recv_ids_by_peer=_wire_peer_ids_by_peer(
                    ids_by_peer=recv_compressed,
                    rank_count=rank_count,
                ),
            ),
            group=group,
            async_op=async_op,
        ),
    )


def accumulate_dsv4_gradient_owner_buckets(
    *,
    buckets: Sequence[Dsv4GradientOwnerBucket],
    query_token_ids: Sequence[int],
    raw_token_ids: Sequence[int],
    compressed_entry_ids: Sequence[int],
    d_attn_sink: torch.Tensor,
) -> Dsv4AttentionGradientResult:
    if len(buckets) == 0:
        raise ValueError("at least one DSV4 gradient owner bucket is required")
    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    raw_ids = tuple(int(token_id) for token_id in raw_token_ids)
    compressed_ids = tuple(int(entry_id) for entry_id in compressed_entry_ids)
    query_index = _row_by_id(ids=query_ids, name="query_token_ids")
    raw_index = _row_by_id(ids=raw_ids, name="raw_token_ids")
    compressed_index = _row_by_id(
        ids=compressed_ids,
        name="compressed_entry_ids",
    )

    first = buckets[0]
    _validate_owner_bucket_shapes(first)
    batch_size, _, head_count, dim = first.dq.shape
    kv_dim = int(first.draw_kv.shape[-1])
    target_dtype = _accum_output_dtype(first.dq.dtype)
    dq = torch.zeros(
        (batch_size, len(query_ids), head_count, dim),
        device=first.dq.device,
        dtype=target_dtype,
    )
    draw_kv = torch.zeros(
        (batch_size, len(raw_ids), kv_dim),
        device=first.draw_kv.device,
        dtype=target_dtype,
    )
    dcompressed_kv = torch.zeros(
        (batch_size, len(compressed_ids), kv_dim),
        device=first.dcompressed_kv.device,
        dtype=target_dtype,
    )

    for bucket in buckets:
        _validate_owner_bucket_shapes(bucket)
        if (
            int(bucket.dq.shape[0]) != batch_size
            or int(bucket.dq.shape[2]) != head_count
            or int(bucket.dq.shape[-1]) != dim
        ):
            raise ValueError(
                "all DSV4 gradient owner bucket dq tensors must share shape"
            )
        if (
            int(bucket.draw_kv.shape[0]) != batch_size
            or int(bucket.draw_kv.shape[-1]) != kv_dim
            or int(bucket.dcompressed_kv.shape[0]) != batch_size
            or int(bucket.dcompressed_kv.shape[-1]) != kv_dim
        ):
            raise ValueError(
                "all DSV4 gradient owner bucket KV tensors must share shape"
            )
        if bucket.query_token_ids:
            dq.index_add_(
                1,
                _indices_for_ids(
                    ids=bucket.query_token_ids,
                    id_index=query_index,
                    name="query_token_ids",
                    device=dq.device,
                ),
                bucket.dq.to(dtype=target_dtype),
            )
        if bucket.raw_token_ids:
            draw_kv.index_add_(
                1,
                _indices_for_ids(
                    ids=bucket.raw_token_ids,
                    id_index=raw_index,
                    name="raw_token_ids",
                    device=draw_kv.device,
                ),
                bucket.draw_kv.to(dtype=target_dtype),
            )
        if bucket.compressed_entry_ids:
            dcompressed_kv.index_add_(
                1,
                _indices_for_ids(
                    ids=bucket.compressed_entry_ids,
                    id_index=compressed_index,
                    name="compressed_entry_ids",
                    device=dcompressed_kv.device,
                ),
                bucket.dcompressed_kv.to(dtype=target_dtype),
            )

    return Dsv4AttentionGradientResult(
        query_token_ids=query_ids,
        raw_token_ids=raw_ids,
        compressed_entry_ids=compressed_ids,
        dq=dq,
        draw_kv=draw_kv,
        dcompressed_kv=dcompressed_kv,
        d_attn_sink=d_attn_sink,
    )


def compute_single_sink_grad(
    grad_out: torch.Tensor,
    global_out: torch.Tensor,
    global_lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    if attn_sink.ndim != 1:
        raise ValueError("attn_sink must be a per-head tensor with shape [H]")
    if global_out.shape != grad_out.shape or global_out.shape[:-1] != global_lse.shape:
        raise ValueError("global_out, grad_out, and global_lse shapes are inconsistent")
    if global_lse.shape[-1] != attn_sink.shape[0]:
        raise ValueError("global_lse head count must match attn_sink")

    target_dtype = _accum_output_dtype(global_out.dtype)
    global_out = global_out.to(dtype=target_dtype)
    grad_out = grad_out.to(dtype=target_dtype)
    global_lse = global_lse.to(dtype=target_dtype)
    sink_lse = attn_sink.to(dtype=target_dtype).view(
        *((1,) * (global_lse.ndim - 1)),
        attn_sink.shape[0],
    )

    delta = (global_out * grad_out).sum(dim=-1)
    p_sink = _safe_exp_diff(sink_lse, global_lse)
    reduce_dims = tuple(range(delta.ndim - 1))
    return -(delta * p_sink).sum(dim=reduce_dims)


def _pack_owner_exchange_rows(
    *,
    rows_by_owner: Sequence[tuple[int, tuple[int, ...], torch.Tensor]],
    empty: torch.Tensor,
    rank: int,
    rank_count: int,
    name: str,
) -> tuple[torch.Tensor, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    send_ids_by_peer: list[tuple[int, ...]] = [() for _ in range(rank_count)]
    tensor_ids: list[int] = []
    pieces: list[torch.Tensor] = []
    for owner_rank, row_ids, rows in rows_by_owner:
        owner_rank = int(owner_rank)
        _validate_exchange_rank(rank=owner_rank, rank_count=rank_count)
        if rows.ndim != 3:
            raise ValueError(f"DSV4 {name} rows must be [B,N,D], got {rows.shape}")
        if int(rows.shape[1]) != len(row_ids):
            raise ValueError(f"DSV4 {name} row count does not match ids")
        wire_ids = _wire_ids_for_peer(
            source_rank=rank,
            rank_count=rank_count,
            ids=row_ids,
        )
        if send_ids_by_peer[owner_rank]:
            raise ValueError(f"DSV4 {name} has duplicate owner rank {owner_rank}")
        send_ids_by_peer[owner_rank] = wire_ids
        tensor_ids.extend(wire_ids)
        if row_ids:
            pieces.append(rows)
    if not pieces:
        return empty, (), tuple(send_ids_by_peer)
    return torch.cat(pieces, dim=1), tuple(tensor_ids), tuple(send_ids_by_peer)


def _normalize_peer_ids(
    ids_by_peer: Sequence[Sequence[int]],
    *,
    rank_count: int,
    name: str,
) -> tuple[tuple[int, ...], ...]:
    if len(ids_by_peer) != int(rank_count):
        raise ValueError(
            f"DSV4 {name} peer count {len(ids_by_peer)} does not match {rank_count}"
        )
    normalized: list[tuple[int, ...]] = []
    for peer, ids in enumerate(ids_by_peer):
        peer_ids = tuple(int(id_) for id_ in ids)
        _validate_nonnegative_ids(peer_ids, name=f"{name}[{peer}]")
        _row_by_id(ids=peer_ids, name=f"{name}[{peer}]")
        normalized.append(peer_ids)
    return tuple(normalized)


def _wire_peer_ids_by_peer(
    *,
    ids_by_peer: tuple[tuple[int, ...], ...],
    rank_count: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        _wire_ids_for_peer(
            source_rank=peer,
            rank_count=rank_count,
            ids=peer_ids,
        )
        for peer, peer_ids in enumerate(ids_by_peer)
    )


def _wire_ids_for_peer(
    *,
    source_rank: int,
    rank_count: int,
    ids: Sequence[int],
) -> tuple[int, ...]:
    _validate_exchange_rank(rank=source_rank, rank_count=rank_count)
    ids = tuple(int(id_) for id_ in ids)
    _validate_nonnegative_ids(ids, name="owner exchange ids")
    return tuple(int(id_) * int(rank_count) + int(source_rank) for id_ in ids)


def _validate_owner_exchange_result_ids(
    *,
    actual: tuple[int, ...],
    expected: tuple[tuple[int, ...], ...],
    name: str,
) -> None:
    flat_expected = tuple(id_ for peer_ids in expected for id_ in peer_ids)
    if actual != flat_expected:
        raise RuntimeError(
            f"DSV4 {name} owner exchange received ids {actual} "
            f"but expected {flat_expected}"
        )


def _validate_exchange_rank(*, rank: int, rank_count: int) -> None:
    if int(rank) < 0 or int(rank) >= int(rank_count):
        raise ValueError(
            f"DSV4 exchange rank {rank} is outside rank count {rank_count}"
        )


def _validate_nonnegative_ids(ids: Sequence[int], *, name: str) -> None:
    if any(int(id_) < 0 for id_ in ids):
        raise ValueError(f"DSV4 {name} must be non-negative")


def _validate_gradient_result_shapes(gradients: Dsv4AttentionGradientResult) -> None:
    if int(gradients.dq.shape[1]) != len(gradients.query_token_ids):
        raise ValueError("DSV4 dq rows must match query_token_ids")
    if int(gradients.draw_kv.shape[1]) != len(gradients.raw_token_ids):
        raise ValueError("DSV4 draw_kv rows must match raw_token_ids")
    if int(gradients.dcompressed_kv.shape[1]) != len(gradients.compressed_entry_ids):
        raise ValueError("DSV4 dcompressed_kv rows must match compressed_entry_ids")
    if gradients.dq.ndim != 4:
        raise ValueError(f"DSV4 dq must have shape [B,Q,H,D], got {gradients.dq.shape}")
    if gradients.draw_kv.ndim != 3 or gradients.dcompressed_kv.ndim != 3:
        raise ValueError(
            "DSV4 KV gradients must have shape [B,K,D], got "
            f"raw={gradients.draw_kv.shape}, compressed={gradients.dcompressed_kv.shape}"
        )
    if int(gradients.draw_kv.shape[0]) != int(gradients.dq.shape[0]) or int(
        gradients.dcompressed_kv.shape[0]
    ) != int(gradients.dq.shape[0]):
        raise ValueError("DSV4 gradient batch dimensions must match")
    if int(gradients.draw_kv.shape[-1]) != int(gradients.dq.shape[-1]) or int(
        gradients.dcompressed_kv.shape[-1]
    ) != int(gradients.dq.shape[-1]):
        raise ValueError("DSV4 gradient head/KV dims must match")


def _validate_owner_bucket_shapes(bucket: Dsv4GradientOwnerBucket) -> None:
    if bucket.dq.ndim != 4:
        raise ValueError(
            f"DSV4 owner bucket dq must be [B,Q,H,D], got {bucket.dq.shape}"
        )
    if bucket.draw_kv.ndim != 3 or bucket.dcompressed_kv.ndim != 3:
        raise ValueError(
            "DSV4 owner bucket KV gradients must be [B,K,D], got "
            f"raw={bucket.draw_kv.shape}, compressed={bucket.dcompressed_kv.shape}"
        )
    if int(bucket.dq.shape[1]) != len(bucket.query_token_ids):
        raise ValueError("DSV4 owner bucket dq rows must match query ids")
    if int(bucket.draw_kv.shape[1]) != len(bucket.raw_token_ids):
        raise ValueError("DSV4 owner bucket raw rows must match raw ids")
    if int(bucket.dcompressed_kv.shape[1]) != len(bucket.compressed_entry_ids):
        raise ValueError("DSV4 owner bucket compressed rows must match compressed ids")
    _row_by_id(ids=bucket.query_token_ids, name="bucket_query_token_ids")
    _row_by_id(ids=bucket.raw_token_ids, name="bucket_raw_token_ids")
    _row_by_id(ids=bucket.compressed_entry_ids, name="bucket_compressed_entry_ids")


def _validate_owner_ranks(
    *,
    ranks: Sequence[int],
    expected_count: int,
    name: str,
) -> tuple[int, ...]:
    ranks = tuple(int(rank) for rank in ranks)
    if len(ranks) != int(expected_count):
        raise ValueError(
            f"DSV4 {name} length {len(ranks)} does not match expected {expected_count}"
        )
    if any(rank < 0 for rank in ranks):
        raise ValueError(f"DSV4 {name} must be non-negative")
    return ranks


def _positions_for_owner(
    owner_ranks: Sequence[int], owner_rank: int
) -> tuple[int, ...]:
    return tuple(
        index
        for index, candidate_rank in enumerate(owner_ranks)
        if int(candidate_rank) == int(owner_rank)
    )


def _ids_at_positions(ids: Sequence[int], positions: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(ids[position]) for position in positions)


def _index_select_positions(
    tensor: torch.Tensor,
    *,
    positions: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    indices = torch.tensor(
        tuple(int(position) for position in positions), device=device, dtype=torch.long
    )
    return tensor.index_select(1, indices)


def _validate_stage_backward_shapes(record: Dsv4StageBackwardRecord) -> None:
    stage = record.materialized_stage
    if record.dq_stage.shape != stage.q_stage.shape:
        raise ValueError(
            "DSV4 replay dq_stage must match q_stage, got "
            f"dq={tuple(record.dq_stage.shape)}, q={tuple(stage.q_stage.shape)}"
        )
    if record.dkv_stage.shape != stage.kv_stage.shape:
        raise ValueError(
            "DSV4 replay dkv_stage must match kv_stage, got "
            f"dkv={tuple(record.dkv_stage.shape)}, kv={tuple(stage.kv_stage.shape)}"
        )
    expected_keys = int(stage.raw_count) + int(stage.compressed_count)
    if expected_keys != int(stage.kv_stage.shape[1]):
        raise ValueError(
            "DSV4 materialized stage raw+compressed count must match kv rows, got "
            f"raw={stage.raw_count}, compressed={stage.compressed_count}, "
            f"kv={tuple(stage.kv_stage.shape)}"
        )


def _stage_raw_and_compressed_key_ids(
    stage: Dsv4MaterializedStage,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    raw_count = int(stage.raw_count)
    compressed_count = int(stage.compressed_count)
    if len(stage.key_global_ids) != raw_count + compressed_count:
        raise ValueError("DSV4 key_global_ids length must match raw+compressed count")
    if len(stage.key_kinds) != len(stage.key_global_ids):
        raise ValueError("DSV4 key_kinds length must match key_global_ids")
    raw_kinds = stage.key_kinds[:raw_count]
    compressed_kinds = stage.key_kinds[raw_count : raw_count + compressed_count]
    if any(kind != Dsv4StageKeyKind.RAW for kind in raw_kinds):
        raise ValueError("DSV4 raw key slice contains non-raw key kind")
    if any(kind != Dsv4StageKeyKind.COMPRESSED for kind in compressed_kinds):
        raise ValueError("DSV4 compressed key slice contains non-compressed key kind")
    return (
        tuple(int(id_) for id_ in stage.key_global_ids[:raw_count]),
        tuple(
            int(id_)
            for id_ in stage.key_global_ids[raw_count : raw_count + compressed_count]
        ),
    )


def _validate_stage_forward_shapes(
    *,
    stage: Dsv4MaterializedStage,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    if out.shape != stage.q_stage.shape:
        raise ValueError(
            "DSV4 stage output must match q_stage, got "
            f"out={tuple(out.shape)}, q={tuple(stage.q_stage.shape)}"
        )
    if lse.shape != stage.q_stage.shape[:-1]:
        raise ValueError(
            "DSV4 stage LSE must match q_stage without head dim, got "
            f"lse={tuple(lse.shape)}, q={tuple(stage.q_stage.shape)}"
        )
    if len(stage.query_token_ids) != int(stage.q_stage.shape[1]):
        raise ValueError(
            "DSV4 materialized stage query id count must match q_stage query dim, got "
            f"ids={len(stage.query_token_ids)}, q={tuple(stage.q_stage.shape)}"
        )


def _query_indices_for_stage(
    *,
    stage_query_ids: Sequence[int],
    query_index: dict[int, int],
    device: torch.device,
) -> torch.Tensor:
    _row_by_id(ids=stage_query_ids, name="stage_query_ids")
    missing = tuple(
        int(token_id)
        for token_id in stage_query_ids
        if int(token_id) not in query_index
    )
    if missing:
        raise ValueError(
            f"DSV4 stage query ids missing from global query ids: {missing}"
        )
    return torch.tensor(
        [query_index[int(token_id)] for token_id in stage_query_ids],
        device=device,
        dtype=torch.long,
    )


def _indices_for_ids(
    *,
    ids: Sequence[int],
    id_index: dict[int, int],
    name: str,
    device: torch.device,
) -> torch.Tensor:
    _row_by_id(ids=ids, name=f"stage_{name}")
    missing = tuple(int(id_) for id_ in ids if int(id_) not in id_index)
    if missing:
        raise ValueError(f"DSV4 stage ids missing from {name}: {missing}")
    return torch.tensor(
        [id_index[int(id_)] for id_ in ids],
        device=device,
        dtype=torch.long,
    )


def _row_by_id(*, ids: Sequence[int], name: str) -> dict[int, int]:
    row_by_id: dict[int, int] = {}
    for row, id_ in enumerate(ids):
        id_int = int(id_)
        if id_int in row_by_id:
            raise ValueError(f"DSV4 {name} contains duplicate id {id_int}")
        row_by_id[id_int] = row
    return row_by_id
