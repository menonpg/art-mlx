from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict
import torch

from . import sparse_kernel
from .comm import Dsv4TensorExchangeWork, launch_dsv4_tensor_exchange
from .types import (
    Dsv4AttentionBackwardReplayResult,
    Dsv4AttentionForwardResult,
    Dsv4AttentionGradientResult,
    Dsv4GradientOwnerBucket,
    Dsv4MaterializedStage,
    Dsv4StageBackwardRecord,
    Dsv4StageForwardRecord,
    Dsv4StageKeyKind,
    Dsv4TensorExchangePlan,
)


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
    """Merge two DSV4 CP stage outputs in natural-log LSE space.

    This is the same softmax merge algebra as generic attention CP, but it is
    DSV4-owned so the later sparse-kernel path can replay stage backward with
    global output/LSE instead of depending on autograd through the merge.
    """
    merged_lse = _safe_logaddexp(prev_lse, stage_lse)
    prev_weight = _safe_exp_diff(prev_lse, merged_lse).unsqueeze(-1)
    stage_weight = _safe_exp_diff(stage_lse, merged_lse).unsqueeze(-1)
    merged_out = prev_weight * prev_out + stage_weight * stage_out
    return _zero_invalid_rows(merged_out, merged_lse), merged_lse


def merge_stage_outputs(
    stage_outputs: Sequence[torch.Tensor],
    stage_lses: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge all real-key DSV4 CP stage outputs before the sink branch.

    Stage LSE tensors must already be converted to natural log. Miles TileLang
    writes log2 LSE, so the sparse-kernel wrapper must multiply by ln(2) before
    calling this function. Invalid rows are represented by LSE=-inf and produce
    exact zero output.
    """
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
    """Run materialized DSV4 CP stages and merge sink-once output.

    This helper owns sparse stage execution and replay metadata after tensors
    have already been materialized. It deliberately does not launch CP
    communication or reduce gradients; those remain separate production steps.
    Custom comm paths should stay eager and outside compiled regions.
    """
    if len(stages) == 0:
        raise ValueError("at least one materialized DSV4 stage is required")
    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    if not query_ids:
        raise ValueError("DSV4 materialized attention requires query_token_ids")

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
    """Create the eager bridge from exchanged DSV4 stages to sparse attention.

    Stage KV exchanges are custom eager communication. This wrapper deliberately
    keeps their wait/materialization boundary out of compiled regions, then uses
    the materialized-stage forward path for sparse kernel execution and global
    real-key plus sink merge.
    """
    works = tuple(stage_works)
    if not works:
        raise ValueError("at least one DSV4 exchanged attention stage is required")
    for position, work in enumerate(works):
        _validate_stage_exchange_work(work=work, position=position)
    query_ids = tuple(int(token_id) for token_id in query_token_ids)
    if not query_ids:
        raise ValueError("DSV4 exchanged attention requires query_token_ids")
    return Dsv4ExchangedAttentionForwardWork(
        stage_works=works,
        query_token_ids=query_ids,
        attn_sink=attn_sink,
        scale=scale,
    )


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
    """Merge the DSV4 sink exactly once after all real-key CP stages.

    CP stage kernels must run with sink disabled, otherwise every stage would add
    the same denominator term and inflate the sink probability. `attn_sink` is a
    per-head natural-logit tensor with shape [H].
    """
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
    """Replay sparse stage backward with global output/LSE.

    Each Miles backward receives the stage's query rows gathered from the
    globally merged output, global LSE, and external output gradient. Sink grad
    is computed once analytically from the global result.
    """
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
    """Accumulate replayed stage grads into explicit local id spaces.

    This is the local reduce step before distributed owner reduction. It sums
    per-stage query gradients by query token id, splits each `dkv_stage` into raw
    and compressed slices using the materialized-stage metadata, and sums those
    slices by raw token id or compressed entry id.
    """
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
    """Pack local DSV4 grads into owner-ranked buckets for eager communication.

    This helper only builds stable send payloads. The actual distributed send
    path should keep custom communication eager and make stream/lifetime
    ordering explicit.
    """
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
    """Launch distributed owner reduction payload exchange for DSV4 grads.

    This is an eager DSV4 communication boundary. Actual ids may repeat across
    source ranks, so the wire id includes the source rank; `wait_post_process`
    converts received rows back to ordinary `Dsv4GradientOwnerBucket`s and
    leaves summation to `accumulate_dsv4_gradient_owner_buckets`.
    """
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
            label="dsv4_gradient_owner_query_exchange",
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
            label="dsv4_gradient_owner_raw_exchange",
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
            label="dsv4_gradient_owner_compressed_exchange",
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
    """Reduce received owner buckets into this rank's explicit id spaces."""
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
    """Compute the per-head DSV4 sink gradient from global output and LSE.

    The sink has zero value, so dL/dsink = -Delta * p_sink where
    Delta = dot(global_out, grad_out). This must be computed once from the
    globally merged CP result, not per stage.
    """
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
