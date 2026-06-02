from __future__ import annotations

from collections.abc import Sequence

import torch

from . import sparse_kernel
from .types import (
    Dsv4AttentionBackwardReplayResult,
    Dsv4AttentionForwardResult,
    Dsv4AttentionGradientResult,
    Dsv4GradientOwnerBucket,
    Dsv4MaterializedStage,
    Dsv4StageBackwardRecord,
    Dsv4StageForwardRecord,
    Dsv4StageKeyKind,
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
