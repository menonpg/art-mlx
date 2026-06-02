from __future__ import annotations

from collections.abc import Sequence

import torch

from . import sparse_kernel
from .types import (
    Dsv4AttentionBackwardReplayResult,
    Dsv4AttentionForwardResult,
    Dsv4MaterializedStage,
    Dsv4StageBackwardRecord,
    Dsv4StageForwardRecord,
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


def _row_by_id(*, ids: Sequence[int], name: str) -> dict[int, int]:
    row_by_id: dict[int, int] = {}
    for row, id_ in enumerate(ids):
        id_int = int(id_)
        if id_int in row_by_id:
            raise ValueError(f"DSV4 {name} contains duplicate id {id_int}")
        row_by_id[id_int] = row
    return row_by_id
