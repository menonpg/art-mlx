from __future__ import annotations

from collections.abc import Sequence

import torch


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
