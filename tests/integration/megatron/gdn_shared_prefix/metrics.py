from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..metrics import (
    DEFAULT_MEAN_ABS_PCT_THRESHOLD,
    mean_abs_pct,
    mean_abs_pct_from_sums,
)

# Testing design: the full model oracle remains fp32 and uses a narrow torch
# reference for the Qwen3.5 GDN recurrent math because the current FLA/TileLang
# stack has no valid fp32 GDN backward path. These real-GDN tests intentionally
# exercise the production bf16 kernels and CP machinery instead. Do not change
# this dtype/threshold split without discussing the oracle coverage tradeoff.
GDN_CORRECTNESS_DTYPE = torch.bfloat16
MEAN_ABS_PCT_THRESHOLD = DEFAULT_MEAN_ABS_PCT_THRESHOLD
MEAN_ABS_PCT_MISMATCH_THRESHOLD = 0.1
REAL_GDN_LOSS_MEAN_ABS_PCT_THRESHOLD = (
    3.0 if GDN_CORRECTNESS_DTYPE == torch.bfloat16 else MEAN_ABS_PCT_THRESHOLD
)
REAL_GDN_OUTPUT_MEAN_ABS_PCT_THRESHOLD = (
    3.0 if GDN_CORRECTNESS_DTYPE == torch.bfloat16 else MEAN_ABS_PCT_THRESHOLD
)
REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD = (
    5.0 if GDN_CORRECTNESS_DTYPE == torch.bfloat16 else MEAN_ABS_PCT_THRESHOLD
)


def assert_mean_abs_pct(
    reference: Tensor,
    candidate: Tensor,
    name: str,
    *,
    threshold: float = MEAN_ABS_PCT_THRESHOLD,
) -> None:
    pct = mean_abs_pct(reference, candidate)
    assert pct <= threshold, f"{name}: mean_abs_pct={pct:.6g}% > {threshold}%"


def assert_scalar_loss_close(
    reference: Tensor,
    candidate: Tensor,
    name: str,
    *,
    threshold: float = REAL_GDN_LOSS_MEAN_ABS_PCT_THRESHOLD,
) -> None:
    pct = mean_abs_pct(reference, candidate)
    assert pct <= threshold, f"{name}: mean_abs_pct={pct:.6g}% > {threshold}%"


def assert_real_gdn_metrics(metrics: Any, name: str) -> None:
    assert metrics.loss_mean_abs_pct <= REAL_GDN_LOSS_MEAN_ABS_PCT_THRESHOLD, (
        f"{name}: loss_mean_abs_pct={metrics.loss_mean_abs_pct:.6g}% > "
        f"{REAL_GDN_LOSS_MEAN_ABS_PCT_THRESHOLD}%"
    )
    assert metrics.output_mean_abs_pct <= REAL_GDN_OUTPUT_MEAN_ABS_PCT_THRESHOLD, (
        f"{name}: output_mean_abs_pct={metrics.output_mean_abs_pct:.6g}% > "
        f"{REAL_GDN_OUTPUT_MEAN_ABS_PCT_THRESHOLD}%"
    )
    assert metrics.hidden_grad_mean_abs_pct <= REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD, (
        f"{name}: hidden_grad_mean_abs_pct={metrics.hidden_grad_mean_abs_pct:.6g}% > "
        f"{REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD}%"
    )
    assert metrics.param_grad_mean_abs_pct <= REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD, (
        f"{name}: param_grad_mean_abs_pct={metrics.param_grad_mean_abs_pct:.6g}% > "
        f"{REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD}%"
    )


def parameter_grad_mean_abs_pct_with_name(
    reference: torch.nn.Module,
    candidate: torch.nn.Module,
) -> tuple[str, float]:
    worst_name = ""
    worst_pct = 0.0
    abs_diff_sum = 0.0
    reference_abs_sum = 0.0
    numel = 0
    candidate_params = dict(candidate.named_parameters())
    for name, reference_param in reference.named_parameters():
        candidate_param = candidate_params[name]
        reference_grad = parameter_grad(reference_param)
        candidate_grad = parameter_grad(candidate_param)
        if reference_grad is None and candidate_grad is None:
            continue
        if reference_grad is None or candidate_grad is None:
            raise AssertionError(f"mismatched parameter grad presence for {name}")
        pct = mean_abs_pct(reference_grad, candidate_grad)
        if pct > worst_pct:
            worst_name = name
            worst_pct = pct
        reference_grad_fp32 = reference_grad.detach().float()
        candidate_grad_fp32 = candidate_grad.detach().float()
        abs_diff_sum += float(
            (candidate_grad_fp32 - reference_grad_fp32).abs().sum().item()
        )
        reference_abs_sum += float(reference_grad_fp32.abs().sum().item())
        numel += int(reference_grad_fp32.numel())
    if numel == 0:
        return worst_name, 0.0
    return worst_name, mean_abs_pct_from_sums(abs_diff_sum, reference_abs_sum, numel)


def assert_parameter_grad_mean_abs_pct(
    reference: torch.nn.Module,
    candidate: torch.nn.Module,
    name: str,
    *,
    threshold: float = MEAN_ABS_PCT_THRESHOLD,
) -> None:
    param_name, pct = parameter_grad_mean_abs_pct_with_name(reference, candidate)
    assert pct <= threshold, (
        f"{name}:{param_name}: mean_abs_pct={pct:.6g}% > {threshold}%"
    )


def parameter_grad(parameter: torch.nn.Parameter) -> Tensor | None:
    main_grad = getattr(parameter, "main_grad", None)
    if parameter.grad is not None and main_grad is not None:
        if not getattr(parameter, "grad_added_to_main_grad", False) or getattr(
            parameter, "zero_out_wgrad", False
        ):
            return main_grad + parameter.grad.to(dtype=main_grad.dtype)
        return main_grad
    if parameter.grad is not None:
        return parameter.grad
    if main_grad is not None:
        return main_grad
    return None
