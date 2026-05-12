from __future__ import annotations

import torch
from torch import Tensor

from ..metrics import (
    DEFAULT_MEAN_ABS_PCT_THRESHOLD,
    mean_abs_pct,
    mean_abs_pct_from_sums,
)

GDN_CORRECTNESS_DTYPE = torch.float32
MEAN_ABS_PCT_THRESHOLD = DEFAULT_MEAN_ABS_PCT_THRESHOLD
MEAN_ABS_PCT_MISMATCH_THRESHOLD = 0.1


def assert_mean_abs_pct(
    reference: Tensor,
    candidate: Tensor,
    name: str,
    *,
    threshold: float = MEAN_ABS_PCT_THRESHOLD,
) -> None:
    pct = mean_abs_pct(reference, candidate)
    assert pct <= threshold, f"{name}: mean_abs_pct={pct:.6g}% > {threshold}%"


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
