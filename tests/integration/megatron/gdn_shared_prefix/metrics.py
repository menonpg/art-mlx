from __future__ import annotations

import torch
from torch import Tensor

GDN_CORRECTNESS_DTYPE = torch.float32
MEAN_ABS_PCT_THRESHOLD = 0.5
MEAN_ABS_PCT_DENOMINATOR_EPS = 1e-18


def mean_abs_pct(reference: Tensor, candidate: Tensor) -> float:
    abs_pct = elementwise_abs_pct(reference, candidate)
    if abs_pct.numel() == 0:
        return 0.0
    return float((abs_pct.mean() * 100.0).item())


def elementwise_abs_pct(reference: Tensor, candidate: Tensor) -> Tensor:
    reference_fp32 = reference.detach().float()
    candidate_fp32 = candidate.detach().float()
    return (candidate_fp32 - reference_fp32).abs() / reference_fp32.abs().clamp_min(
        MEAN_ABS_PCT_DENOMINATOR_EPS
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


def parameter_grad_mean_abs_pct_with_name(
    reference: torch.nn.Module,
    candidate: torch.nn.Module,
) -> tuple[str, float]:
    worst_name = ""
    worst_pct = 0.0
    abs_pct_sum = 0.0
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
        abs_pct = elementwise_abs_pct(reference_grad, candidate_grad)
        pct = float((abs_pct.mean() * 100.0).item())
        if pct > worst_pct:
            worst_name = name
            worst_pct = pct
        abs_pct_sum += float(abs_pct.sum().item())
        numel += int(abs_pct.numel())
    if numel == 0:
        return worst_name, 0.0
    return worst_name, (abs_pct_sum / numel) * 100.0


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
