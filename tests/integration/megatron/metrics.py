from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_MEAN_ABS_PCT_THRESHOLD = 1.0
MEAN_ABS_PCT_DENOMINATOR_EPS = 1e-18


def mean_abs_pct_from_sums(
    abs_diff_sum: float,
    reference_abs_sum: float,
    numel: int,
) -> float:
    if numel == 0:
        return 0.0
    mean_abs_diff = abs_diff_sum / numel
    mean_abs_reference = reference_abs_sum / numel
    return (mean_abs_diff / (mean_abs_reference + MEAN_ABS_PCT_DENOMINATOR_EPS)) * 100.0


def mean_abs_pct(reference: Tensor, candidate: Tensor) -> float:
    reference_fp32 = reference.detach().float()
    candidate_fp32 = candidate.detach().float()
    diff = (candidate_fp32 - reference_fp32).abs()
    return mean_abs_pct_from_sums(
        float(diff.sum().item()),
        float(reference_fp32.abs().sum().item()),
        int(diff.numel()),
    )
