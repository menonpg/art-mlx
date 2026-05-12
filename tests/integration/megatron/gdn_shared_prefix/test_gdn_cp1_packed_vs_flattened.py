from __future__ import annotations

import torch

from .cases import default_phase0_cases
from .metrics import (
    GDN_CORRECTNESS_DTYPE,
    MEAN_ABS_PCT_MISMATCH_THRESHOLD,
    MEAN_ABS_PCT_THRESHOLD,
    mean_abs_pct,
)
from .oracles import (
    ToyGdnConfig,
    ToyStatefulGdn,
    compare_toy_packed_to_flattened,
    compare_toy_packed_to_flattened_with_output_grad,
    run_toy_packed,
    run_toy_physical_stream,
)
from .packed_layout import build_phase0_packed_tensors


def test_toy_stateful_oracle_matches_flattened_grad_accumulation() -> None:
    torch.manual_seed(1234)
    config = ToyGdnConfig(hidden_size=8, conv_width=4)
    module = ToyStatefulGdn(config)
    case = next(
        case
        for case in default_phase0_cases(conv_width=4)
        if case.name == "ragged_family_mix"
    )
    tensors = build_phase0_packed_tensors(case)
    hidden = torch.randn(
        len(case.rows),
        case.sequence_length,
        config.hidden_size,
        dtype=GDN_CORRECTNESS_DTYPE,
    )

    metrics = compare_toy_packed_to_flattened(
        module,
        hidden,
        group_ids=tensors["group_ids"],
        parent_ids=tensors["parent_ids"],
        assistant_mask=tensors["assistant_mask"],
    )

    assert metrics.loss_mean_abs_pct <= MEAN_ABS_PCT_THRESHOLD
    assert metrics.output_mean_abs_pct <= MEAN_ABS_PCT_THRESHOLD
    assert metrics.hidden_grad_mean_abs_pct <= MEAN_ABS_PCT_THRESHOLD
    assert metrics.param_grad_mean_abs_pct <= MEAN_ABS_PCT_THRESHOLD

    real_mask = tensors["group_ids"] != -1
    output_grads = {
        "prefix_only": _expanded_output_mask(
            tensors["group_ids"] == tensors["parent_ids"], config.hidden_size
        ),
        "suffix_only": _expanded_output_mask(
            tensors["assistant_mask"], config.hidden_size
        ),
        "random_all_real_tokens": (
            torch.randn(
                hidden.shape,
                dtype=GDN_CORRECTNESS_DTYPE,
                generator=torch.Generator().manual_seed(4321),
            )
            * _expanded_output_mask(real_mask, config.hidden_size)
        ),
        "single_token_channel": _single_token_channel_grad(hidden, real_mask),
    }
    for name, output_grad in output_grads.items():
        metrics = compare_toy_packed_to_flattened_with_output_grad(
            module,
            hidden,
            group_ids=tensors["group_ids"],
            parent_ids=tensors["parent_ids"],
            output_grad=output_grad,
        )
        assert metrics.loss_mean_abs_pct <= MEAN_ABS_PCT_THRESHOLD, name
        assert metrics.output_mean_abs_pct <= MEAN_ABS_PCT_THRESHOLD, name
        assert metrics.hidden_grad_mean_abs_pct <= MEAN_ABS_PCT_THRESHOLD, name
        assert metrics.param_grad_mean_abs_pct <= MEAN_ABS_PCT_THRESHOLD, name


def test_toy_stateful_oracle_rejects_physical_stream() -> None:
    torch.manual_seed(5678)
    config = ToyGdnConfig(hidden_size=8, conv_width=4)
    module = ToyStatefulGdn(config)
    case = next(
        case
        for case in default_phase0_cases(conv_width=4)
        if case.name == "multi_family_repeated"
    )
    tensors = build_phase0_packed_tensors(case)
    hidden = torch.randn(
        len(case.rows),
        case.sequence_length,
        config.hidden_size,
        dtype=GDN_CORRECTNESS_DTYPE,
    )

    packed = run_toy_packed(
        module,
        hidden,
        group_ids=tensors["group_ids"],
        parent_ids=tensors["parent_ids"],
    )
    physical = run_toy_physical_stream(
        module,
        hidden,
        group_ids=tensors["group_ids"],
    )
    real_mask = tensors["group_ids"] != -1

    assert (
        mean_abs_pct(packed[real_mask], physical[real_mask])
        > MEAN_ABS_PCT_MISMATCH_THRESHOLD
    )


def _expanded_output_mask(mask: torch.Tensor, hidden_size: int) -> torch.Tensor:
    return (
        mask.unsqueeze(-1)
        .expand(*mask.shape, hidden_size)
        .to(dtype=GDN_CORRECTNESS_DTYPE)
    )


def _single_token_channel_grad(
    hidden: torch.Tensor, real_mask: torch.Tensor
) -> torch.Tensor:
    row, position = real_mask.nonzero()[real_mask.sum() // 2].tolist()
    output_grad = torch.zeros_like(hidden)
    output_grad[row, position, 0] = 1.0
    return output_grad
