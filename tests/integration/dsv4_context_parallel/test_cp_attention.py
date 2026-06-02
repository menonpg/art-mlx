from __future__ import annotations

import pytest
import torch

from art.megatron.dsv4 import (
    compute_single_sink_grad,
    merge_single_sink_branch,
    merge_stage_outputs,
)


def _dense_attention_stats(
    logits: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lse = torch.logsumexp(logits, dim=-1)
    diff = logits - lse.unsqueeze(-1)
    both_neg_inf = torch.isneginf(logits) & torch.isneginf(lse).unsqueeze(-1)
    diff = torch.where(both_neg_inf, torch.full_like(diff, float("-inf")), diff)
    weights = torch.exp(diff)
    out = (weights.unsqueeze(-1) * values).sum(dim=-2)
    return torch.where(
        torch.isneginf(lse).unsqueeze(-1), torch.zeros_like(out), out
    ), lse


def test_merge_stage_outputs_matches_dense_split_softmax() -> None:
    torch.manual_seed(17)
    bsz, q_len, heads, dim = 2, 4, 3, 5
    key_counts = (4, 3, 2)
    stage_logits = [
        torch.randn(bsz, q_len, heads, key_count, dtype=torch.float64)
        for key_count in key_counts
    ]
    stage_values = [
        torch.randn(bsz, q_len, heads, key_count, dim, dtype=torch.float64)
        for key_count in key_counts
    ]

    stage_logits[0][0, 1, 2].fill_(float("-inf"))
    for logits in stage_logits:
        logits[1, 2, 0].fill_(float("-inf"))

    stage_outputs: list[torch.Tensor] = []
    stage_lses: list[torch.Tensor] = []
    for logits, values in zip(stage_logits, stage_values):
        stage_out, stage_lse = _dense_attention_stats(logits, values)
        stage_outputs.append(stage_out)
        stage_lses.append(stage_lse)

    dense_out, dense_lse = _dense_attention_stats(
        torch.cat(stage_logits, dim=-1),
        torch.cat(stage_values, dim=-2),
    )
    merged_out, merged_lse = merge_stage_outputs(stage_outputs, stage_lses)

    torch.testing.assert_close(merged_lse, dense_lse)
    torch.testing.assert_close(merged_out, dense_out)
    assert torch.isneginf(merged_lse[1, 2, 0])
    torch.testing.assert_close(
        merged_out[1, 2, 0], torch.zeros(dim, dtype=torch.float64)
    )


def test_merge_stage_outputs_requires_real_stage() -> None:
    with pytest.raises(ValueError, match="at least one"):
        merge_stage_outputs([], [])


def test_merge_single_sink_matches_dense_zero_value_sink() -> None:
    torch.manual_seed(19)
    bsz, q_len, heads, dim, key_count = 2, 5, 4, 6, 7
    real_logits = torch.randn(bsz, q_len, heads, key_count, dtype=torch.float64)
    real_values = torch.randn(bsz, q_len, heads, key_count, dim, dtype=torch.float64)
    real_logits[0, 3, 1].fill_(float("-inf"))
    attn_sink = torch.randn(heads, dtype=torch.float64)

    real_out, real_lse = _dense_attention_stats(real_logits, real_values)
    global_out, global_lse = merge_single_sink_branch(real_out, real_lse, attn_sink)

    sink_logits = attn_sink.view(1, 1, heads, 1).expand(bsz, q_len, heads, 1)
    sink_values = torch.zeros(bsz, q_len, heads, 1, dim, dtype=torch.float64)
    dense_out, dense_lse = _dense_attention_stats(
        torch.cat((real_logits, sink_logits), dim=-1),
        torch.cat((real_values, sink_values), dim=-2),
    )

    torch.testing.assert_close(global_lse, dense_lse)
    torch.testing.assert_close(global_out, dense_out)
    torch.testing.assert_close(
        global_out[0, 3, 1], torch.zeros(dim, dtype=torch.float64)
    )
    torch.testing.assert_close(global_lse[0, 3, 1], attn_sink[1])


def test_compute_single_sink_grad_matches_autograd_reference() -> None:
    torch.manual_seed(23)
    bsz, q_len, heads, dim, key_count = 2, 3, 4, 5, 6
    real_logits = torch.randn(bsz, q_len, heads, key_count, dtype=torch.float64)
    real_values = torch.randn(bsz, q_len, heads, key_count, dim, dtype=torch.float64)
    real_logits[1, 1, 2].fill_(float("-inf"))
    grad_out = torch.randn(bsz, q_len, heads, dim, dtype=torch.float64)
    attn_sink = torch.randn(heads, dtype=torch.float64, requires_grad=True)

    sink_logits = attn_sink.view(1, 1, heads, 1).expand(bsz, q_len, heads, 1)
    sink_values = torch.zeros(bsz, q_len, heads, 1, dim, dtype=torch.float64)
    dense_out, _ = _dense_attention_stats(
        torch.cat((real_logits, sink_logits), dim=-1),
        torch.cat((real_values, sink_values), dim=-2),
    )
    loss = (dense_out * grad_out).sum()
    loss.backward()

    real_out, real_lse = _dense_attention_stats(
        real_logits.detach(), real_values.detach()
    )
    global_out, global_lse = merge_single_sink_branch(
        real_out, real_lse, attn_sink.detach()
    )
    sink_grad = compute_single_sink_grad(
        grad_out,
        global_out,
        global_lse,
        attn_sink.detach(),
    )

    torch.testing.assert_close(sink_grad, attn_sink.grad)
