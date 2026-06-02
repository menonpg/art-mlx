from __future__ import annotations

import pytest
import torch

from art.megatron.dsv4 import (
    Dsv4MaterializedStage,
    Dsv4SparseBackwardResult,
    Dsv4SparseForwardResult,
    Dsv4StageForwardRecord,
    Dsv4StageKeyKind,
    compute_single_sink_grad,
    merge_materialized_stage_records,
    merge_single_sink_branch,
    merge_stage_outputs,
    replay_materialized_dsv4_attention_backward,
    run_materialized_dsv4_attention_forward,
)
import art.megatron.dsv4.cp_attention as cp_attention


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


def test_materialized_attention_forward_merges_partial_stages_and_sink_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = (_materialized_stage(0, (10, 11)), _materialized_stage(1, (11, 12)))
    stage_outputs = (
        _stage_tensor(((1.0, 2.0),)),
        _stage_tensor(((3.0, 4.0),)),
    )
    stage_lses = (
        _stage_lse(((2.0, 4.0),)),
        _stage_lse(((6.0, 8.0),)),
    )
    calls: list[tuple[torch.Tensor, float | None]] = []

    def fake_fwd(
        *,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk: torch.Tensor,
        scale: float | None = None,
    ) -> Dsv4SparseForwardResult:
        del kv, topk
        calls.append((attn_sink, scale))
        index = len(calls) - 1
        assert q.shape == stages[index].q_stage.shape
        return Dsv4SparseForwardResult(out=stage_outputs[index], lse=stage_lses[index])

    monkeypatch.setattr(cp_attention.sparse_kernel, "dsv4_sparse_fwd", fake_fwd)
    attn_sink = torch.log(torch.tensor([5.0, 7.0], dtype=torch.float64))

    result = run_materialized_dsv4_attention_forward(
        stages=stages,
        query_token_ids=(10, 11, 12),
        attn_sink=attn_sink,
        scale=0.5,
    )

    expected_real = _stage_tensor(((1.0, 2.6, 4.0),))
    expected_real_lse = _stage_lse(((2.0, 10.0, 8.0),))
    expected_out, expected_lse = merge_single_sink_branch(
        expected_real,
        expected_real_lse,
        attn_sink,
    )

    torch.testing.assert_close(result.real_out, expected_real)
    torch.testing.assert_close(result.real_lse, expected_real_lse)
    torch.testing.assert_close(result.out, expected_out)
    torch.testing.assert_close(result.lse, expected_lse)
    assert len(calls) == 2
    for disabled_sink, scale in calls:
        assert torch.isneginf(disabled_sink).all()
        assert scale == 0.5


def test_materialized_attention_backward_replays_global_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = (_materialized_stage(0, (10, 11)), _materialized_stage(1, (11, 12)))
    monkeypatch.setattr(
        cp_attention.sparse_kernel,
        "dsv4_sparse_fwd",
        _fake_forward_for_replay,
    )
    attn_sink = torch.log(torch.tensor([5.0, 7.0], dtype=torch.float64))
    forward = run_materialized_dsv4_attention_forward(
        stages=stages,
        query_token_ids=(10, 11, 12),
        attn_sink=attn_sink,
        scale=0.25,
    )
    grad_out = torch.arange(
        forward.out.numel(),
        dtype=forward.out.dtype,
    ).reshape_as(forward.out)
    bwd_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def fake_bwd(
        *,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk: torch.Tensor,
        global_out: torch.Tensor,
        grad_out: torch.Tensor,
        global_lse: torch.Tensor,
        scale: float | None = None,
    ) -> Dsv4SparseBackwardResult:
        del topk
        bwd_calls.append((attn_sink, global_out, grad_out, global_lse))
        stage_value = float(q[0, 0, 0, 0].item())
        assert scale == 0.25
        return Dsv4SparseBackwardResult(
            dq=torch.full_like(q, stage_value + 10.0),
            dkv=torch.full_like(kv, stage_value + 20.0),
            d_attn_sink=torch.full_like(attn_sink, 999.0),
        )

    monkeypatch.setattr(cp_attention.sparse_kernel, "dsv4_sparse_bwd", fake_bwd)

    replay = replay_materialized_dsv4_attention_backward(
        forward_result=forward,
        grad_out=grad_out,
    )

    assert len(replay.stage_records) == 2
    assert len(bwd_calls) == 2
    torch.testing.assert_close(bwd_calls[0][1], forward.out[:, [0, 1]])
    torch.testing.assert_close(bwd_calls[1][1], forward.out[:, [1, 2]])
    torch.testing.assert_close(bwd_calls[0][2], grad_out[:, [0, 1]])
    torch.testing.assert_close(bwd_calls[1][2], grad_out[:, [1, 2]])
    torch.testing.assert_close(bwd_calls[0][3], forward.lse[:, [0, 1]])
    torch.testing.assert_close(bwd_calls[1][3], forward.lse[:, [1, 2]])
    assert torch.isneginf(bwd_calls[0][0]).all()
    assert torch.isneginf(bwd_calls[1][0]).all()
    torch.testing.assert_close(
        replay.d_attn_sink,
        compute_single_sink_grad(
            grad_out=grad_out,
            global_out=forward.out,
            global_lse=forward.lse,
            attn_sink=attn_sink,
        ),
    )
    torch.testing.assert_close(
        replay.stage_records[0].dq_stage,
        torch.full_like(stages[0].q_stage, 10.0),
    )
    torch.testing.assert_close(
        replay.stage_records[1].dkv_stage,
        torch.full_like(stages[1].kv_stage, 21.0),
    )


def test_materialized_stage_merge_rejects_unknown_query_id() -> None:
    stage = _materialized_stage(0, (13,))
    record = Dsv4StageForwardRecord(
        materialized_stage=stage,
        out=torch.zeros_like(stage.q_stage),
        lse=torch.zeros(stage.q_stage.shape[:-1], dtype=stage.q_stage.dtype),
    )

    with pytest.raises(ValueError, match="missing from global query ids"):
        merge_materialized_stage_records(records=(record,), query_token_ids=(10, 11))


def _materialized_stage(
    stage_index: int, query_token_ids: tuple[int, ...]
) -> Dsv4MaterializedStage:
    q_stage = torch.full(
        (1, len(query_token_ids), 2, 3),
        float(stage_index),
        dtype=torch.float64,
    )
    kv_stage = torch.full((1, 2, 3), float(stage_index), dtype=torch.float64)
    return Dsv4MaterializedStage(
        stage_index=stage_index,
        query_token_ids=query_token_ids,
        q_stage=q_stage,
        kv_stage=kv_stage,
        topk_stage_local=torch.zeros(1, len(query_token_ids), 2, dtype=torch.long),
        raw_count=1,
        compressed_count=1,
        key_kinds=(Dsv4StageKeyKind.RAW, Dsv4StageKeyKind.COMPRESSED),
        key_global_ids=(stage_index, stage_index + 100),
    )


def _stage_tensor(values_by_batch: tuple[tuple[float, ...], ...]) -> torch.Tensor:
    rows = []
    for values in values_by_batch:
        rows.append(torch.tensor(values, dtype=torch.float64).view(len(values), 1, 1))
    return torch.stack(rows, dim=0).expand(-1, -1, 2, 3).contiguous()


def _stage_lse(values_by_batch: tuple[tuple[float, ...], ...]) -> torch.Tensor:
    rows = []
    for values in values_by_batch:
        rows.append(
            torch.log(torch.tensor(values, dtype=torch.float64)).view(len(values), 1)
        )
    return torch.stack(rows, dim=0).expand(-1, -1, 2).contiguous()


def _fake_forward_for_replay(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk: torch.Tensor,
    scale: float | None = None,
) -> Dsv4SparseForwardResult:
    del kv, attn_sink, topk, scale
    stage_index = int(q[0, 0, 0, 0].item())
    if stage_index == 0:
        return Dsv4SparseForwardResult(
            out=_stage_tensor(((1.0, 2.0),)),
            lse=_stage_lse(((2.0, 4.0),)),
        )
    return Dsv4SparseForwardResult(
        out=_stage_tensor(((3.0, 4.0),)),
        lse=_stage_lse(((6.0, 8.0),)),
    )
