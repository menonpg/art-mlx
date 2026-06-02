from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict
import pytest
import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.multiprocessing as mp

from art.megatron.dsv4 import (
    Dsv4AttentionBackwardReplayResult,
    Dsv4AttentionGradientResult,
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    Dsv4ExchangedAttentionBackwardWork,
    Dsv4GradientOwnerBucket,
    Dsv4MaterializedStage,
    Dsv4SparseBackwardResult,
    Dsv4SparseForwardResult,
    Dsv4StageBackwardRecord,
    Dsv4StageForwardRecord,
    Dsv4StageKeyKind,
    Dsv4StagePlanGroup,
    accumulate_dsv4_gradient_owner_buckets,
    accumulate_materialized_dsv4_attention_backward,
    build_dsv4_compressed_layout,
    build_dsv4_stage_plan_slots,
    build_stage_local_topk_for_csa,
    compute_single_sink_grad,
    launch_dsv4_attention_backward_from_stage_plan_slots,
    launch_dsv4_attention_forward_from_stage_plan_groups,
    launch_dsv4_csa_attention_forward_from_stage_plan_slots,
    launch_dsv4_hca_attention_forward_from_stage_plan_slots,
    launch_exchanged_dsv4_attention_backward,
    launch_exchanged_dsv4_attention_forward,
    merge_materialized_stage_records,
    merge_single_sink_branch,
    merge_stage_outputs,
    pack_dsv4_gradient_owner_buckets,
    replay_materialized_dsv4_attention_backward,
    run_materialized_dsv4_attention_forward,
)
import art.megatron.dsv4.cp_attention as cp_attention


class _LayoutIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    ownership_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    token_counts_by_rank: tuple[int, ...]


class _Range(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int
    end: int


class _StagePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage_index: int
    global_q_ranges: tuple[_Range, ...]
    global_k_ranges: tuple[_Range, ...]


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


def test_exchanged_attention_forward_materializes_stages_then_merges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = (_materialized_stage(0, (10, 11)), _materialized_stage(1, (11, 12)))
    stage_works = tuple(_FakeStageExchangeWork(stage) for stage in stages)
    monkeypatch.setattr(
        cp_attention.sparse_kernel,
        "dsv4_sparse_fwd",
        _fake_forward_for_replay,
    )
    attn_sink = torch.log(torch.tensor([5.0, 7.0], dtype=torch.float64))

    work = launch_exchanged_dsv4_attention_forward(
        stage_works=stage_works,
        query_token_ids=(10, 11, 12),
        attn_sink=attn_sink,
        scale=0.25,
    )
    work.wait()
    result = work.wait_post_process()

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
    for stage_work in stage_works:
        assert stage_work.wait_count == 1
        assert stage_work.post_process_count == 1


def test_stage_plan_group_attention_forward_launcher_assembles_stage_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cp_attention.sparse_kernel,
        "dsv4_sparse_fwd",
        _fake_forward_for_replay,
    )
    layout = _single_rank_layout()
    stage = build_stage_local_topk_for_csa(
        layout=layout,
        stage_index=0,
        query_token_ids=(3, 7),
        global_k_ranges=(_Range(start=0, end=8),),
        global_topk=torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long),
        window_size=4,
    )
    attn_sink = torch.log(torch.tensor([5.0, 7.0], dtype=torch.float64))

    work = launch_dsv4_attention_forward_from_stage_plan_groups(
        layout=layout,
        rank=0,
        stage_plan_groups=(
            Dsv4StagePlanGroup(stage_index=0, stage_inputs_by_rank=(stage,)),
        ),
        query=torch.zeros(2, 2, 3, dtype=torch.float64),
        query_token_ids=(3, 7),
        raw_kv=torch.zeros(8, 3, dtype=torch.float64),
        raw_token_ids=tuple(range(8)),
        compressed_kv=torch.zeros(2, 3, dtype=torch.float64),
        compressed_entry_ids=(0, 1),
        attn_sink=attn_sink,
        group=None,
        async_op=True,
        scale=0.25,
    )
    result = work.wait_post_process()

    expected_out, expected_lse = merge_single_sink_branch(
        _stage_tensor(((1.0, 2.0),)),
        _stage_lse(((2.0, 4.0),)),
        attn_sink,
    )
    torch.testing.assert_close(result.out, expected_out)
    torch.testing.assert_close(result.lse, expected_lse)
    assert result.stage_records[0].materialized_stage.key_global_ids == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
    )


def test_csa_attention_forward_launcher_uses_local_topk_and_stage_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cp_attention.sparse_kernel,
        "dsv4_sparse_fwd",
        _fake_forward_for_replay,
    )
    layout = _single_rank_layout(Dsv4CompressionKind.CSA)
    slots = build_dsv4_stage_plan_slots(
        stage_plans_by_rank=(
            (
                _stage_plan(
                    stage_index=0,
                    q_ranges=((3, 4), (7, 8)),
                    k_ranges=((0, 8),),
                ),
            ),
        ),
    )
    attn_sink = torch.log(torch.tensor([5.0, 7.0], dtype=torch.float64))
    original_tolist = torch.Tensor.tolist

    def fail_tolist(tensor: torch.Tensor) -> list[object]:
        raise AssertionError(f"unexpected Tensor.tolist on shape {tuple(tensor.shape)}")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)
    try:
        work = launch_dsv4_csa_attention_forward_from_stage_plan_slots(
            layout=layout,
            rank=0,
            stage_plan_slots=slots,
            query=torch.zeros(2, 2, 3, dtype=torch.float64),
            query_token_ids=(3, 7),
            raw_kv=torch.zeros(8, 3, dtype=torch.float64),
            raw_token_ids=tuple(range(8)),
            compressed_kv=torch.zeros(2, 3, dtype=torch.float64),
            compressed_entry_ids=(0, 1),
            indexer_q=torch.tensor(
                [[[1.0, 0.0]], [[1.0, 0.0]]],
                dtype=torch.float32,
            ),
            indexer_weights=torch.ones(2, 1, dtype=torch.float32),
            indexer_kv=torch.tensor(
                [[2.0, 0.0], [3.0, 0.0]],
                dtype=torch.float32,
            ),
            indexer_kv_entry_ids=(0, 1),
            indexer_topk=2,
            attn_sink=attn_sink,
            group=None,
            async_op=True,
            scale=0.25,
            window_size=4,
        )
        result = work.wait_post_process()
    finally:
        monkeypatch.setattr(torch.Tensor, "tolist", original_tolist)

    expected_out, expected_lse = merge_single_sink_branch(
        _stage_tensor(((1.0, 2.0),)),
        _stage_lse(((2.0, 4.0),)),
        attn_sink,
    )
    torch.testing.assert_close(result.out, expected_out)
    torch.testing.assert_close(result.lse, expected_lse)
    stage = result.stage_records[0].materialized_stage
    assert stage.query_token_ids == (3, 7)
    assert stage.raw_count == 8
    assert stage.compressed_count == 2
    assert stage.key_global_ids == (0, 1, 2, 3, 4, 5, 6, 7, 0, 1)


def test_hca_attention_forward_launcher_uses_stage_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cp_attention.sparse_kernel,
        "dsv4_sparse_fwd",
        _fake_forward_for_replay,
    )
    layout = _single_rank_layout(Dsv4CompressionKind.HCA)
    slots = build_dsv4_stage_plan_slots(
        stage_plans_by_rank=(
            (
                _stage_plan(
                    stage_index=0,
                    q_ranges=((3, 4), (7, 8)),
                    k_ranges=((0, 8),),
                ),
            ),
        ),
    )
    attn_sink = torch.log(torch.tensor([5.0, 7.0], dtype=torch.float64))

    work = launch_dsv4_hca_attention_forward_from_stage_plan_slots(
        layout=layout,
        rank=0,
        stage_plan_slots=slots,
        query=torch.zeros(2, 2, 3, dtype=torch.float64),
        query_token_ids=(3, 7),
        raw_kv=torch.zeros(8, 3, dtype=torch.float64),
        raw_token_ids=tuple(range(8)),
        compressed_kv=torch.zeros(2, 3, dtype=torch.float64),
        compressed_entry_ids=(0, 1),
        attn_sink=attn_sink,
        group=None,
        async_op=True,
        scale=0.25,
        window_size=4,
    )
    result = work.wait_post_process()

    expected_out, expected_lse = merge_single_sink_branch(
        _stage_tensor(((1.0, 2.0),)),
        _stage_lse(((2.0, 4.0),)),
        attn_sink,
    )
    torch.testing.assert_close(result.out, expected_out)
    torch.testing.assert_close(result.lse, expected_lse)
    stage = result.stage_records[0].materialized_stage
    assert stage.query_token_ids == (3, 7)
    assert stage.raw_count == 8
    assert stage.compressed_count == 2


def test_exchanged_attention_forward_rejects_bad_stage_work() -> None:
    attn_sink = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="missing wait"):
        launch_exchanged_dsv4_attention_forward(
            stage_works=(object(),),
            query_token_ids=(10,),
            attn_sink=attn_sink,
        )

    work = launch_exchanged_dsv4_attention_forward(
        stage_works=(_BadStageExchangeWork(),),
        query_token_ids=(10,),
        attn_sink=attn_sink,
    )
    with pytest.raises(TypeError, match="expected Dsv4MaterializedStage"):
        work.wait_post_process()


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


def test_exchanged_attention_backward_replays_and_reduces_owner_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = (
        _materialized_stage(0, (10, 11), raw_id=20, compressed_id=100),
        _materialized_stage(1, (11, 12), raw_id=20, compressed_id=101),
    )
    monkeypatch.setattr(
        cp_attention.sparse_kernel,
        "dsv4_sparse_fwd",
        _fake_forward_for_replay,
    )
    monkeypatch.setattr(
        cp_attention.sparse_kernel,
        "dsv4_sparse_bwd",
        _fake_backward_for_bridge,
    )
    attn_sink = torch.log(torch.tensor([5.0, 7.0], dtype=torch.float64))
    forward = run_materialized_dsv4_attention_forward(
        stages=stages,
        query_token_ids=(10, 11, 12),
        attn_sink=attn_sink,
        scale=0.25,
    )
    grad_out = torch.arange(forward.out.numel(), dtype=forward.out.dtype).reshape_as(
        forward.out
    )

    work = launch_exchanged_dsv4_attention_backward(
        forward_result=forward,
        grad_out=grad_out,
        query_token_ids=(10, 11, 12),
        raw_token_ids=(20,),
        compressed_entry_ids=(100, 101),
        query_owner_ranks=(0, 0, 0),
        raw_owner_ranks=(0,),
        compressed_owner_ranks=(0, 0),
        recv_query_token_ids_by_peer=((10, 11, 12),),
        recv_raw_token_ids_by_peer=((20,),),
        recv_compressed_entry_ids_by_peer=((100, 101),),
        owned_query_token_ids=(10, 11, 12),
        owned_raw_token_ids=(20,),
        owned_compressed_entry_ids=(100, 101),
        rank=0,
        rank_count=1,
        group=None,
        async_op=True,
    )
    result = work.wait_post_process()

    torch.testing.assert_close(result.dq, _stage_tensor(((10.0, 21.0, 11.0),)))
    torch.testing.assert_close(
        result.draw_kv,
        torch.full((1, 1, 3), 140.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.dcompressed_kv,
        torch.tensor(
            [[[21.0, 21.0, 21.0], [121.0, 121.0, 121.0]]], dtype=torch.float64
        ),
    )
    torch.testing.assert_close(
        result.d_attn_sink,
        compute_single_sink_grad(
            grad_out=grad_out,
            global_out=forward.out,
            global_lse=forward.lse,
            attn_sink=attn_sink,
        ),
    )


def test_attention_backward_launcher_uses_stage_plan_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cp_attention.sparse_kernel,
        "dsv4_sparse_fwd",
        _fake_forward_for_replay,
    )
    monkeypatch.setattr(
        cp_attention.sparse_kernel,
        "dsv4_sparse_bwd",
        _fake_backward_for_bridge,
    )
    layout = _single_rank_layout(Dsv4CompressionKind.CSA)
    slots = build_dsv4_stage_plan_slots(
        stage_plans_by_rank=(
            (
                _stage_plan(
                    stage_index=0,
                    q_ranges=((3, 4), (7, 8)),
                    k_ranges=((0, 8),),
                ),
            ),
        ),
    )
    attn_sink = torch.log(torch.tensor([5.0, 7.0], dtype=torch.float64))
    forward = launch_dsv4_csa_attention_forward_from_stage_plan_slots(
        layout=layout,
        rank=0,
        stage_plan_slots=slots,
        query=torch.zeros(2, 2, 3, dtype=torch.float64),
        query_token_ids=(3, 7),
        raw_kv=torch.zeros(8, 3, dtype=torch.float64),
        raw_token_ids=tuple(range(8)),
        compressed_kv=torch.zeros(2, 3, dtype=torch.float64),
        compressed_entry_ids=(0, 1),
        indexer_q=torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=torch.float32),
        indexer_weights=torch.ones(2, 1, dtype=torch.float32),
        indexer_kv=torch.tensor([[2.0, 0.0], [3.0, 0.0]], dtype=torch.float32),
        indexer_kv_entry_ids=(0, 1),
        indexer_topk=2,
        attn_sink=attn_sink,
        group=None,
        async_op=True,
        scale=0.25,
        window_size=4,
    ).wait_post_process()
    grad_out = torch.arange(forward.out.numel(), dtype=forward.out.dtype).reshape_as(
        forward.out
    )

    work = launch_dsv4_attention_backward_from_stage_plan_slots(
        layout=layout,
        rank=0,
        stage_plan_slots=slots,
        forward_result=forward,
        grad_out=grad_out,
        group=None,
        async_op=True,
    )
    result = work.wait_post_process()

    torch.testing.assert_close(result.dq, _stage_tensor(((10.0, 10.0),)))
    torch.testing.assert_close(
        result.draw_kv,
        _kv_rows((20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0)),
    )
    torch.testing.assert_close(
        result.dcompressed_kv,
        _kv_rows((28.0, 29.0)),
    )
    torch.testing.assert_close(
        result.d_attn_sink,
        compute_single_sink_grad(
            grad_out=grad_out,
            global_out=forward.out,
            global_lse=forward.lse,
            attn_sink=attn_sink,
        ),
    )


def test_attention_backward_launcher_reduces_owner_grads_from_stage_plan_slots(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_attention_slot_backward_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _slot_backward_worker,
        args=(2, str(init_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def test_exchanged_attention_backward_handles_empty_owner_receives() -> None:
    template = Dsv4AttentionGradientResult(
        query_token_ids=(),
        raw_token_ids=(),
        compressed_entry_ids=(),
        dq=torch.empty(1, 0, 2, 3, dtype=torch.float64),
        draw_kv=torch.empty(1, 0, 3, dtype=torch.float64),
        dcompressed_kv=torch.empty(1, 0, 3, dtype=torch.float64),
        d_attn_sink=torch.tensor([3.0, 4.0], dtype=torch.float64),
    )
    owner_work = _EmptyOwnerExchangeWork()
    work = Dsv4ExchangedAttentionBackwardWork(
        local_gradients=template,
        owner_work=owner_work,
        owned_query_token_ids=(10, 11),
        owned_raw_token_ids=(20,),
        owned_compressed_entry_ids=(100,),
    )

    work.wait()
    result = work.wait_post_process()

    assert owner_work.wait_count == 1
    assert owner_work.post_process_count == 1
    assert result.query_token_ids == (10, 11)
    assert result.raw_token_ids == (20,)
    assert result.compressed_entry_ids == (100,)
    torch.testing.assert_close(result.dq, torch.zeros(1, 2, 2, 3, dtype=torch.float64))
    torch.testing.assert_close(
        result.draw_kv, torch.zeros(1, 1, 3, dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.dcompressed_kv, torch.zeros(1, 1, 3, dtype=torch.float64)
    )
    torch.testing.assert_close(result.d_attn_sink, template.d_attn_sink)


def test_materialized_stage_merge_rejects_unknown_query_id() -> None:
    stage = _materialized_stage(0, (13,))
    record = Dsv4StageForwardRecord(
        materialized_stage=stage,
        out=torch.zeros_like(stage.q_stage),
        lse=torch.zeros(stage.q_stage.shape[:-1], dtype=stage.q_stage.dtype),
    )

    with pytest.raises(ValueError, match="missing from global query ids"):
        merge_materialized_stage_records(records=(record,), query_token_ids=(10, 11))


def test_accumulate_materialized_backward_sums_by_explicit_ids() -> None:
    stage0 = _materialized_stage(0, (10, 11), raw_id=20, compressed_id=100)
    stage1 = _materialized_stage(1, (11, 12), raw_id=20, compressed_id=101)
    replay = Dsv4AttentionBackwardReplayResult(
        stage_records=(
            _stage_backward_record(
                stage0, dq_value=1.0, raw_value=10.0, compressed_value=100.0
            ),
            _stage_backward_record(
                stage1, dq_value=2.0, raw_value=20.0, compressed_value=200.0
            ),
        ),
        d_attn_sink=torch.tensor([3.0, 4.0], dtype=torch.float64),
    )

    result = accumulate_materialized_dsv4_attention_backward(
        replay_result=replay,
        query_token_ids=(10, 11, 12),
        raw_token_ids=(20,),
        compressed_entry_ids=(100, 101),
    )

    torch.testing.assert_close(result.dq, _stage_tensor(((1.0, 3.0, 2.0),)))
    torch.testing.assert_close(
        result.draw_kv,
        torch.full((1, 1, 3), 30.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.dcompressed_kv,
        torch.tensor(
            [[[100.0, 100.0, 100.0], [200.0, 200.0, 200.0]]], dtype=torch.float64
        ),
    )
    torch.testing.assert_close(result.d_attn_sink, replay.d_attn_sink)
    assert result.query_token_ids == (10, 11, 12)
    assert result.raw_token_ids == (20,)
    assert result.compressed_entry_ids == (100, 101)


def test_accumulate_materialized_backward_rejects_bad_id_spaces() -> None:
    stage = _materialized_stage(0, (10,), raw_id=20, compressed_id=100)
    replay = Dsv4AttentionBackwardReplayResult(
        stage_records=(
            _stage_backward_record(
                stage, dq_value=1.0, raw_value=10.0, compressed_value=100.0
            ),
        ),
        d_attn_sink=torch.tensor([3.0, 4.0], dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="missing from raw_token_ids"):
        accumulate_materialized_dsv4_attention_backward(
            replay_result=replay,
            query_token_ids=(10,),
            raw_token_ids=(21,),
            compressed_entry_ids=(100,),
        )
    with pytest.raises(ValueError, match="raw_token_ids contains duplicate"):
        accumulate_materialized_dsv4_attention_backward(
            replay_result=replay,
            query_token_ids=(10,),
            raw_token_ids=(20, 20),
            compressed_entry_ids=(100,),
        )


def test_pack_gradient_owner_buckets_groups_by_owner_rank() -> None:
    gradients = _gradient_result()

    buckets = pack_dsv4_gradient_owner_buckets(
        gradients=gradients,
        query_owner_ranks=(1, 0, 1),
        raw_owner_ranks=(0, 1),
        compressed_owner_ranks=(1, 0),
    )

    assert tuple(bucket.owner_rank for bucket in buckets) == (0, 1)
    bucket0, bucket1 = buckets
    assert bucket0.query_token_ids == (11,)
    assert bucket0.raw_token_ids == (20,)
    assert bucket0.compressed_entry_ids == (101,)
    assert bucket1.query_token_ids == (10, 12)
    assert bucket1.raw_token_ids == (21,)
    assert bucket1.compressed_entry_ids == (100,)
    torch.testing.assert_close(bucket0.dq, gradients.dq[:, [1]])
    torch.testing.assert_close(bucket1.draw_kv, gradients.draw_kv[:, [1]])
    torch.testing.assert_close(bucket0.dcompressed_kv, gradients.dcompressed_kv[:, [1]])


def test_gradient_owner_buckets_support_partial_empty_sections() -> None:
    gradients = _gradient_result()

    buckets = pack_dsv4_gradient_owner_buckets(
        gradients=gradients,
        query_owner_ranks=(0, 0, 0),
        raw_owner_ranks=(1, 1),
        compressed_owner_ranks=(2, 2),
    )

    assert tuple(bucket.owner_rank for bucket in buckets) == (0, 1, 2)
    query_bucket, raw_bucket, compressed_bucket = buckets
    assert raw_bucket.query_token_ids == ()
    assert compressed_bucket.raw_token_ids == ()
    assert query_bucket.compressed_entry_ids == ()
    assert raw_bucket.dq.shape[1] == 0
    assert compressed_bucket.draw_kv.shape[1] == 0
    assert query_bucket.dcompressed_kv.shape[1] == 0

    reduced = accumulate_dsv4_gradient_owner_buckets(
        buckets=buckets,
        query_token_ids=gradients.query_token_ids,
        raw_token_ids=gradients.raw_token_ids,
        compressed_entry_ids=gradients.compressed_entry_ids,
        d_attn_sink=gradients.d_attn_sink,
    )

    torch.testing.assert_close(reduced.dq, gradients.dq)
    torch.testing.assert_close(reduced.draw_kv, gradients.draw_kv)
    torch.testing.assert_close(reduced.dcompressed_kv, gradients.dcompressed_kv)


def test_accumulate_gradient_owner_buckets_sums_received_duplicates() -> None:
    bucket0 = _owner_bucket(
        owner_rank=0,
        query_ids=(10, 11),
        raw_ids=(20,),
        compressed_ids=(100,),
        value=1.0,
    )
    bucket1 = _owner_bucket(
        owner_rank=2,
        query_ids=(11, 12),
        raw_ids=(20,),
        compressed_ids=(101,),
        value=2.0,
    )

    reduced = accumulate_dsv4_gradient_owner_buckets(
        buckets=(bucket0, bucket1),
        query_token_ids=(10, 11, 12),
        raw_token_ids=(20,),
        compressed_entry_ids=(100, 101),
        d_attn_sink=torch.tensor([7.0, 8.0], dtype=torch.float64),
    )

    torch.testing.assert_close(reduced.dq, _stage_tensor(((1.0, 3.0, 2.0),)))
    torch.testing.assert_close(
        reduced.draw_kv,
        torch.full((1, 1, 3), 3.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        reduced.dcompressed_kv,
        torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        reduced.d_attn_sink, torch.tensor([7.0, 8.0], dtype=torch.float64)
    )


def test_gradient_owner_buckets_reject_bad_metadata() -> None:
    gradients = _gradient_result()
    with pytest.raises(ValueError, match="query_owner_ranks length"):
        pack_dsv4_gradient_owner_buckets(
            gradients=gradients,
            query_owner_ranks=(0,),
            raw_owner_ranks=(0, 1),
            compressed_owner_ranks=(0, 1),
        )
    with pytest.raises(ValueError, match="missing from query_token_ids"):
        accumulate_dsv4_gradient_owner_buckets(
            buckets=(
                _owner_bucket(
                    owner_rank=0,
                    query_ids=(13,),
                    raw_ids=(),
                    compressed_ids=(),
                    value=1.0,
                ),
            ),
            query_token_ids=(10,),
            raw_token_ids=(),
            compressed_entry_ids=(),
            d_attn_sink=torch.zeros(2, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="bucket_query_token_ids contains duplicate"):
        accumulate_dsv4_gradient_owner_buckets(
            buckets=(
                _owner_bucket(
                    owner_rank=0,
                    query_ids=(10, 10),
                    raw_ids=(),
                    compressed_ids=(),
                    value=1.0,
                ),
            ),
            query_token_ids=(10,),
            raw_token_ids=(),
            compressed_entry_ids=(),
            d_attn_sink=torch.zeros(2, dtype=torch.float64),
        )


def _materialized_stage(
    stage_index: int,
    query_token_ids: tuple[int, ...],
    raw_id: int | None = None,
    compressed_id: int | None = None,
) -> Dsv4MaterializedStage:
    if raw_id is None:
        raw_id = stage_index
    if compressed_id is None:
        compressed_id = stage_index + 100
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
        key_global_ids=(raw_id, compressed_id),
    )


def _single_rank_layout(
    kind: Dsv4CompressionKind = Dsv4CompressionKind.CSA,
) -> Dsv4CompressedLayout:
    return build_dsv4_compressed_layout(
        group_ids=torch.tensor([[0] * 8]),
        parent_ids=torch.tensor([[0] * 8]),
        token_layout_index=_LayoutIndex(
            ownership_ranges_by_rank=(((0, 8, 0),),),
            token_counts_by_rank=(8,),
        ),
        spec=Dsv4CompressionSpec(kind=kind, ratio=4),
    )


def _two_rank_layout() -> Dsv4CompressedLayout:
    return build_dsv4_compressed_layout(
        group_ids=torch.tensor([[0] * 8]),
        parent_ids=torch.tensor([[0] * 8]),
        token_layout_index=_LayoutIndex(
            ownership_ranges_by_rank=(((0, 4, 0),), ((4, 8, 0),)),
            token_counts_by_rank=(4, 4),
        ),
        spec=Dsv4CompressionSpec(kind=Dsv4CompressionKind.CSA, ratio=4),
    )


def _two_rank_slots() -> tuple[Any, ...]:
    return build_dsv4_stage_plan_slots(
        stage_plans_by_rank=(
            (
                _stage_plan(
                    stage_index=0,
                    q_ranges=((2, 4),),
                    k_ranges=((0, 8),),
                ),
            ),
            (
                _stage_plan(
                    stage_index=0,
                    q_ranges=((6, 8),),
                    k_ranges=((0, 8),),
                ),
            ),
        ),
    )


def _stage_plan(
    *,
    stage_index: int,
    q_ranges: tuple[tuple[int, int], ...],
    k_ranges: tuple[tuple[int, int], ...],
) -> _StagePlan:
    return _StagePlan(
        stage_index=stage_index,
        global_q_ranges=tuple(_Range(start=start, end=end) for start, end in q_ranges),
        global_k_ranges=tuple(_Range(start=start, end=end) for start, end in k_ranges),
    )


def _full_kv_materialized_stage(
    stage_index: int,
    query_token_ids: tuple[int, ...],
) -> Dsv4MaterializedStage:
    return Dsv4MaterializedStage(
        stage_index=stage_index,
        query_token_ids=query_token_ids,
        q_stage=torch.full(
            (1, len(query_token_ids), 2, 3),
            float(stage_index),
            dtype=torch.float64,
        ),
        kv_stage=torch.zeros(1, 10, 3, dtype=torch.float64),
        topk_stage_local=torch.zeros(
            1,
            len(query_token_ids),
            10,
            dtype=torch.long,
        ),
        raw_count=8,
        compressed_count=2,
        key_kinds=(Dsv4StageKeyKind.RAW,) * 8 + (Dsv4StageKeyKind.COMPRESSED,) * 2,
        key_global_ids=tuple(range(8)) + (0, 1),
    )


def _stage_tensor(values_by_batch: tuple[tuple[float, ...], ...]) -> torch.Tensor:
    rows = []
    for values in values_by_batch:
        rows.append(torch.tensor(values, dtype=torch.float64).view(len(values), 1, 1))
    return torch.stack(rows, dim=0).expand(-1, -1, 2, 3).contiguous()


def _kv_rows(values: tuple[float, ...]) -> torch.Tensor:
    return (
        torch.tensor(values, dtype=torch.float64)
        .view(1, len(values), 1)
        .expand(-1, -1, 3)
        .contiguous()
    )


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


def _fake_backward_for_bridge(
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
    del topk, global_out, grad_out, global_lse
    assert scale == 0.25
    stage_index = int(q[0, 0, 0, 0].item())
    dkv = torch.empty_like(kv)
    for row in range(int(kv.shape[1])):
        dkv[:, row].fill_(stage_index * 100.0 + row + 20.0)
    return Dsv4SparseBackwardResult(
        dq=torch.full_like(q, stage_index + 10.0),
        dkv=dkv,
        d_attn_sink=torch.full_like(attn_sink, 999.0),
    )


def _slot_backward_worker(rank: int, world_size: int, init_path: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29631")
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_fwd", _fake_forward_for_replay)
        setattr(
            cp_attention.sparse_kernel, "dsv4_sparse_bwd", _fake_backward_for_bridge
        )
        query_ids = (2, 3) if rank == 0 else (6, 7)
        layout = _two_rank_layout()
        attn_sink = torch.log(torch.tensor([5.0, 7.0], dtype=torch.float64))
        forward = run_materialized_dsv4_attention_forward(
            stages=(_full_kv_materialized_stage(0, query_ids),),
            query_token_ids=query_ids,
            attn_sink=attn_sink,
            scale=0.25,
        )
        grad_out = torch.ones_like(forward.out) * float(rank + 1)

        work = launch_dsv4_attention_backward_from_stage_plan_slots(
            layout=layout,
            rank=rank,
            stage_plan_slots=_two_rank_slots(),
            forward_result=forward,
            grad_out=grad_out,
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
        )
        result = work.wait_post_process()

        expected_sink = compute_single_sink_grad(
            grad_out=torch.ones_like(forward.out),
            global_out=forward.out,
            global_lse=forward.lse,
            attn_sink=attn_sink,
        ) + compute_single_sink_grad(
            grad_out=torch.ones_like(forward.out) * 2.0,
            global_out=forward.out,
            global_lse=forward.lse,
            attn_sink=attn_sink,
        )
        torch.testing.assert_close(result.d_attn_sink, expected_sink)
        if rank == 0:
            assert result.query_token_ids == (2, 3)
            assert result.raw_token_ids == (0, 1, 2, 3)
            assert result.compressed_entry_ids == (0,)
            torch.testing.assert_close(result.dq, _stage_tensor(((10.0, 10.0),)))
            torch.testing.assert_close(
                result.draw_kv,
                _kv_rows((40.0, 42.0, 44.0, 46.0)),
            )
            torch.testing.assert_close(result.dcompressed_kv, _kv_rows((56.0,)))
        else:
            assert result.query_token_ids == (6, 7)
            assert result.raw_token_ids == (4, 5, 6, 7)
            assert result.compressed_entry_ids == (1,)
            torch.testing.assert_close(result.dq, _stage_tensor(((10.0, 10.0),)))
            torch.testing.assert_close(
                result.draw_kv,
                _kv_rows((48.0, 50.0, 52.0, 54.0)),
            )
            torch.testing.assert_close(result.dcompressed_kv, _kv_rows((58.0,)))
    finally:
        destroy_process_group()


def _stage_backward_record(
    stage: Dsv4MaterializedStage,
    *,
    dq_value: float,
    raw_value: float,
    compressed_value: float,
) -> Dsv4StageBackwardRecord:
    return Dsv4StageBackwardRecord(
        materialized_stage=stage,
        dq_stage=torch.full_like(stage.q_stage, dq_value),
        dkv_stage=torch.tensor(
            [
                [
                    [raw_value, raw_value, raw_value],
                    [compressed_value, compressed_value, compressed_value],
                ]
            ],
            dtype=stage.kv_stage.dtype,
        ),
    )


def _gradient_result() -> Dsv4AttentionGradientResult:
    return Dsv4AttentionGradientResult(
        query_token_ids=(10, 11, 12),
        raw_token_ids=(20, 21),
        compressed_entry_ids=(100, 101),
        dq=_stage_tensor(((1.0, 2.0, 3.0),)),
        draw_kv=torch.tensor(
            [[[20.0, 20.0, 20.0], [21.0, 21.0, 21.0]]],
            dtype=torch.float64,
        ),
        dcompressed_kv=torch.tensor(
            [[[100.0, 100.0, 100.0], [101.0, 101.0, 101.0]]],
            dtype=torch.float64,
        ),
        d_attn_sink=torch.tensor([3.0, 4.0], dtype=torch.float64),
    )


def _owner_bucket(
    *,
    owner_rank: int,
    query_ids: tuple[int, ...],
    raw_ids: tuple[int, ...],
    compressed_ids: tuple[int, ...],
    value: float,
) -> Dsv4GradientOwnerBucket:
    return Dsv4GradientOwnerBucket(
        owner_rank=owner_rank,
        query_token_ids=query_ids,
        raw_token_ids=raw_ids,
        compressed_entry_ids=compressed_ids,
        dq=_stage_tensor((tuple(value for _ in query_ids),)),
        draw_kv=torch.full((1, len(raw_ids), 3), value, dtype=torch.float64),
        dcompressed_kv=torch.full(
            (1, len(compressed_ids), 3),
            value,
            dtype=torch.float64,
        ),
    )


class _FakeStageExchangeWork:
    def __init__(self, stage: Dsv4MaterializedStage) -> None:
        self.stage = stage
        self.wait_count = 0
        self.post_process_count = 0
        self._wait_complete = False

    def wait(self) -> None:
        if self._wait_complete:
            return
        self.wait_count += 1
        self._wait_complete = True

    def wait_post_process(self) -> Dsv4MaterializedStage:
        self.post_process_count += 1
        self.wait()
        return self.stage


class _BadStageExchangeWork:
    def wait(self) -> None:
        pass

    def wait_post_process(self) -> object:
        return object()


class _EmptyOwnerExchangeWork:
    def __init__(self) -> None:
        self.wait_count = 0
        self.post_process_count = 0
        self._wait_complete = False

    def wait(self) -> None:
        if self._wait_complete:
            return
        self.wait_count += 1
        self._wait_complete = True

    def wait_post_process(self) -> tuple[Dsv4GradientOwnerBucket, ...]:
        self.post_process_count += 1
        self.wait()
        return ()
