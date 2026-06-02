from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from oracles import dense_dsv4_packed_attention_oracle
from pydantic import BaseModel, ConfigDict
import pytest
import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.multiprocessing as mp

from art.megatron.dsv4 import (
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    Dsv4SparseBackwardResult,
    Dsv4SparseForwardResult,
    Dsv4StagePlanSlot,
    accumulate_materialized_dsv4_attention_backward,
    build_dsv4_compressed_layout,
    build_stage_local_topk_for_csa,
    compress_projected_kv,
    compute_indexer_topk,
    launch_dsv4_csa_projected_attention_forward_from_stage_plan_slots,
    launch_dsv4_hca_projected_attention_forward_from_stage_plan_slots,
    launch_dsv4_projected_attention_backward_from_stage_plan_slots,
    materialize_dsv4_stage_tensors,
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


def test_dense_oracle_matches_unpacked_branch_views_and_shared_prefix() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    torch.manual_seed(101)
    query = torch.randn(18, 3, 5, dtype=torch.float64)
    raw_kv = torch.randn(18, 5, dtype=torch.float64)
    compressed_kv = torch.randn(len(layout.entries), 5, dtype=torch.float64)
    attn_sink = torch.randn(3, dtype=torch.float64)
    topk = _all_visible_topk(layout)

    result = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=query,
        raw_kv=raw_kv,
        compressed_kv=compressed_kv,
        attn_sink=attn_sink,
        topk_by_query=topk,
        window_size=128,
        scale=0.7,
    )

    assert result.query_token_ids == tuple(range(18))
    assert result.out.shape == (1, 18, 3, 5)
    assert result.lse.shape == (1, 18, 3)
    completion_branch_ids = {
        int(branch.branch_stream_id)
        for branch in layout.branch_views
        if branch.suffix_stream_id is not None
    }
    prefix_results = [
        branch.out[:, :8]
        for branch in result.branches
        if int(branch.branch_stream_id) in completion_branch_ids
    ]
    assert len(prefix_results) == 2
    torch.testing.assert_close(prefix_results[0], prefix_results[1])
    torch.testing.assert_close(result.out[:, :8], prefix_results[0])
    assert torch.isfinite(result.out).all()
    assert torch.isfinite(result.lse).all()


def test_dense_oracle_rejects_sibling_compressed_leakage_from_bad_topk() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    torch.manual_seed(103)
    query = torch.randn(18, 2, 4, dtype=torch.float64)
    raw_kv = torch.randn(18, 4, dtype=torch.float64)
    compressed_kv = torch.randn(len(layout.entries), 4, dtype=torch.float64)
    attn_sink = torch.randn(2, dtype=torch.float64)
    topk = torch.full((18, 3), -1, dtype=torch.long)
    query_token_id = 12
    sibling_entry_id = next(
        entry.entry_id
        for entry in layout.entries
        if entry.branch_stream_id == 2 and not entry.shared_prefix_entry
    )
    visible_entry_id = next(
        entry.entry_id
        for entry in layout.entries
        if entry.branch_stream_id == 1 and not entry.shared_prefix_entry
    )
    topk[query_token_id] = torch.tensor([sibling_entry_id, visible_entry_id, -1])
    without_sibling = topk.clone()
    without_sibling[query_token_id] = torch.tensor([visible_entry_id, -1, -1])

    with_bad_topk = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=query,
        raw_kv=raw_kv,
        compressed_kv=compressed_kv,
        attn_sink=attn_sink,
        topk_by_query=topk,
        window_size=128,
        scale=1.0,
    )
    expected = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=query,
        raw_kv=raw_kv,
        compressed_kv=compressed_kv,
        attn_sink=attn_sink,
        topk_by_query=without_sibling,
        window_size=128,
        scale=1.0,
    )
    torch.testing.assert_close(
        with_bad_topk.out[:, query_token_id],
        expected.out[:, query_token_id],
    )
    torch.testing.assert_close(
        with_bad_topk.lse[:, query_token_id],
        expected.lse[:, query_token_id],
    )


def test_materialized_stage_path_matches_packed_oracle_forward_and_backward(
    monkeypatch,
) -> None:
    layout = _layout(Dsv4CompressionKind.CSA, rank_count=1)
    torch.manual_seed(107)
    query = torch.randn(18, 2, 4, dtype=torch.float64)
    raw_kv = torch.randn(18, 4, dtype=torch.float64)
    compressed_kv = torch.randn(len(layout.entries), 4, dtype=torch.float64)
    attn_sink = torch.randn(2, dtype=torch.float64)
    grad_out = torch.randn(1, 18, 2, 4, dtype=torch.float64)
    global_topk = _all_visible_topk(layout)

    monkeypatch.setattr(cp_attention.sparse_kernel, "dsv4_sparse_fwd", _dense_fake_fwd)
    monkeypatch.setattr(cp_attention.sparse_kernel, "dsv4_sparse_bwd", _dense_fake_bwd)

    stage_inputs = build_stage_local_topk_for_csa(
        layout=layout,
        stage_index=0,
        query_token_ids=tuple(range(18)),
        global_k_ranges=(_Range(start=0, end=18),),
        global_topk=global_topk,
        window_size=128,
        raw_list_size=18,
        compressed_list_size=int(global_topk.shape[1]),
    )
    stage = materialize_dsv4_stage_tensors(
        stage_inputs=stage_inputs,
        query=query,
        query_token_ids=tuple(range(18)),
        raw_kv=raw_kv,
        raw_token_ids=tuple(range(18)),
        compressed_kv=compressed_kv,
        compressed_entry_ids=tuple(range(len(layout.entries))),
    )
    actual = run_materialized_dsv4_attention_forward(
        stages=(stage,),
        query_token_ids=tuple(range(18)),
        attn_sink=attn_sink,
        scale=0.5,
    )
    expected = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=query,
        raw_kv=raw_kv,
        compressed_kv=compressed_kv,
        attn_sink=attn_sink,
        topk_by_query=global_topk,
        window_size=128,
        scale=0.5,
    )
    torch.testing.assert_close(actual.out, expected.out, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        actual.lse,
        expected.lse,
        rtol=1e-6,
        atol=1e-6,
        check_dtype=False,
    )

    replay = replay_materialized_dsv4_attention_backward(
        forward_result=actual,
        grad_out=grad_out,
    )
    actual_grad = accumulate_materialized_dsv4_attention_backward(
        replay_result=replay,
        query_token_ids=tuple(range(18)),
        raw_token_ids=tuple(range(18)),
        compressed_entry_ids=tuple(range(len(layout.entries))),
    )

    ref_query = query.detach().clone().requires_grad_()
    ref_raw = raw_kv.detach().clone().requires_grad_()
    ref_compressed = compressed_kv.detach().clone().requires_grad_()
    ref_sink = attn_sink.detach().clone().requires_grad_()
    ref = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=ref_query,
        raw_kv=ref_raw,
        compressed_kv=ref_compressed,
        attn_sink=ref_sink,
        topk_by_query=global_topk,
        window_size=128,
        scale=0.5,
    )
    (ref.out * grad_out).sum().backward()

    assert ref_query.grad is not None
    assert ref_raw.grad is not None
    assert ref_compressed.grad is not None
    assert ref_sink.grad is not None
    torch.testing.assert_close(
        actual_grad.dq,
        ref_query.grad.unsqueeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        actual_grad.draw_kv,
        ref_raw.grad.unsqueeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        actual_grad.dcompressed_kv,
        ref_compressed.grad.unsqueeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        actual_grad.d_attn_sink,
        ref_sink.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    assert not bool(actual_grad.dq.abs().sum().eq(0).item())
    assert not bool(actual_grad.draw_kv.abs().sum().eq(0).item())
    assert not bool(actual_grad.dcompressed_kv.abs().sum().eq(0).item())


def test_projected_hca_wrapper_matches_packed_oracle_forward_and_backward(
    monkeypatch,
) -> None:
    layout = _layout(Dsv4CompressionKind.HCA, rank_count=1)
    torch.manual_seed(109)
    query = torch.randn(18, 2, 4, dtype=torch.float64)
    raw_kv = torch.randn(18, 4, dtype=torch.float64)
    projected_kv = torch.randn(18, 4, dtype=torch.float64)
    projected_gate = torch.randn(18, 4, dtype=torch.float64)
    positional_bias = torch.randn(4, 4, dtype=torch.float64)
    attn_sink = torch.randn(2, dtype=torch.float64)
    grad_out = torch.randn(1, 18, 2, 4, dtype=torch.float64)

    monkeypatch.setattr(cp_attention.sparse_kernel, "dsv4_sparse_fwd", _dense_fake_fwd)
    monkeypatch.setattr(cp_attention.sparse_kernel, "dsv4_sparse_bwd", _dense_fake_bwd)

    forward_work = launch_dsv4_hca_projected_attention_forward_from_stage_plan_slots(
        layout=layout,
        rank=0,
        stage_plan_slots=_single_rank_slot(),
        query=query,
        query_token_ids=tuple(range(18)),
        raw_kv=raw_kv,
        raw_token_ids=tuple(range(18)),
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        positional_bias=positional_bias,
        token_ids=tuple(range(18)),
        attn_sink=attn_sink,
        group=None,
        async_op=False,
        scale=0.25,
        window_size=128,
        raw_list_size=18,
    )
    actual = forward_work.wait_post_process()
    compressed = compress_projected_kv(
        layout=layout,
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        positional_bias=positional_bias,
    )
    expected = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=query,
        raw_kv=raw_kv,
        compressed_kv=compressed,
        attn_sink=attn_sink,
        topk_by_query=None,
        window_size=128,
        scale=0.25,
    )
    torch.testing.assert_close(actual.attention.out, expected.out, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        actual.attention.lse,
        expected.lse,
        rtol=1e-6,
        atol=1e-6,
        check_dtype=False,
    )

    backward = launch_dsv4_projected_attention_backward_from_stage_plan_slots(
        layout=layout,
        rank=0,
        stage_plan_slots=_single_rank_slot(),
        forward_result=actual,
        grad_out=grad_out,
        group=None,
        async_op=False,
    ).wait_post_process()

    ref_query = query.detach().clone().requires_grad_()
    ref_raw = raw_kv.detach().clone().requires_grad_()
    ref_projected = projected_kv.detach().clone().requires_grad_()
    ref_gate = projected_gate.detach().clone().requires_grad_()
    ref_bias = positional_bias.detach().clone().requires_grad_()
    ref_sink = attn_sink.detach().clone().requires_grad_()
    ref_compressed = compress_projected_kv(
        layout=layout,
        projected_kv=ref_projected,
        projected_gate=ref_gate,
        positional_bias=ref_bias,
    )
    ref = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=ref_query,
        raw_kv=ref_raw,
        compressed_kv=ref_compressed,
        attn_sink=ref_sink,
        topk_by_query=None,
        window_size=128,
        scale=0.25,
    )
    (ref.out * grad_out).sum().backward()

    assert ref_query.grad is not None
    assert ref_raw.grad is not None
    assert ref_projected.grad is not None
    assert ref_gate.grad is not None
    assert ref_bias.grad is not None
    assert ref_sink.grad is not None
    torch.testing.assert_close(
        backward.attention.dq,
        ref_query.grad.unsqueeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        backward.attention.draw_kv,
        ref_raw.grad.unsqueeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        backward.main_compressor.dprojected_kv,
        ref_projected.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        backward.main_compressor.dprojected_gate,
        ref_gate.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        backward.main_compressor.dpositional_bias,
        ref_bias.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        backward.attention.d_attn_sink,
        ref_sink.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    assert not bool(backward.main_compressor.dprojected_kv.abs().sum().eq(0).item())
    assert not bool(backward.main_compressor.dprojected_gate.abs().sum().eq(0).item())


def test_projected_csa_wrapper_matches_packed_oracle_forward_and_backward(
    monkeypatch,
) -> None:
    layout = _layout(Dsv4CompressionKind.CSA, rank_count=1)
    torch.manual_seed(113)
    query = torch.randn(18, 2, 4, dtype=torch.float64)
    raw_kv = torch.randn(18, 4, dtype=torch.float64)
    main_projected_kv = torch.randn(18, 8, dtype=torch.float64)
    main_projected_gate = torch.randn(18, 8, dtype=torch.float64)
    main_positional_bias = torch.randn(4, 8, dtype=torch.float64)
    indexer_projected_kv = torch.randn(18, 6, dtype=torch.float64)
    indexer_projected_gate = torch.randn(18, 6, dtype=torch.float64)
    indexer_positional_bias = torch.randn(4, 6, dtype=torch.float64)
    indexer_q = torch.randn(18, 2, 3, dtype=torch.float64)
    indexer_weights = torch.randn(18, 2, dtype=torch.float64)
    attn_sink = torch.randn(2, dtype=torch.float64)
    grad_out = torch.randn(1, 18, 2, 4, dtype=torch.float64)

    monkeypatch.setattr(cp_attention.sparse_kernel, "dsv4_sparse_fwd", _dense_fake_fwd)
    monkeypatch.setattr(cp_attention.sparse_kernel, "dsv4_sparse_bwd", _dense_fake_bwd)

    forward_work = launch_dsv4_csa_projected_attention_forward_from_stage_plan_slots(
        layout=layout,
        rank=0,
        stage_plan_slots=_single_rank_slot(),
        query=query,
        query_token_ids=tuple(range(18)),
        raw_kv=raw_kv,
        raw_token_ids=tuple(range(18)),
        main_projected_kv=main_projected_kv,
        main_projected_gate=main_projected_gate,
        main_positional_bias=main_positional_bias,
        main_token_ids=tuple(range(18)),
        indexer_projected_kv=indexer_projected_kv,
        indexer_projected_gate=indexer_projected_gate,
        indexer_positional_bias=indexer_positional_bias,
        indexer_token_ids=tuple(range(18)),
        indexer_q=indexer_q,
        indexer_weights=indexer_weights,
        indexer_topk=2,
        attn_sink=attn_sink,
        group=None,
        async_op=False,
        scale=0.4,
        window_size=128,
        raw_list_size=18,
        compressed_list_size=2,
    )
    actual = forward_work.wait_post_process()
    main_compressed = compress_projected_kv(
        layout=layout,
        projected_kv=main_projected_kv,
        projected_gate=main_projected_gate,
        positional_bias=main_positional_bias,
    )
    indexer_compressed = compress_projected_kv(
        layout=layout,
        projected_kv=indexer_projected_kv,
        projected_gate=indexer_projected_gate,
        positional_bias=indexer_positional_bias,
    )
    topk = compute_indexer_topk(
        layout=layout,
        query_token_ids=tuple(range(18)),
        indexer_q=indexer_q,
        indexer_kv=indexer_compressed,
        indexer_weights=indexer_weights,
        candidate_entry_ids=tuple(range(len(layout.entries))),
        topk=2,
    ).indices[0]
    expected = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=query,
        raw_kv=raw_kv,
        compressed_kv=main_compressed,
        attn_sink=attn_sink,
        topk_by_query=topk,
        window_size=128,
        scale=0.4,
    )
    torch.testing.assert_close(actual.attention.out, expected.out, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        actual.attention.lse,
        expected.lse,
        rtol=1e-6,
        atol=1e-6,
        check_dtype=False,
    )

    backward = launch_dsv4_projected_attention_backward_from_stage_plan_slots(
        layout=layout,
        rank=0,
        stage_plan_slots=_single_rank_slot(),
        forward_result=actual,
        grad_out=grad_out,
        group=None,
        async_op=False,
    ).wait_post_process()

    ref_query = query.detach().clone().requires_grad_()
    ref_raw = raw_kv.detach().clone().requires_grad_()
    ref_main_projected = main_projected_kv.detach().clone().requires_grad_()
    ref_main_gate = main_projected_gate.detach().clone().requires_grad_()
    ref_main_bias = main_positional_bias.detach().clone().requires_grad_()
    ref_sink = attn_sink.detach().clone().requires_grad_()
    ref_main_compressed = compress_projected_kv(
        layout=layout,
        projected_kv=ref_main_projected,
        projected_gate=ref_main_gate,
        positional_bias=ref_main_bias,
    )
    ref = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=ref_query,
        raw_kv=ref_raw,
        compressed_kv=ref_main_compressed,
        attn_sink=ref_sink,
        topk_by_query=topk,
        window_size=128,
        scale=0.4,
    )
    (ref.out * grad_out).sum().backward()

    assert ref_query.grad is not None
    assert ref_raw.grad is not None
    assert ref_main_projected.grad is not None
    assert ref_main_gate.grad is not None
    assert ref_main_bias.grad is not None
    assert ref_sink.grad is not None
    torch.testing.assert_close(
        backward.attention.dq,
        ref_query.grad.unsqueeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        backward.attention.draw_kv,
        ref_raw.grad.unsqueeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        backward.main_compressor.dprojected_kv,
        ref_main_projected.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        backward.main_compressor.dprojected_gate,
        ref_main_gate.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        backward.main_compressor.dpositional_bias,
        ref_main_bias.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        backward.attention.d_attn_sink,
        ref_sink.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    assert actual.indexer_compressed is not None
    assert not bool(backward.main_compressor.dprojected_kv.abs().sum().eq(0).item())
    assert not bool(backward.main_compressor.dprojected_gate.abs().sum().eq(0).item())


def test_distributed_projected_csa_wrapper_matches_packed_oracle(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_projected_csa_oracle_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_csa_oracle_worker,
        args=(2, str(init_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not cast(Any, torch.distributed).is_nccl_available(),
    reason="DSV4 CUDA/NCCL oracle requires at least two CUDA devices and NCCL",
)
def test_distributed_projected_csa_wrapper_matches_packed_oracle_cuda_nccl(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_projected_csa_oracle_nccl"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_csa_nccl_oracle_worker,
        args=(2, str(init_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def test_distributed_projected_hca_wrapper_matches_packed_oracle(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_projected_hca_oracle_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_hca_oracle_worker,
        args=(2, str(init_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def test_distributed_projected_hca_ratio128_boundary_matches_packed_oracle(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_projected_hca_ratio128_oracle_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_hca_ratio128_oracle_worker,
        args=(2, str(init_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def _all_visible_topk(layout: Dsv4CompressedLayout) -> torch.Tensor:
    max_visible = max(
        sum(
            1
            for entry in layout.entries
            if (
                int(entry.branch_stream_id) == int(branch.branch_stream_id)
                or (
                    entry.shared_prefix_entry
                    and int(entry.prefix_stream_id) == int(branch.prefix_stream_id)
                )
            )
            and int(entry.closure_view_pos) <= int(token.view_pos)
        )
        for branch in layout.branch_views
        for token in branch.tokens
    )
    topk = torch.full((18, max_visible), -1, dtype=torch.long)
    for branch in layout.branch_views:
        for token in branch.tokens:
            visible = [
                int(entry.entry_id)
                for entry in layout.entries
                if (
                    int(entry.branch_stream_id) == int(branch.branch_stream_id)
                    or (
                        entry.shared_prefix_entry
                        and int(entry.prefix_stream_id) == int(branch.prefix_stream_id)
                    )
                )
                and int(entry.closure_view_pos) <= int(token.view_pos)
            ]
            topk[int(token.packed_token_id), : len(visible)] = torch.tensor(visible)
    return topk


def _layout(kind: Dsv4CompressionKind, rank_count: int = 2) -> Dsv4CompressedLayout:
    if rank_count == 1:
        ownership_ranges_by_rank = (((0, 18, 0),),)
        token_counts_by_rank = (18,)
    elif rank_count == 2:
        ownership_ranges_by_rank = (
            ((0, 8, 0),),
            ((8, 18, 0),),
        )
        token_counts_by_rank = (8, 10)
    else:
        raise RuntimeError(f"unsupported test rank_count {rank_count}")
    return build_dsv4_compressed_layout(
        group_ids=torch.tensor([[0] * 8 + [1] * 5 + [2] * 5 + [-1] * 2]),
        parent_ids=torch.tensor([[0] * 8 + [0] * 5 + [0] * 5 + [-1] * 2]),
        token_layout_index=_LayoutIndex(
            ownership_ranges_by_rank=ownership_ranges_by_rank,
            token_counts_by_rank=token_counts_by_rank,
        ),
        spec=Dsv4CompressionSpec(kind=kind, ratio=4),
    )


def _single_rank_slot() -> tuple[Dsv4StagePlanSlot, ...]:
    return (
        Dsv4StagePlanSlot(
            stage_index=0,
            stage_plans_by_rank=(
                _StagePlan(
                    stage_index=0,
                    global_q_ranges=(_Range(start=0, end=18),),
                    global_k_ranges=(_Range(start=0, end=18),),
                ),
            ),
        ),
    )


def _two_rank_slot() -> tuple[Dsv4StagePlanSlot, ...]:
    return _two_rank_full_stage_slot(first_rank_end=8, token_count=18)


def _two_rank_full_stage_slot(
    *,
    first_rank_end: int,
    token_count: int,
) -> tuple[Dsv4StagePlanSlot, ...]:
    return (
        Dsv4StagePlanSlot(
            stage_index=0,
            stage_plans_by_rank=(
                _StagePlan(
                    stage_index=0,
                    global_q_ranges=(_Range(start=0, end=first_rank_end),),
                    global_k_ranges=(_Range(start=0, end=token_count),),
                ),
                _StagePlan(
                    stage_index=0,
                    global_q_ranges=(_Range(start=first_rank_end, end=token_count),),
                    global_k_ranges=(_Range(start=0, end=token_count),),
                ),
            ),
        ),
    )


def _distributed_projected_csa_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29628")
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_fwd", _dense_fake_fwd)
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_bwd", _dense_fake_bwd)
        layout = _layout(Dsv4CompressionKind.CSA, rank_count=2)
        torch.manual_seed(127)
        query = torch.randn(18, 2, 4, dtype=torch.float64)
        raw_kv = torch.randn(18, 4, dtype=torch.float64)
        main_projected_kv = torch.randn(18, 8, dtype=torch.float64)
        main_projected_gate = torch.randn(18, 8, dtype=torch.float64)
        main_positional_bias = torch.randn(4, 8, dtype=torch.float64)
        indexer_projected_kv = torch.randn(18, 6, dtype=torch.float64)
        indexer_projected_gate = torch.randn(18, 6, dtype=torch.float64)
        indexer_positional_bias = torch.randn(4, 6, dtype=torch.float64)
        indexer_q = torch.randn(18, 2, 3, dtype=torch.float64)
        indexer_weights = torch.randn(18, 2, dtype=torch.float64)
        attn_sink = torch.randn(2, dtype=torch.float64)
        grad_out = torch.randn(1, 18, 2, 4, dtype=torch.float64)
        local_token_ids = tuple(range(0, 8)) if rank == 0 else tuple(range(8, 18))

        forward = launch_dsv4_csa_projected_attention_forward_from_stage_plan_slots(
            layout=layout,
            rank=rank,
            stage_plan_slots=_two_rank_slot(),
            query=query[list(local_token_ids)],
            query_token_ids=local_token_ids,
            raw_kv=raw_kv[list(local_token_ids)],
            raw_token_ids=local_token_ids,
            main_projected_kv=main_projected_kv[list(local_token_ids)],
            main_projected_gate=main_projected_gate[list(local_token_ids)],
            main_positional_bias=main_positional_bias,
            main_token_ids=local_token_ids,
            indexer_projected_kv=indexer_projected_kv[list(local_token_ids)],
            indexer_projected_gate=indexer_projected_gate[list(local_token_ids)],
            indexer_positional_bias=indexer_positional_bias,
            indexer_token_ids=local_token_ids,
            indexer_q=indexer_q[list(local_token_ids)],
            indexer_weights=indexer_weights[list(local_token_ids)],
            indexer_topk=2,
            attn_sink=attn_sink,
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
            scale=0.4,
            window_size=128,
            raw_list_size=18,
            compressed_list_size=2,
        ).wait_post_process()

        main_compressed = compress_projected_kv(
            layout=layout,
            projected_kv=main_projected_kv,
            projected_gate=main_projected_gate,
            positional_bias=main_positional_bias,
        )
        indexer_compressed = compress_projected_kv(
            layout=layout,
            projected_kv=indexer_projected_kv,
            projected_gate=indexer_projected_gate,
            positional_bias=indexer_positional_bias,
        )
        topk = compute_indexer_topk(
            layout=layout,
            query_token_ids=tuple(range(18)),
            indexer_q=indexer_q,
            indexer_kv=indexer_compressed,
            indexer_weights=indexer_weights,
            candidate_entry_ids=tuple(range(len(layout.entries))),
            topk=2,
        ).indices[0]
        expected = dense_dsv4_packed_attention_oracle(
            layout=layout,
            query=query,
            raw_kv=raw_kv,
            compressed_kv=main_compressed,
            attn_sink=attn_sink,
            topk_by_query=topk,
            window_size=128,
            scale=0.4,
        )
        local_positions = torch.tensor(local_token_ids, dtype=torch.long)
        torch.testing.assert_close(
            forward.attention.out,
            expected.out.index_select(1, local_positions),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            forward.attention.lse,
            expected.lse.index_select(1, local_positions),
            rtol=1e-6,
            atol=1e-6,
            check_dtype=False,
        )

        backward = launch_dsv4_projected_attention_backward_from_stage_plan_slots(
            layout=layout,
            rank=rank,
            stage_plan_slots=_two_rank_slot(),
            forward_result=forward,
            grad_out=grad_out.index_select(1, local_positions),
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
        ).wait_post_process()

        ref_query = query.detach().clone().requires_grad_()
        ref_raw = raw_kv.detach().clone().requires_grad_()
        ref_main_projected = main_projected_kv.detach().clone().requires_grad_()
        ref_main_gate = main_projected_gate.detach().clone().requires_grad_()
        ref_main_bias = main_positional_bias.detach().clone().requires_grad_()
        ref_sink = attn_sink.detach().clone().requires_grad_()
        ref_main_compressed = compress_projected_kv(
            layout=layout,
            projected_kv=ref_main_projected,
            projected_gate=ref_main_gate,
            positional_bias=ref_main_bias,
        )
        ref = dense_dsv4_packed_attention_oracle(
            layout=layout,
            query=ref_query,
            raw_kv=ref_raw,
            compressed_kv=ref_main_compressed,
            attn_sink=ref_sink,
            topk_by_query=topk,
            window_size=128,
            scale=0.4,
        )
        (ref.out * grad_out).sum().backward()

        assert ref_query.grad is not None
        assert ref_raw.grad is not None
        assert ref_main_projected.grad is not None
        assert ref_main_gate.grad is not None
        assert ref_main_bias.grad is not None
        assert ref_sink.grad is not None
        _assert_id_aligned_rows_close(
            actual=backward.attention.dq,
            actual_ids=backward.attention.query_token_ids,
            expected=ref_query.grad.unsqueeze(0),
        )
        _assert_id_aligned_rows_close(
            actual=backward.attention.draw_kv,
            actual_ids=backward.attention.raw_token_ids,
            expected=ref_raw.grad.unsqueeze(0),
        )
        _assert_id_aligned_rows_close(
            actual=backward.main_compressor.dprojected_kv,
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_main_projected.grad,
        )
        _assert_id_aligned_rows_close(
            actual=backward.main_compressor.dprojected_gate,
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_main_gate.grad,
        )
        torch.testing.assert_close(
            backward.main_compressor.dpositional_bias,
            ref_main_bias.grad,
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            backward.attention.d_attn_sink,
            ref_sink.grad,
            rtol=1e-6,
            atol=1e-6,
        )
        assert not bool(backward.attention.dq.abs().sum().eq(0).item())
        assert not bool(backward.main_compressor.dprojected_kv.abs().sum().eq(0).item())
    finally:
        destroy_process_group()


def _distributed_projected_csa_nccl_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29631")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_process_group(
        "nccl",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_fwd", _dense_fake_fwd)
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_bwd", _dense_fake_bwd)
        layout = _layout(Dsv4CompressionKind.CSA, rank_count=2)
        torch.manual_seed(127)
        query = torch.randn(18, 2, 4, device=device)
        raw_kv = torch.randn(18, 4, device=device)
        main_projected_kv = torch.randn(18, 8, device=device)
        main_projected_gate = torch.randn(18, 8, device=device)
        main_positional_bias = torch.randn(4, 8, device=device)
        indexer_projected_kv = torch.randn(18, 6, device=device)
        indexer_projected_gate = torch.randn(18, 6, device=device)
        indexer_positional_bias = torch.randn(4, 6, device=device)
        indexer_q = torch.randn(18, 2, 3, device=device)
        indexer_weights = torch.randn(18, 2, device=device)
        attn_sink = torch.randn(2, device=device)
        grad_out = torch.randn(1, 18, 2, 4, device=device)
        local_token_ids = tuple(range(0, 8)) if rank == 0 else tuple(range(8, 18))

        forward = launch_dsv4_csa_projected_attention_forward_from_stage_plan_slots(
            layout=layout,
            rank=rank,
            stage_plan_slots=_two_rank_slot(),
            query=query[list(local_token_ids)],
            query_token_ids=local_token_ids,
            raw_kv=raw_kv[list(local_token_ids)],
            raw_token_ids=local_token_ids,
            main_projected_kv=main_projected_kv[list(local_token_ids)],
            main_projected_gate=main_projected_gate[list(local_token_ids)],
            main_positional_bias=main_positional_bias,
            main_token_ids=local_token_ids,
            indexer_projected_kv=indexer_projected_kv[list(local_token_ids)],
            indexer_projected_gate=indexer_projected_gate[list(local_token_ids)],
            indexer_positional_bias=indexer_positional_bias,
            indexer_token_ids=local_token_ids,
            indexer_q=indexer_q[list(local_token_ids)],
            indexer_weights=indexer_weights[list(local_token_ids)],
            indexer_topk=2,
            attn_sink=attn_sink,
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
            scale=0.4,
            window_size=128,
            raw_list_size=18,
            compressed_list_size=2,
        ).wait_post_process()
        assert forward.attention.out.device == device

        main_compressed = compress_projected_kv(
            layout=layout,
            projected_kv=main_projected_kv,
            projected_gate=main_projected_gate,
            positional_bias=main_positional_bias,
        )
        indexer_compressed = compress_projected_kv(
            layout=layout,
            projected_kv=indexer_projected_kv,
            projected_gate=indexer_projected_gate,
            positional_bias=indexer_positional_bias,
        )
        topk = compute_indexer_topk(
            layout=layout,
            query_token_ids=tuple(range(18)),
            indexer_q=indexer_q,
            indexer_kv=indexer_compressed,
            indexer_weights=indexer_weights,
            candidate_entry_ids=tuple(range(len(layout.entries))),
            topk=2,
        ).indices[0]
        expected = dense_dsv4_packed_attention_oracle(
            layout=layout,
            query=query,
            raw_kv=raw_kv,
            compressed_kv=main_compressed,
            attn_sink=attn_sink,
            topk_by_query=topk,
            window_size=128,
            scale=0.4,
        )
        local_positions = torch.tensor(
            local_token_ids,
            device=device,
            dtype=torch.long,
        )
        torch.testing.assert_close(
            forward.attention.out,
            expected.out.index_select(1, local_positions),
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            forward.attention.lse,
            expected.lse.index_select(1, local_positions),
            rtol=1e-4,
            atol=1e-4,
            check_dtype=False,
        )

        backward = launch_dsv4_projected_attention_backward_from_stage_plan_slots(
            layout=layout,
            rank=rank,
            stage_plan_slots=_two_rank_slot(),
            forward_result=forward,
            grad_out=grad_out.index_select(1, local_positions),
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
        ).wait_post_process()

        ref_query = query.detach().clone().requires_grad_()
        ref_raw = raw_kv.detach().clone().requires_grad_()
        ref_main_projected = main_projected_kv.detach().clone().requires_grad_()
        ref_main_gate = main_projected_gate.detach().clone().requires_grad_()
        ref_main_bias = main_positional_bias.detach().clone().requires_grad_()
        ref_sink = attn_sink.detach().clone().requires_grad_()
        ref_main_compressed = compress_projected_kv(
            layout=layout,
            projected_kv=ref_main_projected,
            projected_gate=ref_main_gate,
            positional_bias=ref_main_bias,
        )
        ref = dense_dsv4_packed_attention_oracle(
            layout=layout,
            query=ref_query,
            raw_kv=ref_raw,
            compressed_kv=ref_main_compressed,
            attn_sink=ref_sink,
            topk_by_query=topk,
            window_size=128,
            scale=0.4,
        )
        (ref.out * grad_out).sum().backward()

        assert ref_query.grad is not None
        assert ref_raw.grad is not None
        assert ref_main_projected.grad is not None
        assert ref_main_gate.grad is not None
        assert ref_main_bias.grad is not None
        assert ref_sink.grad is not None
        _assert_id_aligned_rows_close(
            actual=backward.attention.dq,
            actual_ids=backward.attention.query_token_ids,
            expected=ref_query.grad.unsqueeze(0),
            rtol=1e-4,
            atol=1e-4,
        )
        _assert_id_aligned_rows_close(
            actual=backward.attention.draw_kv,
            actual_ids=backward.attention.raw_token_ids,
            expected=ref_raw.grad.unsqueeze(0),
            rtol=1e-4,
            atol=1e-4,
        )
        _assert_id_aligned_rows_close(
            actual=backward.main_compressor.dprojected_kv,
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_main_projected.grad,
            rtol=1e-4,
            atol=1e-4,
        )
        _assert_id_aligned_rows_close(
            actual=backward.main_compressor.dprojected_gate,
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_main_gate.grad,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            backward.main_compressor.dpositional_bias,
            ref_main_bias.grad,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            backward.attention.d_attn_sink,
            ref_sink.grad,
            rtol=1e-4,
            atol=1e-4,
        )
        assert not bool(backward.attention.dq.abs().sum().eq(0).item())
        assert not bool(backward.main_compressor.dprojected_kv.abs().sum().eq(0).item())
        assert not bool(
            backward.main_compressor.dprojected_gate.abs().sum().eq(0).item()
        )
        torch.cuda.synchronize(device)
    finally:
        destroy_process_group()


def _distributed_projected_hca_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29629")
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_fwd", _dense_fake_fwd)
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_bwd", _dense_fake_bwd)
        _run_distributed_projected_hca_oracle_case(
            rank=rank,
            layout=_layout(Dsv4CompressionKind.HCA, rank_count=2),
            stage_plan_slots=_two_rank_slot(),
            local_token_ids=tuple(range(0, 8)) if rank == 0 else tuple(range(8, 18)),
            token_count=18,
            seed=131,
            expect_rank0_halo_grad=False,
        )
    finally:
        destroy_process_group()


def _distributed_projected_hca_ratio128_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29630")
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_fwd", _dense_fake_fwd)
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_bwd", _dense_fake_bwd)
        layout = _hca_ratio128_boundary_layout()
        _run_distributed_projected_hca_oracle_case(
            rank=rank,
            layout=layout,
            stage_plan_slots=_two_rank_full_stage_slot(
                first_rank_end=96,
                token_count=416,
            ),
            local_token_ids=tuple(range(0, 96)) if rank == 0 else tuple(range(96, 416)),
            token_count=416,
            seed=137,
            expect_rank0_halo_grad=True,
        )
    finally:
        destroy_process_group()


def _hca_ratio128_boundary_layout() -> Dsv4CompressedLayout:
    return build_dsv4_compressed_layout(
        group_ids=torch.tensor([[0] * 96 + [1] * 160 + [2] * 160]),
        parent_ids=torch.tensor([[0] * 96 + [0] * 160 + [0] * 160]),
        token_layout_index=_LayoutIndex(
            ownership_ranges_by_rank=(
                ((0, 96, 0),),
                ((96, 416, 0),),
            ),
            token_counts_by_rank=(96, 320),
        ),
        spec=Dsv4CompressionSpec(kind=Dsv4CompressionKind.HCA, ratio=128),
    )


def _run_distributed_projected_hca_oracle_case(
    *,
    rank: int,
    layout: Dsv4CompressedLayout,
    stage_plan_slots: tuple[Dsv4StagePlanSlot, ...],
    local_token_ids: tuple[int, ...],
    token_count: int,
    seed: int,
    expect_rank0_halo_grad: bool,
) -> None:
    if expect_rank0_halo_grad:
        assert layout.halo_transfers
    torch.manual_seed(seed)
    query = torch.randn(token_count, 2, 4, dtype=torch.float64)
    raw_kv = torch.randn(token_count, 4, dtype=torch.float64)
    projected_kv = torch.randn(token_count, 4, dtype=torch.float64)
    projected_gate = torch.randn(token_count, 4, dtype=torch.float64)
    positional_bias = torch.randn(int(layout.spec.ratio), 4, dtype=torch.float64)
    attn_sink = torch.randn(2, dtype=torch.float64)
    grad_out = torch.randn(1, token_count, 2, 4, dtype=torch.float64)

    forward = launch_dsv4_hca_projected_attention_forward_from_stage_plan_slots(
        layout=layout,
        rank=rank,
        stage_plan_slots=stage_plan_slots,
        query=query[list(local_token_ids)],
        query_token_ids=local_token_ids,
        raw_kv=raw_kv[list(local_token_ids)],
        raw_token_ids=local_token_ids,
        projected_kv=projected_kv[list(local_token_ids)],
        projected_gate=projected_gate[list(local_token_ids)],
        positional_bias=positional_bias,
        token_ids=local_token_ids,
        attn_sink=attn_sink,
        group=cast(Any, torch.distributed).group.WORLD,
        async_op=True,
        scale=0.25,
        window_size=128,
        raw_list_size=min(128, token_count),
        compressed_list_size=len(layout.entries),
    ).wait_post_process()

    compressed = compress_projected_kv(
        layout=layout,
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        positional_bias=positional_bias,
    )
    expected = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=query,
        raw_kv=raw_kv,
        compressed_kv=compressed,
        attn_sink=attn_sink,
        topk_by_query=None,
        window_size=128,
        scale=0.25,
    )
    local_positions = torch.tensor(local_token_ids, dtype=torch.long)
    torch.testing.assert_close(
        forward.attention.out,
        expected.out.index_select(1, local_positions),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        forward.attention.lse,
        expected.lse.index_select(1, local_positions),
        rtol=1e-6,
        atol=1e-6,
        check_dtype=False,
    )

    backward = launch_dsv4_projected_attention_backward_from_stage_plan_slots(
        layout=layout,
        rank=rank,
        stage_plan_slots=stage_plan_slots,
        forward_result=forward,
        grad_out=grad_out.index_select(1, local_positions),
        group=cast(Any, torch.distributed).group.WORLD,
        async_op=True,
    ).wait_post_process()

    ref_query = query.detach().clone().requires_grad_()
    ref_raw = raw_kv.detach().clone().requires_grad_()
    ref_projected = projected_kv.detach().clone().requires_grad_()
    ref_gate = projected_gate.detach().clone().requires_grad_()
    ref_bias = positional_bias.detach().clone().requires_grad_()
    ref_sink = attn_sink.detach().clone().requires_grad_()
    ref_compressed = compress_projected_kv(
        layout=layout,
        projected_kv=ref_projected,
        projected_gate=ref_gate,
        positional_bias=ref_bias,
    )
    ref = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=ref_query,
        raw_kv=ref_raw,
        compressed_kv=ref_compressed,
        attn_sink=ref_sink,
        topk_by_query=None,
        window_size=128,
        scale=0.25,
    )
    (ref.out * grad_out).sum().backward()

    assert ref_query.grad is not None
    assert ref_raw.grad is not None
    assert ref_projected.grad is not None
    assert ref_gate.grad is not None
    assert ref_bias.grad is not None
    assert ref_sink.grad is not None
    _assert_id_aligned_rows_close(
        actual=backward.attention.dq,
        actual_ids=backward.attention.query_token_ids,
        expected=ref_query.grad.unsqueeze(0),
    )
    _assert_id_aligned_rows_close(
        actual=backward.attention.draw_kv,
        actual_ids=backward.attention.raw_token_ids,
        expected=ref_raw.grad.unsqueeze(0),
    )
    _assert_id_aligned_rows_close(
        actual=backward.main_compressor.dprojected_kv,
        actual_ids=backward.main_compressor.token_ids,
        expected=ref_projected.grad,
    )
    _assert_id_aligned_rows_close(
        actual=backward.main_compressor.dprojected_gate,
        actual_ids=backward.main_compressor.token_ids,
        expected=ref_gate.grad,
    )
    torch.testing.assert_close(
        backward.main_compressor.dpositional_bias,
        ref_bias.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        backward.attention.d_attn_sink,
        ref_sink.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    assert not bool(backward.attention.dq.abs().sum().eq(0).item())
    assert not bool(backward.main_compressor.dprojected_gate.abs().sum().eq(0).item())
    if expect_rank0_halo_grad and rank == 0:
        assert not bool(backward.main_compressor.dprojected_kv.abs().sum().eq(0).item())
        assert not bool(
            backward.main_compressor.dprojected_gate.abs().sum().eq(0).item()
        )


def _assert_id_aligned_rows_close(
    *,
    actual: torch.Tensor,
    actual_ids: tuple[int, ...],
    expected: torch.Tensor,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    positions = torch.tensor(actual_ids, device=expected.device, dtype=torch.long)
    token_dim = 0 if expected.ndim == 2 else 1
    torch.testing.assert_close(
        actual,
        expected.index_select(token_dim, positions),
        rtol=rtol,
        atol=atol,
    )


def _dense_fake_fwd(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk: torch.Tensor,
    scale: float | None = None,
) -> Dsv4SparseForwardResult:
    del attn_sink
    selected, valid = _selected_stage_kv(kv=kv, topk=topk)
    logits = torch.einsum("bqhd,bqld->bqhl", q.float(), selected.float()) * float(
        1.0 if scale is None else scale
    )
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    weights = _safe_stage_weights(logits=logits, lse=lse)
    out = torch.einsum("bqhl,bqld->bqhd", weights.to(selected.dtype), selected)
    out = torch.where(torch.isneginf(lse).unsqueeze(-1), torch.zeros_like(out), out)
    return Dsv4SparseForwardResult(out=out.to(dtype=q.dtype), lse=lse)


def _dense_fake_bwd(
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
    scale_value = float(1.0 if scale is None else scale)
    selected, valid = _selected_stage_kv(kv=kv, topk=topk)
    logits = torch.einsum("bqhd,bqld->bqhl", q.float(), selected.float()) * scale_value
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    weights = torch.exp(logits - global_lse.unsqueeze(-1).float())
    weights = torch.where(valid.unsqueeze(2), weights, torch.zeros_like(weights))
    value_grad = weights.unsqueeze(-1) * grad_out.float().unsqueeze(-2)
    score_dot = (
        (selected.float().unsqueeze(2) - global_out.float().unsqueeze(-2))
        * grad_out.float().unsqueeze(-2)
    ).sum(dim=-1)
    dlogits = weights * score_dot
    dq = torch.einsum("bqhl,bqld->bqhd", dlogits, selected.float()) * scale_value
    dselected_from_key = (
        torch.einsum("bqhl,bqhd->bqhld", dlogits, q.float()) * scale_value
    )
    dselected = value_grad + dselected_from_key
    dkv = torch.zeros_like(kv)
    safe_topk = topk.clamp_min(0).to(device=kv.device)
    for batch in range(int(kv.shape[0])):
        for query_index in range(int(q.shape[1])):
            for list_index in range(int(topk.shape[-1])):
                if not bool(valid[batch, query_index, list_index].item()):
                    continue
                key_index = int(safe_topk[batch, query_index, list_index].item())
                dkv[batch, key_index] += (
                    dselected[
                        batch,
                        query_index,
                        :,
                        list_index,
                    ]
                    .sum(dim=0)
                    .to(dtype=dkv.dtype)
                )
    return Dsv4SparseBackwardResult(
        dq=dq.to(dtype=q.dtype),
        dkv=dkv,
        d_attn_sink=torch.zeros_like(attn_sink),
    )


def _selected_stage_kv(
    *,
    kv: torch.Tensor,
    topk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk = topk.to(device=kv.device, dtype=torch.long)
    if topk.shape[0] == 1 and int(kv.shape[0]) != 1:
        topk = topk.expand(int(kv.shape[0]), *topk.shape[1:])
    valid = topk >= 0
    safe_topk = topk.clamp_min(0)
    batch_ids = torch.arange(int(kv.shape[0]), device=kv.device).view(-1, 1, 1)
    batch_ids = batch_ids.expand_as(safe_topk)
    selected = kv[batch_ids, safe_topk]
    selected = torch.where(valid.unsqueeze(-1), selected, torch.zeros_like(selected))
    return selected, valid


def _safe_stage_weights(*, logits: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    diff = logits - lse.unsqueeze(-1)
    both_neg_inf = torch.isneginf(logits) & torch.isneginf(lse).unsqueeze(-1)
    diff = torch.where(both_neg_inf, torch.full_like(diff, float("-inf")), diff)
    return torch.exp(diff)
