from __future__ import annotations

import os
from pathlib import Path
import sys
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
    launch_dsv4_csa_projected_attention_forward_from_context_parallel_state,
    launch_dsv4_csa_projected_attention_forward_from_stage_plan_slots,
    launch_dsv4_hca_projected_attention_forward_from_context_parallel_state,
    launch_dsv4_hca_projected_attention_forward_from_stage_plan_slots,
    launch_dsv4_projected_attention_backward_from_context_parallel_state,
    launch_dsv4_projected_attention_backward_from_stage_plan_slots,
    materialize_dsv4_stage_tensors,
    prepare_dsv4_context_parallel_state,
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


def test_distributed_projected_csa_cp4_empty_rank_matches_packed_oracle(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_projected_csa_cp4_empty_rank_oracle_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_csa_cp4_empty_rank_oracle_worker,
        args=(4, str(init_path)),
        nprocs=4,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def test_distributed_projected_csa_cp8_empty_rank_matches_packed_oracle(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_projected_csa_cp8_empty_rank_oracle_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_csa_cp8_empty_rank_oracle_worker,
        args=(8, str(init_path)),
        nprocs=8,
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


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not cast(Any, torch.distributed).is_nccl_available(),
    reason="DSV4 real-Miles CUDA/NCCL oracle requires CUDA devices and NCCL",
)
@pytest.mark.parametrize(
    ("world_size", "master_port"),
    (
        (2, "29636"),
        (4, "29642"),
        (8, "29643"),
    ),
)
def test_distributed_projected_csa_real_miles_matches_packed_oracle_cuda_nccl(
    tmp_path: Path,
    world_size: int,
    master_port: str,
) -> None:
    _ensure_miles_sparse_available()
    if torch.cuda.device_count() < int(world_size):
        pytest.skip(f"requires {world_size} CUDA devices")
    init_path = tmp_path / f"dsv4_projected_csa_real_miles_cp{world_size}_oracle_nccl"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_csa_real_miles_nccl_oracle_worker,
        args=(int(world_size), str(init_path), master_port),
        nprocs=int(world_size),
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not cast(Any, torch.distributed).is_nccl_available(),
    reason="DSV4 real-Miles CUDA/NCCL oracle requires CUDA devices and NCCL",
)
@pytest.mark.parametrize(
    ("world_size", "master_port"),
    (
        (2, "29637"),
        (4, "29644"),
        (8, "29645"),
    ),
)
def test_distributed_projected_hca_real_miles_matches_packed_oracle_cuda_nccl(
    tmp_path: Path,
    world_size: int,
    master_port: str,
) -> None:
    _ensure_miles_sparse_available()
    if torch.cuda.device_count() < int(world_size):
        pytest.skip(f"requires {world_size} CUDA devices")
    init_path = tmp_path / f"dsv4_projected_hca_real_miles_cp{world_size}_oracle_nccl"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_hca_real_miles_nccl_oracle_worker,
        args=(int(world_size), str(init_path), master_port),
        nprocs=int(world_size),
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


@pytest.mark.parametrize(
    ("world_size", "master_port"),
    (
        (2, "29638"),
        (4, "29639"),
        (8, "29640"),
    ),
)
def test_distributed_projected_real_planner_context_state_matches_packed_oracle(
    tmp_path: Path,
    world_size: int,
    master_port: str,
) -> None:
    pytest.importorskip("megatron.core")
    init_path = tmp_path / f"dsv4_projected_real_planner_cp{world_size}_oracle_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_real_planner_oracle_worker,
        args=(int(world_size), str(init_path), master_port),
        nprocs=int(world_size),
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not cast(Any, torch.distributed).is_nccl_available(),
    reason="DSV4 real-planner real-Miles CUDA/NCCL oracle requires CUDA and NCCL",
)
@pytest.mark.parametrize(
    ("world_size", "master_port"),
    (
        (2, "29646"),
        (4, "29647"),
        (8, "29648"),
    ),
)
def test_distributed_projected_real_planner_real_miles_context_state_matches_packed_oracle_cuda_nccl(
    tmp_path: Path,
    world_size: int,
    master_port: str,
) -> None:
    pytest.importorskip("megatron.core")
    _ensure_miles_sparse_available()
    if torch.cuda.device_count() < int(world_size):
        pytest.skip(f"requires {world_size} CUDA devices")
    init_path = tmp_path / f"dsv4_real_planner_real_miles_cp{world_size}_oracle_nccl"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_real_planner_real_miles_nccl_oracle_worker,
        args=(int(world_size), str(init_path), master_port),
        nprocs=int(world_size),
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


def test_distributed_projected_hca_ratio128_cp4_empty_rank_matches_packed_oracle(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_projected_hca_ratio128_cp4_empty_rank_oracle_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_hca_ratio128_empty_rank_oracle_worker,
        args=(4, str(init_path), "29634"),
        nprocs=4,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def test_distributed_projected_hca_ratio128_cp8_empty_rank_matches_packed_oracle(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_projected_hca_ratio128_cp8_empty_rank_oracle_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_hca_ratio128_empty_rank_oracle_worker,
        args=(8, str(init_path), "29635"),
        nprocs=8,
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
def test_distributed_projected_hca_ratio128_boundary_matches_packed_oracle_cuda_nccl(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_projected_hca_ratio128_oracle_nccl"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_hca_ratio128_nccl_oracle_worker,
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
    reason="DSV4 HCA ratio-128 real-Miles CUDA/NCCL oracle requires two CUDA devices and NCCL",
)
def test_distributed_projected_hca_ratio128_real_miles_boundary_matches_packed_oracle_cuda_nccl(
    tmp_path: Path,
) -> None:
    _ensure_miles_sparse_available()
    init_path = tmp_path / "dsv4_projected_hca_ratio128_real_miles_oracle_nccl"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_hca_ratio128_real_miles_nccl_oracle_worker,
        args=(2, str(init_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not cast(Any, torch.distributed).is_nccl_available(),
    reason="DSV4 HCA ratio-128 real-Miles CUDA/NCCL empty-rank oracle requires CUDA devices and NCCL",
)
@pytest.mark.parametrize(
    ("world_size", "master_port"),
    (
        (4, "29650"),
        (8, "29651"),
    ),
)
def test_distributed_projected_hca_ratio128_real_miles_empty_rank_matches_packed_oracle_cuda_nccl(
    tmp_path: Path,
    world_size: int,
    master_port: str,
) -> None:
    _ensure_miles_sparse_available()
    if torch.cuda.device_count() < int(world_size):
        pytest.skip(f"requires {world_size} CUDA devices")
    init_path = (
        tmp_path
        / f"dsv4_projected_hca_ratio128_real_miles_cp{world_size}_empty_rank_oracle_nccl"
    )
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _distributed_projected_hca_ratio128_empty_rank_real_miles_nccl_oracle_worker,
        args=(int(world_size), str(init_path), master_port),
        nprocs=int(world_size),
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
    elif rank_count in (4, 8):
        ownership_ranges_by_rank = (
            ((0, 8, 0),),
            ((8, 13, 0),),
            ((13, 18, 0),),
            *(((),) * (rank_count - 3)),
        )
        token_counts_by_rank = (8, 5, 5) + ((0,) * (rank_count - 3))
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


def _four_rank_empty_slot() -> tuple[Dsv4StagePlanSlot, ...]:
    return _n_rank_empty_slot(rank_count=4)


def _empty_rank_token_ids_by_rank(*, rank_count: int) -> tuple[tuple[int, ...], ...]:
    if int(rank_count) < 3:
        raise RuntimeError(
            f"empty-rank token layout requires at least 3 ranks: {rank_count}"
        )
    return (
        tuple(range(0, 8)),
        tuple(range(8, 13)),
        tuple(range(13, 18)),
        *(((),) * (int(rank_count) - 3)),
    )


def _n_rank_empty_slot(*, rank_count: int) -> tuple[Dsv4StagePlanSlot, ...]:
    if int(rank_count) < 3:
        raise RuntimeError(f"empty-rank slot requires at least 3 ranks: {rank_count}")
    stage_plans = [
        _StagePlan(
            stage_index=0,
            global_q_ranges=(_Range(start=0, end=8),),
            global_k_ranges=(_Range(start=0, end=18),),
        ),
        _StagePlan(
            stage_index=0,
            global_q_ranges=(_Range(start=8, end=13),),
            global_k_ranges=(_Range(start=0, end=18),),
        ),
        _StagePlan(
            stage_index=0,
            global_q_ranges=(_Range(start=13, end=18),),
            global_k_ranges=(_Range(start=0, end=18),),
        ),
    ]
    for _ in range(int(rank_count) - 3):
        stage_plans.append(
            _StagePlan(
                stage_index=0,
                global_q_ranges=(),
                global_k_ranges=(_Range(start=0, end=18),),
            )
        )
    return (
        Dsv4StagePlanSlot(
            stage_index=0,
            stage_plans_by_rank=tuple(stage_plans),
        ),
    )


def _small_stage_slot(*, rank_count: int) -> tuple[Dsv4StagePlanSlot, ...]:
    if int(rank_count) == 2:
        return _two_rank_slot()
    return _n_rank_empty_slot(rank_count=int(rank_count))


def _small_token_ids_by_rank(*, rank_count: int) -> tuple[tuple[int, ...], ...]:
    if int(rank_count) == 2:
        return (tuple(range(0, 8)), tuple(range(8, 18)))
    return _empty_rank_token_ids_by_rank(rank_count=int(rank_count))


def _distributed_projected_real_planner_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
    master_port: str,
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = master_port
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_fwd", _dense_fake_fwd)
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_bwd", _dense_fake_bwd)
        context_state = _real_planner_dsv4_context_state(
            rank=rank,
            world_size=world_size,
        )
        local_token_ids = _local_token_ids_from_context_state(context_state)
        _run_real_planner_csa_context_oracle(
            context_state=context_state,
            local_token_ids=local_token_ids,
        )
        _run_real_planner_hca_context_oracle(
            context_state=context_state,
            local_token_ids=local_token_ids,
        )
    finally:
        destroy_process_group()


def _distributed_projected_real_planner_real_miles_nccl_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
    master_port: str,
) -> None:
    _add_miles_path_to_sys_path()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_process_group(
        "nccl",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        context_state = _real_planner_dsv4_context_state(
            rank=rank,
            world_size=world_size,
        )
        local_token_ids = _local_token_ids_from_context_state(context_state)
        _run_real_planner_csa_context_oracle(
            context_state=context_state,
            local_token_ids=local_token_ids,
            device=device,
            dtype=torch.bfloat16,
            head_count=64,
            head_dim=512,
            indexer_dim=128,
            scale=1.0 / (512**0.5),
            mean_abs_pct_threshold=3.0,
            grad_mean_abs_pct_threshold=5.0,
            name_prefix="real planner real Miles CSA",
        )
        _run_real_planner_hca_context_oracle(
            context_state=context_state,
            local_token_ids=local_token_ids,
            device=device,
            dtype=torch.bfloat16,
            head_count=64,
            head_dim=512,
            scale=1.0 / (512**0.5),
            mean_abs_pct_threshold=3.0,
            grad_mean_abs_pct_threshold=5.0,
            name_prefix="real planner real Miles HCA",
        )
        torch.cuda.synchronize(device)
    finally:
        destroy_process_group()


def _real_planner_dsv4_context_state(*, rank: int, world_size: int) -> Any:
    from art.megatron.context_parallel import ContextParallelConfig, ParallelTopology
    from art.megatron.context_parallel.runtime import (
        prepare_megatron_context_parallel_state,
    )

    cp_state, _rank_plan, _spec, _pad = prepare_megatron_context_parallel_state(
        micro=_real_planner_micro(),
        topology=ParallelTopology(cp=int(world_size)),
        config=ContextParallelConfig(block_size=4, planner_chunk_size=4),
        cp_group=cast(Any, torch.distributed).group.WORLD,
        cp_rank=int(rank),
    )
    return prepare_dsv4_context_parallel_state(cp_state=cp_state, hca_ratio=4)


def _real_planner_micro() -> Any:
    from art.preprocessing.pack import PackedTensors

    group_ids = torch.tensor([[0] * 8 + [1] * 8 + [2] * 8], dtype=torch.long)
    parent_ids = torch.tensor([[0] * 8 + [0] * 8 + [0] * 8], dtype=torch.long)
    seq_len = int(group_ids.shape[1])
    return PackedTensors(
        tokens=torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
        group_ids=group_ids,
        parent_ids=parent_ids,
        input_pos=torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
        assistant_mask=torch.ones(1, seq_len, dtype=torch.bool),
        logprobs=torch.zeros(1, seq_len, dtype=torch.float32),
        advantages=torch.ones(1, seq_len, dtype=torch.float32),
        weights=torch.ones(1, seq_len, dtype=torch.float32),
        pixel_values=[None],
        image_grid_thw=[None],
    )


def _local_token_ids_from_context_state(context_state: Any) -> tuple[int, ...]:
    rank = int(context_state.cp_state.rank_plan.rank)
    ranges = (
        context_state.cp_state.rank_plan.token_layout_index.ownership_ranges_by_rank[
            rank
        ]
    )
    return tuple(
        token_id
        for start, end, _offset in ranges
        for token_id in range(int(start), int(end))
    )


def _run_real_planner_csa_context_oracle(
    *,
    context_state: Any,
    local_token_ids: tuple[int, ...],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    head_count: int = 2,
    head_dim: int = 4,
    indexer_dim: int = 3,
    scale: float = 0.4,
    mean_abs_pct_threshold: float | None = None,
    grad_mean_abs_pct_threshold: float | None = None,
    name_prefix: str = "real planner CSA",
) -> None:
    layout = context_state.dsv4_plan.csa_layout
    if layout is None:
        raise RuntimeError("real planner CSA oracle requires CSA layout")
    token_count = 24
    if device is None:
        device = torch.device("cpu")
    use_mean_abs = mean_abs_pct_threshold is not None
    bias_dtype = torch.float32 if use_mean_abs else dtype
    torch.manual_seed(157)
    query = torch.randn(
        token_count,
        head_count,
        head_dim,
        device=device,
        dtype=dtype,
    )
    raw_kv = torch.randn(token_count, head_dim, device=device, dtype=dtype)
    main_projected_kv = torch.randn(
        token_count,
        2 * head_dim,
        device=device,
        dtype=dtype,
    )
    main_projected_gate = torch.randn(
        token_count,
        2 * head_dim,
        device=device,
        dtype=dtype,
    )
    main_positional_bias = torch.randn(
        4,
        2 * head_dim,
        device=device,
        dtype=bias_dtype,
    )
    indexer_projected_kv = torch.randn(
        token_count,
        2 * indexer_dim,
        device=device,
        dtype=dtype,
    )
    indexer_projected_gate = torch.randn(
        token_count,
        2 * indexer_dim,
        device=device,
        dtype=dtype,
    )
    indexer_positional_bias = torch.randn(
        4,
        2 * indexer_dim,
        device=device,
        dtype=bias_dtype,
    )
    indexer_q = torch.randn(
        token_count,
        head_count,
        indexer_dim,
        device=device,
        dtype=dtype,
    )
    indexer_weights = torch.randn(
        token_count,
        head_count,
        device=device,
        dtype=dtype,
    )
    attn_sink = torch.randn(head_count, device=device, dtype=bias_dtype) * 0.1
    grad_out = torch.randn(
        1,
        token_count,
        head_count,
        head_dim,
        device=device,
        dtype=dtype,
    )
    local_positions = torch.tensor(local_token_ids, device=device, dtype=torch.long)

    forward = launch_dsv4_csa_projected_attention_forward_from_context_parallel_state(
        context_state=context_state,
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
        async_op=True,
        scale=scale,
        window_size=128,
        raw_list_size=token_count,
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
        query_token_ids=tuple(range(token_count)),
        indexer_q=indexer_q,
        indexer_kv=indexer_compressed,
        indexer_weights=indexer_weights,
        candidate_entry_ids=tuple(range(len(layout.entries))),
        topk=2,
    ).indices[0]
    expected = dense_dsv4_packed_attention_oracle(
        layout=layout,
        query=query.float() if use_mean_abs else query,
        raw_kv=raw_kv.float() if use_mean_abs else raw_kv,
        compressed_kv=main_compressed.float() if use_mean_abs else main_compressed,
        attn_sink=attn_sink.float() if use_mean_abs else attn_sink,
        topk_by_query=topk,
        window_size=128,
        scale=scale,
    )
    if use_mean_abs:
        assert mean_abs_pct_threshold is not None
        if local_token_ids:
            threshold = float(mean_abs_pct_threshold)
            _assert_mean_abs_pct(
                forward.attention.out.float(),
                expected.out.index_select(1, local_positions).float(),
                threshold=threshold,
                name=f"{name_prefix} fwd",
            )
            _assert_mean_abs_pct(
                forward.attention.lse.float(),
                expected.lse.index_select(1, local_positions).float(),
                threshold=threshold,
                name=f"{name_prefix} lse",
            )
        else:
            assert forward.attention.out.numel() == 0
            assert forward.attention.lse.numel() == 0
    else:
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

    backward = launch_dsv4_projected_attention_backward_from_context_parallel_state(
        context_state=context_state,
        forward_result=forward,
        grad_out=grad_out.index_select(1, local_positions),
        async_op=True,
    ).wait_post_process()
    ref_query = query.detach().float() if use_mean_abs else query.detach().clone()
    ref_query.requires_grad_()
    ref_raw = raw_kv.detach().float() if use_mean_abs else raw_kv.detach().clone()
    ref_raw.requires_grad_()
    ref_main_projected = (
        main_projected_kv.detach().float()
        if use_mean_abs
        else main_projected_kv.detach().clone()
    )
    ref_main_projected.requires_grad_()
    ref_main_gate = (
        main_projected_gate.detach().float()
        if use_mean_abs
        else main_projected_gate.detach().clone()
    )
    ref_main_gate.requires_grad_()
    ref_main_bias = (
        main_positional_bias.detach().float()
        if use_mean_abs
        else main_positional_bias.detach().clone()
    )
    ref_main_bias.requires_grad_()
    ref_sink = (
        attn_sink.detach().float() if use_mean_abs else attn_sink.detach().clone()
    )
    ref_sink.requires_grad_()
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
        scale=scale,
    )
    (ref.out * (grad_out.float() if use_mean_abs else grad_out)).sum().backward()

    assert ref_query.grad is not None
    assert ref_raw.grad is not None
    assert ref_main_projected.grad is not None
    assert ref_main_gate.grad is not None
    assert ref_main_bias.grad is not None
    assert ref_sink.grad is not None
    if use_mean_abs:
        grad_threshold = (
            mean_abs_pct_threshold
            if grad_mean_abs_pct_threshold is None
            else grad_mean_abs_pct_threshold
        )
        assert grad_threshold is not None
        grad_threshold_value = float(grad_threshold)
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.attention.dq.float(),
            actual_ids=backward.attention.query_token_ids,
            expected=ref_query.grad.unsqueeze(0),
            threshold=grad_threshold_value,
            name=f"{name_prefix} dq",
        )
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.attention.draw_kv.float(),
            actual_ids=backward.attention.raw_token_ids,
            expected=ref_raw.grad.unsqueeze(0),
            threshold=grad_threshold_value,
            name=f"{name_prefix} draw",
        )
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.main_compressor.dprojected_kv.float(),
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_main_projected.grad,
            threshold=grad_threshold_value,
            name=f"{name_prefix} dprojected_kv",
        )
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.main_compressor.dprojected_gate.float(),
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_main_gate.grad,
            threshold=grad_threshold_value,
            name=f"{name_prefix} dprojected_gate",
        )
        _assert_mean_abs_pct(
            backward.main_compressor.dpositional_bias.float(),
            ref_main_bias.grad.float(),
            threshold=grad_threshold_value,
            name=f"{name_prefix} dpositional_bias",
        )
        _assert_mean_abs_pct(
            backward.attention.d_attn_sink.float(),
            ref_sink.grad.float(),
            threshold=grad_threshold_value,
            name=f"{name_prefix} dsink",
        )
    else:
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
            backward.main_compressor.dpositional_bias, ref_main_bias.grad
        )
        torch.testing.assert_close(backward.attention.d_attn_sink, ref_sink.grad)
    _assert_distributed_nonzero(backward.attention.dq, name=f"{name_prefix} dq")
    _assert_distributed_nonzero(
        backward.main_compressor.dprojected_kv,
        name=f"{name_prefix} dprojected_kv",
    )


def _run_real_planner_hca_context_oracle(
    *,
    context_state: Any,
    local_token_ids: tuple[int, ...],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    head_count: int = 2,
    head_dim: int = 4,
    scale: float = 0.25,
    mean_abs_pct_threshold: float | None = None,
    grad_mean_abs_pct_threshold: float | None = None,
    name_prefix: str = "real planner HCA",
) -> None:
    layout = context_state.dsv4_plan.hca_layout
    if layout is None:
        raise RuntimeError("real planner HCA oracle requires HCA layout")
    token_count = 24
    if device is None:
        device = torch.device("cpu")
    use_mean_abs = mean_abs_pct_threshold is not None
    bias_dtype = torch.float32 if use_mean_abs else dtype
    torch.manual_seed(163)
    query = torch.randn(
        token_count,
        head_count,
        head_dim,
        device=device,
        dtype=dtype,
    )
    raw_kv = torch.randn(token_count, head_dim, device=device, dtype=dtype)
    projected_kv = torch.randn(token_count, head_dim, device=device, dtype=dtype)
    projected_gate = torch.randn(token_count, head_dim, device=device, dtype=dtype)
    positional_bias = torch.randn(
        int(layout.spec.ratio),
        head_dim,
        device=device,
        dtype=bias_dtype,
    )
    attn_sink = torch.randn(head_count, device=device, dtype=bias_dtype) * 0.1
    grad_out = torch.randn(
        1,
        token_count,
        head_count,
        head_dim,
        device=device,
        dtype=dtype,
    )
    local_positions = torch.tensor(local_token_ids, device=device, dtype=torch.long)

    forward = launch_dsv4_hca_projected_attention_forward_from_context_parallel_state(
        context_state=context_state,
        query=query[list(local_token_ids)],
        query_token_ids=local_token_ids,
        raw_kv=raw_kv[list(local_token_ids)],
        raw_token_ids=local_token_ids,
        projected_kv=projected_kv[list(local_token_ids)],
        projected_gate=projected_gate[list(local_token_ids)],
        positional_bias=positional_bias,
        token_ids=local_token_ids,
        attn_sink=attn_sink,
        async_op=True,
        scale=scale,
        window_size=128,
        raw_list_size=token_count,
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
        query=query.float() if use_mean_abs else query,
        raw_kv=raw_kv.float() if use_mean_abs else raw_kv,
        compressed_kv=compressed.float() if use_mean_abs else compressed,
        attn_sink=attn_sink.float() if use_mean_abs else attn_sink,
        topk_by_query=None,
        window_size=128,
        scale=scale,
    )
    if use_mean_abs:
        assert mean_abs_pct_threshold is not None
        if local_token_ids:
            threshold = float(mean_abs_pct_threshold)
            _assert_mean_abs_pct(
                forward.attention.out.float(),
                expected.out.index_select(1, local_positions).float(),
                threshold=threshold,
                name=f"{name_prefix} fwd",
            )
            _assert_mean_abs_pct(
                forward.attention.lse.float(),
                expected.lse.index_select(1, local_positions).float(),
                threshold=threshold,
                name=f"{name_prefix} lse",
            )
        else:
            assert forward.attention.out.numel() == 0
            assert forward.attention.lse.numel() == 0
    else:
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

    backward = launch_dsv4_projected_attention_backward_from_context_parallel_state(
        context_state=context_state,
        forward_result=forward,
        grad_out=grad_out.index_select(1, local_positions),
        async_op=True,
    ).wait_post_process()
    ref_query = query.detach().float() if use_mean_abs else query.detach().clone()
    ref_query.requires_grad_()
    ref_raw = raw_kv.detach().float() if use_mean_abs else raw_kv.detach().clone()
    ref_raw.requires_grad_()
    ref_projected = (
        projected_kv.detach().float() if use_mean_abs else projected_kv.detach().clone()
    )
    ref_projected.requires_grad_()
    ref_gate = (
        projected_gate.detach().float()
        if use_mean_abs
        else projected_gate.detach().clone()
    )
    ref_gate.requires_grad_()
    ref_bias = (
        positional_bias.detach().float()
        if use_mean_abs
        else positional_bias.detach().clone()
    )
    ref_bias.requires_grad_()
    ref_sink = (
        attn_sink.detach().float() if use_mean_abs else attn_sink.detach().clone()
    )
    ref_sink.requires_grad_()
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
        scale=scale,
    )
    (ref.out * (grad_out.float() if use_mean_abs else grad_out)).sum().backward()

    assert ref_query.grad is not None
    assert ref_raw.grad is not None
    assert ref_projected.grad is not None
    assert ref_gate.grad is not None
    assert ref_bias.grad is not None
    assert ref_sink.grad is not None
    if use_mean_abs:
        grad_threshold = (
            mean_abs_pct_threshold
            if grad_mean_abs_pct_threshold is None
            else grad_mean_abs_pct_threshold
        )
        assert grad_threshold is not None
        grad_threshold_value = float(grad_threshold)
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.attention.dq.float(),
            actual_ids=backward.attention.query_token_ids,
            expected=ref_query.grad.unsqueeze(0),
            threshold=grad_threshold_value,
            name=f"{name_prefix} dq",
        )
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.attention.draw_kv.float(),
            actual_ids=backward.attention.raw_token_ids,
            expected=ref_raw.grad.unsqueeze(0),
            threshold=grad_threshold_value,
            name=f"{name_prefix} draw",
        )
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.main_compressor.dprojected_kv.float(),
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_projected.grad,
            threshold=grad_threshold_value,
            name=f"{name_prefix} dprojected_kv",
        )
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.main_compressor.dprojected_gate.float(),
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_gate.grad,
            threshold=grad_threshold_value,
            name=f"{name_prefix} dprojected_gate",
        )
        _assert_mean_abs_pct(
            backward.main_compressor.dpositional_bias.float(),
            ref_bias.grad.float(),
            threshold=grad_threshold_value,
            name=f"{name_prefix} dpositional_bias",
        )
        _assert_mean_abs_pct(
            backward.attention.d_attn_sink.float(),
            ref_sink.grad.float(),
            threshold=grad_threshold_value,
            name=f"{name_prefix} dsink",
        )
    else:
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
            backward.main_compressor.dpositional_bias, ref_bias.grad
        )
        torch.testing.assert_close(backward.attention.d_attn_sink, ref_sink.grad)
    _assert_distributed_nonzero(backward.attention.dq, name=f"{name_prefix} dq")
    _assert_distributed_nonzero(
        backward.main_compressor.dprojected_kv,
        name=f"{name_prefix} dprojected_kv",
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


def _distributed_projected_csa_cp4_empty_rank_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29633")
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_fwd", _dense_fake_fwd)
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_bwd", _dense_fake_bwd)
        layout = _layout(Dsv4CompressionKind.CSA, rank_count=world_size)
        torch.manual_seed(141)
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
        local_token_ids_by_rank = _empty_rank_token_ids_by_rank(rank_count=world_size)
        local_token_ids = local_token_ids_by_rank[rank]
        local_positions = torch.tensor(local_token_ids, dtype=torch.long)
        local_index = list(local_token_ids)

        forward = launch_dsv4_csa_projected_attention_forward_from_stage_plan_slots(
            layout=layout,
            rank=rank,
            stage_plan_slots=_n_rank_empty_slot(rank_count=world_size),
            query=query[local_index],
            query_token_ids=local_token_ids,
            raw_kv=raw_kv[local_index],
            raw_token_ids=local_token_ids,
            main_projected_kv=main_projected_kv[local_index],
            main_projected_gate=main_projected_gate[local_index],
            main_positional_bias=main_positional_bias,
            main_token_ids=local_token_ids,
            indexer_projected_kv=indexer_projected_kv[local_index],
            indexer_projected_gate=indexer_projected_gate[local_index],
            indexer_positional_bias=indexer_positional_bias,
            indexer_token_ids=local_token_ids,
            indexer_q=indexer_q[local_index],
            indexer_weights=indexer_weights[local_index],
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
            stage_plan_slots=_n_rank_empty_slot(rank_count=world_size),
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
        if local_token_ids:
            assert not bool(backward.attention.dq.abs().sum().eq(0).item())
            assert not bool(
                backward.main_compressor.dprojected_kv.abs().sum().eq(0).item()
            )
        else:
            assert backward.attention.dq.numel() == 0
            assert backward.attention.draw_kv.numel() == 0
            assert backward.main_compressor.dprojected_kv.numel() == 0
            assert backward.main_compressor.dprojected_gate.numel() == 0
    finally:
        destroy_process_group()


def _distributed_projected_csa_cp8_empty_rank_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
) -> None:
    _distributed_projected_csa_cp4_empty_rank_oracle_worker(
        rank=rank,
        world_size=world_size,
        init_path=init_path,
    )


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


def _distributed_projected_csa_real_miles_nccl_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
    master_port: str,
) -> None:
    _add_miles_path_to_sys_path()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_process_group(
        "nccl",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        layout = _layout(Dsv4CompressionKind.CSA, rank_count=world_size)
        stage_plan_slots = _small_stage_slot(rank_count=world_size)
        local_token_ids = _small_token_ids_by_rank(rank_count=world_size)[rank]
        torch.manual_seed(151)
        query = torch.randn(18, 64, 512, device=device, dtype=torch.bfloat16)
        raw_kv = torch.randn(18, 512, device=device, dtype=torch.bfloat16)
        main_projected_kv = torch.randn(
            18,
            1024,
            device=device,
            dtype=torch.bfloat16,
        )
        main_projected_gate = torch.randn(
            18,
            1024,
            device=device,
            dtype=torch.bfloat16,
        )
        main_positional_bias = torch.randn(4, 1024, device=device)
        indexer_projected_kv = torch.randn(
            18,
            256,
            device=device,
            dtype=torch.bfloat16,
        )
        indexer_projected_gate = torch.randn(
            18,
            256,
            device=device,
            dtype=torch.bfloat16,
        )
        indexer_positional_bias = torch.randn(4, 256, device=device)
        indexer_q = torch.randn(18, 64, 128, device=device, dtype=torch.bfloat16)
        indexer_weights = torch.randn(18, 64, device=device, dtype=torch.bfloat16)
        attn_sink = torch.randn(64, device=device) * 0.1
        grad_out = torch.randn(
            1,
            18,
            64,
            512,
            device=device,
            dtype=torch.bfloat16,
        )
        local_positions = torch.tensor(local_token_ids, device=device, dtype=torch.long)

        forward = launch_dsv4_csa_projected_attention_forward_from_stage_plan_slots(
            layout=layout,
            rank=rank,
            stage_plan_slots=stage_plan_slots,
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
            scale=1.0 / (512**0.5),
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
        assert main_compressed.dtype == query.dtype
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
            query=query.float(),
            raw_kv=raw_kv.float(),
            compressed_kv=main_compressed.float(),
            attn_sink=attn_sink.float(),
            topk_by_query=topk,
            window_size=128,
            scale=1.0 / (512**0.5),
        )
        if local_token_ids:
            _assert_mean_abs_pct(
                forward.attention.out.float(),
                expected.out.index_select(1, local_positions).float(),
                threshold=3.0,
                name="real Miles distributed CSA fwd",
            )
            _assert_mean_abs_pct(
                forward.attention.lse.float(),
                expected.lse.index_select(1, local_positions).float(),
                threshold=3.0,
                name="real Miles distributed CSA lse",
            )
        else:
            assert forward.attention.out.numel() == 0
            assert forward.attention.lse.numel() == 0

        backward = launch_dsv4_projected_attention_backward_from_stage_plan_slots(
            layout=layout,
            rank=rank,
            stage_plan_slots=stage_plan_slots,
            forward_result=forward,
            grad_out=grad_out.index_select(1, local_positions),
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
        ).wait_post_process()

        ref_query = query.detach().float().requires_grad_()
        ref_raw = raw_kv.detach().float().requires_grad_()
        ref_main_projected = main_projected_kv.detach().float().requires_grad_()
        ref_main_gate = main_projected_gate.detach().float().requires_grad_()
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
            scale=1.0 / (512**0.5),
        )
        (ref.out * grad_out.float()).sum().backward()

        assert ref_query.grad is not None
        assert ref_raw.grad is not None
        assert ref_main_projected.grad is not None
        assert ref_main_gate.grad is not None
        assert ref_main_bias.grad is not None
        assert ref_sink.grad is not None
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.attention.dq.float(),
            actual_ids=backward.attention.query_token_ids,
            expected=ref_query.grad.unsqueeze(0),
            threshold=5.0,
            name="real Miles distributed CSA dq",
        )
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.attention.draw_kv.float(),
            actual_ids=backward.attention.raw_token_ids,
            expected=ref_raw.grad.unsqueeze(0),
            threshold=5.0,
            name="real Miles distributed CSA draw",
        )
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.main_compressor.dprojected_kv.float(),
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_main_projected.grad,
            threshold=5.0,
            name="real Miles distributed CSA dprojected_kv",
        )
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.main_compressor.dprojected_gate.float(),
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_main_gate.grad,
            threshold=5.0,
            name="real Miles distributed CSA dprojected_gate",
        )
        _assert_mean_abs_pct(
            backward.main_compressor.dpositional_bias.float(),
            ref_main_bias.grad.float(),
            threshold=5.0,
            name="real Miles distributed CSA dpositional_bias",
        )
        _assert_mean_abs_pct(
            backward.attention.d_attn_sink.float(),
            ref_sink.grad.float(),
            threshold=5.0,
            name="real Miles distributed CSA dsink",
        )
        _assert_distributed_nonzero(
            forward.attention.out,
            name="real Miles distributed CSA fwd",
        )
        _assert_distributed_nonzero(
            backward.attention.dq,
            name="real Miles distributed CSA dq",
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


def _distributed_projected_hca_real_miles_nccl_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
    master_port: str,
) -> None:
    _add_miles_path_to_sys_path()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_process_group(
        "nccl",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        local_token_ids_by_rank = _small_token_ids_by_rank(rank_count=world_size)
        _run_distributed_projected_hca_oracle_case(
            rank=rank,
            layout=_layout(Dsv4CompressionKind.HCA, rank_count=world_size),
            stage_plan_slots=_small_stage_slot(rank_count=world_size),
            local_token_ids=local_token_ids_by_rank[rank],
            token_count=18,
            seed=153,
            expect_rank0_halo_grad=False,
            device=device,
            dtype=torch.bfloat16,
            head_count=64,
            head_dim=512,
            scale=1.0 / (512**0.5),
            mean_abs_pct_threshold=3.0,
            grad_mean_abs_pct_threshold=5.0,
        )
        torch.cuda.synchronize(device)
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
            rtol=1e-4,
            atol=1e-4,
        )
    finally:
        destroy_process_group()


def _distributed_projected_hca_ratio128_nccl_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29632")
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
            seed=139,
            expect_rank0_halo_grad=True,
            device=device,
            dtype=torch.float32,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.cuda.synchronize(device)
    finally:
        destroy_process_group()


def _distributed_projected_hca_ratio128_real_miles_nccl_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
) -> None:
    _add_miles_path_to_sys_path()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29649")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_process_group(
        "nccl",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
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
            seed=155,
            expect_rank0_halo_grad=True,
            device=device,
            dtype=torch.bfloat16,
            head_count=64,
            head_dim=512,
            scale=1.0 / (512**0.5),
            mean_abs_pct_threshold=3.0,
            grad_mean_abs_pct_threshold=5.0,
        )
        torch.cuda.synchronize(device)
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


def _hca_ratio128_empty_rank_layout(*, rank_count: int) -> Dsv4CompressedLayout:
    if int(rank_count) < 3:
        raise RuntimeError(
            f"HCA ratio-128 empty-rank layout requires at least 3 ranks: {rank_count}"
        )
    return build_dsv4_compressed_layout(
        group_ids=torch.tensor([[0] * 96 + [1] * 160 + [2] * 160]),
        parent_ids=torch.tensor([[0] * 96 + [0] * 160 + [0] * 160]),
        token_layout_index=_LayoutIndex(
            ownership_ranges_by_rank=(
                ((0, 96, 0),),
                ((96, 256, 0),),
                ((256, 416, 0),),
                *(((),) * (int(rank_count) - 3)),
            ),
            token_counts_by_rank=(96, 160, 160) + ((0,) * (int(rank_count) - 3)),
        ),
        spec=Dsv4CompressionSpec(kind=Dsv4CompressionKind.HCA, ratio=128),
    )


def _hca_ratio128_empty_token_ids_by_rank(
    *,
    rank_count: int,
) -> tuple[tuple[int, ...], ...]:
    if int(rank_count) < 3:
        raise RuntimeError(
            f"HCA ratio-128 empty-token layout requires at least 3 ranks: {rank_count}"
        )
    return (
        tuple(range(0, 96)),
        tuple(range(96, 256)),
        tuple(range(256, 416)),
        *(((),) * (int(rank_count) - 3)),
    )


def _hca_ratio128_empty_stage_slots(
    *,
    rank_count: int,
) -> tuple[Dsv4StagePlanSlot, ...]:
    if int(rank_count) < 3:
        raise RuntimeError(
            f"HCA ratio-128 empty-rank slot requires at least 3 ranks: {rank_count}"
        )
    stage_plans = [
        _StagePlan(
            stage_index=0,
            global_q_ranges=(_Range(start=0, end=96),),
            global_k_ranges=(_Range(start=0, end=416),),
        ),
        _StagePlan(
            stage_index=0,
            global_q_ranges=(_Range(start=96, end=256),),
            global_k_ranges=(_Range(start=0, end=416),),
        ),
        _StagePlan(
            stage_index=0,
            global_q_ranges=(_Range(start=256, end=416),),
            global_k_ranges=(_Range(start=0, end=416),),
        ),
    ]
    for _ in range(int(rank_count) - 3):
        stage_plans.append(
            _StagePlan(
                stage_index=0,
                global_q_ranges=(),
                global_k_ranges=(_Range(start=0, end=416),),
            )
        )
    return (
        Dsv4StagePlanSlot(
            stage_index=0,
            stage_plans_by_rank=tuple(stage_plans),
        ),
    )


def _distributed_projected_hca_ratio128_empty_rank_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
    master_port: str,
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = master_port
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_fwd", _dense_fake_fwd)
        setattr(cp_attention.sparse_kernel, "dsv4_sparse_bwd", _dense_fake_bwd)
        layout = _hca_ratio128_empty_rank_layout(rank_count=world_size)
        local_token_ids_by_rank = _hca_ratio128_empty_token_ids_by_rank(
            rank_count=world_size
        )
        _run_distributed_projected_hca_oracle_case(
            rank=rank,
            layout=layout,
            stage_plan_slots=_hca_ratio128_empty_stage_slots(rank_count=world_size),
            local_token_ids=local_token_ids_by_rank[rank],
            token_count=416,
            seed=143 + world_size,
            expect_rank0_halo_grad=True,
            rtol=1e-4,
            atol=1e-4,
        )
    finally:
        destroy_process_group()


def _distributed_projected_hca_ratio128_empty_rank_real_miles_nccl_oracle_worker(
    rank: int,
    world_size: int,
    init_path: str,
    master_port: str,
) -> None:
    _add_miles_path_to_sys_path()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_process_group(
        "nccl",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        layout = _hca_ratio128_empty_rank_layout(rank_count=world_size)
        local_token_ids_by_rank = _hca_ratio128_empty_token_ids_by_rank(
            rank_count=world_size
        )
        _run_distributed_projected_hca_oracle_case(
            rank=rank,
            layout=layout,
            stage_plan_slots=_hca_ratio128_empty_stage_slots(rank_count=world_size),
            local_token_ids=local_token_ids_by_rank[rank],
            token_count=416,
            seed=159 + int(world_size),
            expect_rank0_halo_grad=True,
            device=device,
            dtype=torch.bfloat16,
            head_count=64,
            head_dim=512,
            scale=1.0 / (512**0.5),
            mean_abs_pct_threshold=3.0,
            grad_mean_abs_pct_threshold=5.0,
        )
        torch.cuda.synchronize(device)
    finally:
        destroy_process_group()


def _run_distributed_projected_hca_oracle_case(
    *,
    rank: int,
    layout: Dsv4CompressedLayout,
    stage_plan_slots: tuple[Dsv4StagePlanSlot, ...],
    local_token_ids: tuple[int, ...],
    token_count: int,
    seed: int,
    expect_rank0_halo_grad: bool,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    head_count: int = 2,
    head_dim: int = 4,
    scale: float = 0.25,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    mean_abs_pct_threshold: float | None = None,
    grad_mean_abs_pct_threshold: float | None = None,
) -> None:
    if expect_rank0_halo_grad:
        assert layout.halo_transfers
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(seed)
    query = torch.randn(token_count, head_count, head_dim, device=device, dtype=dtype)
    raw_kv = torch.randn(token_count, head_dim, device=device, dtype=dtype)
    projected_kv = torch.randn(token_count, head_dim, device=device, dtype=dtype)
    projected_gate = torch.randn(token_count, head_dim, device=device, dtype=dtype)
    positional_bias = torch.randn(
        int(layout.spec.ratio),
        head_dim,
        device=device,
        dtype=dtype,
    )
    attn_sink = torch.randn(head_count, device=device, dtype=dtype)
    grad_out = torch.randn(
        1,
        token_count,
        head_count,
        head_dim,
        device=device,
        dtype=dtype,
    )

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
        scale=scale,
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
        query=query.float() if mean_abs_pct_threshold is not None else query,
        raw_kv=raw_kv.float() if mean_abs_pct_threshold is not None else raw_kv,
        compressed_kv=(
            compressed.float() if mean_abs_pct_threshold is not None else compressed
        ),
        attn_sink=attn_sink.float()
        if mean_abs_pct_threshold is not None
        else attn_sink,
        topk_by_query=None,
        window_size=128,
        scale=scale,
    )
    local_positions = torch.tensor(local_token_ids, device=device, dtype=torch.long)
    if mean_abs_pct_threshold is None:
        torch.testing.assert_close(
            forward.attention.out,
            expected.out.index_select(1, local_positions),
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            forward.attention.lse,
            expected.lse.index_select(1, local_positions),
            rtol=rtol,
            atol=atol,
            check_dtype=False,
        )
    else:
        if local_token_ids:
            _assert_mean_abs_pct(
                forward.attention.out.float(),
                expected.out.index_select(1, local_positions).float(),
                threshold=mean_abs_pct_threshold,
                name="real Miles distributed HCA fwd",
            )
            _assert_mean_abs_pct(
                forward.attention.lse.float(),
                expected.lse.index_select(1, local_positions).float(),
                threshold=mean_abs_pct_threshold,
                name="real Miles distributed HCA lse",
            )
        else:
            assert forward.attention.out.numel() == 0
            assert forward.attention.lse.numel() == 0

    backward = launch_dsv4_projected_attention_backward_from_stage_plan_slots(
        layout=layout,
        rank=rank,
        stage_plan_slots=stage_plan_slots,
        forward_result=forward,
        grad_out=grad_out.index_select(1, local_positions),
        group=cast(Any, torch.distributed).group.WORLD,
        async_op=True,
    ).wait_post_process()

    use_mean_abs = mean_abs_pct_threshold is not None
    grad_threshold = (
        mean_abs_pct_threshold
        if grad_mean_abs_pct_threshold is None
        else grad_mean_abs_pct_threshold
    )
    ref_query = query.detach().float() if use_mean_abs else query.detach().clone()
    ref_query.requires_grad_()
    ref_raw = raw_kv.detach().float() if use_mean_abs else raw_kv.detach().clone()
    ref_raw.requires_grad_()
    ref_projected = (
        projected_kv.detach().float() if use_mean_abs else projected_kv.detach().clone()
    )
    ref_projected.requires_grad_()
    ref_gate = (
        projected_gate.detach().float()
        if use_mean_abs
        else projected_gate.detach().clone()
    )
    ref_gate.requires_grad_()
    ref_bias = (
        positional_bias.detach().float()
        if use_mean_abs
        else positional_bias.detach().clone()
    )
    ref_bias.requires_grad_()
    ref_sink = (
        attn_sink.detach().float() if use_mean_abs else attn_sink.detach().clone()
    )
    ref_sink.requires_grad_()
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
        scale=scale,
    )
    (ref.out * (grad_out.float() if use_mean_abs else grad_out)).sum().backward()

    assert ref_query.grad is not None
    assert ref_raw.grad is not None
    assert ref_projected.grad is not None
    assert ref_gate.grad is not None
    assert ref_bias.grad is not None
    assert ref_sink.grad is not None
    if use_mean_abs:
        assert grad_threshold is not None
        grad_threshold_value = float(grad_threshold)
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.attention.dq.float(),
            actual_ids=backward.attention.query_token_ids,
            expected=ref_query.grad.unsqueeze(0),
            threshold=grad_threshold_value,
            name="real Miles distributed HCA dq",
        )
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.attention.draw_kv.float(),
            actual_ids=backward.attention.raw_token_ids,
            expected=ref_raw.grad.unsqueeze(0),
            threshold=grad_threshold_value,
            name="real Miles distributed HCA draw",
        )
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.main_compressor.dprojected_kv.float(),
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_projected.grad,
            threshold=grad_threshold_value,
            name="real Miles distributed HCA dprojected_kv",
        )
        _assert_id_aligned_rows_mean_abs_pct(
            actual=backward.main_compressor.dprojected_gate.float(),
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_gate.grad,
            threshold=grad_threshold_value,
            name="real Miles distributed HCA dprojected_gate",
        )
        _assert_mean_abs_pct(
            backward.main_compressor.dpositional_bias.float(),
            ref_bias.grad.float(),
            threshold=grad_threshold_value,
            name="real Miles distributed HCA dpositional_bias",
        )
        _assert_mean_abs_pct(
            backward.attention.d_attn_sink.float(),
            ref_sink.grad.float(),
            threshold=grad_threshold_value,
            name="real Miles distributed HCA dsink",
        )
    else:
        _assert_id_aligned_rows_close(
            actual=backward.attention.dq,
            actual_ids=backward.attention.query_token_ids,
            expected=ref_query.grad.unsqueeze(0),
            rtol=rtol,
            atol=atol,
        )
        _assert_id_aligned_rows_close(
            actual=backward.attention.draw_kv,
            actual_ids=backward.attention.raw_token_ids,
            expected=ref_raw.grad.unsqueeze(0),
            rtol=rtol,
            atol=atol,
        )
        _assert_id_aligned_rows_close(
            actual=backward.main_compressor.dprojected_kv,
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_projected.grad,
            rtol=rtol,
            atol=atol,
        )
        _assert_id_aligned_rows_close(
            actual=backward.main_compressor.dprojected_gate,
            actual_ids=backward.main_compressor.token_ids,
            expected=ref_gate.grad,
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            backward.main_compressor.dpositional_bias,
            ref_bias.grad,
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            backward.attention.d_attn_sink,
            ref_sink.grad,
            rtol=rtol,
            atol=atol,
        )
    if local_token_ids:
        assert not bool(backward.attention.dq.abs().sum().eq(0).item())
        assert not bool(
            backward.main_compressor.dprojected_gate.abs().sum().eq(0).item()
        )
    else:
        assert backward.attention.dq.numel() == 0
        assert backward.attention.draw_kv.numel() == 0
        assert backward.main_compressor.dprojected_kv.numel() == 0
        assert backward.main_compressor.dprojected_gate.numel() == 0
    if expect_rank0_halo_grad and rank == 0:
        assert not bool(backward.main_compressor.dprojected_kv.abs().sum().eq(0).item())
        assert not bool(
            backward.main_compressor.dprojected_gate.abs().sum().eq(0).item()
        )


def _ensure_miles_sparse_available() -> None:
    _add_miles_path_to_sys_path()
    pytest.importorskip(
        "miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_fwd"
    )
    pytest.importorskip(
        "miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_bwd"
    )


def _add_miles_path_to_sys_path() -> None:
    miles_path = os.environ.get(
        "DSV4_MILES_PATH", "/mnt/ws_pvc/ws/scratch/miles_inspect"
    )
    if miles_path and Path(miles_path).exists() and miles_path not in sys.path:
        sys.path.insert(0, miles_path)


def _assert_id_aligned_rows_mean_abs_pct(
    *,
    actual: torch.Tensor,
    actual_ids: tuple[int, ...],
    expected: torch.Tensor,
    threshold: float,
    name: str,
) -> None:
    if not actual_ids:
        assert actual.numel() == 0
        return
    positions = torch.tensor(actual_ids, device=expected.device, dtype=torch.long)
    token_dim = 0 if expected.ndim == 2 else 1
    target = expected.index_select(token_dim, positions).to(device=actual.device)
    _assert_mean_abs_pct(
        actual.float(),
        target.float(),
        threshold=threshold,
        name=name,
    )


def _assert_mean_abs_pct(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    threshold: float,
    name: str,
) -> None:
    denominator = expected.abs().mean().clamp_min(1e-8)
    value = float(((actual - expected).abs().mean() / denominator * 100.0).item())
    assert value <= float(threshold), (
        f"{name} mean_abs_pct {value:.6g}% exceeds threshold {threshold:g}%"
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


def _assert_distributed_nonzero(tensor: torch.Tensor, *, name: str) -> None:
    total = tensor.detach().abs().sum()
    if cast(Any, torch.distributed).is_initialized():
        total = total.clone()
        cast(Any, torch.distributed).all_reduce(total)
    assert not bool(total.eq(0).item()), f"{name} is zero across all ranks"


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
    if int(kv.shape[1]) > 0 and int(topk.shape[-1]) > 0:
        safe_topk = topk.to(device=kv.device, dtype=torch.long)
        if safe_topk.shape[0] == 1 and int(kv.shape[0]) != 1:
            safe_topk = safe_topk.expand(int(kv.shape[0]), *safe_topk.shape[1:])
        safe_topk = safe_topk.clamp_min(0)
        dselected_by_key = dselected.sum(dim=2).to(dtype=dkv.dtype)
        dselected_by_key = torch.where(
            valid.unsqueeze(-1),
            dselected_by_key,
            torch.zeros_like(dselected_by_key),
        )
        flat_index = safe_topk.reshape(int(kv.shape[0]), -1, 1).expand(
            -1,
            -1,
            int(kv.shape[-1]),
        )
        flat_source = dselected_by_key.reshape(int(kv.shape[0]), -1, int(kv.shape[-1]))
        dkv.scatter_add_(dim=1, index=flat_index, src=flat_source)
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
