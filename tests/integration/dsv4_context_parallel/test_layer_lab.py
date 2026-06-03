from __future__ import annotations

import json

from bench_dsv4_layer_lab import (
    RankRuntimeOutput,
    RankRuntimeTiming,
    _load_runtime_timings,
    _runtime_kinds,
    run_dry_run_cases,
    run_planning_benchmark,
)
from cases import default_validation_cases
import pytest


def test_layer_lab_dry_run_writes_manifest_and_summary(tmp_path) -> None:
    case = next(
        case
        for case in default_validation_cases()
        if case.name == "single_family_two_branches"
    )

    result = run_dry_run_cases(
        output_dir=tmp_path,
        cases=(case,),
        command=["bench_dsv4_layer_lab.py", "--dry-run-cases"],
    )

    assert result.status == "pass"
    manifest = json.loads(result.manifest_path.read_text())
    summary = result.summary_path.read_text()
    assert manifest["kind"] == "dsv4_layer_lab_dry_run"
    assert manifest["cases"][0]["name"] == "single_family_two_branches"
    assert "DSV4 Layer Lab Dry Run" in summary
    assert "single_family_two_branches" in summary
    assert "dry-run only" in summary


def test_layer_lab_planning_benchmark_uses_real_cp_and_dsv4_planner(tmp_path) -> None:
    _require_megatron_or_skip()
    case = next(
        case
        for case in default_validation_cases()
        if case.name == "single_family_two_branches"
    )

    result = run_planning_benchmark(
        output_dir=tmp_path,
        cases=(case,),
        topologies=(2,),
        iterations=1,
        warmup=0,
        threshold_ms=10_000.0,
        command=["bench_dsv4_layer_lab.py", "--benchmark-planning"],
        block_size=4,
        planner_chunk_size=4,
        hca_ratio=4,
    )

    assert result.status == "pass"
    manifest = json.loads(result.manifest_path.read_text())
    summary = result.summary_path.read_text()
    metric_names = {metric["name"] for metric in manifest["metrics"]}
    assert manifest["kind"] == "dsv4_layer_lab_planning"
    assert "single_family_two_branches_cp2_total_plan_median" in metric_names
    assert "normal_cp_plan_median" in summary
    assert "dsv4_plan_median" in summary


def test_layer_lab_runtime_metrics_aggregate_rank_max(tmp_path) -> None:
    result_dir = tmp_path / "rank_results"
    result_dir.mkdir()
    rank0 = _runtime_timing(
        rank=0,
        normal_cp_plan_ms=3.0,
        dsv4_plan_ms=4.0,
        forward_total_ms=(10.0, 20.0),
        backward_total_ms=(7.0, 9.0),
        e2e_ms=(30.0, 40.0),
        output_abs_sum=1.0,
        dq_abs_sum=2.0,
        compressor_grad_abs_sum=3.0,
    )
    rank1 = _runtime_timing(
        rank=1,
        normal_cp_plan_ms=5.0,
        dsv4_plan_ms=6.0,
        forward_total_ms=(15.0, 18.0),
        backward_total_ms=(6.0, 12.0),
        e2e_ms=(35.0, 32.0),
        output_abs_sum=4.0,
        dq_abs_sum=5.0,
        compressor_grad_abs_sum=6.0,
    )
    for timing in (rank0, rank1):
        output = RankRuntimeOutput(timings=(timing,))
        path = result_dir / f"single_family_two_branches_cp2_rank{timing.rank}.json"
        path.write_text(output.model_dump_json())

    runtime = _load_runtime_timings(
        result_dir=result_dir,
        case_name="single_family_two_branches",
        topology="cp2",
        kinds=("csa",),
        world_size=2,
    )[0]
    metrics = {metric.name: metric for metric in runtime.metrics()}

    assert _runtime_kinds("both") == ("csa", "hca")
    assert metrics["single_family_two_branches_cp2_csa_normal_cp_plan_max"].value == 5.0
    assert metrics["single_family_two_branches_cp2_csa_dsv4_plan_max"].value == 6.0
    assert (
        metrics["single_family_two_branches_cp2_csa_forward_total_median"].value == 17.5
    )
    assert (
        metrics["single_family_two_branches_cp2_csa_compression_forward_median"].value
        == 3.5
    )
    assert (
        metrics["single_family_two_branches_cp2_csa_attention_forward_median"].value
        == 6.5
    )
    assert (
        metrics["single_family_two_branches_cp2_csa_indexer_forward_median"].value
        == 1.5
    )
    assert (
        metrics[
            "single_family_two_branches_cp2_csa_exposed_forward_comm_wait_median"
        ].value
        == 4.5
    )
    assert (
        metrics["single_family_two_branches_cp2_csa_compression_halo_wait_median"].value
        == 1.5
    )
    assert (
        metrics["single_family_two_branches_cp2_csa_indexer_kv_wait_median"].value
        == 1.0
    )
    assert (
        metrics["single_family_two_branches_cp2_csa_stage_kv_wait_median"].value == 2.0
    )
    assert (
        metrics[
            "single_family_two_branches_cp2_csa_stage_materialization_forward_median"
        ].value
        == 2.5
    )
    assert (
        metrics["single_family_two_branches_cp2_csa_sparse_kernel_forward_median"].value
        == 4.5
    )
    assert (
        metrics["single_family_two_branches_cp2_csa_stage_merge_forward_median"].value
        == 0.5
    )
    assert (
        metrics["single_family_two_branches_cp2_csa_sink_merge_forward_median"].value
        == 0.25
    )
    assert (
        metrics[
            "single_family_two_branches_cp2_csa_exposed_backward_comm_wait_median"
        ].value
        == 0.75
    )
    assert (
        metrics[
            "single_family_two_branches_cp2_csa_planned_forward_comm_send_bytes_total"
        ].value
        == 21.0
    )
    assert (
        metrics[
            "single_family_two_branches_cp2_csa_planned_forward_comm_recv_bytes_total"
        ].value
        == 41.0
    )
    assert (
        metrics[
            "single_family_two_branches_cp2_csa_planned_backward_comm_send_bytes_total"
        ].value
        == 61.0
    )
    assert (
        metrics[
            "single_family_two_branches_cp2_csa_planned_backward_comm_recv_bytes_total"
        ].value
        == 81.0
    )
    assert (
        metrics[
            "single_family_two_branches_cp2_csa_planned_explicit_comm_send_bytes_total"
        ].value
        == 82.0
    )
    assert (
        metrics[
            "single_family_two_branches_cp2_csa_planned_explicit_comm_recv_bytes_total"
        ].value
        == 122.0
    )
    assert (
        metrics[
            "single_family_two_branches_cp2_csa_planned_explicit_comm_send_bytes_rank_max"
        ].value
        == 42.0
    )
    assert (
        metrics[
            "single_family_two_branches_cp2_csa_planned_explicit_comm_recv_bytes_rank_max"
        ].value
        == 62.0
    )
    assert metrics["single_family_two_branches_cp2_csa_e2e_median"].value == 37.5
    assert metrics["single_family_two_branches_cp2_csa_output_abs_sum"].value == 5.0
    assert metrics["single_family_two_branches_cp2_csa_dq_abs_sum"].passed is True


def _require_megatron_or_skip() -> None:
    try:
        import megatron.core  # noqa: F401
    except Exception as exc:
        pytest.skip(f"megatron.core is unavailable: {exc}")


def _runtime_timing(
    *,
    rank: int,
    normal_cp_plan_ms: float,
    dsv4_plan_ms: float,
    forward_total_ms: tuple[float, ...],
    backward_total_ms: tuple[float, ...],
    e2e_ms: tuple[float, ...],
    output_abs_sum: float,
    dq_abs_sum: float,
    compressor_grad_abs_sum: float,
) -> RankRuntimeTiming:
    return RankRuntimeTiming(
        case_name="single_family_two_branches",
        topology="cp2",
        compression_kind="csa",
        rank=rank,
        world_size=2,
        iterations=2,
        warmup=1,
        dtype="bf16",
        token_count=32,
        local_token_count=16,
        completion_count=2,
        normal_cp_plan_ms=normal_cp_plan_ms,
        dsv4_plan_ms=dsv4_plan_ms,
        forward_launch_ms=(1.0, 1.0),
        forward_wait_ms=(1.0, 1.0),
        forward_total_ms=forward_total_ms,
        compression_forward_ms=(3.0, 4.0),
        attention_forward_ms=(6.0, 7.0),
        indexer_forward_ms=(1.0, 2.0),
        exposed_forward_comm_wait_ms=(4.0, 5.0),
        compression_halo_wait_ms=(1.0, 2.0),
        indexer_kv_wait_ms=(0.5, 1.5),
        stage_kv_wait_ms=(1.5, 2.5),
        stage_materialization_forward_ms=(2.0, 3.0),
        sparse_kernel_forward_ms=(4.0, 5.0),
        stage_merge_forward_ms=(0.25, 0.75),
        sink_merge_forward_ms=(0.125, 0.375),
        backward_launch_ms=(1.0, 1.0),
        backward_wait_ms=(1.0, 1.0),
        exposed_backward_comm_wait_ms=(0.5, 1.0),
        backward_total_ms=backward_total_ms,
        e2e_ms=e2e_ms,
        peak_allocated_bytes=1024,
        peak_reserved_bytes=2048,
        planned_forward_comm_send_bytes=10 + rank,
        planned_forward_comm_recv_bytes=20 + rank,
        planned_backward_comm_send_bytes=30 + rank,
        planned_backward_comm_recv_bytes=40 + rank,
        planned_explicit_comm_send_bytes=40 + 2 * rank,
        planned_explicit_comm_recv_bytes=60 + 2 * rank,
        output_abs_sum=output_abs_sum,
        dq_abs_sum=dq_abs_sum,
        compressor_grad_abs_sum=compressor_grad_abs_sum,
        nonfinite_count=0,
    )
