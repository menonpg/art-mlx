from __future__ import annotations

import json

from bench_stacked_context_parallel import (
    RankStackedOutput,
    RankStackedTiming,
    StackedBenchmarkConfig,
    _layer_kinds,
    _load_stacked_timing,
    run_stacked_dry_run_cases,
)
from cases import default_validation_cases


def test_stacked_dry_run_writes_manifest_and_summary(tmp_path) -> None:
    case = next(
        case
        for case in default_validation_cases()
        if case.name == "single_family_two_branches"
    )
    config = _stacked_config(layer_kinds=("csa", "hca"))

    result = run_stacked_dry_run_cases(
        output_dir=tmp_path,
        cases=(case,),
        config=config,
        command=["bench_stacked_context_parallel.py", "--dry-run-cases"],
    )

    assert result.status == "pass"
    manifest = json.loads(result.manifest_path.read_text())
    summary = result.summary_path.read_text()
    assert manifest["kind"] == "dsv4_stacked_context_parallel_dry_run"
    assert manifest["configs"]["layer_kinds"] == ["csa", "hca"]
    assert "DSV4 Stacked Context Parallel Dry Run" in summary
    assert "single_family_two_branches" in summary
    assert "dry-run only" in summary


def test_stacked_layer_kind_selection() -> None:
    assert _layer_kinds(exact=("hca", "csa"), pattern="alternating", count=4) == (
        "hca",
        "csa",
    )
    assert _layer_kinds(exact=(), pattern="alternating", count=4) == (
        "csa",
        "hca",
        "csa",
        "hca",
    )
    assert _layer_kinds(exact=(), pattern="csa", count=3) == ("csa", "csa", "csa")
    assert _layer_kinds(exact=(), pattern="hca", count=2) == ("hca", "hca")


def test_stacked_runtime_metrics_aggregate_rank_max(tmp_path) -> None:
    result_dir = tmp_path / "rank_results"
    result_dir.mkdir()
    rank0 = _rank_timing(
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
    rank1 = _rank_timing(
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
        path = result_dir / f"single_family_two_branches_cp2_rank{timing.rank}.json"
        path.write_text(RankStackedOutput(timing=timing).model_dump_json())

    stacked = _load_stacked_timing(
        result_dir=result_dir,
        case_name="single_family_two_branches",
        topology="cp2",
        world_size=2,
    )
    metrics = {metric.name: metric for metric in stacked.metrics()}

    assert (
        metrics["single_family_two_branches_cp2_stacked_total_plan_max"].value == 11.0
    )
    assert (
        metrics["single_family_two_branches_cp2_stacked_forward_total_median"].value
        == 17.5
    )
    assert (
        metrics["single_family_two_branches_cp2_stacked_backward_total_median"].value
        == 9.5
    )
    assert metrics["single_family_two_branches_cp2_stacked_e2e_median"].value == 37.5
    assert (
        metrics["single_family_two_branches_cp2_stacked_plan_plus_e2e_median"].value
        == 48.5
    )
    assert metrics["single_family_two_branches_cp2_stacked_layer_count"].value == 2.0
    assert (
        metrics[
            "single_family_two_branches_cp2_stacked_planned_explicit_comm_send_bytes_total"
        ].value
        == 82.0
    )
    assert (
        metrics[
            "single_family_two_branches_cp2_stacked_planned_explicit_comm_recv_bytes_total"
        ].value
        == 122.0
    )
    assert metrics["single_family_two_branches_cp2_stacked_output_abs_sum"].value == 5.0
    assert metrics["single_family_two_branches_cp2_stacked_dq_abs_sum"].passed is True


def _stacked_config(*, layer_kinds: tuple[str, ...]) -> StackedBenchmarkConfig:
    return StackedBenchmarkConfig(
        iterations=2,
        warmup=1,
        dtype="bf16",
        head_count=64,
        head_dim=512,
        indexer_dim=128,
        indexer_topk=1024,
        block_size=128,
        planner_chunk_size=512,
        csa_ratio=4,
        hca_ratio=128,
        window_size=128,
        layer_kinds=layer_kinds,
        miles_path="/mnt/ws_pvc/ws/scratch/miles_inspect",
    )


def _rank_timing(
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
) -> RankStackedTiming:
    return RankStackedTiming(
        case_name="single_family_two_branches",
        topology="cp2",
        rank=rank,
        world_size=2,
        iterations=2,
        warmup=1,
        dtype="bf16",
        layer_kinds=("csa", "hca"),
        token_count=32,
        local_token_count=16,
        completion_count=2,
        normal_cp_plan_ms=normal_cp_plan_ms,
        dsv4_plan_ms=dsv4_plan_ms,
        forward_launch_ms=(1.0, 1.0),
        forward_work_wait_ms=(4.0, 5.0),
        forward_total_ms=forward_total_ms,
        backward_launch_ms=(1.0, 1.0),
        backward_owner_wait_ms=(0.5, 1.0),
        backward_total_ms=backward_total_ms,
        e2e_ms=e2e_ms,
        peak_allocated_bytes=100 + rank,
        peak_reserved_bytes=200 + rank,
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
