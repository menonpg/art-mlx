from __future__ import annotations

import json

from bench_dsv4_layer_lab import run_dry_run_cases, run_planning_benchmark
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
    pytest.importorskip("megatron.core")
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
