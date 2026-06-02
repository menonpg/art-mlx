from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from artifacts import Dsv4Metric, write_manifest, write_readable_summary
from cases import Dsv4WorkloadCase, canonical_benchmark_cases, default_validation_cases
from packed_layout import build_dsv4_packed_tensors, summarize_case
from pydantic import BaseModel, ConfigDict, Field
import torch


class PlanningTiming(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_name: str
    topology: str
    cp_rank: int
    iterations: int
    warmup: int
    normal_cp_plan_ms: tuple[float, ...]
    dsv4_plan_ms: tuple[float, ...]
    total_plan_ms: tuple[float, ...]
    threshold_ms: float

    def metrics(self) -> tuple[Dsv4Metric, ...]:
        prefix = f"{self.case_name}_{self.topology}"
        total_median = _median(self.total_plan_ms)
        total_p90 = _percentile(self.total_plan_ms, 90)
        total_max = max(self.total_plan_ms)
        return (
            Dsv4Metric(
                name=f"{prefix}_normal_cp_plan_median",
                value=_median(self.normal_cp_plan_ms),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_dsv4_plan_median",
                value=_median(self.dsv4_plan_ms),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_total_plan_median",
                value=total_median,
                unit="ms",
                threshold=self.threshold_ms,
                passed=total_median < self.threshold_ms,
            ),
            Dsv4Metric(
                name=f"{prefix}_total_plan_p90",
                value=total_p90,
                unit="ms",
                threshold=self.threshold_ms,
                passed=total_p90 < self.threshold_ms,
            ),
            Dsv4Metric(
                name=f"{prefix}_total_plan_max",
                value=total_max,
                unit="ms",
                threshold=self.threshold_ms,
                passed=total_max < self.threshold_ms,
            ),
        )


class LabResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    manifest_path: Path
    summary_path: Path
    timings: tuple[PlanningTiming, ...] = Field(default_factory=tuple)


PlanningTiming.model_rebuild()
LabResult.model_rebuild()


def run_dry_run_cases(
    *,
    output_dir: Path,
    cases: tuple[Dsv4WorkloadCase, ...],
    command: list[str],
) -> LabResult:
    case_summaries = tuple(summarize_case(case) for case in cases)
    manifest_path = write_manifest(
        output_dir,
        kind="dsv4_layer_lab_dry_run",
        command=command,
        configs={"mode": "dry_run_cases"},
        cases=case_summaries,
        caveats=("dry-run only; no CP execution or timing",),
    )
    summary_path = write_readable_summary(
        output_dir,
        title="DSV4 Layer Lab Dry Run",
        status="pass",
        manifest_path=manifest_path,
        case_summaries=case_summaries,
        caveats=("dry-run only; no CP execution or timing",),
    )
    return LabResult(
        status="pass", manifest_path=manifest_path, summary_path=summary_path
    )


def run_planning_benchmark(
    *,
    output_dir: Path,
    cases: tuple[Dsv4WorkloadCase, ...],
    topologies: tuple[int, ...],
    iterations: int,
    warmup: int,
    threshold_ms: float,
    command: list[str],
    block_size: int = 128,
    planner_chunk_size: int = 512,
    csa_ratio: int = 4,
    hca_ratio: int = 128,
    include_csa: bool = True,
    include_hca: bool = True,
) -> LabResult:
    _require_megatron()
    timings: list[PlanningTiming] = []
    for case in cases:
        _require_single_packed_row(case)
        for cp_size in topologies:
            timings.append(
                _time_planning_case(
                    case=case,
                    cp_size=cp_size,
                    iterations=iterations,
                    warmup=warmup,
                    threshold_ms=threshold_ms,
                    block_size=block_size,
                    planner_chunk_size=planner_chunk_size,
                    csa_ratio=csa_ratio,
                    hca_ratio=hca_ratio,
                    include_csa=include_csa,
                    include_hca=include_hca,
                )
            )
    metrics = tuple(metric for timing in timings for metric in timing.metrics())
    passed = all(metric.passed is not False for metric in metrics)
    caveats = (
        "planning benchmark only; no layer projection, kernel, communication, or backward timing",
        "uses fresh group/parent ids per iteration so normal CP planning does not rely on cache hits",
    )
    case_summaries = tuple(summarize_case(case) for case in cases)
    manifest_path = write_manifest(
        output_dir,
        kind="dsv4_layer_lab_planning",
        command=command,
        configs={
            "mode": "benchmark_planning",
            "topologies": [f"cp{cp_size}" for cp_size in topologies],
            "iterations": iterations,
            "warmup": warmup,
            "threshold_ms": threshold_ms,
            "block_size": block_size,
            "planner_chunk_size": planner_chunk_size,
            "csa_ratio": csa_ratio,
            "hca_ratio": hca_ratio,
            "include_csa": include_csa,
            "include_hca": include_hca,
        },
        cases=case_summaries,
        metrics=metrics,
        caveats=caveats,
    )
    summary_path = write_readable_summary(
        output_dir,
        title="DSV4 Layer Lab Planning Benchmark",
        status="pass" if passed else "fail",
        manifest_path=manifest_path,
        case_summaries=case_summaries,
        metrics=metrics,
        caveats=caveats,
    )
    return LabResult(
        status="pass" if passed else "fail",
        manifest_path=manifest_path,
        summary_path=summary_path,
        timings=tuple(timings),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cases = _select_cases(case_set=args.case_set, names=tuple(args.case or ()))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("tests/integration/dsv4_context_parallel/scratch/layer_lab")
        / time.strftime("%Y%m%d_%H%M%S")
    )
    command = [Path(sys.argv[0]).name, *(sys.argv[1:] if argv is None else argv)]
    if args.benchmark_planning:
        result = run_planning_benchmark(
            output_dir=output_dir,
            cases=cases,
            topologies=tuple(_parse_topology(value) for value in args.topology),
            iterations=args.iterations,
            warmup=args.warmup,
            threshold_ms=args.planning_threshold_ms,
            command=command,
            block_size=args.block_size,
            planner_chunk_size=args.planner_chunk_size,
            csa_ratio=args.csa_ratio,
            hca_ratio=args.hca_ratio,
            include_csa=not args.hca_only,
            include_hca=True,
        )
    else:
        result = run_dry_run_cases(
            output_dir=output_dir,
            cases=cases,
            command=command,
        )
    print(f"summary: {result.summary_path}")
    print(f"manifest: {result.manifest_path}")
    return 0 if result.status == "pass" else 1


def _time_planning_case(
    *,
    case: Dsv4WorkloadCase,
    cp_size: int,
    iterations: int,
    warmup: int,
    threshold_ms: float,
    block_size: int,
    planner_chunk_size: int,
    csa_ratio: int,
    hca_ratio: int,
    include_csa: bool,
    include_hca: bool,
) -> PlanningTiming:
    from art.megatron.context_parallel import ContextParallelConfig, ParallelTopology
    from art.megatron.context_parallel.runtime import (
        prepare_megatron_context_parallel_state,
    )
    from art.megatron.dsv4 import prepare_dsv4_context_parallel_state

    if iterations < 1:
        raise RuntimeError(f"planning iterations must be positive, got {iterations}")
    if warmup < 0:
        raise RuntimeError(f"planning warmup must be non-negative, got {warmup}")
    topology = ParallelTopology(cp=int(cp_size))
    config = ContextParallelConfig(
        block_size=int(block_size),
        planner_chunk_size=int(planner_chunk_size),
    )
    cp_times: list[float] = []
    dsv4_times: list[float] = []
    total_count = int(warmup) + int(iterations)
    for iteration in range(total_count):
        micro = _fresh_micro(case, iteration=iteration)
        cp_start = time.perf_counter()
        cp_state, _rank_plan, _spec, _pad = prepare_megatron_context_parallel_state(
            micro=micro,
            topology=topology,
            config=config,
            cp_group=None,
            cp_rank=0,
        )
        cp_elapsed = _elapsed_ms(cp_start)
        dsv4_start = time.perf_counter()
        prepare_dsv4_context_parallel_state(
            cp_state=cp_state,
            csa_ratio=int(csa_ratio),
            hca_ratio=int(hca_ratio),
            include_csa=include_csa,
            include_hca=include_hca,
        )
        dsv4_elapsed = _elapsed_ms(dsv4_start)
        if iteration >= int(warmup):
            cp_times.append(cp_elapsed)
            dsv4_times.append(dsv4_elapsed)
    return PlanningTiming(
        case_name=case.name,
        topology=f"cp{cp_size}",
        cp_rank=0,
        iterations=int(iterations),
        warmup=int(warmup),
        normal_cp_plan_ms=tuple(cp_times),
        dsv4_plan_ms=tuple(dsv4_times),
        total_plan_ms=tuple(cp + dsv4 for cp, dsv4 in zip(cp_times, dsv4_times)),
        threshold_ms=float(threshold_ms),
    )


def _fresh_micro(case: Dsv4WorkloadCase, *, iteration: int) -> Any:
    from art.preprocessing.pack import PackedTensors

    tensors = build_dsv4_packed_tensors(case)
    offset = int(iteration + 1) * 10_000_000
    for name in ("group_ids", "parent_ids"):
        tensor = tensors[name]
        tensors[name] = torch.where(tensor >= 0, tensor + offset, tensor)
    return PackedTensors(
        tokens=tensors["tokens"],
        group_ids=tensors["group_ids"],
        parent_ids=tensors["parent_ids"],
        input_pos=tensors["input_pos"],
        assistant_mask=tensors["assistant_mask"],
        logprobs=tensors["logprobs"],
        advantages=tensors["advantages"],
        weights=tensors["weights"],
        pixel_values=tensors["pixel_values"],
        image_grid_thw=tensors["image_grid_thw"],
    )


def _select_cases(
    *, case_set: str, names: tuple[str, ...]
) -> tuple[Dsv4WorkloadCase, ...]:
    catalog = {case.name: case for case in _case_iter(case_set)}
    if not names:
        return tuple(catalog.values())
    missing = tuple(name for name in names if name not in catalog)
    if missing:
        raise RuntimeError(f"unknown DSV4 lab case names: {missing}")
    return tuple(catalog[name] for name in names)


def _case_iter(case_set: str) -> Iterable[Dsv4WorkloadCase]:
    if case_set == "validation":
        return default_validation_cases()
    if case_set == "canonical":
        return canonical_benchmark_cases()
    if case_set == "all":
        return (*default_validation_cases(), *canonical_benchmark_cases())
    raise RuntimeError(f"unsupported DSV4 lab case set {case_set}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DSV4 CP layer lab")
    parser.add_argument("--dry-run-cases", action="store_true")
    parser.add_argument("--benchmark-planning", action="store_true")
    parser.add_argument(
        "--case-set",
        choices=("validation", "canonical", "all"),
        default="canonical",
    )
    parser.add_argument("--case", action="append")
    parser.add_argument("--topology", action="append")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--planning-threshold-ms", type=float, default=120.0)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--planner-chunk-size", type=int, default=512)
    parser.add_argument("--csa-ratio", type=int, default=4)
    parser.add_argument("--hca-ratio", type=int, default=128)
    parser.add_argument("--hca-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    args = parser.parse_args(argv)
    if args.dry_run_cases and args.benchmark_planning:
        parser.error("--dry-run-cases and --benchmark-planning are mutually exclusive")
    if not args.dry_run_cases and not args.benchmark_planning:
        args.dry_run_cases = True
    if args.topology is None:
        args.topology = ["cp2"]
    return args


def _parse_topology(value: str) -> int:
    normalized = value.lower()
    if not normalized.startswith("cp"):
        raise RuntimeError(f"topology must look like cp2/cp4/cp8, got {value}")
    cp_size = int(normalized[2:])
    if cp_size <= 1:
        raise RuntimeError(f"planning benchmark requires CP > 1, got {value}")
    return cp_size


def _require_single_packed_row(case: Dsv4WorkloadCase) -> None:
    if len(case.rows) != 1:
        raise RuntimeError(
            f"ART CP planning expects one packed row, case {case.name} has {len(case.rows)}"
        )


def _require_megatron() -> None:
    try:
        import megatron.core  # noqa: F401
    except Exception as exc:
        raise RuntimeError("DSV4 planning lab requires megatron.core") from exc


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _median(values: tuple[float, ...]) -> float:
    return float(statistics.median(values))


def _percentile(values: tuple[float, ...], percentile: int) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((int(percentile) / 100.0) * (len(ordered) - 1)))
    return float(ordered[index])


if __name__ == "__main__":
    raise SystemExit(main())
