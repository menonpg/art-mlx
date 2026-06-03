from __future__ import annotations

import argparse
from collections.abc import Iterable
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, cast

from artifacts import Dsv4Metric, write_manifest, write_readable_summary
from cases import Dsv4WorkloadCase, canonical_benchmark_cases, default_validation_cases
from packed_layout import build_dsv4_packed_tensors, summarize_case
from pydantic import BaseModel, ConfigDict, Field
import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.multiprocessing as mp


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


class RuntimeBenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    iterations: int
    warmup: int
    dtype: str
    head_count: int
    head_dim: int
    indexer_dim: int
    indexer_topk: int
    block_size: int
    planner_chunk_size: int
    csa_ratio: int
    hca_ratio: int
    window_size: int
    kinds: tuple[str, ...]
    miles_path: str


class RankRuntimeTiming(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_name: str
    topology: str
    compression_kind: str
    rank: int
    world_size: int
    iterations: int
    warmup: int
    dtype: str
    token_count: int
    local_token_count: int
    completion_count: int
    normal_cp_plan_ms: float
    dsv4_plan_ms: float
    forward_launch_ms: tuple[float, ...]
    forward_wait_ms: tuple[float, ...]
    forward_total_ms: tuple[float, ...]
    compression_forward_ms: tuple[float, ...]
    attention_forward_ms: tuple[float, ...]
    indexer_forward_ms: tuple[float, ...]
    stage_materialization_forward_ms: tuple[float, ...]
    sparse_merge_forward_ms: tuple[float, ...]
    backward_launch_ms: tuple[float, ...]
    backward_wait_ms: tuple[float, ...]
    backward_total_ms: tuple[float, ...]
    e2e_ms: tuple[float, ...]
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    output_abs_sum: float
    dq_abs_sum: float
    compressor_grad_abs_sum: float
    nonfinite_count: int


class RuntimeTiming(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_name: str
    topology: str
    compression_kind: str
    ranks: tuple[RankRuntimeTiming, ...]

    def metrics(self) -> tuple[Dsv4Metric, ...]:
        prefix = f"{self.case_name}_{self.topology}_{self.compression_kind}"
        forward_total = _iteration_rank_max(self.ranks, "forward_total_ms")
        compression_forward = _iteration_rank_max(self.ranks, "compression_forward_ms")
        attention_forward = _iteration_rank_max(self.ranks, "attention_forward_ms")
        indexer_forward = _iteration_rank_max(self.ranks, "indexer_forward_ms")
        stage_materialization_forward = _iteration_rank_max(
            self.ranks, "stage_materialization_forward_ms"
        )
        sparse_merge_forward = _iteration_rank_max(
            self.ranks, "sparse_merge_forward_ms"
        )
        backward_total = _iteration_rank_max(self.ranks, "backward_total_ms")
        e2e = _iteration_rank_max(self.ranks, "e2e_ms")
        token_count = max((rank.token_count for rank in self.ranks), default=0)
        completion_count = max(
            (rank.completion_count for rank in self.ranks), default=0
        )
        median_e2e = _median(e2e)
        seconds = max(median_e2e / 1000.0, 1e-12)
        nonfinite = sum(rank.nonfinite_count for rank in self.ranks)
        output_abs = sum(rank.output_abs_sum for rank in self.ranks)
        dq_abs = sum(rank.dq_abs_sum for rank in self.ranks)
        compressor_grad_abs = sum(rank.compressor_grad_abs_sum for rank in self.ranks)
        return (
            Dsv4Metric(
                name=f"{prefix}_normal_cp_plan_max",
                value=max(rank.normal_cp_plan_ms for rank in self.ranks),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_dsv4_plan_max",
                value=max(rank.dsv4_plan_ms for rank in self.ranks),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_forward_total_median",
                value=_median(forward_total),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_forward_total_p90",
                value=_percentile(forward_total, 90),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_compression_forward_median",
                value=_median(compression_forward),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_compression_forward_p90",
                value=_percentile(compression_forward, 90),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_attention_forward_median",
                value=_median(attention_forward),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_attention_forward_p90",
                value=_percentile(attention_forward, 90),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_indexer_forward_median",
                value=_median(indexer_forward),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_indexer_forward_p90",
                value=_percentile(indexer_forward, 90),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_stage_materialization_forward_median",
                value=_median(stage_materialization_forward),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_stage_materialization_forward_p90",
                value=_percentile(stage_materialization_forward, 90),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_sparse_merge_forward_median",
                value=_median(sparse_merge_forward),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_sparse_merge_forward_p90",
                value=_percentile(sparse_merge_forward, 90),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_backward_total_median",
                value=_median(backward_total),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_backward_total_p90",
                value=_percentile(backward_total, 90),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_e2e_median",
                value=median_e2e,
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_e2e_p90",
                value=_percentile(e2e, 90),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_tokens_per_second",
                value=float(token_count) / seconds,
                unit="tok/s",
            ),
            Dsv4Metric(
                name=f"{prefix}_examples_per_second",
                value=float(completion_count) / seconds,
                unit="examples/s",
            ),
            Dsv4Metric(
                name=f"{prefix}_peak_allocated_max",
                value=float(max(rank.peak_allocated_bytes for rank in self.ranks)),
                unit="bytes",
            ),
            Dsv4Metric(
                name=f"{prefix}_peak_reserved_max",
                value=float(max(rank.peak_reserved_bytes for rank in self.ranks)),
                unit="bytes",
            ),
            Dsv4Metric(
                name=f"{prefix}_nonfinite_count",
                value=float(nonfinite),
                unit="",
                threshold=0.0,
                passed=nonfinite == 0,
            ),
            Dsv4Metric(
                name=f"{prefix}_output_abs_sum",
                value=float(output_abs),
                unit="",
                threshold=0.0,
                passed=output_abs > 0.0,
            ),
            Dsv4Metric(
                name=f"{prefix}_dq_abs_sum",
                value=float(dq_abs),
                unit="",
                threshold=0.0,
                passed=dq_abs > 0.0,
            ),
            Dsv4Metric(
                name=f"{prefix}_compressor_grad_abs_sum",
                value=float(compressor_grad_abs),
                unit="",
                threshold=0.0,
                passed=compressor_grad_abs > 0.0,
            ),
        )


class RankRuntimeOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    timings: tuple[RankRuntimeTiming, ...]


class RuntimeForwardInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    main_projected_kv: torch.Tensor | None = None
    main_projected_gate: torch.Tensor | None = None
    main_positional_bias: torch.Tensor | None = None
    indexer_projected_kv: torch.Tensor | None = None
    indexer_projected_gate: torch.Tensor | None = None
    indexer_positional_bias: torch.Tensor | None = None
    indexer_q: torch.Tensor | None = None
    indexer_weights: torch.Tensor | None = None
    projected_kv: torch.Tensor | None = None
    projected_gate: torch.Tensor | None = None
    positional_bias: torch.Tensor | None = None


class RuntimeForwardPhase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    forward: Any
    launch_ms: float
    wait_ms: float
    total_ms: float
    compression_ms: float
    attention_ms: float
    indexer_ms: float
    stage_materialization_ms: float
    sparse_merge_ms: float


class RuntimeStageAttentionPhase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    attention: Any
    total_ms: float
    launch_ms: float
    materialization_ms: float
    sparse_merge_ms: float


class LabResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    manifest_path: Path
    summary_path: Path
    timings: tuple[PlanningTiming, ...] = Field(default_factory=tuple)
    runtime_timings: tuple[RuntimeTiming, ...] = Field(default_factory=tuple)


PlanningTiming.model_rebuild()
RuntimeBenchmarkConfig.model_rebuild()
RankRuntimeTiming.model_rebuild()
RuntimeTiming.model_rebuild()
RankRuntimeOutput.model_rebuild()
RuntimeForwardInputs.model_rebuild()
RuntimeForwardPhase.model_rebuild()
RuntimeStageAttentionPhase.model_rebuild()
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


def run_runtime_benchmark(
    *,
    output_dir: Path,
    cases: tuple[Dsv4WorkloadCase, ...],
    topologies: tuple[int, ...],
    config: RuntimeBenchmarkConfig,
    command: list[str],
) -> LabResult:
    _require_megatron()
    _require_cuda_for_runtime()
    _add_miles_path_to_sys_path(config.miles_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    timings: list[RuntimeTiming] = []
    for case in cases:
        _require_single_packed_row(case)
        for cp_size in topologies:
            if cp_size > torch.cuda.device_count():
                raise RuntimeError(
                    f"runtime benchmark requested cp{cp_size}, but only "
                    f"{torch.cuda.device_count()} CUDA devices are visible"
                )
            init_path = (
                output_dir
                / f"dist_{_safe_name(case.name)}_cp{cp_size}_{time.time_ns()}"
            )
            if init_path.exists():
                init_path.unlink()
            result_dir = output_dir / "rank_results"
            result_dir.mkdir(parents=True, exist_ok=True)
            mp.start_processes(
                _runtime_worker,
                args=(
                    int(cp_size),
                    str(init_path),
                    str(result_dir),
                    case.model_dump_json(),
                    config.model_dump_json(),
                ),
                nprocs=int(cp_size),
                join=True,
                start_method="spawn",
            )
            timings.extend(
                _load_runtime_timings(
                    result_dir=result_dir,
                    case_name=case.name,
                    topology=f"cp{cp_size}",
                    kinds=config.kinds,
                    world_size=cp_size,
                )
            )
    metrics = tuple(metric for timing in timings for metric in timing.metrics())
    passed = all(metric.passed is not False for metric in metrics)
    caveats = (
        "projected-input runtime benchmark; surrounding DSV4 model projections, RMSNorm, RoPE, and output projection are excluded until the non-CP DSV4 handler exists",
        "uses real prepared ART CP state plus public DSV4 compression, indexer/stage-attention, and projected-backward APIs",
        "forward phase timings split public compression, CSA indexer, stage exchange/materialization, and sparse-kernel/merge execution without production debug hooks",
        "phase medians and p90s are rank-max aggregates per phase and are diagnostic rather than additive",
        "warmup excludes first-use TileLang compilation and setup where the iteration count is large enough to amortize it",
    )
    case_summaries = tuple(summarize_case(case) for case in cases)
    manifest_path = write_manifest(
        output_dir,
        kind="dsv4_layer_lab_runtime",
        command=command,
        configs={
            "mode": "benchmark",
            "topologies": [f"cp{cp_size}" for cp_size in topologies],
            **config.model_dump(),
        },
        cases=case_summaries,
        metrics=metrics,
        caveats=caveats,
    )
    summary_path = write_readable_summary(
        output_dir,
        title="DSV4 Layer Lab Runtime Benchmark",
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
        runtime_timings=tuple(timings),
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
    if args.benchmark:
        result = run_runtime_benchmark(
            output_dir=output_dir,
            cases=cases,
            topologies=tuple(_parse_topology(value) for value in args.topology),
            config=RuntimeBenchmarkConfig(
                iterations=args.iterations,
                warmup=args.warmup,
                dtype=args.dtype,
                head_count=args.head_count,
                head_dim=args.head_dim,
                indexer_dim=args.indexer_dim,
                indexer_topk=args.indexer_topk,
                block_size=args.block_size,
                planner_chunk_size=args.planner_chunk_size,
                csa_ratio=args.csa_ratio,
                hca_ratio=args.hca_ratio,
                window_size=args.window_size,
                kinds=_runtime_kinds(args.runtime_kind),
                miles_path=args.miles_path,
            ),
            command=command,
        )
    elif args.benchmark_planning:
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


def _runtime_worker(
    rank: int,
    world_size: int,
    init_path: str,
    result_dir: str,
    case_json: str,
    config_json: str,
) -> None:
    from art.megatron.context_parallel import ContextParallelConfig, ParallelTopology
    from art.megatron.context_parallel.runtime import (
        prepare_megatron_context_parallel_state,
    )
    from art.megatron.dsv4 import prepare_dsv4_context_parallel_state

    case = Dsv4WorkloadCase.model_validate_json(case_json)
    config = RuntimeBenchmarkConfig.model_validate_json(config_json)
    _add_miles_path_to_sys_path(config.miles_path)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29671")
    torch.cuda.set_device(int(rank))
    device = torch.device("cuda", int(rank))
    init_process_group(
        "nccl",
        init_method=f"file://{init_path}",
        rank=int(rank),
        world_size=int(world_size),
    )
    try:
        micro = _fresh_micro(case, iteration=0)
        cp_start = time.perf_counter()
        cp_state, _rank_plan, _spec, _pad = prepare_megatron_context_parallel_state(
            micro=micro,
            topology=ParallelTopology(cp=int(world_size)),
            config=ContextParallelConfig(
                block_size=int(config.block_size),
                planner_chunk_size=int(config.planner_chunk_size),
            ),
            cp_group=cast(Any, torch.distributed).group.WORLD,
            cp_rank=int(rank),
        )
        normal_cp_plan_ms = _elapsed_ms(cp_start)
        dsv4_start = time.perf_counter()
        context_state = prepare_dsv4_context_parallel_state(
            cp_state=cp_state,
            csa_ratio=int(config.csa_ratio),
            hca_ratio=int(config.hca_ratio),
            include_csa="csa" in config.kinds,
            include_hca="hca" in config.kinds,
        )
        dsv4_plan_ms = _elapsed_ms(dsv4_start)
        local_token_ids = _local_token_ids_from_context_state(context_state)
        timings = tuple(
            _time_runtime_kind(
                context_state=context_state,
                case=case,
                config=config,
                compression_kind=kind,
                rank=int(rank),
                world_size=int(world_size),
                device=device,
                local_token_ids=local_token_ids,
                normal_cp_plan_ms=normal_cp_plan_ms,
                dsv4_plan_ms=dsv4_plan_ms,
            )
            for kind in config.kinds
        )
        output = RankRuntimeOutput(timings=timings)
        path = (
            Path(result_dir) / f"{_safe_name(case.name)}_cp{world_size}_rank{rank}.json"
        )
        path.write_text(output.model_dump_json(indent=2) + "\n")
    finally:
        destroy_process_group()


def _time_runtime_kind(
    *,
    context_state: Any,
    case: Dsv4WorkloadCase,
    config: RuntimeBenchmarkConfig,
    compression_kind: str,
    rank: int,
    world_size: int,
    device: torch.device,
    local_token_ids: tuple[int, ...],
    normal_cp_plan_ms: float,
    dsv4_plan_ms: float,
) -> RankRuntimeTiming:
    from art.megatron.dsv4 import (
        launch_dsv4_projected_attention_backward_from_context_parallel_state,
    )

    dtype = _dtype_from_name(config.dtype)
    token_count = int(context_state.cp_state.rank_plan.original_seq_len)
    local_count = len(local_token_ids)
    completion_count = summarize_case(case).completion_count
    torch.manual_seed(10_000 + rank * 97 + (0 if compression_kind == "csa" else 1))
    query = torch.randn(
        local_count,
        int(config.head_count),
        int(config.head_dim),
        device=device,
        dtype=dtype,
    )
    raw_kv = torch.randn(
        local_count,
        int(config.head_dim),
        device=device,
        dtype=dtype,
    )
    attn_sink = (
        torch.randn(
            int(config.head_count),
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    )
    grad_out = torch.randn(
        1,
        local_count,
        int(config.head_count),
        int(config.head_dim),
        device=device,
        dtype=dtype,
    )
    scale = 1.0 / (int(config.head_dim) ** 0.5)
    forward_launch_ms: list[float] = []
    forward_wait_ms: list[float] = []
    forward_total_ms: list[float] = []
    backward_launch_ms: list[float] = []
    backward_wait_ms: list[float] = []
    backward_total_ms: list[float] = []
    compression_forward_ms: list[float] = []
    attention_forward_ms: list[float] = []
    indexer_forward_ms: list[float] = []
    stage_materialization_forward_ms: list[float] = []
    sparse_merge_forward_ms: list[float] = []
    e2e_ms: list[float] = []
    output_abs_sum = 0.0
    dq_abs_sum = 0.0
    compressor_grad_abs_sum = 0.0
    nonfinite_count = 0
    forward_inputs = _build_runtime_forward_inputs(
        compression_kind=compression_kind,
        local_token_count=local_count,
        config=config,
        device=device,
        dtype=dtype,
    )
    if int(config.warmup) == 0:
        torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(int(config.warmup) + int(config.iterations)):
        if iteration == int(config.warmup):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        e2e_start = time.perf_counter()
        if compression_kind == "csa":
            forward_phase = _run_runtime_csa_forward(
                context_state=context_state,
                query=query,
                raw_kv=raw_kv,
                attn_sink=attn_sink,
                local_token_ids=local_token_ids,
                config=config,
                forward_inputs=forward_inputs,
                scale=scale,
                device=device,
            )
        elif compression_kind == "hca":
            forward_phase = _run_runtime_hca_forward(
                context_state=context_state,
                query=query,
                raw_kv=raw_kv,
                attn_sink=attn_sink,
                local_token_ids=local_token_ids,
                config=config,
                forward_inputs=forward_inputs,
                scale=scale,
                device=device,
            )
        else:
            raise RuntimeError(f"unsupported runtime kind {compression_kind}")
        forward = forward_phase.forward
        bwd_start = time.perf_counter()
        backward_work = (
            launch_dsv4_projected_attention_backward_from_context_parallel_state(
                context_state=context_state,
                forward_result=forward,
                grad_out=grad_out,
                async_op=True,
            )
        )
        bwd_launch = _elapsed_ms(bwd_start)
        bwd_wait_start = time.perf_counter()
        backward = backward_work.wait_post_process()
        torch.cuda.synchronize(device)
        bwd_wait = _elapsed_ms(bwd_wait_start)
        bwd_total = _elapsed_ms(bwd_start)
        total = _elapsed_ms(e2e_start)
        if iteration >= int(config.warmup):
            forward_launch_ms.append(forward_phase.launch_ms)
            forward_wait_ms.append(forward_phase.wait_ms)
            forward_total_ms.append(forward_phase.total_ms)
            compression_forward_ms.append(forward_phase.compression_ms)
            attention_forward_ms.append(forward_phase.attention_ms)
            indexer_forward_ms.append(forward_phase.indexer_ms)
            stage_materialization_forward_ms.append(
                forward_phase.stage_materialization_ms
            )
            sparse_merge_forward_ms.append(forward_phase.sparse_merge_ms)
            backward_launch_ms.append(bwd_launch)
            backward_wait_ms.append(bwd_wait)
            backward_total_ms.append(bwd_total)
            e2e_ms.append(total)
            nonfinite_count += _nonfinite_count(forward.attention.out)
            nonfinite_count += _nonfinite_count(forward.attention.lse)
            nonfinite_count += _nonfinite_count(backward.attention.dq)
            nonfinite_count += _nonfinite_count(backward.main_compressor.dprojected_kv)
            output_abs_sum += _abs_sum(forward.attention.out)
            dq_abs_sum += _abs_sum(backward.attention.dq)
            compressor_grad_abs_sum += _abs_sum(backward.main_compressor.dprojected_kv)
    return RankRuntimeTiming(
        case_name=case.name,
        topology=f"cp{world_size}",
        compression_kind=compression_kind,
        rank=int(rank),
        world_size=int(world_size),
        iterations=int(config.iterations),
        warmup=int(config.warmup),
        dtype=config.dtype,
        token_count=token_count,
        local_token_count=local_count,
        completion_count=completion_count,
        normal_cp_plan_ms=float(normal_cp_plan_ms),
        dsv4_plan_ms=float(dsv4_plan_ms),
        forward_launch_ms=tuple(forward_launch_ms),
        forward_wait_ms=tuple(forward_wait_ms),
        forward_total_ms=tuple(forward_total_ms),
        compression_forward_ms=tuple(compression_forward_ms),
        attention_forward_ms=tuple(attention_forward_ms),
        indexer_forward_ms=tuple(indexer_forward_ms),
        stage_materialization_forward_ms=tuple(stage_materialization_forward_ms),
        sparse_merge_forward_ms=tuple(sparse_merge_forward_ms),
        backward_launch_ms=tuple(backward_launch_ms),
        backward_wait_ms=tuple(backward_wait_ms),
        backward_total_ms=tuple(backward_total_ms),
        e2e_ms=tuple(e2e_ms),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        output_abs_sum=float(output_abs_sum),
        dq_abs_sum=float(dq_abs_sum),
        compressor_grad_abs_sum=float(compressor_grad_abs_sum),
        nonfinite_count=int(nonfinite_count),
    )


def _build_runtime_forward_inputs(
    *,
    compression_kind: str,
    local_token_count: int,
    config: RuntimeBenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> RuntimeForwardInputs:
    if compression_kind == "csa":
        main_projected_kv = torch.randn(
            int(local_token_count),
            2 * int(config.head_dim),
            device=device,
            dtype=dtype,
        )
        indexer_projected_kv = torch.randn(
            int(local_token_count),
            2 * int(config.indexer_dim),
            device=device,
            dtype=dtype,
        )
        return RuntimeForwardInputs(
            main_projected_kv=main_projected_kv,
            main_projected_gate=torch.randn_like(main_projected_kv),
            main_positional_bias=torch.randn(
                int(config.csa_ratio),
                2 * int(config.head_dim),
                device=device,
                dtype=torch.float32,
            ),
            indexer_projected_kv=indexer_projected_kv,
            indexer_projected_gate=torch.randn_like(indexer_projected_kv),
            indexer_positional_bias=torch.randn(
                int(config.csa_ratio),
                2 * int(config.indexer_dim),
                device=device,
                dtype=torch.float32,
            ),
            indexer_q=torch.randn(
                int(local_token_count),
                int(config.head_count),
                int(config.indexer_dim),
                device=device,
                dtype=dtype,
            ),
            indexer_weights=torch.randn(
                int(local_token_count),
                int(config.head_count),
                device=device,
                dtype=dtype,
            ),
        )
    if compression_kind == "hca":
        projected_kv = torch.randn(
            int(local_token_count),
            int(config.head_dim),
            device=device,
            dtype=dtype,
        )
        return RuntimeForwardInputs(
            projected_kv=projected_kv,
            projected_gate=torch.randn_like(projected_kv),
            positional_bias=torch.randn(
                int(config.hca_ratio),
                int(config.head_dim),
                device=device,
                dtype=torch.float32,
            ),
        )
    raise RuntimeError(f"unsupported runtime kind {compression_kind}")


def _run_runtime_csa_forward(
    *,
    context_state: Any,
    query: torch.Tensor,
    raw_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    local_token_ids: tuple[int, ...],
    config: RuntimeBenchmarkConfig,
    forward_inputs: RuntimeForwardInputs,
    scale: float,
    device: torch.device,
) -> RuntimeForwardPhase:
    from art.megatron.dsv4 import (
        Dsv4CompressionKind,
        Dsv4ProjectedAttentionForwardResult,
        launch_dsv4_compressed_kv_forward,
        launch_dsv4_indexer_topk_from_stage_plans,
    )

    plan = context_state.dsv4_plan
    layout = plan.csa_layout
    if layout is None:
        raise RuntimeError("runtime CSA benchmark requires a prepared CSA layout")
    required = (
        forward_inputs.main_projected_kv,
        forward_inputs.main_projected_gate,
        forward_inputs.main_positional_bias,
        forward_inputs.indexer_projected_kv,
        forward_inputs.indexer_projected_gate,
        forward_inputs.indexer_positional_bias,
        forward_inputs.indexer_q,
        forward_inputs.indexer_weights,
    )
    if any(item is None for item in required):
        raise RuntimeError("runtime CSA benchmark inputs are incomplete")

    fwd_start = time.perf_counter()
    with torch.no_grad():
        main_work = launch_dsv4_compressed_kv_forward(
            layout=layout,
            rank=int(context_state.cp_state.rank_plan.rank),
            projected_kv=forward_inputs.main_projected_kv,
            projected_gate=forward_inputs.main_projected_gate,
            positional_bias=forward_inputs.main_positional_bias,
            token_ids=local_token_ids,
            group=context_state.cp_state.cp_group,
            async_op=True,
        )
        indexer_work = launch_dsv4_compressed_kv_forward(
            layout=layout,
            rank=int(context_state.cp_state.rank_plan.rank),
            projected_kv=forward_inputs.indexer_projected_kv,
            projected_gate=forward_inputs.indexer_projected_gate,
            positional_bias=forward_inputs.indexer_positional_bias,
            token_ids=local_token_ids,
            group=context_state.cp_state.cp_group,
            async_op=True,
        )
    compression_launch_ms = _elapsed_ms(fwd_start)
    main_compressed = main_work.wait_post_process()
    indexer_compressed = indexer_work.wait_post_process()
    torch.cuda.synchronize(device)
    compression_ms = _elapsed_ms(fwd_start)

    attention_start = time.perf_counter()
    indexer_start = time.perf_counter()
    topk_result = launch_dsv4_indexer_topk_from_stage_plans(
        layout=layout,
        rank=int(context_state.cp_state.rank_plan.rank),
        indexer_stage_plans=plan.csa_indexer_stage_plans,
        query_token_ids=local_token_ids,
        indexer_q=forward_inputs.indexer_q,
        indexer_weights=forward_inputs.indexer_weights,
        indexer_kv=indexer_compressed.compressed_kv,
        indexer_kv_entry_ids=indexer_compressed.compressed_entry_ids,
        topk=int(config.indexer_topk),
        group=context_state.cp_state.cp_group,
        async_op=True,
        indexer_kv_peer_plans_by_stage=plan.csa_indexer_kv_peer_plans_by_stage or None,
    ).wait_post_process()
    torch.cuda.synchronize(device)
    indexer_ms = _elapsed_ms(indexer_start)
    stage_phase = _run_runtime_stage_attention_forward(
        context_state=context_state,
        layout=layout,
        compression_kind=Dsv4CompressionKind.CSA,
        query=query,
        query_token_ids=local_token_ids,
        raw_kv=raw_kv,
        raw_token_ids=local_token_ids,
        compressed_kv=main_compressed.compressed_kv,
        compressed_entry_ids=main_compressed.compressed_entry_ids,
        global_topk=topk_result.indices,
        attn_sink=attn_sink,
        config=config,
        scale=scale,
        device=device,
        stage_kv_peer_plans_by_slot=plan.csa_stage_kv_peer_plans_by_slot or None,
    )
    attention_ms = _elapsed_ms(attention_start)
    total_ms = _elapsed_ms(fwd_start)
    return RuntimeForwardPhase(
        forward=Dsv4ProjectedAttentionForwardResult(
            compression_kind=Dsv4CompressionKind.CSA,
            attention=stage_phase.attention,
            main_compressed=main_compressed,
            indexer_compressed=indexer_compressed,
        ),
        launch_ms=compression_launch_ms + stage_phase.launch_ms,
        wait_ms=max(total_ms - compression_launch_ms - stage_phase.launch_ms, 0.0),
        total_ms=total_ms,
        compression_ms=compression_ms,
        attention_ms=attention_ms,
        indexer_ms=indexer_ms,
        stage_materialization_ms=stage_phase.materialization_ms,
        sparse_merge_ms=stage_phase.sparse_merge_ms,
    )


def _run_runtime_hca_forward(
    *,
    context_state: Any,
    query: torch.Tensor,
    raw_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    local_token_ids: tuple[int, ...],
    config: RuntimeBenchmarkConfig,
    forward_inputs: RuntimeForwardInputs,
    scale: float,
    device: torch.device,
) -> RuntimeForwardPhase:
    from art.megatron.dsv4 import (
        Dsv4CompressionKind,
        Dsv4ProjectedAttentionForwardResult,
        launch_dsv4_compressed_kv_forward,
    )

    plan = context_state.dsv4_plan
    layout = plan.hca_layout
    if layout is None:
        raise RuntimeError("runtime HCA benchmark requires a prepared HCA layout")
    if (
        forward_inputs.projected_kv is None
        or forward_inputs.projected_gate is None
        or forward_inputs.positional_bias is None
    ):
        raise RuntimeError("runtime HCA benchmark inputs are incomplete")

    fwd_start = time.perf_counter()
    with torch.no_grad():
        compressed_work = launch_dsv4_compressed_kv_forward(
            layout=layout,
            rank=int(context_state.cp_state.rank_plan.rank),
            projected_kv=forward_inputs.projected_kv,
            projected_gate=forward_inputs.projected_gate,
            positional_bias=forward_inputs.positional_bias,
            token_ids=local_token_ids,
            group=context_state.cp_state.cp_group,
            async_op=True,
        )
    compression_launch_ms = _elapsed_ms(fwd_start)
    compressed = compressed_work.wait_post_process()
    torch.cuda.synchronize(device)
    compression_ms = _elapsed_ms(fwd_start)

    attention_start = time.perf_counter()
    stage_phase = _run_runtime_stage_attention_forward(
        context_state=context_state,
        layout=layout,
        compression_kind=Dsv4CompressionKind.HCA,
        query=query,
        query_token_ids=local_token_ids,
        raw_kv=raw_kv,
        raw_token_ids=local_token_ids,
        compressed_kv=compressed.compressed_kv,
        compressed_entry_ids=compressed.compressed_entry_ids,
        global_topk=None,
        attn_sink=attn_sink,
        config=config,
        scale=scale,
        device=device,
        stage_kv_peer_plans_by_slot=plan.hca_stage_kv_peer_plans_by_slot or None,
    )
    attention_ms = _elapsed_ms(attention_start)
    total_ms = _elapsed_ms(fwd_start)
    return RuntimeForwardPhase(
        forward=Dsv4ProjectedAttentionForwardResult(
            compression_kind=Dsv4CompressionKind.HCA,
            attention=stage_phase.attention,
            main_compressed=compressed,
        ),
        launch_ms=compression_launch_ms + stage_phase.launch_ms,
        wait_ms=max(total_ms - compression_launch_ms - stage_phase.launch_ms, 0.0),
        total_ms=total_ms,
        compression_ms=compression_ms,
        attention_ms=attention_ms,
        indexer_ms=0.0,
        stage_materialization_ms=stage_phase.materialization_ms,
        sparse_merge_ms=stage_phase.sparse_merge_ms,
    )


def _run_runtime_stage_attention_forward(
    *,
    context_state: Any,
    layout: Any,
    compression_kind: Any,
    query: torch.Tensor,
    query_token_ids: tuple[int, ...],
    raw_kv: torch.Tensor,
    raw_token_ids: tuple[int, ...],
    compressed_kv: torch.Tensor,
    compressed_entry_ids: tuple[int, ...],
    global_topk: torch.Tensor | None,
    attn_sink: torch.Tensor,
    config: RuntimeBenchmarkConfig,
    scale: float,
    device: torch.device,
    stage_kv_peer_plans_by_slot: tuple[tuple[Any, ...], ...] | None,
) -> RuntimeStageAttentionPhase:
    from art.megatron.dsv4 import (
        Dsv4CompressionKind,
        build_dsv4_stage_inputs_from_stage_plan,
        launch_dsv4_stage_kv_exchange_from_stage_plan_slot,
        run_materialized_dsv4_attention_forward,
    )

    rank = int(context_state.cp_state.rank_plan.rank)
    slots = tuple(context_state.dsv4_plan.stage_plan_slots)
    if not slots:
        raise RuntimeError("runtime stage benchmark requires prepared StagePlan slots")
    stage_start = time.perf_counter()
    stage_works = []
    for stage_position, slot in enumerate(slots):
        stage_inputs = build_dsv4_stage_inputs_from_stage_plan(
            layout=layout,
            stage_plan=slot.stage_plans_by_rank[rank],
            compression_kind=compression_kind,
            global_topk=global_topk,
            topk_query_token_ids=query_token_ids
            if compression_kind == Dsv4CompressionKind.CSA
            else None,
            window_size=int(config.window_size),
            raw_list_size=int(config.window_size),
            compressed_list_size=int(config.indexer_topk)
            if compression_kind == Dsv4CompressionKind.CSA
            else None,
            materialize_compressed_metadata=compression_kind != Dsv4CompressionKind.CSA,
        )
        stage_works.append(
            launch_dsv4_stage_kv_exchange_from_stage_plan_slot(
                layout=layout,
                rank=rank,
                stage_plan_slot=slot,
                local_stage_inputs=stage_inputs,
                query=query,
                query_token_ids=query_token_ids,
                raw_kv=raw_kv,
                raw_token_ids=raw_token_ids,
                compressed_kv=compressed_kv,
                compressed_entry_ids=compressed_entry_ids,
                group=context_state.cp_state.cp_group,
                async_op=True,
                peer_plans=stage_kv_peer_plans_by_slot[stage_position]
                if stage_kv_peer_plans_by_slot is not None
                else None,
            )
        )
    launch_ms = _elapsed_ms(stage_start)
    materialization_start = time.perf_counter()
    stages = tuple(work.wait_post_process() for work in stage_works)
    torch.cuda.synchronize(device)
    materialization_ms = _elapsed_ms(materialization_start)
    sparse_merge_start = time.perf_counter()
    attention = run_materialized_dsv4_attention_forward(
        stages=stages,
        query_token_ids=query_token_ids,
        attn_sink=attn_sink,
        scale=scale,
    )
    torch.cuda.synchronize(device)
    sparse_merge_ms = _elapsed_ms(sparse_merge_start)
    return RuntimeStageAttentionPhase(
        attention=attention,
        total_ms=_elapsed_ms(stage_start),
        launch_ms=launch_ms,
        materialization_ms=materialization_ms,
        sparse_merge_ms=sparse_merge_ms,
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
    parser.add_argument("--benchmark", action="store_true")
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
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--hca-only", action="store_true")
    parser.add_argument(
        "--runtime-kind",
        choices=("both", "csa", "hca"),
        default="both",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--head-count", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--indexer-dim", type=int, default=128)
    parser.add_argument("--indexer-topk", type=int, default=1024)
    parser.add_argument(
        "--miles-path",
        default=os.environ.get(
            "DSV4_MILES_PATH", "/mnt/ws_pvc/ws/scratch/miles_inspect"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    args = parser.parse_args(argv)
    mode_count = sum(
        bool(value)
        for value in (
            args.dry_run_cases,
            args.benchmark,
            args.benchmark_planning,
        )
    )
    if mode_count > 1:
        parser.error(
            "--dry-run-cases, --benchmark, and --benchmark-planning are mutually exclusive"
        )
    if mode_count == 0:
        args.dry_run_cases = True
    if args.topology is None:
        args.topology = ["cp2"]
    return args


def _parse_topology(value: str) -> int:
    normalized = value.lower()
    if not normalized.startswith("cp"):
        raise RuntimeError(f"topology must look like cp2/cp4/cp8, got {value}")
    cp_size = int(normalized[2:])
    if cp_size < 1:
        raise RuntimeError(f"topology must be positive, got {value}")
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


def _require_cuda_for_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "DSV4 runtime benchmark requires CUDA and real Miles kernels"
        )


def _add_miles_path_to_sys_path(miles_path: str) -> None:
    if miles_path and Path(miles_path).exists() and miles_path not in sys.path:
        sys.path.insert(0, miles_path)


def _load_runtime_timings(
    *,
    result_dir: Path,
    case_name: str,
    topology: str,
    kinds: tuple[str, ...],
    world_size: int,
) -> tuple[RuntimeTiming, ...]:
    records_by_kind: dict[str, list[RankRuntimeTiming]] = {kind: [] for kind in kinds}
    cp_size = int(topology[2:])
    for rank in range(int(world_size)):
        path = result_dir / f"{_safe_name(case_name)}_cp{cp_size}_rank{rank}.json"
        output = RankRuntimeOutput.model_validate_json(path.read_text())
        for timing in output.timings:
            records_by_kind[timing.compression_kind].append(timing)
    timings: list[RuntimeTiming] = []
    for kind, records in records_by_kind.items():
        if len(records) != int(world_size):
            raise RuntimeError(
                f"runtime benchmark expected {world_size} rank records for {kind}, "
                f"got {len(records)}"
            )
        timings.append(
            RuntimeTiming(
                case_name=case_name,
                topology=topology,
                compression_kind=kind,
                ranks=tuple(sorted(records, key=lambda item: item.rank)),
            )
        )
    return tuple(timings)


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


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise RuntimeError(f"unsupported DSV4 runtime dtype {name}")


def _runtime_kinds(value: str) -> tuple[str, ...]:
    if value == "both":
        return ("csa", "hca")
    if value in {"csa", "hca"}:
        return (value,)
    raise RuntimeError(f"unsupported DSV4 runtime kind {value}")


def _iteration_rank_max(
    ranks: tuple[RankRuntimeTiming, ...],
    field_name: str,
) -> tuple[float, ...]:
    if not ranks:
        raise RuntimeError("runtime metric aggregation requires rank timings")
    iteration_count = min(
        len(cast(tuple[float, ...], getattr(rank, field_name))) for rank in ranks
    )
    if iteration_count == 0:
        raise RuntimeError(f"runtime metric {field_name} has no timed iterations")
    return tuple(
        max(
            cast(tuple[float, ...], getattr(rank, field_name))[iteration]
            for rank in ranks
        )
        for iteration in range(iteration_count)
    )


def _nonfinite_count(tensor: torch.Tensor) -> int:
    return int(torch.logical_not(torch.isfinite(tensor)).sum().item())


def _abs_sum(tensor: torch.Tensor) -> float:
    return float(tensor.detach().abs().sum().item())


def _safe_name(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"_", "-"} else "_" for char in value
    )


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
