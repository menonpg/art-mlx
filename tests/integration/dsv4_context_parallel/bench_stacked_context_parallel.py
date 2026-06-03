from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, cast

from artifacts import Dsv4Metric, write_manifest, write_readable_summary
from bench_dsv4_layer_lab import (
    RuntimeBenchmarkConfig,
    RuntimeForwardInputs,
    _abs_sum,
    _add_miles_path_to_sys_path,
    _build_runtime_forward_inputs,
    _dtype_from_name,
    _elapsed_ms,
    _fresh_micro,
    _local_token_ids_from_context_state,
    _median,
    _nonfinite_count,
    _parse_topology,
    _percentile,
    _planned_runtime_explicit_comm_bytes,
    _require_cuda_for_runtime,
    _require_megatron,
    _require_single_packed_row,
    _safe_name,
    _select_cases,
)
from cases import Dsv4WorkloadCase
from packed_layout import summarize_case
from pydantic import BaseModel, ConfigDict, Field
import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.multiprocessing as mp


class StackedBenchmarkConfig(BaseModel):
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
    layer_kinds: tuple[str, ...]
    miles_path: str


class StackedLayerInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    kind: str
    query: torch.Tensor
    raw_kv: torch.Tensor
    attn_sink: torch.Tensor
    grad_out: torch.Tensor
    forward_inputs: RuntimeForwardInputs


class RankStackedTiming(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_name: str
    topology: str
    rank: int
    world_size: int
    iterations: int
    warmup: int
    dtype: str
    layer_kinds: tuple[str, ...]
    token_count: int
    local_token_count: int
    completion_count: int
    normal_cp_plan_ms: float
    dsv4_plan_ms: float
    forward_launch_ms: tuple[float, ...]
    forward_compression_wait_ms: tuple[float, ...]
    forward_attention_wait_ms: tuple[float, ...]
    forward_work_wait_ms: tuple[float, ...]
    forward_total_ms: tuple[float, ...]
    backward_launch_ms: tuple[float, ...]
    backward_owner_wait_ms: tuple[float, ...]
    backward_total_ms: tuple[float, ...]
    e2e_ms: tuple[float, ...]
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    planned_forward_comm_send_bytes: int
    planned_forward_comm_recv_bytes: int
    planned_backward_comm_send_bytes: int
    planned_backward_comm_recv_bytes: int
    planned_explicit_comm_send_bytes: int
    planned_explicit_comm_recv_bytes: int
    output_abs_sum: float
    dq_abs_sum: float
    compressor_grad_abs_sum: float
    nonfinite_count: int


class StackedTiming(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_name: str
    topology: str
    ranks: tuple[RankStackedTiming, ...]

    def metrics(self) -> tuple[Dsv4Metric, ...]:
        prefix = f"{self.case_name}_{self.topology}_stacked"
        forward_total = _iteration_rank_max(self.ranks, "forward_total_ms")
        forward_compression_wait = _iteration_rank_max(
            self.ranks,
            "forward_compression_wait_ms",
        )
        forward_attention_wait = _iteration_rank_max(
            self.ranks,
            "forward_attention_wait_ms",
        )
        forward_wait = _iteration_rank_max(
            self.ranks,
            "forward_work_wait_ms",
        )
        backward_total = _iteration_rank_max(self.ranks, "backward_total_ms")
        backward_wait = _iteration_rank_max(
            self.ranks,
            "backward_owner_wait_ms",
        )
        e2e = _iteration_rank_max(self.ranks, "e2e_ms")
        token_count = max((rank.token_count for rank in self.ranks), default=0)
        completion_count = max(
            (rank.completion_count for rank in self.ranks),
            default=0,
        )
        layer_count = max((len(rank.layer_kinds) for rank in self.ranks), default=0)
        median_e2e = _median(e2e)
        seconds = max(median_e2e / 1000.0, 1e-12)
        total_plan_max = max(
            rank.normal_cp_plan_ms + rank.dsv4_plan_ms for rank in self.ranks
        )
        nonfinite = sum(rank.nonfinite_count for rank in self.ranks)
        output_abs = sum(rank.output_abs_sum for rank in self.ranks)
        dq_abs = sum(rank.dq_abs_sum for rank in self.ranks)
        compressor_grad_abs = sum(rank.compressor_grad_abs_sum for rank in self.ranks)
        forward_send = sum(rank.planned_forward_comm_send_bytes for rank in self.ranks)
        forward_recv = sum(rank.planned_forward_comm_recv_bytes for rank in self.ranks)
        backward_send = sum(
            rank.planned_backward_comm_send_bytes for rank in self.ranks
        )
        backward_recv = sum(
            rank.planned_backward_comm_recv_bytes for rank in self.ranks
        )
        explicit_send = sum(
            rank.planned_explicit_comm_send_bytes for rank in self.ranks
        )
        explicit_recv = sum(
            rank.planned_explicit_comm_recv_bytes for rank in self.ranks
        )
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
                name=f"{prefix}_total_plan_max",
                value=total_plan_max,
                unit="ms",
                threshold=120.0,
                passed=total_plan_max < 120.0,
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
                name=f"{prefix}_forward_compression_wait_median",
                value=_median(forward_compression_wait),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_forward_compression_wait_p90",
                value=_percentile(forward_compression_wait, 90),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_forward_attention_wait_median",
                value=_median(forward_attention_wait),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_forward_attention_wait_p90",
                value=_percentile(forward_attention_wait, 90),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_forward_work_wait_median",
                value=_median(forward_wait),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_forward_work_wait_p90",
                value=_percentile(forward_wait, 90),
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
                name=f"{prefix}_backward_owner_wait_median",
                value=_median(backward_wait),
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_backward_owner_wait_p90",
                value=_percentile(backward_wait, 90),
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
                name=f"{prefix}_plan_plus_e2e_median",
                value=median_e2e + total_plan_max,
                unit="ms",
            ),
            Dsv4Metric(
                name=f"{prefix}_tokens_per_second",
                value=float(token_count) / seconds,
                unit="tok/s",
            ),
            Dsv4Metric(
                name=f"{prefix}_layer_tokens_per_second",
                value=float(token_count * layer_count) / seconds,
                unit="layer_tok/s",
            ),
            Dsv4Metric(
                name=f"{prefix}_examples_per_second",
                value=float(completion_count) / seconds,
                unit="examples/s",
            ),
            Dsv4Metric(
                name=f"{prefix}_layer_count",
                value=float(layer_count),
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
                name=f"{prefix}_planned_forward_comm_send_bytes_total",
                value=float(forward_send),
                unit="bytes",
            ),
            Dsv4Metric(
                name=f"{prefix}_planned_forward_comm_recv_bytes_total",
                value=float(forward_recv),
                unit="bytes",
            ),
            Dsv4Metric(
                name=f"{prefix}_planned_backward_comm_send_bytes_total",
                value=float(backward_send),
                unit="bytes",
            ),
            Dsv4Metric(
                name=f"{prefix}_planned_backward_comm_recv_bytes_total",
                value=float(backward_recv),
                unit="bytes",
            ),
            Dsv4Metric(
                name=f"{prefix}_planned_explicit_comm_send_bytes_total",
                value=float(explicit_send),
                unit="bytes",
            ),
            Dsv4Metric(
                name=f"{prefix}_planned_explicit_comm_recv_bytes_total",
                value=float(explicit_recv),
                unit="bytes",
            ),
            Dsv4Metric(
                name=f"{prefix}_planned_explicit_comm_send_bytes_rank_max",
                value=float(
                    max(rank.planned_explicit_comm_send_bytes for rank in self.ranks)
                ),
                unit="bytes",
            ),
            Dsv4Metric(
                name=f"{prefix}_planned_explicit_comm_recv_bytes_rank_max",
                value=float(
                    max(rank.planned_explicit_comm_recv_bytes for rank in self.ranks)
                ),
                unit="bytes",
            ),
            Dsv4Metric(
                name=f"{prefix}_nonfinite_count",
                value=float(nonfinite),
                threshold=0.0,
                passed=nonfinite == 0,
            ),
            Dsv4Metric(
                name=f"{prefix}_output_abs_sum",
                value=float(output_abs),
                threshold=0.0,
                passed=output_abs > 0.0,
            ),
            Dsv4Metric(
                name=f"{prefix}_dq_abs_sum",
                value=float(dq_abs),
                threshold=0.0,
                passed=dq_abs > 0.0,
            ),
            Dsv4Metric(
                name=f"{prefix}_compressor_grad_abs_sum",
                value=float(compressor_grad_abs),
                threshold=0.0,
                passed=compressor_grad_abs > 0.0,
            ),
        )


class RankStackedOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    timing: RankStackedTiming


class StackedBenchmarkResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    manifest_path: Path
    summary_path: Path
    timings: tuple[StackedTiming, ...] = Field(default_factory=tuple)


StackedBenchmarkConfig.model_rebuild()
StackedLayerInputs.model_rebuild()
RankStackedTiming.model_rebuild()
StackedTiming.model_rebuild()
RankStackedOutput.model_rebuild()
StackedBenchmarkResult.model_rebuild()


def run_stacked_dry_run_cases(
    *,
    output_dir: Path,
    cases: tuple[Dsv4WorkloadCase, ...],
    config: StackedBenchmarkConfig,
    command: list[str],
) -> StackedBenchmarkResult:
    case_summaries = tuple(summarize_case(case) for case in cases)
    manifest_path = write_manifest(
        output_dir,
        kind="dsv4_stacked_context_parallel_dry_run",
        command=command,
        configs={"mode": "dry_run_cases", **config.model_dump()},
        cases=case_summaries,
        caveats=("dry-run only; no CP execution or timing",),
    )
    summary_path = write_readable_summary(
        output_dir,
        title="DSV4 Stacked Context Parallel Dry Run",
        status="pass",
        manifest_path=manifest_path,
        case_summaries=case_summaries,
        caveats=("dry-run only; no CP execution or timing",),
    )
    return StackedBenchmarkResult(
        status="pass",
        manifest_path=manifest_path,
        summary_path=summary_path,
    )


def run_stacked_benchmark(
    *,
    output_dir: Path,
    cases: tuple[Dsv4WorkloadCase, ...],
    topologies: tuple[int, ...],
    config: StackedBenchmarkConfig,
    command: list[str],
) -> StackedBenchmarkResult:
    _require_megatron()
    _require_cuda_for_runtime()
    _add_miles_path_to_sys_path(config.miles_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    timings: list[StackedTiming] = []
    for case in cases:
        _require_single_packed_row(case)
        for cp_size in topologies:
            if cp_size > torch.cuda.device_count():
                raise RuntimeError(
                    f"stacked benchmark requested cp{cp_size}, but only "
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
                _stacked_worker,
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
            timings.append(
                _load_stacked_timing(
                    result_dir=result_dir,
                    case_name=case.name,
                    topology=f"cp{cp_size}",
                    world_size=cp_size,
                )
            )
    metrics = tuple(metric for timing in timings for metric in timing.metrics())
    passed = all(metric.passed is not False for metric in metrics)
    caveats = (
        "projected-input stacked benchmark; surrounding DSV4 model projections, RMSNorm, RoPE, inverse RoPE, grouped output projection, and full model-handler integration are excluded until the non-CP DSV4 handler exists",
        "uses real prepared ART CP state, host-ahead DSV4 plan metadata, public projected DSV4 context-state forward APIs, public replay backward APIs, and real Miles TileLang sparse kernels",
        "stacked layers are independent projected attention-core invocations over the same packed row; this measures repeated DSV4 CP layer cost but is not a full model trainability claim",
        "main e2e timing excludes host planning and reports plan_plus_e2e separately so overlapped planning and exposed planning cost stay visible",
        "planned communication bytes are per-iteration explicit row-exchange bytes summed across layers and exclude collective algorithm internals for sink and positional-bias all-reduces",
        "forward_compression_wait is a direct host timing around the public projected-compression halo wait boundary; projected inputs are prebuilt, so this benchmark reports the overlap boundary but does not simulate surrounding q/raw-KV projection compute",
        "forward_attention_wait is a direct host timing around the bound projected-forward work wait boundary after compression halo wait; forward_work_wait is their sum",
        "backward_owner_wait is a direct host timing around the public attention owner/sink-gradient wait boundary; total e2e uses one synchronization after each stacked forward/backward pass",
        "warmup excludes first-use TileLang compilation and setup where the iteration count is large enough to amortize it",
    )
    case_summaries = tuple(summarize_case(case) for case in cases)
    manifest_path = write_manifest(
        output_dir,
        kind="dsv4_stacked_context_parallel_benchmark",
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
        title="DSV4 Stacked Context Parallel Benchmark",
        status="pass" if passed else "fail",
        manifest_path=manifest_path,
        case_summaries=case_summaries,
        metrics=metrics,
        caveats=caveats,
    )
    return StackedBenchmarkResult(
        status="pass" if passed else "fail",
        manifest_path=manifest_path,
        summary_path=summary_path,
        timings=tuple(timings),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    layer_kinds = _layer_kinds(
        exact=tuple(args.layer_kind or ()),
        pattern=args.layer_pattern,
        count=args.layer_count,
    )
    config = StackedBenchmarkConfig(
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
        layer_kinds=layer_kinds,
        miles_path=args.miles_path,
    )
    cases = _select_cases(case_set=args.case_set, names=tuple(args.case or ()))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("tests/integration/dsv4_context_parallel/scratch/stacked")
        / time.strftime("%Y%m%d_%H%M%S")
    )
    command = [Path(sys.argv[0]).name, *(sys.argv[1:] if argv is None else argv)]
    if args.benchmark:
        result = run_stacked_benchmark(
            output_dir=output_dir,
            cases=cases,
            topologies=tuple(_parse_topology(value) for value in args.topology),
            config=config,
            command=command,
        )
    else:
        result = run_stacked_dry_run_cases(
            output_dir=output_dir,
            cases=cases,
            config=config,
            command=command,
        )
    print(f"summary: {result.summary_path}")
    print(f"manifest: {result.manifest_path}")
    return 0 if result.status == "pass" else 1


def _stacked_worker(
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
    config = StackedBenchmarkConfig.model_validate_json(config_json)
    _add_miles_path_to_sys_path(config.miles_path)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29673")
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
            include_csa="csa" in config.layer_kinds,
            include_hca="hca" in config.layer_kinds,
        )
        dsv4_plan_ms = _elapsed_ms(dsv4_start)
        local_token_ids = _local_token_ids_from_context_state(context_state)
        timing = _time_stacked_runtime(
            context_state=context_state,
            case=case,
            config=config,
            rank=int(rank),
            world_size=int(world_size),
            device=device,
            local_token_ids=local_token_ids,
            normal_cp_plan_ms=normal_cp_plan_ms,
            dsv4_plan_ms=dsv4_plan_ms,
        )
        path = (
            Path(result_dir) / f"{_safe_name(case.name)}_cp{world_size}_rank{rank}.json"
        )
        path.write_text(
            RankStackedOutput(timing=timing).model_dump_json(indent=2) + "\n"
        )
    finally:
        destroy_process_group()


def _time_stacked_runtime(
    *,
    context_state: Any,
    case: Dsv4WorkloadCase,
    config: StackedBenchmarkConfig,
    rank: int,
    world_size: int,
    device: torch.device,
    local_token_ids: tuple[int, ...],
    normal_cp_plan_ms: float,
    dsv4_plan_ms: float,
) -> RankStackedTiming:
    from art.megatron.dsv4 import (
        launch_dsv4_projected_attention_backward_from_context_parallel_state,
    )

    dtype = _dtype_from_name(config.dtype)
    token_count = int(context_state.cp_state.rank_plan.original_seq_len)
    local_count = len(local_token_ids)
    completion_count = summarize_case(case).completion_count
    scale = 1.0 / (int(config.head_dim) ** 0.5)
    torch.manual_seed(30_000 + rank * 97 + len(config.layer_kinds))
    layers = tuple(
        _build_layer_inputs(
            kind=kind,
            layer_index=layer_index,
            local_token_count=local_count,
            config=config,
            device=device,
            dtype=dtype,
        )
        for layer_index, kind in enumerate(config.layer_kinds)
    )
    planned_comm_by_kind = {
        kind: _planned_runtime_explicit_comm_bytes(
            context_state=context_state,
            compression_kind=kind,
            config=_runtime_config(config),
            dtype=dtype,
        )
        for kind in set(config.layer_kinds)
    }
    forward_launch_ms: list[float] = []
    forward_compression_wait_ms: list[float] = []
    forward_attention_wait_ms: list[float] = []
    forward_work_wait_ms: list[float] = []
    forward_total_ms: list[float] = []
    backward_launch_ms: list[float] = []
    backward_owner_wait_ms: list[float] = []
    backward_total_ms: list[float] = []
    e2e_ms: list[float] = []
    output_abs_sum = 0.0
    dq_abs_sum = 0.0
    compressor_grad_abs_sum = 0.0
    nonfinite_count = 0
    if int(config.warmup) == 0:
        torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(int(config.warmup) + int(config.iterations)):
        if iteration == int(config.warmup):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        e2e_start = time.perf_counter()
        fwd_start = time.perf_counter()
        forward_results: list[tuple[Any, torch.Tensor]] = []
        iter_forward_launch = 0.0
        iter_forward_compression_wait = 0.0
        iter_forward_attention_wait = 0.0
        iter_forward_wait = 0.0
        for layer in layers:
            layer_launch_start = time.perf_counter()
            compression_work = _launch_layer_compression(
                context_state=context_state,
                layer=layer,
                token_ids=local_token_ids,
                config=config,
            )
            iter_forward_launch += _elapsed_ms(layer_launch_start)
            layer_compression_wait_start = time.perf_counter()
            compression_work.wait()
            compression_wait = _elapsed_ms(layer_compression_wait_start)
            iter_forward_compression_wait += compression_wait
            layer_bind_start = time.perf_counter()
            forward_work = _bind_layer_forward(
                context_state=context_state,
                layer=layer,
                compression_work=compression_work,
                token_ids=local_token_ids,
                config=config,
                scale=scale,
            )
            iter_forward_launch += _elapsed_ms(layer_bind_start)
            layer_attention_wait_start = time.perf_counter()
            forward_work.wait()
            attention_wait = _elapsed_ms(layer_attention_wait_start)
            iter_forward_attention_wait += attention_wait
            iter_forward_wait += compression_wait + attention_wait
            forward_results.append((forward_work.wait_post_process(), layer.grad_out))
        torch.cuda.synchronize(device)
        fwd_total = _elapsed_ms(fwd_start)
        bwd_start = time.perf_counter()
        iter_backward_launch = 0.0
        iter_backward_wait = 0.0
        for forward_result, grad_out in reversed(forward_results):
            layer_bwd_start = time.perf_counter()
            backward_work = (
                launch_dsv4_projected_attention_backward_from_context_parallel_state(
                    context_state=context_state,
                    forward_result=forward_result,
                    grad_out=grad_out,
                    async_op=True,
                )
            )
            iter_backward_launch += _elapsed_ms(layer_bwd_start)
            layer_bwd_wait_start = time.perf_counter()
            backward_work.attention_work.wait()
            iter_backward_wait += _elapsed_ms(layer_bwd_wait_start)
            backward = backward_work.wait_post_process()
            if iteration >= int(config.warmup):
                nonfinite_count += _nonfinite_count(forward_result.attention.out)
                nonfinite_count += _nonfinite_count(forward_result.attention.lse)
                nonfinite_count += _nonfinite_count(backward.attention.dq)
                nonfinite_count += _nonfinite_count(
                    backward.main_compressor.dprojected_kv
                )
                output_abs_sum += _abs_sum(forward_result.attention.out)
                dq_abs_sum += _abs_sum(backward.attention.dq)
                compressor_grad_abs_sum += _abs_sum(
                    backward.main_compressor.dprojected_kv
                )
        torch.cuda.synchronize(device)
        bwd_total = _elapsed_ms(bwd_start)
        total = _elapsed_ms(e2e_start)
        if iteration >= int(config.warmup):
            forward_launch_ms.append(iter_forward_launch)
            forward_compression_wait_ms.append(iter_forward_compression_wait)
            forward_attention_wait_ms.append(iter_forward_attention_wait)
            forward_work_wait_ms.append(iter_forward_wait)
            forward_total_ms.append(fwd_total)
            backward_launch_ms.append(iter_backward_launch)
            backward_owner_wait_ms.append(iter_backward_wait)
            backward_total_ms.append(bwd_total)
            e2e_ms.append(total)
    planned_forward_send = sum(
        planned_comm_by_kind[layer.kind].forward_send_bytes for layer in layers
    )
    planned_forward_recv = sum(
        planned_comm_by_kind[layer.kind].forward_recv_bytes for layer in layers
    )
    planned_backward_send = sum(
        planned_comm_by_kind[layer.kind].backward_send_bytes for layer in layers
    )
    planned_backward_recv = sum(
        planned_comm_by_kind[layer.kind].backward_recv_bytes for layer in layers
    )
    return RankStackedTiming(
        case_name=case.name,
        topology=f"cp{world_size}",
        rank=int(rank),
        world_size=int(world_size),
        iterations=int(config.iterations),
        warmup=int(config.warmup),
        dtype=config.dtype,
        layer_kinds=config.layer_kinds,
        token_count=token_count,
        local_token_count=local_count,
        completion_count=completion_count,
        normal_cp_plan_ms=float(normal_cp_plan_ms),
        dsv4_plan_ms=float(dsv4_plan_ms),
        forward_launch_ms=tuple(forward_launch_ms),
        forward_compression_wait_ms=tuple(forward_compression_wait_ms),
        forward_attention_wait_ms=tuple(forward_attention_wait_ms),
        forward_work_wait_ms=tuple(forward_work_wait_ms),
        forward_total_ms=tuple(forward_total_ms),
        backward_launch_ms=tuple(backward_launch_ms),
        backward_owner_wait_ms=tuple(backward_owner_wait_ms),
        backward_total_ms=tuple(backward_total_ms),
        e2e_ms=tuple(e2e_ms),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        planned_forward_comm_send_bytes=int(planned_forward_send),
        planned_forward_comm_recv_bytes=int(planned_forward_recv),
        planned_backward_comm_send_bytes=int(planned_backward_send),
        planned_backward_comm_recv_bytes=int(planned_backward_recv),
        planned_explicit_comm_send_bytes=int(
            planned_forward_send + planned_backward_send
        ),
        planned_explicit_comm_recv_bytes=int(
            planned_forward_recv + planned_backward_recv
        ),
        output_abs_sum=float(output_abs_sum),
        dq_abs_sum=float(dq_abs_sum),
        compressor_grad_abs_sum=float(compressor_grad_abs_sum),
        nonfinite_count=int(nonfinite_count),
    )


def _build_layer_inputs(
    *,
    kind: str,
    layer_index: int,
    local_token_count: int,
    config: StackedBenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> StackedLayerInputs:
    torch.manual_seed(40_000 + int(layer_index) * 271 + (0 if kind == "csa" else 1))
    runtime_config = _runtime_config(config)
    return StackedLayerInputs(
        kind=kind,
        query=torch.randn(
            int(local_token_count),
            int(config.head_count),
            int(config.head_dim),
            device=device,
            dtype=dtype,
        ),
        raw_kv=torch.randn(
            int(local_token_count),
            int(config.head_dim),
            device=device,
            dtype=dtype,
        ),
        attn_sink=(
            torch.randn(
                int(config.head_count),
                device=device,
                dtype=torch.float32,
            )
            * 0.1
        ),
        grad_out=torch.randn(
            1,
            int(local_token_count),
            int(config.head_count),
            int(config.head_dim),
            device=device,
            dtype=dtype,
        ),
        forward_inputs=_build_runtime_forward_inputs(
            compression_kind=kind,
            local_token_count=int(local_token_count),
            config=runtime_config,
            device=device,
            dtype=dtype,
        ),
    )


def _launch_layer_compression(
    *,
    context_state: Any,
    layer: StackedLayerInputs,
    token_ids: tuple[int, ...],
    config: StackedBenchmarkConfig,
) -> Any:
    from art.megatron.dsv4 import (
        launch_dsv4_csa_projected_compression_forward_from_context_parallel_state,
        launch_dsv4_hca_projected_compression_forward_from_context_parallel_state,
    )

    inputs = layer.forward_inputs
    if layer.kind == "csa":
        if (
            inputs.main_projected_kv is None
            or inputs.main_projected_gate is None
            or inputs.main_positional_bias is None
            or inputs.indexer_projected_kv is None
            or inputs.indexer_projected_gate is None
            or inputs.indexer_positional_bias is None
        ):
            raise RuntimeError("stacked CSA layer inputs are incomplete")
        return (
            launch_dsv4_csa_projected_compression_forward_from_context_parallel_state(
                context_state=context_state,
                main_projected_kv=inputs.main_projected_kv,
                main_projected_gate=inputs.main_projected_gate,
                main_positional_bias=inputs.main_positional_bias,
                main_token_ids=token_ids,
                indexer_projected_kv=inputs.indexer_projected_kv,
                indexer_projected_gate=inputs.indexer_projected_gate,
                indexer_positional_bias=inputs.indexer_positional_bias,
                indexer_token_ids=token_ids,
                async_op=True,
            )
        )
    if layer.kind == "hca":
        if (
            inputs.projected_kv is None
            or inputs.projected_gate is None
            or inputs.positional_bias is None
        ):
            raise RuntimeError("stacked HCA layer inputs are incomplete")
        return (
            launch_dsv4_hca_projected_compression_forward_from_context_parallel_state(
                context_state=context_state,
                projected_kv=inputs.projected_kv,
                projected_gate=inputs.projected_gate,
                positional_bias=inputs.positional_bias,
                token_ids=token_ids,
                async_op=True,
            )
        )
    raise RuntimeError(f"unsupported stacked layer kind {layer.kind}")


def _bind_layer_forward(
    *,
    context_state: Any,
    layer: StackedLayerInputs,
    compression_work: Any,
    token_ids: tuple[int, ...],
    config: StackedBenchmarkConfig,
    scale: float,
) -> Any:
    from art.megatron.dsv4 import (
        launch_dsv4_csa_projected_attention_forward_from_context_parallel_state_and_compression_work,
        launch_dsv4_hca_projected_attention_forward_from_context_parallel_state_and_compression_work,
    )

    inputs = layer.forward_inputs
    if layer.kind == "csa":
        if inputs.indexer_q is None or inputs.indexer_weights is None:
            raise RuntimeError("stacked CSA layer inputs are incomplete")
        return launch_dsv4_csa_projected_attention_forward_from_context_parallel_state_and_compression_work(
            context_state=context_state,
            compression_work=compression_work,
            query=layer.query,
            query_token_ids=token_ids,
            raw_kv=layer.raw_kv,
            raw_token_ids=token_ids,
            indexer_q=inputs.indexer_q,
            indexer_weights=inputs.indexer_weights,
            indexer_topk=int(config.indexer_topk),
            attn_sink=layer.attn_sink,
            async_op=True,
            scale=scale,
            window_size=int(config.window_size),
            raw_list_size=int(config.window_size),
            compressed_list_size=int(config.indexer_topk),
        )
    if layer.kind == "hca":
        return launch_dsv4_hca_projected_attention_forward_from_context_parallel_state_and_compression_work(
            context_state=context_state,
            compression_work=compression_work,
            query=layer.query,
            query_token_ids=token_ids,
            raw_kv=layer.raw_kv,
            raw_token_ids=token_ids,
            attn_sink=layer.attn_sink,
            async_op=True,
            scale=scale,
            window_size=int(config.window_size),
            raw_list_size=int(config.window_size),
        )
    raise RuntimeError(f"unsupported stacked layer kind {layer.kind}")


def _runtime_config(config: StackedBenchmarkConfig) -> RuntimeBenchmarkConfig:
    return RuntimeBenchmarkConfig(
        iterations=config.iterations,
        warmup=config.warmup,
        dtype=config.dtype,
        head_count=config.head_count,
        head_dim=config.head_dim,
        indexer_dim=config.indexer_dim,
        indexer_topk=config.indexer_topk,
        block_size=config.block_size,
        planner_chunk_size=config.planner_chunk_size,
        csa_ratio=config.csa_ratio,
        hca_ratio=config.hca_ratio,
        window_size=config.window_size,
        kinds=tuple(sorted(set(config.layer_kinds))),
        miles_path=config.miles_path,
    )


def _load_stacked_timing(
    *,
    result_dir: Path,
    case_name: str,
    topology: str,
    world_size: int,
) -> StackedTiming:
    cp_size = int(topology[2:])
    ranks: list[RankStackedTiming] = []
    for rank in range(int(world_size)):
        path = result_dir / f"{_safe_name(case_name)}_cp{cp_size}_rank{rank}.json"
        ranks.append(RankStackedOutput.model_validate_json(path.read_text()).timing)
    if len(ranks) != int(world_size):
        raise RuntimeError(
            f"stacked benchmark expected {world_size} rank records, got {len(ranks)}"
        )
    return StackedTiming(
        case_name=case_name,
        topology=topology,
        ranks=tuple(sorted(ranks, key=lambda item: item.rank)),
    )


def _iteration_rank_max(
    ranks: tuple[RankStackedTiming, ...],
    field_name: str,
) -> tuple[float, ...]:
    if not ranks:
        raise RuntimeError("stacked metric aggregation requires rank timings")
    iteration_count = min(
        len(cast(tuple[float, ...], getattr(rank, field_name))) for rank in ranks
    )
    if iteration_count == 0:
        raise RuntimeError(f"stacked metric {field_name} has no timed iterations")
    return tuple(
        max(
            cast(tuple[float, ...], getattr(rank, field_name))[iteration]
            for rank in ranks
        )
        for iteration in range(iteration_count)
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DSV4 stacked CP benchmark")
    parser.add_argument("--dry-run-cases", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--case-set",
        choices=("validation", "canonical", "all"),
        default="canonical",
    )
    parser.add_argument("--case", action="append")
    parser.add_argument("--topology", action="append")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--planner-chunk-size", type=int, default=512)
    parser.add_argument("--csa-ratio", type=int, default=4)
    parser.add_argument("--hca-ratio", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--head-count", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--indexer-dim", type=int, default=128)
    parser.add_argument("--indexer-topk", type=int, default=1024)
    parser.add_argument("--layer-count", type=int, default=2)
    parser.add_argument(
        "--layer-pattern",
        choices=("alternating", "csa", "hca"),
        default="alternating",
    )
    parser.add_argument("--layer-kind", choices=("csa", "hca"), action="append")
    parser.add_argument(
        "--miles-path",
        default=os.environ.get(
            "DSV4_MILES_PATH",
            "/mnt/ws_pvc/ws/scratch/miles_inspect",
        ),
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    if args.dry_run_cases and args.benchmark:
        parser.error("--dry-run-cases and --benchmark are mutually exclusive")
    if not args.dry_run_cases and not args.benchmark:
        args.dry_run_cases = True
    if args.topology is None:
        args.topology = ["cp2"]
    return args


def _layer_kinds(
    *,
    exact: tuple[str, ...],
    pattern: str,
    count: int,
) -> tuple[str, ...]:
    if exact:
        return exact
    if int(count) < 1:
        raise RuntimeError(f"layer-count must be positive, got {count}")
    if pattern == "csa":
        return tuple("csa" for _ in range(int(count)))
    if pattern == "hca":
        return tuple("hca" for _ in range(int(count)))
    if pattern == "alternating":
        return tuple("csa" if index % 2 == 0 else "hca" for index in range(int(count)))
    raise RuntimeError(f"unsupported layer pattern {pattern}")


if __name__ == "__main__":
    raise SystemExit(main())
