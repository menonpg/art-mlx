from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.multiprocessing as mp

from art.megatron.gdn.gdn_shared_prefix import (
    GdnPlannerConfig,
    build_gdn_rank_execution_plan,
    move_gdn_rank_execution_plan_to_device,
    parse_gdn_shared_prefix_segments,
)
from art.megatron.gdn.operator import gdn_nvtx_ranges, run_gdn_layer

from .artifacts import write_manifest
from .bench_single_gdn_operation import (
    _GDN_NVTX_PREFIXES_WITH_AUTOGRAD,
    TimingSummary,
    _backward_with_optional_autograd_nvtx,
    _nvtx_range,
    _selected_or_repeated_case,
    _summary,
)
from .benchmark_gdn import (
    QWEN35_GDN_LINEAR_POLICY,
    make_qwen35_gdn_pair,
    qwen35_gdn_module_config,
)
from .distributed_grad import all_reduce_parameter_grads_coalesced
from .nsys_profile_tables import export_nsys_sqlite, parse_nsys_sqlite
from .packed_layout import build_gdn_group_parent_tensors
from .real_gdn_oracle import zero_parameter_grads

BENCHMARK_DTYPE = torch.bfloat16

_CP_PACKED_REQUIRED_NVTX_RANGES = (
    "art_gdn_lab_forward",
    "art_gdn_lab_loss",
    "art_gdn_lab_backward",
    "art_gdn_in_proj",
    "art_gdn_causal_conv_forward",
    "art_gdn_output_norm_gate",
    "art_gdn_out_proj",
)


class RankPackedCpTiming(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=0)
    attention_tokens: int = Field(ge=0)
    gdn_tokens: int = Field(ge=0)
    plan_ms: TimingSummary
    plan_raw_ms: tuple[float, ...]
    fwd_ms: TimingSummary
    bwd_ms: TimingSummary
    e2e_ms: TimingSummary
    e2e_with_param_reduce_ms: TimingSummary
    local_prefix_bucket_count: int = Field(ge=0)
    local_completion_bucket_count: int = Field(ge=0)
    chain_prefix_bucket_count: int = Field(ge=0)
    chain_completion_bucket_count: int = Field(ge=0)
    parent_state_exchange_family_count: int = Field(ge=0)


class PackedCpGdnBenchmark(BaseModel):
    model_config = ConfigDict(frozen=True)

    cp_size: int = Field(ge=1)
    dtype: str
    gdn_linear_policy: str
    hidden_size: int = Field(ge=1)
    case_name: str
    sequence_length: int = Field(ge=1)
    real_tokens: int = Field(ge=1)
    family_count: int = Field(ge=1)
    completion_count: int = Field(ge=1)
    plan_ms: TimingSummary
    max_rank_fwd_ms: float
    max_rank_bwd_ms: float
    max_rank_e2e_ms: float
    max_rank_e2e_with_param_reduce_ms: float
    max_local_prefix_bucket_count: int = Field(ge=0)
    max_local_completion_bucket_count: int = Field(ge=0)
    max_chain_prefix_bucket_count: int = Field(ge=0)
    max_chain_completion_bucket_count: int = Field(ge=0)
    max_parent_state_exchange_family_count: int = Field(ge=0)
    tokens_per_second: float
    tokens_per_second_with_param_reduce: float
    ranks: tuple[RankPackedCpTiming, ...]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark native packed CP GDN")
    parser.add_argument("--cp-sizes", default="2,4")
    parser.add_argument("--case-name", default="sampled_repeated_family")
    parser.add_argument("--conv-width", type=int, default=4)
    parser.add_argument("--target-seq-len", type=int, default=40960)
    parser.add_argument("--prefix-len", type=int, default=5000)
    parser.add_argument("--suffix-len", type=int, default=100)
    parser.add_argument("--completions-per-family", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prefix-length-std", type=int, default=0)
    parser.add_argument("--prefix-length-clip-delta", type=int, default=0)
    parser.add_argument("--branch-length-std", type=int, default=0)
    parser.add_argument("--branch-length-clip-delta", type=int, default=0)
    parser.add_argument("--background-prefix-len", type=int, default=512)
    parser.add_argument("--background-suffix-len", type=int, default=64)
    parser.add_argument("--background-completions-per-family", type=int, default=4)
    parser.add_argument("--background-prefix-length-std", type=int, default=64)
    parser.add_argument("--background-prefix-length-clip-delta", type=int, default=128)
    parser.add_argument("--background-branch-length-std", type=int, default=16)
    parser.add_argument("--background-branch-length-clip-delta", type=int, default=32)
    parser.add_argument(
        "--gdn-linear-policy",
        choices=QWEN35_GDN_LINEAR_POLICY,
        default="noop",
    )
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--nsys-profile", action="store_true")
    parser.add_argument("--top-kernels", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.nsys_profile:
        return _run_nsys_profile(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for cp_size in tuple(int(value) for value in args.cp_sizes.split(",") if value):
        run_args = _args_for_cp_size(args, cp_size)
        run_dir = args.output_dir / f"cp{cp_size}"
        run_dir.mkdir(parents=True, exist_ok=True)
        port = _find_free_port()
        mp.spawn(
            _worker,
            args=(cp_size, port, run_args, str(run_dir)),
            nprocs=cp_size,
            join=True,
        )
        results.append(
            PackedCpGdnBenchmark.model_validate_json(
                (run_dir / "result_rank0.json").read_text()
            )
        )
        print(results[-1].model_dump_json(), flush=True)
    (args.output_dir / "result.json").write_text(
        json.dumps([result.model_dump() for result in results], indent=2) + "\n"
    )
    (args.output_dir / "benchmark_report.md").write_text(
        _render_report(tuple(results)),
        encoding="utf-8",
    )
    manifest_path = write_manifest(
        args.output_dir,
        kind="gdn_cp_packed_layer_benchmark",
        command=sys.argv,
        configs=_manifest_configs(args),
        cases=tuple(result.model_dump() for result in results),
    )
    print(json.dumps({"manifest": str(manifest_path)}), flush=True)
    return 0


def _worker(
    rank: int, cp_size: int, port: int, args: argparse.Namespace, run_dir: str
) -> None:
    from megatron.core import parallel_state as ps

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=cp_size,
    )
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
            expert_model_parallel_size=1,
        )
        config = qwen35_gdn_module_config().model_copy(
            update={"linear_conv_kernel_dim": args.conv_width}
        )
        _, gdn = make_qwen35_gdn_pair(
            params_dtype=BENCHMARK_DTYPE,
            linear_policy=args.gdn_linear_policy,
            config=config,
        )
        if rank == 0:
            case = _selected_or_repeated_case(args)
            tensors = build_gdn_group_parent_tensors(case)
            group_ids_cpu = tensors["group_ids"]
            parent_ids_cpu = tensors["parent_ids"]
        else:
            case = None
            group_ids_cpu = torch.empty((0, 0), dtype=torch.long)
            parent_ids_cpu = torch.empty((0, 0), dtype=torch.long)
        if cp_size == 1:
            group_ids = group_ids_cpu.cuda()
            parent_ids = parent_ids_cpu.cuda()
        else:
            group_ids = group_ids_cpu
            parent_ids = parent_ids_cpu
        spec = _build_distributed_execution_spec(
            group_ids_cpu,
            parent_ids_cpu,
            cp_rank=rank,
        )
        plan_times = []
        plan: Any | None = None
        for _ in range(args.warmup_iters):
            plan = _build_rank_execution_plan_from_spec(
                spec,
                cp_rank=rank,
                cp_size=cp_size,
                device=torch.device("cpu"),
            )
        torch.distributed.barrier()
        for _ in range(args.iters):
            torch.distributed.barrier()
            start = time.perf_counter()
            plan = _build_rank_execution_plan_from_spec(
                spec,
                cp_rank=rank,
                cp_size=cp_size,
                device=torch.device("cpu"),
            )
            plan_times.append((time.perf_counter() - start) * 1000.0)
            torch.distributed.barrier()
        if plan is None:
            raise RuntimeError("distributed CP GDN plan was not built")
        plan = move_gdn_rank_execution_plan_to_device(
            plan, torch.device("cuda", torch.cuda.current_device())
        )
        torch.cuda.synchronize()
        hidden, output_grad = _hidden_and_grad(
            case,
            plan,
            seed=20500426 + cp_size + rank * 10_000,
            hidden_size=config.hidden_size,
        )
        local_hidden_template = hidden
        local_output_grad = output_grad
        for _ in range(args.warmup_iters):
            _timed_iteration(
                gdn,
                local_hidden_template,
                local_output_grad,
                group_ids=group_ids,
                parent_ids=parent_ids,
                spec=spec,
                plan=plan,
                profile=False,
            )
        timings = [
            _timed_iteration(
                gdn,
                local_hidden_template,
                local_output_grad,
                group_ids=group_ids,
                parent_ids=parent_ids,
                spec=spec,
                plan=plan,
                profile=bool(args.profile),
            )
            for _ in range(args.iters)
        ]
        rank_result = RankPackedCpTiming(
            rank=rank,
            attention_tokens=_rank_attention_token_count(plan, spec),
            gdn_tokens=_rank_gdn_token_count(plan, spec),
            plan_ms=_summary(plan_times),
            plan_raw_ms=tuple(plan_times),
            fwd_ms=_summary([timing["fwd_ms"] for timing in timings]),
            bwd_ms=_summary([timing["bwd_ms"] for timing in timings]),
            e2e_ms=_summary([timing["e2e_ms"] for timing in timings]),
            e2e_with_param_reduce_ms=_summary(
                [timing["e2e_with_param_reduce_ms"] for timing in timings]
            ),
            local_prefix_bucket_count=_local_prefix_bucket_count(plan),
            local_completion_bucket_count=_local_completion_bucket_count(plan),
            chain_prefix_bucket_count=len(plan.chain_prefix_buckets),
            chain_completion_bucket_count=len(plan.chain_completion_buckets),
            parent_state_exchange_family_count=len(
                plan.parent_state_exchange_family_indices
            ),
        )
        gathered: list[Any] = [None for _ in range(cp_size)]
        torch.distributed.all_gather_object(  # ty: ignore[possibly-missing-attribute]
            gathered, rank_result.model_dump()
        )
        if rank == 0:
            if case is None:
                raise RuntimeError("rank 0 must retain benchmark case metadata")
            ranks = tuple(RankPackedCpTiming.model_validate(item) for item in gathered)
            e2e = max(result.e2e_ms.median_ms for result in ranks)
            e2e_reduce = max(
                result.e2e_with_param_reduce_ms.median_ms for result in ranks
            )
            plan_times_by_iter = [
                max(result.plan_raw_ms[index] for result in ranks)
                for index in range(args.iters)
            ]
            result = PackedCpGdnBenchmark(
                cp_size=cp_size,
                dtype=str(BENCHMARK_DTYPE),
                gdn_linear_policy=str(args.gdn_linear_policy),
                hidden_size=config.hidden_size,
                case_name=case.name,
                sequence_length=case.sequence_length,
                real_tokens=spec.real_token_count,
                family_count=spec.family_count,
                completion_count=spec.completion_count,
                plan_ms=_summary(plan_times_by_iter),
                max_rank_fwd_ms=max(result.fwd_ms.median_ms for result in ranks),
                max_rank_bwd_ms=max(result.bwd_ms.median_ms for result in ranks),
                max_rank_e2e_ms=e2e,
                max_rank_e2e_with_param_reduce_ms=e2e_reduce,
                max_local_prefix_bucket_count=max(
                    result.local_prefix_bucket_count for result in ranks
                ),
                max_local_completion_bucket_count=max(
                    result.local_completion_bucket_count for result in ranks
                ),
                max_chain_prefix_bucket_count=max(
                    result.chain_prefix_bucket_count for result in ranks
                ),
                max_chain_completion_bucket_count=max(
                    result.chain_completion_bucket_count for result in ranks
                ),
                max_parent_state_exchange_family_count=max(
                    result.parent_state_exchange_family_count for result in ranks
                ),
                tokens_per_second=1000.0 * spec.real_token_count / e2e,
                tokens_per_second_with_param_reduce=(
                    1000.0 * spec.real_token_count / e2e_reduce
                ),
                ranks=ranks,
            )
            Path(run_dir, "result_rank0.json").write_text(
                result.model_dump_json(indent=2) + "\n"
            )
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def _timed_iteration(
    gdn: torch.nn.Module,
    local_hidden_template: torch.Tensor,
    local_output_grad: torch.Tensor,
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    spec: Any,
    plan: Any,
    profile: bool,
) -> dict[str, float]:
    zero_parameter_grads(gdn)
    local_hidden = local_hidden_template.clone().detach().requires_grad_(True)
    start = torch.cuda.Event(enable_timing=True)
    after_fwd = torch.cuda.Event(enable_timing=True)
    after_bwd = torch.cuda.Event(enable_timing=True)
    after_reduce = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    with gdn_nvtx_ranges(profile):
        with _nvtx_range("art_gdn_lab_forward", enabled=profile):
            output, _ = run_gdn_layer(
                gdn,
                local_hidden,
                group_ids=group_ids,
                parent_ids=parent_ids,
                execution_spec=spec,
                execution_plan=plan,
                cp_group=torch.distributed.group.WORLD,  # ty: ignore[possibly-missing-attribute]
            )
    after_fwd.record()
    with _nvtx_range("art_gdn_lab_loss", enabled=profile):
        loss = (output * local_output_grad).sum()
    _backward_with_optional_autograd_nvtx(loss, enabled=profile)
    after_bwd.record()
    all_reduce_parameter_grads_coalesced(gdn)
    after_reduce.record()
    torch.cuda.synchronize()
    return {
        "fwd_ms": float(start.elapsed_time(after_fwd)),
        "bwd_ms": float(after_fwd.elapsed_time(after_bwd)),
        "e2e_ms": float(start.elapsed_time(after_bwd)),
        "e2e_with_param_reduce_ms": float(start.elapsed_time(after_reduce)),
    }


def _rank_attention_token_count(plan: Any, spec: Any) -> int:
    if int(plan.cp_size) == 1:
        return int(spec.real_token_count)
    return int(plan.attention_token_count)


def _rank_gdn_token_count(plan: Any, spec: Any) -> int:
    if int(plan.cp_size) == 1:
        return int(spec.real_token_count)
    return int(plan.gdn_token_count)


def _local_prefix_bucket_count(plan: Any) -> int:
    return (
        len(plan.local_prefix_buckets)
        + len(plan.prefix_boundary_buckets)
        + len(plan.prefix_tail_buckets)
    )


def _local_completion_bucket_count(plan: Any) -> int:
    return len(plan.local_completion_buckets) + len(
        plan.completion_with_prefix_tail_buckets
    )


def _build_distributed_execution_spec(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    *,
    cp_rank: int,
) -> Any:
    spec_payload: list[Any] = [None]
    if cp_rank == 0:
        spec_payload[0] = parse_gdn_shared_prefix_segments(
            group_ids, parent_ids, min_completions_per_family=0
        )
    torch.distributed.broadcast_object_list(  # ty: ignore[possibly-missing-attribute]
        spec_payload,
        src=0,
        group=torch.distributed.group.WORLD,  # ty: ignore[possibly-missing-attribute]
    )
    return spec_payload[0]


def _build_rank_execution_plan_from_spec(
    spec: Any,
    *,
    cp_rank: int,
    cp_size: int,
    device: torch.device,
) -> Any:
    return build_gdn_rank_execution_plan(
        spec,
        device=device,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )


def _hidden_and_grad(
    case: Any | None, plan: Any, *, seed: int, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if int(plan.cp_size) > 1:
        shape = (int(plan.attention_token_count), 1, hidden_size)
        hidden = torch.randn(
            shape,
            device="cuda",
            dtype=BENCHMARK_DTYPE,
            generator=generator,
        )
        grad = torch.randn(
            shape,
            device="cuda",
            dtype=BENCHMARK_DTYPE,
            generator=generator,
        )
        return hidden, grad
    if case is None:
        raise ValueError("CP1 packed layer benchmark requires a full packed case")
    hidden = torch.randn(
        case.sequence_length,
        len(case.rows),
        hidden_size,
        device="cuda",
        dtype=BENCHMARK_DTYPE,
        generator=generator,
    )
    grad = torch.randn(
        hidden.shape,
        device="cuda",
        dtype=BENCHMARK_DTYPE,
        generator=generator,
    )
    return hidden, grad


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _args_for_cp_size(args: argparse.Namespace, cp_size: int) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.target_seq_len = args.target_seq_len * cp_size
    return run_args


def _manifest_configs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "cp_sizes": args.cp_sizes,
        "case_name": args.case_name,
        "conv_width": args.conv_width,
        "base_cp1_target_seq_len": args.target_seq_len,
        "cp_target_seq_len_rule": (
            "effective_target_seq_len = base_cp1_target_seq_len * cp_size; "
            "per-family prefix/completion lengths stay fixed and additional "
            "families are packed to target"
        ),
        "prefix_len": args.prefix_len,
        "suffix_len": args.suffix_len,
        "completions_per_family": args.completions_per_family,
        "seed": args.seed,
        "prefix_length_std": args.prefix_length_std,
        "prefix_length_clip_delta": args.prefix_length_clip_delta,
        "branch_length_std": args.branch_length_std,
        "branch_length_clip_delta": args.branch_length_clip_delta,
        "background_prefix_len": args.background_prefix_len,
        "background_suffix_len": args.background_suffix_len,
        "background_completions_per_family": args.background_completions_per_family,
        "background_prefix_length_std": args.background_prefix_length_std,
        "background_prefix_length_clip_delta": args.background_prefix_length_clip_delta,
        "background_branch_length_std": args.background_branch_length_std,
        "background_branch_length_clip_delta": args.background_branch_length_clip_delta,
        "gdn_linear_policy": str(args.gdn_linear_policy),
        "warmup_iters": args.warmup_iters,
        "iters": args.iters,
        "benchmark_dtype": str(BENCHMARK_DTYPE),
        "worker_torch_num_threads": 1,
        "plan_timing_scope": (
            "CPU rank execution plan from a parsed distributed spec; metadata "
            "parse/broadcast and CPU-to-CUDA plan transfer run outside timing"
        ),
        "benchmark_qwen35_gdn": qwen35_gdn_module_config()
        .model_copy(update={"linear_conv_kernel_dim": args.conv_width})
        .model_dump(),
        "profile": bool(args.profile),
        "nsys_profile": bool(args.nsys_profile),
        "planner_config": GdnPlannerConfig().model_dump(),
    }


def _render_report(results: tuple[PackedCpGdnBenchmark, ...]) -> str:
    lines = [
        "# Packed Native CP GDN Benchmark",
        "",
        "| CP | dtype | linear policy | hidden | case | real tokens | families | completions | plan median ms | fwd ms | bwd ms | e2e+reduce ms | tok/s incl reduce | local buckets | chain buckets | parent exchanges |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.cp_size} | {result.dtype} | {result.gdn_linear_policy} | "
            f"{result.hidden_size} | {result.case_name} | "
            f"{result.real_tokens} | "
            f"{result.family_count} | "
            f"{result.completion_count} | {result.plan_ms.median_ms:.3f} | "
            f"{result.max_rank_fwd_ms:.3f} | {result.max_rank_bwd_ms:.3f} | "
            f"{result.max_rank_e2e_with_param_reduce_ms:.3f} | "
            f"{result.tokens_per_second_with_param_reduce:.0f} | "
            f"{result.max_local_prefix_bucket_count}/"
            f"{result.max_local_completion_bucket_count} | "
            f"{result.max_chain_prefix_bucket_count}/"
            f"{result.max_chain_completion_bucket_count} | "
            f"{result.max_parent_state_exchange_family_count} |"
        )
    lines.extend(
        [
            "",
            "Per-rank medians use the slowest rank as the topology-level time.",
            "The target sequence length is weak-scaled by adding more fixed-shape families; the final family may use fewer completions to fit the target.",
            "Planning is measured as CPU rank-plan construction from an already parsed distributed execution spec; metadata parse/broadcast and CPU-to-CUDA plan transfer are prepared outside the timed planner loop.",
            "The parameter-reduce column uses one coalesced all-reduce bucket per dtype/device, matching production gradient-sync shape better than per-parameter test reductions.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_nsys_profile(args: argparse.Namespace) -> int:
    cp_sizes = tuple(int(value) for value in args.cp_sizes.split(",") if value)
    if len(cp_sizes) != 1:
        raise ValueError("--nsys-profile expects exactly one CP size")
    output_dir = args.output_dir
    benchmark_dir = output_dir / "benchmark"
    report_stem = output_dir / f"nsys_gdn_cp{cp_sizes[0]}_packed_profile"
    report_path = report_stem.with_suffix(".nsys-rep")
    sqlite_path = output_dir / f"nsys_gdn_cp{cp_sizes[0]}_packed_profile.sqlite"
    profile_tables_dir = output_dir / "profile_tables"
    nsys_command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--force-overwrite=true",
        "-o",
        str(report_stem),
        sys.executable,
        "-m",
        "tests.integration.megatron.gdn_shared_prefix.bench_gdn_cp_packed_layer",
        "--cp-sizes",
        args.cp_sizes,
        "--case-name",
        args.case_name,
        "--conv-width",
        str(args.conv_width),
        "--target-seq-len",
        str(args.target_seq_len),
        "--prefix-len",
        str(args.prefix_len),
        "--suffix-len",
        str(args.suffix_len),
        "--completions-per-family",
        str(args.completions_per_family),
        "--seed",
        str(args.seed),
        "--prefix-length-std",
        str(args.prefix_length_std),
        "--prefix-length-clip-delta",
        str(args.prefix_length_clip_delta),
        "--branch-length-std",
        str(args.branch_length_std),
        "--branch-length-clip-delta",
        str(args.branch_length_clip_delta),
        "--background-prefix-len",
        str(args.background_prefix_len),
        "--background-suffix-len",
        str(args.background_suffix_len),
        "--background-completions-per-family",
        str(args.background_completions_per_family),
        "--background-prefix-length-std",
        str(args.background_prefix_length_std),
        "--background-prefix-length-clip-delta",
        str(args.background_prefix_length_clip_delta),
        "--background-branch-length-std",
        str(args.background_branch_length_std),
        "--background-branch-length-clip-delta",
        str(args.background_branch_length_clip_delta),
        "--gdn-linear-policy",
        str(args.gdn_linear_policy),
        "--warmup-iters",
        str(args.warmup_iters),
        "--iters",
        str(args.iters),
        "--profile",
        "--output-dir",
        str(benchmark_dir),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nsys_command.json").write_text(
        json.dumps({"profile_command": nsys_command}, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run(nsys_command, check=True, text=True)
    export_nsys_sqlite(report_path, sqlite_path)
    tables = parse_nsys_sqlite(
        sqlite_path,
        profile_tables_dir,
        expected_ranges=_CP_PACKED_REQUIRED_NVTX_RANGES,
        nvtx_prefixes=_GDN_NVTX_PREFIXES_WITH_AUTOGRAD,
        top_kernels=args.top_kernels,
    )
    result = {
        "mode": "nsys-profile",
        "cp_size": cp_sizes[0],
        "case_name": _selected_or_repeated_case(args).name,
        "report_path": str(report_path),
        "sqlite_path": str(sqlite_path),
        "profile_tables": tables.model_dump(),
        "benchmark_dir": str(benchmark_dir),
    }
    (output_dir / "nsys_profile_result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = write_manifest(
        output_dir,
        kind="gdn_cp_packed_layer_nsys_profile",
        command=sys.argv,
        configs=_manifest_configs(args),
        cases=(result,),
    )
    print(json.dumps({"manifest": str(manifest_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
