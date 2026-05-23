from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import torch
from torch import Tensor
from torch.distributed import destroy_process_group, init_process_group, is_initialized

from art.megatron.gdn.gdn_shared_prefix import (
    GdnPackedExecutionSpec,
    GdnRankExecutionPlan,
    build_gdn_rank_execution_plan,
    parse_gdn_shared_prefix_segments,
)
from art.megatron.gdn.operator import gdn_nvtx_ranges, gdn_shared_prefix_forward

from .artifacts import write_manifest
from .benchmark_gdn import (
    QWEN35_GDN_LINEAR_POLICY,
    make_qwen35_gdn_pair,
    qwen35_gdn_module_config,
)
from .cases import (
    GdnFamilyShape,
    GdnPackedRowShape,
    GdnPhase0Case,
    default_phase0_cases,
    fit_gdn_family_to_remaining,
    gdn_family_token_count,
)
from .metrics import GDN_CORRECTNESS_DTYPE, MEAN_ABS_PCT_THRESHOLD
from .nsys_profile_tables import export_nsys_sqlite, parse_nsys_sqlite
from .packed_layout import (
    build_phase0_packed_tensors,
    format_case_summary,
    summarize_case,
)
from .real_gdn_oracle import (
    RealGdnOracleMetrics,
    attach_main_grads,
    compare_real_gdn_cp1_to_flattened,
    run_real_gdn_flattened_reference,
    zero_parameter_grads,
)

CORRECTNESS_DTYPE = GDN_CORRECTNESS_DTYPE
BENCHMARK_DTYPE = torch.bfloat16

_NVTX_RANGES = (
    "art_gdn_lab_forward",
    "art_gdn_lab_loss",
    "art_gdn_lab_backward",
    "art_gdn_plan_shared_prefix_layout",
    "art_gdn_input_layout_gather_reorder",
    "art_gdn_in_proj",
    "art_gdn_qkv_gate_beta_alpha_split_reshape",
    "art_gdn_causal_conv_forward",
    "art_gdn_qkv_head_prepare",
    "art_gdn_recurrent_gate_prepare",
    "art_gdn_recurrent_forward",
    "art_gdn_output_norm_gate",
    "art_gdn_out_proj",
    "art_gdn_scatter_back_attention_layout",
    "art_gdn_cp_layout_plan",
    "art_gdn_cp_attention_to_gdn_exchange",
    "art_gdn_cp_exchange_backward",
    "art_gdn_cp_gdn_to_attention_exchange",
    "art_gdn_cp_conv_boundary_exchange",
    "art_gdn_cp_recurrent_summary_scan",
    "art_gdn_cp_prefix_segment",
    "art_gdn_cp_completion_segment",
    "art_gdn_local_prefix_segment",
    "art_gdn_local_completion_segment",
    "art_gdn_cp_parent_state_exchange",
    "art_gdn_conv_state_materialization",
    "art_gdn_recurrent_state_materialization",
    "art_gdn_prefix_segment",
    "art_gdn_completion_segment",
    "art_gdn_state_fanout",
)

_GDN_NVTX_PREFIXES_WITH_AUTOGRAD = ("art_gdn", "autograd::", "aten::")


class TimingSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    median_ms: float
    p90_ms: float
    max_ms: float


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str
    topology: str
    case_name: str
    dtype: str
    gdn_linear_policy: str
    hidden_size: int = Field(ge=1)
    real_tokens: int = Field(ge=1)
    family_count: int = Field(ge=1)
    completion_count: int = Field(ge=1)
    warmup_iters: int = Field(ge=0)
    timed_iters: int = Field(ge=1)
    gdn_plan_ms: TimingSummary
    fwd_ms: TimingSummary
    bwd_ms: TimingSummary
    e2e_ms: TimingSummary
    tokens_per_second: float
    examples_per_second: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    layout_bytes_moved: int
    state_bytes_materialized: int
    cp_comm_bytes: int = 0
    exposed_comm_wait_ms: float = 0.0
    nvtx_ranges: tuple[str, ...] = _NVTX_RANGES


class CorrectnessResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str
    topology: str
    case_name: str
    dtype: str
    gdn_linear_policy: str
    hidden_size: int = Field(ge=1)
    real_tokens: int = Field(ge=1)
    family_count: int = Field(ge=1)
    completion_count: int = Field(ge=1)
    metrics: RealGdnOracleMetrics


class SavedTensorRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape: tuple[int, ...]
    dtype: str
    bytes: int


class MemoryDebugResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str
    topology: str
    case_name: str
    dtype: str
    gdn_linear_policy: str
    hidden_size: int = Field(ge=1)
    real_tokens: int = Field(ge=1)
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    saved_tensor_count: int
    saved_tensor_bytes: int
    top_saved_tensors: tuple[SavedTensorRecord, ...]


class NsysProfileResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str
    topology: str
    case_name: str
    nsys_report_path: str | None = None
    sqlite_path: str
    profile_json_path: str
    profile_markdown_path: str
    nvtx_csv_path: str
    kernel_by_range_csv_path: str
    top_kernels_csv_path: str
    missing_expected_ranges: tuple[str, ...]


class BaselineComparisonResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str
    topology: str
    case_name: str
    dtype: str
    gdn_linear_policy: str
    hidden_size: int = Field(ge=1)
    attention_heads: int = Field(ge=1)
    attention_head_dim: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    packed_batch_size: int = Field(ge=1)
    art_training_realistic_batch_size: bool
    real_tokens: int = Field(ge=1)
    family_count: int = Field(ge=1)
    completion_count: int = Field(ge=1)
    warmup_iters: int = Field(ge=0)
    timed_iters: int = Field(ge=1)
    gdn_plan_ms: TimingSummary
    packed_gdn_ms: TimingSummary
    flattened_gdn_ms: TimingSummary
    flex_attention_kernel_ms: TimingSummary
    flex_attention_with_mask_build_ms: TimingSummary
    gdn_plan_raw_ms: tuple[float, ...]
    packed_gdn_raw_ms: tuple[float, ...]
    flattened_gdn_raw_ms: tuple[float, ...]
    flex_attention_kernel_raw_ms: tuple[float, ...]
    flex_attention_with_mask_build_raw_ms: tuple[float, ...]
    packed_gdn_tokens_per_second: float
    flattened_gdn_tokens_per_second: float
    flex_attention_tokens_per_second: float
    flattened_gdn_slowdown_vs_packed: float
    flex_attention_slowdown_vs_packed_gdn: float
    flex_attention_projection_policy: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GDN shared-prefix single-operation lab"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run-cases", action="store_true")
    mode.add_argument("--correctness-only", action="store_true")
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--memory-debug", action="store_true")
    mode.add_argument("--nsys-profile", action="store_true")
    mode.add_argument("--parse-profile-sqlite", type=Path)
    mode.add_argument("--benchmark-baselines", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--conv-width", type=int, default=4)
    parser.add_argument("--case-name", default="ragged_family_mix")
    parser.add_argument("--cp-sizes", default="2,4,8")
    parser.add_argument(
        "--topology",
        choices=("cp1", "cp2-layout", "cp4-layout", "cp8-layout"),
        default="cp1",
    )
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--top-kernels", type=int, default=20)
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
        help=(
            "Benchmark-side GDN projection policy. Default no-ops in/out "
            "linear layers so timings isolate shared-prefix GDN recurrence, "
            "layout, planning, and setup."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    cp_layout_status = _maybe_run_cp_layout_topology(args)
    if cp_layout_status is not None:
        return cp_layout_status
    if args.dry_run_cases:
        return _dry_run_cases(args)
    if args.correctness_only:
        results = _run_correctness(args)
        return _write_lab_results(args, "gdn_single_operation_correctness", results)
    if args.benchmark:
        results = (_run_benchmark(args),)
        return _write_lab_results(args, "gdn_single_operation_benchmark", results)
    if args.memory_debug:
        results = (_run_memory_debug(args),)
        return _write_lab_results(args, "gdn_single_operation_memory_debug", results)
    if args.nsys_profile:
        results = (_run_nsys_profile(args),)
        return _write_lab_results(args, "gdn_single_operation_nsys_profile", results)
    if args.parse_profile_sqlite is not None:
        results = (_run_parse_profile_sqlite(args),)
        return _write_lab_results(args, "gdn_single_operation_nsys_parse", results)
    if args.benchmark_baselines:
        results = (_run_baseline_comparison(args),)
        return _write_lab_results(
            args, "gdn_single_operation_baseline_comparison", results
        )
    raise AssertionError("unreachable")


def _maybe_run_cp_layout_topology(args: argparse.Namespace) -> int | None:
    if args.topology == "cp1":
        return None
    if not args.benchmark:
        raise ValueError(
            f"{args.topology} is a CP layout-exchange topology; use --benchmark"
        )
    if args.output_dir is None:
        raise ValueError(f"{args.topology} benchmark requires --output-dir")
    cp_size = args.topology.removeprefix("cp").removesuffix("-layout")
    from . import bench_gdn_cp_layout_exchange

    return bench_gdn_cp_layout_exchange.main(
        [
            "--cp-sizes",
            cp_size,
            "--target-seq-len",
            str(args.target_seq_len),
            "--prefix-len",
            str(args.prefix_len),
            "--suffix-len",
            str(args.suffix_len),
            "--completions-per-family",
            str(args.completions_per_family),
            "--warmup-iters",
            str(args.warmup_iters),
            "--iters",
            str(args.iters),
            "--output-dir",
            str(args.output_dir),
        ]
    )


def _dry_run_cases(args: argparse.Namespace) -> int:
    cp_sizes = tuple(int(value) for value in args.cp_sizes.split(",") if value)
    summaries = []
    for case in default_phase0_cases(conv_width=args.conv_width):
        tensors = build_phase0_packed_tensors(case)
        summary = summarize_case(
            case, tensors, conv_width=args.conv_width, cp_sizes=cp_sizes
        )
        summaries.append(summary)
        print(format_case_summary(summary), flush=True)

    if args.output_dir is not None:
        manifest_path = write_manifest(
            args.output_dir,
            kind="gdn_shared_prefix_dry_run_cases",
            command=sys.argv,
            configs=_manifest_configs(args),
            cases=tuple(summary.model_dump() for summary in summaries),
            caveats=(
                "Phase 0 dry-run only; no GDN kernels or distributed CP executed.",
            ),
        )
        print(json.dumps({"manifest": str(manifest_path)}), flush=True)
    return 0


def _run_correctness(args: argparse.Namespace) -> tuple[CorrectnessResult, ...]:
    _require_cuda()
    cases = _selected_cases(args.case_name, conv_width=args.conv_width)
    with _single_rank_model_parallel():
        packed_gdn, flat_gdn = _make_matching_qwen35_gdn_pair(
            args.conv_width,
            params_dtype=CORRECTNESS_DTYPE,
        )
        results = []
        for case_index, case in enumerate(cases):
            zero_parameter_grads(packed_gdn)
            zero_parameter_grads(flat_gdn)
            tensors = build_phase0_packed_tensors(case)
            group_ids = tensors["group_ids"].cuda()
            parent_ids = tensors["parent_ids"].cuda()
            assistant_mask = tensors["assistant_mask"].cuda()
            hidden_states = _hidden_states(
                case,
                seed=20260425 + case_index,
                dtype=CORRECTNESS_DTYPE,
            )
            metrics = compare_real_gdn_cp1_to_flattened(
                packed_gdn=packed_gdn,
                flat_gdn=flat_gdn,
                hidden_states=hidden_states,
                group_ids=group_ids,
                parent_ids=parent_ids,
                assistant_mask=assistant_mask,
            )
            _assert_correctness_thresholds(case.name, metrics)
            counts = _case_counts(tensors)
            result = CorrectnessResult(
                mode="correctness-only",
                topology=args.topology,
                case_name=case.name,
                dtype=str(hidden_states.dtype),
                gdn_linear_policy="real",
                hidden_size=int(hidden_states.shape[-1]),
                real_tokens=counts["real_tokens"],
                family_count=counts["family_count"],
                completion_count=counts["completion_count"],
                metrics=metrics,
            )
            results.append(result)
            print(result.model_dump_json(), flush=True)
        return tuple(results)


def _run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    _require_cuda()
    case = _selected_or_repeated_case(args)
    tensors = build_phase0_packed_tensors(case)
    with _single_rank_model_parallel():
        config = qwen35_gdn_module_config().model_copy(
            update={"linear_conv_kernel_dim": args.conv_width}
        )
        packed_gdn, _ = make_qwen35_gdn_pair(
            params_dtype=BENCHMARK_DTYPE,
            linear_policy=args.gdn_linear_policy,
            config=config,
        )
        group_ids = tensors["group_ids"].cuda()
        parent_ids = tensors["parent_ids"].cuda()
        assistant_mask = tensors["assistant_mask"].cuda()
        hidden_template = _hidden_states(
            case,
            seed=20270425,
            dtype=BENCHMARK_DTYPE,
            hidden_size=config.hidden_size,
        )
        _measure_gdn_plan_iterations(
            group_ids=group_ids,
            parent_ids=parent_ids,
            iters=args.warmup_iters,
            profile=False,
        )
        gdn_plan_times = _measure_gdn_plan_iterations(
            group_ids=group_ids,
            parent_ids=parent_ids,
            iters=args.iters,
            profile=args.profile,
        )
        execution_spec, execution_plan = _build_gdn_execution_plan(
            group_ids, parent_ids
        )
        _run_timed_iterations(
            packed_gdn=packed_gdn,
            hidden_template=hidden_template,
            group_ids=group_ids,
            parent_ids=parent_ids,
            execution_spec=execution_spec,
            execution_plan=execution_plan,
            assistant_mask=assistant_mask,
            iters=args.warmup_iters,
            profile=False,
        )
        torch.cuda.reset_peak_memory_stats()
        timings = _run_timed_iterations(
            packed_gdn=packed_gdn,
            hidden_template=hidden_template,
            group_ids=group_ids,
            parent_ids=parent_ids,
            execution_spec=execution_spec,
            execution_plan=execution_plan,
            assistant_mask=assistant_mask,
            iters=args.iters,
            profile=args.profile,
        )

        e2e_summary = _summary(timings["e2e_ms"])
        counts = _case_counts(tensors)
        tokens_per_second = 1000.0 * counts["real_tokens"] / e2e_summary.median_ms
        result = BenchmarkResult(
            mode="benchmark",
            topology=args.topology,
            case_name=case.name,
            dtype=str(hidden_template.dtype),
            gdn_linear_policy=str(args.gdn_linear_policy),
            hidden_size=config.hidden_size,
            real_tokens=counts["real_tokens"],
            family_count=counts["family_count"],
            completion_count=counts["completion_count"],
            warmup_iters=args.warmup_iters,
            timed_iters=args.iters,
            gdn_plan_ms=_summary(gdn_plan_times),
            fwd_ms=_summary(timings["fwd_ms"]),
            bwd_ms=_summary(timings["bwd_ms"]),
            e2e_ms=e2e_summary,
            tokens_per_second=tokens_per_second,
            examples_per_second=1000.0 / e2e_summary.median_ms,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            layout_bytes_moved=_layout_bytes_moved(hidden_template, tensors),
            state_bytes_materialized=_state_bytes_materialized(packed_gdn, tensors),
        )
        print(result.model_dump_json(), flush=True)
        return result


def _run_memory_debug(args: argparse.Namespace) -> MemoryDebugResult:
    _require_cuda()
    case = _selected_cases(args.case_name, conv_width=args.conv_width)[0]
    tensors = build_phase0_packed_tensors(case)
    saved: list[SavedTensorRecord] = []

    def pack(tensor: Tensor) -> Tensor:
        saved.append(
            SavedTensorRecord(
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=str(tensor.dtype),
                bytes=int(tensor.numel() * tensor.element_size()),
            )
        )
        return tensor

    def unpack(tensor: Tensor) -> Tensor:
        return tensor

    with _single_rank_model_parallel():
        config = qwen35_gdn_module_config().model_copy(
            update={"linear_conv_kernel_dim": args.conv_width}
        )
        packed_gdn, _ = make_qwen35_gdn_pair(
            params_dtype=BENCHMARK_DTYPE,
            linear_policy=args.gdn_linear_policy,
            config=config,
        )
        group_ids = tensors["group_ids"].cuda()
        parent_ids = tensors["parent_ids"].cuda()
        assistant_mask = tensors["assistant_mask"].cuda()
        hidden_template = _hidden_states(
            case,
            seed=20280425,
            dtype=BENCHMARK_DTYPE,
            hidden_size=config.hidden_size,
        )
        execution_spec, execution_plan = _build_gdn_execution_plan(
            group_ids, parent_ids
        )
        torch.cuda.reset_peak_memory_stats()
        with torch.autograd.graph.saved_tensors_hooks(
            _dynamo_disabled(pack), _dynamo_disabled(unpack)
        ):
            _one_iteration(
                packed_gdn=packed_gdn,
                hidden_template=hidden_template,
                group_ids=group_ids,
                parent_ids=parent_ids,
                execution_spec=execution_spec,
                execution_plan=execution_plan,
                assistant_mask=assistant_mask,
                profile=args.profile,
            )

        top_saved = tuple(
            sorted(saved, key=lambda record: record.bytes, reverse=True)[:12]
        )
        result = MemoryDebugResult(
            mode="memory-debug",
            topology=args.topology,
            case_name=case.name,
            dtype=str(hidden_template.dtype),
            gdn_linear_policy=str(args.gdn_linear_policy),
            hidden_size=config.hidden_size,
            real_tokens=_case_counts(tensors)["real_tokens"],
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            saved_tensor_count=len(saved),
            saved_tensor_bytes=sum(record.bytes for record in saved),
            top_saved_tensors=top_saved,
        )
        print(result.model_dump_json(), flush=True)
        return result


def _run_nsys_profile(args: argparse.Namespace) -> NsysProfileResult:
    output_dir = _require_output_dir(args)
    benchmark_dir = output_dir / "benchmark"
    report_stem = output_dir / "nsys_gdn_profile"
    report_path = report_stem.with_suffix(".nsys-rep")
    sqlite_path = output_dir / "nsys_gdn_profile.sqlite"
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
        "tests.integration.megatron.gdn_shared_prefix.bench_single_gdn_operation",
        "--benchmark",
        "--profile",
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
        "--topology",
        args.topology,
        "--warmup-iters",
        str(args.warmup_iters),
        "--iters",
        str(args.iters),
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
        expected_ranges=_NVTX_RANGES,
        nvtx_prefixes=_GDN_NVTX_PREFIXES_WITH_AUTOGRAD,
        top_kernels=args.top_kernels,
    )
    result = _nsys_result(
        mode="nsys-profile",
        topology=args.topology,
        case_name=_selected_or_repeated_case(args).name,
        report_path=report_path,
        tables=tables,
    )
    print(result.model_dump_json(), flush=True)
    return result


def _run_parse_profile_sqlite(args: argparse.Namespace) -> NsysProfileResult:
    output_dir = _require_output_dir(args)
    tables = parse_nsys_sqlite(
        args.parse_profile_sqlite,
        output_dir / "profile_tables",
        expected_ranges=_NVTX_RANGES,
        nvtx_prefixes=_GDN_NVTX_PREFIXES_WITH_AUTOGRAD,
        top_kernels=args.top_kernels,
    )
    result = _nsys_result(
        mode="parse-profile-sqlite",
        topology=args.topology,
        case_name=args.case_name,
        report_path=None,
        tables=tables,
    )
    print(result.model_dump_json(), flush=True)
    return result


def _run_baseline_comparison(args: argparse.Namespace) -> BaselineComparisonResult:
    _require_cuda()
    case = _selected_or_repeated_case(args)
    tensors = build_phase0_packed_tensors(case)
    counts = _case_counts(tensors)
    with _single_rank_model_parallel():
        config = qwen35_gdn_module_config().model_copy(
            update={"linear_conv_kernel_dim": args.conv_width}
        )
        packed_gdn, flat_gdn = make_qwen35_gdn_pair(
            params_dtype=BENCHMARK_DTYPE,
            linear_policy=args.gdn_linear_policy,
            config=config,
        )
        group_ids = tensors["group_ids"].cuda()
        parent_ids = tensors["parent_ids"].cuda()
        assistant_mask = tensors["assistant_mask"].cuda()
        hidden_template = _hidden_states(
            case,
            seed=20290425,
            dtype=BENCHMARK_DTYPE,
            hidden_size=config.hidden_size,
        )
        _measure_gdn_plan_iterations(
            group_ids=group_ids,
            parent_ids=parent_ids,
            iters=args.warmup_iters,
            profile=False,
        )
        gdn_plan_times = _measure_gdn_plan_iterations(
            group_ids=group_ids,
            parent_ids=parent_ids,
            iters=args.iters,
            profile=False,
        )
        execution_spec, execution_plan = _build_gdn_execution_plan(
            group_ids, parent_ids
        )
        _run_packed_gdn_baseline_iterations(
            packed_gdn=packed_gdn,
            hidden_template=hidden_template,
            group_ids=group_ids,
            parent_ids=parent_ids,
            execution_spec=execution_spec,
            execution_plan=execution_plan,
            assistant_mask=assistant_mask,
            iters=args.warmup_iters,
        )
        packed_gdn_times = _run_packed_gdn_baseline_iterations(
            packed_gdn=packed_gdn,
            hidden_template=hidden_template,
            group_ids=group_ids,
            parent_ids=parent_ids,
            execution_spec=execution_spec,
            execution_plan=execution_plan,
            assistant_mask=assistant_mask,
            iters=args.iters,
        )
        _run_flattened_gdn_baseline_iterations(
            flat_gdn=flat_gdn,
            hidden_template=hidden_template,
            group_ids=group_ids,
            parent_ids=parent_ids,
            execution_spec=execution_spec,
            assistant_mask=assistant_mask,
            iters=args.warmup_iters,
        )
        flattened_gdn_times = _run_flattened_gdn_baseline_iterations(
            flat_gdn=flat_gdn,
            hidden_template=hidden_template,
            group_ids=group_ids,
            parent_ids=parent_ids,
            execution_spec=execution_spec,
            assistant_mask=assistant_mask,
            iters=args.iters,
        )
        flex_attention = _make_flex_attention_inputs(
            case,
            tensors,
            dtype=BENCHMARK_DTYPE,
            heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
        )
        _run_flex_attention_baseline_iterations(
            flex_attention,
            rebuild_mask=False,
            iters=args.warmup_iters,
        )
        flex_kernel_times = _run_flex_attention_baseline_iterations(
            flex_attention,
            rebuild_mask=False,
            iters=args.iters,
        )
        _run_flex_attention_baseline_iterations(
            flex_attention,
            rebuild_mask=True,
            iters=args.warmup_iters,
        )
        flex_with_mask_times = _run_flex_attention_baseline_iterations(
            flex_attention,
            rebuild_mask=True,
            iters=args.iters,
        )

    packed_summary = _summary(packed_gdn_times)
    flattened_summary = _summary(flattened_gdn_times)
    flex_summary = _summary(flex_kernel_times)
    real_tokens = counts["real_tokens"]
    result = BaselineComparisonResult(
        mode="benchmark-baselines",
        topology=args.topology,
        case_name=case.name,
        dtype=str(hidden_template.dtype),
        gdn_linear_policy=str(args.gdn_linear_policy),
        hidden_size=config.hidden_size,
        attention_heads=config.num_attention_heads,
        attention_head_dim=config.hidden_size // config.num_attention_heads,
        sequence_length=case.sequence_length,
        packed_batch_size=len(case.rows),
        art_training_realistic_batch_size=len(case.rows) == 1,
        real_tokens=real_tokens,
        family_count=counts["family_count"],
        completion_count=counts["completion_count"],
        warmup_iters=args.warmup_iters,
        timed_iters=args.iters,
        gdn_plan_ms=_summary(gdn_plan_times),
        packed_gdn_ms=packed_summary,
        flattened_gdn_ms=flattened_summary,
        flex_attention_kernel_ms=flex_summary,
        flex_attention_with_mask_build_ms=_summary(flex_with_mask_times),
        gdn_plan_raw_ms=tuple(gdn_plan_times),
        packed_gdn_raw_ms=tuple(packed_gdn_times),
        flattened_gdn_raw_ms=tuple(flattened_gdn_times),
        flex_attention_kernel_raw_ms=tuple(flex_kernel_times),
        flex_attention_with_mask_build_raw_ms=tuple(flex_with_mask_times),
        packed_gdn_tokens_per_second=1000.0 * real_tokens / packed_summary.median_ms,
        flattened_gdn_tokens_per_second=1000.0
        * real_tokens
        / flattened_summary.median_ms,
        flex_attention_tokens_per_second=1000.0 * real_tokens / flex_summary.median_ms,
        flattened_gdn_slowdown_vs_packed=flattened_summary.median_ms
        / packed_summary.median_ms,
        flex_attention_slowdown_vs_packed_gdn=flex_summary.median_ms
        / packed_summary.median_ms,
        flex_attention_projection_policy=(
            "Canonical flex baseline times compiled ART flex_attention only: q/k/v "
            "projections, output projection, and block-mask construction are excluded. "
            "Packed and flattened GDN timings follow --gdn-linear-policy; the "
            "default no-op policy excludes GDN in_proj/out_proj while the real "
            "policy measures a full layer-style GDN path. GDN shared-prefix "
            "planning and flex mask-build timing are diagnostics only."
        ),
    )
    print(result.model_dump_json(), flush=True)
    return result


def _nsys_result(
    *,
    mode: str,
    topology: str,
    case_name: str,
    report_path: Path | None,
    tables: Any,
) -> NsysProfileResult:
    return NsysProfileResult(
        mode=mode,
        topology=topology,
        case_name=case_name,
        nsys_report_path=None if report_path is None else str(report_path),
        sqlite_path=tables.paths.sqlite_path,
        profile_json_path=tables.paths.json_path,
        profile_markdown_path=tables.paths.markdown_path,
        nvtx_csv_path=tables.paths.nvtx_csv_path,
        kernel_by_range_csv_path=tables.paths.kernel_by_range_csv_path,
        top_kernels_csv_path=tables.paths.top_kernels_csv_path,
        missing_expected_ranges=tables.missing_expected_ranges,
    )


def _write_lab_results(
    args: argparse.Namespace,
    kind: str,
    results: tuple[BaseModel, ...],
) -> int:
    if args.output_dir is None:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(
            [result.model_dump() for result in results], indent=2, sort_keys=True
        )
        + "\n"
    )
    extra_paths = _write_extra_result_artifacts(args.output_dir, results)
    manifest_path = write_manifest(
        args.output_dir,
        kind=kind,
        command=sys.argv,
        configs=_manifest_configs(args),
        cases=tuple(result.model_dump() for result in results),
        caveats=_caveats(args),
    )
    print(
        json.dumps(
            {"manifest": str(manifest_path), "result": str(result_path), **extra_paths}
        ),
        flush=True,
    )
    return 0


def _write_extra_result_artifacts(
    output_dir: Path,
    results: tuple[BaseModel, ...],
) -> dict[str, str]:
    if len(results) == 1 and isinstance(results[0], BaselineComparisonResult):
        result = results[0]
        report_path = output_dir / "baseline_report.md"
        csv_path = output_dir / "baseline_table.csv"
        report_path.write_text(_render_baseline_report(result), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "name",
                    "median_ms",
                    "tokens_per_second",
                    "slowdown_vs_packed_gdn",
                    "raw_ms",
                ),
            )
            writer.writeheader()
            for row in _baseline_rows(result):
                writer.writerow(row)
        return {
            "baseline_report": str(report_path),
            "baseline_table": str(csv_path),
        }
    return {}


def _render_baseline_report(result: BaselineComparisonResult) -> str:
    lines = [
        "# GDN Packed Baseline Comparison",
        "",
        "Definitions:",
        "",
        "- `packed_gdn` is the current CP1 shared-prefix GDN path: prefixes are computed once, completions fork from prefix state.",
        "- `flattened_gdn` is the correctness baseline: each completion runs as an independent `prefix + suffix` sequence.",
        "- `flex_attention_kernel` is the canonical flex baseline: ART's compiled flex attention on the same packed row with the already-built shared-prefix block mask, excluding q/k/v and output projections.",
        "- `gdn_plan` is the CPU shared-prefix segment plan. It is measured separately and excluded from the single-layer GDN timings.",
        "- `flex_attention_with_mask_build` is recorded in `result.json` as a diagnostic only and is intentionally excluded from this comparison table.",
        "",
        f"Workload: `{result.case_name}`",
        "",
        f"- sequence_length: `{result.sequence_length}`",
        f"- real_tokens: `{result.real_tokens}`",
        f"- packed_batch_size: `{result.packed_batch_size}`",
        f"- ART-realistic batch size: `{result.art_training_realistic_batch_size}`",
        f"- hidden_size: `{result.hidden_size}`",
        f"- GDN linear policy: `{result.gdn_linear_policy}`",
        f"- flex attention heads/head_dim: `{result.attention_heads}/{result.attention_head_dim}`",
        f"- families: `{result.family_count}`",
        f"- completions: `{result.completion_count}`",
        f"- timed_iters: `{result.timed_iters}`",
        "",
        "| name | median_ms | tokens_per_second | slowdown_vs_packed_gdn | raw_ms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in _baseline_rows(result):
        lines.append(
            "| {name} | {median_ms:.3f} | {tokens_per_second:.0f} | {slowdown_vs_packed_gdn:.3f} | {raw_ms} |".format(
                **row
            )
        )
    lines.extend(
        (
            "",
            "Planning diagnostics:",
            "",
            f"- gdn_plan median_ms: `{result.gdn_plan_ms.median_ms:.3f}`",
            f"- flex_attention_with_mask_build median_ms: `{result.flex_attention_with_mask_build_ms.median_ms:.3f}`",
            "",
            "Projection policy:",
            "",
            result.flex_attention_projection_policy,
            "",
        )
    )
    return "\n".join(lines)


def _baseline_rows(result: BaselineComparisonResult) -> tuple[dict[str, Any], ...]:
    packed_ms = result.packed_gdn_ms.median_ms
    return (
        {
            "name": "packed_gdn",
            "median_ms": packed_ms,
            "tokens_per_second": result.packed_gdn_tokens_per_second,
            "slowdown_vs_packed_gdn": 1.0,
            "raw_ms": _format_raw_ms(result.packed_gdn_raw_ms),
        },
        {
            "name": "flattened_gdn",
            "median_ms": result.flattened_gdn_ms.median_ms,
            "tokens_per_second": result.flattened_gdn_tokens_per_second,
            "slowdown_vs_packed_gdn": result.flattened_gdn_ms.median_ms / packed_ms,
            "raw_ms": _format_raw_ms(result.flattened_gdn_raw_ms),
        },
        {
            "name": "flex_attention_kernel",
            "median_ms": result.flex_attention_kernel_ms.median_ms,
            "tokens_per_second": result.flex_attention_tokens_per_second,
            "slowdown_vs_packed_gdn": result.flex_attention_kernel_ms.median_ms
            / packed_ms,
            "raw_ms": _format_raw_ms(result.flex_attention_kernel_raw_ms),
        },
    )


def _format_raw_ms(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"


def _build_gdn_execution_spec(
    group_ids: Tensor, parent_ids: Tensor
) -> GdnPackedExecutionSpec:
    return parse_gdn_shared_prefix_segments(
        group_ids, parent_ids, min_completions_per_family=0
    )


def _build_gdn_execution_plan(
    group_ids: Tensor, parent_ids: Tensor
) -> tuple[GdnPackedExecutionSpec, GdnRankExecutionPlan]:
    spec = _build_gdn_execution_spec(group_ids, parent_ids)
    return spec, build_gdn_rank_execution_plan(spec, device=group_ids.device)


def _measure_gdn_plan_iterations(
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
    iters: int,
    profile: bool,
) -> list[float]:
    timings = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with _nvtx_range("art_gdn_plan_shared_prefix_layout", enabled=profile):
            _build_gdn_execution_plan(group_ids, parent_ids)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def _run_timed_iterations(
    *,
    packed_gdn: torch.nn.Module,
    hidden_template: Tensor,
    group_ids: Tensor,
    parent_ids: Tensor,
    execution_spec: GdnPackedExecutionSpec,
    execution_plan: GdnRankExecutionPlan,
    assistant_mask: Tensor,
    iters: int,
    profile: bool,
) -> dict[str, list[float]]:
    timings: dict[str, list[float]] = {"fwd_ms": [], "bwd_ms": [], "e2e_ms": []}
    for _ in range(iters):
        fwd_ms, bwd_ms, e2e_ms = _one_iteration(
            packed_gdn=packed_gdn,
            hidden_template=hidden_template,
            group_ids=group_ids,
            parent_ids=parent_ids,
            execution_spec=execution_spec,
            execution_plan=execution_plan,
            assistant_mask=assistant_mask,
            profile=profile,
        )
        timings["fwd_ms"].append(fwd_ms)
        timings["bwd_ms"].append(bwd_ms)
        timings["e2e_ms"].append(e2e_ms)
    return timings


def _run_packed_gdn_baseline_iterations(
    *,
    packed_gdn: torch.nn.Module,
    hidden_template: Tensor,
    group_ids: Tensor,
    parent_ids: Tensor,
    execution_spec: GdnPackedExecutionSpec,
    execution_plan: GdnRankExecutionPlan,
    assistant_mask: Tensor,
    iters: int,
) -> list[float]:
    timings = []
    for _ in range(iters):
        zero_parameter_grads(packed_gdn)
        hidden_states = hidden_template.clone().detach().requires_grad_(True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        output, _ = gdn_shared_prefix_forward(
            packed_gdn,
            hidden_states,
            group_ids=group_ids,
            parent_ids=parent_ids,
            execution_spec=execution_spec,
            execution_plan=execution_plan,
        )
        _masked_quadratic_loss(output, assistant_mask).backward()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return timings


def _run_flattened_gdn_baseline_iterations(
    *,
    flat_gdn: torch.nn.Module,
    hidden_template: Tensor,
    group_ids: Tensor,
    parent_ids: Tensor,
    execution_spec: GdnPackedExecutionSpec,
    assistant_mask: Tensor,
    iters: int,
) -> list[float]:
    timings = []
    for _ in range(iters):
        zero_parameter_grads(flat_gdn)
        hidden_states = hidden_template.clone().detach().requires_grad_(True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        output = run_real_gdn_flattened_reference(
            flat_gdn,
            hidden_states,
            group_ids=group_ids,
            parent_ids=parent_ids,
            execution_spec=execution_spec,
        )
        _masked_quadratic_loss(output, assistant_mask).backward()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return timings


class FlexAttentionInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    q_template: Tensor
    k_template: Tensor
    v_template: Tensor
    group_ids: Tensor
    parent_ids: Tensor
    assistant_mask: Tensor
    scale: float


def _make_flex_attention_inputs(
    case: GdnPhase0Case,
    tensors: dict[str, Any],
    *,
    dtype: torch.dtype,
    heads: int,
    head_dim: int,
) -> FlexAttentionInputs:
    generator = torch.Generator(device="cuda").manual_seed(20300425)
    shape = (len(case.rows), heads, case.sequence_length, head_dim)
    return FlexAttentionInputs(
        q_template=torch.randn(shape, device="cuda", dtype=dtype, generator=generator),
        k_template=torch.randn(shape, device="cuda", dtype=dtype, generator=generator),
        v_template=torch.randn(shape, device="cuda", dtype=dtype, generator=generator),
        group_ids=tensors["group_ids"].cuda(),
        parent_ids=tensors["parent_ids"].cuda(),
        assistant_mask=tensors["assistant_mask"].cuda(),
        scale=1.0 / (head_dim**0.5),
    )


def _run_flex_attention_baseline_iterations(
    inputs: FlexAttentionInputs,
    *,
    rebuild_mask: bool,
    iters: int,
) -> list[float]:
    from art.megatron.flex_attn.attention import FlexAttentionWrapper
    from art.megatron.shared_prefix_state import create_shared_prefix_state

    wrapper = FlexAttentionWrapper().cuda()
    attention_state = create_shared_prefix_state(
        group_ids=inputs.group_ids,
        parent_ids=inputs.parent_ids,
        build_gdn_execution_spec=False,
    )
    timings = []
    for _ in range(iters):
        q = inputs.q_template.clone().detach().requires_grad_(True)
        k = inputs.k_template.clone().detach().requires_grad_(True)
        v = inputs.v_template.clone().detach().requires_grad_(True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        state = (
            create_shared_prefix_state(
                group_ids=inputs.group_ids,
                parent_ids=inputs.parent_ids,
                build_gdn_execution_spec=False,
            )
            if rebuild_mask
            else attention_state
        )
        output = wrapper(
            q,
            k,
            v,
            block_mask=state.block_mask,
            scale=inputs.scale,
            enable_gqa=False,
        )
        _masked_quadratic_loss(
            output.permute(2, 0, 1, 3).reshape(output.shape[2], output.shape[0], -1),
            inputs.assistant_mask,
        ).backward()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return timings


def _one_iteration(
    *,
    packed_gdn: torch.nn.Module,
    hidden_template: Tensor,
    group_ids: Tensor,
    parent_ids: Tensor,
    execution_spec: GdnPackedExecutionSpec,
    execution_plan: GdnRankExecutionPlan,
    assistant_mask: Tensor,
    profile: bool,
) -> tuple[float, float, float]:
    zero_parameter_grads(packed_gdn)
    hidden_states = hidden_template.clone().detach().requires_grad_(True)
    start = torch.cuda.Event(enable_timing=True)
    after_fwd = torch.cuda.Event(enable_timing=True)
    after_bwd = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    with gdn_nvtx_ranges(profile):
        with _nvtx_range("art_gdn_lab_forward", enabled=profile):
            output, _ = gdn_shared_prefix_forward(
                packed_gdn,
                hidden_states,
                group_ids=group_ids,
                parent_ids=parent_ids,
                execution_spec=execution_spec,
                execution_plan=execution_plan,
            )
    after_fwd.record()
    with _nvtx_range("art_gdn_lab_loss", enabled=profile):
        loss = _masked_quadratic_loss(output, assistant_mask)
    _backward_with_optional_autograd_nvtx(loss, enabled=profile)
    after_bwd.record()
    torch.cuda.synchronize()
    return (
        float(start.elapsed_time(after_fwd)),
        float(after_fwd.elapsed_time(after_bwd)),
        float(start.elapsed_time(after_bwd)),
    )


def _backward_with_optional_autograd_nvtx(loss: Tensor, *, enabled: bool) -> None:
    if not enabled:
        loss.backward()
        return
    with _nvtx_range("art_gdn_lab_backward", enabled=True):
        with torch.autograd.profiler.emit_nvtx(record_shapes=False):
            loss.backward()


def _make_matching_qwen35_gdn_pair(
    conv_width: int,
    *,
    params_dtype: torch.dtype = CORRECTNESS_DTYPE,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(1234)
    packed_model = _make_qwen35_language_model(
        conv_width,
        params_dtype=params_dtype,
    )
    model_parallel_cuda_manual_seed(5678)
    flat_model = _make_qwen35_language_model(
        conv_width,
        params_dtype=params_dtype,
    )
    packed_gdn = _first_gdn(packed_model)
    flat_gdn = _first_gdn(flat_model)
    flat_gdn.load_state_dict(packed_gdn.state_dict())
    attach_main_grads(packed_gdn)
    attach_main_grads(flat_gdn)
    return packed_gdn, flat_gdn


def _make_qwen35_language_model(
    conv_width: int,
    *,
    params_dtype: torch.dtype = CORRECTNESS_DTYPE,
) -> torch.nn.Module:
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
        Qwen3_5MoeVisionConfig,
        Qwen35VLMoEModelProvider,
    )

    assert Qwen3_5MoeVisionConfig is not None
    provider = Qwen35VLMoEModelProvider(
        num_layers=4,
        hidden_size=64,
        ffn_hidden_size=128,
        moe_ffn_hidden_size=32,
        moe_shared_expert_intermediate_size=16,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=16,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_output_gate=True,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=4,
        linear_conv_kernel_dim=conv_width,
        vocab_size=128,
        seq_length=128,
        position_embedding_type="mrope",
        vision_config=Qwen3_5MoeVisionConfig(),
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        params_dtype=params_dtype,
    )
    provider.finalize()
    return provider.provide_language_model(pre_process=True, post_process=True).cuda()


def _first_gdn(model: torch.nn.Module) -> torch.nn.Module:
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    for module in model.modules():
        if isinstance(module, GatedDeltaNet):
            return module
    raise AssertionError("expected Qwen3.5 provider to build at least one GDN layer")


def _hidden_states(
    case: GdnPhase0Case,
    *,
    seed: int,
    dtype: torch.dtype = CORRECTNESS_DTYPE,
    hidden_size: int = 64,
) -> Tensor:
    return torch.randn(
        case.sequence_length,
        len(case.rows),
        hidden_size,
        device="cuda",
        dtype=dtype,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    )


def _selected_cases(case_name: str, *, conv_width: int) -> tuple[GdnPhase0Case, ...]:
    cases = default_phase0_cases(conv_width=conv_width)
    if case_name == "all":
        return cases
    for case in cases:
        if case.name == case_name:
            return (case,)
    names = ", ".join(case.name for case in cases)
    raise ValueError(f"unknown case {case_name!r}; expected one of: all, {names}")


def _selected_or_repeated_case(args: argparse.Namespace) -> GdnPhase0Case:
    if args.case_name == "repeated_family":
        return _repeated_family_case(
            target_seq_len=args.target_seq_len,
            prefix_len=args.prefix_len,
            suffix_len=args.suffix_len,
            completions_per_family=args.completions_per_family,
        )
    if args.case_name == "sampled_repeated_family":
        return _sampled_repeated_family_case(
            target_seq_len=args.target_seq_len,
            prefix_len=args.prefix_len,
            suffix_len=args.suffix_len,
            completions_per_family=args.completions_per_family,
            seed=args.seed,
            prefix_length_std=args.prefix_length_std,
            prefix_length_clip_delta=args.prefix_length_clip_delta,
            branch_length_std=args.branch_length_std,
            branch_length_clip_delta=args.branch_length_clip_delta,
        )
    if args.case_name == "sampled_single_family":
        return _sampled_single_family_case(
            prefix_len=args.prefix_len,
            suffix_len=args.suffix_len,
            completions_per_family=args.completions_per_family,
            seed=args.seed,
            prefix_length_std=args.prefix_length_std,
            prefix_length_clip_delta=args.prefix_length_clip_delta,
            branch_length_std=args.branch_length_std,
            branch_length_clip_delta=args.branch_length_clip_delta,
        )
    if args.case_name == "deterministic_jitter_repeated_family":
        return _deterministic_jitter_repeated_family_case(
            target_seq_len=args.target_seq_len,
            prefix_len=args.prefix_len,
            suffix_len=args.suffix_len,
            completions_per_family=args.completions_per_family,
        )
    if args.case_name == "deterministic_many_small_families":
        return _deterministic_many_small_family_case(
            target_seq_len=args.target_seq_len,
            prefix_len=args.prefix_len,
            suffix_len=args.suffix_len,
            completions_per_family=args.completions_per_family,
        )
    if args.case_name == "fixed_dominant_family_benchmark":
        return _fixed_dominant_family_benchmark_case(
            target_seq_len=args.target_seq_len,
            prefix_len=args.prefix_len,
            suffix_len=args.suffix_len,
            completions_per_family=args.completions_per_family,
        )
    if args.case_name == "sampled_dominant_family_benchmark":
        return _sampled_dominant_family_benchmark_case(
            target_seq_len=args.target_seq_len,
            prefix_len=args.prefix_len,
            suffix_len=args.suffix_len,
            completions_per_family=args.completions_per_family,
            seed=args.seed,
            prefix_length_std=args.prefix_length_std,
            prefix_length_clip_delta=args.prefix_length_clip_delta,
            branch_length_std=args.branch_length_std,
            branch_length_clip_delta=args.branch_length_clip_delta,
            background_prefix_len=args.background_prefix_len,
            background_suffix_len=args.background_suffix_len,
            background_completions_per_family=args.background_completions_per_family,
            background_prefix_length_std=args.background_prefix_length_std,
            background_prefix_length_clip_delta=args.background_prefix_length_clip_delta,
            background_branch_length_std=args.background_branch_length_std,
            background_branch_length_clip_delta=args.background_branch_length_clip_delta,
        )
    return _selected_cases(args.case_name, conv_width=args.conv_width)[0]


def _repeated_family_case(
    *,
    target_seq_len: int,
    prefix_len: int,
    suffix_len: int,
    completions_per_family: int,
) -> GdnPhase0Case:
    family = GdnFamilyShape(
        prefix_length=prefix_len,
        suffix_lengths=(suffix_len,) * completions_per_family,
    )
    if gdn_family_token_count(family) <= 0:
        raise ValueError(
            "repeated family must contain positive prefix and completion lengths"
        )
    families: list[GdnFamilyShape] = []
    used = 0
    while fitted := fit_gdn_family_to_remaining(family, target_seq_len - used):
        families.append(fitted)
        used += gdn_family_token_count(fitted)
        if len(fitted.suffix_lengths) != len(family.suffix_lengths):
            break
    if not families:
        raise ValueError(
            "target_seq_len must fit at least one repeated prefix plus completion, "
            f"got target={target_seq_len}, family_tokens={gdn_family_token_count(family)}"
        )
    return GdnPhase0Case(
        name=(
            f"repeated_{prefix_len}_plus_{completions_per_family}x"
            f"{suffix_len}_target_{target_seq_len}"
        ),
        sequence_length=target_seq_len,
        rows=(GdnPackedRowShape(families=tuple(families)),),
        seed=41,
        description=(
            "One ART-realistic packed row with complete repeated prompt families "
            "packed up to the target sequence length."
        ),
    )


def _sample_length(
    *,
    mean: int,
    std: int,
    clip_delta: int,
    rng: random.Random,
    min_value: int = 1,
) -> int:
    if std == 0 or clip_delta == 0:
        return max(min_value, int(mean))
    lower = max(int(min_value), int(mean) - int(clip_delta))
    upper = max(lower, int(mean) + int(clip_delta))
    sampled = int(round(rng.gauss(mu=float(mean), sigma=float(std))))
    return max(lower, min(upper, sampled))


def _sampled_repeated_family_case(
    *,
    target_seq_len: int,
    prefix_len: int,
    suffix_len: int,
    completions_per_family: int,
    seed: int,
    prefix_length_std: int,
    prefix_length_clip_delta: int,
    branch_length_std: int,
    branch_length_clip_delta: int,
) -> GdnPhase0Case:
    rng = random.Random(seed)
    families: list[GdnFamilyShape] = []
    used = 0
    while True:
        prefix = _sample_length(
            mean=prefix_len,
            std=prefix_length_std,
            clip_delta=prefix_length_clip_delta,
            rng=rng,
        )
        suffixes = tuple(
            _sample_length(
                mean=suffix_len,
                std=branch_length_std,
                clip_delta=branch_length_clip_delta,
                rng=rng,
                min_value=2,
            )
            for _ in range(completions_per_family)
        )
        family = GdnFamilyShape(prefix_length=prefix, suffix_lengths=suffixes)
        fitted = fit_gdn_family_to_remaining(family, target_seq_len - used)
        if fitted is None:
            break
        families.append(fitted)
        used += gdn_family_token_count(fitted)
        if len(fitted.suffix_lengths) != len(family.suffix_lengths):
            break
    if not families:
        raise ValueError(
            "target_seq_len must fit at least one sampled repeated family, got "
            f"target={target_seq_len}"
        )
    return GdnPhase0Case(
        name=(
            f"sampled_{prefix_len}_plus_{completions_per_family}x"
            f"{suffix_len}_target_{target_seq_len}_seed_{seed}"
        ),
        sequence_length=target_seq_len,
        rows=(GdnPackedRowShape(families=tuple(families)),),
        seed=seed,
        description=(
            "One ART-realistic packed row with clipped-normal sampled prefix "
            "and completion lengths packed up to the target sequence length."
        ),
    )


def _sampled_single_family_case(
    *,
    prefix_len: int,
    suffix_len: int,
    completions_per_family: int,
    seed: int,
    prefix_length_std: int,
    prefix_length_clip_delta: int,
    branch_length_std: int,
    branch_length_clip_delta: int,
) -> GdnPhase0Case:
    rng = random.Random(seed)
    prefix = _sample_length(
        mean=prefix_len,
        std=prefix_length_std,
        clip_delta=prefix_length_clip_delta,
        rng=rng,
    )
    suffixes = tuple(
        _sample_length(
            mean=suffix_len,
            std=branch_length_std,
            clip_delta=branch_length_clip_delta,
            rng=rng,
            min_value=2,
        )
        for _ in range(completions_per_family)
    )
    family = GdnFamilyShape(prefix_length=prefix, suffix_lengths=suffixes)
    target_seq_len = gdn_family_token_count(family)
    return GdnPhase0Case(
        name=(
            f"sampled_single_{prefix_len}_plus_{completions_per_family}x"
            f"{suffix_len}_target_{target_seq_len}_seed_{seed}"
        ),
        sequence_length=target_seq_len,
        rows=(GdnPackedRowShape(families=(family,)),),
        seed=seed,
        description=(
            "One ART-realistic packed row with a single clipped-normal sampled "
            "prefix family and exact sequence length."
        ),
    )


def _deterministic_jitter_repeated_family_case(
    *,
    target_seq_len: int,
    prefix_len: int,
    suffix_len: int,
    completions_per_family: int,
) -> GdnPhase0Case:
    prefix_jitter = (0, -512, 384, 128, -256, 640, -128, 256)
    suffix_jitter = (
        -36,
        12,
        -20,
        28,
        -8,
        40,
        -28,
        16,
        -12,
        32,
        -4,
        24,
        -32,
        8,
        -16,
        36,
    )
    families: list[GdnFamilyShape] = []
    used = 0
    family_index = 0
    while True:
        prefix = max(1, prefix_len + prefix_jitter[family_index % len(prefix_jitter)])
        suffixes = tuple(
            max(
                2,
                suffix_len + suffix_jitter[(family_index + child) % len(suffix_jitter)],
            )
            for child in range(completions_per_family)
        )
        family = GdnFamilyShape(prefix_length=prefix, suffix_lengths=suffixes)
        fitted = fit_gdn_family_to_remaining(family, target_seq_len - used)
        if fitted is None:
            break
        families.append(fitted)
        used += gdn_family_token_count(fitted)
        if len(fitted.suffix_lengths) != len(family.suffix_lengths):
            break
        family_index += 1
    if not families:
        raise ValueError(
            "target_seq_len must fit at least one varied repeated family, got "
            f"target={target_seq_len}"
        )
    return GdnPhase0Case(
        name=(
            f"deterministic_jitter_{prefix_len}_plus_{completions_per_family}x"
            f"{suffix_len}_target_{target_seq_len}"
        ),
        sequence_length=target_seq_len,
        rows=(GdnPackedRowShape(families=tuple(families)),),
        seed=43,
        description=(
            "One deterministic benchmark row with periodic prefix and completion "
            "length jitter packed up to the target sequence length."
        ),
    )


def _deterministic_many_small_family_case(
    *,
    target_seq_len: int,
    prefix_len: int,
    suffix_len: int,
    completions_per_family: int,
) -> GdnPhase0Case:
    prefix_base = max(2, min(prefix_len, 96))
    suffix_base = max(2, min(suffix_len, 32))
    branch_count = max(2, min(completions_per_family, 8))
    prefix_jitter = (0, 7, -5, 11, -3, 5, -7, 13)
    suffix_jitter = (0, 3, -2, 5, -1, 2, -3, 4)
    families: list[GdnFamilyShape] = []
    used = 0
    family_index = 0
    while True:
        prefix = max(2, prefix_base + prefix_jitter[family_index % len(prefix_jitter)])
        suffixes = tuple(
            max(
                2,
                suffix_base
                + suffix_jitter[(family_index + child) % len(suffix_jitter)],
            )
            for child in range(branch_count)
        )
        family = GdnFamilyShape(prefix_length=prefix, suffix_lengths=suffixes)
        fitted = fit_gdn_family_to_remaining(family, target_seq_len - used)
        if fitted is None:
            break
        families.append(fitted)
        used += gdn_family_token_count(fitted)
        if len(fitted.suffix_lengths) != len(family.suffix_lengths):
            break
        family_index += 1
    if not families:
        raise ValueError(
            "target_seq_len must fit at least one many-small family, got "
            f"target={target_seq_len}"
        )
    return GdnPhase0Case(
        name=(
            f"deterministic_many_small_{prefix_base}_plus_{branch_count}x"
            f"{suffix_base}_target_{target_seq_len}"
        ),
        sequence_length=target_seq_len,
        rows=(GdnPackedRowShape(families=tuple(families)),),
        seed=47,
        description=(
            "One deterministic benchmark row with many independent small prompt "
            "families and periodic short completion length jitter."
        ),
    )


def _fixed_dominant_family_benchmark_case(
    *,
    target_seq_len: int,
    prefix_len: int,
    suffix_len: int,
    completions_per_family: int,
) -> GdnPhase0Case:
    branch_count = max(2, completions_per_family)
    dominant_family = GdnFamilyShape(
        prefix_length=prefix_len,
        suffix_lengths=(max(2, suffix_len),) * branch_count,
    )
    fitted = fit_gdn_family_to_remaining(dominant_family, target_seq_len)
    if fitted is None:
        raise ValueError(
            "target_seq_len must fit at least one dominant prefix plus completion, "
            f"got target={target_seq_len}, family_tokens={gdn_family_token_count(dominant_family)}"
        )
    families = [fitted]
    used = gdn_family_token_count(fitted)
    background_prefix = max(4, min(256, max(1, prefix_len // 16)))
    background_suffix = max(2, min(64, max(1, suffix_len // 2)))
    background_branches = max(2, min(4, completions_per_family))
    family_index = 0
    while True:
        prefix = background_prefix + (family_index % 5) * 3
        small_suffixes = tuple(
            background_suffix + ((family_index + child) % 4)
            for child in range(background_branches)
        )
        family = GdnFamilyShape(prefix_length=prefix, suffix_lengths=small_suffixes)
        fitted = fit_gdn_family_to_remaining(family, target_seq_len - used)
        if fitted is None:
            break
        families.append(fitted)
        used += gdn_family_token_count(fitted)
        if len(fitted.suffix_lengths) != len(family.suffix_lengths):
            break
        family_index += 1
    return GdnPhase0Case(
        name=(
            f"fixed_dominant_{prefix_len}_plus_{branch_count}x"
            f"{max(2, suffix_len)}_target_{target_seq_len}"
        ),
        sequence_length=target_seq_len,
        rows=(GdnPackedRowShape(families=tuple(families)),),
        seed=51,
        description=(
            "One fixed benchmark row with a dominant long prompt family and "
            "deterministic small background families."
        ),
    )


def _sampled_dominant_family_benchmark_case(
    *,
    target_seq_len: int,
    prefix_len: int,
    suffix_len: int,
    completions_per_family: int,
    seed: int,
    prefix_length_std: int,
    prefix_length_clip_delta: int,
    branch_length_std: int,
    branch_length_clip_delta: int,
    background_prefix_len: int,
    background_suffix_len: int,
    background_completions_per_family: int,
    background_prefix_length_std: int,
    background_prefix_length_clip_delta: int,
    background_branch_length_std: int,
    background_branch_length_clip_delta: int,
) -> GdnPhase0Case:
    rng = random.Random(seed)
    branch_count = max(2, completions_per_family)
    dominant_family = GdnFamilyShape(
        prefix_length=_sample_length(
            mean=prefix_len,
            std=prefix_length_std,
            clip_delta=prefix_length_clip_delta,
            rng=rng,
        ),
        suffix_lengths=tuple(
            _sample_length(
                mean=suffix_len,
                std=branch_length_std,
                clip_delta=branch_length_clip_delta,
                rng=rng,
                min_value=2,
            )
            for _ in range(branch_count)
        ),
    )
    fitted = fit_gdn_family_to_remaining(dominant_family, target_seq_len)
    if fitted is None:
        raise ValueError(
            "target_seq_len must fit at least one sampled dominant prefix plus "
            f"completion, got target={target_seq_len}, "
            f"family_tokens={gdn_family_token_count(dominant_family)}"
        )
    families = [fitted]
    used = gdn_family_token_count(fitted)
    background_branches = max(2, background_completions_per_family)
    while True:
        family = GdnFamilyShape(
            prefix_length=_sample_length(
                mean=background_prefix_len,
                std=background_prefix_length_std,
                clip_delta=background_prefix_length_clip_delta,
                rng=rng,
            ),
            suffix_lengths=tuple(
                _sample_length(
                    mean=background_suffix_len,
                    std=background_branch_length_std,
                    clip_delta=background_branch_length_clip_delta,
                    rng=rng,
                    min_value=2,
                )
                for _ in range(background_branches)
            ),
        )
        fitted = fit_gdn_family_to_remaining(family, target_seq_len - used)
        if fitted is None:
            break
        families.append(fitted)
        used += gdn_family_token_count(fitted)
        if len(fitted.suffix_lengths) != len(family.suffix_lengths):
            break
    return GdnPhase0Case(
        name=(
            f"sampled_dominant_{prefix_len}_plus_{branch_count}x"
            f"{max(2, suffix_len)}_target_{target_seq_len}_seed_{seed}"
        ),
        sequence_length=target_seq_len,
        rows=(GdnPackedRowShape(families=tuple(families)),),
        seed=seed,
        description=(
            "One ART-realistic packed row with a clipped-normal sampled "
            "dominant long prompt family and sampled smaller background families."
        ),
    )


def _case_counts(tensors: dict[str, Tensor]) -> dict[str, int]:
    spec = parse_gdn_shared_prefix_segments(
        tensors["group_ids"], tensors["parent_ids"], min_completions_per_family=1
    )
    return {
        "real_tokens": spec.real_token_count,
        "family_count": spec.family_count,
        "completion_count": spec.completion_count,
    }


def _layout_bytes_moved(hidden_states: Tensor, tensors: dict[str, Tensor]) -> int:
    return int(
        _case_counts(tensors)["real_tokens"]
        * hidden_states.shape[-1]
        * hidden_states.element_size()
        * 2
    )


def _state_bytes_materialized(gdn: Any, tensors: dict[str, Tensor]) -> int:
    counts = _case_counts(tensors)
    family_count = counts["family_count"]
    completion_count = counts["completion_count"]
    conv_state_elems = int(gdn.conv_dim_local_tp) * int(gdn.conv_kernel_dim - 1)
    rec_state_elems = (
        int(gdn.num_v_heads_local_tp) * int(gdn.key_head_dim) * int(gdn.value_head_dim)
    )
    conv_bytes = conv_state_elems * 4 * (family_count + completion_count)
    rec_bytes = rec_state_elems * 4 * (family_count + completion_count)
    return int(conv_bytes + rec_bytes)


def _masked_quadratic_loss(output: Tensor, assistant_mask: Tensor) -> Tensor:
    selected = output.transpose(0, 1)[assistant_mask]
    if selected.numel() == 0:
        raise ValueError("assistant_mask selects no tokens")
    return selected.square().sum()


def _summary(values: list[float]) -> TimingSummary:
    if not values:
        raise ValueError("at least one timing value is required")
    sorted_values = sorted(values)
    return TimingSummary(
        median_ms=float(torch.tensor(values).median().item()),
        p90_ms=sorted_values[
            min(len(sorted_values) - 1, int(0.9 * (len(sorted_values) - 1)))
        ],
        max_ms=max(values),
    )


def _assert_correctness_thresholds(
    case_name: str, metrics: RealGdnOracleMetrics
) -> None:
    if metrics.loss_mean_abs_pct > MEAN_ABS_PCT_THRESHOLD:
        raise AssertionError(
            f"{case_name}: loss_mean_abs_pct={metrics.loss_mean_abs_pct}%"
        )
    if metrics.output_mean_abs_pct > MEAN_ABS_PCT_THRESHOLD:
        raise AssertionError(
            f"{case_name}: output_mean_abs_pct={metrics.output_mean_abs_pct}%"
        )
    if metrics.hidden_grad_mean_abs_pct > MEAN_ABS_PCT_THRESHOLD:
        raise AssertionError(
            f"{case_name}: hidden_grad_mean_abs_pct={metrics.hidden_grad_mean_abs_pct}%"
        )
    if metrics.param_grad_mean_abs_pct > MEAN_ABS_PCT_THRESHOLD:
        raise AssertionError(
            f"{case_name}: param_grad_mean_abs_pct={metrics.param_grad_mean_abs_pct}%"
        )


def _caveats(args: argparse.Namespace) -> tuple[str, ...]:
    caveats = [
        "Phase 2 CP1 single-operation lab only; no CP2/CP4/CP8 GDN math, real distributed collectives, stacked benchmark, or isolated backend training claim.",
    ]
    if args.memory_debug:
        caveats.append(
            "Memory-debug uses saved_tensors_hooks and is not authoritative for speed."
        )
    if args.benchmark:
        caveats.append(
            "Benchmark mode intentionally skips flattened correctness to avoid polluting timing; run --correctness-only as the paired correctness gate."
        )
    if args.profile:
        caveats.append("Profile mode emits NVTX ranges for external nsys capture.")
    if args.benchmark_baselines:
        caveats.append(
            "Baseline comparison is CP1 only. The repeated_family workload uses one packed row, matching ART training's batch-size-one packed microbatch assumption."
        )
        caveats.append(
            "Canonical flex attention baseline excludes q/k/v projections, output projection, and block-mask construction; packed and flattened GDN include GDN projections."
        )
    if args.nsys_profile:
        caveats.append(
            "Nsys profile mode wraps benchmark --profile, exports SQLite, and writes parsed JSON/CSV/Markdown tables."
        )
    if args.parse_profile_sqlite is not None:
        caveats.append(
            "Parse-profile mode summarizes an existing nsys SQLite export and does not execute kernels."
        )
    return tuple(caveats)


def _active_params_dtype_name(args: argparse.Namespace) -> str:
    if (
        args.benchmark
        or args.memory_debug
        or args.benchmark_baselines
        or args.nsys_profile
    ):
        return str(BENCHMARK_DTYPE)
    return str(CORRECTNESS_DTYPE)


def _manifest_configs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "lab_args": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "dtype_policy": {
            "correctness_dtype": str(CORRECTNESS_DTYPE),
            "benchmark_dtype": str(BENCHMARK_DTYPE),
        },
        "benchmark_qwen35_gdn": qwen35_gdn_module_config()
        .model_copy(update={"linear_conv_kernel_dim": args.conv_width})
        .model_dump(),
        "gdn_linear_policy": str(args.gdn_linear_policy),
        "qwen35_tiny_gdn": {
            "num_layers": 4,
            "hidden_size": 64,
            "ffn_hidden_size": 128,
            "moe_ffn_hidden_size": 32,
            "moe_shared_expert_intermediate_size": 16,
            "num_attention_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_conv_kernel_dim": args.conv_width,
            "tensor_model_parallel_size": 1,
            "context_parallel_size": 1,
            "params_dtype": _active_params_dtype_name(args),
        },
    }


def _dynamo_disabled(function: Any) -> Any:
    disable = getattr(torch, "_dynamo", None)
    if disable is None:
        return function
    disable_fn = getattr(disable, "disable", None)
    if not callable(disable_fn):
        return function
    return disable_fn(function)


@contextmanager
def _single_rank_model_parallel() -> Iterator[None]:
    from megatron.core import parallel_state as ps

    if is_initialized():
        raise RuntimeError("torch.distributed is already initialized in this process")
    torch.cuda.set_device(0)
    init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{_find_free_port()}",
        rank=0,
        world_size=1,
    )
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        yield
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        if is_initialized():
            destroy_process_group()


@contextmanager
def _nvtx_range(label: str, *, enabled: bool) -> Iterator[None]:
    if enabled:
        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
        return
    yield


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real Megatron/FLA GDN lab modes")


def _require_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is None:
        raise ValueError("--output-dir is required for this mode")
    return args.output_dir


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    raise SystemExit(main())
