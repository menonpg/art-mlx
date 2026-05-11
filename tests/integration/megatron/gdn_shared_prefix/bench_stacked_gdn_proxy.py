from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import random
import socket
import statistics
import sys
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.multiprocessing as mp

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.context_parallel.runtime import _normalized_chunk_size
from art.megatron.context_parallel.types import ContextParallelConfig
from art.megatron.gdn.gdn_shared_prefix import (
    GdnPlannerConfig,
    build_gdn_rank_execution_plan,
    move_gdn_rank_execution_plan_to_device,
    parse_gdn_shared_prefix_segments,
)
from art.megatron.gdn.operator import (
    gdn_cp_attention_to_gdn_layout,
    gdn_cp_gdn_to_attention_layout,
    gdn_nvtx_ranges,
    run_gdn_layer,
)

from .artifacts import write_manifest
from .bench_single_gdn_operation import _selected_or_repeated_case
from .benchmark_gdn import QWEN35_GDN_LINEAR_POLICY, apply_gdn_linear_policy
from .cases import (
    GdnFamilyShape,
    GdnPackedRowShape,
    GdnPhase0Case,
    fit_gdn_family_to_remaining,
    gdn_family_token_count,
)
from .distributed_grad import all_reduce_parameter_grads_coalesced
from .packed_layout import build_gdn_group_parent_tensors
from .real_gdn_oracle import attach_main_grads, zero_parameter_grads
from .test_real_gdn_native_fla_cp import (
    Qwen3_5MoeVisionConfig,
    Qwen35VLMoEModelProvider,
    _first_gdn,
    model_parallel_cuda_manual_seed,
)
from .test_real_gdn_native_fla_cp import (
    _make_model as _make_toy_model,
)

BENCHMARK_DTYPE = torch.bfloat16


class StackedWorkloadConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    prefix_length_mode: str = "fixed"
    family_pattern: str = "uniform"
    base_target_seq_len: int = Field(ge=1)
    prefix_length_mean: int = Field(ge=1)
    prefix_length_std: int = Field(ge=0)
    prefix_length_clip_delta: int = Field(ge=0)
    branch_length_mean: int = Field(ge=2)
    branch_length_std: int = Field(ge=0)
    branch_length_clip_delta: int = Field(ge=0)
    branches_per_prefix: int = Field(ge=1)
    background_prefix_length_mean: int | None = Field(default=None, ge=1)
    background_prefix_length_std: int | None = Field(default=None, ge=0)
    background_prefix_length_clip_delta: int | None = Field(default=None, ge=0)
    background_branch_length_mean: int | None = Field(default=None, ge=2)
    background_branch_length_std: int | None = Field(default=None, ge=0)
    background_branch_length_clip_delta: int | None = Field(default=None, ge=0)
    background_branches_per_prefix: int | None = Field(default=None, ge=1)
    description: str = ""


class LayerSchedule(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    model_layer_count: int = Field(ge=1)
    gdn_layer_count: int = Field(ge=1)
    attention_layer_count: int = Field(ge=0)
    gdn_group_lengths: tuple[int, ...]
    layer_types: tuple[str, ...]
    description: str = ""


class GdnModuleConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    hidden_size: int = Field(ge=1)
    model_builder_layers: int = Field(ge=1)
    ffn_hidden_size: int = Field(ge=1)
    moe_ffn_hidden_size: int = Field(ge=1)
    moe_shared_expert_intermediate_size: int = Field(ge=1)
    num_attention_heads: int = Field(ge=1)
    num_query_groups: int = Field(ge=1)
    kv_channels: int = Field(ge=1)
    linear_key_head_dim: int = Field(ge=1)
    linear_value_head_dim: int = Field(ge=1)
    linear_num_key_heads: int = Field(ge=1)
    linear_num_value_heads: int = Field(ge=1)
    linear_conv_kernel_dim: int = Field(ge=1)
    num_moe_experts: int = Field(ge=1)
    moe_router_topk: int = Field(ge=1)
    description: str = ""


class WorkloadHistogram(BaseModel):
    model_config = ConfigDict(frozen=True)

    prefix_min: int = Field(ge=0)
    prefix_max: int = Field(ge=0)
    prefix_mean: float
    suffix_min: int = Field(ge=0)
    suffix_max: int = Field(ge=0)
    suffix_mean: float


class PreparedGdnSequence(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sequence_index: int = Field(ge=0)
    case_name: str
    case: GdnPhase0Case | None
    tensors: dict[str, Any]
    group_ids: torch.Tensor
    parent_ids: torch.Tensor
    spec: Any
    plan: Any
    setup_total_ms: float
    setup_blocking_ms: float
    plan_host_ms: float
    device_setup_sync_ms: float = 0.0
    overlap_window_ms: float = 0.0
    setup_event: torch.cuda.Event | None = None
    workload_histogram: WorkloadHistogram


class LayerLaunch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    start_wall_s: float
    layer_count: int = Field(ge=1)
    start_event: torch.cuda.Event
    reduce_start_event: torch.cuda.Event
    reduce_event: torch.cuda.Event
    event_ranges: tuple["CudaEventRange", ...]


class CudaEventRange(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    label: str
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event


class RankSequenceTiming(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=0)
    sequence_index: int = Field(ge=0)
    case_name: str
    attention_tokens: int = Field(ge=0)
    gdn_tokens: int = Field(ge=0)
    real_tokens: int = Field(ge=1)
    family_count: int = Field(ge=1)
    completion_count: int = Field(ge=1)
    setup_total_ms: float
    setup_blocking_ms: float
    plan_host_ms: float
    device_setup_sync_ms: float
    fwd_ms: float
    bwd_ms: float
    boundary_fwd_ms: float
    boundary_bwd_ms: float
    gdn_fwd_ms: float
    gdn_bwd_ms: float
    param_reduce_ms: float
    cuda_gap_ms: float
    layers_total_ms: float
    layer_window_ms: float
    e2e_ms: float
    e2e_with_param_reduce_ms: float
    sync_overhang_ms: float
    local_prefix_bucket_count: int = Field(ge=0)
    local_completion_bucket_count: int = Field(ge=0)
    chain_prefix_bucket_count: int = Field(ge=0)
    chain_completion_bucket_count: int = Field(ge=0)
    parent_state_exchange_family_count: int = Field(ge=0)
    layout_cross_rank_token_count: int = Field(ge=0)
    layout_cross_rank_bytes_per_direction: int = Field(ge=0)
    bucket_count: int = Field(ge=0)
    bucket_real_tokens: int = Field(ge=0)
    bucket_padded_tokens: int = Field(ge=0)
    bucket_padding_ratio: float
    max_bucket_length: int = Field(ge=0)
    max_bucket_segments: int = Field(ge=0)
    max_bucket_padding_ratio: float
    prefix_bucket_real_tokens: int = Field(ge=0)
    prefix_bucket_padded_tokens: int = Field(ge=0)
    prefix_bucket_padding_ratio: float
    completion_bucket_real_tokens: int = Field(ge=0)
    completion_bucket_padded_tokens: int = Field(ge=0)
    completion_bucket_padding_ratio: float
    chain_bucket_real_tokens: int = Field(ge=0)
    chain_bucket_padded_tokens: int = Field(ge=0)
    chain_bucket_padding_ratio: float


class SequenceSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    sequence_index: int = Field(ge=0)
    case_name: str
    real_tokens: int = Field(ge=1)
    family_count: int = Field(ge=1)
    completion_count: int = Field(ge=1)
    workload_histogram: WorkloadHistogram
    max_rank_setup_total_ms: float
    max_rank_setup_blocking_ms: float
    max_rank_plan_host_ms: float
    max_rank_device_setup_sync_ms: float
    max_rank_layers_total_ms: float
    max_rank_layer_window_ms: float
    max_rank_sync_overhang_ms: float
    max_rank_fwd_ms: float
    max_rank_bwd_ms: float
    max_rank_boundary_fwd_ms: float
    max_rank_boundary_bwd_ms: float
    max_rank_gdn_fwd_ms: float
    max_rank_gdn_bwd_ms: float
    max_rank_param_reduce_ms: float
    max_rank_cuda_gap_ms: float
    max_rank_e2e_with_param_reduce_ms: float
    max_rank_attention_tokens: int = Field(ge=0)
    max_rank_gdn_tokens: int = Field(ge=0)
    max_local_prefix_bucket_count: int = Field(ge=0)
    max_local_completion_bucket_count: int = Field(ge=0)
    max_chain_prefix_bucket_count: int = Field(ge=0)
    max_chain_completion_bucket_count: int = Field(ge=0)
    max_parent_state_exchange_family_count: int = Field(ge=0)
    max_layout_cross_rank_token_count: int = Field(ge=0)
    max_layout_cross_rank_bytes_per_direction: int = Field(ge=0)
    max_bucket_count: int = Field(ge=0)
    max_bucket_real_tokens: int = Field(ge=0)
    max_bucket_padded_tokens: int = Field(ge=0)
    max_bucket_padding_ratio: float
    max_bucket_length: int = Field(ge=0)
    max_bucket_segments: int = Field(ge=0)
    max_single_bucket_padding_ratio: float
    max_prefix_bucket_real_tokens: int = Field(ge=0)
    max_prefix_bucket_padded_tokens: int = Field(ge=0)
    max_prefix_bucket_padding_ratio: float
    max_completion_bucket_real_tokens: int = Field(ge=0)
    max_completion_bucket_padded_tokens: int = Field(ge=0)
    max_completion_bucket_padding_ratio: float
    max_chain_bucket_real_tokens: int = Field(ge=0)
    max_chain_bucket_padded_tokens: int = Field(ge=0)
    max_chain_bucket_padding_ratio: float
    end_to_end_ms: float
    end_to_end_per_layer_ms: float
    layer_window_per_layer_ms: float
    sync_overhang_per_layer_ms: float
    tokens_per_second: float
    ranks: tuple[RankSequenceTiming, ...]


class StackedRollup(BaseModel):
    model_config = ConfigDict(frozen=True)

    sequence_count: int = Field(ge=0)
    setup_total_ms: float
    setup_blocking_ms: float
    plan_host_ms: float
    device_setup_sync_ms: float
    layers_total_ms: float
    layer_window_ms: float
    layer_window_per_layer_ms: float
    sync_overhang_ms: float
    sync_overhang_per_layer_ms: float
    fwd_ms: float
    bwd_ms: float
    boundary_fwd_ms: float
    boundary_bwd_ms: float
    gdn_fwd_ms: float
    gdn_bwd_ms: float
    param_reduce_ms: float
    cuda_gap_ms: float
    end_to_end_ms: float
    end_to_end_per_layer_ms: float
    tokens_per_second: float
    layout_cross_rank_token_count: float
    layout_cross_rank_bytes_per_direction: float
    bucket_count: float
    bucket_real_tokens: float
    bucket_padded_tokens: float
    bucket_padding_ratio: float
    max_bucket_length: float
    max_bucket_segments: float
    max_single_bucket_padding_ratio: float
    prefix_bucket_padded_tokens: float
    prefix_bucket_padding_ratio: float
    completion_bucket_padded_tokens: float
    completion_bucket_padding_ratio: float
    chain_bucket_padded_tokens: float
    chain_bucket_padding_ratio: float


class StackedGdnProxyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    cp_size: int = Field(ge=1)
    dtype: str
    workload_name: str
    architecture: str
    gdn_module_config: GdnModuleConfig
    gdn_linear_policy: str
    cp_attention_layout: str
    model_layer_count: int = Field(ge=1)
    gdn_layer_count: int = Field(ge=1)
    attention_layer_count: int = Field(ge=0)
    gdn_group_lengths: tuple[int, ...]
    layer_types: tuple[str, ...]
    sequence_length: int = Field(ge=1)
    prefix_length_mode: str
    num_sequences: int = Field(ge=1)
    tail_window: int = Field(ge=1)
    all_sequences_median: StackedRollup
    tail_sequences_median: StackedRollup
    sequences: tuple[SequenceSummary, ...]


def _resolve_layer_schedule(args: argparse.Namespace) -> LayerSchedule:
    architecture = args.architecture or (
        "gdn_only" if args.layers is not None else "qwen3_5_35b_a3b"
    )
    if architecture == "gdn_only":
        gdn_layers = int(args.layers or 48)
        if gdn_layers < 1:
            raise ValueError("--layers must be >= 1")
        return LayerSchedule(
            name="gdn_only",
            model_layer_count=gdn_layers,
            gdn_layer_count=gdn_layers,
            attention_layer_count=0,
            gdn_group_lengths=(gdn_layers,),
            layer_types=tuple("linear_attention" for _ in range(gdn_layers)),
            description="Legacy controlled stack with every layer executed as GDN.",
        )
    if architecture != "qwen3_5_35b_a3b":
        raise ValueError(f"unknown architecture {architecture!r}")
    model_layers = int(args.layers or 40)
    if model_layers < 1 or model_layers > 40:
        raise ValueError("Qwen3.5-35B-A3B model-layer count must be in [1, 40]")
    layer_types = tuple(
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(model_layers)
    )
    group_lengths: list[int] = []
    current = 0
    for layer_type in layer_types:
        if layer_type == "linear_attention":
            current += 1
            continue
        if current:
            group_lengths.append(current)
            current = 0
    if current:
        group_lengths.append(current)
    gdn_layers = sum(group_lengths)
    if gdn_layers < 1:
        raise ValueError("Qwen3.5-35B-A3B schedule has no GDN layers to benchmark")
    return LayerSchedule(
        name="qwen3_5_35b_a3b",
        model_layer_count=model_layers,
        gdn_layer_count=gdn_layers,
        attention_layer_count=model_layers - gdn_layers,
        gdn_group_lengths=tuple(group_lengths),
        layer_types=layer_types,
        description=(
            "Qwen3.5-35B-A3B text schedule: three GDN/linear-attention layers "
            "followed by one full-attention layer."
        ),
    )


def _resolve_gdn_module_config(args: argparse.Namespace) -> GdnModuleConfig:
    name = str(args.gdn_module_config or "qwen3_5_35b_a3b")
    if name == "toy":
        return GdnModuleConfig(
            name="toy",
            hidden_size=64,
            model_builder_layers=4,
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
            linear_conv_kernel_dim=2,
            num_moe_experts=4,
            moe_router_topk=2,
            description="Small correctness-lab GDN dimensions for smoke/debug runs only.",
        )
    if name != "qwen3_5_35b_a3b":
        raise ValueError(f"unknown GDN module config {name!r}")
    return GdnModuleConfig(
        name="qwen3_5_35b_a3b",
        hidden_size=2048,
        model_builder_layers=1,
        ffn_hidden_size=12288,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        num_attention_heads=16,
        num_query_groups=2,
        kv_channels=256,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_conv_kernel_dim=4,
        num_moe_experts=4,
        moe_router_topk=2,
        description=(
            "Qwen3.5-35B-A3B GDN-relevant dimensions from the public config. "
            "MoE count/top-k are kept small because this benchmark extracts and "
            "runs only the GDN module."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    _configure_rank_cpu_threads()
    parser = argparse.ArgumentParser(
        description="Training-shaped stacked packed shared-prefix GDN proxy"
    )
    parser.add_argument("--cp-sizes", default="1,2,4")
    parser.add_argument(
        "--architecture",
        choices=("gdn_only", "qwen3_5_35b_a3b"),
        default=None,
        help=(
            "Layer schedule to model. Default is qwen3_5_35b_a3b unless "
            "--layers is provided for GDN-only runs."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help=(
            "GDN-only layer count, or a Qwen model-layer truncation "
            "when --architecture=qwen3_5_35b_a3b is explicit."
        ),
    )
    parser.add_argument("--num-sequences", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tail-window", type=int, default=16)
    parser.add_argument("--workloads", default="default_5k_16x100")
    parser.add_argument(
        "--case-name",
        default="",
        help="Legacy deterministic single-case generator. Empty uses --workloads.",
    )
    parser.add_argument(
        "--prefix-length-mode",
        choices=("fixed", "clipped_normal"),
        default=None,
        help="Override workload prefix-length mode. Workload defaults keep prefixes fixed unless the selected workload is explicitly varied.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--gdn-module-config",
        choices=("qwen3_5_35b_a3b", "toy"),
        default=None,
        help=(
            "GDN module dimensions. Default uses Qwen3.5-35B-A3B GDN-relevant "
            "parameters; toy is for fast smoke/debug runs only."
        ),
    )
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
    parser.add_argument(
        "--cp-attention-layout",
        choices=(
            "planner_default",
            "contiguous",
            "striped",
            "reversed_striped",
            "randomized_cp_chunks",
        ),
        default="planner_default",
        help=(
            "CP attention-token ownership fed into the GDN planner. "
            "planner_default lets the GDN planner choose the current low-exchange "
            "layout; reversed_striped reverses CP-sized chunk assignment order "
            "as a layout sensitivity check; randomized_cp_chunks "
            "shuffles attention-CP-sized token chunks across ranks."
        ),
    )
    parser.add_argument("--conv-width", type=int, default=None)
    parser.add_argument(
        "--target-seq-len",
        "--sequence-length",
        dest="target_seq_len",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--prefix-len",
        "--prefix-length-mean",
        dest="prefix_len",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--suffix-len",
        "--branch-length-mean",
        dest="suffix_len",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--completions-per-family",
        "--branches-per-prefix",
        dest="completions_per_family",
        type=int,
        default=None,
    )
    parser.add_argument("--prefix-length-std", type=int, default=None)
    parser.add_argument("--prefix-length-clip-delta", type=int, default=None)
    parser.add_argument("--branch-length-std", type=int, default=None)
    parser.add_argument("--branch-length-clip-delta", type=int, default=None)
    parser.add_argument(
        "--overlap-next-state-prep",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Emit stacked-proxy and GDN-operator NVTX ranges for external nsys capture.",
    )
    parser.add_argument(
        "--activation-checkpoint-gdn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Checkpoint each contiguous GDN group in the stacked proxy. Defaults "
            "on for Qwen-width GDN modules and off for toy smoke/debug modules."
        ),
    )
    parser.add_argument(
        "--output-dir", "--results-dir", dest="output_dir", type=Path, required=True
    )
    args = parser.parse_args(argv)
    args.num_sequences = int(
        args.num_sequences if args.num_sequences is not None else args.iters or 32
    )

    args.layer_schedule = _resolve_layer_schedule(args)
    args.gdn_module = _resolve_gdn_module_config(args)
    args.conv_width = int(args.conv_width or args.gdn_module.linear_conv_kernel_dim)
    if args.activation_checkpoint_gdn:
        raise ValueError(
            "--activation-checkpoint-gdn is not valid for the attention-style "
            "stacked proxy; each GDN layer is already an independent fwd/bwd."
        )
    args.activation_checkpoint_gdn = False
    if args.num_sequences < 1:
        raise ValueError("--num-sequences must be >= 1")
    if args.tail_window < 1:
        raise ValueError("--tail-window must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workloads = _selected_workloads(args)
    results: list[StackedGdnProxyResult] = []
    for workload in workloads:
        for cp_size in tuple(int(value) for value in args.cp_sizes.split(",") if value):
            run_args = _args_for_run(args, workload, cp_size)
            run_dir = args.output_dir / workload.name / f"cp{cp_size}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if cp_size == 1:
                results.append(_run_cp1(run_args, run_dir))
            else:
                port = _find_free_port()
                mp.spawn(
                    _worker,
                    args=(cp_size, port, run_args, str(run_dir)),
                    nprocs=cp_size,
                    join=True,
                )
                results.append(
                    StackedGdnProxyResult.model_validate_json(
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
        kind="gdn_stacked_training_proxy_benchmark",
        command=sys.argv,
        configs=_manifest_configs(args, workloads),
        cases=tuple(result.model_dump() for result in results),
    )
    print(json.dumps({"manifest": str(manifest_path)}), flush=True)
    return 0


def _run_cp1(args: argparse.Namespace, run_dir: Path) -> StackedGdnProxyResult:
    from megatron.core import parallel_state as ps

    _configure_rank_cpu_threads()
    torch.cuda.set_device(0)
    init_process_group(
        backend="gloo",
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
        _, gdn = _make_benchmark_gdn_pair(
            cp_size=1,
            config=args.gdn_module,
            linear_policy=args.gdn_linear_policy,
        )
        result = _run_rank_sequence_stream(
            rank=0,
            cp_size=1,
            gdn=gdn,
            args=args,
            run_dir=run_dir,
            cp_group=None,
            reduce_params=False,
        )
        return result
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def _worker(
    rank: int, cp_size: int, port: int, args: argparse.Namespace, run_dir: str
) -> None:
    from megatron.core import parallel_state as ps

    _configure_rank_cpu_threads()
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
        cp_group = ps.get_context_parallel_group()
        _, gdn = _make_benchmark_gdn_pair(
            cp_size=cp_size,
            config=args.gdn_module,
            linear_policy=args.gdn_linear_policy,
        )
        _run_rank_sequence_stream(
            rank=rank,
            cp_size=cp_size,
            gdn=gdn,
            args=args,
            run_dir=Path(run_dir),
            cp_group=cp_group,
            reduce_params=True,
        )
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def _run_rank_sequence_stream(
    *,
    rank: int,
    cp_size: int,
    gdn: torch.nn.Module,
    args: argparse.Namespace,
    run_dir: Path,
    cp_group: Any | None,
    reduce_params: bool,
) -> StackedGdnProxyResult:
    setup_stream = torch.cuda.Stream()
    pending = _prepare_sequence(
        sequence_index=0,
        args=args,
        cp_rank=rank,
        cp_size=cp_size,
        cp_group=cp_group,
        setup_stream=setup_stream,
    )
    summaries: list[SequenceSummary] = []
    for sequence_index in range(args.num_sequences):
        context = _apply_setup_overlap(pending)
        context = _sync_pending_setup(context)
        hidden, output_grad = _hidden_and_grad(
            context.case,
            context.plan,
            cp_size=cp_size,
            seed=int(args.seed) + sequence_index * 97 + rank * 10_000,
            hidden_size=args.gdn_module.hidden_size,
        )
        _dist_barrier()
        launch = _launch_layers(
            gdn,
            hidden,
            output_grad,
            group_ids=context.group_ids,
            parent_ids=context.parent_ids,
            spec=context.spec,
            plan=context.plan,
            layer_schedule=args.layer_schedule,
            cp_group=cp_group,
            reduce_params=reduce_params,
            profile=bool(args.profile),
        )

        next_context = None
        if (
            bool(args.overlap_next_state_prep)
            and sequence_index + 1 < args.num_sequences
        ):
            next_context = _prepare_sequence(
                sequence_index=sequence_index + 1,
                args=args,
                cp_rank=rank,
                cp_size=cp_size,
                cp_group=cp_group,
                setup_stream=setup_stream,
            )

        timing = _finalize_layers(launch)
        if next_context is not None:
            next_context.overlap_window_ms = float(timing["layers_total_ms"])
        rank_timing = _rank_sequence_timing(
            rank=rank,
            context=context,
            timing=timing,
            hidden_size=args.gdn_module.hidden_size,
        )
        gathered = _gather_rank_timing(rank_timing, cp_size)
        if rank == 0:
            summary = _summarize_sequence(
                tuple(gathered),
                layer_count=args.layer_schedule.gdn_layer_count,
                workload_histogram=context.workload_histogram,
            )
            summaries.append(summary)
            _write_progress(run_dir, args, tuple(summaries), is_final=False)
        if next_context is None and sequence_index + 1 < args.num_sequences:
            next_context = _prepare_sequence(
                sequence_index=sequence_index + 1,
                args=args,
                cp_rank=rank,
                cp_size=cp_size,
                cp_group=cp_group,
                setup_stream=setup_stream,
            )
        pending = next_context

    if pending is not None:
        raise RuntimeError("internal benchmark loop left an unused pending sequence")
    if rank != 0:
        return _empty_nonzero_rank_result(args)
    result = _aggregate_result(
        args=args,
        sequences=tuple(summaries),
    )
    (run_dir / "result_rank0.json").write_text(result.model_dump_json(indent=2) + "\n")
    _write_progress(run_dir, args, tuple(summaries), is_final=True)
    return result


def _prepare_sequence(
    *,
    sequence_index: int,
    args: argparse.Namespace,
    cp_rank: int,
    cp_size: int,
    cp_group: Any | None,
    setup_stream: torch.cuda.Stream,
) -> PreparedGdnSequence:
    start = time.perf_counter()
    case = _build_sequence_case(
        args=args,
        sequence_index=sequence_index,
    )
    tensors = build_gdn_group_parent_tensors(case)
    plan_group_ids = tensors["group_ids"]
    plan_parent_ids = tensors["parent_ids"]
    case_name = case.name
    workload_histogram = _workload_histogram(case)
    with torch.cuda.stream(setup_stream):
        plan_start = time.perf_counter()
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            spec, plan = _build_execution_plan(
                plan_group_ids,
                plan_parent_ids,
                cp_rank=cp_rank,
                cp_size=cp_size,
                cp_group=cp_group,
                cp_attention_layout=args.cp_attention_layout,
                seed=int(args.seed),
                device=torch.device("cpu"),
            )
        finally:
            if gc_was_enabled:
                gc.enable()
        plan_host_ms = (time.perf_counter() - plan_start) * 1000.0
        plan = move_gdn_rank_execution_plan_to_device(
            plan, torch.device("cuda", torch.cuda.current_device())
        )
        if cp_size == 1:
            group_ids = plan_group_ids.cuda(non_blocking=True)
            parent_ids = plan_parent_ids.cuda(non_blocking=True)
        else:
            group_ids = plan_group_ids
            parent_ids = plan_parent_ids
        setup_event = torch.cuda.Event()
        setup_event.record(setup_stream)
    setup_total_ms = (time.perf_counter() - start) * 1000.0
    return PreparedGdnSequence(
        sequence_index=sequence_index,
        case_name=case_name,
        case=case,
        tensors=tensors,
        group_ids=group_ids,
        parent_ids=parent_ids,
        spec=spec,
        plan=plan,
        setup_total_ms=setup_total_ms,
        setup_blocking_ms=setup_total_ms,
        plan_host_ms=plan_host_ms,
        setup_event=setup_event,
        workload_histogram=workload_histogram,
    )


def _build_execution_plan(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
    cp_group: Any | None,
    cp_attention_layout: str,
    seed: int,
    device: torch.device,
) -> tuple[Any, Any]:
    if cp_size == 1:
        spec = parse_gdn_shared_prefix_segments(
            group_ids, parent_ids, min_completions_per_family=0
        )
        return spec, build_gdn_rank_execution_plan(spec, device=device)
    spec = parse_gdn_shared_prefix_segments(
        group_ids, parent_ids, min_completions_per_family=0
    )
    attention_token_layout_index = _attention_layout_index_for_mode(
        spec,
        cp_size=cp_size,
        mode=cp_attention_layout,
        seed=seed,
    )
    return spec, build_gdn_rank_execution_plan(
        spec,
        device=device,
        cp_rank=cp_rank,
        cp_size=cp_size,
        attention_token_layout_index=attention_token_layout_index,
    )


def _attention_layout_index_for_mode(
    spec: Any,
    *,
    cp_size: int,
    mode: str,
    seed: int,
) -> TokenLayoutIndex | None:
    if mode == "planner_default":
        return None
    ranges_by_rank = _attention_layout_ranges_for_mode(
        spec,
        cp_size=cp_size,
        mode=mode,
        seed=seed,
    )
    return TokenLayoutIndex(
        ownership_ranges_by_rank=ranges_by_rank,
        token_counts_by_rank=tuple(
            sum(end - start for start, end, _ in ranges) for ranges in ranges_by_rank
        ),
    )


def _attention_layout_ranges_for_mode(
    spec: Any,
    *,
    cp_size: int,
    mode: str,
    seed: int,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    chunks = _cp_chunk_ranges(spec, cp_size=cp_size)
    if mode == "contiguous":
        return _assign_chunks_contiguous(chunks, cp_size=cp_size)
    if mode == "striped":
        return _assign_chunks_round_robin(chunks, cp_size=cp_size)
    if mode == "reversed_striped":
        return _assign_chunks_round_robin(tuple(reversed(chunks)), cp_size=cp_size)
    if mode == "randomized_cp_chunks":
        shuffled = list(chunks)
        rng = random.Random(int(seed) + 1009 * int(cp_size) + 9176 * len(shuffled))
        rng.shuffle(shuffled)
        return _assign_chunks_round_robin(tuple(shuffled), cp_size=cp_size)
    raise ValueError(f"unknown CP attention layout mode {mode!r}")


def _cp_chunk_ranges(
    spec: Any,
    *,
    cp_size: int,
) -> tuple[tuple[int, int], ...]:
    config = ContextParallelConfig()
    chunks = []
    for row_index, valid_length in enumerate(spec.valid_lengths):
        row_valid_tokens = int(valid_length)
        row_start = int(row_index) * int(spec.sequence_length)
        chunk_size = _normalized_chunk_size(
            valid_tokens=row_valid_tokens,
            block_size=int(config.block_size),
            requested_chunk_size=int(config.planner_chunk_size),
            cp_size=cp_size,
            config=config,
        )
        for start in range(0, row_valid_tokens, chunk_size):
            chunks.append(
                (
                    row_start + start,
                    row_start + min(start + chunk_size, row_valid_tokens),
                )
            )
    return tuple(chunks)


def _assign_chunks_round_robin(
    chunks: tuple[tuple[int, int], ...],
    *,
    cp_size: int,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    ranks: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
    rank_positions = [0] * cp_size
    for offset, (start, end) in enumerate(chunks):
        rank = offset % cp_size
        position = rank_positions[rank]
        ranks[rank].append((start, end, position))
        rank_positions[rank] += end - start
    return tuple(tuple(ranges) for ranges in ranks)


def _assign_chunks_contiguous(
    chunks: tuple[tuple[int, int], ...],
    *,
    cp_size: int,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    total_tokens = sum(end - start for start, end in chunks)
    ranks: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
    rank_positions = [0] * cp_size
    rank = 0
    target_end = (total_tokens * (rank + 1)) // cp_size
    seen = 0
    for start, end in chunks:
        while rank + 1 < cp_size and seen >= target_end:
            rank += 1
            target_end = (total_tokens * (rank + 1)) // cp_size
        position = rank_positions[rank]
        ranks[rank].append((start, end, position))
        length = end - start
        rank_positions[rank] += length
        seen += length
    return tuple(tuple(ranges) for ranges in ranks)


def _configure_rank_cpu_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)


def _make_benchmark_gdn_pair(
    *, cp_size: int, config: GdnModuleConfig, linear_policy: str
) -> tuple[torch.nn.Module, torch.nn.Module]:
    ref_gdn = _make_single_benchmark_gdn(
        config=config,
        cp_size=cp_size,
        seed=1234,
        params_dtype=BENCHMARK_DTYPE,
    )
    cp_gdn = _make_single_benchmark_gdn(
        config=config,
        cp_size=cp_size,
        seed=5678,
        params_dtype=BENCHMARK_DTYPE,
    )
    cp_gdn.load_state_dict(ref_gdn.state_dict())
    apply_gdn_linear_policy(ref_gdn, linear_policy)
    apply_gdn_linear_policy(cp_gdn, linear_policy)
    attach_main_grads(ref_gdn)
    attach_main_grads(cp_gdn)
    return ref_gdn, cp_gdn


def _make_single_benchmark_gdn(
    *,
    config: GdnModuleConfig,
    cp_size: int,
    seed: int,
    params_dtype: torch.dtype,
) -> torch.nn.Module:
    if config.name == "toy":
        model_parallel_cuda_manual_seed(seed)
        return _first_gdn(_make_toy_model(cp_size=cp_size, params_dtype=params_dtype))
    model_parallel_cuda_manual_seed(seed)
    return _first_gdn(_make_benchmark_model(config, params_dtype=params_dtype))


def _make_benchmark_model(
    config: GdnModuleConfig,
    *,
    params_dtype: torch.dtype,
) -> torch.nn.Module:
    assert Qwen3_5MoeVisionConfig is not None
    provider = Qwen35VLMoEModelProvider(
        num_layers=config.model_builder_layers,
        hidden_size=config.hidden_size,
        ffn_hidden_size=config.ffn_hidden_size,
        moe_ffn_hidden_size=config.moe_ffn_hidden_size,
        moe_shared_expert_intermediate_size=config.moe_shared_expert_intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_query_groups,
        kv_channels=config.kv_channels,
        linear_key_head_dim=config.linear_key_head_dim,
        linear_value_head_dim=config.linear_value_head_dim,
        linear_num_key_heads=config.linear_num_key_heads,
        linear_num_value_heads=config.linear_num_value_heads,
        num_moe_experts=config.num_moe_experts,
        moe_router_topk=config.moe_router_topk,
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
        linear_conv_kernel_dim=config.linear_conv_kernel_dim,
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


def _apply_setup_overlap(context: PreparedGdnSequence) -> PreparedGdnSequence:
    blocking = max(
        0.0,
        float(context.setup_total_ms) - float(context.overlap_window_ms),
    )
    context.setup_blocking_ms = blocking
    return context


def _sync_pending_setup(context: PreparedGdnSequence) -> PreparedGdnSequence:
    start = time.perf_counter()
    if context.setup_event is None:
        torch.cuda.synchronize()
    else:
        context.setup_event.synchronize()
    sync_ms = (time.perf_counter() - start) * 1000.0
    context.device_setup_sync_ms = sync_ms
    context.setup_total_ms += sync_ms
    context.setup_blocking_ms += sync_ms
    return context


@contextmanager
def _nvtx_range(label: str, *, enabled: bool) -> Iterator[None]:
    if enabled:
        torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def _event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def _hidden_and_grad(
    case: GdnPhase0Case | None,
    plan: Any,
    *,
    cp_size: int,
    seed: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if cp_size > 1:
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
        raise ValueError("CP1 stacked benchmark requires a full packed case")
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


def _launch_layers(
    gdn: torch.nn.Module,
    hidden_template: torch.Tensor,
    output_grad: torch.Tensor,
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    spec: Any,
    plan: Any,
    layer_schedule: LayerSchedule,
    cp_group: Any | None,
    reduce_params: bool,
    profile: bool,
) -> LayerLaunch:
    if getattr(plan, "cp_size", 1) != 1:
        return _launch_grouped_cp_layers(
            gdn,
            hidden_template,
            output_grad,
            group_ids=group_ids,
            parent_ids=parent_ids,
            plan=plan,
            layer_schedule=layer_schedule,
            cp_group=cp_group,
            reduce_params=reduce_params,
            profile=profile,
        )
    zero_parameter_grads(gdn)
    start_event = torch.cuda.Event(enable_timing=True)
    reduce_start_event = torch.cuda.Event(enable_timing=True)
    reduce_event = torch.cuda.Event(enable_timing=True)
    event_ranges: list[CudaEventRange] = []
    start_wall_s = time.perf_counter()
    start_event.record()
    with gdn_nvtx_ranges(profile):
        with _nvtx_range("art_gdn_stacked_sequence_layers", enabled=profile):
            for _ in range(layer_schedule.gdn_layer_count):
                hidden = hidden_template.detach().requires_grad_(True)
                fwd_start, fwd_end = _event_pair()
                fwd_start.record()
                with _nvtx_range("art_gdn_stacked_gdn_forward", enabled=profile):
                    out, _ = run_gdn_layer(
                        gdn,
                        hidden,
                        group_ids=group_ids,
                        parent_ids=parent_ids,
                        execution_spec=spec,
                        execution_plan=plan,
                        cp_group=cp_group,
                    )
                fwd_end.record()
                event_ranges.append(
                    CudaEventRange(
                        label="gdn_forward",
                        start_event=fwd_start,
                        end_event=fwd_end,
                    )
                )
                bwd_start, bwd_end = _event_pair()
                bwd_start.record()
                with _nvtx_range("art_gdn_stacked_gdn_backward", enabled=profile):
                    (out * output_grad).sum().backward()
                bwd_end.record()
                event_ranges.append(
                    CudaEventRange(
                        label="gdn_backward",
                        start_event=bwd_start,
                        end_event=bwd_end,
                    )
                )
            reduce_start_event.record()
            with _nvtx_range("art_gdn_stacked_param_reduce", enabled=profile):
                if reduce_params:
                    all_reduce_parameter_grads_coalesced(gdn, group=cp_group)
    reduce_event.record()
    return LayerLaunch(
        start_wall_s=start_wall_s,
        layer_count=layer_schedule.gdn_layer_count,
        start_event=start_event,
        reduce_start_event=reduce_start_event,
        reduce_event=reduce_event,
        event_ranges=tuple(event_ranges),
    )


def _launch_grouped_cp_layers(
    gdn: torch.nn.Module,
    hidden_template: torch.Tensor,
    output_grad: torch.Tensor,
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    plan: Any,
    layer_schedule: LayerSchedule,
    cp_group: Any | None,
    reduce_params: bool,
    profile: bool,
) -> LayerLaunch:
    if cp_group is None:
        raise ValueError("CP grouped GDN benchmark requires a context-parallel group")
    if sum(layer_schedule.gdn_group_lengths) != layer_schedule.gdn_layer_count:
        raise ValueError("GDN group lengths must sum to the counted GDN layer count")
    zero_parameter_grads(gdn)
    gdn_output_grad = torch.ones(
        int(plan.gdn_token_count),
        1,
        hidden_template.shape[-1],
        device=hidden_template.device,
        dtype=hidden_template.dtype,
    )
    start_event = torch.cuda.Event(enable_timing=True)
    reduce_start_event = torch.cuda.Event(enable_timing=True)
    reduce_event = torch.cuda.Event(enable_timing=True)
    event_ranges: list[CudaEventRange] = []
    start_wall_s = time.perf_counter()
    start_event.record()
    with gdn_nvtx_ranges(profile):
        with _nvtx_range("art_gdn_stacked_sequence_layers", enabled=profile):
            for group_length in layer_schedule.gdn_group_lengths:
                hidden = hidden_template.detach().requires_grad_(True)
                boundary_fwd_start, boundary_fwd_end = _event_pair()
                boundary_fwd_start.record()
                with _nvtx_range("art_gdn_stacked_boundary_forward", enabled=profile):
                    gdn_hidden, original_shape = gdn_cp_attention_to_gdn_layout(
                        hidden, plan, cp_group
                    )
                    gdn_hidden_template = gdn_hidden.detach()
                    attention_output = gdn_cp_gdn_to_attention_layout(
                        gdn_hidden, plan, original_shape, cp_group
                    )
                boundary_fwd_end.record()
                event_ranges.append(
                    CudaEventRange(
                        label="boundary_forward",
                        start_event=boundary_fwd_start,
                        end_event=boundary_fwd_end,
                    )
                )
                boundary_bwd_start, boundary_bwd_end = _event_pair()
                boundary_bwd_start.record()
                with _nvtx_range("art_gdn_stacked_boundary_backward", enabled=profile):
                    (attention_output * output_grad).sum().backward()
                boundary_bwd_end.record()
                event_ranges.append(
                    CudaEventRange(
                        label="boundary_backward",
                        start_event=boundary_bwd_start,
                        end_event=boundary_bwd_end,
                    )
                )
                for _ in range(group_length):
                    gdn_hidden = gdn_hidden_template.detach().requires_grad_(True)
                    fwd_start, fwd_end = _event_pair()
                    fwd_start.record()
                    with _nvtx_range("art_gdn_stacked_gdn_forward", enabled=profile):
                        out, _ = run_gdn_layer(
                            gdn,
                            gdn_hidden,
                            group_ids=group_ids,
                            parent_ids=parent_ids,
                            execution_plan=plan,
                            cp_group=cp_group,
                            input_layout="gdn",
                            output_layout="gdn",
                        )
                    fwd_end.record()
                    event_ranges.append(
                        CudaEventRange(
                            label="gdn_forward",
                            start_event=fwd_start,
                            end_event=fwd_end,
                        )
                    )
                    bwd_start, bwd_end = _event_pair()
                    bwd_start.record()
                    with _nvtx_range("art_gdn_stacked_gdn_backward", enabled=profile):
                        (out * gdn_output_grad).sum().backward()
                    bwd_end.record()
                    event_ranges.append(
                        CudaEventRange(
                            label="gdn_backward",
                            start_event=bwd_start,
                            end_event=bwd_end,
                        )
                    )
            reduce_start_event.record()
            with _nvtx_range("art_gdn_stacked_param_reduce", enabled=profile):
                if reduce_params:
                    all_reduce_parameter_grads_coalesced(gdn, group=cp_group)
    reduce_event.record()
    return LayerLaunch(
        start_wall_s=start_wall_s,
        layer_count=layer_schedule.gdn_layer_count,
        start_event=start_event,
        reduce_start_event=reduce_start_event,
        reduce_event=reduce_event,
        event_ranges=tuple(event_ranges),
    )


def _finalize_layers(launch: LayerLaunch) -> dict[str, float]:
    launch.reduce_event.synchronize()
    layer_window_ms = float(launch.start_event.elapsed_time(launch.reduce_event))
    layers_total_ms = (time.perf_counter() - launch.start_wall_s) * 1000.0
    by_label = {
        "boundary_forward": 0.0,
        "boundary_backward": 0.0,
        "gdn_forward": 0.0,
        "gdn_backward": 0.0,
    }
    for event_range in launch.event_ranges:
        by_label[event_range.label] = by_label.get(event_range.label, 0.0) + float(
            event_range.start_event.elapsed_time(event_range.end_event)
        )
    param_reduce_ms = float(launch.reduce_start_event.elapsed_time(launch.reduce_event))
    attributed_cuda_ms = sum(by_label.values()) + param_reduce_ms
    fwd_ms = by_label["boundary_forward"] + by_label["gdn_forward"]
    bwd_ms = by_label["boundary_backward"] + by_label["gdn_backward"]
    return {
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "boundary_fwd_ms": by_label["boundary_forward"],
        "boundary_bwd_ms": by_label["boundary_backward"],
        "gdn_fwd_ms": by_label["gdn_forward"],
        "gdn_bwd_ms": by_label["gdn_backward"],
        "param_reduce_ms": param_reduce_ms,
        "cuda_gap_ms": max(0.0, layer_window_ms - attributed_cuda_ms),
        "e2e_ms": fwd_ms + bwd_ms,
        "e2e_with_param_reduce_ms": layer_window_ms,
        "layer_window_ms": layer_window_ms,
        "layers_total_ms": layers_total_ms,
        "sync_overhang_ms": max(0.0, layers_total_ms - layer_window_ms),
    }


def _rank_sequence_timing(
    *,
    rank: int,
    context: PreparedGdnSequence,
    timing: dict[str, float],
    hidden_size: int,
) -> RankSequenceTiming:
    plan = context.plan
    all_bucket_stats = _bucket_stats(_all_execution_buckets(plan))
    prefix_bucket_stats = _bucket_stats(
        (
            *plan.local_prefix_buckets,
            *plan.prefix_boundary_buckets,
            *plan.prefix_tail_buckets,
            *plan.remote_prefix_tail_buckets,
        )
    )
    completion_bucket_stats = _bucket_stats(
        (
            *plan.local_completion_buckets,
            *plan.completion_with_prefix_tail_buckets,
            *plan.remote_completion_with_prefix_tail_buckets,
        )
    )
    chain_bucket_stats = _bucket_stats(
        (*plan.chain_prefix_buckets, *plan.chain_completion_buckets)
    )
    return RankSequenceTiming(
        rank=rank,
        sequence_index=context.sequence_index,
        case_name=context.case_name,
        attention_tokens=int(plan.attention_token_count)
        if plan.cp_size > 1
        else context.spec.real_token_count,
        gdn_tokens=int(plan.gdn_token_count)
        if plan.cp_size > 1
        else context.spec.real_token_count,
        real_tokens=context.spec.real_token_count,
        family_count=context.spec.family_count,
        completion_count=context.spec.completion_count,
        setup_total_ms=context.setup_total_ms,
        setup_blocking_ms=context.setup_blocking_ms,
        plan_host_ms=context.plan_host_ms,
        device_setup_sync_ms=context.device_setup_sync_ms,
        fwd_ms=timing["fwd_ms"],
        bwd_ms=timing["bwd_ms"],
        boundary_fwd_ms=timing["boundary_fwd_ms"],
        boundary_bwd_ms=timing["boundary_bwd_ms"],
        gdn_fwd_ms=timing["gdn_fwd_ms"],
        gdn_bwd_ms=timing["gdn_bwd_ms"],
        param_reduce_ms=timing["param_reduce_ms"],
        cuda_gap_ms=timing["cuda_gap_ms"],
        layers_total_ms=timing["layers_total_ms"],
        layer_window_ms=timing["layer_window_ms"],
        e2e_ms=timing["e2e_ms"],
        e2e_with_param_reduce_ms=timing["e2e_with_param_reduce_ms"],
        sync_overhang_ms=timing["sync_overhang_ms"],
        local_prefix_bucket_count=(
            len(plan.local_prefix_buckets)
            + len(plan.prefix_boundary_buckets)
            + len(plan.prefix_tail_buckets)
            + len(plan.remote_prefix_tail_buckets)
        ),
        local_completion_bucket_count=(
            len(plan.local_completion_buckets)
            + len(plan.completion_with_prefix_tail_buckets)
        ),
        chain_prefix_bucket_count=len(plan.chain_prefix_buckets),
        chain_completion_bucket_count=len(plan.chain_completion_buckets),
        parent_state_exchange_family_count=len(
            plan.parent_state_exchange_family_indices
        ),
        layout_cross_rank_token_count=_layout_cross_rank_token_count(plan),
        layout_cross_rank_bytes_per_direction=_layout_cross_rank_bytes_per_direction(
            plan,
            hidden_size=hidden_size,
        ),
        bucket_count=all_bucket_stats["bucket_count"],
        bucket_real_tokens=all_bucket_stats["real_tokens"],
        bucket_padded_tokens=all_bucket_stats["padded_tokens"],
        bucket_padding_ratio=all_bucket_stats["padding_ratio"],
        max_bucket_length=all_bucket_stats["max_length"],
        max_bucket_segments=all_bucket_stats["max_segments"],
        max_bucket_padding_ratio=all_bucket_stats["max_padding_ratio"],
        prefix_bucket_real_tokens=prefix_bucket_stats["real_tokens"],
        prefix_bucket_padded_tokens=prefix_bucket_stats["padded_tokens"],
        prefix_bucket_padding_ratio=prefix_bucket_stats["padding_ratio"],
        completion_bucket_real_tokens=completion_bucket_stats["real_tokens"],
        completion_bucket_padded_tokens=completion_bucket_stats["padded_tokens"],
        completion_bucket_padding_ratio=completion_bucket_stats["padding_ratio"],
        chain_bucket_real_tokens=chain_bucket_stats["real_tokens"],
        chain_bucket_padded_tokens=chain_bucket_stats["padded_tokens"],
        chain_bucket_padding_ratio=chain_bucket_stats["padding_ratio"],
    )


def _all_execution_buckets(plan: Any) -> tuple[Any, ...]:
    return (
        *plan.local_prefix_buckets,
        *plan.local_completion_buckets,
        *plan.chain_prefix_buckets,
        *plan.chain_completion_buckets,
        *plan.prefix_boundary_buckets,
        *plan.prefix_tail_buckets,
        *plan.completion_with_prefix_tail_buckets,
        *plan.remote_prefix_tail_buckets,
        *plan.remote_completion_with_prefix_tail_buckets,
    )


def _bucket_stats(buckets: tuple[Any, ...]) -> dict[str, int | float]:
    padded_tokens = 0
    real_tokens = 0
    max_length = 0
    max_segments = 0
    max_padding_ratio = 0.0
    for bucket in buckets:
        segment_count = int(bucket.segment_count)
        padded = int(bucket.length) * segment_count
        real = int(bucket.real_token_count_static)
        padded_tokens += padded
        real_tokens += real
        max_length = max(max_length, int(bucket.length))
        max_segments = max(max_segments, segment_count)
        max_padding_ratio = max(max_padding_ratio, _ratio(padded, real))
    return {
        "bucket_count": len(buckets),
        "real_tokens": real_tokens,
        "padded_tokens": padded_tokens,
        "padding_ratio": _ratio(padded_tokens, real_tokens),
        "max_length": max_length,
        "max_segments": max_segments,
        "max_padding_ratio": max_padding_ratio,
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if float(denominator) == 0.0 else float(numerator) / float(denominator)


def _layout_cross_rank_token_count(plan: Any) -> int:
    exchange = getattr(plan, "attention_to_gdn", None)
    if exchange is None:
        return 0
    return int(getattr(exchange, "cross_rank_token_count", 0))


def _layout_cross_rank_bytes_per_direction(plan: Any, *, hidden_size: int) -> int:
    element_size = torch.tensor((), dtype=BENCHMARK_DTYPE).element_size()
    return _layout_cross_rank_token_count(plan) * int(hidden_size) * int(element_size)


def _gather_rank_timing(
    rank_timing: RankSequenceTiming, cp_size: int
) -> tuple[RankSequenceTiming, ...]:
    if cp_size == 1:
        return (rank_timing,)
    gathered: list[Any] = [None for _ in range(cp_size)]
    torch.distributed.all_gather_object(  # ty: ignore[possibly-missing-attribute]
        gathered, rank_timing.model_dump()
    )
    return tuple(RankSequenceTiming.model_validate(item) for item in gathered)


def _summarize_sequence(
    ranks: tuple[RankSequenceTiming, ...],
    *,
    layer_count: int,
    workload_histogram: WorkloadHistogram,
) -> SequenceSummary:
    first = ranks[0]
    setup_blocking_ms = max(
        rank.setup_blocking_ms + rank.sync_overhang_ms for rank in ranks
    )
    layers_total_ms = max(rank.layers_total_ms for rank in ranks)
    layer_window_ms = max(rank.layer_window_ms for rank in ranks)
    sync_overhang_ms = max(rank.sync_overhang_ms for rank in ranks)
    end_to_end_ms = setup_blocking_ms + layer_window_ms
    return SequenceSummary(
        sequence_index=first.sequence_index,
        case_name=first.case_name,
        real_tokens=first.real_tokens,
        family_count=first.family_count,
        completion_count=first.completion_count,
        workload_histogram=workload_histogram,
        max_rank_setup_total_ms=max(rank.setup_total_ms for rank in ranks),
        max_rank_setup_blocking_ms=setup_blocking_ms,
        max_rank_plan_host_ms=max(rank.plan_host_ms for rank in ranks),
        max_rank_device_setup_sync_ms=max(rank.device_setup_sync_ms for rank in ranks),
        max_rank_layers_total_ms=layers_total_ms,
        max_rank_layer_window_ms=layer_window_ms,
        max_rank_sync_overhang_ms=sync_overhang_ms,
        max_rank_fwd_ms=max(rank.fwd_ms for rank in ranks),
        max_rank_bwd_ms=max(rank.bwd_ms for rank in ranks),
        max_rank_boundary_fwd_ms=max(rank.boundary_fwd_ms for rank in ranks),
        max_rank_boundary_bwd_ms=max(rank.boundary_bwd_ms for rank in ranks),
        max_rank_gdn_fwd_ms=max(rank.gdn_fwd_ms for rank in ranks),
        max_rank_gdn_bwd_ms=max(rank.gdn_bwd_ms for rank in ranks),
        max_rank_param_reduce_ms=max(rank.param_reduce_ms for rank in ranks),
        max_rank_cuda_gap_ms=max(rank.cuda_gap_ms for rank in ranks),
        max_rank_e2e_with_param_reduce_ms=max(
            rank.e2e_with_param_reduce_ms for rank in ranks
        ),
        max_rank_attention_tokens=max(rank.attention_tokens for rank in ranks),
        max_rank_gdn_tokens=max(rank.gdn_tokens for rank in ranks),
        max_local_prefix_bucket_count=max(
            rank.local_prefix_bucket_count for rank in ranks
        ),
        max_local_completion_bucket_count=max(
            rank.local_completion_bucket_count for rank in ranks
        ),
        max_chain_prefix_bucket_count=max(
            rank.chain_prefix_bucket_count for rank in ranks
        ),
        max_chain_completion_bucket_count=max(
            rank.chain_completion_bucket_count for rank in ranks
        ),
        max_parent_state_exchange_family_count=max(
            rank.parent_state_exchange_family_count for rank in ranks
        ),
        max_layout_cross_rank_token_count=max(
            rank.layout_cross_rank_token_count for rank in ranks
        ),
        max_layout_cross_rank_bytes_per_direction=max(
            rank.layout_cross_rank_bytes_per_direction for rank in ranks
        ),
        max_bucket_count=max(rank.bucket_count for rank in ranks),
        max_bucket_real_tokens=max(rank.bucket_real_tokens for rank in ranks),
        max_bucket_padded_tokens=max(rank.bucket_padded_tokens for rank in ranks),
        max_bucket_padding_ratio=max(rank.bucket_padding_ratio for rank in ranks),
        max_bucket_length=max(rank.max_bucket_length for rank in ranks),
        max_bucket_segments=max(rank.max_bucket_segments for rank in ranks),
        max_single_bucket_padding_ratio=max(
            rank.max_bucket_padding_ratio for rank in ranks
        ),
        max_prefix_bucket_real_tokens=max(
            rank.prefix_bucket_real_tokens for rank in ranks
        ),
        max_prefix_bucket_padded_tokens=max(
            rank.prefix_bucket_padded_tokens for rank in ranks
        ),
        max_prefix_bucket_padding_ratio=max(
            rank.prefix_bucket_padding_ratio for rank in ranks
        ),
        max_completion_bucket_real_tokens=max(
            rank.completion_bucket_real_tokens for rank in ranks
        ),
        max_completion_bucket_padded_tokens=max(
            rank.completion_bucket_padded_tokens for rank in ranks
        ),
        max_completion_bucket_padding_ratio=max(
            rank.completion_bucket_padding_ratio for rank in ranks
        ),
        max_chain_bucket_real_tokens=max(
            rank.chain_bucket_real_tokens for rank in ranks
        ),
        max_chain_bucket_padded_tokens=max(
            rank.chain_bucket_padded_tokens for rank in ranks
        ),
        max_chain_bucket_padding_ratio=max(
            rank.chain_bucket_padding_ratio for rank in ranks
        ),
        end_to_end_ms=end_to_end_ms,
        end_to_end_per_layer_ms=end_to_end_ms / layer_count,
        layer_window_per_layer_ms=layer_window_ms / layer_count,
        sync_overhang_per_layer_ms=sync_overhang_ms / layer_count,
        tokens_per_second=1000.0 * first.real_tokens / max(end_to_end_ms, 1e-9),
        ranks=ranks,
    )


def _aggregate_result(
    *,
    args: argparse.Namespace,
    sequences: tuple[SequenceSummary, ...],
) -> StackedGdnProxyResult:
    tail_count = min(args.tail_window, len(sequences))
    return StackedGdnProxyResult(
        cp_size=args.cp_size,
        dtype=str(BENCHMARK_DTYPE),
        workload_name=args.workload.name,
        architecture=args.layer_schedule.name,
        gdn_module_config=args.gdn_module,
        gdn_linear_policy=str(args.gdn_linear_policy),
        cp_attention_layout=str(args.cp_attention_layout),
        model_layer_count=args.layer_schedule.model_layer_count,
        gdn_layer_count=args.layer_schedule.gdn_layer_count,
        attention_layer_count=args.layer_schedule.attention_layer_count,
        gdn_group_lengths=args.layer_schedule.gdn_group_lengths,
        layer_types=args.layer_schedule.layer_types,
        sequence_length=args.target_seq_len,
        prefix_length_mode=args.prefix_length_mode,
        num_sequences=args.num_sequences,
        tail_window=args.tail_window,
        all_sequences_median=_rollup(sequences),
        tail_sequences_median=_rollup(sequences[-tail_count:]),
        sequences=sequences,
    )


def _rollup(sequences: tuple[SequenceSummary, ...]) -> StackedRollup:
    if not sequences:
        return StackedRollup(
            sequence_count=0,
            setup_total_ms=0.0,
            setup_blocking_ms=0.0,
            plan_host_ms=0.0,
            device_setup_sync_ms=0.0,
            layers_total_ms=0.0,
            layer_window_ms=0.0,
            layer_window_per_layer_ms=0.0,
            sync_overhang_ms=0.0,
            sync_overhang_per_layer_ms=0.0,
            fwd_ms=0.0,
            bwd_ms=0.0,
            boundary_fwd_ms=0.0,
            boundary_bwd_ms=0.0,
            gdn_fwd_ms=0.0,
            gdn_bwd_ms=0.0,
            param_reduce_ms=0.0,
            cuda_gap_ms=0.0,
            end_to_end_ms=0.0,
            end_to_end_per_layer_ms=0.0,
            tokens_per_second=0.0,
            layout_cross_rank_token_count=0.0,
            layout_cross_rank_bytes_per_direction=0.0,
            bucket_count=0.0,
            bucket_real_tokens=0.0,
            bucket_padded_tokens=0.0,
            bucket_padding_ratio=0.0,
            max_bucket_length=0.0,
            max_bucket_segments=0.0,
            max_single_bucket_padding_ratio=0.0,
            prefix_bucket_padded_tokens=0.0,
            prefix_bucket_padding_ratio=0.0,
            completion_bucket_padded_tokens=0.0,
            completion_bucket_padding_ratio=0.0,
            chain_bucket_padded_tokens=0.0,
            chain_bucket_padding_ratio=0.0,
        )

    def median_of(field: str) -> float:
        return float(statistics.median(float(getattr(row, field)) for row in sequences))

    return StackedRollup(
        sequence_count=len(sequences),
        setup_total_ms=median_of("max_rank_setup_total_ms"),
        setup_blocking_ms=median_of("max_rank_setup_blocking_ms"),
        plan_host_ms=median_of("max_rank_plan_host_ms"),
        device_setup_sync_ms=median_of("max_rank_device_setup_sync_ms"),
        layers_total_ms=median_of("max_rank_layers_total_ms"),
        layer_window_ms=median_of("max_rank_layer_window_ms"),
        layer_window_per_layer_ms=median_of("layer_window_per_layer_ms"),
        sync_overhang_ms=median_of("max_rank_sync_overhang_ms"),
        sync_overhang_per_layer_ms=median_of("sync_overhang_per_layer_ms"),
        fwd_ms=median_of("max_rank_fwd_ms"),
        bwd_ms=median_of("max_rank_bwd_ms"),
        boundary_fwd_ms=median_of("max_rank_boundary_fwd_ms"),
        boundary_bwd_ms=median_of("max_rank_boundary_bwd_ms"),
        gdn_fwd_ms=median_of("max_rank_gdn_fwd_ms"),
        gdn_bwd_ms=median_of("max_rank_gdn_bwd_ms"),
        param_reduce_ms=median_of("max_rank_param_reduce_ms"),
        cuda_gap_ms=median_of("max_rank_cuda_gap_ms"),
        end_to_end_ms=median_of("end_to_end_ms"),
        end_to_end_per_layer_ms=median_of("end_to_end_per_layer_ms"),
        tokens_per_second=median_of("tokens_per_second"),
        layout_cross_rank_token_count=median_of("max_layout_cross_rank_token_count"),
        layout_cross_rank_bytes_per_direction=median_of(
            "max_layout_cross_rank_bytes_per_direction"
        ),
        bucket_count=median_of("max_bucket_count"),
        bucket_real_tokens=median_of("max_bucket_real_tokens"),
        bucket_padded_tokens=median_of("max_bucket_padded_tokens"),
        bucket_padding_ratio=median_of("max_bucket_padding_ratio"),
        max_bucket_length=median_of("max_bucket_length"),
        max_bucket_segments=median_of("max_bucket_segments"),
        max_single_bucket_padding_ratio=median_of("max_single_bucket_padding_ratio"),
        prefix_bucket_padded_tokens=median_of("max_prefix_bucket_padded_tokens"),
        prefix_bucket_padding_ratio=median_of("max_prefix_bucket_padding_ratio"),
        completion_bucket_padded_tokens=median_of(
            "max_completion_bucket_padded_tokens"
        ),
        completion_bucket_padding_ratio=median_of(
            "max_completion_bucket_padding_ratio"
        ),
        chain_bucket_padded_tokens=median_of("max_chain_bucket_padded_tokens"),
        chain_bucket_padding_ratio=median_of("max_chain_bucket_padding_ratio"),
    )


def _build_sequence_case(
    *,
    args: argparse.Namespace,
    sequence_index: int,
) -> GdnPhase0Case:
    if args.case_name:
        case_args = argparse.Namespace(
            case_name=args.case_name,
            conv_width=args.conv_width,
            target_seq_len=args.target_seq_len,
            prefix_len=args.prefix_len,
            suffix_len=args.suffix_len,
            completions_per_family=args.completions_per_family,
        )
        case = _selected_or_repeated_case(case_args)
        return case.model_copy(
            update={
                "name": f"{case.name}_seq{sequence_index}",
                "seed": int(args.seed) + sequence_index * 97,
            }
        )
    workload: StackedWorkloadConfig = args.workload
    rng = random.Random(int(args.seed) + sequence_index * 97)
    families: list[GdnFamilyShape] = []
    used = 0
    if workload.family_pattern == "dominant_with_background":
        dominant = _sample_family(
            workload=workload, rng=rng, prefix_mode=args.prefix_length_mode
        )
        used = _append_family_if_it_fits(
            families=families,
            family=dominant,
            used=used,
            target_seq_len=args.target_seq_len,
        )
        workload = _background_workload(workload)
    while True:
        family = _sample_family(
            workload=workload,
            rng=rng,
            prefix_mode=args.prefix_length_mode,
        )
        fitted = fit_gdn_family_to_remaining(family, int(args.target_seq_len) - used)
        if fitted is None:
            if families:
                break
            raise ValueError(
                f"workload {workload.name!r} cannot fit one prefix plus completion in target_seq_len={args.target_seq_len}"
            )
        families.append(fitted)
        used += gdn_family_token_count(fitted)
        if len(fitted.suffix_lengths) != len(family.suffix_lengths):
            break
    return GdnPhase0Case(
        name=f"{workload.name}_seq{sequence_index}",
        sequence_length=args.target_seq_len,
        rows=(GdnPackedRowShape(families=tuple(families)),),
        seed=int(args.seed) + sequence_index * 97,
        description=workload.description,
    )


def _sequence_case_name(args: argparse.Namespace, sequence_index: int) -> str:
    if args.case_name:
        return f"case_{args.case_name}_seq{sequence_index}"
    return f"{args.workload.name}_seq{sequence_index}"


def _sample_length(
    *,
    mean: int,
    std: int,
    clip_delta: int,
    mode: str,
    rng: random.Random,
    min_value: int = 1,
) -> int:
    if mode == "fixed" or std == 0 or clip_delta == 0:
        return max(min_value, int(mean))
    lower = max(int(min_value), int(mean) - int(clip_delta))
    upper = max(lower, int(mean) + int(clip_delta))
    sampled = int(round(rng.gauss(mu=float(mean), sigma=float(std))))
    return max(lower, min(upper, sampled))


def _sample_family(
    *,
    workload: StackedWorkloadConfig,
    rng: random.Random,
    prefix_mode: str,
) -> GdnFamilyShape:
    prefix = _sample_length(
        mean=workload.prefix_length_mean,
        std=workload.prefix_length_std,
        clip_delta=workload.prefix_length_clip_delta,
        mode=prefix_mode,
        rng=rng,
    )
    suffixes = tuple(
        _sample_length(
            mean=workload.branch_length_mean,
            std=workload.branch_length_std,
            clip_delta=workload.branch_length_clip_delta,
            mode="clipped_normal",
            rng=rng,
            min_value=2,
        )
        for _ in range(workload.branches_per_prefix)
    )
    return GdnFamilyShape(prefix_length=prefix, suffix_lengths=suffixes)


def _append_family_if_it_fits(
    *,
    families: list[GdnFamilyShape],
    family: GdnFamilyShape,
    used: int,
    target_seq_len: int,
) -> int:
    fitted = fit_gdn_family_to_remaining(family, int(target_seq_len) - used)
    if fitted is None:
        raise ValueError(
            "dominant family requires at least one prefix plus completion within "
            f"target_seq_len={target_seq_len}"
        )
    families.append(fitted)
    return used + gdn_family_token_count(fitted)


def _background_workload(workload: StackedWorkloadConfig) -> StackedWorkloadConfig:
    return workload.model_copy(
        update={
            "family_pattern": "uniform",
            "prefix_length_mean": workload.background_prefix_length_mean or 512,
            "prefix_length_std": workload.background_prefix_length_std or 64,
            "prefix_length_clip_delta": workload.background_prefix_length_clip_delta
            or 128,
            "branch_length_mean": workload.background_branch_length_mean or 64,
            "branch_length_std": workload.background_branch_length_std or 16,
            "branch_length_clip_delta": workload.background_branch_length_clip_delta
            or 32,
            "branches_per_prefix": workload.background_branches_per_prefix or 4,
        }
    )


def _workload_histogram(case: GdnPhase0Case) -> WorkloadHistogram:
    prefixes = [family.prefix_length for row in case.rows for family in row.families]
    suffixes = [
        suffix
        for row in case.rows
        for family in row.families
        for suffix in family.suffix_lengths
    ]
    return WorkloadHistogram(
        prefix_min=min(prefixes, default=0),
        prefix_max=max(prefixes, default=0),
        prefix_mean=float(statistics.mean(prefixes)) if prefixes else 0.0,
        suffix_min=min(suffixes, default=0),
        suffix_max=max(suffixes, default=0),
        suffix_mean=float(statistics.mean(suffixes)) if suffixes else 0.0,
    )


def _selected_workloads(args: argparse.Namespace) -> tuple[StackedWorkloadConfig, ...]:
    if args.case_name:
        return (
            StackedWorkloadConfig(
                name=f"case_{args.case_name}",
                prefix_length_mode=str(args.prefix_length_mode or "fixed"),
                base_target_seq_len=int(args.target_seq_len or 40960),
                prefix_length_mean=int(args.prefix_len or 5000),
                prefix_length_std=int(args.prefix_length_std or 0),
                prefix_length_clip_delta=int(args.prefix_length_clip_delta or 0),
                branch_length_mean=int(args.suffix_len or 100),
                branch_length_std=int(args.branch_length_std or 0),
                branch_length_clip_delta=int(args.branch_length_clip_delta or 0),
                branches_per_prefix=int(args.completions_per_family or 16),
                description="Deterministic case-name mode.",
            ),
        )
    available = _workload_matrix()
    names = [name.strip() for name in str(args.workloads).split(",") if name.strip()]
    if names == ["all"]:
        return tuple(available.values())
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(
            f"unknown workload(s) {missing}; expected one of: "
            f"{', '.join((*available.keys(), 'all'))}"
        )
    return tuple(available[name] for name in names)


def _workload_matrix() -> dict[str, StackedWorkloadConfig]:
    return {
        "fixed_5k_16x100": StackedWorkloadConfig(
            name="fixed_5k_16x100",
            prefix_length_mode="fixed",
            base_target_seq_len=40960,
            prefix_length_mean=5000,
            prefix_length_std=0,
            prefix_length_clip_delta=0,
            branch_length_mean=100,
            branch_length_std=0,
            branch_length_clip_delta=0,
            branches_per_prefix=16,
            description="Fixed repeated 5k prefix plus 16x100 completions, complete families only.",
        ),
        "default_5k_16x100": StackedWorkloadConfig(
            name="default_5k_16x100",
            prefix_length_mode="fixed",
            base_target_seq_len=40960,
            prefix_length_mean=5000,
            prefix_length_std=512,
            prefix_length_clip_delta=1024,
            branch_length_mean=100,
            branch_length_std=32,
            branch_length_clip_delta=64,
            branches_per_prefix=16,
            description="Attention-benchmark default: 5k prefix, 16 completions near 100 tokens.",
        ),
        "varied_5k_16x100": StackedWorkloadConfig(
            name="varied_5k_16x100",
            prefix_length_mode="clipped_normal",
            base_target_seq_len=40960,
            prefix_length_mean=5000,
            prefix_length_std=512,
            prefix_length_clip_delta=1024,
            branch_length_mean=100,
            branch_length_std=32,
            branch_length_clip_delta=64,
            branches_per_prefix=16,
            description="Varied 5k plus 16x100 workload with jittered prefix and completion lengths.",
        ),
        "many_small_64_4x16": StackedWorkloadConfig(
            name="many_small_64_4x16",
            prefix_length_mode="clipped_normal",
            base_target_seq_len=40960,
            prefix_length_mean=64,
            prefix_length_std=7,
            prefix_length_clip_delta=13,
            branch_length_mean=16,
            branch_length_std=5,
            branch_length_clip_delta=10,
            branches_per_prefix=4,
            description="Many small prompt families, kept on the backburner but selectable.",
        ),
        "varied_dominant_14745_16x921": StackedWorkloadConfig(
            name="varied_dominant_14745_16x921",
            prefix_length_mode="clipped_normal",
            family_pattern="dominant_with_background",
            base_target_seq_len=40960,
            prefix_length_mean=14745,
            prefix_length_std=1024,
            prefix_length_clip_delta=2048,
            branch_length_mean=921,
            branch_length_std=256,
            branch_length_clip_delta=512,
            branches_per_prefix=16,
            background_prefix_length_mean=512,
            background_prefix_length_std=64,
            background_prefix_length_clip_delta=128,
            background_branch_length_mean=64,
            background_branch_length_std=16,
            background_branch_length_clip_delta=32,
            background_branches_per_prefix=4,
            description="One sampled dominant long family with sampled smaller background families.",
        ),
        "long_8k_16x8k": StackedWorkloadConfig(
            name="long_8k_16x8k",
            prefix_length_mode="fixed",
            base_target_seq_len=147456,
            prefix_length_mean=8192,
            prefix_length_std=512,
            prefix_length_clip_delta=1024,
            branch_length_mean=8192,
            branch_length_std=512,
            branch_length_clip_delta=1024,
            branches_per_prefix=16,
            description="Long-branch 8k plus 16x8k workload.",
        ),
        "long_64k_8x64k": StackedWorkloadConfig(
            name="long_64k_8x64k",
            prefix_length_mode="fixed",
            base_target_seq_len=600000,
            prefix_length_mean=65536,
            prefix_length_std=1024,
            prefix_length_clip_delta=2048,
            branch_length_mean=65536,
            branch_length_std=1024,
            branch_length_clip_delta=2048,
            branches_per_prefix=8,
            description="Very long 64k plus 8x64k workload.",
        ),
    }


def _args_for_run(
    args: argparse.Namespace,
    workload: StackedWorkloadConfig,
    cp_size: int,
) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.workload = workload
    run_args.cp_size = cp_size
    run_args.target_seq_len = int(args.target_seq_len or workload.base_target_seq_len)
    run_args.target_seq_len *= cp_size
    run_args.prefix_len = int(args.prefix_len or workload.prefix_length_mean)
    run_args.suffix_len = int(args.suffix_len or workload.branch_length_mean)
    run_args.completions_per_family = int(
        args.completions_per_family or workload.branches_per_prefix
    )
    if args.prefix_length_std is not None:
        workload = workload.model_copy(
            update={"prefix_length_std": int(args.prefix_length_std)}
        )
    if args.prefix_length_clip_delta is not None:
        workload = workload.model_copy(
            update={"prefix_length_clip_delta": int(args.prefix_length_clip_delta)}
        )
    if args.branch_length_std is not None:
        workload = workload.model_copy(
            update={"branch_length_std": int(args.branch_length_std)}
        )
    if args.branch_length_clip_delta is not None:
        workload = workload.model_copy(
            update={"branch_length_clip_delta": int(args.branch_length_clip_delta)}
        )
    if args.prefix_len is not None:
        workload = workload.model_copy(
            update={"prefix_length_mean": int(args.prefix_len)}
        )
    if args.suffix_len is not None:
        workload = workload.model_copy(
            update={"branch_length_mean": int(args.suffix_len)}
        )
    if args.completions_per_family is not None:
        workload = workload.model_copy(
            update={"branches_per_prefix": int(args.completions_per_family)}
        )
    run_args.prefix_length_mode = str(
        args.prefix_length_mode or workload.prefix_length_mode
    )
    run_args.workload = workload
    return run_args


def _dist_barrier() -> None:
    if (
        not torch.distributed.is_available()  # ty: ignore[possibly-missing-attribute]
        or not torch.distributed.is_initialized()  # ty: ignore[possibly-missing-attribute]
        or torch.distributed.get_world_size() <= 1  # ty: ignore[possibly-missing-attribute]
    ):
        return
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])  # ty: ignore[possibly-missing-attribute]


def _group_global_rank(group: Any | None, group_rank: int) -> int:
    if group is None:
        return group_rank
    try:
        return int(
            torch.distributed.get_global_rank(  # ty: ignore[possibly-missing-attribute]
                group, group_rank
            )
        )
    except Exception:
        ranks = torch.distributed.get_process_group_ranks(  # ty: ignore[possibly-missing-attribute]
            group
        )
        return int(ranks[group_rank])


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _empty_nonzero_rank_result(args: argparse.Namespace) -> StackedGdnProxyResult:
    empty = _rollup(())
    return StackedGdnProxyResult(
        cp_size=args.cp_size,
        dtype=str(BENCHMARK_DTYPE),
        workload_name=args.workload.name,
        architecture=args.layer_schedule.name,
        gdn_module_config=args.gdn_module,
        gdn_linear_policy=str(args.gdn_linear_policy),
        cp_attention_layout=str(args.cp_attention_layout),
        model_layer_count=args.layer_schedule.model_layer_count,
        gdn_layer_count=args.layer_schedule.gdn_layer_count,
        attention_layer_count=args.layer_schedule.attention_layer_count,
        gdn_group_lengths=args.layer_schedule.gdn_group_lengths,
        layer_types=args.layer_schedule.layer_types,
        sequence_length=args.target_seq_len,
        prefix_length_mode=args.prefix_length_mode,
        num_sequences=args.num_sequences,
        tail_window=args.tail_window,
        all_sequences_median=empty,
        tail_sequences_median=empty,
        sequences=(),
    )


def _write_progress(
    run_dir: Path,
    args: argparse.Namespace,
    sequences: tuple[SequenceSummary, ...],
    *,
    is_final: bool,
) -> None:
    payload = {
        "config": _run_config(args),
        "completed_sequences": len(sequences),
        "is_final": is_final,
        "summary": {
            "all_sequences_median": _rollup(sequences).model_dump(),
            "tail_sequences_median": _rollup(
                sequences[-min(args.tail_window, len(sequences)) :]
            ).model_dump(),
        },
        "sequences": [sequence.model_dump() for sequence in sequences],
    }
    (run_dir / "progress.json").write_text(json.dumps(payload, indent=2) + "\n")


def _manifest_configs(
    args: argparse.Namespace,
    workloads: tuple[StackedWorkloadConfig, ...],
) -> dict[str, object]:
    return {
        "cp_sizes": args.cp_sizes,
        "requested_layers": args.layers,
        "layer_schedule": args.layer_schedule.model_dump(),
        "gdn_module": args.gdn_module.model_dump(),
        "gdn_linear_policy": str(args.gdn_linear_policy),
        "cp_attention_layout": str(args.cp_attention_layout),
        "num_sequences": args.num_sequences,
        "tail_window": args.tail_window,
        "workloads": [workload.model_dump() for workload in workloads],
        "case_name": args.case_name,
        "prefix_length_mode_override": args.prefix_length_mode,
        "base_cp1_target_seq_len": args.target_seq_len,
        "cp_target_seq_len_rule": (
            "effective_target_seq_len = base_cp1_target_seq_len * cp_size; "
            "per-family prefix/completion lengths stay fixed and additional "
            "families are packed to target"
        ),
        "overlap_next_state_prep": args.overlap_next_state_prep,
        "activation_checkpoint_gdn": args.activation_checkpoint_gdn,
        "profile": args.profile,
        "layer_execution_pattern": "attention_style_independent_fwd_bwd",
        "benchmark_dtype": str(BENCHMARK_DTYPE),
        "rank_torch_num_threads": torch.get_num_threads(),
        "planner_config": GdnPlannerConfig().model_dump(),
    }


def _run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cp_size": args.cp_size,
        "workload": args.workload.model_dump(),
        "layer_schedule": args.layer_schedule.model_dump(),
        "gdn_module": args.gdn_module.model_dump(),
        "gdn_linear_policy": str(args.gdn_linear_policy),
        "cp_attention_layout": str(args.cp_attention_layout),
        "sequence_length": args.target_seq_len,
        "num_sequences": args.num_sequences,
        "tail_window": args.tail_window,
        "prefix_length_mode": args.prefix_length_mode,
        "overlap_next_state_prep": args.overlap_next_state_prep,
        "activation_checkpoint_gdn": args.activation_checkpoint_gdn,
        "profile": args.profile,
        "layer_execution_pattern": "attention_style_independent_fwd_bwd",
        "benchmark_dtype": str(BENCHMARK_DTYPE),
        "rank_torch_num_threads": torch.get_num_threads(),
    }


def _render_report(results: tuple[StackedGdnProxyResult, ...]) -> str:
    lines = [
        "# Stacked Packed Shared-Prefix GDN Training Proxy Benchmark",
        "",
        "| workload | CP | dtype | linear policy | CP attention layout | arch | GDN dims | model layers | GDN layers | GDN groups | seq len | sequences | tail n | xrank layout tok | xrank layout MiB/dir | setup block ms | layer window/GDN layer ms | sync overhang/GDN layer ms | e2e/GDN layer ms | tok/s |",
        "|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        tail = result.tail_sequences_median
        lines.append(
            f"| {result.workload_name} | {result.cp_size} | {result.dtype} | "
            f"{result.gdn_linear_policy} | {result.cp_attention_layout} | "
            f"{result.architecture} | "
            f"{result.gdn_module_config.name}:{result.gdn_module_config.hidden_size} | "
            f"{result.model_layer_count} | {result.gdn_layer_count} | "
            f"{len(result.gdn_group_lengths)} | {result.sequence_length} | "
            f"{result.num_sequences} | "
            f"{tail.sequence_count} | "
            f"{tail.layout_cross_rank_token_count:.0f} | "
            f"{tail.layout_cross_rank_bytes_per_direction / (1024 * 1024):.1f} | "
            f"{tail.setup_blocking_ms:.3f} | "
            f"{tail.layer_window_per_layer_ms:.3f} | "
            f"{tail.sync_overhang_per_layer_ms:.3f} | "
            f"{tail.end_to_end_per_layer_ms:.3f} | "
            f"{tail.tokens_per_second:.0f} |"
        )
    lines.extend(
        [
            "",
            "| workload | CP | plan host ms | setup total ms | device setup sync ms | fwd ms | bwd ms | layers total ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        tail = result.tail_sequences_median
        lines.append(
            f"| {result.workload_name} | {result.cp_size} | "
            f"{tail.plan_host_ms:.3f} | {tail.setup_total_ms:.3f} | "
            f"{tail.device_setup_sync_ms:.3f} | {tail.fwd_ms:.3f} | "
            f"{tail.bwd_ms:.3f} | {tail.layers_total_ms:.3f} |"
        )
    lines.extend(
        [
            "",
            "| workload | CP | boundary fwd ms | boundary bwd ms | GDN fwd ms | GDN bwd ms | param reduce ms | CUDA gap ms | layer window ms | host overhang ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        tail = result.tail_sequences_median
        lines.append(
            f"| {result.workload_name} | {result.cp_size} | "
            f"{tail.boundary_fwd_ms:.3f} | {tail.boundary_bwd_ms:.3f} | "
            f"{tail.gdn_fwd_ms:.3f} | {tail.gdn_bwd_ms:.3f} | "
            f"{tail.param_reduce_ms:.3f} | {tail.cuda_gap_ms:.3f} | "
            f"{tail.layer_window_ms:.3f} | {tail.sync_overhang_ms:.3f} |"
        )
    lines.extend(
        [
            "",
            "| workload | CP | boundary/GDN layer ms | GDN/GDN layer ms | reduce/GDN layer ms | CUDA gap/GDN layer ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        tail = result.tail_sequences_median
        layer_count = float(result.gdn_layer_count)
        boundary_total = tail.boundary_fwd_ms + tail.boundary_bwd_ms
        gdn_total = tail.gdn_fwd_ms + tail.gdn_bwd_ms
        lines.append(
            f"| {result.workload_name} | {result.cp_size} | "
            f"{boundary_total / layer_count:.3f} | "
            f"{gdn_total / layer_count:.3f} | "
            f"{tail.param_reduce_ms / layer_count:.3f} | "
            f"{tail.cuda_gap_ms / layer_count:.3f} |"
        )
    lines.extend(
        [
            "",
            "| workload | CP | buckets | bucket real tok | bucket padded tok | pad x | max len | max seg | max bucket pad x | prefix padded tok | prefix pad x | completion padded tok | completion pad x | chain padded tok | chain pad x |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        tail = result.tail_sequences_median
        lines.append(
            f"| {result.workload_name} | {result.cp_size} | "
            f"{tail.bucket_count:.0f} | "
            f"{tail.bucket_real_tokens:.0f} | "
            f"{tail.bucket_padded_tokens:.0f} | "
            f"{tail.bucket_padding_ratio:.3f} | "
            f"{tail.max_bucket_length:.0f} | "
            f"{tail.max_bucket_segments:.0f} | "
            f"{tail.max_single_bucket_padding_ratio:.3f} | "
            f"{tail.prefix_bucket_padded_tokens:.0f} | "
            f"{tail.prefix_bucket_padding_ratio:.3f} | "
            f"{tail.completion_bucket_padded_tokens:.0f} | "
            f"{tail.completion_bucket_padding_ratio:.3f} | "
            f"{tail.chain_bucket_padded_tokens:.0f} | "
            f"{tail.chain_bucket_padding_ratio:.3f} |"
        )
    lines.extend(
        [
            "",
            "The benchmark follows the attention CP training proxy shape: a stream of packed sequences, repeated independent layer fwd/bwd calls, max-rank sequence records, and tail-window medians.",
            "By default it uses the Qwen3.5-35B-A3B text schedule: 40 model layers with 30 GDN/linear-attention layers in ten groups of three, separated by 10 full-attention boundaries. This branch does not execute full attention CP in this benchmark, so per-layer metrics are normalized by executed GDN layer count.",
            "The default GDN module uses Qwen3.5-35B-A3B GDN-relevant dimensions: hidden size 2048, 16 linear key heads, 32 linear value heads, 128-dimensional GDN keys/values, and convolution width 4. The stacked proxy reuses one representative GDN module across executed GDN applications to keep long-sequence activation and CP timing measurable without adding parameter-footprint pressure that is orthogonal to the GDN sequence path.",
            "By default --gdn-linear-policy=noop replaces GDN in/out projection modules inside this benchmark only, so reported times isolate the shared-prefix GDN recurrence/layout/setup path. Use --gdn-linear-policy=real for a full layer-style projection timing.",
            "Each counted GDN layer receives a fresh detached input and runs backward immediately, matching the stacked attention proxy rather than retaining activations through a full model stack. Activation checkpointing is disabled because there is no cross-layer autograd graph in this benchmark.",
            "Target sequence length is weak-scaled by adding more fixed-shape families; the final family may use fewer completions to fit the target.",
            "Distributed GDN token exchange, parent-state exchange, native FLA CP scans, and parameter-gradient all-reduce use Megatron's context-parallel process group.",
            "CP token layout conversion is charged at Qwen3.5 GDN/full-attention boundaries: attention layout to GDN layout once per contiguous GDN group, GDN layout reused by every layer in that group, then GDN layout back to attention layout once at the next full-attention boundary.",
            "`--cp-attention-layout=planner_default` lets the GDN planner pick its low-exchange rank ownership. `reversed_striped` reverses CP-sized chunk assignment order and `randomized_cp_chunks` shuffles those chunks to check layout sensitivity without relying on token-list ownership.",
            "GDN planning is built once per packed sequence. Setup blocking includes any exposed next-sequence prep that appears as sync overhang after the current layer-window event, so e2e is layer-window plus blocking setup without dropping that training gap.",
            "The default workload keeps prefixes fixed and samples completion lengths, matching the attention proxy contract. Select `varied_5k_16x100`, `varied_dominant_14745_16x921`, or `--prefix-length-mode clipped_normal` to sample prefixes.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
