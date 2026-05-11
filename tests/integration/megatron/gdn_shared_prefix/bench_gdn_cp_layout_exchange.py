from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import socket
import sys
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import torch
from torch.distributed import barrier, destroy_process_group, init_process_group
import torch.multiprocessing as mp

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.context_parallel.runtime import _normalized_chunk_size
from art.megatron.context_parallel.types import ContextParallelConfig
from art.megatron.gdn.layout import (
    GdnCpLayoutPlan,
    build_gdn_cp_layout_plan,
    exchange_rank_tensor_all_to_all,
)

from .artifacts import write_manifest
from .benchmark_gdn import qwen35_gdn_module_config
from .cases import (
    GdnFamilyShape,
    GdnPackedRowShape,
    GdnPhase0Case,
    fit_gdn_family_to_remaining,
    gdn_family_token_count,
)
from .packed_layout import build_gdn_group_parent_tensors
from .parser_import import parse_gdn_shared_prefix_segments

BENCHMARK_DTYPE = torch.bfloat16

_NVTX_RANGES = (
    "art_gdn_cp_layout_plan",
    "art_gdn_cp_attention_to_gdn_exchange",
    "art_gdn_cp_exchange_backward",
)


class TimingSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    median_ms: float
    p90_ms: float
    max_ms: float
    raw_ms: tuple[float, ...]


class RankExchangeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=0)
    attention_tokens: int = Field(ge=0)
    gdn_tokens: int = Field(ge=0)
    forward_ms: TimingSummary
    backward_ms: TimingSummary
    e2e_ms: TimingSummary


class CpLayoutExchangeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    cp_size: int = Field(ge=1)
    backend: str
    device_type: str
    dtype: str
    hidden_size: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    real_tokens: int = Field(ge=1)
    family_count: int = Field(ge=1)
    completion_count: int = Field(ge=1)
    warmup_iters: int = Field(ge=0)
    timed_iters: int = Field(ge=1)
    plan_build_ms: TimingSummary
    cross_rank_token_count: int = Field(ge=0)
    cross_rank_bytes_per_direction: int = Field(ge=0)
    packed_buffer_bytes_per_direction: int = Field(ge=0)
    max_rank_forward_ms: float
    max_rank_backward_ms: float
    max_rank_e2e_ms: float
    rank_results: tuple[RankExchangeResult, ...]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark GDN CP layout exchange")
    parser.add_argument("--cp-sizes", default="2,4")
    parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--target-seq-len", type=int, default=40960)
    parser.add_argument("--prefix-len", type=int, default=5000)
    parser.add_argument("--suffix-len", type=int, default=100)
    parser.add_argument("--completions-per-family", type=int, default=16)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.hidden_size = int(args.hidden_size or qwen35_gdn_module_config().hidden_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for cp_size in tuple(int(value) for value in args.cp_sizes.split(",") if value):
        run_args = argparse.Namespace(**vars(args))
        run_args.target_seq_len = args.target_seq_len * cp_size
        case = _repeated_family_case(
            target_seq_len=run_args.target_seq_len,
            prefix_len=run_args.prefix_len,
            suffix_len=run_args.suffix_len,
            completions_per_family=run_args.completions_per_family,
        )
        tensors = build_gdn_group_parent_tensors(case)
        backend = _select_backend(args.backend, cp_size)
        result = _run_cp_size(run_args, case, tensors, cp_size=cp_size, backend=backend)
        results.append(result)
        print(result.model_dump_json(), flush=True)

    _write_outputs(args.output_dir, tuple(results))
    manifest = write_manifest(
        args.output_dir,
        kind="gdn_cp_layout_exchange_benchmark",
        command=sys.argv,
        configs=_manifest_configs(args),
        cases=tuple(result.model_dump() for result in results),
        caveats=(
            "Layout exchange benchmark only; no GDN recurrence kernels are executed.",
            "Planning is measured separately and should run once per training sequence.",
            "Timings include explicit synchronization/barriers to expose communication.",
        ),
    )
    print(json.dumps({"manifest": str(manifest)}), flush=True)
    return 0


def _run_cp_size(
    args: argparse.Namespace,
    case: GdnPhase0Case,
    tensors: dict[str, Any],
    *,
    cp_size: int,
    backend: str,
) -> CpLayoutExchangeResult:
    plan_build_ms = _measure_plan_build(
        tensors,
        cp_size=cp_size,
        iters=args.iters,
    )
    port = _find_free_port()
    run_dir = args.output_dir / f"cp{cp_size}_{backend}"
    run_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _distributed_worker,
        args=(
            cp_size,
            backend,
            port,
            args.hidden_size,
            args.warmup_iters,
            args.iters,
            case.model_dump(),
            str(run_dir),
        ),
        nprocs=cp_size,
        join=True,
    )
    rank_results = tuple(
        RankExchangeResult.model_validate_json(
            (run_dir / f"rank_{rank}.json").read_text()
        )
        for rank in range(cp_size)
    )
    plan = _layout_plan(tensors, cp_size=cp_size)
    spec = parse_gdn_shared_prefix_segments(
        tensors["group_ids"], tensors["parent_ids"], min_completions_per_family=0
    )
    dtype = str(BENCHMARK_DTYPE)
    element_size = torch.tensor((), dtype=BENCHMARK_DTYPE).element_size()
    cross_rank_tokens = plan.attention_to_gdn.cross_rank_token_count
    return CpLayoutExchangeResult(
        cp_size=cp_size,
        backend=backend,
        device_type="cuda" if backend == "nccl" else "cpu",
        dtype=dtype,
        hidden_size=args.hidden_size,
        sequence_length=case.sequence_length,
        real_tokens=spec.real_token_count,
        family_count=spec.family_count,
        completion_count=spec.completion_count,
        warmup_iters=args.warmup_iters,
        timed_iters=args.iters,
        plan_build_ms=plan_build_ms,
        cross_rank_token_count=cross_rank_tokens,
        cross_rank_bytes_per_direction=cross_rank_tokens
        * args.hidden_size
        * element_size,
        packed_buffer_bytes_per_direction=spec.real_token_count
        * args.hidden_size
        * element_size,
        max_rank_forward_ms=max(result.forward_ms.median_ms for result in rank_results),
        max_rank_backward_ms=max(
            result.backward_ms.median_ms for result in rank_results
        ),
        max_rank_e2e_ms=max(result.e2e_ms.median_ms for result in rank_results),
        rank_results=rank_results,
    )


def _distributed_worker(
    rank: int,
    cp_size: int,
    backend: str,
    port: int,
    hidden_size: int,
    warmup_iters: int,
    iters: int,
    case_dump: dict[str, Any],
    run_dir: str,
) -> None:
    if backend == "nccl":
        torch.cuda.set_device(rank)
    init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=cp_size,
    )
    try:
        case = GdnPhase0Case.model_validate(case_dump)
        tensors = build_gdn_group_parent_tensors(case)
        plan = _layout_plan(tensors, cp_size=cp_size)
        device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
        generator = torch.Generator(device=device).manual_seed(20400426 + rank)
        local_template = torch.randn(
            plan.attention_to_gdn.source_token_counts_by_rank[rank],
            hidden_size,
            device=device,
            dtype=BENCHMARK_DTYPE,
            generator=generator,
        )
        for _ in range(warmup_iters):
            _time_exchange_iteration(local_template, plan, rank=rank, backward=True)
        forward_ms = []
        backward_ms = []
        e2e_ms = []
        for _ in range(iters):
            forward_ms.append(
                _time_exchange_iteration(
                    local_template, plan, rank=rank, backward=False
                )
            )
            backward_ms.append(_time_exchange_backward(local_template, plan, rank=rank))
            e2e_ms.append(
                _time_exchange_iteration(local_template, plan, rank=rank, backward=True)
            )
        result = RankExchangeResult(
            rank=rank,
            attention_tokens=plan.attention_to_gdn.source_token_counts_by_rank[rank],
            gdn_tokens=plan.attention_to_gdn.dest_token_counts_by_rank[rank],
            forward_ms=_summary(forward_ms),
            backward_ms=_summary(backward_ms),
            e2e_ms=_summary(e2e_ms),
        )
        Path(run_dir, f"rank_{rank}.json").write_text(
            result.model_dump_json(indent=2) + "\n"
        )
    finally:
        destroy_process_group()


def _time_exchange_iteration(
    local_template: torch.Tensor,
    plan: GdnCpLayoutPlan,
    *,
    rank: int,
    backward: bool,
) -> float:
    local_tensor = local_template.clone().detach().requires_grad_(backward)
    _sync()
    start = time.perf_counter()
    with _nvtx_range("art_gdn_cp_attention_to_gdn_exchange"):
        output = exchange_rank_tensor_all_to_all(
            local_tensor,
            plan.attention_to_gdn,
            rank=rank,
            backward_plan=plan.gdn_to_attention,
        )
    if backward:
        with _nvtx_range("art_gdn_cp_exchange_backward"):
            output.square().sum().backward()
    _sync()
    return (time.perf_counter() - start) * 1000.0


def _time_exchange_backward(
    local_template: torch.Tensor,
    plan: GdnCpLayoutPlan,
    *,
    rank: int,
) -> float:
    local_tensor = local_template.clone().detach().requires_grad_(True)
    output = exchange_rank_tensor_all_to_all(
        local_tensor,
        plan.attention_to_gdn,
        rank=rank,
        backward_plan=plan.gdn_to_attention,
    )
    loss = output.square().sum()
    _sync()
    start = time.perf_counter()
    with _nvtx_range("art_gdn_cp_exchange_backward"):
        loss.backward()
    _sync()
    return (time.perf_counter() - start) * 1000.0


def _measure_plan_build(
    tensors: dict[str, Any],
    *,
    cp_size: int,
    iters: int,
) -> TimingSummary:
    elapsed = []
    for _ in range(iters):
        start = time.perf_counter()
        with _nvtx_range("art_gdn_cp_layout_plan"):
            _layout_plan(tensors, cp_size=cp_size)
        elapsed.append((time.perf_counter() - start) * 1000.0)
    return _summary(elapsed)


def _layout_plan(tensors: dict[str, Any], *, cp_size: int) -> GdnCpLayoutPlan:
    spec = parse_gdn_shared_prefix_segments(
        tensors["group_ids"],
        tensors["parent_ids"],
        min_completions_per_family=0,
    )
    return build_gdn_cp_layout_plan(
        execution_spec=spec,
        cp_size=cp_size,
        attention_token_layout_index=_reverse_striped_chunk_layout(
            spec, cp_size=cp_size
        ),
    )


def _write_outputs(
    output_dir: Path,
    results: tuple[CpLayoutExchangeResult, ...],
) -> None:
    (output_dir / "result.json").write_text(
        json.dumps([result.model_dump() for result in results], indent=2) + "\n"
    )
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "cp_size",
                "backend",
                "real_tokens",
                "plan_build_ms",
                "fwd_ms",
                "bwd_ms",
                "e2e_ms",
                "cross_rank_bytes_per_direction",
            ),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "cp_size": result.cp_size,
                    "backend": result.backend,
                    "real_tokens": result.real_tokens,
                    "plan_build_ms": result.plan_build_ms.median_ms,
                    "fwd_ms": result.max_rank_forward_ms,
                    "bwd_ms": result.max_rank_backward_ms,
                    "e2e_ms": result.max_rank_e2e_ms,
                    "cross_rank_bytes_per_direction": (
                        result.cross_rank_bytes_per_direction
                    ),
                }
            )
    lines = [
        "# GDN CP Layout Exchange Benchmark",
        "",
        "| CP | Backend | Real tokens | Plan ms | Fwd ms | Bwd ms | E2E ms | Cross-rank bytes/dir |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.cp_size} | {result.backend} | {result.real_tokens} | "
            f"{result.plan_build_ms.median_ms:.3f} | "
            f"{result.max_rank_forward_ms:.3f} | "
            f"{result.max_rank_backward_ms:.3f} | "
            f"{result.max_rank_e2e_ms:.3f} | "
            f"{result.cross_rank_bytes_per_direction} |"
        )
    lines.extend(
        (
            "",
            "Planning is reported separately because it is once per training sequence, not per GDN layer.",
            "Forward/backward timings synchronize all ranks to expose layout communication.",
        )
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def _sync() -> None:
    if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
        torch.cuda.synchronize()
    barrier()


def _summary(values: list[float]) -> TimingSummary:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize empty timings")
    return TimingSummary(
        median_ms=ordered[len(ordered) // 2],
        p90_ms=ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        max_ms=ordered[-1],
        raw_ms=tuple(values),
    )


def _manifest_configs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "layout_exchange_args": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "benchmark_dtype": str(BENCHMARK_DTYPE),
        "hidden_size_default": "qwen3_5_35b_a3b hidden_size",
        "cp_target_seq_len_rule": (
            "effective_target_seq_len = base_cp1_target_seq_len * cp_size; "
            "per-family prefix/completion lengths stay fixed and additional "
            "families are packed to target"
        ),
        "nvtx_ranges": _NVTX_RANGES,
    }


@contextmanager
def _nvtx_range(label: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
        return
    yield


def _select_backend(requested: str, cp_size: int) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available() and torch.cuda.device_count() >= cp_size:
        return "nccl"
    return "gloo"


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
        raise ValueError("target sequence must fit at least one complete family")
    families: list[GdnFamilyShape] = []
    used = 0
    while fitted := fit_gdn_family_to_remaining(family, target_seq_len - used):
        families.append(fitted)
        used += gdn_family_token_count(fitted)
        if len(fitted.suffix_lengths) != len(family.suffix_lengths):
            break
    if not families:
        raise ValueError("target sequence must fit at least one prefix plus completion")
    return GdnPhase0Case(
        name=(
            f"repeated_{prefix_len}_plus_{completions_per_family}x"
            f"{suffix_len}_target_{target_seq_len}"
        ),
        sequence_length=target_seq_len,
        rows=(GdnPackedRowShape(families=tuple(families)),),
        seed=43,
    )


def _reverse_striped_chunk_layout(spec: Any, *, cp_size: int) -> TokenLayoutIndex:
    chunks = list(_cp_chunk_ranges(spec, cp_size=cp_size))
    chunks.reverse()
    ranges_by_rank = _assign_chunks_round_robin(tuple(chunks), cp_size=cp_size)
    return TokenLayoutIndex(
        ownership_ranges_by_rank=ranges_by_rank,
        token_counts_by_rank=tuple(
            sum(end - start for start, end, _ in ranges) for ranges in ranges_by_rank
        ),
    )


def _cp_chunk_ranges(spec: Any, *, cp_size: int) -> tuple[tuple[int, int], ...]:
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
    return tuple(tuple(rank_ranges) for rank_ranges in ranks)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    raise SystemExit(main())
