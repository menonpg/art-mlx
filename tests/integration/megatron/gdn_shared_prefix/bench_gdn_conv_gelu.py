from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import json
import math
from pathlib import Path
import socket
import statistics
import sys
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import torch
from torch import Tensor
from torch.distributed import destroy_process_group, init_process_group, is_initialized

from art.megatron.gdn.conv_gelu import varlen_causal_conv_gelu
from art.megatron.gdn.gdn_shared_prefix import (
    GdnPlannerConfig,
    GdnSegmentBucketPlan,
    build_gdn_rank_execution_plan,
    parse_gdn_shared_prefix_segments,
)
from art.megatron.gdn.operator import (
    _causal_conv1d_fn,
    _causal_conv1d_with_state,
    _conv_final_from_varlen_qkv,
)
from tests.integration.megatron.gdn_shared_prefix.benchmark_gdn import (
    make_qwen35_gdn_pair,
    qwen35_gdn_module_config,
)
from tests.integration.megatron.gdn_shared_prefix.cases import (
    GdnFamilyShape,
    GdnPackedRowShape,
    GdnPhase0Case,
    default_phase0_cases,
    fit_gdn_family_to_remaining,
    gdn_family_token_count,
)
from tests.integration.megatron.gdn_shared_prefix.metrics import mean_abs_pct
from tests.integration.megatron.gdn_shared_prefix.packed_layout import (
    build_phase0_packed_tensors,
)

SCRATCH_DIR = Path(__file__).resolve().parent / "scratch"


class PathSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    fn: Callable[..., tuple[Tensor, Tensor]]


class BucketCase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    source_case: str
    kind: str
    bucket: GdnSegmentBucketPlan

    @property
    def segment_count(self) -> int:
        return int(self.bucket.segment_count)

    @property
    def max_len(self) -> int:
        return int(self.bucket.length)

    @property
    def real_tokens(self) -> int:
        return int(self.bucket.real_token_count)


class CorrectnessMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_pct: float
    final_pct: float
    qkv_grad_pct: float
    conv_initial_grad_pct: float
    weight_grad_pct: float
    bias_grad_pct: float | None = None

    @property
    def worst_pct(self) -> float:
        values = (
            self.output_pct,
            self.final_pct,
            self.qkv_grad_pct,
            self.conv_initial_grad_pct,
            self.weight_grad_pct,
        )
        bias = () if self.bias_grad_pct is None else (self.bias_grad_pct,)
        return max(*values, *bias)


class CorrectnessResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case: str
    path: str
    dtype: str
    segments: int = Field(ge=1)
    max_len: int = Field(ge=1)
    channels: int = Field(ge=1)
    kernel_width: int = Field(ge=1)
    real_tokens: int = Field(ge=1)
    metrics: CorrectnessMetrics


class TimingSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    median_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float


class PerfResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case: str
    path: str
    dtype: str
    segments: int = Field(ge=1)
    max_len: int = Field(ge=1)
    channels: int = Field(ge=1)
    kernel_width: int = Field(ge=1)
    real_tokens: int = Field(ge=1)
    fwd_ms: TimingSummary
    bwd_ms: TimingSummary
    e2e_ms: TimingSummary
    e2e_tokens_per_second: float
    speedup_vs_production: float | None = None


class LaunchCountResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case: str
    path: str
    launches: int
    top_kernels: tuple[tuple[str, int], ...]


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    torch_version: str
    triton_version: str
    device_name: str
    production_backend: str
    correctness: tuple[CorrectnessResult, ...]
    performance: tuple[PerfResult, ...]
    launch_counts: tuple[LaunchCountResult, ...] = ()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GDN conv+GELU benchmark")
    torch.cuda.set_device(args.device)
    output_dir = args.output_dir or SCRATCH_DIR / "gdn_conv_gelu"
    output_dir.mkdir(parents=True, exist_ok=True)
    with _single_rank_model_parallel(args.device):
        config = qwen35_gdn_module_config().model_copy(
            update={"linear_conv_kernel_dim": args.conv_width}
        )
        gdn, _ = make_qwen35_gdn_pair(
            params_dtype=_dtype(args.dtype),
            linear_policy="noop",
            config=config,
        )
        gdn.eval()
        cases = _bucket_cases(args)
        paths = (
            PathSpec(name="production", fn=_production_path),
            PathSpec(name="triton_fused", fn=_fused_path),
        )
        correctness = _run_correctness(gdn, cases, paths, args)
        performance = _run_performance(gdn, cases, paths, args)
        performance = _with_speedups(performance)
        launch_counts = (
            _run_launch_counts(gdn, cases, paths, args) if args.count_launches else ()
        )
        report = BenchmarkReport(
            torch_version=torch.__version__,
            triton_version=_triton_version(),
            device_name=torch.cuda.get_device_name(args.device),
            production_backend=(
                "causal_conv1d"
                if _causal_conv1d_fn() is not None
                else "torch_conv1d_native_fallback"
            ),
            correctness=correctness,
            performance=performance,
            launch_counts=launch_counts,
        )
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(report.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    table_path = output_dir / "summary.md"
    table_path.write_text(_render_summary(report), encoding="utf-8")
    print(json.dumps({"result": str(result_path), "summary": str(table_path)}))
    print(_render_summary(report))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fused prepared-varlen GDN causal conv+GELU."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--conv-width", type=int, default=4)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--correctness-cases", default="all")
    parser.add_argument("--perf-cases", default="all")
    parser.add_argument("--target-seq-len", type=int, default=40960)
    parser.add_argument("--prefix-len", type=int, default=5000)
    parser.add_argument("--suffix-len", type=int, default=100)
    parser.add_argument("--completions-per-family", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--count-launches", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def _bucket_cases(args: argparse.Namespace) -> tuple[BucketCase, ...]:
    repeated = _repeated_family_case(
        target_seq_len=args.target_seq_len,
        prefix_len=args.prefix_len,
        suffix_len=args.suffix_len,
        completions_per_family=args.completions_per_family,
    )
    varied = _varied_repeated_family_case(
        target_seq_len=args.target_seq_len,
        prefix_len=args.prefix_len,
        suffix_len=args.suffix_len,
        completions_per_family=args.completions_per_family,
    )
    edge = next(
        case
        for case in default_phase0_cases(args.conv_width)
        if case.name == "conv_tail_boundary"
    )
    return (
        _largest_bucket(repeated, "prefix", "repeated_prefix"),
        _largest_bucket(repeated, "completion", "repeated_completion"),
        _largest_bucket(varied, "prefix", "varied_prefix"),
        _largest_bucket(varied, "completion", "varied_completion"),
        _largest_bucket(edge, "completion", "conv_tail_boundary_completion"),
    )


def _largest_bucket(case: GdnPhase0Case, kind: str, name: str) -> BucketCase:
    tensors = build_phase0_packed_tensors(case)
    spec = parse_gdn_shared_prefix_segments(
        tensors["group_ids"].cuda(),
        tensors["parent_ids"].cuda(),
        min_completions_per_family=1,
    )
    plan = build_gdn_rank_execution_plan(
        spec,
        device=torch.device("cuda"),
        planner_config=GdnPlannerConfig(
            max_padding_ratio=4.0, max_segments_per_batch=4096
        ),
    )
    buckets = (
        plan.prefix_boundary_buckets + plan.prefix_tail_buckets
        if kind == "prefix"
        else plan.completion_with_prefix_tail_buckets
    )
    if not buckets:
        raise RuntimeError(f"{case.name} has no {kind} buckets")
    bucket = max(buckets, key=lambda item: item.real_token_count)
    return BucketCase(name=name, source_case=case.name, kind=kind, bucket=bucket)


def _run_correctness(
    gdn: Any,
    cases: tuple[BucketCase, ...],
    paths: tuple[PathSpec, ...],
    args: argparse.Namespace,
) -> tuple[CorrectnessResult, ...]:
    selected = _select_cases(cases, args.correctness_cases)
    results = []
    for case_index, case in enumerate(selected):
        inputs = _make_inputs(
            gdn, case.bucket, _dtype("float32"), args.seed + case_index
        )
        reference = _run_once(gdn, paths[0].fn, inputs)
        _assert_not_all_zero("production output", reference["out"])
        for path in paths[1:]:
            candidate = _run_once(gdn, path.fn, inputs)
            metrics = CorrectnessMetrics(
                output_pct=mean_abs_pct(reference["out"], candidate["out"]),
                final_pct=mean_abs_pct(reference["final"], candidate["final"]),
                qkv_grad_pct=mean_abs_pct(reference["qkv_grad"], candidate["qkv_grad"]),
                conv_initial_grad_pct=mean_abs_pct(
                    reference["conv_initial_grad"], candidate["conv_initial_grad"]
                ),
                weight_grad_pct=mean_abs_pct(
                    reference["weight_grad"], candidate["weight_grad"]
                ),
                bias_grad_pct=(
                    None
                    if reference["bias_grad"] is None
                    else mean_abs_pct(reference["bias_grad"], candidate["bias_grad"])
                ),
            )
            if metrics.worst_pct > 0.5:
                raise AssertionError(
                    f"{case.name} {path.name} mean_abs_pct exceeded 0.5: {metrics}"
                )
            results.append(
                _correctness_result(case, path.name, "float32", inputs, metrics)
            )
    return tuple(results)


def _run_performance(
    gdn: Any,
    cases: tuple[BucketCase, ...],
    paths: tuple[PathSpec, ...],
    args: argparse.Namespace,
) -> tuple[PerfResult, ...]:
    selected = _select_cases(cases, args.perf_cases)
    results = []
    dtype = _dtype(args.dtype)
    for case_index, case in enumerate(selected):
        inputs = _make_inputs(gdn, case.bucket, dtype, args.seed + 100 + case_index)
        for path in paths:
            fwd = _time_many(
                lambda: _run_fwd_only(gdn, path.fn, inputs),
                args.warmup_iters,
                args.iters,
            )
            bwd = _time_backward_many(
                gdn,
                path.fn,
                inputs,
                args.warmup_iters,
                args.iters,
            )
            e2e = _time_many(
                lambda: _run_e2e(gdn, path.fn, inputs),
                args.warmup_iters,
                args.iters,
            )
            e2e_summary = _summary(e2e)
            results.append(
                PerfResult(
                    case=case.name,
                    path=path.name,
                    dtype=str(dtype),
                    segments=case.segment_count,
                    max_len=case.max_len,
                    channels=int(inputs["qkv"].shape[1]),
                    kernel_width=int(inputs["weight"].shape[1]),
                    real_tokens=case.real_tokens,
                    fwd_ms=_summary(fwd),
                    bwd_ms=_summary(bwd),
                    e2e_ms=e2e_summary,
                    e2e_tokens_per_second=1000.0
                    * case.real_tokens
                    / e2e_summary.median_ms,
                )
            )
        torch.cuda.empty_cache()
    return tuple(results)


def _run_launch_counts(
    gdn: Any,
    cases: tuple[BucketCase, ...],
    paths: tuple[PathSpec, ...],
    args: argparse.Namespace,
) -> tuple[LaunchCountResult, ...]:
    selected = _select_cases(cases, args.perf_cases)
    if not selected:
        return ()
    case = selected[0]
    inputs = _make_inputs(gdn, case.bucket, _dtype(args.dtype), args.seed + 700)
    results = []
    from torch.profiler import ProfilerActivity, profile

    for path in paths:
        _run_e2e(gdn, path.fn, inputs)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            _run_e2e(gdn, path.fn, inputs)
            torch.cuda.synchronize()
        counts: dict[str, int] = {}
        for event in prof.events():
            if str(event.device_type).endswith("CUDA"):
                counts[event.name] = counts.get(event.name, 0) + 1
        top = tuple(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10])
        results.append(
            LaunchCountResult(
                case=case.name,
                path=path.name,
                launches=sum(counts.values()),
                top_kernels=top,
            )
        )
    return tuple(results)


def _make_inputs(
    gdn: Any, bucket: GdnSegmentBucketPlan, dtype: torch.dtype, seed: int
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch = int(bucket.segment_count)
    channels = int(gdn.conv_dim_local_tp)
    max_len = int(bucket.length)
    kernel_width = int(gdn.conv_kernel_dim)
    qkv = torch.randn(
        batch, channels, max_len, device="cuda", dtype=dtype, generator=generator
    )
    real_mask = bucket.real_mask.transpose(0, 1).unsqueeze(1)
    qkv = qkv.masked_fill(~real_mask, 0)
    conv_initial = torch.randn(
        batch,
        channels,
        kernel_width - 1,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    weight = gdn.conv1d.weight.detach().squeeze(1).to(dtype=dtype).contiguous()
    bias = (
        None
        if gdn.conv1d.bias is None
        else gdn.conv1d.bias.detach().to(dtype=dtype).contiguous()
    )
    out_grad = torch.randn(qkv.shape, device="cuda", dtype=dtype, generator=generator)
    out_grad = out_grad.masked_fill(~real_mask, 0)
    final_grad = torch.randn(
        conv_initial.shape, device="cuda", dtype=dtype, generator=generator
    )
    return {
        "qkv": qkv.contiguous(),
        "conv_initial": conv_initial.contiguous(),
        "weight": weight,
        "bias": bias,
        "lengths": bucket.lengths,
        "out_grad": out_grad.contiguous(),
        "final_grad": final_grad.contiguous(),
    }


def _run_once(
    gdn: Any,
    fn: Callable[..., tuple[Tensor, Tensor]],
    inputs: dict[str, Any],
) -> dict[str, Tensor | None]:
    qkv, conv_initial, weight, bias = _leaves(inputs)
    with _patch_gdn_conv(gdn, weight, bias) as (weight_param, bias_param):
        out, final = fn(gdn, qkv, conv_initial, inputs["lengths"])
        loss = (out * inputs["out_grad"]).sum() + (final * inputs["final_grad"]).sum()
        loss.backward()
        return {
            "out": out.detach(),
            "final": final.detach(),
            "qkv_grad": _grad(qkv),
            "conv_initial_grad": _grad(conv_initial),
            "weight_grad": _grad_or_param(weight, weight_param).reshape_as(weight),
            "bias_grad": None if bias is None else _grad_or_param(bias, bias_param),
        }


def _run_fwd_only(
    gdn: Any, fn: Callable[..., tuple[Tensor, Tensor]], inputs: dict[str, Any]
) -> None:
    with torch.no_grad(), _patch_gdn_conv(gdn, inputs["weight"], inputs["bias"]):
        out, final = fn(gdn, inputs["qkv"], inputs["conv_initial"], inputs["lengths"])
        _keep_alive(out, final)


def _run_bwd_timed(
    gdn: Any, fn: Callable[..., tuple[Tensor, Tensor]], inputs: dict[str, Any]
) -> float:
    qkv, conv_initial, weight, bias = _leaves(inputs)
    with _patch_gdn_conv(gdn, weight, bias):
        out, final = fn(gdn, qkv, conv_initial, inputs["lengths"])
        loss = (out * inputs["out_grad"]).sum() + (final * inputs["final_grad"]).sum()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))


def _run_e2e(
    gdn: Any, fn: Callable[..., tuple[Tensor, Tensor]], inputs: dict[str, Any]
) -> None:
    qkv, conv_initial, weight, bias = _leaves(inputs)
    with _patch_gdn_conv(gdn, weight, bias):
        out, final = fn(gdn, qkv, conv_initial, inputs["lengths"])
        (
            (out * inputs["out_grad"]).sum() + (final * inputs["final_grad"]).sum()
        ).backward()


def _production_path(
    gdn: Any, qkv: Tensor, conv_initial: Tensor, lengths: Tensor
) -> tuple[Tensor, Tensor]:
    final = _conv_final_from_varlen_qkv(qkv, conv_initial, lengths)
    out, _ = _causal_conv1d_with_state(gdn, qkv, conv_initial, output_final_state=False)
    return out, final


def _fused_path(
    gdn: Any, qkv: Tensor, conv_initial: Tensor, lengths: Tensor
) -> tuple[Tensor, Tensor]:
    weight = gdn.conv1d.weight.squeeze(1)
    out, final = varlen_causal_conv_gelu(
        qkv,
        conv_initial,
        weight,
        gdn.conv1d.bias,
        lengths,
        output_final_state=True,
    )
    assert final is not None
    return out, final


def _time_many(fn: Callable[[], None], warmups: int, iters: int) -> list[float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def _time_backward_many(
    gdn: Any,
    fn: Callable[..., tuple[Tensor, Tensor]],
    inputs: dict[str, Any],
    warmups: int,
    iters: int,
) -> list[float]:
    for _ in range(warmups):
        _run_bwd_timed(gdn, fn, inputs)
    return [_run_bwd_timed(gdn, fn, inputs) for _ in range(iters)]


@contextmanager
def _patch_gdn_conv(
    gdn: Any, weight: Tensor, bias: Tensor | None
) -> Iterator[tuple[Tensor, Tensor | None]]:
    old_weight = gdn.conv1d.weight
    old_bias = gdn.conv1d.bias
    weight_param = torch.nn.Parameter(
        weight.reshape(weight.shape[0], 1, weight.shape[1])
    )
    bias_param = None if bias is None else torch.nn.Parameter(bias)
    gdn.conv1d.weight = weight_param
    gdn.conv1d.bias = bias_param
    try:
        yield weight_param, bias_param
    finally:
        gdn.conv1d.weight = old_weight
        gdn.conv1d.bias = old_bias


def _leaves(inputs: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    qkv = inputs["qkv"].detach().clone().requires_grad_(True)
    conv_initial = inputs["conv_initial"].detach().clone().requires_grad_(True)
    weight = inputs["weight"].detach().clone().requires_grad_(True)
    bias = None
    if inputs["bias"] is not None:
        bias = inputs["bias"].detach().clone().requires_grad_(True)
    return qkv, conv_initial, weight, bias


def _grad(tensor: Tensor) -> Tensor:
    if tensor.grad is None:
        raise AssertionError("missing gradient")
    return tensor.grad.detach()


def _grad_or_param(leaf: Tensor, parameter: Tensor | None) -> Tensor:
    if leaf.grad is not None:
        return leaf.grad.detach()
    if parameter is not None and parameter.grad is not None:
        return parameter.grad.detach()
    raise AssertionError("missing gradient")


def _keep_alive(*tensors: Tensor | None) -> None:
    for tensor in tensors:
        if tensor is not None and tensor.numel() == -1:
            raise AssertionError("unreachable")


def _summary(values: list[float]) -> TimingSummary:
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return TimingSummary(
        median_ms=statistics.median(values),
        p90_ms=ordered[p90_index],
        min_ms=min(values),
        max_ms=max(values),
    )


def _with_speedups(results: tuple[PerfResult, ...]) -> tuple[PerfResult, ...]:
    production = {
        result.case: result.e2e_ms.median_ms
        for result in results
        if result.path == "production"
    }
    updated = []
    for result in results:
        base = production.get(result.case)
        speedup = (
            None
            if base is None or result.e2e_ms.median_ms <= 0
            else base / result.e2e_ms.median_ms
        )
        updated.append(result.model_copy(update={"speedup_vs_production": speedup}))
    return tuple(updated)


def _correctness_result(
    case: BucketCase,
    path: str,
    dtype: str,
    inputs: dict[str, Any],
    metrics: CorrectnessMetrics,
) -> CorrectnessResult:
    return CorrectnessResult(
        case=case.name,
        path=path,
        dtype=dtype,
        segments=case.segment_count,
        max_len=case.max_len,
        channels=int(inputs["qkv"].shape[1]),
        kernel_width=int(inputs["weight"].shape[1]),
        real_tokens=case.real_tokens,
        metrics=metrics,
    )


def _select_cases(
    cases: tuple[BucketCase, ...], selection: str
) -> tuple[BucketCase, ...]:
    if selection == "all":
        return cases
    names = {name.strip() for name in selection.split(",") if name.strip()}
    selected = tuple(case for case in cases if case.name in names)
    missing = names - {case.name for case in selected}
    if missing:
        raise ValueError(f"unknown case names: {sorted(missing)}")
    return selected


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
    families: list[GdnFamilyShape] = []
    used = 0
    while fitted := fit_gdn_family_to_remaining(family, target_seq_len - used):
        families.append(fitted)
        used += gdn_family_token_count(fitted)
        if len(fitted.suffix_lengths) != len(family.suffix_lengths):
            break
    if not families:
        raise ValueError("target_seq_len does not fit one repeated family")
    return GdnPhase0Case(
        name=f"repeated_{prefix_len}_plus_{completions_per_family}x{suffix_len}",
        sequence_length=target_seq_len,
        rows=(GdnPackedRowShape(families=tuple(families)),),
        seed=41,
    )


def _varied_repeated_family_case(
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
    families = []
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
        raise ValueError("target_seq_len does not fit one varied family")
    return GdnPhase0Case(
        name=f"varied_{prefix_len}_plus_{completions_per_family}x{suffix_len}",
        sequence_length=target_seq_len,
        rows=(GdnPackedRowShape(families=tuple(families)),),
        seed=43,
    )


def _assert_not_all_zero(name: str, tensor: Tensor | None) -> None:
    if tensor is None or tensor.numel() == 0:
        return
    if not bool(torch.any(tensor.detach() != 0).item()):
        raise AssertionError(f"{name} is all zero")


def _dtype(name: str) -> torch.dtype:
    return torch.float32 if name == "float32" else torch.bfloat16


def _triton_version() -> str:
    import triton

    return triton.__version__


@contextmanager
def _single_rank_model_parallel(device: int) -> Iterator[None]:
    from megatron.core import parallel_state as ps

    if is_initialized():
        raise RuntimeError("torch.distributed is already initialized")
    torch.cuda.set_device(device)
    init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{_free_port()}",
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _render_summary(report: BenchmarkReport) -> str:
    lines = [
        "# GDN Conv+GELU Benchmark",
        "",
        f"- torch: `{report.torch_version}`",
        f"- triton: `{report.triton_version}`",
        f"- device: `{report.device_name}`",
        f"- production backend: `{report.production_backend}`",
        "",
        "## Correctness",
        "",
        "| case | path | dtype | shape | out% | final% | qkv grad% | init grad% | weight grad% | bias grad% |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report.correctness:
        metrics = result.metrics
        bias = (
            "n/a" if metrics.bias_grad_pct is None else f"{metrics.bias_grad_pct:.6g}"
        )
        lines.append(
            f"| {result.case} | {result.path} | {result.dtype} | "
            f"{result.segments}x{result.channels}x{result.max_len}/k{result.kernel_width} | "
            f"{metrics.output_pct:.6g} | {metrics.final_pct:.6g} | "
            f"{metrics.qkv_grad_pct:.6g} | {metrics.conv_initial_grad_pct:.6g} | "
            f"{metrics.weight_grad_pct:.6g} | {bias} |"
        )
    lines.extend(
        [
            "",
            "## Performance",
            "",
            "| case | path | dtype | shape | fwd ms | bwd ms | e2e ms | toks/s | speedup |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in report.performance:
        speedup = (
            "n/a"
            if result.speedup_vs_production is None
            else f"{result.speedup_vs_production:.3f}"
        )
        lines.append(
            f"| {result.case} | {result.path} | {result.dtype} | "
            f"{result.segments}x{result.channels}x{result.max_len}/k{result.kernel_width} | "
            f"{result.fwd_ms.median_ms:.3f} | {result.bwd_ms.median_ms:.3f} | "
            f"{result.e2e_ms.median_ms:.3f} | {result.e2e_tokens_per_second:.0f} | {speedup} |"
        )
    if report.launch_counts:
        lines.extend(
            [
                "",
                "## Launch Counts",
                "",
                "| case | path | launches | top kernels |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for result in report.launch_counts:
            top = ", ".join(
                f"{name} x{count}" for name, count in result.top_kernels[:5]
            )
            lines.append(
                f"| {result.case} | {result.path} | {result.launches} | {top} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
