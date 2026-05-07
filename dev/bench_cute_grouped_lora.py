#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from megatron.core.transformer.moe import grouped_gemm_util
from pydantic import BaseModel, ConfigDict, Field, field_validator
import torch

from art.megatron.kernels.cute_grouped_lora_quack import quack_grouped_lora

GroupedLoraFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def _parse_dtype(name: str) -> torch.dtype:
    value = name.strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}")


def _as_cpu_counts(counts: torch.Tensor) -> torch.Tensor:
    return counts.to(device="cpu", dtype=torch.int64).contiguous()


def _mean_ms(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_tokens: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    out_features: int = Field(gt=0)
    num_experts: int = Field(gt=0)
    rank: int = Field(gt=0)
    dtype_name: str
    warmup: int = Field(ge=0)
    iters: int = Field(gt=0)
    seed: int
    input_scale: float = Field(gt=0.0)
    weight_scale: float = Field(gt=0.0)
    skew: float = Field(ge=0.0)
    atol: float = Field(ge=0.0)
    rtol: float = Field(ge=0.0)

    @field_validator("rank")
    @classmethod
    def _validate_rank(cls, value: int) -> int:
        if value < 1 or value > 128 or (value & (value - 1)) != 0:
            raise ValueError("rank must be a power of 2 in [1, 128]")
        return value

    @property
    def dtype(self) -> torch.dtype:
        return _parse_dtype(self.dtype_name)


def _make_group_counts(spec: BenchmarkSpec) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed)
    raw = torch.rand(spec.num_experts, generator=generator, dtype=torch.float32)
    probs = raw.pow(1.0 + spec.skew)
    probs = probs / probs.sum()
    expert_ids = torch.multinomial(
        probs,
        spec.total_tokens,
        replacement=True,
        generator=generator,
    )
    return torch.bincount(expert_ids, minlength=spec.num_experts).to(dtype=torch.int64)


def _build_problem(
    spec: BenchmarkSpec,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(spec.seed)
    torch.cuda.manual_seed_all(spec.seed)
    counts = _make_group_counts(spec)
    loss_grad = torch.randn(
        spec.total_tokens,
        spec.out_features,
        device=device,
        dtype=spec.dtype,
    )
    return {
        "counts": counts,
        "x": (
            torch.randn(
                spec.total_tokens,
                spec.hidden_size,
                device=device,
                dtype=spec.dtype,
            )
            * spec.input_scale
        ),
        "a_t": (
            torch.randn(
                spec.num_experts,
                spec.hidden_size,
                spec.rank,
                device=device,
                dtype=spec.dtype,
            )
            * spec.weight_scale
        ),
        "b_t": (
            torch.randn(
                spec.num_experts,
                spec.rank,
                spec.out_features,
                device=device,
                dtype=spec.dtype,
            )
            * spec.weight_scale
        ),
        "loss_grad": loss_grad,
    }


def eager_fused_grouped_lora(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b_t: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    counts_list = [int(v) for v in counts.tolist()]
    outputs: list[torch.Tensor] = []
    start = 0
    for expert_idx, token_count in enumerate(counts_list):
        if token_count == 0:
            continue
        stop = start + token_count
        outputs.append(x[start:stop] @ a_t[expert_idx] @ b_t[expert_idx])
        start = stop
    if start != x.shape[0]:
        raise RuntimeError(
            f"Grouped split mismatch: consumed {start} tokens for shape {tuple(x.shape)}"
        )
    if not outputs:
        return x.new_empty((0, b_t.shape[-1]))
    return torch.cat(outputs, dim=0)


def grouped_gemm_grouped_lora(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b_t: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    counts_cpu = _as_cpu_counts(counts)
    tmp = grouped_gemm_util.ops.gmm(x, a_t, counts_cpu, trans_b=False)  # type: ignore[attr-defined]
    return grouped_gemm_util.ops.gmm(tmp, b_t, counts_cpu, trans_b=False)  # type: ignore[attr-defined]


def _backend_registry() -> dict[str, GroupedLoraFn]:
    registry: dict[str, GroupedLoraFn] = {
        "grouped_gemm": grouped_gemm_grouped_lora,
        "quack_final": quack_grouped_lora,
    }
    return registry


def _default_output_json_path(spec: BenchmarkSpec) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = (
        f"rank_{spec.rank}_tokens_{spec.total_tokens}_hidden_{spec.hidden_size}"
        f"_out_{spec.out_features}_{timestamp}.json"
    )
    return repo_root / ".local" / "bench_cute_grouped_lora" / filename


def _clone_problem(problem: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "counts": problem["counts"].clone(),
        "loss_grad": problem["loss_grad"].clone(),
        "x": problem["x"].detach().clone().requires_grad_(True),
        "a_t": problem["a_t"].detach().clone().requires_grad_(True),
        "b_t": problem["b_t"].detach().clone().requires_grad_(True),
    }


def _run_backward(
    fn: GroupedLoraFn,
    problem: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    out = fn(problem["x"], problem["a_t"], problem["b_t"], problem["counts"])
    if not isinstance(out, torch.Tensor):
        raise RuntimeError(
            f"Backend returned {type(out).__name__} instead of torch.Tensor"
        )
    if out.shape != problem["loss_grad"].shape:
        raise RuntimeError(
            "Output shape mismatch: "
            f"expected {tuple(problem['loss_grad'].shape)}, got {tuple(out.shape)}"
        )
    loss = (out.float() * problem["loss_grad"].float()).sum() / max(
        1, problem["loss_grad"].numel()
    )
    loss.backward()
    return {
        "out": out.detach(),
        "x_grad": problem["x"].grad.detach().clone(),
        "a_grad": problem["a_t"].grad.detach().clone(),
        "b_grad": problem["b_t"].grad.detach().clone(),
    }


def _tensor_summary(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    diff = (reference - candidate).abs()
    ref_abs_max = float(reference.abs().max().item()) if reference.numel() else 0.0
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    max_rel = max_abs / max(ref_abs_max, 1e-12)
    close = bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol))
    return {
        "close": close,
        "max_abs_diff": max_abs,
        "max_rel_to_ref_abs_max": max_rel,
    }


def validate_backend(
    *,
    backend_name: str,
    backend_fn: GroupedLoraFn,
    spec: BenchmarkSpec,
    problem: dict[str, torch.Tensor],
) -> dict[str, Any]:
    reference_outputs = _run_backward(eager_fused_grouped_lora, _clone_problem(problem))
    candidate_outputs = _run_backward(backend_fn, _clone_problem(problem))
    output_summary = _tensor_summary(
        reference_outputs["out"],
        candidate_outputs["out"],
        atol=spec.atol,
        rtol=spec.rtol,
    )
    x_grad_summary = _tensor_summary(
        reference_outputs["x_grad"],
        candidate_outputs["x_grad"],
        atol=spec.atol,
        rtol=spec.rtol,
    )
    a_grad_summary = _tensor_summary(
        reference_outputs["a_grad"],
        candidate_outputs["a_grad"],
        atol=spec.atol,
        rtol=spec.rtol,
    )
    b_grad_summary = _tensor_summary(
        reference_outputs["b_grad"],
        candidate_outputs["b_grad"],
        atol=spec.atol,
        rtol=spec.rtol,
    )
    passed = all(
        summary["close"]
        for summary in (
            output_summary,
            x_grad_summary,
            a_grad_summary,
            b_grad_summary,
        )
    )
    return {
        "backend": backend_name,
        "passed": passed,
        "output": output_summary,
        "x_grad": x_grad_summary,
        "a_grad": a_grad_summary,
        "b_grad": b_grad_summary,
    }


def benchmark_backend(
    *,
    backend_name: str,
    backend_fn: GroupedLoraFn,
    spec: BenchmarkSpec,
    device: torch.device,
    problem: dict[str, torch.Tensor],
) -> dict[str, Any]:
    validation_pre = validate_backend(
        backend_name=backend_name,
        backend_fn=backend_fn,
        spec=spec,
        problem=problem,
    )
    if not validation_pre["passed"]:
        raise RuntimeError(
            f"{backend_name} failed pre-benchmark validation: "
            f"{json.dumps(validation_pre, indent=2)}"
        )

    candidate = _clone_problem(problem)
    for _ in range(spec.warmup):
        candidate["x"].grad = None
        candidate["a_t"].grad = None
        candidate["b_t"].grad = None
        _run_backward(backend_fn, candidate)
    torch.cuda.synchronize(device)

    fwd_ms: list[float] = []
    bwd_ms: list[float] = []
    total_ms: list[float] = []
    peak_alloc_bytes = 0

    for _ in range(spec.iters):
        candidate = _clone_problem(problem)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        middle = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        out = backend_fn(
            candidate["x"], candidate["a_t"], candidate["b_t"], candidate["counts"]
        )
        loss = (out.float() * candidate["loss_grad"].float()).sum() / max(
            1, candidate["loss_grad"].numel()
        )
        middle.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize(device)

        fwd_ms.append(float(start.elapsed_time(middle)))
        bwd_ms.append(float(middle.elapsed_time(end)))
        total_ms.append(float(start.elapsed_time(end)))
        peak_alloc_bytes = max(
            peak_alloc_bytes,
            int(torch.cuda.max_memory_allocated(device=device)),
        )

    validation_post = validate_backend(
        backend_name=backend_name,
        backend_fn=backend_fn,
        spec=spec,
        problem=problem,
    )
    if not validation_post["passed"]:
        raise RuntimeError(
            f"{backend_name} failed post-benchmark validation: "
            f"{json.dumps(validation_post, indent=2)}"
        )

    counts = problem["counts"]
    nonzero_counts = counts[counts > 0]
    return {
        "backend": backend_name,
        "timing_ms": {
            "forward_mean": _mean_ms(fwd_ms),
            "backward_mean": _mean_ms(bwd_ms),
            "total_mean": _mean_ms(total_ms),
        },
        "peak_alloc_gib": peak_alloc_bytes / (1024**3),
        "counts_summary": {
            "num_groups": int(counts.numel()),
            "nonzero_groups": int(nonzero_counts.numel()),
            "zero_groups": int((counts == 0).sum().item()),
            "max_tokens_per_group": int(counts.max().item()),
            "mean_tokens_per_nonzero_group": float(nonzero_counts.float().mean().item())
            if nonzero_counts.numel()
            else 0.0,
        },
        "validation_pre": validation_pre,
        "validation_post": validation_post,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark grouped LoRA kernels with mandatory forward+backward "
            "validation before and after timing."
        )
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["grouped_gemm", "quack_final"],
        help=("Built-in backends to benchmark. Built-ins: grouped_gemm, quack_final."),
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=32768,
        help="Default is a Qwen3-30B-A3B-style batch size target for realistic sweeps.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=2048,
        help="Qwen3-30B-A3B hidden size.",
    )
    parser.add_argument(
        "--out-features",
        type=int,
        default=768,
        help="Qwen3-30B-A3B moe_intermediate_size for expert up/gate-style shapes.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=128,
        help="Qwen3-30B-A3B number of routed experts.",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-scale", type=float, default=0.05)
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument(
        "--skew",
        type=float,
        default=2.0,
        help="Higher values create a longer-tailed per-expert token distribution.",
    )
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Optional override for the JSON output path. If omitted, results are "
            "written under .local/bench_cute_grouped_lora/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    spec = BenchmarkSpec(
        total_tokens=args.total_tokens,
        hidden_size=args.hidden_size,
        out_features=args.out_features,
        num_experts=args.num_experts,
        rank=args.rank,
        dtype_name=args.dtype,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        input_scale=args.input_scale,
        weight_scale=args.weight_scale,
        skew=args.skew,
        atol=args.atol,
        rtol=args.rtol,
    )

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    problem = _build_problem(spec, device=device)
    registry = _backend_registry()

    missing = [name for name in args.backends if name not in registry]
    if missing:
        raise SystemExit(
            f"Unknown backends {missing}. Known backends: {sorted(registry)}"
        )

    results: dict[str, Any] = {
        "config": {
            "total_tokens": spec.total_tokens,
            "hidden_size": spec.hidden_size,
            "out_features": spec.out_features,
            "num_experts": spec.num_experts,
            "rank": spec.rank,
            "dtype": spec.dtype_name,
            "warmup": spec.warmup,
            "iters": spec.iters,
            "seed": spec.seed,
            "input_scale": spec.input_scale,
            "weight_scale": spec.weight_scale,
            "skew": spec.skew,
            "atol": spec.atol,
            "rtol": spec.rtol,
            "device": str(device),
        },
        "results": {},
    }

    for backend_name in args.backends:
        results["results"][backend_name] = benchmark_backend(
            backend_name=backend_name,
            backend_fn=registry[backend_name],
            spec=spec,
            device=device,
            problem=problem,
        )

    baseline_name = "grouped_gemm" if "grouped_gemm" in results["results"] else None
    baseline_total = (
        results["results"][baseline_name]["timing_ms"]["total_mean"]
        if baseline_name is not None
        else None
    )
    if baseline_total is not None:
        for backend_name, backend_result in results["results"].items():
            delta_ms = backend_result["timing_ms"]["total_mean"] - baseline_total
            backend_result["delta_vs_grouped_gemm_ms"] = delta_ms
            backend_result["delta_vs_grouped_gemm_pct"] = (
                100.0 * delta_ms / baseline_total
            )

    output_json = args.output_json or _default_output_json_path(spec)
    payload = json.dumps(results, indent=2, sort_keys=True)
    print(payload)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(payload + "\n")


if __name__ == "__main__":
    main()
