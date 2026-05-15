from __future__ import annotations

from typing import Any

import torch

from .oracle_harness import (
    FlexBackend,
    MetricThresholdRule,
    OracleCaseConfig,
    OracleObjective,
    PhasePassFn,
    ProviderPrecisionOverrides,
    VariantReport,
    VariantRunner,
    VariantSpec,
    oracle_output_slug,
    oracle_topology,
    selected_oracle_objectives,
)

FP8_BASE_WEIGHT_VALIDATOR = (
    "integration.megatron.model_support.fp8_oracle:"
    "assert_fp8_base_weight_overrides_effective"
)


def fp8_base_weight_precision_overrides() -> ProviderPrecisionOverrides:
    """Returns the FP8 base-weight settings used by the oracle sanity gate."""
    return {
        "fp8": "e4m3",
        "fp8_recipe": "blockwise",
        "fp8_param": True,
        "fp8_wgrad": True,
        "fp8_dot_product_attention": False,
        "fp8_multi_head_attention": False,
        "moe_router_padding_for_quantization": True,
        "moe_token_dispatcher_type": "alltoall",
        "art_lora_dtype": "bf16",
    }


def _fp8_base_weight_phase_pass_fns() -> dict[str, PhasePassFn]:
    """Builds FP8-vs-BF16 sanity limits while preserving exact routing replay checks."""
    non_zero_scales = {"typical_abs_scale": 0.0, "candidate_abs_scale": 0.0}
    fwd_out_loss = MetricThresholdRule(limits={"mean_abs_pct": 12.0})
    fwd_out = MetricThresholdRule(
        limits={"mean_abs_pct": 12.0},
        minimums=non_zero_scales,
    )
    grads = MetricThresholdRule(
        limits={"mean_abs_pct": 25.0},
        minimums=non_zero_scales,
    )
    deltas = MetricThresholdRule(
        limits={"mean_abs_pct": 15.0},
        minimums=non_zero_scales,
    )
    router_scores_rule = MetricThresholdRule(
        limits={"relative_l2": 0.0, "mean_abs_pct": 0.0}
    )
    router_topk_rule = MetricThresholdRule(
        limits={"topk_mismatch_fraction": 0.0, "top1_mismatch_fraction": 0.0}
    )
    return {"forward": fwd_out, "outputs": fwd_out, "losses": fwd_out_loss} | {
        "grads": grads,
        "deltas": deltas,
        "router_scores": router_scores_rule,
        "router_topk_ids": router_topk_rule,
    }


def _fp8_base_weight_variants(
    objective: OracleObjective,
    *,
    is_moe: bool = True,
) -> list[VariantSpec]:
    """Builds the single-rank FP8 base-weight candidate against the canonical oracle."""
    topology = oracle_topology(is_moe=is_moe)
    return [
        VariantSpec(
            name=f"{objective}_fp8_base_weights_{topology.slug()}",
            objective=objective,
            topology=topology,
            output_slug=oracle_output_slug(objective, topology, "fp8_base_weights"),
            pass_fn_by_phase=_fp8_base_weight_phase_pass_fns(),
            provider_precision_overrides=fp8_base_weight_precision_overrides(),
            provider_precision_validator=FP8_BASE_WEIGHT_VALIDATOR,
        )
    ]


def run_fp8_base_weight_suite(
    *,
    case_config: OracleCaseConfig,
    oracle_flex_backend: FlexBackend | None = None,
    variant_flex_backend: FlexBackend | None = None,
) -> list[VariantReport]:
    """Runs a single-rank FP8 base-weight sanity candidate against the canonical oracle."""
    reports: list[VariantReport] = []
    for objective in selected_oracle_objectives():
        runner = VariantRunner(
            objective=objective,
            case_config=case_config,
            oracle_flex_backend=oracle_flex_backend,
            variant_flex_backend=variant_flex_backend,
        )
        reports.extend(
            runner.run_suite(
                _fp8_base_weight_variants(
                    objective,
                    is_moe=case_config.is_moe,
                )
            )
        )
    return reports


def _iter_named_unique_parameters(
    model_chunks: list[Any],
) -> list[tuple[str, torch.nn.Parameter]]:
    seen: set[int] = set()
    params: list[tuple[str, torch.nn.Parameter]] = []
    for chunk_index, chunk in enumerate(model_chunks):
        for name, param in chunk.named_parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append((f"chunk{chunk_index}.{name}", param))
    return params


def assert_fp8_base_weight_overrides_effective(
    model_chunks: list[Any],
    provider: Any,
    precision_overrides: ProviderPrecisionOverrides,
) -> None:
    """Checks that an FP8 oracle variant exercises FP8 base params and BF16 LoRA."""
    if precision_overrides.get("fp8_param"):
        from megatron.core.fp8_utils import is_float8tensor

        fp8_param_count = sum(
            1
            for _, parameter in _iter_named_unique_parameters(model_chunks)
            if is_float8tensor(parameter)
        )
        if fp8_param_count == 0:
            raise RuntimeError("Expected fp8_param=True to create FP8 base parameters.")

    if precision_overrides.get("art_lora_dtype") != "bf16":
        return
    mismatched_lora_params = [
        f"{name}:{parameter.dtype}"
        for name, parameter in _iter_named_unique_parameters(model_chunks)
        if hasattr(parameter, "lora_shard_domain") and parameter.dtype != torch.bfloat16
    ]
    if mismatched_lora_params:
        raise RuntimeError(
            "LoRA parameter dtype mismatch under FP8 precision override: "
            + ", ".join(mismatched_lora_params[:8])
        )
