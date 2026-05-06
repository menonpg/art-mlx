"""ART harness Megatron worker entrypoint with CE and GDN timing overrides."""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from typing import Any

CE_IMPL_ENV = "ART_HARNESS_CROSS_ENTROPY_FUSION_IMPL"
HARNESS_ROOT = "/mnt/ws_pvc/ws/projects/art_harness"


def _install_harness_import_path() -> None:
    if HARNESS_ROOT not in sys.path:
        sys.path.insert(0, HARNESS_ROOT)


def _install_ce_impl_override() -> None:
    impl = os.environ.get(CE_IMPL_ENV, "").strip()
    if not impl:
        return

    import art.megatron.provider as provider_module

    original_prepare_provider_bundle = provider_module.prepare_provider_bundle

    def prepare_provider_bundle_with_ce_impl(*args: Any, **kwargs: Any) -> Any:
        bundle = original_prepare_provider_bundle(*args, **kwargs)
        bundle.provider.cross_entropy_loss_fusion = True
        bundle.provider.cross_entropy_fusion_impl = impl
        return bundle

    provider_module.prepare_provider_bundle = prepare_provider_bundle_with_ce_impl


def _install_gdn_timing_overrides(timing_worker: Any) -> None:
    profiler_cls = timing_worker.LayerTimingProfiler
    original_infer_layer_type = profiler_cls._infer_layer_type
    original_estimate_module_flops = profiler_cls._estimate_module_flops
    original_build_exclusive_categories = profiler_cls._build_exclusive_categories
    original_install_timing_patches = timing_worker._install_timing_patches

    def infer_layer_type_with_gdn(
        self: Any,
        module: Any,
        *,
        module_name: str = "",
    ) -> str | None:
        if isinstance(module, self._lora_cls):
            prefix = str(getattr(module, "adapter_model_prefix", ""))
            if ".linear_attn" in prefix:
                return "gdn_lora"
        class_name = module.__class__.__name__
        lowered_name = str(module_name).lower()
        if class_name == "GatedDeltaNet" or lowered_name.endswith(".linear_attn"):
            return "gdn"
        return original_infer_layer_type(self, module, module_name=module_name)

    def estimate_module_flops_with_gdn(
        self: Any,
        *,
        record: Any,
        module: Any,
        is_forward: bool,
    ) -> tuple[int, int, float, float, dict[str, float]]:
        if record.layer_type not in {"gdn", "gdn_lora"}:
            return original_estimate_module_flops(
                self,
                record=record,
                module=module,
                is_forward=is_forward,
            )
        token_count = self._resolve_token_count(layer_type=record.layer_type)
        active_params, active_trainable_params = self._effective_param_counts_for_call(
            record=record,
        )
        linear_flops = 2.0 * float(token_count) * float(active_params)
        if not is_forward:
            linear_flops += 2.0 * float(token_count) * float(active_trainable_params)
        return (token_count, 0, linear_flops, 0.0, {})

    def build_exclusive_categories_with_gdn(
        self: Any,
        raw_categories: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        exclusive = original_build_exclusive_categories(self, raw_categories)
        gdn_raw = raw_categories.get("gdn")
        if gdn_raw is None:
            return exclusive
        gdn_lora_raw = raw_categories.get("gdn_lora", _empty_category())
        exclusive["gdn"] = _subtract_categories(self, gdn_raw, gdn_lora_raw)
        exclusive["gdn_lora"] = gdn_lora_raw
        return exclusive

    def install_timing_patches_with_gdn(timer: Any, state: Any) -> None:
        original_install_timing_patches(timer, state)
        if state.layer_profiler is not None:
            _install_gdn_operator_timing(state.layer_profiler)

    profiler_cls._infer_layer_type = infer_layer_type_with_gdn
    profiler_cls._estimate_module_flops = estimate_module_flops_with_gdn
    profiler_cls._build_exclusive_categories = build_exclusive_categories_with_gdn
    timing_worker._install_timing_patches = install_timing_patches_with_gdn


def _empty_category() -> dict[str, Any]:
    return {
        "fwd_ms": 0.0,
        "bwd_ms": 0.0,
        "total_ms": 0.0,
        "fwd_calls": 0,
        "bwd_calls": 0,
        "fwd_tokens": 0,
        "bwd_tokens": 0,
        "fwd_attention_pairs": 0,
        "bwd_attention_pairs": 0,
        "fwd_flops_est": 0.0,
        "bwd_flops_est": 0.0,
        "fwd_linear_flops_est": 0.0,
        "bwd_linear_flops_est": 0.0,
        "fwd_attention_flops_est": 0.0,
        "bwd_attention_flops_est": 0.0,
        "fwd_elementwise_flops_est": 0.0,
        "bwd_elementwise_flops_est": 0.0,
        "fwd_routing_flops_est": 0.0,
        "bwd_routing_flops_est": 0.0,
        "fwd_dispatch_flops_est": 0.0,
        "bwd_dispatch_flops_est": 0.0,
        "fwd_combine_flops_est": 0.0,
        "bwd_combine_flops_est": 0.0,
        "fwd_loss_flops_est": 0.0,
        "bwd_loss_flops_est": 0.0,
        "total_flops_est": 0.0,
        "fwd_tflops_est": 0.0,
        "bwd_tflops_est": 0.0,
        "total_tflops_est": 0.0,
        "fwd_mfu": None,
        "bwd_mfu": None,
        "mfu": None,
    }


def _subtract_categories(
    profiler: Any,
    base: dict[str, Any],
    sub: dict[str, Any],
) -> dict[str, Any]:
    out = _empty_category()
    for key in (
        "fwd_ms",
        "bwd_ms",
        "fwd_flops_est",
        "bwd_flops_est",
        "fwd_linear_flops_est",
        "bwd_linear_flops_est",
        "fwd_attention_flops_est",
        "bwd_attention_flops_est",
        "fwd_elementwise_flops_est",
        "bwd_elementwise_flops_est",
        "fwd_routing_flops_est",
        "bwd_routing_flops_est",
        "fwd_dispatch_flops_est",
        "bwd_dispatch_flops_est",
        "fwd_combine_flops_est",
        "bwd_combine_flops_est",
        "fwd_loss_flops_est",
        "bwd_loss_flops_est",
    ):
        out[key] = round(
            max(0.0, float(base.get(key, 0.0)) - float(sub.get(key, 0.0))), 6
        )
    out["total_ms"] = round(float(out["fwd_ms"]) + float(out["bwd_ms"]), 6)
    out["total_flops_est"] = round(
        float(out["fwd_flops_est"]) + float(out["bwd_flops_est"]), 2
    )
    out["fwd_tflops_est"] = round(
        profiler._to_tflops(float(out["fwd_flops_est"]), float(out["fwd_ms"])),
        6,
    )
    out["bwd_tflops_est"] = round(
        profiler._to_tflops(float(out["bwd_flops_est"]), float(out["bwd_ms"])),
        6,
    )
    out["total_tflops_est"] = round(
        profiler._to_tflops(float(out["total_flops_est"]), float(out["total_ms"])),
        6,
    )
    for key in (
        "fwd_calls",
        "bwd_calls",
        "fwd_tokens",
        "bwd_tokens",
        "fwd_attention_pairs",
        "bwd_attention_pairs",
    ):
        out[key] = int(base.get(key, 0))
    out["fwd_mfu"] = profiler._to_mfu(float(out["fwd_tflops_est"]))
    out["bwd_mfu"] = profiler._to_mfu(float(out["bwd_tflops_est"]))
    out["mfu"] = profiler._to_mfu(float(out["total_tflops_est"]))
    return out


def _install_gdn_operator_timing(profiler: Any) -> None:
    import art.megatron.gdn.operator as gdn_operator

    if getattr(gdn_operator, "_art_harness_gdn_timing_installed", False):
        return

    _wrap_gdn_function(
        profiler=profiler,
        owner=gdn_operator,
        name="_in_proj",
        layer_type="gdn_in_proj",
    )
    _wrap_gdn_function(
        profiler=profiler,
        owner=gdn_operator,
        name="_causal_conv1d_with_state",
        layer_type="gdn_conv",
    )
    _wrap_gdn_function(
        profiler=profiler,
        owner=gdn_operator,
        name="_causal_conv1d_varlen_with_state",
        layer_type="gdn_conv",
    )
    _wrap_gdn_function(
        profiler=profiler,
        owner=gdn_operator,
        name="_causal_conv1d_packed_varlen_with_state",
        layer_type="gdn_conv",
    )
    _wrap_gdn_function(
        profiler=profiler,
        owner=gdn_operator,
        name="_chunk_gated_delta_rule",
        layer_type="gdn_recurrent",
    )
    _wrap_gdn_function(
        profiler=profiler,
        owner=gdn_operator,
        name="_apply_gated_rms_norm",
        layer_type="gdn_norm_gate",
    )
    _wrap_gdn_function(
        profiler=profiler,
        owner=gdn_operator,
        name="_out_proj",
        layer_type="gdn_out_proj",
    )
    _wrap_gdn_nvtx_ranges(profiler=profiler, gdn_operator=gdn_operator)
    gdn_operator._art_harness_gdn_timing_installed = True


def _wrap_gdn_function(
    *,
    profiler: Any,
    owner: Any,
    name: str,
    layer_type: str,
) -> None:
    original = getattr(owner, name)
    if getattr(original, "__art_harness_gdn_timed__", False):
        return

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        tensor = profiler._find_first_tensor((args, kwargs))
        if tensor is None:
            return original(*args, **kwargs)
        token_count = profiler._tensor_token_count(tensor)
        record_name = _gdn_record_name(profiler, layer_type)
        record_id = profiler.start_synthetic_forward(
            module_name=record_name,
            layer_type=layer_type,
            device=tensor.device,
            token_count=token_count,
        )
        invocation = profiler.create_synthetic_backward_invocation(
            record_id=record_id,
            input_tensor_count=profiler.count_grad_tensors((args, kwargs)),
            token_count=token_count,
        )
        wrapped_args = profiler.wrap_input_boundaries(args, invocation)
        wrapped_kwargs = profiler.wrap_input_boundaries(kwargs, invocation)
        try:
            with profiler._active_forward_record(record_id):
                out = original(*wrapped_args, **wrapped_kwargs)
        finally:
            profiler.stop_synthetic_forward(record_id)
        return profiler.wrap_output_boundaries(out, invocation)

    setattr(wrapped, "__art_harness_gdn_timed__", True)
    setattr(owner, name, wrapped)


def _wrap_gdn_nvtx_ranges(*, profiler: Any, gdn_operator: Any) -> None:
    original_nvtx_range = gdn_operator._nvtx_range
    if getattr(original_nvtx_range, "__art_harness_gdn_timed__", False):
        return

    @contextmanager
    def timed_nvtx_range(label: str, tensor: Any = None) -> Any:
        if tensor is None:
            with original_nvtx_range(label, tensor):
                yield
            return
        record_id = profiler.start_synthetic_forward(
            module_name=f"{_gdn_record_name(profiler, 'gdn_range')}.{label}",
            layer_type="gdn_range",
            device=getattr(tensor, "device", None),
            token_count=profiler._tensor_token_count(tensor),
        )
        try:
            with original_nvtx_range(label, tensor):
                yield
        finally:
            profiler.stop_synthetic_forward(record_id)

    setattr(timed_nvtx_range, "__art_harness_gdn_timed__", True)
    gdn_operator._nvtx_range = timed_nvtx_range


def _gdn_record_name(profiler: Any, layer_type: str) -> str:
    parent_id = profiler._current_active_forward_module_id()
    if parent_id is None:
        return f"gdn_global.{layer_type}"
    parent = profiler._records.get(int(parent_id))
    parent_name = getattr(parent, "module_name", f"record_{parent_id}")
    return f"{parent_name}.{layer_type}"


def _run_harness_worker() -> int:
    _install_harness_import_path()
    from art_harness import megatron_train_with_provider_patch as provider_patch
    from art_harness import megatron_train_with_timing as timing_worker

    overrides = provider_patch._read_overrides()
    provider_patch._install_distributed_timeout_patch()
    provider_patch._install_provider_patch(overrides)
    _install_gdn_timing_overrides(timing_worker)
    return int(timing_worker.main())


def main() -> int:
    _install_ce_impl_override()
    return _run_harness_worker()


if __name__ == "__main__":
    raise SystemExit(main())
