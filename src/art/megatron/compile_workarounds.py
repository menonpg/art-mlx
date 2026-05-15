from __future__ import annotations

from importlib import import_module
import json
import os
import time
from typing import Any

import torch
import torch.distributed as dist

from art.megatron.model_support.spec import CompileWorkaroundConfig

_INSTALLED_CONFIG: tuple[frozenset[str], str] | None = None
_DEEPEP_DEBUG_COUNTERS: dict[str, int] = {}
_MOE_DEBUG_COUNTERS: dict[str, int] = {}
_SELF_ATTN_LINEAR_PROJ_REDUCE_SCATTER_WORKAROUND_FLAG = (
    "disable_compile_self_attn_linear_proj_reduce_scatter"
)


def _disable(fn):
    if getattr(fn, "__art_compile_disabled__", False):
        return fn
    fn = getattr(fn, "_torchdynamo_orig_callable", fn)
    wrapped = torch.compiler.disable(fn)
    setattr(wrapped, "__art_compile_disabled__", True)
    return wrapped


def _selected_workaround_flags(
    config: CompileWorkaroundConfig | None,
) -> set[str]:
    flags = set(() if config is None else config.flags)
    raw = os.environ.get("ART_MEGATRON_COMPILE_WORKAROUNDS", "").strip()
    if not raw:
        return flags
    if raw.lower() in {"none", "off"}:
        return flags
    return flags | {part.strip() for part in raw.split(",") if part.strip()}


def _optional_import_module(name: str) -> Any | None:
    try:
        return import_module(name)
    except ImportError:
        return None


def _install_context_parallel_attention_workaround() -> None:
    from art.megatron.context_parallel import core_attention, executor

    # CP attention owns custom comm and side-stream lifetime management. Keep
    # that wrapper eager; the inner flex attention kernels compile separately.
    executor.run_context_parallel = _disable(executor.run_context_parallel)
    core_attention.run_context_parallel = _disable(core_attention.run_context_parallel)
    core_attention.ArtContextParallelCoreAttention.forward = _disable(
        core_attention.ArtContextParallelCoreAttention.forward
    )


def _install_self_attn_linear_proj_reduce_scatter_workaround() -> None:
    from megatron.core.tensor_parallel import mappings

    from art.megatron import lora as art_lora

    # SelfAttentionLinearProjLoRA imports this symbol directly from
    # art.megatron.lora, so rebinding only megatron.core.tensor_parallel.mappings
    # leaves the compiled LoRA path untouched.
    wrapped = _disable(mappings.reduce_scatter_to_sequence_parallel_region)
    mappings.reduce_scatter_to_sequence_parallel_region = wrapped  # type: ignore[assignment]
    art_lora.reduce_scatter_to_sequence_parallel_region = wrapped  # type: ignore[assignment]


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _distributed_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():  # ty: ignore[possibly-missing-attribute]
        return int(os.environ.get("RANK", "0"))
    return int(dist.get_rank())  # ty: ignore[possibly-missing-attribute]


def _tensor_shape(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape)
    return None


def _cuda_memory_payload() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    return {
        "device": int(torch.cuda.current_device()),
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
    }


def _next_deepep_debug_count(name: str) -> int:
    count = _DEEPEP_DEBUG_COUNTERS.get(name, 0)
    _DEEPEP_DEBUG_COUNTERS[name] = count + 1
    return count


def _next_moe_debug_count(name: str) -> int:
    count = _MOE_DEBUG_COUNTERS.get(name, 0)
    _MOE_DEBUG_COUNTERS[name] = count + 1
    return count


def _deepep_debug_log(event: str, **payload: Any) -> None:
    if not _env_enabled("ART_MEGATRON_DEEPEP_DEBUG"):
        return
    message = (
        "ART_MEGATRON_DEEPEP_DEBUG_JSON="
        + json.dumps(
            {
                "event": event,
                "rank": _distributed_rank(),
                "time": time.time(),
                **_cuda_memory_payload(),
                **payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    os.write(1, message.encode("utf-8"))


def _moe_debug_log(event: str, **payload: Any) -> None:
    if not _env_enabled("ART_MEGATRON_MOE_DEBUG"):
        return
    message = (
        "ART_MEGATRON_MOE_DEBUG_JSON="
        + json.dumps(
            {
                "event": event,
                "rank": _distributed_rank(),
                "time": time.time(),
                **_cuda_memory_payload(),
                **payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    os.write(1, message.encode("utf-8"))


def _tokens_per_expert_payload(tokens_per_expert: Any) -> dict[str, Any]:
    if isinstance(tokens_per_expert, (list, tuple)):
        counts = torch.tensor(tokens_per_expert, dtype=torch.int64)
    elif isinstance(tokens_per_expert, torch.Tensor):
        counts = tokens_per_expert.detach().cpu().to(torch.int64)
    else:
        return {}
    if counts.numel() == 0:
        return {
            "tokens_per_expert_shape": tuple(int(dim) for dim in counts.shape),
            "tokens_total": 0,
            "tokens_max": 0,
            "tokens_min": 0,
            "tokens_nonzero": 0,
            "tokens_top": [],
        }
    top_count = min(8, int(counts.numel()))
    top_values, top_indices = torch.topk(counts, top_count)
    return {
        "tokens_per_expert_shape": tuple(int(dim) for dim in counts.shape),
        "tokens_total": int(counts.sum().item()),
        "tokens_max": int(counts.max().item()),
        "tokens_min": int(counts.min().item()),
        "tokens_nonzero": int((counts != 0).sum().item()),
        "tokens_top": [
            [int(index), int(value)]
            for index, value in zip(
                top_indices.tolist(), top_values.tolist(), strict=True
            )
        ],
    }


def _install_moe_debug_wrappers(moe_experts: Any) -> None:
    grouped_mlp = getattr(moe_experts, "TEGroupedMLP", None)
    if grouped_mlp is None:
        return

    def install_inline_grouped_mlp_forward() -> bool:
        if not _env_enabled("ART_MEGATRON_MOE_DEBUG_INLINE_FORWARD"):
            return False
        original = getattr(grouped_mlp, "forward", None)
        if original is None or getattr(original, "__art_moe_debug_wrapped__", False):
            return True

        from megatron.core import tensor_parallel
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )
        from megatron.core.typed_torch import apply_module

        def wrapped(
            self: Any,
            permuted_local_hidden_states: torch.Tensor,
            tokens_per_expert: Any,
            permuted_probs: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            counter = _next_moe_debug_count("te_grouped_mlp_forward")
            tokens_payload = tokens_per_expert
            start_time = time.time()
            _moe_debug_log(
                "te_grouped_mlp_forward_enter",
                count=counter,
                module_id=id(self),
                hidden_shape=_tensor_shape(permuted_local_hidden_states),
                probs_shape=_tensor_shape(permuted_probs),
                activation_recompute=bool(getattr(self, "activation_recompute", False)),
                offload_expert_fc1=bool(getattr(self, "offload_expert_fc1", False)),
                offload_moe_act=bool(getattr(self, "offload_moe_act", False)),
                inline_forward=True,
                **_tokens_per_expert_payload(tokens_payload),
            )
            if isinstance(tokens_per_expert, torch.Tensor):
                tokens_per_expert = tokens_per_expert.tolist()
            else:
                tokens_per_expert = list(tokens_per_expert)
            if self.config.fp8 or self.config.fp4:
                actual_tokens_per_expert = tokens_per_expert
                permuted_local_hidden_states, tokens_per_expert = (
                    self.quantization_padding(
                        permuted_local_hidden_states, tokens_per_expert
                    )
                )
                permuted_probs, _ = self.quantization_padding(
                    permuted_probs.unsqueeze(-1), actual_tokens_per_expert
                )
            else:
                actual_tokens_per_expert = None
                permuted_probs = permuted_probs.unsqueeze(-1)

            if self.config.moe_apply_probs_on_input:
                assert self.config.moe_router_topk == 1
                original_dtype = permuted_local_hidden_states.dtype
                permuted_local_hidden_states = (
                    permuted_probs * permuted_local_hidden_states
                )
                permuted_local_hidden_states = permuted_local_hidden_states.to(
                    original_dtype
                )
                permuted_probs = torch.ones_like(permuted_probs)

            with off_interface(
                self.offload_expert_fc1, permuted_local_hidden_states, "expert_fc1"
            ) as permuted_local_hidden_states:
                fc1_output, bias_parallel = apply_module(self.linear_fc1)(
                    permuted_local_hidden_states, tokens_per_expert
                )
            if self.offload_expert_fc1:
                fc1_output = off_interface.group_commit(
                    fc1_output,
                    name="expert_fc1",
                    forced_released_tensors=[permuted_local_hidden_states],
                )

            if self.activation_recompute:
                self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
                with off_interface(
                    self.offload_moe_act, fc1_output, "moe_act"
                ) as fc1_output:
                    bias_act_output = self.activation_checkpoint.checkpoint(
                        self.bias_act_func, fc1_output, bias_parallel, permuted_probs
                    )
            else:
                with off_interface(
                    self.offload_moe_act, fc1_output, "moe_act"
                ) as fc1_output:
                    bias_act_output = self.bias_act_func(
                        fc1_output, bias_parallel, permuted_probs
                    )

            _moe_debug_log(
                "te_grouped_mlp_inline_before_fc2",
                count=counter,
                module_id=id(self),
                hidden_shape=_tensor_shape(bias_act_output),
                **_tokens_per_expert_payload(tokens_payload),
            )
            output, output_bias = apply_module(self.linear_fc2)(
                bias_act_output, tokens_per_expert
            )
            _moe_debug_log(
                "te_grouped_mlp_inline_after_fc2",
                count=counter,
                module_id=id(self),
                result_shape=_tensor_shape(output),
                bias_shape=_tensor_shape(output_bias),
                activation_recompute=bool(self.activation_recompute),
                offload_moe_act=bool(self.offload_moe_act),
            )
            if self.activation_recompute:
                _moe_debug_log(
                    "te_grouped_mlp_inline_before_recompute_discard",
                    count=counter,
                    module_id=id(self),
                    result_shape=_tensor_shape(output),
                )
                self.activation_checkpoint.discard_output_and_register_recompute(output)
            if self.offload_moe_act:
                _moe_debug_log(
                    "te_grouped_mlp_inline_before_moe_act_commit",
                    count=counter,
                    module_id=id(self),
                    result_shape=_tensor_shape(output),
                )
                output = off_interface.group_commit(
                    output, name="moe_act", forced_released_tensors=[fc1_output]
                )
            _moe_debug_log(
                "te_grouped_mlp_inline_before_apply_bias",
                count=counter,
                module_id=id(self),
                result_shape=_tensor_shape(output),
                bias_shape=_tensor_shape(output_bias),
                probs_shape=_tensor_shape(permuted_probs),
            )
            output = self._apply_bias(
                output, output_bias, tokens_per_expert, permuted_probs
            )
            _moe_debug_log(
                "te_grouped_mlp_inline_after_apply_bias",
                count=counter,
                module_id=id(self),
                result_shape=_tensor_shape(output),
            )
            if self.config.fp8 or self.config.fp4:
                output = self.quantization_unpadding(output, actual_tokens_per_expert)
            _moe_debug_log(
                "te_grouped_mlp_forward_exit",
                count=counter,
                module_id=id(self),
                elapsed_ms=(time.time() - start_time) * 1000.0,
                result_shape=_tensor_shape(output),
                inline_forward=True,
            )
            return output, None

        setattr(wrapped, "__art_moe_debug_wrapped__", True)
        grouped_mlp.forward = wrapped
        return True

    def wrap_grouped_mlp_forward() -> None:
        original = getattr(grouped_mlp, "forward", None)
        if original is None or getattr(original, "__art_moe_debug_wrapped__", False):
            return

        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            counter = _next_moe_debug_count("te_grouped_mlp_forward")
            hidden_states = (
                args[0]
                if len(args) >= 1
                else kwargs.get("permuted_local_hidden_states")
            )
            tokens_per_expert = (
                args[1] if len(args) >= 2 else kwargs.get("tokens_per_expert")
            )
            permuted_probs = args[2] if len(args) >= 3 else kwargs.get("permuted_probs")
            start_time = time.time()
            _moe_debug_log(
                "te_grouped_mlp_forward_enter",
                count=counter,
                module_id=id(self),
                hidden_shape=_tensor_shape(hidden_states),
                probs_shape=_tensor_shape(permuted_probs),
                activation_recompute=bool(getattr(self, "activation_recompute", False)),
                offload_expert_fc1=bool(getattr(self, "offload_expert_fc1", False)),
                offload_moe_act=bool(getattr(self, "offload_moe_act", False)),
                **_tokens_per_expert_payload(tokens_per_expert),
            )
            result = original(self, *args, **kwargs)
            elapsed_ms = (time.time() - start_time) * 1000.0
            output = result[0] if isinstance(result, tuple) and result else result
            _moe_debug_log(
                "te_grouped_mlp_forward_exit",
                count=counter,
                module_id=id(self),
                elapsed_ms=elapsed_ms,
                result_shape=_tensor_shape(output),
            )
            return result

        setattr(wrapped, "__art_moe_debug_wrapped__", True)
        grouped_mlp.forward = _disable(wrapped)

    def wrap_bias_act_func() -> None:
        original = getattr(grouped_mlp, "bias_act_func", None)
        if original is None or getattr(original, "__art_moe_debug_wrapped__", False):
            return

        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            counter = _next_moe_debug_count("te_grouped_mlp_bias_act")
            start_time = time.time()
            _moe_debug_log(
                "te_grouped_mlp_bias_act_enter",
                count=counter,
                module_id=id(self),
                hidden_shape=_tensor_shape(args[0] if args else None),
                probs_shape=_tensor_shape(args[2] if len(args) >= 3 else None),
            )
            result = original(self, *args, **kwargs)
            _moe_debug_log(
                "te_grouped_mlp_bias_act_exit",
                count=counter,
                module_id=id(self),
                elapsed_ms=(time.time() - start_time) * 1000.0,
                result_shape=_tensor_shape(result),
            )
            return result

        setattr(wrapped, "__art_moe_debug_wrapped__", True)
        grouped_mlp.bias_act_func = _disable(wrapped)

    def wrap_apply_bias() -> None:
        original = getattr(grouped_mlp, "_apply_bias", None)
        if original is None or getattr(original, "__art_moe_debug_wrapped__", False):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            counter = _next_moe_debug_count("te_grouped_mlp_apply_bias")
            start_time = time.time()
            output = args[0] if len(args) >= 1 else None
            bias = args[1] if len(args) >= 2 else None
            tokens_per_expert = args[2] if len(args) >= 3 else None
            probs = args[3] if len(args) >= 4 else None
            _moe_debug_log(
                "te_grouped_mlp_apply_bias_enter",
                count=counter,
                hidden_shape=_tensor_shape(output),
                bias_shape=_tensor_shape(bias),
                probs_shape=_tensor_shape(probs),
                **_tokens_per_expert_payload(tokens_per_expert),
            )
            result = original(*args, **kwargs)
            _moe_debug_log(
                "te_grouped_mlp_apply_bias_exit",
                count=counter,
                elapsed_ms=(time.time() - start_time) * 1000.0,
                result_shape=_tensor_shape(result),
            )
            return result

        setattr(wrapped, "__art_moe_debug_wrapped__", True)
        grouped_mlp._apply_bias = staticmethod(_disable(wrapped))

    def wrap_grouped_linear(cls: Any, label: str) -> None:
        original = getattr(cls, "forward", None)
        if original is None or getattr(original, "__art_moe_debug_wrapped__", False):
            return

        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            counter = _next_moe_debug_count(label)
            start_time = time.time()
            hidden_states = args[0] if args else None
            tokens_per_expert = args[1] if len(args) >= 2 else None
            _moe_debug_log(
                f"{label}_enter",
                count=counter,
                module_id=id(self),
                hidden_shape=_tensor_shape(hidden_states),
                **_tokens_per_expert_payload(tokens_per_expert),
            )
            result = original(self, *args, **kwargs)
            output = result[0] if isinstance(result, tuple) and result else result
            sync_target = os.environ.get("ART_MEGATRON_MOE_DEBUG_SYNC_LINEAR", "")
            if sync_target and sync_target in {"1", "true", "all", label}:
                _moe_debug_log(
                    f"{label}_sync_enter",
                    count=counter,
                    module_id=id(self),
                    result_shape=_tensor_shape(output),
                )
                torch.cuda.synchronize()
                _moe_debug_log(
                    f"{label}_sync_exit",
                    count=counter,
                    module_id=id(self),
                    result_shape=_tensor_shape(output),
                )
            _moe_debug_log(
                f"{label}_exit",
                count=counter,
                module_id=id(self),
                elapsed_ms=(time.time() - start_time) * 1000.0,
                result_shape=_tensor_shape(output),
            )
            return result

        setattr(wrapped, "__art_moe_debug_wrapped__", True)
        cls.forward = _disable(wrapped)

    def wrap_offload_group_commit() -> None:
        try:
            from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
                FineGrainedActivationOffloadingInterface as off_interface,
            )
        except ImportError:
            return
        original = getattr(off_interface, "group_commit", None)
        if original is None or getattr(original, "__art_moe_debug_wrapped__", False):
            return

        def wrapped(
            tensor: torch.Tensor,
            name: str,
            forced_released_tensors: Any = None,
            delay_offload: bool = False,
        ) -> torch.Tensor:
            counter = _next_moe_debug_count("fine_grained_group_commit")
            start_time = time.time()
            _moe_debug_log(
                "fine_grained_group_commit_enter",
                count=counter,
                name=name,
                hidden_shape=_tensor_shape(tensor),
                forced_count=(
                    len(forced_released_tensors)
                    if forced_released_tensors is not None
                    else 0
                ),
                delay_offload=bool(delay_offload),
            )
            result = original(tensor, name, forced_released_tensors, delay_offload)
            _moe_debug_log(
                "fine_grained_group_commit_exit",
                count=counter,
                name=name,
                elapsed_ms=(time.time() - start_time) * 1000.0,
                result_shape=_tensor_shape(result),
            )
            return result

        setattr(wrapped, "__art_moe_debug_wrapped__", True)
        setattr(off_interface, "group_commit", staticmethod(_disable(wrapped)))

    inline_forward = install_inline_grouped_mlp_forward()
    if not inline_forward:
        wrap_grouped_mlp_forward()
    wrap_bias_act_func()
    wrap_apply_bias()
    wrap_offload_group_commit()
    try:
        from megatron.core.extensions import transformer_engine as te_ext
    except ImportError:
        return
    wrap_grouped_linear(
        getattr(te_ext, "TEColumnParallelGroupedLinear", None),
        "te_grouped_mlp_fc1",
    )
    wrap_grouped_linear(
        getattr(te_ext, "TERowParallelGroupedLinear", None),
        "te_grouped_mlp_fc2",
    )


def _install_deepep_debug_wrappers(deepep_manager: Any) -> None:
    force_sync = _env_enabled("ART_MEGATRON_DEEPEP_FORCE_SYNC")
    if (
        getattr(deepep_manager, "__art_deepep_debug_wrapped__", False)
        and not force_sync
    ):
        return

    def wrap_method(name: str) -> None:
        original = getattr(deepep_manager, name, None)
        if original is None or getattr(original, "__art_deepep_debug_wrapped__", False):
            return

        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            if force_sync and name in {"dispatch", "combine"}:
                args_list = list(args)
                if len(args_list) >= 2:
                    args_list[1] = False
                else:
                    kwargs["async_finish"] = False
                if len(args_list) >= 3:
                    args_list[2] = False
                else:
                    kwargs["allocate_on_comm_stream"] = False
                args = tuple(args_list)
            counter = _next_deepep_debug_count(name)
            start_time = time.time()
            _deepep_debug_log(
                f"{name}_enter",
                count=counter,
                manager_id=id(self),
                hidden_shape=_tensor_shape(args[0] if args else None),
                token_indices_shape=_tensor_shape(getattr(self, "token_indices", None)),
                token_probs_shape=_tensor_shape(getattr(self, "token_probs", None)),
                async_finish=(
                    (args[1] if len(args) >= 2 else kwargs.get("async_finish"))
                    if name in {"dispatch", "combine"}
                    else None
                ),
                allocate_on_comm_stream=(
                    (
                        args[2]
                        if len(args) >= 3
                        else kwargs.get("allocate_on_comm_stream")
                    )
                    if name in {"dispatch", "combine"}
                    else None
                ),
                force_sync=force_sync,
            )
            result = original(self, *args, **kwargs)
            payload = {}
            if _env_enabled("ART_MEGATRON_DEEPEP_DEBUG"):
                payload.update(
                    _tokens_per_expert_payload(getattr(self, "tokens_per_expert", None))
                )
            _deepep_debug_log(
                f"{name}_exit",
                count=counter,
                elapsed_ms=(time.time() - start_time) * 1000.0,
                manager_id=id(self),
                result_shape=_tensor_shape(result),
                force_sync=force_sync,
                **payload,
            )
            return result

        setattr(wrapped, "__art_deepep_debug_wrapped__", True)
        setattr(deepep_manager, name, _disable(wrapped))

    for method_name in (
        "setup_metadata",
        "dispatch",
        "get_permuted_hidden_states_by_experts",
        "get_restored_hidden_states_by_experts",
        "combine",
    ):
        wrap_method(method_name)
    setattr(deepep_manager, "__art_deepep_debug_wrapped__", True)


def install_torch_compile_workarounds(
    config: CompileWorkaroundConfig | None = None,
) -> None:
    global _INSTALLED_CONFIG
    flags = _selected_workaround_flags(config)
    shared_expert_state = "none" if config is None else config.shared_expert_state
    installed_config = (frozenset(flags), shared_expert_state)
    if _INSTALLED_CONFIG is not None:
        if _INSTALLED_CONFIG != installed_config:
            raise RuntimeError(
                "torch.compile workarounds already installed with a different config"
            )
        return
    from megatron.core.extensions import transformer_engine as te_ext
    from megatron.core.transformer.moe import experts as moe_experts
    from megatron.core.transformer.moe import moe_layer, moe_utils, token_dispatcher

    if "fake_sync_dealloc" in flags:
        try:

            @torch.library.register_fake("streams::sync_dealloc")
            def _sync_dealloc_fake(
                wait_event_index: int,
                src_stream_index: int,
                to_dealloc: torch.Tensor,
            ) -> None:
                del wait_event_index, src_stream_index, to_dealloc
                return None
        except RuntimeError as exc:
            if "already has a fake impl registered" not in str(exc):
                raise

    if "context_parallel_attention" in flags:
        _install_context_parallel_attention_workaround()
    if _SELF_ATTN_LINEAR_PROJ_REDUCE_SCATTER_WORKAROUND_FLAG in flags:
        _install_self_attn_linear_proj_reduce_scatter_workaround()

    deepep_manager = getattr(token_dispatcher, "_DeepepManager", None)
    if deepep_manager is not None:
        if "deepep_permute_restore" in flags:
            deepep_manager.get_permuted_hidden_states_by_experts = _disable(
                deepep_manager.get_permuted_hidden_states_by_experts
            )
            deepep_manager.get_restored_hidden_states_by_experts = _disable(
                deepep_manager.get_restored_hidden_states_by_experts
            )
        if "deepep_dispatch_combine" in flags:
            deepep_manager.dispatch = _disable(deepep_manager.dispatch)
            deepep_manager.combine = _disable(deepep_manager.combine)
        if _env_enabled("ART_MEGATRON_DEEPEP_DEBUG") or _env_enabled(
            "ART_MEGATRON_DEEPEP_FORCE_SYNC"
        ):
            _install_deepep_debug_wrappers(deepep_manager)
    if "alltoall_dtoh" in flags:
        token_dispatcher.MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize = (
            _disable(
                token_dispatcher.MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize
            )
        )
    if "alltoall_dispatch_preprocess" in flags:
        token_dispatcher.MoEAlltoAllTokenDispatcher.dispatch_preprocess = _disable(
            token_dispatcher.MoEAlltoAllTokenDispatcher.dispatch_preprocess
        )
    if "alltoall_combine_postprocess" in flags:
        token_dispatcher.MoEAlltoAllTokenDispatcher.combine_postprocess = _disable(
            token_dispatcher.MoEAlltoAllTokenDispatcher.combine_postprocess
        )
    if "te_moe_permute_with_probs" in flags:
        te_permutation = _optional_import_module(
            "transformer_engine.pytorch.permutation"
        )
        if te_permutation is not None:
            te_permutation.moe_permute_with_probs = _disable(
                te_permutation.moe_permute_with_probs
            )
        if te_ext.fused_permute_with_probs is not None:
            te_ext.fused_permute_with_probs = _disable(te_ext.fused_permute_with_probs)
        fused_permute_with_probs = getattr(moe_utils, "fused_permute_with_probs", None)
        if fused_permute_with_probs is not None:
            moe_utils.fused_permute_with_probs = _disable(fused_permute_with_probs)
    if "te_triton_permute_with_mask_map" in flags:
        te_triton_permutation = _optional_import_module(
            "transformer_engine.pytorch.triton.permutation"
        )
        if te_triton_permutation is not None:
            te_triton_permutation.make_row_id_map = _disable(
                te_triton_permutation.make_row_id_map
            )
            te_triton_permutation.permute_with_mask_map = _disable(
                te_triton_permutation.permute_with_mask_map
            )
            te_triton_permutation.unpermute_with_mask_map = _disable(
                te_triton_permutation.unpermute_with_mask_map
            )
    if "te_moe_unpermute" in flags:
        te_permutation = _optional_import_module(
            "transformer_engine.pytorch.permutation"
        )
        if te_permutation is not None:
            te_permutation.moe_unpermute = _disable(te_permutation.moe_unpermute)
        if te_ext.fused_unpermute is not None:
            te_ext.fused_unpermute = _disable(te_ext.fused_unpermute)
        fused_unpermute = getattr(moe_utils, "fused_unpermute", None)
        if fused_unpermute is not None:
            moe_utils.fused_unpermute = _disable(fused_unpermute)
    if "moe_utils_permute" in flags:
        moe_utils.permute = _disable(moe_utils.permute)
    if "moe_utils_unpermute" in flags:
        moe_utils.unpermute = _disable(moe_utils.unpermute)
    if "te_moe_unpermute_backward" in flags:
        te_permutation = _optional_import_module(
            "transformer_engine.pytorch.permutation"
        )
        if te_permutation is not None:
            setattr(
                te_permutation._moe_unpermute_mask_map,
                "backward",
                staticmethod(_disable(te_permutation._moe_unpermute_mask_map.backward)),
            )
    if "te_triton_unpermute_bwd_with_merging_probs" in flags:
        te_triton_permutation = _optional_import_module(
            "transformer_engine.pytorch.triton.permutation"
        )
        if te_triton_permutation is not None:
            te_triton_permutation.unpermute_with_mask_map_bwd_with_merging_probs = (
                _disable(
                    te_triton_permutation.unpermute_with_mask_map_bwd_with_merging_probs
                )
            )
    if "flex_token_dispatch_combine" in flags:
        token_dispatcher.MoEFlexTokenDispatcher.token_dispatch = _disable(
            token_dispatcher.MoEFlexTokenDispatcher.token_dispatch
        )
        token_dispatcher.MoEFlexTokenDispatcher.token_combine = _disable(
            token_dispatcher.MoEFlexTokenDispatcher.token_combine
        )
    if "flex_token_dispatch_preprocess" in flags:
        token_dispatcher.MoEFlexTokenDispatcher.dispatch_preprocess = _disable(
            token_dispatcher.MoEFlexTokenDispatcher.dispatch_preprocess
        )
    if "moe_preprocess" in flags:
        moe_layer.MoELayer.preprocess = _disable(moe_layer.MoELayer.preprocess)
    if "moe_forward" in flags:
        moe_layer.MoELayer.forward = _disable(moe_layer.MoELayer.forward)
    if "moe_routed_experts_compute" in flags:
        moe_layer.MoELayer.routed_experts_compute = _disable(
            moe_layer.MoELayer.routed_experts_compute
        )
    if "grouped_mlp_forward" in flags:
        moe_experts.GroupedMLP.forward = _disable(moe_experts.GroupedMLP.forward)
    if "te_grouped_mlp_forward" in flags:
        moe_experts.TEGroupedMLP.forward = _disable(moe_experts.TEGroupedMLP.forward)
    if _env_enabled("ART_MEGATRON_MOE_DEBUG"):
        _install_moe_debug_wrappers(moe_experts)
    _INSTALLED_CONFIG = installed_config


def install_debug_wrappers_if_requested() -> None:
    if not (
        _env_enabled("ART_MEGATRON_DEEPEP_DEBUG")
        or _env_enabled("ART_MEGATRON_DEEPEP_FORCE_SYNC")
        or _env_enabled("ART_MEGATRON_MOE_DEBUG")
    ):
        return
    from megatron.core.transformer.moe import experts as moe_experts
    from megatron.core.transformer.moe import token_dispatcher

    deepep_manager = getattr(token_dispatcher, "_DeepepManager", None)
    if deepep_manager is not None and (
        _env_enabled("ART_MEGATRON_DEEPEP_DEBUG")
        or _env_enabled("ART_MEGATRON_DEEPEP_FORCE_SYNC")
    ):
        _install_deepep_debug_wrappers(deepep_manager)
    if _env_enabled("ART_MEGATRON_MOE_DEBUG"):
        _install_moe_debug_wrappers(moe_experts)
