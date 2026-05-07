from __future__ import annotations

from collections.abc import Iterable, Mapping
import contextlib
import fnmatch
from typing import Any

from megatron.bridge.models.common.unimodal import to_empty_if_meta_device
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    MegatronParamMapping,
    ReplicatedMapping,
    get_module_and_param_from_name,
)
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.utils import get_model_config
import torch


def _pin_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu" or not torch.cuda.is_available():
        return tensor
    try:
        return tensor if tensor.is_pinned() else tensor.pin_memory()
    except RuntimeError:
        return tensor


def _iter_hf_param_names(hf_param: Any) -> Iterable[str]:
    if isinstance(hf_param, str):
        yield hf_param
        return
    if isinstance(hf_param, Mapping):
        for value in hf_param.values():
            yield from _iter_hf_param_names(value)


def _needs_local_hf_prefetch(task: Any) -> bool:
    if task is None or task.megatron_module is None:
        return False
    mapping = task.mapping
    tp_size = int(getattr(mapping, "tp_size", 1))
    if tp_size <= 1:
        return True
    if type(mapping).__name__ == "DirectMapping":
        return True
    return int(getattr(mapping, "tp_rank", 0)) == 0


def load_unique_hf_keys_once(
    tasks: Iterable[Any],
    hf_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    keys = sorted(
        {
            key
            for task in tasks
            if _needs_local_hf_prefetch(task)
            for key in _iter_hf_param_names(task.mapping.hf_param)
        }
    )
    if not keys:
        return {}
    if hasattr(hf_state_dict, "__getitem__"):
        loaded = (
            hf_state_dict[keys]
            if not isinstance(hf_state_dict, dict)
            else {key: hf_state_dict[key] for key in keys}
        )
    else:
        loaded = {key: hf_state_dict[key] for key in keys}
    return {key: _pin_cpu_tensor(value) for key, value in loaded.items()}


class _CachedStateLookup(Mapping[str, torch.Tensor]):
    def __init__(
        self,
        *,
        cache: Mapping[str, torch.Tensor],
        source: Mapping[str, torch.Tensor],
    ) -> None:
        self._cache = cache
        self._source = source

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self._cache:
            return self._cache[key]
        return _pin_cpu_tensor(self._source[key])

    def __iter__(self):
        seen = set(self._cache)
        yield from self._cache
        for key in self._source:
            if key not in seen:
                yield key

    def __len__(self) -> int:
        return len(set(self._cache).union(self._source))


def _materialization_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _apply_pre_wrap_hook(
    model: list[MegatronModule],
    pre_wrap_hook: Any,
) -> list[MegatronModule]:
    if pre_wrap_hook is None:
        return model
    if not callable(pre_wrap_hook):
        raise RuntimeError("pre_wrap_hook must be callable")
    updated = pre_wrap_hook(model)
    return model if updated is None else updated


def _set_tp_attrs(model: list[MegatronModule]) -> None:
    from megatron.core import tensor_parallel

    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                param
            )


def _wrap_with_mp_wrapper(
    model: list[MegatronModule],
    model_config: Any,
    mixed_precision_wrapper: Any,
) -> list[MegatronModule]:
    if not (model_config.fp16 or model_config.bf16) or mixed_precision_wrapper is None:
        return model
    keep_in_fp32: list[tuple[Any, torch.Tensor]] = []
    for model_module in model:
        for submodule in model_module.modules():
            if hasattr(submodule, "_maintain_float32_expert_bias"):
                expert_bias = getattr(submodule, "expert_bias", None)
                if expert_bias is not None:
                    keep_in_fp32.append((submodule, expert_bias.data.clone()))
    wrapped = [
        mixed_precision_wrapper(model_config, model_module) for model_module in model
    ]
    for submodule, fp32_data in keep_in_fp32:
        submodule.expert_bias.data = fp32_data
    return wrapped


def _art_get_model(
    model_provider: ModelProviderMixin,
    ddp_config: DistributedDataParallelConfig,
    model_type=ModelType.encoder_or_decoder,
    overlap_param_gather_with_optimizer_step: bool = False,
    fp16: bool | None = None,
    bf16: bool | None = None,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    wrap_with_ddp: bool = True,
    data_parallel_random_init: bool = False,
    use_cpu_initialization: None | bool = False,
    init_model_with_meta_device: bool | None = None,
    pre_wrap_hook: Any = None,
    mixed_precision_wrapper: Any = Float16Module,
    *,
    pg_collection: ProcessGroupCollection,
) -> list[MegatronModule]:
    from megatron.bridge.models import model_provider as model_provider_module

    if fp16:
        model_provider.fp16 = fp16
    if bf16:
        model_provider.bf16 = bf16

    model_provider.use_cpu_initialization = bool(use_cpu_initialization)
    if init_model_with_meta_device:
        model_provider.init_model_with_meta_device = True
        with torch.device("meta"):
            model = model_provider_module._create_model(
                model_provider,
                model_type,
                pg_collection=pg_collection,
            )
    else:
        model = model_provider_module._create_model(
            model_provider,
            model_type,
            pg_collection=pg_collection,
        )

    if init_model_with_meta_device and not use_torch_fsdp2 and not use_megatron_fsdp:
        device = _materialization_device()
        model = [
            to_empty_if_meta_device(model_module, device=device)
            for model_module in model
        ]

    model = _apply_pre_wrap_hook(model, pre_wrap_hook)
    _set_tp_attrs(model)
    model_provider_module._print_num_params(model, pg_collection=pg_collection)
    model_config = get_model_config(model[0])

    if (
        not use_torch_fsdp2
        and not model_config.use_cpu_initialization
        and not model_config.init_model_with_meta_device
    ):
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    model = _wrap_with_mp_wrapper(model, model_config, mixed_precision_wrapper)
    if model_provider_module.correct_amax_history_if_needed is not None:
        model_provider_module.correct_amax_history_if_needed(model)
    if wrap_with_ddp:
        model = model_provider_module._ddp_wrap(
            model,
            data_parallel_random_init,
            ddp_config,
            overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp=use_megatron_fsdp,
            use_torch_fsdp2=use_torch_fsdp2,
            pg_collection=pg_collection,
        )
    return model


def _column_parallel_hf_to_megatron(
    self: ColumnParallelMapping,
    hf_weights: torch.Tensor,
    megatron_module: torch.nn.Module,
) -> torch.Tensor:
    if self.tp_size == 1:
        return hf_weights
    normalized_param = self._normalize_expert_param_name(self.megatron_param)
    _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
    if self.tp_rank == 0:
        full_size = hf_weights.shape[0]
        if full_size % self.tp_size != 0:
            raise ValueError(
                f"Cannot evenly split dimension 0 size {full_size} across {self.tp_size} TP ranks"
            )
        splits = torch.chunk(hf_weights, self.tp_size, dim=0)
    else:
        splits = None
    return self.scatter_to_tp_ranks(
        splits,
        target_param.shape,
        target_param.dtype,
        target_param.device,
    )


def _scatter_to_tp_ranks(
    self: MegatronParamMapping,
    splits: list[torch.Tensor] | None,
    output_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    src_rank: int = 0,
) -> torch.Tensor:
    if self.tp_size == 1:
        if not splits:
            return None
        return splits[0].to(device=device, dtype=dtype, non_blocking=True)
    output = torch.empty(output_shape, dtype=dtype, device=device)
    global_src = torch.distributed.get_global_rank(
        group=self.tp_group, group_rank=src_rank
    )
    scatter_list = None
    if self.tp_rank == src_rank and splits:
        scatter_list = [
            shard.to(device=device, dtype=dtype, non_blocking=True) for shard in splits
        ]
    torch.distributed.scatter(output, scatter_list, src=global_src, group=self.tp_group)
    return output


def _replicated_hf_to_megatron(
    self: ReplicatedMapping,
    hf_weights: torch.Tensor,
    megatron_module: torch.nn.Module,
) -> torch.Tensor:
    if hasattr(megatron_module, "weight"):
        target_device = megatron_module.weight.device
    else:
        target_device = next(megatron_module.parameters()).device
    if self.tp_size == 1:
        return hf_weights.to(device=target_device, non_blocking=True)
    broadcast_device = target_device
    if (
        broadcast_device.type != "cuda"
        or broadcast_device.index != torch.cuda.current_device()
    ):
        broadcast_device = _materialization_device()
    if self.tp_rank == 0:
        tensor = hf_weights.to(device=broadcast_device, non_blocking=True)
    else:
        tensor = torch.empty_like(hf_weights, device=broadcast_device)
    return self.broadcast_tensor_to_tp_ranks(tensor, src_rank=0)


def _optimized_load_weights_hf_to_megatron(
    self: MegatronModelBridge,
    hf_pretrained: Any,
    megatron_model: Any,
    allowed_mismatched_params: list[str] | None = None,
) -> list[Any]:
    if not isinstance(megatron_model, list):
        megatron_model = [megatron_model]
    with contextlib.ExitStack() as stack:
        if hasattr(megatron_model[0], "hide_teacher_model"):
            stack.enter_context(megatron_model[0].hide_teacher_model())
        if hasattr(megatron_model[0], "hide_loss_modules"):
            stack.enter_context(megatron_model[0].hide_loss_modules())
        tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)
    hf_state_dict = hf_pretrained.state
    raw_cache = load_unique_hf_keys_once(tasks, hf_state_dict)
    cached_state = _CachedStateLookup(cache=raw_cache, source=hf_state_dict)
    description = f"Loading from {hf_pretrained.model_name_or_path}"
    pending_device_copy = False
    for task in self._with_progress_tracking(tasks, description):
        if task is None or task.megatron_module is None:
            continue
        hf_weights = self.maybe_modify_loaded_hf_weight(
            task.mapping.hf_param, cached_state
        )
        converted_weights = task.mapping.hf_to_megatron(
            hf_weights, task.megatron_module
        )
        if converted_weights is None:
            continue
        assert task.param_weight is not None, (
            "param_weight is required for HF->Megatron conversion"
        )
        if converted_weights.shape != task.param_weight.shape:
            is_whitelisted = False
            if allowed_mismatched_params:
                for pattern in allowed_mismatched_params:
                    if fnmatch.fnmatch(
                        task.mapping.megatron_param, pattern
                    ) or fnmatch.fnmatch(task.param_name, pattern):
                        is_whitelisted = True
                        break
            if is_whitelisted:
                continue
            raise ValueError(
                f"Shape mismatch for megatron param {task.mapping.megatron_param}:\n"
                f"  Expected shape: {task.param_weight.shape}\n"
                f"  Got shape: {converted_weights.shape}\n"
                f"  Bridge type: {type(task.mapping).__name__}\n"
                f"  HF mapping: {task.mapping.hf_param}"
            )
        task.param_weight.data.copy_(converted_weights, non_blocking=True)
        if task.param_weight.device.type == "cuda":
            pending_device_copy = True
    if pending_device_copy and torch.cuda.is_available():
        torch.cuda.synchronize()
    self._broadcast_shared_embeddings(megatron_model)
    return megatron_model


def install_art_bridge_runtime_patches() -> None:
    from megatron.bridge.models import model_provider as model_provider_module

    if not getattr(
        model_provider_module.get_model, "__art_meta_materialization__", False
    ):
        setattr(_art_get_model, "__art_meta_materialization__", True)
        model_provider_module.get_model = _art_get_model
    if not getattr(
        MegatronParamMapping.scatter_to_tp_ranks, "__art_non_blocking__", False
    ):
        setattr(_scatter_to_tp_ranks, "__art_non_blocking__", True)
        MegatronParamMapping.scatter_to_tp_ranks = _scatter_to_tp_ranks
    if not getattr(ColumnParallelMapping.hf_to_megatron, "__art_cast_last__", False):
        setattr(_column_parallel_hf_to_megatron, "__art_cast_last__", True)
        ColumnParallelMapping.hf_to_megatron = _column_parallel_hf_to_megatron
    if not getattr(ReplicatedMapping.hf_to_megatron, "__art_cast_last__", False):
        setattr(_replicated_hf_to_megatron, "__art_cast_last__", True)
        ReplicatedMapping.hf_to_megatron = _replicated_hf_to_megatron
    if not getattr(
        MegatronModelBridge.load_weights_hf_to_megatron, "__art_cached_load__", False
    ):
        setattr(_optimized_load_weights_hf_to_megatron, "__art_cached_load__", True)
        MegatronModelBridge.load_weights_hf_to_megatron = (
            _optimized_load_weights_hf_to_megatron
        )
