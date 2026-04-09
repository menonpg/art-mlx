from itertools import chain
from typing import Any, Iterator, cast

from pydantic import BaseModel, ConfigDict
import torch

from art.megatron.model_chunks import ModelChunks, as_megatron_api_chunks
from art.megatron.param_name_canonicalization import (
    canonical_art_param_name,
    is_art_adapter_param_name,
)


class MergedWeightExport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bridge: Any
    model: ModelChunks
    model_config_value: Any
    conversion_tasks: list[Any]
    adapter_weights_by_base: dict[str, list[Any]]


def _mapping_hf_weights_exist(mapping: Any, hf_keys: set[str]) -> bool:
    if getattr(mapping, "allow_hf_name_mismatch", False):
        return True
    hf_param = mapping.hf_param
    if isinstance(hf_param, str):
        return hf_param in hf_keys
    if isinstance(hf_param, dict):
        return all(param in hf_keys for param in hf_param.values())
    return False


def build_art_conversion_tasks(*, bridge: Any, model: ModelChunks) -> list[Any]:
    from megatron.bridge.models.conversion.model_bridge import (
        WeightConversionTask,
        _megatron_local_name_to_global,
    )
    from megatron.bridge.models.conversion.utils import (
        get_module_and_param_from_name,
        persistent_buffers,
    )

    mapping_registry = bridge._model_bridge.mapping_registry()
    hf_source = bridge.hf_pretrained.state.source
    hf_keys = set(hf_source.get_all_keys())
    megatron_models = as_megatron_api_chunks(model)
    model_config = cast(Any, model[0].config)
    tasks: list[Any] = []
    for vp_stage, chunk in enumerate(model):
        for local_name, _ in chain(
            chunk.named_parameters(),
            persistent_buffers(chunk),
        ):
            if "_extra_state" in local_name or is_art_adapter_param_name(local_name):
                continue
            global_name = _megatron_local_name_to_global(
                megatron_models,
                model_config,
                canonical_art_param_name(local_name),
                vp_stage,
            )
            mapping = mapping_registry.megatron_to_hf_lookup(global_name)
            if mapping is None or not _mapping_hf_weights_exist(mapping, hf_keys):
                continue
            local_module, local_weights = cast(
                tuple[Any, torch.Tensor],
                get_module_and_param_from_name(
                    megatron_models,
                    local_name,
                    vp_stage,
                ),
            )
            if local_module is not None and not hasattr(local_module, "config"):
                setattr(local_module, "config", model_config)
            tasks.append(
                WeightConversionTask(
                    pp_rank=0,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    global_param_name=global_name,
                    megatron_module=local_module,
                    param_weight=local_weights,
                    mapping=mapping,
                )
            )
    return tasks


def build_merged_weight_export(
    *,
    bridge: Any,
    model: ModelChunks,
    model_support_handler: Any,
) -> MergedWeightExport:
    return MergedWeightExport(
        bridge=bridge,
        model=model,
        model_config_value=model[0].config,
        conversion_tasks=build_art_conversion_tasks(
            bridge=bridge,
            model=model,
        ),
        adapter_weights_by_base=model_support_handler.build_adapter_weights_by_base(
            model
        ),
    )


def iter_merged_vllm_weights(
    weight_export: MergedWeightExport,
) -> Iterator[tuple[str, torch.Tensor]]:
    bridge = weight_export.bridge
    model_bridge = bridge._model_bridge
    hf_state_dict = bridge.hf_pretrained.state
    grouped_buffers: dict[str, dict[int, torch.Tensor]] = {}
    for task in weight_export.conversion_tasks:
        converted_weights_dict = task.mapping.megatron_to_hf(
            task.param_weight,
            task.megatron_module,
        )
        adapter_weights = weight_export.adapter_weights_by_base.get(
            task.global_param_name
        )
        if adapter_weights is not None:
            converted_weights_dict = model_bridge._merge_lora_adapter_weights(
                weight_export.model,
                converted_weights_dict,
                adapter_weights,
            )
        if getattr(task.mapping, "is_grouped_export", False):
            merged_result = model_bridge._accumulate_grouped_export(
                task,
                converted_weights_dict,
                weight_export.model_config_value,
                grouped_buffers,
                hf_state_dict,
            )
            if merged_result is None:
                continue
            converted_weights_dict = merged_result
        else:
            converted_weights_dict = model_bridge.maybe_modify_converted_hf_weight(
                task,
                converted_weights_dict,
                hf_state_dict,
            )
        yield from converted_weights_dict.items()


__all__ = [
    "MergedWeightExport",
    "build_art_conversion_tasks",
    "build_merged_weight_export",
    "iter_merged_vllm_weights",
]
