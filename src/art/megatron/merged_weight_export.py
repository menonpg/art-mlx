from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import time
from typing import Any, Iterator, cast

from pydantic import BaseModel, ConfigDict
import torch

from art.megatron.jobs import (
    MergedWeightTransferInitInfo,
    MergedWeightTransferSpec,
)
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
            try:
                converted_weights_dict = model_bridge._merge_lora_adapter_weights(
                    weight_export.model,
                    converted_weights_dict,
                    adapter_weights,
                )
            except Exception as exc:
                converted_shapes = {
                    key: tuple(value.shape)
                    for key, value in converted_weights_dict.items()
                }
                adapter_summaries = [
                    {
                        "base_prefix": adapter_weight.global_base_prefix,
                        "adapter_key": adapter_weight.adapter_key,
                        "linear_in": tuple(
                            adapter_weight.linear_in_weight.weight.shape
                        ),
                        "linear_out": tuple(
                            adapter_weight.linear_out_weight.weight.shape
                        ),
                    }
                    for adapter_weight in adapter_weights
                ]
                raise RuntimeError(
                    "Failed merged LoRA export for "
                    f"{task.global_param_name}: converted={converted_shapes} "
                    f"adapter_weights={adapter_summaries}"
                ) from exc
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


def ensure_merged_weight_transfer_group(
    *,
    rank: int,
    world_size: int,
    merged_weight_transfer_group: Any | None,
    merged_weight_transfer_init_info: MergedWeightTransferInitInfo | None,
    spec: MergedWeightTransferSpec,
) -> tuple[Any, MergedWeightTransferInitInfo]:
    assert rank == 0
    assert world_size == 1
    if merged_weight_transfer_init_info == spec.init_info:
        assert merged_weight_transfer_group is not None
        assert merged_weight_transfer_init_info is not None
        return merged_weight_transfer_group, merged_weight_transfer_init_info

    import httpx
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    def _remote_init() -> None:
        response = httpx.post(
            f"{spec.vllm_base_url}/init_weight_transfer_engine",
            json={"init_info": spec.init_info.model_dump()},
            timeout=300.0,
        )
        response.raise_for_status()

    with ThreadPoolExecutor(max_workers=1) as executor:
        remote_future = executor.submit(_remote_init)
        time.sleep(1.0)
        merged_weight_transfer_group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": spec.init_info.master_address,
                "master_port": spec.init_info.master_port,
                "world_size": spec.init_info.world_size,
            }
        )
        remote_future.result()
    return merged_weight_transfer_group, spec.init_info


def sync_merged_weights_to_vllm(
    *,
    bridge: Any,
    model: ModelChunks,
    model_support_handler: Any,
    rank: int,
    world_size: int,
    merged_weight_transfer_group: Any | None,
    merged_weight_transfer_init_info: MergedWeightTransferInitInfo | None,
    spec: MergedWeightTransferSpec,
    pause_generation: bool,
) -> tuple[Any, MergedWeightTransferInitInfo]:
    assert rank == 0
    assert world_size == 1

    import httpx
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    (
        merged_weight_transfer_group,
        merged_weight_transfer_init_info,
    ) = ensure_merged_weight_transfer_group(
        rank=rank,
        world_size=world_size,
        merged_weight_transfer_group=merged_weight_transfer_group,
        merged_weight_transfer_init_info=merged_weight_transfer_init_info,
        spec=spec,
    )
    weight_export = build_merged_weight_export(
        bridge=bridge,
        model=model,
        model_support_handler=model_support_handler,
    )

    def _send_weights() -> None:
        NCCLWeightTransferEngine.trainer_send_weights(
            iter_merged_vllm_weights(weight_export),
            {"group": merged_weight_transfer_group},
        )

    with httpx.Client() as client:
        if pause_generation:
            response = client.post(
                f"{spec.vllm_base_url}/pause",
                params={"mode": "wait"},
                timeout=300.0,
            )
            response.raise_for_status()
        try:
            torch.cuda.synchronize()
            names: list[str] = []
            dtype_names: list[str] = []
            shapes: list[list[int]] = []
            for name, tensor in iter_merged_vllm_weights(weight_export):
                names.append(name)
                dtype_names.append(str(tensor.dtype).removeprefix("torch."))
                shapes.append(list(tensor.shape))
            with ThreadPoolExecutor(max_workers=1) as executor:
                send_future = executor.submit(_send_weights)
                response = client.post(
                    f"{spec.vllm_base_url}/update_weights",
                    json={
                        "update_info": {
                            "names": names,
                            "dtype_names": dtype_names,
                            "shapes": shapes,
                            "is_checkpoint_format": True,
                        }
                    },
                    timeout=600.0,
                )
                response.raise_for_status()
                send_future.result()
            response = client.post(
                f"{spec.vllm_base_url}/art/set_served_model_name",
                json={"name": spec.served_model_name},
                timeout=30.0,
            )
            response.raise_for_status()
            torch.cuda.synchronize()
        finally:
            if pause_generation:
                response = client.post(
                    f"{spec.vllm_base_url}/resume",
                    timeout=30.0,
                )
                response.raise_for_status()
    return merged_weight_transfer_group, merged_weight_transfer_init_info


__all__ = [
    "MergedWeightExport",
    "build_art_conversion_tasks",
    "build_merged_weight_export",
    "ensure_merged_weight_transfer_group",
    "iter_merged_vllm_weights",
    "sync_merged_weights_to_vllm",
]
