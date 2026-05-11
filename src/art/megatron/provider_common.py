import copy
import inspect
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from art.megatron.model_support.spec import ModelSupportSpec


class ProviderBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Any
    bridge: Any
    handler: Any
    spec: ModelSupportSpec


def resolve_layer_spec(
    base_layer_spec: Any,
    config: Any,
    vp_stage: int | None = None,
) -> Any:
    module_spec_type = _optional_module_spec_type()
    if module_spec_type is not None and isinstance(base_layer_spec, module_spec_type):
        return copy.deepcopy(base_layer_spec)
    kwargs = (
        {"vp_stage": vp_stage}
        if vp_stage in inspect.signature(base_layer_spec).parameters
        else {}
    )
    return base_layer_spec(config, **kwargs)


def patch_core_attention(layer_spec: object, core_attention: object) -> None:
    submodules = getattr(layer_spec, "submodules", None)
    self_attention = getattr(submodules, "self_attention", None)
    attention_submodules = getattr(self_attention, "submodules", None)
    if attention_submodules is None or not hasattr(
        attention_submodules,
        "core_attention",
    ):
        return
    attention_submodules.core_attention = core_attention


def patch_layer_spec_tree(layer_spec: object, core_attention: object) -> None:
    layer_specs = getattr(layer_spec, "layer_specs", None)
    if layer_specs is None:
        patch_core_attention(layer_spec, core_attention)
        return
    for block_layer_spec in layer_specs:
        patch_core_attention(block_layer_spec, core_attention)


def art_context_parallel_size(config: object) -> int:
    configured = int(getattr(config, "context_parallel_size", 1) or 1)
    return max(configured, _runtime_context_parallel_size())


def patch_art_flex_attention(layer_spec: object, config: object) -> None:
    patch_layer_spec_tree(layer_spec, _art_flex_core_attention(config))


def _art_flex_core_attention(config: object) -> object:
    if art_context_parallel_size(config) > 1:
        from art.megatron.context_parallel.core_attention import (
            ArtContextParallelCoreAttention,
        )

        return ArtContextParallelCoreAttention
    from art.megatron.flex_attention import FlexDotProductAttention

    return FlexDotProductAttention


def _runtime_context_parallel_size() -> int:
    try:
        from megatron.core import parallel_state

        if not parallel_state.model_parallel_is_initialized():
            return 1
        return int(parallel_state.get_context_parallel_world_size())
    except (AssertionError, ImportError, RuntimeError, ValueError):
        return 1


def _optional_module_spec_type() -> type[Any] | None:
    try:
        from megatron.core.transformer.spec_utils import ModuleSpec
    except ImportError:
        return None
    return ModuleSpec
