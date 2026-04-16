from types import MethodType
from typing import Any, Callable, Sequence, cast

from art.megatron.model_chunks import ModelChunks
from art.megatron.model_support.handlers.default_dense import DefaultDenseHandler
from art.megatron.model_support.spec import LayerFamilyInstance
from art.megatron.provider_common import patch_layer_spec_tree


class Qwen35MoeHandler(DefaultDenseHandler):
    key = "qwen3_5_moe"

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        from art.megatron.train import _install_gpt_preprocess_hook

        _install_gpt_preprocess_hook(cast(ModelChunks, list(model_chunks)))

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        linear_attention_pattern = _linear_attention_pattern(provider)
        gated_delta_net_layer_index = (
            linear_attention_pattern.index(1) if 1 in linear_attention_pattern else 0
        )
        standard_attention_layer_index = (
            linear_attention_pattern.index(0) if 0 in linear_attention_pattern else 0
        )
        return [
            LayerFamilyInstance(
                key="standard_attention",
                layer_index=standard_attention_layer_index,
            ),
            LayerFamilyInstance(
                key="gated_delta_net_attention",
                layer_index=gated_delta_net_layer_index,
            ),
            LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
            LayerFamilyInstance(key="shared_experts_mlp", layer_index=0),
        ]

    def patch_provider(self, provider: Any, bridge: Any) -> None:
        del bridge
        if not _is_qwen35_vl_provider(provider):
            return
        use_flex_attention = (
            getattr(provider, "_art_runtime_profile", "art_training") == "art_training"
        )
        (
            qwen3_vl_model,
            qwen3_vl_self_attention,
            qwen35_provider_type,
            patch_standard_attention_specs,
            transformer_block_spec_factory,
        ) = _require_qwen35_provider_symbols()
        if use_flex_attention:
            from art.megatron.flex_attention import FlexDotProductAttention

        def _patch_qwen35_block_spec(block_spec: object) -> None:
            patch_standard_attention_specs(block_spec, qwen3_vl_self_attention)
            if use_flex_attention:
                for layer_spec in getattr(block_spec, "layer_specs", ()):
                    patch_layer_spec_tree(layer_spec, FlexDotProductAttention)

        def _qwen35_layer_spec(config: Any, vp_stage: int | None = None) -> object:
            block_spec = transformer_block_spec_factory(config, vp_stage=vp_stage)
            _patch_qwen35_block_spec(block_spec)
            return block_spec

        def _provide_qwen35_with_flex_attention(
            self: Any,
            pre_process: bool | None = None,
            post_process: bool | None = None,
            vp_stage: int | None = None,
        ) -> Any:
            language_transformer_config = self
            hf_vision_config = self.vision_config
            hf_vision_config.torch_dtype = self.params_dtype
            block_spec = transformer_block_spec_factory(
                language_transformer_config,
                vp_stage=vp_stage,
            )
            _patch_qwen35_block_spec(block_spec)
            model = qwen3_vl_model(
                language_transformer_config=language_transformer_config,
                language_transformer_layer_spec=block_spec,
                vision_transformer_config=hf_vision_config,
                pre_process=pre_process,
                post_process=post_process,
                pg_collection=self._pg_collection,
            )
            if (
                self.freeze_language_model
                or self.freeze_vision_model
                or self.freeze_vision_projection
            ):
                model.freeze(
                    freeze_language_model=self.freeze_language_model,
                    freeze_vision_model=self.freeze_vision_model,
                    freeze_vision_projection=self.freeze_vision_projection,
                )
            return model

        if isinstance(provider, qwen35_provider_type):
            provider.transformer_layer_spec = _qwen35_layer_spec
            provider.provide = MethodType(_provide_qwen35_with_flex_attention, provider)

    def apply_lora_adapters(
        self,
        model_chunks: Sequence[Any],
        provider: Any,
        *,
        target_modules: list[str],
        rank: int,
        alpha: int,
    ) -> None:
        from megatron.core.transformer.attention import SelfAttention
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.lora import (
            _adapter_model_prefix,
            _is_language_transformer_layer_name,
            wrap_dense_mlp,
            wrap_gated_delta_net_attention,
            wrap_grouped_moe_experts,
            wrap_shared_experts_mlp,
            wrap_standard_self_attention,
        )

        target_set = set(target_modules)
        gated_delta_net_type = _optional_gated_delta_net_type()
        for chunk in model_chunks:
            for module_name, module in chunk.named_modules():
                if not isinstance(module, TransformerLayer):
                    continue
                if not _is_language_transformer_layer_name(module_name):
                    continue
                adapter_model_prefix = _adapter_model_prefix(module)
                if isinstance(module.self_attention, SelfAttention):
                    wrap_standard_self_attention(
                        module.self_attention,
                        adapter_model_prefix=adapter_model_prefix,
                        provider=provider,
                        target_modules=target_set,
                        rank=rank,
                        alpha=alpha,
                    )
                elif gated_delta_net_type is not None and isinstance(
                    module.self_attention, gated_delta_net_type
                ):
                    wrap_gated_delta_net_attention(
                        module.self_attention,
                        adapter_model_prefix=adapter_model_prefix,
                        provider=provider,
                        target_modules=target_set,
                        rank=rank,
                        alpha=alpha,
                    )
                else:
                    raise TypeError(
                        "Unsupported self_attention module type for Megatron LoRA: "
                        f"{type(module.self_attention)}"
                    )
                experts = getattr(module.mlp, "experts", None)
                if experts is not None:
                    wrap_grouped_moe_experts(
                        experts,
                        adapter_model_prefix=adapter_model_prefix,
                        target_modules=target_set,
                        rank=rank,
                        alpha=alpha,
                    )
                else:
                    wrap_dense_mlp(
                        module.mlp,
                        adapter_model_prefix=adapter_model_prefix,
                        provider=provider,
                        target_modules=target_set,
                        rank=rank,
                        alpha=alpha,
                    )
                shared_experts = getattr(module.mlp, "shared_experts", None)
                if shared_experts is not None:
                    wrap_shared_experts_mlp(
                        shared_experts,
                        adapter_model_prefix=adapter_model_prefix,
                        provider=provider,
                        target_modules=target_set,
                        rank=rank,
                        alpha=alpha,
                    )

    def build_adapter_weights_by_base(
        self,
        model_chunks: Sequence[Any],
    ) -> dict[str, list[Any]]:
        from megatron.core.transformer.attention import SelfAttention
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.adapter_export import (
            add_dense_mlp_adapter_weights,
            add_gated_delta_net_adapter_weights,
            add_grouped_moe_adapter_weights,
            add_shared_experts_adapter_weights,
            add_standard_self_attention_adapter_weights,
            layer_base_prefix,
        )
        from art.megatron.lora import _is_language_transformer_layer_name

        _ensure_bridge_qwen35_adapter_name_map()
        adapter_weights_by_base: dict[str, list[Any]] = {}
        gated_delta_net_type = _optional_gated_delta_net_type()
        for chunk in model_chunks:
            for module_name, module in chunk.named_modules():
                if not isinstance(module, TransformerLayer):
                    continue
                if not _is_language_transformer_layer_name(module_name):
                    continue
                layer_prefix = layer_base_prefix(module)
                if isinstance(module.self_attention, SelfAttention):
                    add_standard_self_attention_adapter_weights(
                        adapter_weights_by_base,
                        layer_prefix=layer_prefix,
                        self_attention=module.self_attention,
                    )
                elif gated_delta_net_type is not None and isinstance(
                    module.self_attention, gated_delta_net_type
                ):
                    add_gated_delta_net_adapter_weights(
                        adapter_weights_by_base,
                        layer_prefix=layer_prefix,
                        self_attention=module.self_attention,
                    )
                experts = getattr(module.mlp, "experts", None)
                if experts is not None:
                    add_grouped_moe_adapter_weights(
                        adapter_weights_by_base,
                        layer_prefix=layer_prefix,
                        experts=experts,
                    )
                else:
                    add_dense_mlp_adapter_weights(
                        adapter_weights_by_base,
                        layer_prefix=layer_prefix,
                        mlp=module.mlp,
                    )
                shared_experts = getattr(module.mlp, "shared_experts", None)
                if shared_experts is not None:
                    add_shared_experts_adapter_weights(
                        adapter_weights_by_base,
                        layer_prefix=layer_prefix,
                        shared_experts=shared_experts,
                    )
        return adapter_weights_by_base

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]:
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        if type(unwrapped).__name__ == "Qwen3VLModel":
            return {"extra_block_kwargs": {"extra_block_kwargs": kwargs}}
        return {"extra_block_kwargs": kwargs}


QWEN3_5_MOE_HANDLER = Qwen35MoeHandler()


def _ensure_bridge_qwen35_adapter_name_map() -> None:
    from megatron.bridge.models.conversion import peft_bridge

    extra_entries = {
        ".in_proj_qkv.weight": "adapter_qkv",
        ".in_proj_z.weight": "adapter_z",
        ".in_proj_b.weight": "adapter_b",
        ".in_proj_a.weight": "adapter_a",
    }
    for suffix, adapter_key in extra_entries.items():
        peft_bridge.ADAPTER_NAME_MAP.setdefault(suffix, adapter_key)
        peft_bridge.ADAPTER_KEY_TO_SUFFIX.setdefault(adapter_key, suffix)


def supported_qwen_moe_bridge_types() -> tuple[type[Any], ...]:
    from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge

    bridge_types: tuple[type[Any], ...] = (Qwen3MoEBridge,)
    try:
        from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge
    except ImportError:
        return bridge_types
    return bridge_types + (Qwen35VLMoEBridge,)


def _is_qwen35_vl_provider(provider: object) -> bool:
    qwen35_provider_type = _optional_qwen35_provider_type()
    return qwen35_provider_type is not None and isinstance(
        provider, qwen35_provider_type
    )


def _optional_qwen35_provider_type() -> type[Any] | None:
    try:
        from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
            Qwen35VLMoEModelProvider,
        )
    except ImportError:
        return None
    return Qwen35VLMoEModelProvider


def _require_qwen35_provider_symbols() -> tuple[Any, ...]:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import (
        Qwen3VLSelfAttention,
    )
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
        Qwen35VLMoEModelProvider,
        _patch_standard_attention_specs,
    )
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )

    return (
        Qwen3VLModel,
        Qwen3VLSelfAttention,
        Qwen35VLMoEModelProvider,
        _patch_standard_attention_specs,
        get_transformer_block_with_experimental_attention_variant_spec,
    )


def _optional_gated_delta_net_type() -> type[Any] | None:
    try:
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    except ImportError:
        return None
    return GatedDeltaNet


def _linear_attention_pattern(provider: Any) -> list[int]:
    try:
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_linear_attention_pattern,
        )
    except ImportError:
        frequency = int(getattr(provider, "linear_attention_freq", 1) or 1)
        layer_count = int(getattr(provider, "num_layers", 1) or 1)
        return [
            0 if frequency > 0 and (layer_index + 1) % frequency == 0 else 1
            for layer_index in range(layer_count)
        ]
    return list(get_linear_attention_pattern(provider))
