from typing import Any, Sequence

from art.megatron.model_support.spec import LayerFamilyInstance


class DefaultDenseHandler:
    key = "default_dense"

    def patch_provider(self, provider: Any, bridge: Any) -> None:
        return None

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        return []

    def apply_lora_adapters(
        self,
        model_chunks: Sequence[Any],
        provider: Any,
        *,
        target_modules: list[str],
        rank: int,
        alpha: int,
    ) -> None:
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.lora import (
            _adapter_model_prefix,
            wrap_grouped_moe_experts,
            wrap_standard_self_attention,
        )

        target_set = set(target_modules)
        for chunk in model_chunks:
            for module in chunk.modules():
                if not isinstance(module, TransformerLayer):
                    continue
                wrap_standard_self_attention(
                    module.self_attention,
                    adapter_model_prefix=_adapter_model_prefix(module),
                    provider=provider,
                    target_modules=target_set,
                    rank=rank,
                    alpha=alpha,
                )
                wrap_grouped_moe_experts(
                    module.mlp.experts,
                    adapter_model_prefix=_adapter_model_prefix(module),
                    target_modules=target_set,
                    rank=rank,
                    alpha=alpha,
                )

    def build_adapter_weights_by_base(
        self,
        model_chunks: Sequence[Any],
    ) -> dict[str, list[Any]]:
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.adapter_export import (
            add_dense_mlp_adapter_weights,
            add_grouped_moe_adapter_weights,
            add_shared_experts_adapter_weights,
            add_standard_self_attention_adapter_weights,
            layer_base_prefix,
        )

        adapter_weights_by_base: dict[str, list[Any]] = {}
        for chunk in model_chunks:
            for module in chunk.modules():
                if not isinstance(module, TransformerLayer):
                    continue
                layer_prefix = layer_base_prefix(module)
                add_standard_self_attention_adapter_weights(
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
        return kwargs


DEFAULT_DENSE_HANDLER = DefaultDenseHandler()
