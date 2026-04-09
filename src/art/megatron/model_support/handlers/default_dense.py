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

    def build_adapter_weights(self, model_chunks: Sequence[Any]) -> dict[str, Any]:
        return {}

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]:
        return kwargs


DEFAULT_DENSE_HANDLER = DefaultDenseHandler()
