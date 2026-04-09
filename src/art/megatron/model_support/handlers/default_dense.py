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
        return None

    def build_adapter_weights(self, model_chunks: Sequence[Any]) -> dict[str, Any]:
        return {}

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]:
        return kwargs


DEFAULT_DENSE_HANDLER = DefaultDenseHandler()
