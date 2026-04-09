from types import MethodType
from typing import Any, Callable

from art.megatron.model_support.handlers.default_dense import DefaultDenseHandler
from art.megatron.provider_common import patch_layer_spec_tree


class Qwen35MoeHandler(DefaultDenseHandler):
    key = "qwen3_5_moe"

    def patch_provider(self, provider: Any, bridge: Any) -> None:
        del bridge
        if not _is_qwen35_vl_provider(provider):
            return
        (
            qwen3_vl_model,
            qwen3_vl_self_attention,
            qwen35_provider_type,
            patch_standard_attention_specs,
            transformer_block_spec_factory,
            mtp_block_spec,
        ) = _require_qwen35_provider_symbols()
        from art.megatron.flex_attention import FlexDotProductAttention

        def _patch_qwen35_block_spec(block_spec: object) -> None:
            patch_standard_attention_specs(block_spec, qwen3_vl_self_attention)
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
                mtp_block_spec=mtp_block_spec(self, vp_stage=vp_stage),
                vp_stage=vp_stage,
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


QWEN3_5_MOE_HANDLER = Qwen35MoeHandler()


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


def _require_qwen35_provider_symbols() -> tuple[
    type[Any],
    type[Any],
    type[Any],
    Callable[[object, type[Any]], None],
    Callable[..., Any],
    Callable[..., Any],
]:
    from megatron.bridge.models.gpt_provider import mtp_block_spec
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
        mtp_block_spec,
    )
