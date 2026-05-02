from copy import copy
from types import MethodType
from typing import Any, Callable, Sequence, cast

from megatron.core.models.gpt.gpt_model import GPTModel
import torch

from art.megatron.model_chunks import ModelChunks
from art.megatron.model_support.handlers.default_dense import DefaultDenseHandler
from art.megatron.model_support.spec import (
    CompileWorkaroundConfig,
    LayerFamilyInstance,
)
from art.megatron.provider_common import patch_layer_spec_tree

_QWEN35_MOE_COMPILE_WORKAROUND_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
)


class Qwen35MoeHandler(DefaultDenseHandler):
    key = "qwen3_5_moe"

    def identity_lora_model_config(self, base_config: Any) -> Any:
        return getattr(base_config, "text_config", base_config)

    def _identity_lora_parameter_suffixes(
        self,
        target_modules: list[str],
    ) -> tuple[str, ...]:
        suffixes = list(super()._identity_lora_parameter_suffixes(target_modules))
        target_set = set(target_modules)
        if "in_proj_qkv" in target_set:
            suffixes.append("linear_attn.in_proj_qkv.weight")
        if "in_proj_z" in target_set:
            suffixes.append("linear_attn.in_proj_z.weight")
        if "out_proj" in target_set:
            suffixes.append("linear_attn.out_proj.weight")
        return tuple(dict.fromkeys(suffixes))

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        from art.megatron.gdn.operator import (
            install_gdn_island_hooks,
            install_shared_prefix_gdn_hooks,
        )

        install_shared_prefix_gdn_hooks(model_chunks)
        install_gdn_island_hooks(model_chunks)
        for chunk in cast(ModelChunks, list(model_chunks)):
            module: Any = chunk
            while hasattr(module, "module"):
                module = module.module
            gpt_module = (
                module
                if isinstance(module, GPTModel)
                else cast(GPTModel, getattr(module, "language_model"))
            )
            preprocess = gpt_module._preprocess

            def preprocess_hook(*args, _preprocess=preprocess, **kwargs):
                position_ids = kwargs.get("position_ids")
                if isinstance(position_ids, torch.Tensor) and position_ids.ndim == 2:
                    kwargs = dict(kwargs)
                    kwargs["position_ids"] = position_ids.unsqueeze(0).expand(
                        3,
                        position_ids.shape[0],
                        position_ids.shape[1],
                    )
                preproc_output = list(_preprocess(*args, **kwargs))
                decoder_input = cast(torch.Tensor, preproc_output[0])
                if not decoder_input.requires_grad and decoder_input.is_leaf:
                    decoder_input.requires_grad_(True)
                return tuple(preproc_output)

            gpt_module._preprocess = preprocess_hook  # type: ignore[attr-defined]

    def configure_provider_for_runtime(self, provider: Any) -> None:
        provider.moe_shared_expert_overlap = False

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        linear_attention_pattern = _linear_attention_pattern(provider)
        gated_delta_net_layer_index = (
            linear_attention_pattern.index(1) if 1 in linear_attention_pattern else 0
        )
        standard_attention_layer_index = (
            linear_attention_pattern.index(0) if 0 in linear_attention_pattern else 0
        )
        layer_families = [
            LayerFamilyInstance(
                key="standard_attention",
                layer_index=standard_attention_layer_index,
            ),
            LayerFamilyInstance(
                key="gated_delta_net_attention",
                layer_index=gated_delta_net_layer_index,
            ),
        ]
        if int(getattr(provider, "num_moe_experts", 0) or 0) > 0:
            layer_families.append(LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0))
        else:
            layer_families.append(LayerFamilyInstance(key="dense_mlp", layer_index=0))
        if int(getattr(provider, "moe_shared_expert_intermediate_size", 0) or 0) > 0:
            layer_families.append(
                LayerFamilyInstance(key="shared_experts_mlp", layer_index=0)
            )
        return layer_families

    def patch_bridge(self, bridge: Any) -> None:
        del bridge
        _ensure_qwen35_text_only_bridge_registered()

    def patch_provider(self, provider: Any, bridge: Any) -> None:
        del bridge
        if not _is_qwen35_vl_provider(provider):
            return
        (
            qwen3_vl_self_attention,
            qwen35_provider_types,
            patch_standard_attention_specs,
            transformer_block_spec_factory,
        ) = _require_qwen35_provider_symbols()
        from art.megatron.flex_attention import FlexDotProductAttention
        matched_provider_type = next(
            (
                provider_type
                for provider_type in qwen35_provider_types
                if isinstance(provider, provider_type)
            ),
            None,
        )
        if matched_provider_type is None:
            return

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
            return matched_provider_type.provide_language_model(
                self,
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            )

        provider.scatter_embedding_sequence_parallel = True
        provider.transformer_layer_spec = _qwen35_layer_spec
        provider.provide = MethodType(_provide_qwen35_with_flex_attention, provider)
        setattr(provider, "_art_text_only_language_model", True)

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
                layer_prefix = layer_base_prefix(module, module_name=module_name)
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

    def compile_workaround_config(
        self,
        provider: Any,
    ) -> CompileWorkaroundConfig:
        if bool(getattr(provider, "moe_shared_expert_overlap", False)):
            return CompileWorkaroundConfig(
                flags=("moe_forward",),
                shared_expert_state="shared_expert_overlap",
                disable_compile=True,
            )
        return CompileWorkaroundConfig(
            flags=_QWEN35_MOE_COMPILE_WORKAROUND_FLAGS,
            shared_expert_state="shared_experts",
            disable_compile=False,
        )

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
    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import (
        Qwen35VLBridge,
        Qwen35VLMoEBridge,
    )

    return (Qwen3MoEBridge, Qwen35VLBridge, Qwen35VLMoEBridge)


def _is_qwen35_vl_provider(provider: object) -> bool:
    return isinstance(provider, _optional_qwen35_provider_types())


def _optional_qwen35_provider_types() -> tuple[type[Any], ...]:
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
        Qwen35VLModelProvider,
        Qwen35VLMoEModelProvider,
    )

    return (Qwen35VLModelProvider, Qwen35VLMoEModelProvider)


def _optional_qwen35_provider_type() -> type[Any] | None:
    provider_types = _optional_qwen35_provider_types()
    return provider_types[0] if provider_types else None


def _require_qwen35_provider_symbols() -> tuple[Any, ...]:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import (
        Qwen3VLSelfAttention,
    )
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
        Qwen35VLModelProvider,
        Qwen35VLMoEModelProvider,
        _patch_standard_attention_specs,
    )
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )

    return (
        Qwen3VLSelfAttention,
        (Qwen35VLModelProvider, Qwen35VLMoEModelProvider),
        _patch_standard_attention_specs,
        get_transformer_block_with_experimental_attention_variant_spec,
    )


def _register_qwen35_text_only_module_types() -> None:
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    AutoMapping.register_module_type("SharedExpertMLP", "column")
    AutoMapping.register_module_type("GatedDeltaNet", "column")


def _qwen35_text_only_mapping_registry() -> Any:
    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge

    _register_qwen35_text_only_module_types()
    upstream_registry = Qwen35VLMoEBridge().mapping_registry()
    language_mappings = [
        _text_only_qwen35_mapping(mapping)
        for mapping in upstream_registry.mappings
        if mapping.megatron_param.startswith("language_model.")
    ]
    return MegatronMappingRegistry(*language_mappings)


def _text_only_qwen35_mapping(mapping: Any) -> Any:
    from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import (
        ExpertMLPDownProjMapping,
        ExpertMLPGateUpProjMapping,
    )

    megatron_param = mapping.megatron_param.removeprefix("language_model.")
    if isinstance(mapping, ExpertMLPGateUpProjMapping):
        return _ArtExpertMLPGateUpProjMapping(megatron_param, mapping.hf_param)
    if isinstance(mapping, ExpertMLPDownProjMapping):
        return _ArtExpertMLPDownProjMapping(megatron_param, mapping.hf_param)
    cloned = copy(mapping)
    cloned.megatron_param = megatron_param
    return cloned


from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import (
    ExpertMLPDownProjMapping as _BridgeExpertMLPDownProjMapping,
)
from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import (
    ExpertMLPGateUpProjMapping as _BridgeExpertMLPGateUpProjMapping,
)


class _ArtExpertMLPGateUpProjMapping(_BridgeExpertMLPGateUpProjMapping):
    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor | dict[str, torch.Tensor],
        megatron_module: Any,
    ) -> torch.Tensor:
        from megatron.bridge.models.conversion.utils import (
            get_module_and_param_from_name,
        )
        from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import (
            _align_weight_to_shape,
        )
        from megatron.bridge.utils.common_utils import (
            extract_expert_number_from_param,
        )

        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        expert_weight = (
            hf_weights[global_expert_number]
            if isinstance(hf_weights, torch.Tensor) and hf_weights.ndim >= 3
            else hf_weights
        )
        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(
            megatron_module, normalized_param
        )
        full_target_shape = (
            target_param.shape[0] * self.tp_size,
            target_param.shape[1],
        )
        gate_target_shape = (
            full_target_shape[0] // 2,
            full_target_shape[1],
        )
        if full_target_shape[0] % 2 != 0:
            raise ValueError(
                f"Expected even fused dim for {self.megatron_param}, got {full_target_shape}."
            )
        if (
            isinstance(expert_weight, torch.Tensor)
            and expert_weight.ndim == 3
            and expert_weight.shape[0] == 2
        ):
            gate = _align_weight_to_shape(expert_weight[0], gate_target_shape, "gate")
            up = _align_weight_to_shape(expert_weight[1], gate_target_shape, "up")
        else:
            fused = _align_weight_to_shape(
                cast(torch.Tensor, expert_weight),
                torch.Size(full_target_shape),
                "gate_up",
            )
            gate, up = torch.chunk(fused, 2, dim=0)
        return self._gated_mapping.hf_to_megatron(
            {"gate": gate, "up": up},
            megatron_module,
        )


class _ArtExpertMLPDownProjMapping(_BridgeExpertMLPDownProjMapping):
    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: Any,
    ) -> torch.Tensor:
        from megatron.bridge.models.conversion.param_mapping import (
            ColumnParallelMapping,
            RowParallelMapping,
        )
        from megatron.bridge.models.conversion.utils import (
            get_module_and_param_from_name,
        )
        from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import (
            _align_weight_to_shape,
        )
        from megatron.bridge.utils.common_utils import (
            extract_expert_number_from_param,
        )

        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        expert_weight = (
            hf_weights[global_expert_number] if hf_weights.ndim >= 3 else hf_weights
        )
        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(
            megatron_module, normalized_param
        )
        if self._mapping is None:
            self._detected_type = self._detect_parallelism_type(megatron_module)
            self._mapping = self._get_or_create_mapping(self._detected_type)
        if isinstance(self._mapping, ColumnParallelMapping):
            full_target_shape = (
                target_param.shape[0] * self.tp_size,
                target_param.shape[1],
            )
        elif isinstance(self._mapping, RowParallelMapping):
            full_target_shape = (
                target_param.shape[0],
                target_param.shape[1] * self.tp_size,
            )
        else:
            full_target_shape = tuple(target_param.shape)
        aligned = _align_weight_to_shape(
            expert_weight,
            torch.Size(full_target_shape),
            "down_proj",
        )
        return self._mapping.hf_to_megatron(aligned, megatron_module)


def _ensure_qwen35_text_only_bridge_registered() -> None:
    return None


from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import (
    _QWEN3_5_DENSE_HF_CLASS_NAME,
    _QWEN3_5_MOE_HF_CLASS_NAME,
    Qwen35VLBridge,
    Qwen35VLMoEBridge,
)
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    Qwen35VLModelProvider,
    Qwen35VLMoEModelProvider,
)


@MegatronModelBridge.register_bridge(
    source=_QWEN3_5_DENSE_HF_CLASS_NAME,
    target=GPTModel,
    provider=Qwen35VLModelProvider,
    model_type="qwen3_5_moe",
)
class _ArtQwen35DenseTextOnlyBridge(Qwen35VLBridge):
    def mapping_registry(self) -> Any:
        return _qwen35_text_only_mapping_registry()


@MegatronModelBridge.register_bridge(
    source=_QWEN3_5_MOE_HF_CLASS_NAME,
    target=GPTModel,
    provider=Qwen35VLMoEModelProvider,
    model_type="qwen3_5_moe",
)
class _ArtQwen35TextOnlyBridge(Qwen35VLMoEBridge):
    def mapping_registry(self) -> Any:
        return _qwen35_text_only_mapping_registry()


def _optional_gated_delta_net_type() -> type[Any] | None:
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    return GatedDeltaNet


def _linear_attention_pattern(provider: Any) -> list[int]:
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_linear_attention_pattern,
    )

    return list(get_linear_attention_pattern(provider))
