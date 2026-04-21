from typing import Any, Sequence

from art.megatron.model_support.spec import (
    CompileWorkaroundConfig,
    LayerFamilyInstance,
    SharedExpertCompileState,
)


class DefaultDenseHandler:
    key = "default_dense"

    def identity_lora_model_config(self, base_config: Any) -> Any:
        return base_config

    def identity_lora_target_parameters(
        self,
        model: Any,
        *,
        target_modules: list[str],
    ) -> list[str]:
        suffixes = self._identity_lora_parameter_suffixes(target_modules)
        return [
            name for name, _ in model.named_parameters() if name.endswith(suffixes)
        ]

    def _identity_lora_parameter_suffixes(
        self,
        target_modules: list[str],
    ) -> tuple[str, ...]:
        target_set = set(target_modules)
        suffixes: list[str] = []
        if "q_proj" in target_set:
            suffixes.append("q_proj.weight")
        if "k_proj" in target_set:
            suffixes.append("k_proj.weight")
        if "v_proj" in target_set:
            suffixes.append("v_proj.weight")
        if "o_proj" in target_set:
            suffixes.append("o_proj.weight")
        if "gate_proj" in target_set:
            suffixes.extend(("gate_proj.weight", "mlp.experts.gate_up_proj"))
        if "up_proj" in target_set:
            suffixes.extend(("up_proj.weight", "mlp.experts.gate_up_proj"))
        if "down_proj" in target_set:
            suffixes.extend(("down_proj.weight", "mlp.experts.down_proj"))
        return tuple(dict.fromkeys(suffixes))

    def patch_provider(self, provider: Any, bridge: Any) -> None:
        return None

    def patch_bridge(self, bridge: Any) -> None:
        del bridge
        return None

    def configure_provider_for_runtime(self, provider: Any) -> None:
        del provider
        return None

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        del model_chunks
        return None

    def _shared_expert_compile_state(
        self,
        provider: Any,
    ) -> SharedExpertCompileState:
        if int(getattr(provider, "moe_shared_expert_intermediate_size", 0) or 0) <= 0:
            return "none"
        if bool(getattr(provider, "moe_shared_expert_overlap", False)):
            return "shared_expert_overlap"
        return "shared_experts"

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        layer_families = [LayerFamilyInstance(key="standard_attention", layer_index=0)]
        if int(getattr(provider, "num_moe_experts", 0) or 0) > 0:
            layer_families.append(
                LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0)
            )
            if (
                int(getattr(provider, "moe_shared_expert_intermediate_size", 0) or 0)
                > 0
            ):
                layer_families.append(
                    LayerFamilyInstance(key="shared_experts_mlp", layer_index=0)
                )
            return layer_families
        layer_families.append(LayerFamilyInstance(key="dense_mlp", layer_index=0))
        return layer_families

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
            for module_name, module in chunk.named_modules():
                if not isinstance(module, TransformerLayer):
                    continue
                layer_prefix = layer_base_prefix(module, module_name=module_name)
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

    def compile_workaround_config(
        self,
        provider: Any,
    ) -> CompileWorkaroundConfig:
        return CompileWorkaroundConfig(
            shared_expert_state=self._shared_expert_compile_state(provider)
        )

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]:
        del model
        return {"extra_block_kwargs": kwargs}


DEFAULT_DENSE_HANDLER = DefaultDenseHandler()
