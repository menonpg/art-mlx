from __future__ import annotations

from copy import copy
import re
from typing import Any, Sequence

import torch

from art.megatron.model_support.handlers.default_dense import (
    DefaultMoeHandler,
    _compile_workaround_flags_for_provider,
    _require_moe_experts,
)
from art.megatron.model_support.spec import (
    CompileWorkaroundConfig,
    ExpertPackedLoraGroup,
    ExpertPackedLoraSlot,
    LayerFamilyInstance,
)

_GEMMA4_MOE_COMPILE_WORKAROUND_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
    "deepep_dispatch_combine",
    "deepep_permute_restore",
    "flex_token_dispatch_combine",
    "te_triton_permute_with_mask_map",
)
_ART_MOE_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\."
    r"(?P<module>gate_up_proj|down_proj)\.(?P<lora>lora_[AB])\.weight$"
)
_VLLM_MOE_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.moe\.experts)\."
    r"(?:(?P<base_layer>base_layer)\.)?(?P<lora>lora_[AB])\.weight$"
)
_VLLM_MOE_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.moe\.experts)\.(?P<expert>\d+)\."
    r"(?P<module>gate_proj|up_proj|down_proj)\.(?P<lora>lora_[AB])\.weight$"
)
_DENSE_MLP_LORA_KEY_RE = re.compile(
    r"(?P<prefix>\.mlp)\.(?P<module>gate_proj|up_proj|down_proj)\."
    r"(?P<lora>lora_[AB])\.weight$"
)
_MEGATRON_LAYER_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\.")


class Gemma4MoeHandler(DefaultMoeHandler):
    key = "gemma4_moe"
    is_moe = True
    native_vllm_lora_status = "wip"

    def identity_lora_model_config(self, base_config: Any) -> Any:
        return getattr(base_config, "text_config", base_config)

    def _identity_lora_parameter_suffixes(
        self,
        target_modules: list[str],
    ) -> tuple[str, ...]:
        suffixes = list(super()._identity_lora_parameter_suffixes(target_modules))
        target_set = set(target_modules)
        if {"experts", "gate_proj", "up_proj"} & target_set:
            suffixes.append("experts.gate_up_proj")
        if {"experts", "down_proj"} & target_set:
            suffixes.append("experts.down_proj")
        return tuple(dict.fromkeys(suffixes))

    def configure_provider_for_runtime(self, provider: Any) -> None:
        _patch_gemma4_router_for_mcore()
        provider.moe_shared_expert_overlap = False

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        if int(getattr(provider, "num_moe_experts", 0) or 0) <= 0:
            raise TypeError("Gemma 4 MoE handler received a dense provider")
        sliding_count, global_count = _gemma4_attention_pattern(provider)
        families = [
            LayerFamilyInstance(key="gemma4_sliding_attention", layer_index=0),
            LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
        ]
        if global_count > 0:
            families.append(
                LayerFamilyInstance(
                    key="gemma4_global_attention",
                    layer_index=sliding_count,
                )
            )
        if int(getattr(provider, "moe_shared_expert_intermediate_size", 0) or 0) > 0:
            families.append(
                LayerFamilyInstance(key="shared_experts_mlp", layer_index=0)
            )
        return families

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
            wrap_grouped_moe_experts_3d,
            wrap_shared_experts_mlp,
            wrap_standard_self_attention,
        )

        target_set = set(target_modules)
        for chunk in model_chunks:
            for module_name, module in chunk.named_modules():
                if not isinstance(module, TransformerLayer):
                    continue
                if not _is_language_transformer_layer_name(module_name):
                    continue
                adapter_model_prefix = _adapter_model_prefix(module)
                if not isinstance(module.self_attention, SelfAttention):
                    raise TypeError(
                        "Gemma 4 expected a SelfAttention module, got "
                        f"{type(module.self_attention)}"
                    )
                wrap_standard_self_attention(
                    module.self_attention,
                    adapter_model_prefix=adapter_model_prefix,
                    provider=_attention_provider_for_layer(provider, module),
                    target_modules=target_set,
                    rank=rank,
                    alpha=alpha,
                )
                wrap_grouped_moe_experts_3d(
                    _require_moe_experts(module),
                    adapter_model_prefix=adapter_model_prefix,
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
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.lora import _is_language_transformer_layer_name
        from art.megatron.weights.adapter_export import (
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
                if not _is_language_transformer_layer_name(module_name):
                    continue
                layer_prefix = layer_base_prefix(module, module_name=module_name)
                add_standard_self_attention_adapter_weights(
                    adapter_weights_by_base,
                    layer_prefix=layer_prefix,
                    self_attention=module.self_attention,
                )
                add_grouped_moe_adapter_weights(
                    adapter_weights_by_base,
                    layer_prefix=layer_prefix,
                    experts=_require_moe_experts(module),
                )
                shared_experts = getattr(module.mlp, "shared_experts", None)
                if shared_experts is not None:
                    add_shared_experts_adapter_weights(
                        adapter_weights_by_base,
                        layer_prefix=layer_prefix,
                        shared_experts=shared_experts,
                    )
        return adapter_weights_by_base

    def to_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return _to_vllm_lora_tensors(tensors, adapter_config=adapter_config)

    def from_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        return _from_vllm_lora_tensors(tensors, adapter_config=adapter_config)

    def expert_packed_lora_groups(self) -> tuple[ExpertPackedLoraGroup, ...]:
        return (
            ExpertPackedLoraGroup(
                art_group_suffix=".mlp.experts",
                slots=(
                    ExpertPackedLoraSlot(
                        source_projection="gate_up_proj",
                        source_lora="lora_A",
                        output_suffix="base_layer.lora_A.weight",
                        pack_layout="expert_rows",
                    ),
                    ExpertPackedLoraSlot(
                        source_projection="gate_up_proj",
                        source_lora="lora_B",
                        output_suffix="base_layer.lora_B.weight",
                        pack_layout="rank_major_expert_cols",
                    ),
                    ExpertPackedLoraSlot(
                        source_projection="down_proj",
                        source_lora="lora_A",
                        output_suffix="lora_A.weight",
                        pack_layout="expert_rows",
                    ),
                    ExpertPackedLoraSlot(
                        source_projection="down_proj",
                        source_lora="lora_B",
                        output_suffix="lora_B.weight",
                        pack_layout="rank_major_expert_cols",
                    ),
                ),
            ),
        )

    def compile_workaround_config(
        self,
        provider: Any,
    ) -> CompileWorkaroundConfig:
        return CompileWorkaroundConfig(
            flags=_compile_workaround_flags_for_provider(
                provider,
                _GEMMA4_MOE_COMPILE_WORKAROUND_FLAGS,
            ),
            shared_expert_state="shared_experts",
            disable_compile=False,
        )


GEMMA4_MOE_HANDLER = Gemma4MoeHandler()

_GEMMA4_ROUTER_PATCHED = False


def _patch_gemma4_router_for_mcore() -> None:
    global _GEMMA4_ROUTER_PATCHED
    if _GEMMA4_ROUTER_PATCHED:
        return
    from megatron.bridge.models.gemma import gemma4_provider
    from megatron.core.transformer.moe.router import TopKRouter

    def _art_gemma4_router_routing(
        self: Any,
        logits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del input_ids
        routing_probs, routing_map = TopKRouter.routing(
            self,
            logits,
            padding_mask=padding_mask,
        )
        if routing_map is not None:
            prob_sums = routing_probs.sum(dim=-1, keepdim=True).clamp(min=1e-20)
            routing_probs = routing_probs / prob_sums
            routing_probs = routing_probs * self.per_expert_scale.unsqueeze(0)
        return routing_probs, routing_map

    gemma4_provider.Gemma4TopKRouter.routing = _art_gemma4_router_routing
    _GEMMA4_ROUTER_PATCHED = True


def _gemma4_attention_pattern(provider: Any) -> tuple[int, int]:
    pattern = getattr(provider, "interleaved_attn_pattern", (0, 1))
    if not pattern:
        return (0, 1)
    if len(pattern) == 1:
        return (int(pattern[0]), 0)
    return (int(pattern[0]), int(pattern[1]))


def _is_gemma4_global_layer(layer_number: int, provider: Any) -> bool:
    sliding_count, global_count = _gemma4_attention_pattern(provider)
    if global_count <= 0:
        return False
    cycle = sliding_count + global_count
    if cycle <= 0:
        return False
    return (layer_number - 1) % cycle >= sliding_count


def _attention_provider_for_layer(provider: Any, module: Any) -> Any:
    if not _is_gemma4_global_layer(int(module.layer_number), provider):
        return provider
    global_provider = copy(provider)
    global_provider.kv_channels = getattr(provider, "global_head_dim")
    global_provider.num_query_groups = getattr(provider, "num_global_key_value_heads")
    return global_provider


def _to_vllm_key(key: str) -> str:
    return key.replace(".mlp.shared_expert.", ".mlp.").replace(
        ".mlp.experts",
        ".moe.experts",
    )


def _from_vllm_key(key: str) -> str:
    key = key.replace(".moe.experts", ".mlp.experts")
    return _DENSE_MLP_LORA_KEY_RE.sub(
        r"\g<prefix>.shared_expert.\g<module>.\g<lora>.weight",
        key,
    )


def _pack_vllm_3d_lora_b(blocks: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(blocks, dim=0)
    return stacked.permute(1, 2, 0).reshape(stacked.shape[1], -1).contiguous()


def _unpack_vllm_3d_lora_b(
    tensor: torch.Tensor,
    *,
    num_experts: int,
    rank: int,
) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], rank, num_experts).permute(2, 0, 1)


def _clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone().contiguous()


def _vllm_moe_config(adapter_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(adapter_config)
    target_modules = list(config.get("target_modules") or [])
    if "experts" not in target_modules:
        target_modules.append("experts")
    config["target_modules"] = target_modules
    return config


def _group_art_moe_tensors(
    tensors: dict[str, torch.Tensor],
) -> dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]]:
    grouped: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]] = {}
    for key, tensor in tensors.items():
        match = _ART_MOE_EXPERT_KEY_RE.match(key)
        if match is None:
            continue
        grouped.setdefault(match.group("prefix"), {}).setdefault(
            int(match.group("expert")),
            {},
        ).setdefault(match.group("module"), {})[match.group("lora")] = tensor
    return grouped


def _to_vllm_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    grouped = _group_art_moe_tensors(tensors)
    if not grouped:
        transformed = {_to_vllm_key(key): tensor for key, tensor in tensors.items()}
        if len(transformed) != len(tensors):
            raise RuntimeError("Duplicate Gemma 4 LoRA tensor after vLLM conversion")
        has_fused_experts = any(_VLLM_MOE_KEY_RE.match(key) for key in tensors)
        return (
            transformed,
            _vllm_moe_config(adapter_config) if has_fused_experts else adapter_config,
        )

    transformed: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()
    for prefix, experts in grouped.items():
        vllm_prefix = _to_vllm_key(prefix)
        gate_up_a: list[torch.Tensor] = []
        gate_up_b: list[torch.Tensor] = []
        down_a: list[torch.Tensor] = []
        down_b: list[torch.Tensor] = []
        for expert in sorted(experts):
            modules = experts[expert]
            try:
                gate_up_a_tensor = modules["gate_up_proj"]["lora_A"]
                gate_up_b_tensor = modules["gate_up_proj"]["lora_B"]
                down_a_tensor = modules["down_proj"]["lora_A"]
                down_b_tensor = modules["down_proj"]["lora_B"]
            except KeyError as exc:
                raise RuntimeError(
                    f"Incomplete Gemma 4 MoE LoRA block for {prefix}.{expert}"
                ) from exc
            gate_up_a.append(gate_up_a_tensor.contiguous())
            gate_up_b.append(gate_up_b_tensor.contiguous())
            down_a.append(down_a_tensor.contiguous())
            down_b.append(down_b_tensor.contiguous())
            for module_name in ("gate_up_proj", "down_proj"):
                for lora_name in ("lora_A", "lora_B"):
                    used_keys.add(f"{prefix}.{expert}.{module_name}.{lora_name}.weight")
        transformed[f"{vllm_prefix}.base_layer.lora_A.weight"] = torch.cat(
            gate_up_a,
            dim=0,
        ).contiguous()
        transformed[f"{vllm_prefix}.base_layer.lora_B.weight"] = _pack_vllm_3d_lora_b(
            gate_up_b
        )
        transformed[f"{vllm_prefix}.lora_A.weight"] = torch.cat(
            down_a,
            dim=0,
        ).contiguous()
        transformed[f"{vllm_prefix}.lora_B.weight"] = _pack_vllm_3d_lora_b(down_b)

    for key, tensor in tensors.items():
        if key in used_keys:
            continue
        vllm_key = _to_vllm_key(key)
        if vllm_key in transformed:
            raise RuntimeError(
                f"Duplicate Gemma 4 LoRA tensor after conversion: {vllm_key}"
            )
        transformed[vllm_key] = tensor
    return transformed, _vllm_moe_config(adapter_config)


def _from_vllm_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    expert_grouped: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]] = {}
    for key, tensor in tensors.items():
        match = _VLLM_MOE_EXPERT_KEY_RE.match(key)
        if match is None:
            continue
        expert_grouped.setdefault(match.group("prefix"), {}).setdefault(
            int(match.group("expert")),
            {},
        ).setdefault(match.group("module"), {})[match.group("lora")] = tensor
    if expert_grouped:
        return _from_vllm_per_expert_lora_tensors(
            tensors,
            expert_grouped=expert_grouped,
            adapter_config=adapter_config,
        )

    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        match = _VLLM_MOE_KEY_RE.match(key)
        if match is None:
            continue
        slot = (
            f"{'base_layer.' if match.group('base_layer') else ''}{match.group('lora')}"
        )
        grouped.setdefault(match.group("prefix"), {})[slot] = tensor
    if not grouped:
        return {_from_vllm_key(key): tensor for key, tensor in tensors.items()}

    rank = int(adapter_config["r"])
    transformed: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()
    for prefix, slots in grouped.items():
        try:
            gate_up_a = slots["base_layer.lora_A"]
            gate_up_b = slots["base_layer.lora_B"]
            down_a = slots["lora_A"]
            down_b = slots["lora_B"]
        except KeyError as exc:
            raise RuntimeError(
                f"Incomplete Gemma 4 vLLM MoE LoRA block for {prefix}"
            ) from exc
        if gate_up_a.shape[0] % rank != 0:
            raise RuntimeError(
                f"{prefix}: gate/up lora_A shape {tuple(gate_up_a.shape)} "
                f"is not divisible by rank {rank}"
            )
        num_experts = gate_up_a.shape[0] // rank
        art_prefix = _from_vllm_key(prefix)
        gate_up_b_by_expert = _unpack_vllm_3d_lora_b(
            gate_up_b,
            num_experts=num_experts,
            rank=rank,
        )
        down_b_by_expert = _unpack_vllm_3d_lora_b(
            down_b,
            num_experts=num_experts,
            rank=rank,
        )
        for expert in range(num_experts):
            row = expert * rank
            transformed[f"{art_prefix}.{expert}.gate_up_proj.lora_A.weight"] = (
                gate_up_a[row : row + rank].contiguous()
            )
            transformed[f"{art_prefix}.{expert}.gate_up_proj.lora_B.weight"] = (
                gate_up_b_by_expert[expert].contiguous()
            )
            transformed[f"{art_prefix}.{expert}.down_proj.lora_A.weight"] = down_a[
                row : row + rank
            ].contiguous()
            transformed[f"{art_prefix}.{expert}.down_proj.lora_B.weight"] = (
                down_b_by_expert[expert].contiguous()
            )
        used_keys.update(
            {
                f"{prefix}.base_layer.lora_A.weight",
                f"{prefix}.base_layer.lora_B.weight",
                f"{prefix}.lora_A.weight",
                f"{prefix}.lora_B.weight",
            }
        )
    for key, tensor in tensors.items():
        if key in used_keys:
            continue
        art_key = _from_vllm_key(key)
        if art_key in transformed:
            raise RuntimeError(
                f"Duplicate Gemma 4 LoRA tensor after conversion: {art_key}"
            )
        transformed[art_key] = tensor
    return transformed


def _from_vllm_per_expert_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    expert_grouped: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]],
    adapter_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    del adapter_config
    transformed: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()
    for prefix, experts in expert_grouped.items():
        art_prefix = _from_vllm_key(prefix)
        for expert, modules in experts.items():
            try:
                gate_a = modules["gate_proj"]["lora_A"]
                gate_b = modules["gate_proj"]["lora_B"]
                up_a = modules["up_proj"]["lora_A"]
                up_b = modules["up_proj"]["lora_B"]
                down_a = modules["down_proj"]["lora_A"]
                down_b = modules["down_proj"]["lora_B"]
            except KeyError as exc:
                raise RuntimeError(
                    f"Incomplete Gemma 4 vLLM MoE LoRA block for {prefix}.{expert}"
                ) from exc
            if not torch.equal(gate_a, up_a):
                raise RuntimeError(
                    "Gemma 4 Megatron gate_up_proj requires gate/up LoRA-A "
                    f"tensors to match for {prefix}.{expert}"
                )
            transformed[f"{art_prefix}.{expert}.gate_up_proj.lora_A.weight"] = _clone(
                gate_a
            )
            transformed[f"{art_prefix}.{expert}.gate_up_proj.lora_B.weight"] = (
                torch.cat([gate_b, up_b], dim=0).contiguous()
            )
            transformed[f"{art_prefix}.{expert}.down_proj.lora_A.weight"] = _clone(
                down_a
            )
            transformed[f"{art_prefix}.{expert}.down_proj.lora_B.weight"] = _clone(
                down_b
            )
            for module_name in ("gate_proj", "up_proj", "down_proj"):
                for lora_name in ("lora_A", "lora_B"):
                    used_keys.add(f"{prefix}.{expert}.{module_name}.{lora_name}.weight")
    for key, tensor in tensors.items():
        if key in used_keys:
            continue
        if _VLLM_MOE_KEY_RE.match(key) is not None:
            raise RuntimeError(
                "Mixed fused and per-expert Gemma 4 vLLM MoE LoRA tensors"
            )
        transformed[_from_vllm_key(key)] = tensor
    return transformed


def _gemma4_text_only_mapping_registry(hf_config: Any | None = None) -> Any:
    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.gemma.gemma4_bridge import _Gemma4QKVMapping
    from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge

    upstream_registry = Gemma4VLBridge().mapping_registry()
    global_layer_indices = _gemma4_global_layer_indices(hf_config)

    class _ArtGemma4TextOnlyQKVMapping(_Gemma4QKVMapping):
        def __init__(
            self,
            megatron_param: str,
            q: str,
            k: str,
            v: str,
            *,
            global_layer_indices: tuple[int, ...],
        ) -> None:
            super().__init__(megatron_param, q, k, v)
            self._global_layer_indices = global_layer_indices
            self._export_hf_param = dict(self.hf_param)

        def resolve(self, captures: tuple[str, ...]) -> Any:
            megatron_param, hf_param = self._resolve_names(captures)
            resolved = type(self)(
                megatron_param,
                hf_param["q"],
                hf_param["k"],
                hf_param["v"],
                global_layer_indices=self._global_layer_indices,
            )
            layer_index = _megatron_layer_index(megatron_param)
            if layer_index in self._global_layer_indices:
                resolved.hf_param = dict(resolved.hf_param)
                resolved.hf_param["v"] = resolved.hf_param["k"]
            return resolved

        def megatron_to_hf(
            self,
            megatron_weights: torch.Tensor | None,
            megatron_module: Any | None,
        ) -> dict[str, torch.Tensor]:
            import_hf_param = self.hf_param
            self.hf_param = self._export_hf_param
            try:
                return super().megatron_to_hf(megatron_weights, megatron_module)
            finally:
                self.hf_param = import_hf_param

    language_mappings = [
        _text_only_gemma4_mapping(
            mapping,
            qkv_mapping_type=_ArtGemma4TextOnlyQKVMapping,
            global_layer_indices=global_layer_indices,
        )
        for mapping in upstream_registry.mappings
        if mapping.megatron_param.startswith("language_model.")
    ]
    return MegatronMappingRegistry(*language_mappings)


def _text_only_gemma4_mapping(
    mapping: Any,
    *,
    qkv_mapping_type: type[Any],
    global_layer_indices: tuple[int, ...],
) -> Any:
    megatron_param = mapping.megatron_param.removeprefix("language_model.")
    hf_param = getattr(mapping, "hf_param", None)
    if (
        megatron_param.endswith(".self_attention.linear_qkv.weight")
        and isinstance(hf_param, dict)
        and set(hf_param) == {"q", "k", "v"}
    ):
        return qkv_mapping_type(
            megatron_param,
            hf_param["q"],
            hf_param["k"],
            hf_param["v"],
            global_layer_indices=global_layer_indices,
        )
    cloned = copy(mapping)
    cloned.megatron_param = megatron_param
    return cloned


def _gemma4_global_layer_indices(hf_config: Any | None) -> tuple[int, ...]:
    text_config = getattr(hf_config, "text_config", hf_config)
    layer_types = getattr(text_config, "layer_types", None)
    if not layer_types:
        return ()
    return tuple(
        layer_index
        for layer_index, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    )


def _megatron_layer_index(megatron_param: str) -> int | None:
    match = _MEGATRON_LAYER_RE.search(megatron_param)
    return None if match is None else int(match.group("layer"))


_GEMMA4_TEXT_ONLY_BRIDGE_REGISTERED = False


def ensure_gemma4_text_only_bridge_registered() -> None:
    global _GEMMA4_TEXT_ONLY_BRIDGE_REGISTERED
    if _GEMMA4_TEXT_ONLY_BRIDGE_REGISTERED:
        return

    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.conversion.transformers_compat import (
        rope_local_base_freq_from_hf,
        rope_theta_from_hf,
    )
    from megatron.bridge.models.gemma.gemma4_bridge import (
        Gemma4Bridge,
        _infer_attn_pattern,
    )
    from megatron.bridge.models.gemma.gemma4_provider import Gemma4ModelProvider
    from megatron.core.models.gpt.gpt_model import GPTModel

    @MegatronModelBridge.register_bridge(
        source="Gemma4ForConditionalGeneration",
        target=GPTModel,
        provider=Gemma4ModelProvider,
        model_type="gemma4",
    )
    class _ArtGemma4TextOnlyBridge(Gemma4Bridge):
        def provider_bridge(self, hf_pretrained: Any) -> Any:
            text_config = getattr(
                hf_pretrained.config,
                "text_config",
                hf_pretrained.config,
            )
            if (
                not getattr(text_config, "enable_moe_block", False)
                or int(getattr(text_config, "hidden_size_per_layer_input", 0) or 0) > 0
            ):
                raise ValueError(
                    "ART Gemma 4 support currently targets the MoE text backbone "
                    "without per-layer embeddings."
                )

            provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
            provider = Gemma4ModelProvider(**provider_kwargs)
            provider.window_size = getattr(text_config, "sliding_window", 1024)
            provider.rotary_base = (
                rope_local_base_freq_from_hf(text_config),
                rope_theta_from_hf(text_config),
            )
            provider.softmax_scale = 1.0
            provider.kv_channels = getattr(text_config, "head_dim", 256)
            provider.qk_layernorm = True
            provider.global_head_dim = getattr(text_config, "global_head_dim", 512)
            provider.num_global_key_value_heads = getattr(
                text_config,
                "num_global_key_value_heads",
                2,
            )
            provider.attention_k_eq_v = getattr(text_config, "attention_k_eq_v", False)
            rope_params = getattr(text_config, "rope_parameters", {})
            if isinstance(rope_params, dict):
                full_attn_rope = rope_params.get("full_attention", {})
                provider.global_rotary_percent = full_attn_rope.get(
                    "partial_rotary_factor",
                    0.25,
                )
            layer_types = getattr(text_config, "layer_types", None)
            if layer_types:
                provider.interleaved_attn_pattern = _infer_attn_pattern(layer_types)

            provider.num_moe_experts = getattr(text_config, "num_experts", 128)
            provider.moe_router_topk = getattr(text_config, "top_k_experts", 8)
            provider.moe_ffn_hidden_size = getattr(
                text_config,
                "moe_intermediate_size",
                704,
            )
            provider.moe_shared_expert_intermediate_size = getattr(
                text_config,
                "intermediate_size",
                2112,
            )
            provider.moe_shared_expert_overlap = False
            provider.moe_shared_expert_gate = False
            provider.moe_layer_freq = 1
            provider.final_logit_softcapping = getattr(
                text_config,
                "final_logit_softcapping",
                30.0,
            )
            provider.bf16 = True
            provider.params_dtype = torch.bfloat16
            provider.autocast_dtype = torch.bfloat16
            provider.make_vocab_size_divisible_by = 128
            return provider

        def mapping_registry(self) -> Any:
            return _gemma4_text_only_mapping_registry(getattr(self, "hf_config", None))

    _GEMMA4_TEXT_ONLY_BRIDGE_REGISTERED = True
