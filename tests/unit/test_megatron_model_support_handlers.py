from types import SimpleNamespace

import pytest
import torch

from art.megatron.flex_attention import FlexDotProductAttention
from art.megatron.model_support.handlers import (
    DEFAULT_DENSE_HANDLER,
    QWEN3_5_MOE_HANDLER,
    QWEN3_MOE_HANDLER,
)
from art.megatron.model_support.handlers.qwen3_5_moe import (
    _ensure_qwen35_text_only_bridge_registered,
    _qwen35_text_only_mapping_registry,
)
from art.megatron.model_support.spec import LayerFamilyInstance


class _FakeModel:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def named_parameters(self):
        return [(name, object()) for name in self._names]


def test_default_dense_handler_returns_standard_attention_kwargs() -> None:
    assert DEFAULT_DENSE_HANDLER.get_forward_kwargs(
        object(),
        attention_bias="bias",
    ) == {"extra_block_kwargs": {"attention_bias": "bias"}}


def test_qwen_handler_wraps_qwen3vl_forward_kwargs() -> None:
    qwen_model = type("Qwen3VLModel", (), {})()

    assert QWEN3_5_MOE_HANDLER.get_forward_kwargs(
        qwen_model,
        attention_bias="bias",
    ) == {"extra_block_kwargs": {"extra_block_kwargs": {"attention_bias": "bias"}}}


def test_qwen_handler_unwraps_model_wrappers() -> None:
    qwen_model = type("Qwen3VLModel", (), {})()
    wrapper = type("Wrapper", (), {"module": qwen_model})()

    assert QWEN3_5_MOE_HANDLER.get_forward_kwargs(
        wrapper,
        attention_bias="bias",
    ) == {"extra_block_kwargs": {"extra_block_kwargs": {"attention_bias": "bias"}}}


def test_default_dense_handler_collects_dense_layer_families() -> None:
    provider = type("Provider", (), {"num_moe_experts": 0})()

    assert DEFAULT_DENSE_HANDLER.collect_layer_families(provider) == [
        LayerFamilyInstance(key="standard_attention", layer_index=0),
        LayerFamilyInstance(key="dense_mlp", layer_index=0),
    ]


def test_default_dense_handler_collects_moe_layer_families() -> None:
    provider = type(
        "Provider",
        (),
        {
            "num_moe_experts": 8,
            "moe_shared_expert_intermediate_size": 4096,
        },
    )()

    assert DEFAULT_DENSE_HANDLER.collect_layer_families(provider) == [
        LayerFamilyInstance(key="standard_attention", layer_index=0),
        LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
        LayerFamilyInstance(key="shared_experts_mlp", layer_index=0),
    ]


def test_qwen_handler_collects_expected_layer_families() -> None:
    provider = type("Provider", (), {"linear_attention_freq": 4, "num_layers": 8})()

    assert QWEN3_5_MOE_HANDLER.collect_layer_families(provider) == [
        LayerFamilyInstance(key="standard_attention", layer_index=3),
        LayerFamilyInstance(key="gated_delta_net_attention", layer_index=0),
        LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
        LayerFamilyInstance(key="shared_experts_mlp", layer_index=0),
    ]


def test_qwen35_handler_expands_rank2_position_ids_for_text_only_mrope() -> None:
    seen_shapes: list[tuple[int, ...]] = []

    def _preprocess(*args, **kwargs):
        del args
        seen_shapes.append(tuple(kwargs["position_ids"].shape))
        return (torch.zeros(1, requires_grad=False),)

    language_model = type(
        "LanguageModel",
        (),
        {"_preprocess": staticmethod(_preprocess)},
    )()
    wrapper = type("Wrapper", (), {"language_model": language_model})()

    assert QWEN3_5_MOE_HANDLER.install_preprocess_patch([wrapper]) is None

    output = language_model._preprocess(position_ids=torch.arange(4).view(1, 4))

    assert seen_shapes == [(3, 1, 4)]
    assert output[0].requires_grad is True


def test_default_dense_handler_reports_shared_expert_compile_state() -> None:
    provider = type(
        "Provider",
        (),
        {
            "moe_shared_expert_intermediate_size": 4096,
            "moe_shared_expert_overlap": True,
        },
    )()

    assert DEFAULT_DENSE_HANDLER.compile_workaround_config(provider).model_dump() == {
        "flags": (),
        "shared_expert_state": "shared_expert_overlap",
        "disable_compile": False,
    }


def test_qwen3_handler_uses_qwen3_compile_workaround_pair() -> None:
    assert QWEN3_MOE_HANDLER.compile_workaround_config(object()).model_dump() == {
        "flags": (
            "alltoall_dtoh",
            "alltoall_dispatch_preprocess",
        ),
        "shared_expert_state": "none",
        "disable_compile": False,
    }


def test_qwen35_handler_disables_shared_expert_overlap_by_default() -> None:
    provider = type("Provider", (), {"moe_shared_expert_overlap": True})()

    QWEN3_5_MOE_HANDLER.configure_provider_for_runtime(provider)

    assert provider.moe_shared_expert_overlap is False


def test_qwen35_handler_uses_shared_expert_workaround_pair_when_overlap_disabled() -> None:
    provider = type("Provider", (), {"moe_shared_expert_overlap": False})()

    assert QWEN3_5_MOE_HANDLER.compile_workaround_config(provider).model_dump() == {
        "flags": (
            "alltoall_dtoh",
            "alltoall_dispatch_preprocess",
        ),
        "shared_expert_state": "shared_experts",
        "disable_compile": False,
    }


def test_qwen35_handler_falls_back_to_moe_forward_when_overlap_enabled() -> None:
    provider = type("Provider", (), {"moe_shared_expert_overlap": True})()

    assert QWEN3_5_MOE_HANDLER.compile_workaround_config(provider).model_dump() == {
        "flags": ("moe_forward",),
        "shared_expert_state": "shared_expert_overlap",
        "disable_compile": True,
    }


def test_qwen35_handler_rebinds_provider_to_language_only_runtime(
    monkeypatch,
) -> None:
    class _FakeQwen35Provider:
        def __init__(self) -> None:
            self.transformer_layer_spec = object()
            self.freeze_language_model = False
            self.language_only_calls: list[tuple[bool | None, bool | None, int | None]] = []

        def provide_language_model(
            self,
            pre_process: bool | None = None,
            post_process: bool | None = None,
            vp_stage: int | None = None,
        ) -> SimpleNamespace:
            self.language_only_calls.append((pre_process, post_process, vp_stage))
            return SimpleNamespace(kind="language_only")

    def _patch_standard_attention_specs(block_spec: object, attention_cls: object) -> None:
        del attention_cls
        return None

    def _transformer_block_spec_factory(
        config: object,
        vp_stage: int | None = None,
    ) -> SimpleNamespace:
        del config, vp_stage
        gdn_layer = SimpleNamespace(
            submodules=SimpleNamespace(
                self_attention=SimpleNamespace(submodules=SimpleNamespace())
            )
        )
        attention_layer = SimpleNamespace(
            submodules=SimpleNamespace(
                self_attention=SimpleNamespace(
                    submodules=SimpleNamespace(core_attention=object())
                )
            )
        )
        return SimpleNamespace(layer_specs=[gdn_layer, attention_layer])

    monkeypatch.setattr(
        "art.megatron.model_support.handlers.qwen3_5_moe._optional_qwen35_provider_type",
        lambda: _FakeQwen35Provider,
    )
    monkeypatch.setattr(
        "art.megatron.model_support.handlers.qwen3_5_moe._require_qwen35_provider_symbols",
        lambda: (
            object(),
            _FakeQwen35Provider,
            _patch_standard_attention_specs,
            _transformer_block_spec_factory,
        ),
    )

    provider = _FakeQwen35Provider()
    QWEN3_5_MOE_HANDLER.patch_provider(provider, bridge=object())

    model = provider.provide(pre_process=True, post_process=False, vp_stage=7)
    layer_spec = provider.transformer_layer_spec(provider, vp_stage=7)

    assert model.kind == "language_only"
    assert provider.language_only_calls == [(True, False, 7)]
    assert getattr(provider, "_art_text_only_language_model") is True
    gdn_layer, attention_layer = layer_spec.layer_specs
    assert not hasattr(gdn_layer.submodules.self_attention.submodules, "core_attention")
    assert (
        attention_layer.submodules.self_attention.submodules.core_attention
        is FlexDotProductAttention
    )


def test_qwen35_handler_requests_text_only_bridge_registration(monkeypatch) -> None:
    calls: list[None] = []

    monkeypatch.setattr(
        "art.megatron.model_support.handlers.qwen3_5_moe._ensure_qwen35_text_only_bridge_registered",
        lambda: calls.append(None),
    )

    QWEN3_5_MOE_HANDLER.patch_bridge(object())

    assert calls == [None]


def test_qwen35_text_only_bridge_registry_uses_decoder_root_names() -> None:
    _ensure_qwen35_text_only_bridge_registered()
    names = {
        mapping.megatron_param
        for mapping in _qwen35_text_only_mapping_registry().mappings
    }

    assert "embedding.word_embeddings.weight" in names
    assert "decoder.layers.*.self_attention.linear_qkv.weight" in names
    assert "language_model.embedding.word_embeddings.weight" not in names


def test_default_dense_handler_identity_lora_targets_dense_shared_and_moe_params() -> None:
    model = _FakeModel(
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.mlp.shared_expert_gate.weight",
        ]
    )

    assert DEFAULT_DENSE_HANDLER.identity_lora_target_parameters(
        model,
        target_modules=["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ) == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    ]


def test_qwen35_handler_identity_lora_targets_linear_attn_and_shared_experts() -> None:
    model = _FakeModel(
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.layers.0.linear_attn.in_proj_z.weight",
            "model.layers.0.linear_attn.out_proj.weight",
            "model.layers.0.linear_attn.in_proj_b.weight",
            "model.layers.0.linear_attn.in_proj_a.weight",
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.layers.0.mlp.shared_expert_gate.weight",
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.down_proj",
        ]
    )

    assert QWEN3_5_MOE_HANDLER.identity_lora_target_parameters(
        model,
        target_modules=[
            "q_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    ) == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
        "model.layers.0.linear_attn.out_proj.weight",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    ]
