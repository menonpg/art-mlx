from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.bridge.models.qwen.qwen3_moe_bridge")
pytest.importorskip("megatron.bridge.models.qwen_vl.qwen35_vl_bridge")

from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import (
    Qwen35VLBridge,
    Qwen35VLMoEBridge,
)
from megatron.core.transformer.enums import AttnBackend

from art.megatron.flex_attention import FlexDotProductAttention
import art.megatron.provider as provider_module


class _FakeProvider:
    def __init__(self) -> None:
        self.transformer_layer_spec = self._base_layer_spec
        self.finalized = False
        self.overlap_moe_expert_parallel_comm = False
        self.delay_wgrad_compute = False
        self.ep_overlap_early_attn_memory_release = False
        self.moe_apply_probs_on_input = False
        self.bias_activation_fusion = True
        self.fine_grained_activation_offloading = False
        self.offload_modules = []
        self.recompute_modules = None

    def _base_layer_spec(
        self, config: object, vp_stage: int | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(
            submodules=SimpleNamespace(
                self_attention=SimpleNamespace(
                    submodules=SimpleNamespace(core_attention=object())
                )
            ),
        )

    def finalize(self) -> None:
        self.finalized = True


class _FakeHybridProvider(_FakeProvider):
    def _base_layer_spec(
        self, config: object, vp_stage: int | None = None
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


class _FakeBridge:
    def __init__(self, *, model_bridge: object, provider: _FakeProvider) -> None:
        self._model_bridge = model_bridge
        self._provider = provider
        self.hf_pretrained = SimpleNamespace(model_name_or_path="unused")

    def to_megatron_provider(self) -> _FakeProvider:
        return self._provider


@pytest.mark.parametrize(
    (
        "bridge_type",
        "num_moe_experts",
        "expected_expert_model_parallel_size",
        "expected_moe_shared_expert_overlap",
    ),
    [
        (Qwen3MoEBridge, 8, 2, False),
        (Qwen35VLBridge, 0, 1, False),
        (Qwen35VLMoEBridge, 8, 2, False),
    ],
)
def test_get_provider_accepts_supported_qwen_bridges(
    monkeypatch: pytest.MonkeyPatch,
    bridge_type: type[object],
    num_moe_experts: int,
    expected_expert_model_parallel_size: int,
    expected_moe_shared_expert_overlap: bool,
) -> None:
    provider = _FakeProvider()
    provider.num_moe_experts = num_moe_experts
    fake_bridge = _FakeBridge(
        model_bridge=object.__new__(bridge_type),
        provider=provider,
    )
    monkeypatch.setattr(
        provider_module.AutoBridge,
        "from_hf_pretrained",
        lambda *args, **kwargs: fake_bridge,
    )
    monkeypatch.setattr(provider_module.torch.cuda, "device_count", lambda: 2)

    resolved = provider_module.get_provider("unused-model")

    assert resolved is provider
    assert provider.finalized is True
    assert resolved.attention_backend is AttnBackend.auto
    assert resolved.recompute_granularity == "full"
    assert resolved.recompute_method == "uniform"
    assert resolved.recompute_num_layers == 1
    assert resolved.tensor_model_parallel_size == 2
    assert resolved.context_parallel_size == 1
    assert resolved.pipeline_model_parallel_size == 1
    assert (
        resolved.expert_model_parallel_size == expected_expert_model_parallel_size
    )
    assert resolved.expert_tensor_parallel_size == 1
    assert resolved.sequence_parallel is True
    assert resolved.moe_shared_expert_overlap is expected_moe_shared_expert_overlap
    if num_moe_experts:
        assert resolved.moe_router_dtype == "fp32"
        assert resolved.moe_aux_loss_coeff == 0.0
    assert resolved.calculate_per_token_loss is True

    layer_spec = provider_module._resolve_layer_spec(
        resolved.transformer_layer_spec,
        resolved,
        vp_stage=7,
    )
    layer_spec = cast(Any, layer_spec)
    assert (
        layer_spec.submodules.self_attention.submodules.core_attention
        is FlexDotProductAttention
    )


def test_get_provider_rejects_unsupported_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bridge = _FakeBridge(model_bridge=object(), provider=_FakeProvider())
    monkeypatch.setattr(
        provider_module.AutoBridge,
        "from_hf_pretrained",
        lambda *args, **kwargs: fake_bridge,
    )

    with pytest.raises(
        AssertionError,
        match="Only Qwen3 MoE and Qwen3.5/3.6 dense or MoE models are supported",
    ):
        provider_module.get_provider("unsupported-model")


def test_get_provider_preserves_hybrid_qwen35_layer_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeHybridProvider()
    fake_bridge = _FakeBridge(
        model_bridge=object.__new__(Qwen35VLMoEBridge),
        provider=provider,
    )
    monkeypatch.setattr(
        provider_module.AutoBridge,
        "from_hf_pretrained",
        lambda *args, **kwargs: fake_bridge,
    )
    monkeypatch.setattr(provider_module.torch.cuda, "device_count", lambda: 1)

    resolved = provider_module.get_provider("unused-qwen35")
    layer_spec = provider_module._resolve_layer_spec(
        resolved.transformer_layer_spec,
        resolved,
        vp_stage=0,
    )

    layer_specs = getattr(layer_spec, "layer_specs", None)
    assert layer_specs is not None
    gdn_layer, attention_layer = layer_specs
    assert not hasattr(gdn_layer.submodules.self_attention.submodules, "core_attention")
    assert (
        attention_layer.submodules.self_attention.submodules.core_attention
        is FlexDotProductAttention
    )
