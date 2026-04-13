from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.bridge.models.qwen.qwen3_moe_bridge")

from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.core.transformer.enums import AttnBackend

from art.megatron.flex_attention import FlexDotProductAttention
import art.megatron.provider as provider_module


class _FakeProvider:
    def __init__(self) -> None:
        self.transformer_layer_spec = self._base_layer_spec
        self.finalized = False
        self.overlap_moe_expert_parallel_comm = False

    def _base_layer_spec(
        self, config: object, vp_stage: int | None = None
    ) -> SimpleNamespace:
        del config, vp_stage
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
            ),
        )
        return SimpleNamespace(layer_specs=[gdn_layer, attention_layer])


class _FakeBridge:
    def __init__(self, *, model_bridge: object, provider: _FakeProvider) -> None:
        self._model_bridge = model_bridge
        self._provider = provider
        self.hf_pretrained = SimpleNamespace(model_name_or_path="unused")

    def to_megatron_provider(self) -> _FakeProvider:
        return self._provider


def test_get_provider_accepts_supported_qwen_moe_bridges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider()
    fake_bridge = _FakeBridge(
        model_bridge=object.__new__(Qwen3MoEBridge),
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
    assert resolved.expert_model_parallel_size == 2
    assert resolved.expert_tensor_parallel_size == 1
    assert resolved.sequence_parallel is True
    assert resolved.moe_shared_expert_overlap is True
    assert resolved.moe_router_dtype == "fp32"
    assert resolved.moe_aux_loss_coeff == 0.0
    assert resolved.calculate_per_token_loss is True

    layer_spec = cast(Any, resolved.transformer_layer_spec)(resolved, vp_stage=7)
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
        match="Only Qwen3 and Qwen3.5 MoE models are supported",
    ):
        provider_module.get_provider("unsupported-model")


def test_get_provider_preserves_hybrid_layer_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeHybridProvider()
    fake_bridge = _FakeBridge(
        model_bridge=object.__new__(Qwen3MoEBridge),
        provider=provider,
    )
    monkeypatch.setattr(
        provider_module.AutoBridge,
        "from_hf_pretrained",
        lambda *args, **kwargs: fake_bridge,
    )
    monkeypatch.setattr(provider_module.torch.cuda, "device_count", lambda: 1)

    resolved = provider_module.get_provider("unused-qwen")
    layer_spec = cast(Any, resolved).transformer_layer_spec(resolved, vp_stage=0)

    assert hasattr(layer_spec, "layer_specs")
    gdn_layer, attention_layer = cast(Any, layer_spec).layer_specs
    assert not hasattr(gdn_layer.submodules.self_attention.submodules, "core_attention")
    assert (
        attention_layer.submodules.self_attention.submodules.core_attention
        is FlexDotProductAttention
    )


def test_finalize_provider_bundle_uses_post_prepare_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider()
    setattr(provider, "num_moe_experts", 8)
    fake_bridge = _FakeBridge(
        model_bridge=object.__new__(Qwen3MoEBridge),
        provider=provider,
    )
    dispatcher_calls: list[tuple[int, int, str]] = []
    monkeypatch.setattr(
        provider_module.AutoBridge,
        "from_hf_pretrained",
        lambda *args, **kwargs: fake_bridge,
    )
    monkeypatch.setattr(provider_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        provider_module,
        "apply_flex_dispatcher_backend",
        lambda provider, moe_flex_dispatcher_backend: dispatcher_calls.append(
            (
                int(provider.tensor_model_parallel_size),
                int(provider.expert_model_parallel_size),
                cast(str, moe_flex_dispatcher_backend),
            )
        ),
    )

    bundle = provider_module.prepare_provider_bundle("unused-model")

    assert provider.finalized is False
    assert getattr(provider, "tensor_model_parallel_size") == 2
    assert getattr(provider, "expert_model_parallel_size") == 2

    bundle.provider.tensor_model_parallel_size = 1
    bundle.provider.expert_model_parallel_size = 1
    bundle.provider.sequence_parallel = False
    provider_module.finalize_provider_bundle(bundle)

    assert dispatcher_calls == []
    assert provider.finalized is True
    assert getattr(provider, "sequence_parallel") is False


def test_get_provider_bundle_single_gpu_parity_uses_clean_runtime_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider()
    fake_bridge = _FakeBridge(
        model_bridge=object.__new__(Qwen3MoEBridge),
        provider=provider,
    )
    monkeypatch.setattr(
        provider_module.AutoBridge,
        "from_hf_pretrained",
        lambda *args, **kwargs: fake_bridge,
    )
    monkeypatch.setattr(provider_module.torch.cuda, "device_count", lambda: 2)

    bundle = provider_module.get_provider_bundle(
        "unused-model",
        runtime_profile="single_gpu_parity",
    )
    resolved = bundle.provider

    assert resolved.tensor_model_parallel_size == 1
    assert resolved.context_parallel_size == 1
    assert resolved.pipeline_model_parallel_size == 1
    assert resolved.expert_model_parallel_size == 1
    assert resolved.expert_tensor_parallel_size == 1
    assert resolved.sequence_parallel is False
    assert resolved.recompute_granularity is None
    assert resolved.recompute_method is None
    assert resolved.recompute_num_layers is None
    assert resolved.overlap_moe_expert_parallel_comm is False
    assert resolved.moe_token_dispatcher_type == "alltoall"
    assert resolved.moe_shared_expert_overlap is False

    layer_spec = resolved.transformer_layer_spec(resolved, vp_stage=0)
    assert (
        layer_spec.submodules.self_attention.submodules.core_attention
        is not FlexDotProductAttention
    )
