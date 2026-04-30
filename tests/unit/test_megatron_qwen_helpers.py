from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("megatron.bridge")

import torch

from art.megatron import lora as megatron_lora
from art.megatron.lora import SelfAttentionLinearQKVLoRA
from art.megatron.train import _canonical_art_param_name


def test_canonical_art_param_name_strips_art_wrapper_segments() -> None:
    assert (
        _canonical_art_param_name(
            "module.language_model.decoder.layers.0.self_attention.out_proj.linear_proj.weight"
        )
        == "language_model.decoder.layers.0.self_attention.out_proj.weight"
    )
    assert (
        _canonical_art_param_name(
            "module.language_model.decoder.layers.0.mlp.linear_fc2.row_parallel_lora.linear_proj.weight"
        )
        == "language_model.decoder.layers.0.mlp.linear_fc2.weight"
    )


def test_self_attention_linear_qkv_lora_accepts_nongated_qwen3_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "art.megatron.lora.ps.get_tensor_model_parallel_world_size", lambda: 1
    )
    provider: Any = SimpleNamespace(
        kv_channels=128,
        num_query_groups=4,
        num_attention_heads=32,
        attention_output_gate=False,
    )
    q_out_features = provider.kv_channels * provider.num_attention_heads
    kv_out_features = provider.kv_channels * provider.num_query_groups
    linear_qkv: Any = SimpleNamespace(
        weight=torch.empty(q_out_features + 2 * kv_out_features, 16),
        in_features=16,
        return_layernorm_output=False,
        return_layernorm_output_gathered=False,
    )

    wrapped = SelfAttentionLinearQKVLoRA(
        adapter_model_prefix="base_model.model.model.layers.0.self_attn",
        linear_qkv=cast(Any, linear_qkv),
        rank=4,
        alpha=8.0,
        provider=cast(Any, provider),
    )

    assert wrapped.attention_output_gate is False
    assert wrapped.q_proj_lora.B_T.shape[-1] == q_out_features


def test_match_sequence_parallel_output_shape_scatters_first_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_out = torch.arange(12).reshape(4, 3)
    base_out = torch.empty(2, 3)
    scatter_calls: list[torch.Tensor] = []

    def fake_scatter(tensor: torch.Tensor) -> torch.Tensor:
        scatter_calls.append(tensor)
        return tensor[:2].contiguous()

    monkeypatch.setattr(megatron_lora, "_get_shard_world_size", lambda _domain: 2)
    monkeypatch.setattr(
        megatron_lora,
        "scatter_to_sequence_parallel_region",
        fake_scatter,
    )

    result = megatron_lora._match_sequence_parallel_output_shape(
        adapter_out,
        base_out,
        adapter_model_prefix="model.layers.0.mlp.shared_expert",
    )

    assert scatter_calls == [adapter_out]
    assert result.shape == base_out.shape
    assert torch.equal(result, adapter_out[:2])


def test_match_sequence_parallel_output_shape_gathers_first_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_out = torch.arange(6).reshape(2, 3)
    base_out = torch.empty(4, 3)
    gather_calls: list[tuple[torch.Tensor, bool]] = []

    def fake_gather(
        tensor: torch.Tensor,
        tensor_parallel_output_grad: bool = True,
    ) -> torch.Tensor:
        gather_calls.append((tensor, tensor_parallel_output_grad))
        return torch.cat([tensor, tensor], dim=0)

    monkeypatch.setattr(megatron_lora, "_get_shard_world_size", lambda _domain: 2)
    monkeypatch.setattr(
        megatron_lora,
        "gather_from_sequence_parallel_region",
        fake_gather,
    )

    result = megatron_lora._match_sequence_parallel_output_shape(
        adapter_out,
        base_out,
        adapter_model_prefix="model.layers.0.mlp.shared_expert",
    )

    assert gather_calls == [(adapter_out, True)]
    assert result.shape == base_out.shape
    assert torch.equal(result, torch.cat([adapter_out, adapter_out], dim=0))
