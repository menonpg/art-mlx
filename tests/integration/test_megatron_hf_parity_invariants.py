from types import SimpleNamespace

import pytest
import torch

from .megatron_hf_parity import (
    build_parity_sample_indices,
    run_hf_parity,
    set_hf_config_num_layers,
)
from .megatron_hf_parity_worker import (
    _mapping_supports_derivative_parity,
    _normalize_hf_tensor_map_for_bridge,
)
from .megatron_oracle_harness import OracleCaseConfig


def test_build_parity_sample_indices_pads_with_none() -> None:
    assert build_parity_sample_indices(
        num_sequences=2,
        global_grad_accumulation_sequences=4,
    ) == [0, 1, None, None]


def test_set_hf_config_num_layers_updates_supported_field() -> None:
    config = SimpleNamespace(num_hidden_layers=28)

    field = set_hf_config_num_layers(config, 4)

    assert field == "num_hidden_layers"
    assert config.num_hidden_layers == 4


def test_set_hf_config_num_layers_updates_nested_text_config() -> None:
    text_config = SimpleNamespace(
        num_hidden_layers=40,
        layer_types=["linear_attention", "linear_attention", "full_attention"] * 2,
        mlp_only_layers=[1, 4, 7],
    )
    config = SimpleNamespace(text_config=text_config)

    field = set_hf_config_num_layers(config, 4)

    assert field == "text_config.num_hidden_layers"
    assert text_config.num_hidden_layers == 4
    assert text_config.layer_types == [
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
    ]
    assert text_config.mlp_only_layers == [1]


def test_run_hf_parity_rejects_uncovered_toy_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "integration.megatron_hf_parity.assess_minimal_layer_coverage",
        lambda **_: SimpleNamespace(
            covered=False,
            missing_layer_families=["standard_attention"],
            unresolved_risks=[],
        ),
    )

    with pytest.raises(
        AssertionError,
        match="HF parity toy model does not cover required layer families",
    ):
        run_hf_parity(
            case_config=OracleCaseConfig(
                base_model="Qwen/Qwen3.5-35B-A3B",
                num_layers=2,
            )
        )


def test_normalize_hf_tensor_map_for_bridge_adds_language_model_prefix() -> None:
    normalized = _normalize_hf_tensor_map_for_bridge(
        {
            "model.layers.0.input_layernorm.weight": torch.ones(1),
            "lm_head.weight": torch.ones(1),
        },
        {
            "model.language_model.layers.0.input_layernorm.weight",
            "lm_head.weight",
        },
    )

    assert set(normalized) == {
        "model.language_model.layers.0.input_layernorm.weight",
        "lm_head.weight",
    }


def test_mapping_supports_derivative_parity_rejects_affine_weight_exports() -> None:
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        RMSNorm2ZeroCenteredRMSNormMapping,
    )

    assert _mapping_supports_derivative_parity(AutoMapping("a", "b")) is True
    assert (
        _mapping_supports_derivative_parity(
            RMSNorm2ZeroCenteredRMSNormMapping("a", "b")
        )
        is False
    )
