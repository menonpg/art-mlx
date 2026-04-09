from types import SimpleNamespace

import pytest

from .megatron_hf_parity import (
    build_parity_sample_indices,
    run_hf_parity,
    set_hf_config_num_layers,
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
