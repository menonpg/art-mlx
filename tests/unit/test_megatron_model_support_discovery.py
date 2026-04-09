from types import SimpleNamespace

from art.megatron.model_support.discovery import (
    inspect_architecture,
    recommended_min_layers,
    summarize_layer_families,
)
from art.megatron.model_support.spec import LayerFamilyInstance, ModelSupportSpec
from art.megatron.provider_common import ProviderBundle


def test_summarize_layer_families_counts_duplicate_keys() -> None:
    summarized = summarize_layer_families(
        [
            LayerFamilyInstance(key="standard_attention", layer_index=3),
            LayerFamilyInstance(key="dense_mlp", layer_index=0),
            LayerFamilyInstance(key="standard_attention", layer_index=5),
        ]
    )

    assert summarized == [
        LayerFamilyInstance(key="dense_mlp", count=1, layer_index=0),
        LayerFamilyInstance(key="standard_attention", count=2, layer_index=3),
    ]


def test_inspect_architecture_uses_handler_report(monkeypatch) -> None:
    handler = SimpleNamespace(
        key="qwen3_5_moe",
        collect_layer_families=lambda provider: [
            LayerFamilyInstance(key="standard_attention", layer_index=3),
            LayerFamilyInstance(key="gated_delta_net_attention", layer_index=0),
            LayerFamilyInstance(key="standard_attention", layer_index=7),
        ],
    )
    provider_bundle = ProviderBundle(
        provider=SimpleNamespace(),
        bridge=SimpleNamespace(_model_bridge=SimpleNamespace()),
        handler=handler,
        spec=ModelSupportSpec(
            key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
            default_target_modules=("q_proj",),
        ),
    )
    monkeypatch.setattr(
        "art.megatron.model_support.discovery.get_provider_bundle",
        lambda *args, **kwargs: provider_bundle,
    )

    report = inspect_architecture("Qwen/Qwen3.5-35B-A3B")

    assert report.base_model == "Qwen/Qwen3.5-35B-A3B"
    assert report.model_key == "qwen3_5_moe"
    assert report.handler_key == "qwen3_5_moe"
    assert report.bridge_type == "SimpleNamespace"
    assert report.provider_type == "SimpleNamespace"
    assert report.layer_families == [
        LayerFamilyInstance(key="gated_delta_net_attention", count=1, layer_index=0),
        LayerFamilyInstance(key="standard_attention", count=2, layer_index=3),
    ]
    assert report.recommended_min_layers == 4
    assert report.unresolved_risks == []


def test_recommended_min_layers_uses_highest_representative_layer_index() -> None:
    assert (
        recommended_min_layers(
            [
                LayerFamilyInstance(key="standard_attention", layer_index=3),
                LayerFamilyInstance(key="gated_delta_net_attention", layer_index=0),
            ]
        )
        == 4
    )
