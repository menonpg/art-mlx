from art.megatron.model_support.spec import ArchitectureReport, LayerFamilyInstance
from art.megatron.model_support.workflow import (
    MANDATORY_VALIDATION_STAGES,
    NATIVE_VLLM_LORA_STAGE,
    assess_minimal_layer_coverage,
    build_validation_report,
    build_validation_stage_names,
)


def test_build_validation_stage_names_has_fixed_order() -> None:
    assert build_validation_stage_names() == list(MANDATORY_VALIDATION_STAGES)
    assert build_validation_stage_names(include_native_vllm_lora=True) == [
        *MANDATORY_VALIDATION_STAGES,
        NATIVE_VLLM_LORA_STAGE,
    ]


def test_build_validation_report_populates_architecture_stage(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.inspect_architecture",
        lambda base_model: ArchitectureReport(
            base_model=base_model,
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
            layer_families=[LayerFamilyInstance(key="standard_attention", count=2)],
            recommended_min_layers=1,
        ),
    )
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.detect_dependency_versions",
        lambda: {"transformers": "5.2.0"},
    )

    report = build_validation_report(base_model="Qwen/Qwen3.5-35B-A3B")

    assert report.base_model == "Qwen/Qwen3.5-35B-A3B"
    assert report.model_key == "qwen3_5_moe"
    assert report.dependency_versions == {"transformers": "5.2.0"}
    architecture_stage = next(
        stage for stage in report.stages if stage.name == "architecture_discovery"
    )
    assert architecture_stage.passed is True
    assert architecture_stage.metrics == {
        "recommended_min_layers": 1,
        "layer_families": [
            {
                "key": "standard_attention",
                "count": 2,
                "layer_index": None,
                "module_path": None,
                "module_type": None,
            }
        ],
        "unresolved_risks": [],
    }


def test_assess_minimal_layer_coverage_reports_missing_families(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.inspect_architecture",
        lambda base_model: ArchitectureReport(
            base_model=base_model,
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
            layer_families=[
                LayerFamilyInstance(key="gated_delta_net_attention", layer_index=0),
                LayerFamilyInstance(key="standard_attention", layer_index=3),
                LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
                LayerFamilyInstance(key="shared_experts_mlp", layer_index=0),
            ],
            recommended_min_layers=4,
        ),
    )

    coverage = assess_minimal_layer_coverage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        num_layers=2,
    )

    assert coverage.covered is False
    assert coverage.requested_num_layers == 2
    assert coverage.recommended_min_layers == 4
    assert coverage.missing_layer_families == ["standard_attention"]
    assert coverage.unresolved_risks == []


def test_assess_minimal_layer_coverage_passes_when_prefix_covers_all_families(
    monkeypatch,
) -> None:
    architecture = ArchitectureReport(
        base_model="Qwen/Qwen3.5-35B-A3B",
        model_key="qwen3_5_moe",
        handler_key="qwen3_5_moe",
        layer_families=[
            LayerFamilyInstance(key="gated_delta_net_attention", layer_index=0),
            LayerFamilyInstance(key="standard_attention", layer_index=3),
            LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
            LayerFamilyInstance(key="shared_experts_mlp", layer_index=0),
        ],
        recommended_min_layers=4,
    )

    coverage = assess_minimal_layer_coverage(
        base_model=architecture.base_model,
        num_layers=4,
        architecture=architecture,
    )

    assert coverage.covered is True
    assert coverage.missing_layer_families == []
