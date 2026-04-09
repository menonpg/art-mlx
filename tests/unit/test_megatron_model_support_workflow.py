from art.megatron.model_support.spec import ArchitectureReport, LayerFamilyInstance
from art.megatron.model_support.workflow import (
    MANDATORY_VALIDATION_STAGES,
    NATIVE_VLLM_LORA_STAGE,
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
