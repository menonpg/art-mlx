from types import SimpleNamespace

from art.megatron.model_support.spec import (
    ArchitectureReport,
    LayerFamilyInstance,
    ValidationStageResult,
)
from art.megatron.model_support.workflow import (
    MANDATORY_VALIDATION_STAGES,
    NATIVE_VLLM_LORA_STAGE,
    assess_minimal_layer_coverage,
    build_validation_report,
    build_validation_stage_names,
    run_correctness_sensitivity_stage,
    run_lora_coverage_stage,
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
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.run_hf_parity_stage",
        lambda *, base_model, architecture: ValidationStageResult(
            name="hf_parity",
            passed=True,
            metrics={"signal": "pass", "requested_num_layers": 1},
            artifact_dir="/tmp/hf_parity",
        ),
    )
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.run_lora_coverage_stage",
        lambda *, base_model, architecture: ValidationStageResult(
            name="lora_coverage",
            passed=True,
            metrics={"wrapped_adapter_prefix_count": 12},
        ),
    )
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.run_correctness_sensitivity_stage",
        lambda *, base_model, architecture: ValidationStageResult(
            name="correctness_sensitivity",
            passed=True,
            metrics={"correctness_variant_count": 4, "sensitivity_variant_count": 9},
            artifact_dir="/tmp/correctness",
        ),
    )

    report = build_validation_report(base_model="Qwen/Qwen3.5-35B-A3B")

    assert report.base_model == "Qwen/Qwen3.5-35B-A3B"
    assert report.model_key == "qwen3_5_moe"
    assert report.dependency_versions == {"transformers": "5.2.0"}
    dependency_stage = next(
        stage for stage in report.stages if stage.name == "dependency_resolution"
    )
    assert dependency_stage.passed is True
    assert dependency_stage.metrics == {"transformers": "5.2.0"}
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
    hf_parity_stage = next(
        stage for stage in report.stages if stage.name == "hf_parity"
    )
    assert hf_parity_stage.passed is True
    assert hf_parity_stage.metrics == {"signal": "pass", "requested_num_layers": 1}
    assert hf_parity_stage.artifact_dir == "/tmp/hf_parity"
    lora_coverage_stage = next(
        stage for stage in report.stages if stage.name == "lora_coverage"
    )
    assert lora_coverage_stage.passed is True
    assert lora_coverage_stage.metrics == {"wrapped_adapter_prefix_count": 12}
    correctness_stage = next(
        stage for stage in report.stages if stage.name == "correctness_sensitivity"
    )
    assert correctness_stage.passed is True
    assert correctness_stage.metrics == {
        "correctness_variant_count": 4,
        "sensitivity_variant_count": 9,
    }
    assert correctness_stage.artifact_dir == "/tmp/correctness"


def test_build_validation_report_captures_hf_parity_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.inspect_architecture",
        lambda base_model: ArchitectureReport(
            base_model=base_model,
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
            layer_families=[],
            recommended_min_layers=4,
        ),
    )
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.detect_dependency_versions",
        lambda: {},
    )

    def _fail_hf_parity(*, base_model: str, architecture: ArchitectureReport) -> None:
        del base_model, architecture
        raise AssertionError("parity failed")

    monkeypatch.setattr(
        "art.megatron.model_support.workflow.run_hf_parity_stage",
        _fail_hf_parity,
    )
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.run_lora_coverage_stage",
        lambda *, base_model, architecture: ValidationStageResult(
            name="lora_coverage",
            passed=True,
            metrics={},
        ),
    )
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.run_correctness_sensitivity_stage",
        lambda *, base_model, architecture: ValidationStageResult(
            name="correctness_sensitivity",
            passed=True,
            metrics={},
        ),
    )

    report = build_validation_report(base_model="Qwen/Qwen3.5-35B-A3B")

    hf_parity_stage = next(
        stage for stage in report.stages if stage.name == "hf_parity"
    )
    assert hf_parity_stage.passed is False
    assert hf_parity_stage.metrics == {"error": "AssertionError: parity failed"}
    assert hf_parity_stage.artifact_dir is None


def test_build_validation_report_captures_lora_coverage_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.inspect_architecture",
        lambda base_model: ArchitectureReport(
            base_model=base_model,
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
            layer_families=[],
            recommended_min_layers=4,
        ),
    )
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.detect_dependency_versions",
        lambda: {},
    )
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.run_hf_parity_stage",
        lambda *, base_model, architecture: ValidationStageResult(
            name="hf_parity",
            passed=True,
            metrics={},
        ),
    )

    def _fail_lora_coverage(
        *,
        base_model: str,
        architecture: ArchitectureReport,
    ) -> None:
        del base_model, architecture
        raise RuntimeError("missing wrapped targets")

    monkeypatch.setattr(
        "art.megatron.model_support.workflow.run_lora_coverage_stage",
        _fail_lora_coverage,
    )
    monkeypatch.setattr(
        "art.megatron.model_support.workflow.run_correctness_sensitivity_stage",
        lambda *, base_model, architecture: ValidationStageResult(
            name="correctness_sensitivity",
            passed=True,
            metrics={},
        ),
    )

    report = build_validation_report(base_model="Qwen/Qwen3.5-35B-A3B")

    lora_coverage_stage = next(
        stage for stage in report.stages if stage.name == "lora_coverage"
    )
    assert lora_coverage_stage.passed is False
    assert lora_coverage_stage.metrics == {
        "error": "RuntimeError: missing wrapped targets"
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


def test_run_lora_coverage_stage_reports_missing_targets(monkeypatch) -> None:
    architecture = ArchitectureReport(
        base_model="Qwen/Qwen3.5-35B-A3B",
        model_key="qwen3_5_moe",
        handler_key="qwen3_5_moe",
        recommended_min_layers=4,
    )
    oracle_module = SimpleNamespace(
        OracleCaseConfig=lambda **kwargs: SimpleNamespace(**kwargs)
    )
    coverage_report = SimpleNamespace(
        missing_wrapped_target_modules=["in_proj_z"],
        missing_exported_target_modules=[],
        model_dump=lambda mode="json": {
            "base_model": "Qwen/Qwen3.5-35B-A3B",
            "missing_wrapped_target_modules": ["in_proj_z"],
        },
    )
    coverage_module = SimpleNamespace(
        run_lora_coverage=lambda case_config: coverage_report
    )

    def _import_integration_module(name: str):
        if name == "integration.megatron_oracle_harness":
            return oracle_module
        if name == "integration.megatron_lora_coverage":
            return coverage_module
        raise AssertionError(name)

    monkeypatch.setattr(
        "art.megatron.model_support.workflow._import_integration_module",
        _import_integration_module,
    )

    stage = run_lora_coverage_stage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture=architecture,
    )

    assert stage.name == "lora_coverage"
    assert stage.passed is False
    assert stage.metrics == {
        "base_model": "Qwen/Qwen3.5-35B-A3B",
        "missing_wrapped_target_modules": ["in_proj_z"],
    }


def test_run_correctness_sensitivity_stage_summarizes_reports(monkeypatch) -> None:
    architecture = ArchitectureReport(
        base_model="Qwen/Qwen3.5-35B-A3B",
        model_key="qwen3_5_moe",
        handler_key="qwen3_5_moe",
        recommended_min_layers=4,
    )
    oracle_module = SimpleNamespace(
        OracleCaseConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        TOPOLOGIES=[SimpleNamespace(world_size=lambda: 2)],
        EXTENDED_TOPOLOGIES=[SimpleNamespace(world_size=lambda: 4)],
        extended_topologies_enabled=lambda: False,
        selected_oracle_objectives=lambda: ["sft"],
        supported_sensitivity_mutations_for_objective=lambda objective: (
            ["skip_finalize"] if objective == "sft" else []
        ),
        sensitivity_required_world_size=lambda mutations: 2,
        available_gpu_count=lambda: 2,
        run_suite=lambda case_config: [
            SimpleNamespace(
                variant="sft_topology_tp2",
                topology="tp2",
                signal="pass",
                fail_count=0,
            )
        ],
        run_sensitivity_suite=lambda case_config, mutations: [
            SimpleNamespace(
                variant="sft_sensitivity_skip_finalize",
                topology="tp2",
                signal="fail",
                expected_signal="fail",
                fail_count=1,
            )
        ],
        ensure_case_artifacts=lambda case_config: SimpleNamespace(
            case_dir="/tmp/oracle"
        ),
    )
    monkeypatch.setattr(
        "art.megatron.model_support.workflow._import_integration_module",
        lambda name: oracle_module,
    )

    stage = run_correctness_sensitivity_stage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture=architecture,
    )

    assert stage.name == "correctness_sensitivity"
    assert stage.passed is True
    assert stage.metrics["requested_num_layers"] == 4
    assert stage.metrics["objectives"] == ["sft"]
    assert stage.metrics["sensitivity_mutations"] == ["skip_finalize"]
    assert stage.metrics["required_gpu_count"] == 2
    assert stage.metrics["correctness_variant_count"] == 1
    assert stage.metrics["sensitivity_variant_count"] == 1
    assert stage.artifact_dir == "/tmp/oracle"
