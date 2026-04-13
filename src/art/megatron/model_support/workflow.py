import importlib
import importlib.metadata
from pathlib import Path
import sys
from typing import Any

from art.megatron.model_support.discovery import inspect_architecture
from art.megatron.model_support.registry import get_model_support_spec
from art.megatron.model_support.spec import (
    ArchitectureReport,
    MinimalLayerCoverageReport,
    ValidationReport,
    ValidationStageResult,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
TESTS_DIR = REPO_ROOT / "tests"

MANDATORY_VALIDATION_STAGES = (
    "dependency_resolution",
    "architecture_discovery",
    "hf_parity",
    "lora_coverage",
    "merged_vllm_serving",
    "correctness_sensitivity",
    "chat_template_rollout",
    "yes_no_trainability",
)
NATIVE_VLLM_LORA_STAGE = "native_vllm_lora"


def build_validation_stage_names(
    *,
    include_native_vllm_lora: bool = False,
) -> list[str]:
    stages = list(MANDATORY_VALIDATION_STAGES)
    if include_native_vllm_lora:
        stages.append(NATIVE_VLLM_LORA_STAGE)
    return stages


def detect_dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in ("transformers", "vllm", "megatron-bridge"):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def initialize_validation_report(
    *,
    base_model: str,
    include_native_vllm_lora: bool = False,
) -> ValidationReport:
    spec = get_model_support_spec(base_model)
    return ValidationReport(
        base_model=base_model,
        model_key=spec.key,
        dependency_versions=detect_dependency_versions(),
        stages=[
            ValidationStageResult(name=stage_name)
            for stage_name in build_validation_stage_names(
                include_native_vllm_lora=include_native_vllm_lora
            )
        ],
    )


def _stage_error_metrics(exc: Exception) -> dict[str, Any]:
    return {"error": f"{type(exc).__name__}: {exc}"}


def _import_integration_module(module_name: str) -> Any:
    tests_dir = str(TESTS_DIR)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    return importlib.import_module(module_name)


def run_hf_parity_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
) -> ValidationStageResult:
    hf_parity = _import_integration_module("integration.megatron_hf_parity")
    oracle_harness = _import_integration_module("integration.megatron_oracle_harness")
    case_config = oracle_harness.OracleCaseConfig(
        base_model=base_model,
        precision="fp32",
        num_layers=max(1, architecture.recommended_min_layers),
        num_steps=1,
    )
    report = hf_parity.run_hf_parity(case_config=case_config)
    case_artifacts = oracle_harness.ensure_case_artifacts(case_config)
    artifact_dir = str(
        Path(case_artifacts.case_dir) / hf_parity.HF_PARITY_OUTPUT_DIRNAME
    )
    return ValidationStageResult(
        name="hf_parity",
        passed=report.signal == "pass",
        metrics={
            "requested_num_layers": report.requested_num_layers,
            "coverage": report.coverage.model_dump(mode="json"),
            "signal": report.signal,
            "pass_count": report.pass_count,
            "fail_count": report.fail_count,
            "phases": [row.model_dump(mode="json") for row in report.metrics],
        },
        artifact_dir=artifact_dir,
    )


def build_validation_report(
    *,
    base_model: str,
    include_native_vllm_lora: bool = False,
) -> ValidationReport:
    report = initialize_validation_report(
        base_model=base_model,
        include_native_vllm_lora=include_native_vllm_lora,
    )
    architecture = inspect_architecture(base_model)
    hf_parity_stage: ValidationStageResult | None = None
    try:
        hf_parity_stage = run_hf_parity_stage(
            base_model=base_model,
            architecture=architecture,
        )
    except Exception as exc:
        hf_parity_stage = ValidationStageResult(
            name="hf_parity",
            passed=False,
            metrics=_stage_error_metrics(exc),
        )
    for stage in report.stages:
        if stage.name == "dependency_resolution":
            stage.passed = True
            stage.metrics = dict(report.dependency_versions)
            continue
        if stage.name != "architecture_discovery":
            if stage.name == "hf_parity":
                stage.passed = hf_parity_stage.passed
                stage.metrics = dict(hf_parity_stage.metrics)
                stage.artifact_dir = hf_parity_stage.artifact_dir
            continue
        stage.passed = not architecture.unresolved_risks
        stage.metrics = {
            "recommended_min_layers": architecture.recommended_min_layers,
            "layer_families": [
                family.model_dump() for family in architecture.layer_families
            ],
            "unresolved_risks": list(architecture.unresolved_risks),
        }
    return report


def assess_minimal_layer_coverage(
    *,
    base_model: str,
    num_layers: int,
    architecture: ArchitectureReport | None = None,
) -> MinimalLayerCoverageReport:
    architecture_report = architecture or inspect_architecture(base_model)
    missing_layer_families = [
        family.key
        for family in architecture_report.layer_families
        if family.layer_index is not None and family.layer_index >= num_layers
    ]
    return MinimalLayerCoverageReport(
        base_model=base_model,
        model_key=architecture_report.model_key,
        requested_num_layers=num_layers,
        recommended_min_layers=architecture_report.recommended_min_layers,
        covered=not missing_layer_families and not architecture_report.unresolved_risks,
        missing_layer_families=missing_layer_families,
        unresolved_risks=list(architecture_report.unresolved_risks),
    )
