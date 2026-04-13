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


def run_lora_coverage_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
) -> ValidationStageResult:
    lora_coverage = _import_integration_module("integration.megatron_lora_coverage")
    oracle_harness = _import_integration_module("integration.megatron_oracle_harness")
    case_config = oracle_harness.OracleCaseConfig(
        base_model=base_model,
        precision="fp32",
        num_layers=max(1, architecture.recommended_min_layers),
        num_steps=1,
    )
    report = lora_coverage.run_lora_coverage(case_config)
    return ValidationStageResult(
        name="lora_coverage",
        passed=not report.missing_wrapped_target_modules
        and not report.missing_exported_target_modules,
        metrics=report.model_dump(mode="json"),
    )


def run_correctness_sensitivity_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
) -> ValidationStageResult:
    oracle_harness = _import_integration_module("integration.megatron_oracle_harness")
    case_config = oracle_harness.OracleCaseConfig(
        base_model=base_model,
        precision="fp32",
        num_layers=max(1, architecture.recommended_min_layers),
        num_steps=1,
    )
    suite_topologies = list(oracle_harness.TOPOLOGIES)
    if oracle_harness.extended_topologies_enabled():
        suite_topologies.extend(oracle_harness.EXTENDED_TOPOLOGIES)
    suite_world_size = max(topology.world_size() for topology in suite_topologies)
    objectives = list(oracle_harness.selected_oracle_objectives())
    mutations: list[str] = []
    for objective in objectives:
        for mutation in oracle_harness.supported_sensitivity_mutations_for_objective(
            objective
        ):
            if mutation not in mutations:
                mutations.append(mutation)
    sensitivity_world_size = oracle_harness.sensitivity_required_world_size(mutations)
    available_gpu_count = oracle_harness.available_gpu_count()
    required_gpu_count = max(suite_world_size, sensitivity_world_size)
    if available_gpu_count < required_gpu_count:
        raise RuntimeError(
            "Need "
            f"{required_gpu_count} GPUs for correctness/sensitivity, found {available_gpu_count}"
        )
    suite_reports = oracle_harness.run_suite(case_config=case_config)
    sensitivity_reports = oracle_harness.run_sensitivity_suite(
        case_config=case_config,
        mutations=mutations,
    )
    case_artifacts = oracle_harness.ensure_case_artifacts(case_config)
    return ValidationStageResult(
        name="correctness_sensitivity",
        passed=True,
        metrics={
            "requested_num_layers": case_config.num_layers,
            "objectives": objectives,
            "sensitivity_mutations": mutations,
            "required_gpu_count": required_gpu_count,
            "correctness_variant_count": len(suite_reports),
            "correctness_variants": [
                {
                    "variant": report.variant,
                    "topology": report.topology,
                    "signal": report.signal,
                    "fail_count": report.fail_count,
                }
                for report in suite_reports
            ],
            "sensitivity_variant_count": len(sensitivity_reports),
            "sensitivity_variants": [
                {
                    "variant": report.variant,
                    "topology": report.topology,
                    "signal": report.signal,
                    "expected_signal": report.expected_signal,
                    "fail_count": report.fail_count,
                }
                for report in sensitivity_reports
            ],
        },
        artifact_dir=case_artifacts.case_dir,
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
    stage_runners = {
        "hf_parity": run_hf_parity_stage,
        "lora_coverage": run_lora_coverage_stage,
        "correctness_sensitivity": run_correctness_sensitivity_stage,
    }
    stage_results: dict[str, ValidationStageResult] = {}
    for stage_name, stage_runner in stage_runners.items():
        try:
            stage_results[stage_name] = stage_runner(
                base_model=base_model,
                architecture=architecture,
            )
        except Exception as exc:
            stage_results[stage_name] = ValidationStageResult(
                name=stage_name,
                passed=False,
                metrics=_stage_error_metrics(exc),
            )
    for stage in report.stages:
        if stage.name == "dependency_resolution":
            stage.passed = True
            stage.metrics = dict(report.dependency_versions)
            continue
        if stage.name != "architecture_discovery":
            stage_result = stage_results.get(stage.name)
            if stage_result is not None:
                stage.passed = stage_result.passed
                stage.metrics = dict(stage_result.metrics)
                stage.artifact_dir = stage_result.artifact_dir
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
