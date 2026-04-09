import importlib.metadata

from art.megatron.model_support.discovery import inspect_architecture
from art.megatron.model_support.registry import get_model_support_spec
from art.megatron.model_support.spec import (
    ArchitectureReport,
    MinimalLayerCoverageReport,
    ValidationReport,
    ValidationStageResult,
)

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
    for stage in report.stages:
        if stage.name != "architecture_discovery":
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
