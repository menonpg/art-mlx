import importlib.metadata

from art.megatron.model_support.discovery import inspect_architecture
from art.megatron.model_support.registry import get_model_support_spec
from art.megatron.model_support.spec import ValidationReport, ValidationStageResult

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
