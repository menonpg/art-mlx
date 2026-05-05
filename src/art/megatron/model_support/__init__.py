from art.megatron.model_support.registry import (
    DEFAULT_DENSE_SPEC,
    PROBE_ONLY_MODEL_SUPPORT_SPECS,
    QWEN3_5_DENSE_MODELS,
    QWEN3_5_DENSE_SPEC,
    QWEN3_5_MODELS,
    QWEN3_5_MOE_MODELS,
    QWEN3_5_MOE_SPEC,
    QWEN3_DENSE_MODELS,
    QWEN3_DENSE_SPEC,
    QWEN3_MOE_MODELS,
    QWEN3_MOE_SPEC,
    VALIDATED_MODEL_SUPPORT_SPECS,
    UnsupportedModelArchitectureError,
    default_target_modules_for_model,
    get_model_support_handler,
    get_model_support_handler_for_spec,
    get_model_support_spec,
    is_model_support_registered,
    list_model_support_specs,
    model_requires_merged_rollout,
    model_uses_expert_parallel,
    native_vllm_lora_status_for_model,
)
from art.megatron.model_support.spec import (
    ArchitectureReport,
    DependencyFloor,
    LayerFamilyInstance,
    MinimalLayerCoverageReport,
    ModelSupportHandler,
    ModelSupportSpec,
    NativeVllmLoraStatus,
    RolloutWeightsMode,
    ValidationReport,
    ValidationStageResult,
)

_LAZY_EXPORT_MODULES = {
    "inspect_architecture": "art.megatron.model_support.discovery",
    "summarize_layer_families": "art.megatron.model_support.discovery",
    "MANDATORY_VALIDATION_STAGES": "art.megatron.model_support.workflow",
    "NATIVE_VLLM_LORA_STAGE": "art.megatron.model_support.workflow",
    "assess_minimal_layer_coverage": "art.megatron.model_support.workflow",
    "build_validation_report": "art.megatron.model_support.workflow",
    "build_validation_stage_names": "art.megatron.model_support.workflow",
    "detect_dependency_versions": "art.megatron.model_support.workflow",
    "initialize_validation_report": "art.megatron.model_support.workflow",
}


def __getattr__(name: str):
    import importlib

    try:
        module_name = _LAZY_EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "ArchitectureReport",
    "DEFAULT_DENSE_SPEC",
    "DependencyFloor",
    "LayerFamilyInstance",
    "MANDATORY_VALIDATION_STAGES",
    "MinimalLayerCoverageReport",
    "ModelSupportHandler",
    "ModelSupportSpec",
    "NativeVllmLoraStatus",
    "NATIVE_VLLM_LORA_STAGE",
    "QWEN3_5_DENSE_MODELS",
    "QWEN3_5_DENSE_SPEC",
    "QWEN3_5_MODELS",
    "QWEN3_5_MOE_MODELS",
    "QWEN3_DENSE_MODELS",
    "QWEN3_DENSE_SPEC",
    "QWEN3_MOE_MODELS",
    "QWEN3_MOE_SPEC",
    "QWEN3_5_MOE_SPEC",
    "PROBE_ONLY_MODEL_SUPPORT_SPECS",
    "RolloutWeightsMode",
    "ValidationReport",
    "ValidationStageResult",
    "UnsupportedModelArchitectureError",
    "VALIDATED_MODEL_SUPPORT_SPECS",
    "assess_minimal_layer_coverage",
    "build_validation_report",
    "build_validation_stage_names",
    "default_target_modules_for_model",
    "detect_dependency_versions",
    "get_model_support_handler",
    "get_model_support_handler_for_spec",
    "get_model_support_spec",
    "initialize_validation_report",
    "inspect_architecture",
    "is_model_support_registered",
    "list_model_support_specs",
    "model_uses_expert_parallel",
    "model_requires_merged_rollout",
    "native_vllm_lora_status_for_model",
    "summarize_layer_families",
]
