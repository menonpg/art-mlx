from art.megatron.model_support.handlers import (
    DEFAULT_DENSE_HANDLER,
    QWEN3_5_MOE_HANDLER,
    QWEN3_MOE_HANDLER,
)
from art.megatron.model_support.spec import (
    DependencyFloor,
    ModelSupportHandler,
    ModelSupportSpec,
)

_DENSE_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_QWEN3_5_MOE_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DEFAULT_DENSE_SPEC = ModelSupportSpec(
    key="default_dense",
    handler_key=DEFAULT_DENSE_HANDLER.key,
    default_target_modules=_DENSE_TARGET_MODULES,
)

QWEN3_MOE_SPEC = ModelSupportSpec(
    key="qwen3_moe",
    handler_key=QWEN3_MOE_HANDLER.key,
    default_target_modules=_DENSE_TARGET_MODULES,
)

QWEN3_5_MOE_SPEC = ModelSupportSpec(
    key="qwen3_5_moe",
    handler_key=QWEN3_5_MOE_HANDLER.key,
    model_names=(
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-397B-A17B",
    ),
    default_target_modules=_QWEN3_5_MOE_TARGET_MODULES,
    default_rollout_weights_mode="merged",
    native_vllm_lora_status="wip",
    dependency_floor=DependencyFloor(
        megatron_bridge="e049cc00c24d03e2ae45d2608c7a44e2d2364e3d",
    ),
)

_SPECS_BY_KEY = {
    DEFAULT_DENSE_SPEC.key: DEFAULT_DENSE_SPEC,
    QWEN3_MOE_SPEC.key: QWEN3_MOE_SPEC,
    QWEN3_5_MOE_SPEC.key: QWEN3_5_MOE_SPEC,
}
_SPECS_BY_MODEL = {
    model_name: QWEN3_5_MOE_SPEC for model_name in QWEN3_5_MOE_SPEC.model_names
}
_HANDLERS_BY_KEY: dict[str, ModelSupportHandler] = {
    DEFAULT_DENSE_HANDLER.key: DEFAULT_DENSE_HANDLER,
    QWEN3_MOE_HANDLER.key: QWEN3_MOE_HANDLER,
    QWEN3_5_MOE_HANDLER.key: QWEN3_5_MOE_HANDLER,
}

QWEN3_5_MOE_MODELS = frozenset(QWEN3_5_MOE_SPEC.model_names)


def get_model_support_spec(base_model: str) -> ModelSupportSpec:
    if _is_qwen3_moe_model(base_model):
        return QWEN3_MOE_SPEC
    return _SPECS_BY_MODEL.get(base_model, DEFAULT_DENSE_SPEC)


def get_model_support_handler(base_model: str) -> ModelSupportHandler:
    return get_model_support_handler_for_spec(get_model_support_spec(base_model))


def get_model_support_handler_for_spec(
    spec: ModelSupportSpec,
) -> ModelSupportHandler:
    return _HANDLERS_BY_KEY[spec.handler_key]


def default_target_modules_for_model(base_model: str) -> list[str]:
    return list(get_model_support_spec(base_model).default_target_modules)


def model_requires_merged_rollout(base_model: str) -> bool:
    return get_model_support_spec(base_model).default_rollout_weights_mode == "merged"


def is_model_support_registered(base_model: str) -> bool:
    return base_model in _SPECS_BY_MODEL


def list_model_support_specs() -> list[ModelSupportSpec]:
    return list(_SPECS_BY_KEY.values())


def _is_qwen3_moe_model(base_model: str) -> bool:
    return (
        base_model.startswith("Qwen/Qwen3-")
        and "Qwen3.5" not in base_model
        and "-VL-" not in base_model
        and ("-A3B" in base_model or "-A22B" in base_model)
    )
