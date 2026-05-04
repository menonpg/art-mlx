import pytest

from art.megatron.model_support import (
    QWEN3_5_DENSE_MODELS,
    QWEN3_5_MODELS,
    QWEN3_5_MOE_MODELS,
    QWEN3_MOE_MODELS,
    UnsupportedModelArchitectureError,
    default_target_modules_for_model,
    get_model_support_handler,
    get_model_support_spec,
    list_model_support_specs,
    model_requires_merged_rollout,
    model_uses_expert_parallel,
    native_vllm_lora_status_for_model,
)


def test_unsupported_model_support_requires_explicit_opt_in():
    with pytest.raises(UnsupportedModelArchitectureError):
        get_model_support_spec("test-model")

    spec = get_model_support_spec("test-model", allow_unvalidated_arch=True)
    assert spec.key == "default_dense"
    assert spec.handler_key == "default_dense"
    assert list(spec.default_target_modules) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_qwen3_5_model_support_spec():
    spec = get_model_support_spec("Qwen/Qwen3.5-35B-A3B")
    assert spec.key == "qwen3_5_moe"
    assert spec.handler_key == "qwen3_5_moe"
    assert spec.default_rollout_weights_mode == "lora"
    assert native_vllm_lora_status_for_model("Qwen/Qwen3.5-35B-A3B") == "validated"
    assert spec.dependency_floor.megatron_bridge == (
        "e049cc00c24d03e2ae45d2608c7a44e2d2364e3d"
    )


def test_qwen3_5_dense_model_support_spec():
    with pytest.raises(UnsupportedModelArchitectureError):
        get_model_support_spec("Qwen/Qwen3.5-4B")

    spec = get_model_support_spec("Qwen/Qwen3.5-4B", allow_unvalidated_arch=True)
    assert spec.key == "qwen3_5_dense"
    assert spec.handler_key == "qwen3_5_dense"
    assert spec.default_rollout_weights_mode == "lora"
    assert (
        native_vllm_lora_status_for_model(
            "Qwen/Qwen3.5-4B",
            allow_unvalidated_arch=True,
        )
        == "validated"
    )
    assert spec.dependency_floor.megatron_bridge == (
        "e049cc00c24d03e2ae45d2608c7a44e2d2364e3d"
    )


def test_qwen3_5_registry_exports():
    assert QWEN3_5_DENSE_MODELS == {
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.6-27B",
    }
    assert QWEN3_5_MOE_MODELS == {
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.6-35B-A3B",
    }
    assert QWEN3_5_MODELS == QWEN3_5_MOE_MODELS
    assert default_target_modules_for_model(
        "Qwen/Qwen3.6-27B",
        allow_unvalidated_arch=True,
    ) == [
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
    ]
    assert model_requires_merged_rollout("Qwen/Qwen3.6-35B-A3B") is False
    assert model_uses_expert_parallel("Qwen/Qwen3.6-35B-A3B") is True
    assert (
        model_uses_expert_parallel(
            "Qwen/Qwen3.6-27B",
            allow_unvalidated_arch=True,
        )
        is False
    )
    assert (
        get_model_support_handler(
            "Qwen/Qwen3.6-27B",
            allow_unvalidated_arch=True,
        ).key
        == "qwen3_5_dense"
    )
    assert get_model_support_handler("Qwen/Qwen3.6-35B-A3B").key == "qwen3_5_moe"


def test_qwen3_moe_model_support_spec():
    assert QWEN3_MOE_MODELS == {
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-30B-A3B-Base",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
    }
    spec = get_model_support_spec("Qwen/Qwen3-30B-A3B-Instruct-2507")
    assert spec.key == "qwen3_moe"
    assert spec.handler_key == "qwen3_moe"
    assert spec.default_rollout_weights_mode == "lora"
    assert (
        native_vllm_lora_status_for_model("Qwen/Qwen3-30B-A3B-Instruct-2507")
        == "validated"
    )
    assert get_model_support_handler("Qwen/Qwen3-30B-A3B-Instruct-2507").key == (
        "qwen3_moe"
    )


def test_qwen3_dense_uses_default_dense_only_in_unsupported_probe_mode():
    with pytest.raises(UnsupportedModelArchitectureError):
        get_model_support_spec("Qwen/Qwen3-4B-Instruct-2507")

    spec = get_model_support_spec(
        "Qwen/Qwen3-4B-Instruct-2507",
        allow_unvalidated_arch=True,
    )
    assert spec.key == "qwen3_dense"
    assert spec.handler_key == "qwen3_dense"
    assert (
        model_uses_expert_parallel(
            "Qwen/Qwen3-4B-Instruct-2507",
            allow_unvalidated_arch=True,
        )
        is False
    )


def test_model_support_specs_list_is_stable():
    specs = list_model_support_specs()
    assert [spec.key for spec in specs] == [
        "qwen3_moe",
        "qwen3_5_moe",
    ]
