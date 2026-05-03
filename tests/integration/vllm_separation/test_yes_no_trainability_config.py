import pytest

from art.megatron.model_support import UnsupportedModelArchitectureError

from .yes_no_trainability import (
    _build_internal_config,
    _default_variant_name,
    _TrainabilityVariant,
    _variant_init_args,
    _variant_max_steps,
    _variant_packed_sequence_length,
    _variant_rollouts_per_prompt,
    _variant_train_kwargs,
)


def test_megatron_variants_keep_short_packed_sequence_default(monkeypatch) -> None:
    monkeypatch.delenv("ART_MODEL_SUPPORT_YES_NO_PACKED_SEQUENCE_LENGTH", raising=False)
    variant = _TrainabilityVariant(
        name="megatron_shared",
        backend_name="megatron",
        placement_mode="shared",
        trainer_gpu_ids=[0, 1],
        inference_gpu_ids=[0, 1],
    )

    assert _variant_packed_sequence_length(variant) == 1024
    assert _variant_train_kwargs(variant) == {"packed_sequence_length": 1024}
    config = _build_internal_config(
        variant, base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"
    )
    assert config["init_args"]["max_seq_length"] == 1024
    assert config["rollout_weights_mode"] == "lora"
    assert (
        _default_variant_name("Qwen/Qwen3-30B-A3B-Instruct-2507") == "megatron_shared"
    )
    assert _variant_rollouts_per_prompt(variant) == 4
    assert _variant_max_steps(variant) == 4


def test_unsloth_variant_uses_chunk_aligned_training_length(monkeypatch) -> None:
    monkeypatch.delenv("ART_MODEL_SUPPORT_YES_NO_PACKED_SEQUENCE_LENGTH", raising=False)
    variant = _TrainabilityVariant(
        name="unsloth_dedicated",
        backend_name="local",
        placement_mode="dedicated",
        trainer_gpu_ids=[0],
        inference_gpu_ids=[1],
    )

    assert _variant_packed_sequence_length(variant) == 1024
    assert _variant_train_kwargs(variant) == {"packed_sequence_length": 1024}
    assert _variant_init_args(variant) == {"max_seq_length": 1024}
    assert _build_internal_config(
        variant, base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"
    )["init_args"] == {"max_seq_length": 1024}
    assert _variant_rollouts_per_prompt(variant) == 8
    assert _variant_max_steps(variant) == 12


def test_qwen3_5_defaults_to_shared_lora_rollout() -> None:
    variant = _TrainabilityVariant(
        name="megatron_shared",
        backend_name="megatron",
        placement_mode="shared",
        trainer_gpu_ids=[0, 1],
        inference_gpu_ids=[0, 1],
    )

    config = _build_internal_config(variant, base_model="Qwen/Qwen3.5-35B-A3B")

    assert _default_variant_name("Qwen/Qwen3.5-35B-A3B") == "megatron_shared"
    assert config["rollout_weights_mode"] == "lora"
    assert "trainer_gpu_ids" not in config
    assert "inference_gpu_ids" not in config


def test_unvalidated_dense_model_is_not_default_megatron_trainability_model(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ART_MODEL_SUPPORT_SHARED_GPU_IDS", "0,1")
    variant = _TrainabilityVariant(
        name="megatron_shared",
        backend_name="megatron",
        placement_mode="shared",
        trainer_gpu_ids=[0, 1],
        inference_gpu_ids=[0, 1],
    )

    with pytest.raises(UnsupportedModelArchitectureError):
        _build_internal_config(variant, base_model="Qwen/Qwen3.5-4B")

    config = _build_internal_config(
        variant,
        base_model="Qwen/Qwen3.5-4B",
        allow_unsupported_arch=True,
    )
    assert config["rollout_weights_mode"] == "lora"
    assert config["engine_args"]["enable_sleep_mode"] is True
    assert "enable_expert_parallel" not in config["engine_args"]


def test_qwen3_5_moe_shared_variant_enables_expert_parallel(monkeypatch) -> None:
    monkeypatch.setenv("ART_MODEL_SUPPORT_SHARED_GPU_IDS", "0,1")
    variant = _TrainabilityVariant(
        name="megatron_shared",
        backend_name="megatron",
        placement_mode="shared",
        trainer_gpu_ids=[0, 1],
        inference_gpu_ids=[0, 1],
    )

    config = _build_internal_config(variant, base_model="Qwen/Qwen3.5-35B-A3B")

    assert config["rollout_weights_mode"] == "lora"
    assert config["engine_args"]["enable_expert_parallel"] is True
