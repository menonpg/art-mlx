import torch

from art.utils.convert_megatron_moe_lora import (
    convert_megatron_moe_lora_to_peft_target_parameter,
    convert_peft_target_parameter_moe_lora_to_megatron,
)


def _peft_tensors(prefix: str) -> dict[str, torch.Tensor]:
    num_experts = 2
    rank = 2
    hidden_size = 3
    intermediate_size = 4
    return {
        f"{prefix}.base_layer.lora_A.weight": torch.arange(
            num_experts * rank * hidden_size
        ).reshape(num_experts * rank, hidden_size),
        f"{prefix}.base_layer.lora_B.weight": torch.arange(
            100,
            100 + 2 * intermediate_size * num_experts * rank,
        ).reshape(2 * intermediate_size, num_experts * rank),
        f"{prefix}.lora_A.weight": torch.arange(
            200,
            200 + num_experts * rank * intermediate_size,
        ).reshape(num_experts * rank, intermediate_size),
        f"{prefix}.lora_B.weight": torch.arange(
            300,
            300 + hidden_size * num_experts * rank,
        ).reshape(hidden_size, num_experts * rank),
    }


def test_convert_peft_target_parameter_moe_lora_to_megatron_round_trips() -> None:
    prefix = "base_model.model.model.layers.0.mlp.experts"
    original = _peft_tensors(prefix)
    original["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"] = (
        torch.ones(1)
    )

    megatron = convert_peft_target_parameter_moe_lora_to_megatron(
        original,
        rank=2,
    )
    converted = convert_megatron_moe_lora_to_peft_target_parameter(megatron)

    assert set(converted) == set(original)
    for key, tensor in original.items():
        assert torch.equal(converted[key], tensor)


def test_convert_peft_target_parameter_moe_lora_uses_rank_major_b_layout() -> None:
    prefix = "base_model.model.model.layers.0.mlp.experts"
    original = _peft_tensors(prefix)

    megatron = convert_peft_target_parameter_moe_lora_to_megatron(
        original,
        rank=2,
    )

    gate_up_b = original[f"{prefix}.base_layer.lora_B.weight"]
    down_b = original[f"{prefix}.lora_B.weight"]

    assert torch.equal(
        megatron[f"{prefix}.1.gate_proj.lora_B.weight"],
        gate_up_b.reshape(8, 2, 2).permute(2, 0, 1)[1, :4],
    )
    assert torch.equal(
        megatron[f"{prefix}.1.up_proj.lora_B.weight"],
        gate_up_b.reshape(8, 2, 2).permute(2, 0, 1)[1, 4:],
    )
    assert torch.equal(
        megatron[f"{prefix}.1.down_proj.lora_B.weight"],
        down_b.reshape(3, 2, 2).permute(2, 0, 1)[1],
    )
