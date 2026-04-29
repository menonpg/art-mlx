"""Convert PEFT target-parameter MoE LoRA to ART Megatron per-expert LoRA.

PEFT saves LoRA for fused MoE expert parameters as tensors under:
  mlp.experts.base_layer.lora_*  (gate_up_proj)
  mlp.experts.lora_*             (down_proj)

ART's Megatron LoRA loader currently consumes per-expert module keys:
  mlp.experts.0.gate_proj.lora_A.weight
  mlp.experts.0.up_proj.lora_A.weight
  mlp.experts.0.down_proj.lora_A.weight

TODO: Teach Megatron's LoRA loader to accept PEFT fused target_parameters
directly, then delete this converter entirely.
"""

import json
import os
import re
from typing import Any

import safetensors.torch
import torch

_FUSED_EXPERT_PATTERN = re.compile(
    r"(?P<prefix>.*\.mlp\.experts)\."
    r"(?P<base_layer>base_layer\.)?"
    r"(?P<lora>lora_[AB])\.weight$"
)


def _has_peft_target_parameter_moe_lora(tensors: dict[str, torch.Tensor]) -> bool:
    """Check whether the adapter contains PEFT fused target-parameter MoE LoRA."""
    return any(_FUSED_EXPERT_PATTERN.search(key) for key in tensors)


def _rank_from_adapter_config(adapter_config: dict[str, Any]) -> int:
    rank = adapter_config.get("r", adapter_config.get("lora_rank", 8))
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"Invalid LoRA rank in adapter_config: {rank!r}")
    return rank


def _reshape_expert_a(
    key: str,
    tensor: torch.Tensor,
    *,
    rank: int,
) -> tuple[int, torch.Tensor]:
    if tensor.ndim != 2:
        raise ValueError(f"{key}: expected 2D lora_A tensor, got shape={tensor.shape}")
    num_experts_times_rank, in_features = tensor.shape
    if num_experts_times_rank % rank != 0:
        raise ValueError(
            f"{key}: first dimension {num_experts_times_rank} is not divisible "
            f"by LoRA rank {rank}"
        )
    num_experts = num_experts_times_rank // rank
    return num_experts, tensor.reshape(num_experts, rank, in_features)


def _reshape_expert_b(
    key: str,
    tensor: torch.Tensor,
    *,
    num_experts: int,
    rank: int,
) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"{key}: expected 2D lora_B tensor, got shape={tensor.shape}")
    out_features, num_experts_times_rank = tensor.shape
    expected = num_experts * rank
    if num_experts_times_rank != expected:
        raise ValueError(
            f"{key}: second dimension {num_experts_times_rank} does not match "
            f"num_experts * rank ({expected})"
        )
    return tensor.reshape(out_features, num_experts, rank).permute(1, 0, 2)


def _convert_gate_up_lora(
    *,
    prefix: str,
    lora_a_key: str,
    lora_a: torch.Tensor,
    lora_b_key: str,
    lora_b: torch.Tensor,
    rank: int,
) -> dict[str, torch.Tensor]:
    num_experts, per_expert_a = _reshape_expert_a(lora_a_key, lora_a, rank=rank)
    per_expert_b = _reshape_expert_b(
        lora_b_key,
        lora_b,
        num_experts=num_experts,
        rank=rank,
    )
    if per_expert_b.shape[1] % 2 != 0:
        raise ValueError(
            f"{lora_b_key}: gate_up output dimension must be even, got "
            f"{per_expert_b.shape[1]}"
        )
    gate_b, up_b = per_expert_b.chunk(2, dim=1)

    converted: dict[str, torch.Tensor] = {}
    for expert_idx in range(num_experts):
        expert_a = per_expert_a[expert_idx].contiguous()
        converted[f"{prefix}.{expert_idx}.gate_proj.lora_A.weight"] = expert_a
        converted[f"{prefix}.{expert_idx}.up_proj.lora_A.weight"] = expert_a.clone()
        converted[f"{prefix}.{expert_idx}.gate_proj.lora_B.weight"] = gate_b[
            expert_idx
        ].contiguous()
        converted[f"{prefix}.{expert_idx}.up_proj.lora_B.weight"] = up_b[
            expert_idx
        ].contiguous()
    return converted


def _convert_down_lora(
    *,
    prefix: str,
    lora_a_key: str,
    lora_a: torch.Tensor,
    lora_b_key: str,
    lora_b: torch.Tensor,
    rank: int,
) -> dict[str, torch.Tensor]:
    num_experts, per_expert_a = _reshape_expert_a(lora_a_key, lora_a, rank=rank)
    per_expert_b = _reshape_expert_b(
        lora_b_key,
        lora_b,
        num_experts=num_experts,
        rank=rank,
    )

    converted: dict[str, torch.Tensor] = {}
    for expert_idx in range(num_experts):
        converted[f"{prefix}.{expert_idx}.down_proj.lora_A.weight"] = per_expert_a[
            expert_idx
        ].contiguous()
        converted[f"{prefix}.{expert_idx}.down_proj.lora_B.weight"] = per_expert_b[
            expert_idx
        ].contiguous()
    return converted


def convert_peft_target_parameter_moe_lora_to_megatron(
    tensors: dict[str, torch.Tensor],
    *,
    rank: int,
) -> dict[str, torch.Tensor]:
    """Convert PEFT fused MoE target-parameter LoRA tensors to Megatron keys."""
    converted: dict[str, torch.Tensor] = {}
    fused_by_prefix: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}

    for key, tensor in tensors.items():
        match = _FUSED_EXPERT_PATTERN.match(key)
        if match is None:
            converted[key] = tensor
            continue

        prefix = match.group("prefix")
        lora_name = match.group("lora")
        is_gate_up = match.group("base_layer") is not None
        group = "gate_up" if is_gate_up else "down"
        fused_by_prefix.setdefault(prefix, {})[f"{group}_{lora_name}"] = (key, tensor)

    for prefix, fused_tensors in fused_by_prefix.items():
        gate_up_a = fused_tensors.get("gate_up_lora_A")
        gate_up_b = fused_tensors.get("gate_up_lora_B")
        if gate_up_a is not None or gate_up_b is not None:
            if gate_up_a is None or gate_up_b is None:
                raise ValueError(f"{prefix}: missing gate_up lora_A or lora_B tensor")
            converted.update(
                _convert_gate_up_lora(
                    prefix=prefix,
                    lora_a_key=gate_up_a[0],
                    lora_a=gate_up_a[1],
                    lora_b_key=gate_up_b[0],
                    lora_b=gate_up_b[1],
                    rank=rank,
                )
            )

        down_a = fused_tensors.get("down_lora_A")
        down_b = fused_tensors.get("down_lora_B")
        if down_a is not None or down_b is not None:
            if down_a is None or down_b is None:
                raise ValueError(f"{prefix}: missing down lora_A or lora_B tensor")
            converted.update(
                _convert_down_lora(
                    prefix=prefix,
                    lora_a_key=down_a[0],
                    lora_a=down_a[1],
                    lora_b_key=down_b[0],
                    lora_b=down_b[1],
                    rank=rank,
                )
            )

    return converted


def convert_checkpoint_to_megatron_moe_lora_if_needed(checkpoint_dir: str) -> None:
    """Convert a PEFT MoE target-parameter adapter to Megatron format if needed."""
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    config_path = os.path.join(checkpoint_dir, "adapter_config.json")

    if not os.path.exists(adapter_path) or not os.path.exists(config_path):
        return

    tensors = safetensors.torch.load_file(adapter_path)
    if not _has_peft_target_parameter_moe_lora(tensors):
        return

    with open(config_path) as f:
        adapter_config = json.load(f)

    rank = _rank_from_adapter_config(adapter_config)
    converted = convert_peft_target_parameter_moe_lora_to_megatron(
        tensors,
        rank=rank,
    )

    safetensors.torch.save_file(converted, adapter_path)

    adapter_config["target_modules"] = [
        module
        for module in adapter_config.get("target_modules", [])
        if "experts" not in module
    ] + ["gate_proj", "up_proj", "down_proj"]
    adapter_config.pop("target_parameters", None)

    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
