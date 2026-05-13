"""Convert PEFT fused MoE LoRA target-parameter adapters for vLLM.

Unsloth with transformers v5 saves MoE expert LoRA as fused 2D tensors:
  mlp.experts.base_layer.lora_A  [num_experts*rank, intermediate*2]
  mlp.experts.base_layer.lora_B  [hidden, num_experts*rank]
  mlp.experts.lora_A             [num_experts*rank, hidden]
  mlp.experts.lora_B             [intermediate, num_experts*rank]

vLLM's 3D MoE LoRA path expects the same fused keys with standard LoRA
orientation, so conversion swaps/transposes each A/B pair and keeps target
modules at "experts".
"""

import json
import os
import re

import safetensors.torch
import torch


def _has_peft_fused_moe_lora(
    tensors: dict[str, torch.Tensor],
    adapter_config: dict,
) -> bool:
    """Check if the adapter contains PEFT target-parameter fused MoE tensors."""
    if not adapter_config.get("target_parameters"):
        return False
    return any(
        re.search(r"mlp\.experts\.(base_layer\.)?lora_[AB]\.weight$", key)
        for key in tensors
    )


def convert_fused_moe_lora(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert PEFT fused MoE LoRA tensors to vLLM's fused experts layout."""
    new_tensors: dict[str, torch.Tensor] = {}

    for key, tensor in tensors.items():
        m = re.match(
            r"(.*\.mlp\.experts)\.(base_layer\.lora_(A|B)|lora_(A|B))\.weight$",
            key,
        )
        if not m:
            new_tensors[key] = tensor
            continue

        prefix = m.group(1)
        if m.group(2) == "base_layer.lora_A":
            new_tensors[f"{prefix}.base_layer.lora_B.weight"] = tensor.T.contiguous()
        elif m.group(2) == "base_layer.lora_B":
            new_tensors[f"{prefix}.base_layer.lora_A.weight"] = tensor.T.contiguous()
        elif m.group(2) == "lora_A":
            new_tensors[f"{prefix}.lora_B.weight"] = tensor.T.contiguous()
        elif m.group(2) == "lora_B":
            new_tensors[f"{prefix}.lora_A.weight"] = tensor.T.contiguous()
        else:
            raise AssertionError(f"Unhandled MoE LoRA tensor key: {key}")

    return new_tensors


def convert_checkpoint_if_needed(checkpoint_dir: str) -> None:
    """Convert a checkpoint's MoE LoRA adapter to per-expert format if needed.

    This is a no-op for non-MoE adapters.
    """
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    config_path = os.path.join(checkpoint_dir, "adapter_config.json")

    if not os.path.exists(adapter_path) or not os.path.exists(config_path):
        return

    tensors = safetensors.torch.load_file(adapter_path)
    with open(config_path) as f:
        adapter_config = json.load(f)

    if not _has_peft_fused_moe_lora(tensors, adapter_config):
        return

    new_tensors = convert_fused_moe_lora(tensors)

    # Overwrite the adapter with the converted tensors
    safetensors.torch.save_file(new_tensors, adapter_path)

    # Update adapter_config.json target_modules
    adapter_config["target_modules"] = [
        m
        for m in adapter_config.get("target_modules", [])
        if m not in {"experts", "gate_proj", "up_proj", "down_proj"}
    ] + ["experts"]
    adapter_config.pop("target_parameters", None)

    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
