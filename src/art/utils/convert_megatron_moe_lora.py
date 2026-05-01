"""Convert between PEFT target-parameter and ART Megatron MoE LoRA tensors.

PEFT saves LoRA for fused MoE expert parameters as tensors under:
  mlp.experts.base_layer.lora_*  (gate_up_proj)
  mlp.experts.lora_*             (down_proj)

ART's Megatron LoRA loader currently consumes per-expert module keys:
  mlp.experts.0.gate_proj.lora_A.weight
  mlp.experts.0.up_proj.lora_A.weight
  mlp.experts.0.down_proj.lora_A.weight

Checkpoints stay in the PEFT fused format on disk because vLLM expects that
layout. Megatron converts to per-expert keys in memory while loading, then the
Megatron shard merger converts trained tensors back before writing the final
adapter_model.safetensors.

TODO: Teach Megatron's LoRA loader to accept PEFT fused target_parameters
directly, then delete this converter entirely.
"""

import re

import torch

_FUSED_EXPERT_PATTERN = re.compile(
    r"(?P<prefix>.*\.mlp\.experts)\."
    r"(?P<base_layer>base_layer\.)?"
    r"(?P<lora>lora_[AB])\.weight$"
)
_MEGATRON_EXPERT_PATTERN = re.compile(
    r"(?P<prefix>.*\.mlp\.experts)\."
    r"(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<lora>lora_[AB])\.weight$"
)
_TEXT_LAYER_PREFIX = "base_model.model.model.layers."
_LANGUAGE_MODEL_LAYER_PREFIX = "base_model.model.model.language_model.layers."


def uses_qwen_language_model_prefix(base_model: str) -> bool:
    return base_model.startswith(("Qwen/Qwen3.5", "Qwen/Qwen3.6"))


def add_language_model_prefix_for_vllm(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rewrite Megatron text-model LoRA keys to vLLM's Qwen3.5/3.6 wrapper path."""
    return {
        key.replace(_TEXT_LAYER_PREFIX, _LANGUAGE_MODEL_LAYER_PREFIX, 1): tensor
        for key, tensor in tensors.items()
    }


def strip_language_model_prefix_for_megatron(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rewrite vLLM/HF Qwen3.5/3.6 wrapper LoRA keys to Megatron text-model keys."""
    return {
        key.replace(_LANGUAGE_MODEL_LAYER_PREFIX, _TEXT_LAYER_PREFIX, 1): tensor
        for key, tensor in tensors.items()
    }


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
    return tensor.reshape(out_features, rank, num_experts).permute(2, 0, 1)


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


def _stack_expert_tensors(
    prefix: str,
    projection_tensors: dict[int, torch.Tensor],
    *,
    projection: str,
    lora: str,
) -> torch.Tensor:
    expert_ids = sorted(projection_tensors)
    if expert_ids != list(range(len(expert_ids))):
        raise ValueError(
            f"{prefix}.{projection}.{lora}: expected contiguous expert ids, got "
            f"{expert_ids}"
        )
    tensors = [projection_tensors[expert_id] for expert_id in expert_ids]
    first_shape = tensors[0].shape
    for expert_id, tensor in zip(expert_ids, tensors):
        if tensor.shape != first_shape:
            raise ValueError(
                f"{prefix}.{expert_id}.{projection}.{lora}: expected shape "
                f"{first_shape}, got {tensor.shape}"
            )
    return torch.stack(tensors)


def _flatten_expert_a(per_expert_a: torch.Tensor) -> torch.Tensor:
    num_experts, rank, in_features = per_expert_a.shape
    return per_expert_a.reshape(num_experts * rank, in_features).contiguous()


def _flatten_expert_b(per_expert_b: torch.Tensor) -> torch.Tensor:
    num_experts, out_features, rank = per_expert_b.shape
    return (
        per_expert_b.permute(1, 2, 0)
        .reshape(out_features, num_experts * rank)
        .contiguous()
    )


def convert_megatron_moe_lora_to_peft_target_parameter(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert ART Megatron per-expert MoE LoRA tensors to PEFT fused keys."""
    converted: dict[str, torch.Tensor] = {}
    grouped: dict[
        str,
        dict[str, dict[str, dict[int, torch.Tensor]]],
    ] = {}

    for key, tensor in tensors.items():
        match = _MEGATRON_EXPERT_PATTERN.match(key)
        if match is None:
            converted[key] = tensor
            continue
        prefix = match.group("prefix")
        projection = match.group("projection")
        lora = match.group("lora")
        expert_id = int(match.group("expert"))
        grouped.setdefault(prefix, {}).setdefault(projection, {}).setdefault(lora, {})[
            expert_id
        ] = tensor

    for prefix, projections in grouped.items():
        required = {
            "gate_proj": {"lora_A", "lora_B"},
            "up_proj": {"lora_A", "lora_B"},
            "down_proj": {"lora_A", "lora_B"},
        }
        for projection, loras in required.items():
            missing_loras = loras - set(projections.get(projection, {}))
            if missing_loras:
                raise ValueError(
                    f"{prefix}.{projection}: missing {sorted(missing_loras)}"
                )

        gate_a = _stack_expert_tensors(
            prefix,
            projections["gate_proj"]["lora_A"],
            projection="gate_proj",
            lora="lora_A",
        )
        up_a = _stack_expert_tensors(
            prefix,
            projections["up_proj"]["lora_A"],
            projection="up_proj",
            lora="lora_A",
        )
        if not torch.equal(gate_a, up_a):
            raise ValueError(
                f"{prefix}: cannot convert Megatron gate/up LoRA to PEFT "
                "target_parameters because gate_proj.lora_A and up_proj.lora_A differ"
            )
        gate_b = _stack_expert_tensors(
            prefix,
            projections["gate_proj"]["lora_B"],
            projection="gate_proj",
            lora="lora_B",
        )
        up_b = _stack_expert_tensors(
            prefix,
            projections["up_proj"]["lora_B"],
            projection="up_proj",
            lora="lora_B",
        )
        down_a = _stack_expert_tensors(
            prefix,
            projections["down_proj"]["lora_A"],
            projection="down_proj",
            lora="lora_A",
        )
        down_b = _stack_expert_tensors(
            prefix,
            projections["down_proj"]["lora_B"],
            projection="down_proj",
            lora="lora_B",
        )

        converted[f"{prefix}.base_layer.lora_A.weight"] = _flatten_expert_a(gate_a)
        converted[f"{prefix}.base_layer.lora_B.weight"] = _flatten_expert_b(
            torch.cat([gate_b, up_b], dim=1)
        )
        converted[f"{prefix}.lora_A.weight"] = _flatten_expert_a(down_a)
        converted[f"{prefix}.lora_B.weight"] = _flatten_expert_b(down_b)

    return converted
