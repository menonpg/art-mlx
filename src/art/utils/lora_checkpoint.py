import importlib
import json
from pathlib import Path
import re
from typing import Any

import torch

_TEXT_LAYER_PREFIX = "base_model.model.model.layers."
_LANGUAGE_MODEL_LAYER_PREFIX = "base_model.model.model.language_model.layers."

safetensors = importlib.import_module("safetensors")
safetensors_torch = importlib.import_module("safetensors.torch")
safe_open = safetensors.safe_open
save_file = safetensors_torch.save_file

_MOE_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\.(?P<module>gate_proj|up_proj|down_proj)\.(?P<lora>lora_[AB])\.weight$"
)


def uses_qwen_language_model_prefix(base_model: str | None) -> bool:
    return isinstance(base_model, str) and base_model.startswith(
        ("Qwen/Qwen3.5", "Qwen/Qwen3.6")
    )


def load_adapter_config(checkpoint_dir: str) -> dict[str, Any]:
    config_path = Path(checkpoint_dir) / "adapter_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def resolve_adapter_base_model(
    checkpoint_dir: str,
    *,
    base_model: str | None = None,
) -> str | None:
    if base_model is not None:
        return base_model
    value = load_adapter_config(checkpoint_dir).get("base_model_name_or_path")
    return value if isinstance(value, str) and value else None


def to_runtime_adapter_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    base_model: str | None,
) -> dict[str, torch.Tensor]:
    if not uses_qwen_language_model_prefix(base_model):
        return tensors
    return {
        (
            key.replace(_TEXT_LAYER_PREFIX, _LANGUAGE_MODEL_LAYER_PREFIX, 1)
            if key.startswith(_TEXT_LAYER_PREFIX)
            else key
        ): tensor
        for key, tensor in tensors.items()
    }


def to_megatron_adapter_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    base_model: str | None,
) -> dict[str, torch.Tensor]:
    if not uses_qwen_language_model_prefix(base_model):
        return tensors
    return {
        (
            key.replace(_LANGUAGE_MODEL_LAYER_PREFIX, _TEXT_LAYER_PREFIX, 1)
            if key.startswith(_LANGUAGE_MODEL_LAYER_PREFIX)
            else key
        ): tensor
        for key, tensor in tensors.items()
    }


def normalize_runtime_lora_checkpoint(
    checkpoint_dir: str,
    *,
    base_model: str | None = None,
) -> None:
    adapter_model_path = Path(checkpoint_dir) / "adapter_model.safetensors"
    if not adapter_model_path.exists():
        return
    resolved_base_model = resolve_adapter_base_model(
        checkpoint_dir,
        base_model=base_model,
    )
    if not uses_qwen_language_model_prefix(resolved_base_model):
        return
    with safe_open(adapter_model_path, framework="pt") as file:
        tensors = {key: file.get_tensor(key) for key in file.keys()}
    normalized = to_runtime_adapter_tensors(
        tensors,
        base_model=resolved_base_model,
    )
    if set(normalized) == set(tensors) and all(
        normalized[key] is tensor for key, tensor in tensors.items()
    ):
        return
    save_file(normalized, adapter_model_path)


def _build_qwen_moe_native_vllm_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | None:
    grouped: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]] = {}
    for key, tensor in tensors.items():
        match = _MOE_EXPERT_KEY_RE.match(key)
        if match is None:
            continue
        prefix = match.group("prefix")
        expert = int(match.group("expert"))
        module = match.group("module")
        lora_name = match.group("lora")
        grouped.setdefault(prefix, {}).setdefault(expert, {}).setdefault(module, {})[
            lora_name
        ] = tensor
    if not grouped:
        return None

    original_rank = int(adapter_config.get("r", 0) or 0)
    if original_rank <= 0:
        raise RuntimeError("LoRA adapter config is missing a positive rank")
    fused_rank = original_rank * 2
    transformed: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()

    def _pad_a(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] == fused_rank:
            return tensor
        padded = tensor.new_zeros((fused_rank, tensor.shape[1]))
        padded[: tensor.shape[0], :] = tensor
        return padded

    def _pad_b(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] == fused_rank:
            return tensor
        padded = tensor.new_zeros((tensor.shape[0], fused_rank))
        padded[:, : tensor.shape[1]] = tensor
        return padded

    for prefix, experts in grouped.items():
        fused_a_blocks: list[torch.Tensor] = []
        fused_b_blocks: list[torch.Tensor] = []
        down_a_blocks: list[torch.Tensor] = []
        down_b_blocks: list[torch.Tensor] = []
        for expert in sorted(experts):
            modules = experts[expert]
            try:
                gate_a = modules["gate_proj"]["lora_A"]
                gate_b = modules["gate_proj"]["lora_B"]
                up_a = modules["up_proj"]["lora_A"]
                up_b = modules["up_proj"]["lora_B"]
                down_a = modules["down_proj"]["lora_A"]
                down_b = modules["down_proj"]["lora_B"]
            except KeyError as exc:
                raise RuntimeError(
                    f"Incomplete MoE LoRA expert block for {prefix}. expert={expert}"
                ) from exc
            fused_a_blocks.append(torch.cat((gate_a, up_a), dim=0).contiguous())
            gate_rank = int(gate_a.shape[0])
            up_rank = int(up_a.shape[0])
            gate_up_b = gate_b.new_zeros(
                (gate_b.shape[0] + up_b.shape[0], gate_rank + up_rank)
            )
            gate_up_b[: gate_b.shape[0], :gate_rank] = gate_b
            gate_up_b[gate_b.shape[0] :, gate_rank:] = up_b
            fused_b_blocks.append(gate_up_b.contiguous())
            down_a_blocks.append(_pad_a(down_a).contiguous())
            down_b_blocks.append(_pad_b(down_b).contiguous())
            used_keys.update(
                {
                    f"{prefix}.{expert}.gate_proj.lora_A.weight",
                    f"{prefix}.{expert}.gate_proj.lora_B.weight",
                    f"{prefix}.{expert}.up_proj.lora_A.weight",
                    f"{prefix}.{expert}.up_proj.lora_B.weight",
                    f"{prefix}.{expert}.down_proj.lora_A.weight",
                    f"{prefix}.{expert}.down_proj.lora_B.weight",
                }
            )
        transformed[f"{prefix}.base_layer.lora_A.weight"] = torch.cat(
            fused_a_blocks,
            dim=0,
        ).contiguous()
        transformed[f"{prefix}.base_layer.lora_B.weight"] = torch.cat(
            fused_b_blocks,
            dim=1,
        ).contiguous()
        transformed[f"{prefix}.lora_A.weight"] = torch.cat(
            down_a_blocks,
            dim=0,
        ).contiguous()
        transformed[f"{prefix}.lora_B.weight"] = torch.cat(
            down_b_blocks,
            dim=1,
        ).contiguous()

    if not transformed:
        return None

    for key, tensor in tensors.items():
        if key in used_keys:
            continue
        match = re.search(r"\.lora_A\.weight$|\.lora_B\.weight$", key)
        if match is None:
            transformed[key] = tensor
            continue
        if key.endswith(".lora_A.weight"):
            transformed[key] = _pad_a(tensor).contiguous()
        else:
            transformed[key] = _pad_b(tensor).contiguous()

    updated_config = dict(adapter_config)
    updated_config["r"] = fused_rank
    if "lora_alpha" in updated_config and updated_config["lora_alpha"] is not None:
        updated_config["lora_alpha"] = int(updated_config["lora_alpha"]) * 2
    target_modules = list(updated_config.get("target_modules") or [])
    if "experts" not in target_modules:
        target_modules.append("experts")
    updated_config["target_modules"] = target_modules
    return transformed, updated_config


def prepare_runtime_lora_checkpoint(
    checkpoint_dir: str,
    *,
    runtime_checkpoint_dir: str,
    base_model: str | None = None,
) -> str:
    adapter_model_path = Path(checkpoint_dir) / "adapter_model.safetensors"
    if not adapter_model_path.exists():
        return checkpoint_dir
    resolved_base_model = resolve_adapter_base_model(
        checkpoint_dir,
        base_model=base_model,
    )
    with safe_open(adapter_model_path, framework="pt") as file:
        tensors = {key: file.get_tensor(key) for key in file.keys()}
    runtime_tensors = to_runtime_adapter_tensors(
        tensors,
        base_model=resolved_base_model,
    )
    runtime_config = load_adapter_config(checkpoint_dir)
    runtime_config.setdefault("base_model_name_or_path", resolved_base_model)
    moe_transformed = _build_qwen_moe_native_vllm_tensors(
        runtime_tensors,
        adapter_config=runtime_config,
    )
    if moe_transformed is not None:
        runtime_tensors, runtime_config = moe_transformed
    runtime_dir = Path(runtime_checkpoint_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    save_file(runtime_tensors, runtime_dir / "adapter_model.safetensors")
    with (runtime_dir / "adapter_config.json").open("w", encoding="utf-8") as handle:
        json.dump(runtime_config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(runtime_dir)
