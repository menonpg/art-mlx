import importlib
import json
from pathlib import Path
from typing import Any

import torch

_TEXT_LAYER_PREFIX = "base_model.model.model.layers."
_LANGUAGE_MODEL_LAYER_PREFIX = "base_model.model.model.language_model.layers."

safetensors = importlib.import_module("safetensors")
safetensors_torch = importlib.import_module("safetensors.torch")
safe_open = safetensors.safe_open
save_file = safetensors_torch.save_file


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
