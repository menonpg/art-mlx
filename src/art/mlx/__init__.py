"""
ART MLX Backend — GRPO training on Apple Silicon.

This module provides an MLX-native backend for ART, enabling agent 
reinforcement training on Apple Silicon Macs without CUDA dependencies.

Components:
- MLXBackend: Drop-in replacement for LocalBackend using MLX
- GRPOTrainer: GRPO algorithm implementation in MLX
- LoRA utilities: Apply and train LoRA adapters
"""

from .backend import MLXBackend
from .grpo import GRPOTrainer, GRPOConfig
from .lora import (
    LoRALinear,
    apply_lora_to_model,
    save_lora_weights,
    load_lora_weights,
    get_lora_parameters,
    count_lora_parameters,
)
from .checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint

__all__ = [
    "MLXBackend",
    "GRPOTrainer",
    "GRPOConfig",
    "LoRALinear",
    "apply_lora_to_model",
    "save_lora_weights",
    "load_lora_weights",
    "get_lora_parameters",
    "count_lora_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
]
