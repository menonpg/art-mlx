"""
ART MLX Backend — GRPO training on Apple Silicon.

This module provides an MLX-native backend for ART, enabling agent 
reinforcement training on Apple Silicon Macs without CUDA dependencies.

Components:
- MLXBackend: Drop-in replacement for LocalBackend using MLX
- MLX-LM integration for inference
- Native MLX LoRA training with GRPO
"""

from .backend import MLXBackend

__all__ = ["MLXBackend"]
