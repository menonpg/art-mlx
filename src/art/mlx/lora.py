"""
MLX LoRA utilities — Apply and train LoRA adapters on MLX models.

This module provides the LoRA layer implementation and utilities
for applying LoRA to MLX language models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) linear layer for MLX.
    
    Wraps an existing linear layer and adds low-rank adapters:
    output = W @ x + (B @ A) @ x * scale
    
    Where:
    - W is the frozen original weights
    - A is the low-rank down-projection (in_features -> rank)
    - B is the low-rank up-projection (rank -> out_features)
    - scale = alpha / rank
    """
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> 'LoRALinear':
        """Create a LoRA layer from an existing linear layer."""
        out_features, in_features = linear.weight.shape
        lora = cls(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        # Copy original weights (frozen)
        lora.linear = linear
        return lora
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = dropout
        
        # Original linear layer (will be set by from_linear or created)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA adapters - initialized for stable training
        # A: down-projection, initialized with small random values
        self.lora_A = mx.random.normal(shape=(rank, in_features)) * 0.01
        # B: up-projection, initialized to zero (so LoRA starts as identity)
        self.lora_B = mx.zeros((out_features, rank))
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with LoRA."""
        # Original output
        output = self.linear(x)
        
        # LoRA output: (B @ A) @ x * scale
        # Compute in steps for memory efficiency
        lora_out = x @ self.lora_A.T  # [batch, seq, rank]
        lora_out = lora_out @ self.lora_B.T  # [batch, seq, out_features]
        lora_out = lora_out * self.scale
        
        return output + lora_out
    
    def lora_parameters(self) -> dict[str, mx.array]:
        """Return only the LoRA parameters (for training)."""
        return {
            "lora_A": self.lora_A,
            "lora_B": self.lora_B,
        }


def apply_lora_to_model(
    model: Any,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list[str] | None = None,
) -> tuple[Any, list[LoRALinear]]:
    """
    Apply LoRA adapters to a model's linear layers.
    
    Args:
        model: MLX model (from mlx_lm.load)
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        target_modules: List of module name patterns to apply LoRA to.
                       Defaults to ["q_proj", "v_proj"] for attention.
    
    Returns:
        (modified_model, list_of_lora_layers)
    """
    if target_modules is None:
        # Default: apply to attention Q and V projections
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    lora_layers = []
    
    def apply_lora_recursive(module: Any, path: str = "") -> Any:
        """Recursively apply LoRA to matching modules."""
        
        # Check if this module should get LoRA
        module_name = path.split(".")[-1] if path else ""
        
        if isinstance(module, nn.Linear):
            if any(target in module_name for target in target_modules):
                logger.debug(f"Applying LoRA to: {path}")
                lora_layer = LoRALinear.from_linear(module, rank=rank, alpha=alpha)
                lora_layers.append(lora_layer)
                return lora_layer
        
        # Recursively process children
        if hasattr(module, '__dict__'):
            for name, child in list(vars(module).items()):
                if isinstance(child, nn.Module):
                    new_child = apply_lora_recursive(child, f"{path}.{name}" if path else name)
                    setattr(module, name, new_child)
                elif isinstance(child, list):
                    new_list = []
                    for i, item in enumerate(child):
                        if isinstance(item, nn.Module):
                            new_list.append(apply_lora_recursive(item, f"{path}.{name}[{i}]"))
                        else:
                            new_list.append(item)
                    setattr(module, name, new_list)
        
        return module
    
    modified_model = apply_lora_recursive(model)
    logger.info(f"Applied LoRA to {len(lora_layers)} layers (rank={rank}, alpha={alpha})")
    
    return modified_model, lora_layers


def save_lora_weights(
    lora_layers: list[LoRALinear],
    path: str | Path,
    metadata: dict | None = None,
) -> None:
    """Save LoRA weights to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    weights = {}
    for i, layer in enumerate(lora_layers):
        weights[f"layer_{i}_A"] = layer.lora_A
        weights[f"layer_{i}_B"] = layer.lora_B
    
    # Save weights
    mx.savez(str(path.with_suffix('.npz')), **weights)
    
    # Save metadata
    meta = {
        "num_layers": len(lora_layers),
        "rank": lora_layers[0].rank if lora_layers else 0,
        "alpha": lora_layers[0].alpha if lora_layers else 0,
    }
    if metadata:
        meta.update(metadata)
    
    with open(path.with_suffix('.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"Saved LoRA weights to {path}")


def load_lora_weights(
    lora_layers: list[LoRALinear],
    path: str | Path,
) -> dict:
    """Load LoRA weights from a file."""
    path = Path(path)
    
    # Load weights
    weights = dict(mx.load(str(path.with_suffix('.npz'))))
    
    for i, layer in enumerate(lora_layers):
        layer.lora_A = weights[f"layer_{i}_A"]
        layer.lora_B = weights[f"layer_{i}_B"]
    
    # Load metadata
    meta = {}
    meta_path = path.with_suffix('.json')
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    
    logger.info(f"Loaded LoRA weights from {path}")
    return meta


def get_lora_parameters(lora_layers: list[LoRALinear]) -> dict[str, mx.array]:
    """Get all trainable LoRA parameters as a flat dict."""
    params = {}
    for i, layer in enumerate(lora_layers):
        params[f"layer_{i}.lora_A"] = layer.lora_A
        params[f"layer_{i}.lora_B"] = layer.lora_B
    return params


def set_lora_parameters(lora_layers: list[LoRALinear], params: dict[str, mx.array]) -> None:
    """Set LoRA parameters from a flat dict."""
    for i, layer in enumerate(lora_layers):
        if f"layer_{i}.lora_A" in params:
            layer.lora_A = params[f"layer_{i}.lora_A"]
        if f"layer_{i}.lora_B" in params:
            layer.lora_B = params[f"layer_{i}.lora_B"]


def count_lora_parameters(lora_layers: list[LoRALinear]) -> int:
    """Count total trainable parameters in LoRA layers."""
    total = 0
    for layer in lora_layers:
        total += layer.lora_A.size + layer.lora_B.size
    return total
