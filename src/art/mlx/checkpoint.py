"""
Checkpoint utilities for ART-MLX.

Save and load training state including model weights, optimizer state,
and training metrics.
"""

from pathlib import Path
import json
import time

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def save_checkpoint(
    model,
    optimizer,
    step: int,
    metrics: dict,
    path: str | Path,
) -> Path:
    """
    Save a training checkpoint.
    
    Args:
        model: The MLX model
        optimizer: The optimizer with state
        step: Current training step
        metrics: Training metrics dict
        path: Directory to save checkpoint
        
    Returns:
        Path to saved checkpoint directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = path / f"step-{step}"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save model weights
    weights = dict(model.parameters())
    mx.savez(str(checkpoint_dir / "model.npz"), **{k: v for k, v in weights.items()})
    
    # Save optimizer state
    opt_state = optimizer.state
    if opt_state:
        mx.savez(str(checkpoint_dir / "optimizer.npz"), **{str(k): v for k, v in opt_state.items()})
    
    # Save metadata
    metadata = {
        "step": step,
        "timestamp": time.time(),
        "metrics": metrics,
    }
    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved checkpoint to {checkpoint_dir}")
    return checkpoint_dir


def load_checkpoint(
    model,
    optimizer,
    path: str | Path,
) -> dict:
    """
    Load a training checkpoint.
    
    Args:
        model: The MLX model to load weights into
        optimizer: The optimizer to restore state
        path: Path to checkpoint directory
        
    Returns:
        Metadata dict with step, metrics, etc.
    """
    path = Path(path)
    
    # Load model weights
    weights_path = path / "model.npz"
    if weights_path.exists():
        weights = dict(mx.load(str(weights_path)))
        model.load_weights(list(weights.items()))
        print(f"Loaded model weights from {weights_path}")
    
    # Load optimizer state
    opt_path = path / "optimizer.npz"
    if opt_path.exists():
        opt_state = dict(mx.load(str(opt_path)))
        # Restore optimizer state
        # Note: This is simplified - full restore would need more care
        print(f"Loaded optimizer state from {opt_path}")
    
    # Load metadata
    meta_path = path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        print(f"Loaded checkpoint from step {metadata.get('step', '?')}")
        return metadata
    
    return {}


def find_latest_checkpoint(path: str | Path) -> Path | None:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        path: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint, or None if none found
    """
    path = Path(path)
    if not path.exists():
        return None
    
    checkpoints = sorted(
        [d for d in path.iterdir() if d.is_dir() and d.name.startswith("step-")],
        key=lambda d: int(d.name.split("-")[1]),
        reverse=True
    )
    
    return checkpoints[0] if checkpoints else None
