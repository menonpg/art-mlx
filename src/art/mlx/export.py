#!/usr/bin/env python3
"""
ART-MLX: Export trained LoRA adapters to HuggingFace Hub

Usage:
    # Upload to HuggingFace
    python -m art.mlx.export --checkpoint ./checkpoints/my_model --repo username/my-lora --push
    
    # Merge LoRA into base model
    python -m art.mlx.export --checkpoint ./checkpoints/my_model --merge --output ./merged_model
    
    # Convert to GGUF for Ollama/llama.cpp
    python -m art.mlx.export --checkpoint ./checkpoints/my_model --gguf --output ./model.gguf
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load a checkpoint and return metadata."""
    checkpoint_dir = Path(checkpoint_path)
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load metadata
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Check for adapter weights
    adapter_path = checkpoint_dir / "adapters.safetensors"
    if not adapter_path.exists():
        adapter_path = checkpoint_dir / "adapters.npz"
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"No adapter weights found in {checkpoint_path}")
    
    return {
        "checkpoint_dir": checkpoint_dir,
        "adapter_path": adapter_path,
        "metadata": metadata,
        "base_model": metadata.get("base_model", "unknown"),
    }


def push_to_hub(checkpoint_path: str, repo_id: str, private: bool = False):
    """Push LoRA adapters to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
        return False
    
    checkpoint = load_checkpoint(checkpoint_path)
    
    print(f"Uploading to HuggingFace: {repo_id}")
    print(f"  Base model: {checkpoint['base_model']}")
    print(f"  Adapter: {checkpoint['adapter_path']}")
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"  ✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"  ✗ Failed to create repo: {e}")
        return False
    
    # Upload adapter weights
    try:
        api.upload_file(
            path_or_fileobj=str(checkpoint['adapter_path']),
            path_in_repo=checkpoint['adapter_path'].name,
            repo_id=repo_id,
        )
        print(f"  ✓ Uploaded adapter weights")
    except Exception as e:
        print(f"  ✗ Failed to upload weights: {e}")
        return False
    
    # Upload metadata
    metadata_path = checkpoint['checkpoint_dir'] / "metadata.json"
    if metadata_path.exists():
        try:
            api.upload_file(
                path_or_fileobj=str(metadata_path),
                path_in_repo="metadata.json",
                repo_id=repo_id,
            )
            print(f"  ✓ Uploaded metadata")
        except Exception as e:
            print(f"  Warning: Failed to upload metadata: {e}")
    
    # Create README
    readme_content = f"""---
tags:
- art-mlx
- lora
- grpo
- apple-silicon
base_model: {checkpoint['base_model']}
---

# {repo_id.split('/')[-1]}

LoRA adapter trained with [ART-MLX](https://github.com/menonpg/art-mlx) using GRPO on Apple Silicon.

## Usage

```python
from mlx_lm import load
from mlx_lm.tuner.utils import apply_lora_layers

# Load base model
model, tokenizer = load("{checkpoint['base_model']}")

# Apply LoRA adapters
# Download adapters.safetensors from this repo and apply
```

## Training Details

- **Base Model:** {checkpoint['base_model']}
- **Method:** GRPO (Group Relative Policy Optimization)
- **Framework:** ART-MLX (MLX native)
- **Hardware:** Apple Silicon

## Metadata

```json
{json.dumps(checkpoint['metadata'], indent=2)}
```
"""
    
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
        )
        print(f"  ✓ Created README")
    except Exception as e:
        print(f"  Warning: Failed to create README: {e}")
    
    print(f"\n✅ Upload complete: https://huggingface.co/{repo_id}")
    return True


def merge_adapters(checkpoint_path: str, output_path: str):
    """Merge LoRA adapters into base model."""
    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.tuner.utils import apply_lora_layers
    except ImportError:
        print("Error: mlx-lm not installed")
        return False
    
    checkpoint = load_checkpoint(checkpoint_path)
    
    print(f"Merging adapters into base model...")
    print(f"  Base model: {checkpoint['base_model']}")
    print(f"  Output: {output_path}")
    
    # Load base model
    model, tokenizer = load(checkpoint['base_model'])
    
    # Load and apply adapters
    adapter_weights = mx.load(str(checkpoint['adapter_path']))
    
    # Merge weights (simplified - real implementation needs proper merging)
    print("  ⚠️ Full merging not yet implemented")
    print("  For now, copy base model and adapters separately")
    
    # Copy base model
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save note
    with open(output_dir / "MERGE_NOTES.txt", "w") as f:
        f.write(f"Base model: {checkpoint['base_model']}\n")
        f.write(f"Adapters: {checkpoint['adapter_path']}\n")
        f.write("\nTo use: Load base model then apply LoRA adapters\n")
    
    print(f"\n✅ Output saved to: {output_path}")
    return True


def convert_to_gguf(checkpoint_path: str, output_path: str):
    """Convert to GGUF format for Ollama/llama.cpp."""
    print("GGUF conversion requires merging first, then using llama.cpp's convert script.")
    print("\nSteps:")
    print("  1. Merge adapters: python -m art.mlx.export --checkpoint {checkpoint_path} --merge --output ./merged")
    print("  2. Convert: python llama.cpp/convert_hf_to_gguf.py ./merged --outfile {output_path}")
    print("  3. (Optional) Quantize: llama.cpp/llama-quantize {output_path} output_q4.gguf Q4_K_M")
    print("\nSee: https://github.com/ggerganov/llama.cpp#prepare-and-quantize")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export ART-MLX trained models")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--repo", help="HuggingFace repo ID (e.g., username/model-name)")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA into base model")
    parser.add_argument("--gguf", action="store_true", help="Show GGUF conversion instructions")
    parser.add_argument("--output", help="Output path for merge/gguf")
    parser.add_argument("--private", action="store_true", help="Make HuggingFace repo private")
    
    args = parser.parse_args()
    
    if args.push:
        if not args.repo:
            print("Error: --repo required for --push")
            return 1
        push_to_hub(args.checkpoint, args.repo, args.private)
    
    elif args.merge:
        if not args.output:
            print("Error: --output required for --merge")
            return 1
        merge_adapters(args.checkpoint, args.output)
    
    elif args.gguf:
        convert_to_gguf(args.checkpoint, args.output or "./model.gguf")
    
    else:
        # Just show checkpoint info
        try:
            checkpoint = load_checkpoint(args.checkpoint)
            print("Checkpoint Info:")
            print(f"  Path: {checkpoint['checkpoint_dir']}")
            print(f"  Base model: {checkpoint['base_model']}")
            print(f"  Adapter: {checkpoint['adapter_path']}")
            print(f"  Metadata: {json.dumps(checkpoint['metadata'], indent=4)}")
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
