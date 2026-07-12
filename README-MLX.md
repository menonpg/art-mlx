# ART-MLX: Agent Reinforcement Training for Apple Silicon

[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-black?logo=apple)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**First GRPO implementation for Apple Silicon.** Train multi-step AI agents with reinforcement learning on your Mac.

This is a fork of [OpenPipe/ART](https://github.com/openpipe/art) that replaces CUDA dependencies with native MLX, enabling agent RL training on M1/M2/M3/M4 Macs.

## Why?

The original ART requires NVIDIA GPUs (vLLM + Unsloth). This fork lets you:

- **Train locally on Mac** — No cloud GPU needed for experimentation
- **64GB M1 Max = serious training** — Run 7B-14B models with LoRA
- **Iterate faster** — No network latency, no GPU rental costs
- **Full GRPO** — Same algorithm that trains reasoning models like DeepSeek-R1

## Quick Start

```bash
# Clone
git clone https://github.com/menonpg/art-mlx
cd art-mlx

# Install
pip install -e .
pip install mlx mlx-lm

# Verify
python examples/mlx_quicktest.py
```

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4)
- **macOS 13.5+**
- **Python 3.10+**
- **16GB+ RAM** (64GB recommended for 7B+ models)

## Usage

```python
from art import TrainableModel
from art.mlx import MLXBackend

# Initialize MLX backend
backend = MLXBackend()

# Create model (use mlx-community quantized models)
model = TrainableModel(
    name="my-agent",
    project="email-task",
    base_model="mlx-community/Qwen2.5-7B-Instruct-4bit"
)

# Register with backend
await model.register(backend)

# Your training loop here...
```

## Architecture

| Component | Original ART | ART-MLX |
|-----------|-------------|---------|
| Inference | vLLM (CUDA) | mlx-lm |
| LoRA Training | Unsloth (CUDA) | MLX native |
| GRPO Algorithm | PyTorch CUDA | MLX native |
| Supported Hardware | NVIDIA GPU | Apple Silicon |

## Status

🚧 **Work in Progress** — Core GRPO algorithm implemented, testing in progress.

- [x] MLXBackend class structure
- [x] GRPO algorithm in MLX
- [x] Basic inference with mlx-lm
- [ ] Full LoRA training integration
- [ ] Checkpoint save/load
- [ ] Example notebooks (Tic Tac Toe, 2048)
- [ ] Benchmarks vs original ART

## Contributing

This is an experimental fork. PRs welcome, especially for:

- GRPO training loop testing
- Memory optimization for larger models
- Example tasks and benchmarks

## Credits

- [OpenPipe/ART](https://github.com/openpipe/art) — Original GRPO framework
- [Apple MLX](https://github.com/ml-explore/mlx) — ML framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LLM inference and LoRA for MLX

## License

Apache 2.0 (same as original ART)
