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
# Clone and checkout MLX branch
git clone https://github.com/menonpg/art-mlx
cd art-mlx
git checkout mlx-backend

# Install with MLX dependencies
pip install -e ".[mlx]"

# Verify your environment
python examples/mlx_quicktest.py

# Run Tic Tac Toe training example
python examples/mlx_tictactoe.py
```

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4)
- **macOS 13.5+**
- **Python 3.12+**
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

## Examples

| Example | Description | Runtime |
|---------|-------------|---------|
| `mlx_quicktest.py` | Verify MLX environment | 1 min |
| `mlx_tictactoe.py` | Train agent to play Tic Tac Toe | ~30 min |

## Architecture

| Component | Original ART | ART-MLX |
|-----------|-------------|--------|
| Inference | vLLM (CUDA) | mlx-lm |
| LoRA Training | Unsloth (CUDA) | MLX native |
| GRPO Algorithm | PyTorch CUDA | MLX native |
| Supported Hardware | NVIDIA GPU | Apple Silicon |

## Project Structure

```
src/art/mlx/
├── __init__.py      # Module exports
├── backend.py       # MLXBackend class
├── grpo.py          # GRPO algorithm
└── lora.py          # LoRA utilities

examples/
├── mlx_quicktest.py # Environment verification
└── mlx_tictactoe.py # Training example
```

## Status

🚧 **Active Development** — Core GRPO algorithm implemented, testing in progress.

- [x] MLXBackend class structure
- [x] GRPO algorithm in MLX  
- [x] LoRA layer implementation
- [x] Basic inference with mlx-lm
- [x] Tic Tac Toe training example
- [ ] Full gradient updates in training loop
- [ ] Checkpoint save/load
- [ ] 2048 game example
- [ ] Benchmarks vs original ART
- [ ] PR to upstream OpenPipe/ART

## Contributing

This is an experimental fork. PRs welcome, especially for:

- Testing on different Mac hardware (M1/M2/M3/M4, various RAM configs)
- GRPO training loop testing and debugging
- Memory optimization for larger models
- Additional example tasks

## Credits

- [OpenPipe/ART](https://github.com/openpipe/art) — Original GRPO framework
- [Apple MLX](https://github.com/ml-explore/mlx) — ML framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LLM inference and LoRA for MLX

## License

Apache 2.0 (same as original ART)

---

**Created by [ThinkCreate.AI](https://thinkcreateai.com)** — First GRPO on Apple Silicon
