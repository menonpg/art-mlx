import importlib
import json
from pathlib import Path

import torch

from art.utils.lora_checkpoint import prepare_runtime_lora_checkpoint

safetensors = importlib.import_module("safetensors")
safetensors_torch = importlib.import_module("safetensors.torch")
save_file = safetensors_torch.save_file


def test_prepare_runtime_lora_checkpoint_rewrites_qwen_moe_for_native_vllm(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    runtime_dir = tmp_path / "runtime"
    source_dir.mkdir()
    tensors = {
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]]
        ),
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.tensor(
            [[10.0], [11.0], [12.0]]
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]]
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.tensor(
            [[5.0], [6.0]]
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.0.up_proj.lora_A.weight": torch.tensor(
            [[7.0, 8.0, 9.0, 10.0]]
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.0.up_proj.lora_B.weight": torch.tensor(
            [[11.0], [12.0]]
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.0.down_proj.lora_A.weight": torch.tensor(
            [[13.0, 14.0]]
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.0.down_proj.lora_B.weight": torch.tensor(
            [[15.0], [16.0], [17.0], [18.0]]
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.1.gate_proj.lora_A.weight": torch.tensor(
            [[21.0, 22.0, 23.0, 24.0]]
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.1.gate_proj.lora_B.weight": torch.tensor(
            [[25.0], [26.0]]
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.1.up_proj.lora_A.weight": torch.tensor(
            [[27.0, 28.0, 29.0, 30.0]]
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.1.up_proj.lora_B.weight": torch.tensor(
            [[31.0], [32.0]]
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.1.down_proj.lora_A.weight": torch.tensor(
            [[33.0, 34.0]]
        ),
        "base_model.model.model.language_model.layers.0.mlp.experts.1.down_proj.lora_B.weight": torch.tensor(
            [[35.0], [36.0], [37.0], [38.0]]
        ),
    }
    save_file(tensors, source_dir / "adapter_model.safetensors")
    (source_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.6-35B-A3B",
                "lora_alpha": 32,
                "r": 1,
                "target_modules": ["q_proj", "gate_proj", "up_proj", "down_proj"],
            }
        ),
        encoding="utf-8",
    )

    prepared_path = prepare_runtime_lora_checkpoint(
        str(source_dir),
        runtime_checkpoint_dir=str(runtime_dir),
        base_model="Qwen/Qwen3.6-35B-A3B",
    )

    assert prepared_path == str(runtime_dir)
    with safetensors.safe_open(
        runtime_dir / "adapter_model.safetensors",
        framework="pt",
    ) as file:
        runtime_tensors = {key: file.get_tensor(key) for key in file.keys()}
    assert (
        runtime_tensors[
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
        ].shape
        == (2, 4)
    )
    assert (
        runtime_tensors[
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight"
        ].shape
        == (3, 2)
    )
    assert torch.equal(
        runtime_tensors[
            "base_model.model.model.language_model.layers.0.mlp.experts.base_layer.lora_A.weight"
        ],
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [7.0, 8.0, 9.0, 10.0],
                [21.0, 22.0, 23.0, 24.0],
                [27.0, 28.0, 29.0, 30.0],
            ]
        ),
    )
    assert torch.equal(
        runtime_tensors[
            "base_model.model.model.language_model.layers.0.mlp.experts.base_layer.lora_B.weight"
        ],
        torch.tensor(
            [
                [5.0, 0.0, 25.0, 0.0],
                [6.0, 0.0, 26.0, 0.0],
                [0.0, 11.0, 0.0, 31.0],
                [0.0, 12.0, 0.0, 32.0],
            ]
        ),
    )
    assert torch.equal(
        runtime_tensors[
            "base_model.model.model.language_model.layers.0.mlp.experts.lora_A.weight"
        ],
        torch.tensor(
            [
                [13.0, 14.0],
                [0.0, 0.0],
                [33.0, 34.0],
                [0.0, 0.0],
            ]
        ),
    )
    assert torch.equal(
        runtime_tensors[
            "base_model.model.model.language_model.layers.0.mlp.experts.lora_B.weight"
        ],
        torch.tensor(
            [
                [15.0, 0.0, 35.0, 0.0],
                [16.0, 0.0, 36.0, 0.0],
                [17.0, 0.0, 37.0, 0.0],
                [18.0, 0.0, 38.0, 0.0],
            ]
        ),
    )
    config = json.loads((runtime_dir / "adapter_config.json").read_text("utf-8"))
    assert config["r"] == 2
    assert config["lora_alpha"] == 64
    assert "experts" in config["target_modules"]
