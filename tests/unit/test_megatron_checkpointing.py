from __future__ import annotations

import json

from safetensors.torch import save_file
import torch

from art.megatron.merge import load_lora_adapter_state_dict, merge_lora_adapter


def test_load_lora_adapter_state_dict_from_shards(tmp_path) -> None:
    save_file(
        {"layer.lora_A.weight": torch.tensor([[1.0], [2.0]])},
        tmp_path / "adapter_model-01-of-02.safetensors",
    )
    save_file(
        {"layer.lora_A.weight": torch.tensor([[3.0], [4.0]])},
        tmp_path / "adapter_model-02-of-02.safetensors",
    )
    (tmp_path / "adapter_manifest-01-of-02.json").write_text(
        json.dumps(
            {
                "layer.lora_A.weight": {
                    "sharded": True,
                    "shard_world_size": 2,
                    "shard_rank": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "adapter_manifest-02-of-02.json").write_text(
        json.dumps(
            {
                "layer.lora_A.weight": {
                    "sharded": True,
                    "shard_world_size": 2,
                    "shard_rank": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    adapter_model = load_lora_adapter_state_dict(str(tmp_path))

    assert torch.equal(
        adapter_model["layer.lora_A.weight"],
        torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
    )
    assert list(tmp_path.glob("adapter_model-*-of-*.safetensors"))


def test_merge_lora_adapter_writes_merged_file_and_cleans_shards(tmp_path) -> None:
    save_file(
        {"layer.lora_B.weight": torch.tensor([[1.0]])},
        tmp_path / "adapter_model-01-of-01.safetensors",
    )
    (tmp_path / "adapter_manifest-01-of-01.json").write_text(
        json.dumps(
            {
                "layer.lora_B.weight": {
                    "sharded": False,
                    "shard_world_size": 1,
                    "shard_rank": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    merge_lora_adapter(str(tmp_path))

    merged = load_lora_adapter_state_dict(str(tmp_path))
    assert torch.equal(merged["layer.lora_B.weight"], torch.tensor([[1.0]]))
    assert not list(tmp_path.glob("adapter_model-*-of-*.safetensors"))
    assert not list(tmp_path.glob("adapter_manifest-*-of-*.json"))
