import json
from pathlib import Path
import subprocess
import sys

from safetensors.torch import save_file
import torch

from art.megatron.merge import load_lora_adapter_state_dict, merge_lora_adapter
from art.megatron.model_support.handlers import (
    DEFAULT_DENSE_HANDLER,
    QWEN3_5_MOE_HANDLER,
    QWEN3_MOE_HANDLER,
)

REPO_ROOT = Path(__file__).parents[3]
VLLM_PYTHON = REPO_ROOT / "vllm_runtime/.venv/bin/python"


def _config(base_model: str, rank: int = 2, alpha: int = 4) -> dict:
    return {
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "bias": "none",
    }


def _assert_tensors_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert set(actual) == set(expected)
    for key, tensor in expected.items():
        assert torch.equal(actual[key], tensor), key


def _save_adapter(path: Path, tensors: dict[str, torch.Tensor], config: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


def _assert_stock_vllm_loads(
    path: Path,
    *,
    expected_modules: set[str],
    mapper: str = "none",
) -> list[str]:
    script = r"""
import json
import sys
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper

path = sys.argv[1]
expected = set(json.loads(sys.argv[2]))
mapper_name = sys.argv[3]
weights_mapper = None
if mapper_name == "qwen35":
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
    weights_mapper = Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
peft = PEFTHelper.from_local_dir(path, max_position_embeddings=None)
lora = LoRAModel.from_local_checkpoint(
    path,
    expected,
    peft,
    lora_model_id=1,
    device="cpu",
    weights_mapper=weights_mapper,
)
print(json.dumps(sorted(lora.loras)))
"""
    result = subprocess.run(
        [
            str(VLLM_PYTHON),
            "-c",
            script,
            str(path),
            json.dumps(sorted(expected_modules)),
            mapper,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _qwen35_moe_art_tensors(prefix: str, *, rank: int = 2) -> dict[str, torch.Tensor]:
    hidden = 3
    intermediate = 4
    tensors: dict[str, torch.Tensor] = {
        f"{prefix}.self_attn.q_proj.lora_A.weight": torch.arange(
            rank * hidden,
            dtype=torch.float32,
        ).reshape(rank, hidden),
        f"{prefix}.self_attn.q_proj.lora_B.weight": torch.arange(
            hidden * rank,
            dtype=torch.float32,
        ).reshape(hidden, rank)
        + 100,
    }
    offset = 200
    for expert in range(2):
        for module in ("gate_proj", "up_proj", "down_proj"):
            out_dim = hidden if module == "down_proj" else intermediate
            in_dim = intermediate if module == "down_proj" else hidden
            tensors[f"{prefix}.mlp.experts.{expert}.{module}.lora_A.weight"] = (
                torch.arange(rank * in_dim, dtype=torch.float32).reshape(rank, in_dim)
                + offset
            )
            offset += 100
            tensors[f"{prefix}.mlp.experts.{expert}.{module}.lora_B.weight"] = (
                torch.arange(out_dim * rank, dtype=torch.float32).reshape(out_dim, rank)
                + offset
            )
            offset += 100
    return tensors


def test_qwen35_and_qwen36_vllm_canonical_roundtrip_and_stock_loader(tmp_path: Path):
    art_prefix = "base_model.model.model.layers.0"
    original = _qwen35_moe_art_tensors(art_prefix)
    for base_model in ("Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.6-35B-A3B"):
        vllm_tensors, vllm_config = QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
            original,
            adapter_config=_config(base_model),
        )
        assert vllm_config["r"] == 4
        assert vllm_config["lora_alpha"] == 8
        assert "experts" in vllm_config["target_modules"]
        assert all("language_model.layers" in key for key in vllm_tensors)
        roundtrip = QWEN3_5_MOE_HANDLER.from_vllm_lora_tensors(
            vllm_tensors,
            adapter_config=vllm_config,
        )
        _assert_tensors_equal(roundtrip, original)
        adapter_dir = tmp_path / base_model.replace("/", "_")
        _save_adapter(adapter_dir, vllm_tensors, vllm_config)
        loaded_modules = _assert_stock_vllm_loads(
            adapter_dir,
            expected_modules=set(vllm_config["target_modules"]) | {"experts"},
            mapper="qwen35",
        )
        assert "language_model.model.layers.0.mlp.experts" in loaded_modules
        assert "language_model.model.layers.0.mlp.experts.base_layer" in loaded_modules


def test_qwen35_and_qwen36_dense_prefix_roundtrip_and_stock_loader(tmp_path: Path):
    original = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
            2,
            3,
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
            3,
            2,
        ),
    }
    for base_model in ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-4B"):
        vllm_tensors, vllm_config = QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
            original,
            adapter_config=_config(base_model),
        )
        assert set(vllm_tensors) == {
            key.replace(
                "base_model.model.model.layers.",
                "base_model.model.model.language_model.layers.",
            )
            for key in original
        }
        roundtrip = QWEN3_5_MOE_HANDLER.from_vllm_lora_tensors(
            vllm_tensors,
            adapter_config=vllm_config,
        )
        _assert_tensors_equal(roundtrip, original)
        adapter_dir = tmp_path / base_model.replace("/", "_")
        _save_adapter(adapter_dir, vllm_tensors, vllm_config)
        loaded_modules = _assert_stock_vllm_loads(
            adapter_dir,
            expected_modules={"q_proj"},
            mapper="qwen35",
        )
        assert loaded_modules == ["language_model.model.layers.0.self_attn.q_proj"]


def test_qwen3_dense_and_moe_are_already_vllm_canonical(tmp_path: Path):
    dense = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
            2,
            3,
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
            3,
            2,
        ),
    }
    assert (
        DEFAULT_DENSE_HANDLER.to_vllm_lora_tensors(
            dense,
            adapter_config=_config("Qwen/Qwen3-0.6B"),
        )[0]
        == dense
    )
    dense_dir = tmp_path / "qwen3_dense"
    _save_adapter(dense_dir, dense, _config("Qwen/Qwen3-0.6B"))
    assert _assert_stock_vllm_loads(dense_dir, expected_modules={"q_proj"}) == [
        "model.layers.0.self_attn.q_proj"
    ]

    moe = {
        "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.ones(
            2,
            3,
        ),
        "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.ones(
            4,
            2,
        ),
    }
    assert (
        QWEN3_MOE_HANDLER.to_vllm_lora_tensors(
            moe,
            adapter_config=_config("Qwen/Qwen3-30B-A3B"),
        )[0]
        == moe
    )
    moe_dir = tmp_path / "qwen3_moe"
    _save_adapter(moe_dir, moe, _config("Qwen/Qwen3-30B-A3B"))
    assert _assert_stock_vllm_loads(
        moe_dir,
        expected_modules={"experts.0.gate_proj"},
    ) == ["model.layers.0.mlp.experts.0.gate_proj"]


def test_qwen35_megatron_shards_merge_to_vllm_checkpoint_and_roundtrip(
    tmp_path: Path,
):
    prefix = "base_model.model.model.layers.0.mlp.experts.0"
    rank = 1
    hidden = 2
    intermediate = 4
    full = {
        f"{prefix}.gate_proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        f"{prefix}.gate_proj.lora_B.weight": torch.arange(
            intermediate * rank,
            dtype=torch.float32,
        ).reshape(intermediate, rank),
        f"{prefix}.up_proj.lora_A.weight": torch.tensor([[3.0, 4.0]]),
        f"{prefix}.up_proj.lora_B.weight": torch.arange(
            intermediate * rank,
            dtype=torch.float32,
        ).reshape(intermediate, rank)
        + 10,
        f"{prefix}.down_proj.lora_A.weight": torch.arange(
            rank * intermediate,
            dtype=torch.float32,
        ).reshape(rank, intermediate)
        + 20,
        f"{prefix}.down_proj.lora_B.weight": torch.arange(
            hidden * rank,
            dtype=torch.float32,
        ).reshape(hidden, rank)
        + 30,
    }

    def unsharded() -> dict:
        return {"sharded": False, "shard_world_size": 1, "shard_rank": 0}

    def sharded(rank_id: int, dim: int) -> dict:
        return {
            "sharded": True,
            "shard_world_size": 2,
            "shard_rank": rank_id,
            "export_shard_dim": dim,
            "export_shard_strategy": "uniform",
        }

    shard0 = {
        f"{prefix}.gate_proj.lora_A.weight": full[f"{prefix}.gate_proj.lora_A.weight"],
        f"{prefix}.up_proj.lora_A.weight": full[f"{prefix}.up_proj.lora_A.weight"],
        f"{prefix}.down_proj.lora_B.weight": full[f"{prefix}.down_proj.lora_B.weight"],
        f"{prefix}.gate_proj.lora_B.weight": full[f"{prefix}.gate_proj.lora_B.weight"][
            :2
        ],
        f"{prefix}.up_proj.lora_B.weight": full[f"{prefix}.up_proj.lora_B.weight"][:2],
        f"{prefix}.down_proj.lora_A.weight": full[f"{prefix}.down_proj.lora_A.weight"][
            :, :2
        ],
    }
    manifest0 = {
        f"{prefix}.gate_proj.lora_A.weight": unsharded(),
        f"{prefix}.up_proj.lora_A.weight": unsharded(),
        f"{prefix}.down_proj.lora_B.weight": unsharded(),
        f"{prefix}.gate_proj.lora_B.weight": sharded(0, 0),
        f"{prefix}.up_proj.lora_B.weight": sharded(0, 0),
        f"{prefix}.down_proj.lora_A.weight": sharded(0, 1),
    }
    shard1 = {
        f"{prefix}.gate_proj.lora_B.weight": full[f"{prefix}.gate_proj.lora_B.weight"][
            2:
        ],
        f"{prefix}.up_proj.lora_B.weight": full[f"{prefix}.up_proj.lora_B.weight"][2:],
        f"{prefix}.down_proj.lora_A.weight": full[f"{prefix}.down_proj.lora_A.weight"][
            :, 2:
        ],
    }
    manifest1 = {
        f"{prefix}.gate_proj.lora_B.weight": sharded(1, 0),
        f"{prefix}.up_proj.lora_B.weight": sharded(1, 0),
        f"{prefix}.down_proj.lora_A.weight": sharded(1, 1),
    }
    adapter_dir = tmp_path / "qwen35_megatron_shards"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(_config("Qwen/Qwen3.5-35B-A3B", rank=rank, alpha=rank)),
        encoding="utf-8",
    )
    save_file(shard0, adapter_dir / "adapter_model-01-of-02.safetensors")
    save_file(shard1, adapter_dir / "adapter_model-02-of-02.safetensors")
    (adapter_dir / "adapter_manifest-01-of-02.json").write_text(
        json.dumps(manifest0),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_manifest-02-of-02.json").write_text(
        json.dumps(manifest1),
        encoding="utf-8",
    )

    merge_lora_adapter(str(adapter_dir))

    assert not list(adapter_dir.glob("adapter_model-*-of-*.safetensors"))
    assert not list(adapter_dir.glob("adapter_manifest-*-of-*.json"))
    roundtrip = load_lora_adapter_state_dict(
        str(adapter_dir),
        handler=QWEN3_5_MOE_HANDLER,
    )
    _assert_tensors_equal(roundtrip, full)
    final_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    loaded_modules = _assert_stock_vllm_loads(
        adapter_dir,
        expected_modules=set(final_config["target_modules"]),
        mapper="qwen35",
    )
    assert "language_model.model.layers.0.mlp.experts" in loaded_modules
    assert "language_model.model.layers.0.mlp.experts.base_layer" in loaded_modules
