import json
from pathlib import Path
import subprocess

import torch

from art.megatron.model_support.handlers import QWEN3_5_MOE_HANDLER

ROOT = Path(__file__).resolve().parents[4]


def _config(base_model: str, *, rank: int) -> dict:
    return {
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": rank,
        "target_modules": [
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "bias": "none",
    }


def _sentinel(
    expert: int,
    module_id: int,
    lora_id: int,
    shape: tuple[int, int],
) -> torch.Tensor:
    return (
        torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
        + expert * 10_000
        + module_id * 1_000
        + lora_id * 100
    )


def _qwen35_art_moe_tensors(
    prefix: str,
    *,
    num_experts: int,
    rank: int,
    hidden: int,
    intermediate: int,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    module_ids = {"gate_proj": 1, "up_proj": 2, "down_proj": 3}
    for expert in range(num_experts):
        for module, module_id in module_ids.items():
            in_dim = intermediate if module == "down_proj" else hidden
            out_dim = hidden if module == "down_proj" else intermediate
            module_prefix = f"{prefix}.mlp.experts.{expert}.{module}"
            tensors[f"{module_prefix}.lora_A.weight"] = _sentinel(
                expert,
                module_id,
                0,
                (rank, in_dim),
            )
            tensors[f"{module_prefix}.lora_B.weight"] = _sentinel(
                expert,
                module_id,
                1,
                (out_dim, rank),
            )
    return tensors


def _expected_vllm_stack(
    art_tensors: dict[str, torch.Tensor],
    art_prefix: str,
    experts: list[int],
    *,
    rank: int,
    vllm_rank: int,
    hidden: int,
    intermediate: int,
) -> dict[str, torch.Tensor]:
    gate_up_a = torch.zeros(len(experts), vllm_rank, hidden)
    gate_up_b = torch.zeros(len(experts), 2 * intermediate, vllm_rank)
    down_a = torch.zeros(len(experts), vllm_rank, intermediate)
    down_b = torch.zeros(len(experts), hidden, vllm_rank)
    for local_expert, global_expert in enumerate(experts):
        expert_prefix = f"{art_prefix}.mlp.experts.{global_expert}"
        gate_up_a[local_expert, :rank] = art_tensors[
            f"{expert_prefix}.gate_proj.lora_A.weight"
        ]
        gate_up_a[local_expert, rank:vllm_rank] = art_tensors[
            f"{expert_prefix}.up_proj.lora_A.weight"
        ]
        gate_up_b[local_expert, :intermediate, :rank] = art_tensors[
            f"{expert_prefix}.gate_proj.lora_B.weight"
        ]
        gate_up_b[local_expert, intermediate:, rank:vllm_rank] = art_tensors[
            f"{expert_prefix}.up_proj.lora_B.weight"
        ]
        down_a[local_expert, :rank] = art_tensors[
            f"{expert_prefix}.down_proj.lora_A.weight"
        ]
        down_b[local_expert, :, :rank] = art_tensors[
            f"{expert_prefix}.down_proj.lora_B.weight"
        ]
    return {
        "gate_up_a": gate_up_a,
        "gate_up_b": gate_up_b,
        "down_a": down_a,
        "down_b": down_b,
    }


def _run_vllm_stack_probe(
    artifact_dir: Path,
    tensors: dict[str, torch.Tensor],
    *,
    vllm_prefix: str,
    rank: int,
    hidden: int,
    num_local_experts: int,
    expert_map: list[int] | None,
) -> dict[str, torch.Tensor]:
    tensors_path = artifact_dir / (
        "ep_vllm_tensors.pt" if expert_map is not None else "vllm_tensors.pt"
    )
    torch.save(tensors, tensors_path)
    script = r"""
import json
from types import SimpleNamespace
import sys

import torch

from vllm.lora.layers import fused_moe


class FakeFusedMoE3DWithLoRA:
    pass


fused_moe.FusedMoE3DWithLoRA = FakeFusedMoE3DWithLoRA

from art_vllm_runtime.patches import apply_vllm_runtime_patches

apply_vllm_runtime_patches()

from vllm.lora.model_manager import LoRAModelManager

tensors = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
prefix = sys.argv[2]
rank = int(sys.argv[3])
hidden = int(sys.argv[4])
num_local_experts = int(sys.argv[5])
expert_map_values = json.loads(sys.argv[6])
module_name = "language_model.model.layers.0.mlp.experts"
down = SimpleNamespace(
    lora_a=tensors[f"{prefix}.lora_A.weight"].clone(),
    lora_b=tensors[f"{prefix}.lora_B.weight"].clone(),
    rank=rank,
)
gate_up = SimpleNamespace(
    lora_a=tensors[f"{prefix}.base_layer.lora_A.weight"].clone(),
    lora_b=tensors[f"{prefix}.base_layer.lora_B.weight"].clone(),
    rank=rank,
)
lora_model = SimpleNamespace(
    loras={module_name: down, module_name + ".base_layer": gate_up}
)


class FakeManager:
    _is_3d_moe_model = True

    def _get_lora_layer_weights(self, lora_model, name):
        return lora_model.loras.get(name)


module = FakeFusedMoE3DWithLoRA()
use_ep = expert_map_values is not None
expert_map = (
    torch.tensor(expert_map_values, dtype=torch.int32)
    if expert_map_values is not None
    else None
)
module.base_layer = SimpleNamespace(
    use_ep=use_ep,
    local_num_experts=num_local_experts,
    _expert_map=expert_map,
)
module.w13_lora_a_stacked = (torch.empty(1, num_local_experts, rank, hidden),)
LoRAModelManager._stack_moe_lora_weights(
    FakeManager(),
    lora_model,
    module,
    module_name,
)
stacked = lora_model.loras[module_name]
print(json.dumps({
    "gate_up_a": stacked.lora_a[0].tolist(),
    "down_a": stacked.lora_a[1].tolist(),
    "gate_up_b": stacked.lora_b[0].tolist(),
    "down_b": stacked.lora_b[1].tolist(),
}))
"""
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT / "vllm_runtime"),
            "python",
            "-c",
            script,
            str(tensors_path),
            vllm_prefix,
            str(rank),
            str(hidden),
            str(num_local_experts),
            json.dumps(expert_map),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    suffix = "ep_" if expert_map is not None else ""
    (artifact_dir / f"{suffix}vllm_stack_stdout.txt").write_text(result.stdout)
    (artifact_dir / f"{suffix}vllm_stack_stderr.txt").write_text(result.stderr)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return {key: torch.tensor(value) for key, value in payload.items()}


def _assert_exact_stack(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert set(actual) == set(expected)
    for key, expected_tensor in expected.items():
        assert torch.equal(actual[key], expected_tensor), key


def test_qwen35_vllm_lora_stack_preserves_expert_rank_layout(
    artifact_dir: Path,
) -> None:
    rank = 2
    vllm_rank = 2 * rank
    hidden = 3
    intermediate = 4
    num_experts = 4
    art_prefix = "base_model.model.model.layers.0"
    vllm_prefix = "base_model.model.model.language_model.layers.0.mlp.experts"
    art_tensors = _qwen35_art_moe_tensors(
        art_prefix,
        num_experts=num_experts,
        rank=rank,
        hidden=hidden,
        intermediate=intermediate,
    )
    vllm_tensors, vllm_config = QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
        art_tensors,
        adapter_config=_config("Qwen/Qwen3.5-35B-A3B", rank=rank),
    )
    (artifact_dir / "adapter_config.json").write_text(
        json.dumps(vllm_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    actual = _run_vllm_stack_probe(
        artifact_dir,
        vllm_tensors,
        vllm_prefix=vllm_prefix,
        rank=vllm_rank,
        hidden=hidden,
        num_local_experts=num_experts,
        expert_map=None,
    )
    _assert_exact_stack(
        actual,
        _expected_vllm_stack(
            art_tensors,
            art_prefix,
            list(range(num_experts)),
            rank=rank,
            vllm_rank=vllm_rank,
            hidden=hidden,
            intermediate=intermediate,
        ),
    )

    expert_map = [1, -1, 0, -1]
    actual_ep = _run_vllm_stack_probe(
        artifact_dir,
        vllm_tensors,
        vllm_prefix=vllm_prefix,
        rank=vllm_rank,
        hidden=hidden,
        num_local_experts=2,
        expert_map=expert_map,
    )
    _assert_exact_stack(
        actual_ep,
        _expected_vllm_stack(
            art_tensors,
            art_prefix,
            [2, 0],
            rank=rank,
            vllm_rank=vllm_rank,
            hidden=hidden,
            intermediate=intermediate,
        ),
    )
