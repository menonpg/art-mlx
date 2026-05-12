import torch

from art.megatron.model_support.handlers import QWEN3_5_MOE_HANDLER


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


def _small_q_gate_config(*, rank: int) -> dict:
    config = _config("Qwen/Qwen3.5-35B-A3B", rank=rank)
    config.update(
        {
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 3,
        }
    )
    return config


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
    module_ids = {"gate_up_proj": 1, "down_proj": 2}
    for expert in range(num_experts):
        for module, module_id in module_ids.items():
            in_dim = intermediate if module == "down_proj" else hidden
            out_dim = hidden if module == "down_proj" else 2 * intermediate
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


def _q_proj_lora_b_to_vllm_expected(
    tensor: torch.Tensor,
    *,
    num_heads: int,
    num_groups: int,
    head_dim: int,
) -> torch.Tensor:
    heads_per_group = num_heads // num_groups
    grouped = tensor.reshape(num_groups, 2 * heads_per_group, head_dim, tensor.shape[1])
    query = grouped[:, :heads_per_group]
    gate = grouped[:, heads_per_group:]
    return torch.cat((query, gate), dim=2).reshape(tensor.shape).contiguous()


def test_qwen35_q_proj_lora_b_translates_grouped_gate_layout() -> None:
    rank = 2
    num_heads = 4
    num_groups = 2
    head_dim = 3
    rows = num_groups * 2 * (num_heads // num_groups) * head_dim
    art_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    vllm_key = (
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight"
    )
    art_tensor = torch.arange(rows * rank, dtype=torch.float32).reshape(rows, rank)
    adapter_config = _small_q_gate_config(rank=rank)

    vllm_tensors, vllm_config = QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
        {art_key: art_tensor},
        adapter_config=adapter_config,
    )

    assert vllm_config == adapter_config
    assert torch.equal(
        vllm_tensors[vllm_key],
        _q_proj_lora_b_to_vllm_expected(
            art_tensor,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
        ),
    )
    roundtrip = QWEN3_5_MOE_HANDLER.from_vllm_lora_tensors(
        vllm_tensors,
        adapter_config=adapter_config,
    )
    assert torch.equal(roundtrip[art_key], art_tensor)


def test_qwen35_moe_layout_exports_vllm_3d_without_rank_rewrite() -> None:
    rank = 2
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

    assert vllm_config["r"] == rank
    assert vllm_config["lora_alpha"] == rank
    assert vllm_config["target_modules"] == [
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
        "experts",
    ]
    assert set(vllm_tensors) == {
        f"{vllm_prefix}.base_layer.lora_A.weight",
        f"{vllm_prefix}.base_layer.lora_B.weight",
        f"{vllm_prefix}.lora_A.weight",
        f"{vllm_prefix}.lora_B.weight",
    }
    assert vllm_tensors[f"{vllm_prefix}.base_layer.lora_A.weight"].shape == (
        num_experts * rank,
        hidden,
    )
    assert vllm_tensors[f"{vllm_prefix}.base_layer.lora_B.weight"].shape == (
        2 * intermediate,
        num_experts * rank,
    )
    assert vllm_tensors[f"{vllm_prefix}.lora_A.weight"].shape == (
        num_experts * rank,
        intermediate,
    )
    assert vllm_tensors[f"{vllm_prefix}.lora_B.weight"].shape == (
        hidden,
        num_experts * rank,
    )
    roundtrip = QWEN3_5_MOE_HANDLER.from_vllm_lora_tensors(
        vllm_tensors,
        adapter_config=vllm_config,
    )
    assert set(roundtrip) == set(art_tensors)
    for key, tensor in art_tensors.items():
        assert torch.equal(roundtrip[key], tensor), key


def test_qwen35_moe_path_keeps_dense_lora_rank_when_moe_is_present() -> None:
    rank = 1
    num_heads = 4
    num_groups = 2
    head_dim = 3
    rows = num_groups * 2 * (num_heads // num_groups) * head_dim
    art_prefix = "base_model.model.model.layers.0"
    art_key = f"{art_prefix}.self_attn.q_proj.lora_B.weight"
    vllm_key = (
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight"
    )
    art_tensor = torch.arange(rows * rank, dtype=torch.float32).reshape(rows, rank)
    art_tensors = {
        **_qwen35_art_moe_tensors(
            art_prefix,
            num_experts=1,
            rank=rank,
            hidden=3,
            intermediate=4,
        ),
        art_key: art_tensor,
    }

    vllm_tensors, vllm_config = QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
        art_tensors,
        adapter_config=_small_q_gate_config(rank=rank),
    )

    expected = _q_proj_lora_b_to_vllm_expected(
        art_tensor,
        num_heads=num_heads,
        num_groups=num_groups,
        head_dim=head_dim,
    )
    assert vllm_config["r"] == rank
    assert torch.equal(vllm_tensors[vllm_key], expected)
    roundtrip = QWEN3_5_MOE_HANDLER.from_vllm_lora_tensors(
        vllm_tensors,
        adapter_config=vllm_config,
    )
    assert torch.equal(roundtrip[art_key], art_tensor)
