from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import socket

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.bridge.models.qwen_vl.qwen35_vl_provider")

from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    Qwen3_5MoeVisionConfig,
    Qwen35VLMoEModelProvider,
)
from megatron.core import parallel_state as ps
from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_layer import TransformerLayer
from torch.distributed import destroy_process_group, init_process_group, is_initialized

from art.megatron.lora import (
    GatedDeltaNetInProjLoRA,
    SelfAttentionLinearProjLoRA,
    SharedExpertsLinearFC1LoRA,
    SharedExpertsLinearFC2LoRA,
    apply_lora_adapters,
)
from art.megatron.model_support import QWEN3_5_MOE_SPEC
from art.megatron.model_support.handlers import QWEN3_5_MOE_HANDLER


class _DenseMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        linear_fc1: TELayerNormColumnParallelLinear,
        linear_fc2: TERowParallelLinear,
    ) -> None:
        super().__init__()
        self.linear_fc1 = linear_fc1
        self.linear_fc2 = linear_fc2


def _make_qwen35_provider() -> Qwen35VLMoEModelProvider:
    assert Qwen3_5MoeVisionConfig is not None
    provider = Qwen35VLMoEModelProvider(
        num_layers=4,
        hidden_size=64,
        ffn_hidden_size=128,
        moe_ffn_hidden_size=32,
        moe_shared_expert_intermediate_size=16,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=16,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_output_gate=True,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=4,
        linear_conv_kernel_dim=2,
        vocab_size=128,
        seq_length=128,
        position_embedding_type="mrope",
        vision_config=Qwen3_5MoeVisionConfig(),
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        params_dtype=torch.bfloat16,
    )
    provider.art_lora_dtype = torch.bfloat16
    provider.finalize()
    setattr(provider, "_art_model_support_handler", QWEN3_5_MOE_HANDLER)
    setattr(provider, "_art_model_support_spec", QWEN3_5_MOE_SPEC)
    return provider


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _single_rank_model_parallel() -> Iterator[None]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Megatron Qwen3.5 LoRA coverage.")
    if is_initialized():
        pytest.skip("torch.distributed is already initialized in this process.")

    torch.cuda.set_device(0)
    init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{_find_free_port()}",
        rank=0,
        world_size=1,
    )
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(1234)
        yield
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        if is_initialized():
            destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No CUDA available in this environment",
)
def test_apply_lora_adapters_wraps_qwen35_gdn_and_shared_experts() -> None:
    with _single_rank_model_parallel():
        provider = _make_qwen35_provider()
        model = provider.provide_language_model(pre_process=True, post_process=True)
        apply_lora_adapters([model], provider)

        gdn_in_proj_qkv_prefixes: list[str] = []
        gdn_in_proj_z_prefixes: list[str] = []
        gdn_out_proj_prefixes: list[str] = []
        shared_fc1_gate_prefixes: list[str] = []
        shared_fc1_up_prefixes: list[str] = []
        shared_fc2_prefixes: list[str] = []

        for module in model.modules():
            in_proj = getattr(module, "in_proj", None)
            if isinstance(in_proj, GatedDeltaNetInProjLoRA):
                gdn_in_proj_qkv_prefixes.append(in_proj.qkv_lora.adapter_model_prefix)
                gdn_in_proj_z_prefixes.append(in_proj.z_lora.adapter_model_prefix)

            out_proj = getattr(module, "out_proj", None)
            if isinstance(out_proj, SelfAttentionLinearProjLoRA):
                prefix = out_proj.lora.adapter_model_prefix
                if prefix.endswith(".linear_attn.out_proj"):
                    gdn_out_proj_prefixes.append(prefix)

            linear_fc1 = getattr(module, "linear_fc1", None)
            if isinstance(linear_fc1, SharedExpertsLinearFC1LoRA):
                shared_fc1_gate_prefixes.append(
                    linear_fc1.gate_lora.adapter_model_prefix
                )
                shared_fc1_up_prefixes.append(linear_fc1.up_lora.adapter_model_prefix)

            linear_fc2 = getattr(module, "linear_fc2", None)
            if isinstance(linear_fc2, SharedExpertsLinearFC2LoRA):
                shared_fc2_prefixes.append(
                    linear_fc2.row_parallel_lora.lora.adapter_model_prefix
                )

        assert gdn_in_proj_qkv_prefixes
        assert gdn_in_proj_z_prefixes
        assert gdn_out_proj_prefixes
        assert shared_fc1_gate_prefixes
        assert shared_fc1_up_prefixes
        assert shared_fc2_prefixes
        assert len(gdn_in_proj_qkv_prefixes) == len(gdn_in_proj_z_prefixes)
        assert len(gdn_in_proj_qkv_prefixes) == len(gdn_out_proj_prefixes)
        assert len(shared_fc1_gate_prefixes) == len(shared_fc1_up_prefixes)
        assert len(shared_fc1_gate_prefixes) == len(shared_fc2_prefixes)
        assert all(
            prefix.startswith("base_model.model.model.layers.")
            and prefix.endswith(".linear_attn.in_proj_qkv")
            for prefix in gdn_in_proj_qkv_prefixes
        )
        assert all(
            prefix.startswith("base_model.model.model.layers.")
            and prefix.endswith(".linear_attn.in_proj_z")
            for prefix in gdn_in_proj_z_prefixes
        )
        assert all(
            prefix.startswith("base_model.model.model.layers.")
            and prefix.endswith(".linear_attn.out_proj")
            for prefix in gdn_out_proj_prefixes
        )
        assert all(
            prefix.startswith("base_model.model.model.layers.")
            and prefix.endswith(".mlp.shared_expert.gate_proj")
            for prefix in shared_fc1_gate_prefixes
        )
        assert all(
            prefix.startswith("base_model.model.model.layers.")
            and prefix.endswith(".mlp.shared_expert.up_proj")
            for prefix in shared_fc1_up_prefixes
        )
        assert all(
            prefix.startswith("base_model.model.model.layers.")
            and prefix.endswith(".mlp.shared_expert.down_proj")
            for prefix in shared_fc2_prefixes
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No CUDA available in this environment",
)
def test_apply_lora_adapters_accepts_layernorm_column_fc1_dense_path() -> None:
    with _single_rank_model_parallel():
        provider = _make_qwen35_provider()
        model = provider.provide_language_model(pre_process=True, post_process=True)

        target_layer = next(
            module
            for module in model.modules()
            if isinstance(module, TransformerLayer)
            and isinstance(module.self_attention, SelfAttention)
            and isinstance(getattr(module.mlp, "shared_experts", None), SharedExpertMLP)
        )
        dense_fc1 = target_layer.self_attention.linear_qkv
        dense_fc2 = target_layer.self_attention.linear_proj
        assert isinstance(dense_fc1, TELayerNormColumnParallelLinear)
        assert isinstance(dense_fc2, TERowParallelLinear)
        target_layer.mlp = _DenseMLP(
            linear_fc1=dense_fc1,
            linear_fc2=dense_fc2,
        )

        apply_lora_adapters([model], provider)

        assert isinstance(target_layer.mlp.linear_fc1, SharedExpertsLinearFC1LoRA)
        assert isinstance(target_layer.mlp.linear_fc2, SharedExpertsLinearFC2LoRA)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No CUDA available in this environment",
)
def test_qwen35_handler_builds_canonical_adapter_weights_by_base() -> None:
    with _single_rank_model_parallel():
        provider = _make_qwen35_provider()
        model = provider.provide_language_model(pre_process=True, post_process=True)
        apply_lora_adapters([model], provider)

        adapter_weights_by_base = QWEN3_5_MOE_HANDLER.build_adapter_weights_by_base(
            [model]
        )

        qkv_key = next(
            key
            for key in adapter_weights_by_base
            if key.endswith(".self_attention.linear_qkv.weight")
        )
        qkv_weights = adapter_weights_by_base[qkv_key]
        assert len(qkv_weights) == 3
        assert {weight.adapter_key for weight in qkv_weights} == {
            "adapter_q",
            "adapter_k",
            "adapter_v",
        }

        gdn_key = next(
            key
            for key in adapter_weights_by_base
            if key.endswith(".self_attention.in_proj.weight")
        )
        gdn_weights = adapter_weights_by_base[gdn_key]
        assert len(gdn_weights) == 4
        assert {weight.adapter_key for weight in gdn_weights} == {
            "adapter_qkv",
            "adapter_z",
            "adapter_b",
            "adapter_a",
        }

        shared_fc1_key = next(
            key
            for key in adapter_weights_by_base
            if key.endswith(".mlp.shared_experts.linear_fc1.weight")
        )
        shared_fc1_weights = adapter_weights_by_base[shared_fc1_key]
        assert len(shared_fc1_weights) == 2
        assert {weight.adapter_key for weight in shared_fc1_weights} == {
            "adapter_gate",
            "adapter_up",
        }

        grouped_fc1_keys = [
            key
            for key in adapter_weights_by_base
            if ".mlp.experts.linear_fc1.weight" in key
        ]
        grouped_fc2_keys = [
            key
            for key in adapter_weights_by_base
            if ".mlp.experts.linear_fc2.weight" in key
        ]
        assert grouped_fc1_keys
        assert grouped_fc2_keys
        assert all(len(adapter_weights_by_base[key]) == 1 for key in grouped_fc1_keys)
        assert all(len(adapter_weights_by_base[key]) == 1 for key in grouped_fc2_keys)
