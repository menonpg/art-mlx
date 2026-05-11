from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import torch

QWEN35_GDN_LINEAR_POLICY = ("noop", "real")


class GdnModuleConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    hidden_size: int = Field(ge=1)
    model_builder_layers: int = Field(ge=1)
    ffn_hidden_size: int = Field(ge=1)
    moe_ffn_hidden_size: int = Field(ge=1)
    moe_shared_expert_intermediate_size: int = Field(ge=1)
    num_attention_heads: int = Field(ge=1)
    num_query_groups: int = Field(ge=1)
    kv_channels: int = Field(ge=1)
    linear_key_head_dim: int = Field(ge=1)
    linear_value_head_dim: int = Field(ge=1)
    linear_num_key_heads: int = Field(ge=1)
    linear_num_value_heads: int = Field(ge=1)
    linear_conv_kernel_dim: int = Field(ge=1)
    num_moe_experts: int = Field(ge=1)
    moe_router_topk: int = Field(ge=1)
    description: str = ""


def qwen35_gdn_module_config() -> GdnModuleConfig:
    return GdnModuleConfig(
        name="qwen3_5_35b_a3b",
        hidden_size=2048,
        model_builder_layers=1,
        ffn_hidden_size=12288,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        num_attention_heads=16,
        num_query_groups=2,
        kv_channels=256,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_conv_kernel_dim=4,
        num_moe_experts=4,
        moe_router_topk=2,
        description=(
            "Qwen3.5-35B-A3B GDN-relevant dimensions. MoE count/top-k stay "
            "small because these benchmarks extract and run only the GDN module."
        ),
    )


def make_qwen35_gdn_pair(
    *,
    params_dtype: torch.dtype,
    linear_policy: str,
    config: GdnModuleConfig | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    resolved = config or qwen35_gdn_module_config()
    model_parallel_cuda_manual_seed(1234)
    ref_gdn = first_gdn(make_qwen35_language_model(resolved, params_dtype=params_dtype))
    model_parallel_cuda_manual_seed(5678)
    test_gdn = first_gdn(make_qwen35_language_model(resolved, params_dtype=params_dtype))
    test_gdn.load_state_dict(ref_gdn.state_dict())
    apply_gdn_linear_policy(ref_gdn, linear_policy)
    apply_gdn_linear_policy(test_gdn, linear_policy)
    _attach_main_grads(ref_gdn)
    _attach_main_grads(test_gdn)
    return ref_gdn, test_gdn


def make_qwen35_language_model(
    config: GdnModuleConfig,
    *,
    params_dtype: torch.dtype,
) -> torch.nn.Module:
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
        Qwen3_5MoeVisionConfig,
        Qwen35VLMoEModelProvider,
    )

    assert Qwen3_5MoeVisionConfig is not None
    provider = Qwen35VLMoEModelProvider(
        num_layers=config.model_builder_layers,
        hidden_size=config.hidden_size,
        ffn_hidden_size=config.ffn_hidden_size,
        moe_ffn_hidden_size=config.moe_ffn_hidden_size,
        moe_shared_expert_intermediate_size=config.moe_shared_expert_intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_query_groups,
        kv_channels=config.kv_channels,
        linear_key_head_dim=config.linear_key_head_dim,
        linear_value_head_dim=config.linear_value_head_dim,
        linear_num_key_heads=config.linear_num_key_heads,
        linear_num_value_heads=config.linear_num_value_heads,
        num_moe_experts=config.num_moe_experts,
        moe_router_topk=config.moe_router_topk,
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
        linear_conv_kernel_dim=config.linear_conv_kernel_dim,
        vocab_size=128,
        seq_length=128,
        position_embedding_type="mrope",
        vision_config=Qwen3_5MoeVisionConfig(),
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        params_dtype=params_dtype,
    )
    provider.finalize()
    return provider.provide_language_model(pre_process=True, post_process=True).cuda()


def first_gdn(model: torch.nn.Module) -> torch.nn.Module:
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    for module in model.modules():
        if isinstance(module, GatedDeltaNet):
            return module
    raise AssertionError("expected Qwen3.5 provider to build at least one GDN layer")


def apply_gdn_linear_policy(gdn: torch.nn.Module, policy: str) -> None:
    if policy == "real":
        gdn._art_benchmark_linear_policy = "real"
        return
    if policy != "noop":
        raise ValueError(f"unknown GDN benchmark linear policy {policy!r}")
    gdn.in_proj = _NoopGdnInProj(gdn)  # type: ignore[assignment]
    gdn.out_proj = _NoopGdnOutProj(int(gdn.hidden_size))  # type: ignore[assignment]
    gdn._art_benchmark_linear_policy = "noop"
    if hasattr(gdn, "_art_reentrant_te_linear_transpose_cache_disabled"):
        delattr(gdn, "_art_reentrant_te_linear_transpose_cache_disabled")


class _NoopGdnInProj(torch.nn.Module):
    def __init__(self, gdn: torch.nn.Module) -> None:
        super().__init__()
        self.out_features = int(gdn.in_proj_dim) // int(gdn.tp_size)
        self.register_buffer("_template", torch.empty(0), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        shape = (*hidden_states.shape[:-1], self.out_features)
        if (
            tuple(self._template.shape) != tuple(shape)
            or self._template.device != hidden_states.device
            or self._template.dtype != hidden_states.dtype
        ):
            template = torch.empty(
                shape, device=hidden_states.device, dtype=hidden_states.dtype
            )
            template.normal_(mean=0.0, std=0.02)
            self._template = template
        return self._template.detach().requires_grad_(hidden_states.requires_grad), None


class _NoopGdnOutProj(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, norm_out: torch.Tensor) -> tuple[torch.Tensor, None]:
        in_features = int(norm_out.shape[-1])
        if in_features == self.hidden_size:
            return norm_out, None
        if in_features > self.hidden_size and in_features % self.hidden_size == 0:
            shape = (*norm_out.shape[:-1], in_features // self.hidden_size, self.hidden_size)
            return norm_out.reshape(shape).sum(dim=-2), None
        if in_features > self.hidden_size:
            return norm_out[..., : self.hidden_size], None
        repeats = (self.hidden_size + in_features - 1) // in_features
        return norm_out.repeat_interleave(repeats, dim=-1)[..., : self.hidden_size], None


def benchmark_linear_policy(model: Any) -> str:
    return str(getattr(model, "_art_benchmark_linear_policy", "real"))


def _attach_main_grads(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        if not hasattr(parameter, "main_grad"):
            setattr(parameter, "main_grad", torch.zeros_like(parameter))
