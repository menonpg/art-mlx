from collections.abc import Callable
import math
from typing import Any

from megatron.bridge.models.conversion.model_bridge import MegatronWeightTuple
from megatron.bridge.models.conversion.peft_bridge import AdapterWeight
from megatron.core.transformer.transformer_layer import TransformerLayer
import torch

from art.megatron.lora import (
    GatedDeltaNetInProjLoRA,
    LoRA,
    MLPExpertsLinearFC1FusedLoRA,
    MLPExpertsLinearFC1LoRA,
    MLPExpertsLinearFC2LoRA,
    SelfAttentionLinearProjLoRA,
    SelfAttentionLinearQKVLoRA,
    SharedExpertsLinearFC1LoRA,
    SharedExpertsLinearFC2LoRA,
)
from art.megatron.weights.param_name_canonicalization import canonical_art_param_name


def layer_base_prefix(
    module: TransformerLayer,
    *,
    module_name: str | None = None,
) -> str:
    if module_name is not None:
        canonical_name = canonical_art_param_name(module_name)
        if canonical_name.startswith(
            ("decoder.layers.", "language_model.decoder.layers.")
        ):
            return canonical_name
    return f"language_model.decoder.layers.{module.layer_number - 1}"


def _adapter_alpha_dim(lora: LoRA) -> tuple[int, int]:
    dim = int(lora.A_T.shape[-1])
    alpha = float(lora.scale) * dim
    rounded_alpha = round(alpha)
    assert math.isclose(alpha, rounded_alpha)
    return rounded_alpha, dim


def _adapter_tensors(
    lora: LoRA,
    expert_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_t = lora.A_T if expert_idx is None else lora.A_T[expert_idx]
    b_t = lora.B_T if expert_idx is None else lora.B_T[expert_idx]
    return a_t.transpose(-1, -2).contiguous(), b_t.transpose(-1, -2).contiguous()


def _adapter_param_prefix(base_prefix: str, adapter_key: str | None) -> str:
    if adapter_key is None:
        return f"{base_prefix}.adapter"
    return f"{base_prefix}.adapter.{adapter_key}"


def _adapter_weight(
    *,
    base_prefix: str,
    adapter_key: str | None,
    alpha: int,
    dim: int,
    linear_in: torch.Tensor,
    linear_out: torch.Tensor,
) -> AdapterWeight:
    param_prefix = _adapter_param_prefix(base_prefix, adapter_key)
    return AdapterWeight(
        global_base_prefix=base_prefix,
        adapter_key=adapter_key,
        alpha=alpha,
        dim=dim,
        linear_in_weight=MegatronWeightTuple(
            param_name=f"{param_prefix}.linear_in.weight",
            weight=linear_in,
            vp_stage=0,
        ),
        linear_out_weight=MegatronWeightTuple(
            param_name=f"{param_prefix}.linear_out.weight",
            weight=linear_out,
            vp_stage=0,
        ),
    )


def _simple_adapter_weight(
    base_prefix: str,
    lora: LoRA,
    *,
    adapter_key: str | None = None,
    expert_idx: int | None = None,
) -> AdapterWeight:
    alpha, dim = _adapter_alpha_dim(lora)
    linear_in, linear_out = _adapter_tensors(lora, expert_idx)
    return _adapter_weight(
        base_prefix=base_prefix,
        adapter_key=adapter_key,
        alpha=alpha,
        dim=dim,
        linear_in=linear_in,
        linear_out=linear_out,
    )


def _zero_adapter_weight(
    *,
    base_prefix: str,
    adapter_key: str,
    input_dim: int,
    output_dim: int,
    like: torch.Tensor,
) -> AdapterWeight:
    return _adapter_weight(
        base_prefix=base_prefix,
        adapter_key=adapter_key,
        alpha=1,
        dim=1,
        linear_in=like.new_zeros((1, input_dim)),
        linear_out=like.new_zeros((output_dim, 1)),
    )


def _fused_pair_adapter_weight(
    base_prefix: str,
    first_lora: LoRA,
    second_lora: LoRA,
    *,
    first_expert_idx: int | None = None,
    second_expert_idx: int | None = None,
) -> AdapterWeight:
    first_linear_in, first_linear_out = _adapter_tensors(first_lora, first_expert_idx)
    second_linear_in, second_linear_out = _adapter_tensors(
        second_lora,
        second_expert_idx,
    )
    assert math.isclose(float(first_lora.scale), float(second_lora.scale))
    total_dim = int(first_linear_in.shape[0] + second_linear_in.shape[0])
    alpha = round(float(first_lora.scale) * total_dim)

    first_rank = int(first_linear_in.shape[0])
    second_rank = int(second_linear_in.shape[0])
    first_out = int(first_linear_out.shape[0])
    second_out = int(second_linear_out.shape[0])

    first_padding = first_linear_out.new_zeros((first_out, second_rank))
    second_padding = second_linear_out.new_zeros((second_out, first_rank))
    return _adapter_weight(
        base_prefix=base_prefix,
        adapter_key=None,
        alpha=alpha,
        dim=total_dim,
        linear_in=torch.cat([first_linear_in, second_linear_in], dim=0),
        linear_out=torch.cat(
            [
                torch.cat([first_linear_out, first_padding], dim=1),
                torch.cat([second_padding, second_linear_out], dim=1),
            ],
            dim=0,
        ),
    )


def _set_adapter_weights(
    out: dict[str, list[Any]],
    base_prefix: str,
    *weights: AdapterWeight,
    weight_suffix: str = ".weight",
) -> None:
    out[f"{base_prefix}{weight_suffix}"] = list(weights)


def _set_expert_adapter_weights(
    out: dict[str, list[Any]],
    base_prefix: str,
    lora: LoRA,
    build_weight: Callable[[int], AdapterWeight],
) -> None:
    for local_expert_idx in range(lora.num_local_experts):
        global_expert_idx = local_expert_idx + lora._expert_offset
        _set_adapter_weights(
            out,
            base_prefix,
            build_weight(local_expert_idx),
            weight_suffix=f".weight{global_expert_idx}",
        )


def add_standard_self_attention_adapter_weights(
    adapter_weights_by_base: dict[str, list[Any]],
    *,
    layer_prefix: str,
    self_attention: Any,
) -> None:
    linear_proj = getattr(self_attention, "linear_proj", None)
    if isinstance(linear_proj, SelfAttentionLinearProjLoRA):
        base_prefix = f"{layer_prefix}.self_attention.linear_proj"
        _set_adapter_weights(
            adapter_weights_by_base,
            base_prefix,
            _simple_adapter_weight(base_prefix, linear_proj.lora),
        )

    linear_qkv = getattr(self_attention, "linear_qkv", None)
    if isinstance(linear_qkv, SelfAttentionLinearQKVLoRA):
        base_prefix = f"{layer_prefix}.self_attention.linear_qkv"
        _set_adapter_weights(
            adapter_weights_by_base,
            base_prefix,
            *(
                _simple_adapter_weight(base_prefix, lora, adapter_key=key)
                for lora, key in (
                    (linear_qkv.q_proj_lora, "adapter_q"),
                    (linear_qkv.k_proj_lora, "adapter_k"),
                    (linear_qkv.v_proj_lora, "adapter_v"),
                )
            ),
        )


def add_gated_delta_net_adapter_weights(
    adapter_weights_by_base: dict[str, list[Any]],
    *,
    layer_prefix: str,
    self_attention: Any,
) -> None:
    out_proj = getattr(self_attention, "out_proj", None)
    if isinstance(out_proj, SelfAttentionLinearProjLoRA):
        base_prefix = f"{layer_prefix}.self_attention.out_proj"
        _set_adapter_weights(
            adapter_weights_by_base,
            base_prefix,
            _simple_adapter_weight(base_prefix, out_proj.lora),
        )

    in_proj = getattr(self_attention, "in_proj", None)
    if isinstance(in_proj, GatedDeltaNetInProjLoRA):
        base_prefix = f"{layer_prefix}.self_attention.in_proj"
        input_dim = int(in_proj.qkv_lora.A_T.shape[-2])
        output_dim = int(in_proj.num_value_heads_per_partition)
        _set_adapter_weights(
            adapter_weights_by_base,
            base_prefix,
            _simple_adapter_weight(
                base_prefix,
                in_proj.qkv_lora,
                adapter_key="adapter_qkv",
            ),
            _simple_adapter_weight(
                base_prefix,
                in_proj.z_lora,
                adapter_key="adapter_z",
            ),
            *(
                _zero_adapter_weight(
                    base_prefix=base_prefix,
                    adapter_key=adapter_key,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    like=in_proj.qkv_lora.B_T,
                )
                for adapter_key in ("adapter_b", "adapter_a")
            ),
        )


def add_grouped_moe_adapter_weights(
    adapter_weights_by_base: dict[str, list[Any]],
    *,
    layer_prefix: str,
    experts: Any,
) -> None:
    linear_fc1 = getattr(experts, "linear_fc1", None)
    base_prefix = f"{layer_prefix}.mlp.experts.linear_fc1"
    if isinstance(linear_fc1, MLPExpertsLinearFC1FusedLoRA):
        _set_expert_adapter_weights(
            adapter_weights_by_base,
            base_prefix,
            linear_fc1.lora,
            lambda local_expert_idx: _simple_adapter_weight(
                base_prefix,
                linear_fc1.lora,
                expert_idx=local_expert_idx,
            ),
        )
    elif isinstance(linear_fc1, MLPExpertsLinearFC1LoRA):
        _set_expert_adapter_weights(
            adapter_weights_by_base,
            base_prefix,
            linear_fc1.gate_lora,
            lambda local_expert_idx: _fused_pair_adapter_weight(
                base_prefix,
                linear_fc1.gate_lora,
                linear_fc1.up_lora,
                first_expert_idx=local_expert_idx,
                second_expert_idx=local_expert_idx,
            ),
        )

    linear_fc2 = getattr(experts, "linear_fc2", None)
    if isinstance(linear_fc2, MLPExpertsLinearFC2LoRA):
        base_prefix = f"{layer_prefix}.mlp.experts.linear_fc2"
        _set_expert_adapter_weights(
            adapter_weights_by_base,
            base_prefix,
            linear_fc2.lora,
            lambda local_expert_idx: _simple_adapter_weight(
                base_prefix,
                linear_fc2.lora,
                expert_idx=local_expert_idx,
            ),
        )


def _add_split_mlp_adapter_weights(
    adapter_weights_by_base: dict[str, list[Any]],
    *,
    base_prefix: str,
    mlp: Any,
) -> None:
    linear_fc1 = getattr(mlp, "linear_fc1", None)
    if isinstance(linear_fc1, SharedExpertsLinearFC1LoRA):
        fc1_prefix = f"{base_prefix}.linear_fc1"
        _set_adapter_weights(
            adapter_weights_by_base,
            fc1_prefix,
            *(
                _simple_adapter_weight(fc1_prefix, lora, adapter_key=adapter_key)
                for lora, adapter_key in (
                    (linear_fc1.gate_lora, "adapter_gate"),
                    (linear_fc1.up_lora, "adapter_up"),
                )
            ),
        )

    linear_fc2 = getattr(mlp, "linear_fc2", None)
    if isinstance(linear_fc2, SharedExpertsLinearFC2LoRA):
        fc2_prefix = f"{base_prefix}.linear_fc2"
        _set_adapter_weights(
            adapter_weights_by_base,
            fc2_prefix,
            _simple_adapter_weight(fc2_prefix, linear_fc2.row_parallel_lora.lora),
        )


def add_dense_mlp_adapter_weights(
    adapter_weights_by_base: dict[str, list[Any]],
    *,
    layer_prefix: str,
    mlp: Any,
) -> None:
    _add_split_mlp_adapter_weights(
        adapter_weights_by_base,
        base_prefix=f"{layer_prefix}.mlp",
        mlp=mlp,
    )


def add_shared_experts_adapter_weights(
    adapter_weights_by_base: dict[str, list[Any]],
    *,
    layer_prefix: str,
    shared_experts: Any,
) -> None:
    _add_split_mlp_adapter_weights(
        adapter_weights_by_base,
        base_prefix=f"{layer_prefix}.mlp.shared_experts",
        mlp=shared_experts,
    )
