from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType
from typing import Any, Callable, Iterator, Literal, Sequence, cast

from causal_conv1d import causal_conv1d_fn
from fla.modules.l2norm import l2norm
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.transformer.transformer_layer import TransformerLayer
from pydantic import BaseModel, ConfigDict
import torch
from torch import Tensor
import torch.nn.functional as F

from .conv_gelu import gdn_varlen_causal_conv_gelu, packed_varlen_causal_conv
from .gdn_shared_prefix import (
    GdnPackedExecutionSpec,
    GdnParentStateTransferPlan,
    GdnRankExecutionPlan,
    GdnSegmentBucketPlan,
    build_gdn_rank_execution_plan,
    parse_gdn_shared_prefix_segments,
)
from .segment_layout import (
    gather_bucket_streams_compact as _gather_bucket_streams_compact_fused,
)
from .segment_layout import (
    prepare_packed_recurrent_inputs as _prepare_packed_recurrent_inputs_fused,
)
from .segment_layout import (
    scatter_bucket_output_compact as _scatter_bucket_output_fused,
)

_NVTX_ENABLED: ContextVar[bool] = ContextVar("art_gdn_nvtx_enabled", default=False)


class _BucketFlatLayout(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    padded_indices: Tensor
    padded_mask: Tensor
    real_indices: Tensor
    output_indices: Tensor
    output_selector: Tensor | None


def install_shared_prefix_gdn_hooks(model_chunks: Sequence[Any]) -> None:
    """Patch Megatron GatedDeltaNet modules to honor ART shared-prefix packing."""

    for chunk in model_chunks:
        if not hasattr(chunk, "modules"):
            continue
        for module in chunk.modules():
            if not isinstance(module, GatedDeltaNet):
                continue
            if getattr(module, "_art_shared_prefix_gdn_hooked", False):
                continue
            original_forward = module.forward
            module._art_physical_forward = original_forward
            module.forward = MethodType(_shared_prefix_forward, module)
            module._art_shared_prefix_gdn_hooked = True


def install_gdn_island_hooks(model_chunks: Sequence[Any]) -> None:
    """Hoist CP layout conversion across consecutive Transformer GDN layers."""

    for chunk in model_chunks:
        if not hasattr(chunk, "modules"):
            continue
        _install_empty_safe_norm_hooks(chunk)
        layers = [
            module
            for module in chunk.modules()
            if isinstance(module, TransformerLayer)
            and hasattr(module, "self_attention")
        ]
        layer_is_gdn = [
            isinstance(layer.self_attention, GatedDeltaNet) for layer in layers
        ]
        for index, layer in enumerate(layers):
            is_gdn = layer_is_gdn[index]
            layer._art_gdn_island_is_gdn = is_gdn
            layer._art_gdn_island_prev_is_gdn = index > 0 and layer_is_gdn[index - 1]
            layer._art_gdn_island_next_is_gdn = (
                index + 1 < len(layers) and layer_is_gdn[index + 1]
            )
            if getattr(layer, "_art_gdn_island_hooked", False):
                continue
            layer._art_gdn_island_physical_forward = layer.forward
            layer.forward = MethodType(_gdn_island_layer_forward, layer)
            layer._art_gdn_island_hooked = True


def _gdn_island_layer_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
    attention_bias = kwargs.get("attention_bias")
    plan = getattr(attention_bias, "gdn_execution_plan", None)
    original_forward = cast(Callable[..., Any], self._art_gdn_island_physical_forward)
    if plan is None or int(getattr(plan, "cp_size", 1)) <= 1:
        return original_forward(*args, **kwargs)

    hidden_states = _layer_forward_hidden_states(args, kwargs)
    if hidden_states is None:
        return original_forward(*args, **kwargs)

    is_gdn = bool(getattr(self, "_art_gdn_island_is_gdn", False))
    if not is_gdn:
        if getattr(attention_bias, "gdn_hidden_layout", "attention") == "gdn":
            _mark_attention_layout_active(attention_bias)
        return original_forward(*args, **kwargs)

    prev_is_gdn = bool(getattr(self, "_art_gdn_island_prev_is_gdn", False))
    next_is_gdn = bool(getattr(self, "_art_gdn_island_next_is_gdn", False))
    if prev_is_gdn:
        _mark_gdn_layout_active(attention_bias, hidden_states)
    else:
        hidden_states = _enter_gdn_island_layout(
            hidden_states, attention_bias, force=True
        )
        args, kwargs = _replace_layer_hidden_states(args, kwargs, hidden_states)

    output = (
        _empty_gdn_island_layer_forward(self, hidden_states, kwargs)
        if int(hidden_states.shape[0]) == 0
        else original_forward(*args, **kwargs)
    )
    if next_is_gdn:
        _mark_gdn_layout_active(attention_bias, _layer_output_hidden_states(output))
        return output

    hidden_out = _leave_gdn_island_layout(
        _layer_output_hidden_states(output), attention_bias
    )
    return _replace_layer_output_hidden_states(output, hidden_out)


def _layer_forward_hidden_states(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Tensor | None:
    hidden_states = kwargs.get("hidden_states")
    if isinstance(hidden_states, Tensor):
        return hidden_states
    if args and isinstance(args[0], Tensor):
        return args[0]
    return None


def _replace_layer_hidden_states(
    args: tuple[Any, ...], kwargs: dict[str, Any], hidden_states: Tensor
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if "hidden_states" in kwargs:
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = hidden_states
        return args, kwargs
    if args:
        return (hidden_states, *args[1:]), kwargs
    kwargs = dict(kwargs)
    kwargs["hidden_states"] = hidden_states
    return args, kwargs


def _layer_output_hidden_states(output: Any) -> Tensor:
    if isinstance(output, tuple):
        return cast(Tensor, output[0])
    return cast(Tensor, output)


def _replace_layer_output_hidden_states(output: Any, hidden_states: Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden_states, *output[1:])
    return hidden_states


def _install_empty_safe_norm_hooks(root: Any) -> None:
    if not isinstance(root, torch.nn.Module):
        return
    for module in root.modules():
        if getattr(module, "_art_empty_safe_norm_hooked", False):
            continue
        if not _is_empty_safe_norm_target(module):
            continue
        module._art_empty_safe_norm_physical_forward = module.forward
        module.forward = MethodType(_empty_safe_norm_forward, module)
        module._art_empty_safe_norm_hooked = True


def _is_empty_safe_norm_target(module: Any) -> bool:
    if not isinstance(getattr(module, "weight", None), Tensor):
        return False
    module_name = type(module).__name__
    module_path = type(module).__module__
    return module_name in {"RMSNorm", "LayerNorm"} and module_path.startswith(
        "transformer_engine."
    )


def _empty_safe_norm_forward(
    self: Any, input_: Tensor, *args: Any, **kwargs: Any
) -> Any:
    if isinstance(input_, Tensor) and int(input_.numel()) == 0:
        return _apply_explicit_norm(
            self,
            input_,
            config=None,
            weight_name="weight",
            bias_name="bias",
        )
    original_forward = cast(
        Callable[..., Any], self._art_empty_safe_norm_physical_forward
    )
    return original_forward(input_, *args, **kwargs)


def _empty_gdn_island_layer_forward(
    layer: Any, hidden_states: Tensor, kwargs: dict[str, Any]
) -> tuple[Tensor, Tensor | None]:
    with _nvtx_range("art_gdn_empty_island_layer", hidden_states):
        attention_output = layer.self_attention(
            hidden_states,
            attention_mask=kwargs.get("attention_mask"),
            inference_context=kwargs.get(
                "inference_context", kwargs.get("inference_params")
            ),
            rotary_pos_emb=kwargs.get("rotary_pos_emb"),
            rotary_pos_cos=kwargs.get("rotary_pos_cos"),
            rotary_pos_sin=kwargs.get("rotary_pos_sin"),
            rotary_pos_cos_sin=kwargs.get("rotary_pos_cos_sin"),
            attention_bias=kwargs.get("attention_bias"),
            packed_seq_params=kwargs.get("packed_seq_params"),
            sequence_len_offset=kwargs.get("sequence_len_offset"),
        )
    context = kwargs.get("context")
    if isinstance(attention_output, dict) and "context" in attention_output:
        context = attention_output["context"]
    attention_hidden = (
        attention_output[0] if isinstance(attention_output, tuple) else attention_output
    )
    return hidden_states + cast(Tensor, attention_hidden), context


def _shared_prefix_forward(
    self: Any,
    hidden_states: Tensor,
    attention_mask: Tensor,
    key_value_states: Tensor | None = None,
    inference_context: Any | None = None,
    attention_bias: Any | None = None,
    packed_seq_params: Any | None = None,
    sequence_len_offset: int | None = None,
    *,
    inference_params: Any | None = None,
    **kwargs: Any,
) -> tuple[Tensor, Tensor | None]:
    group_ids = getattr(attention_bias, "group_ids", None)
    parent_ids = getattr(attention_bias, "parent_ids", None)
    execution_spec = getattr(attention_bias, "gdn_execution_spec", None)
    execution_plan = getattr(attention_bias, "gdn_execution_plan", None)
    if group_ids is None or parent_ids is None:
        original_forward = cast(
            Callable[..., tuple[Tensor, Tensor | None]], self._art_physical_forward
        )
        return original_forward(
            hidden_states,
            attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
            **kwargs,
        )

    del attention_mask, key_value_states, sequence_len_offset, kwargs
    if inference_context is not None or inference_params is not None:
        raise NotImplementedError("ART shared-prefix GDN does not support inference.")
    if packed_seq_params is not None:
        raise NotImplementedError(
            "PackedSeqParams is not used in ART shared-prefix GDN."
        )
    return gdn_shared_prefix_forward(
        self,
        hidden_states,
        group_ids=cast(Tensor, group_ids),
        parent_ids=cast(Tensor, parent_ids),
        execution_spec=cast(GdnPackedExecutionSpec | None, execution_spec),
        execution_plan=cast(GdnRankExecutionPlan | None, execution_plan),
        input_layout=(
            "gdn"
            if getattr(attention_bias, "gdn_hidden_layout", "attention") == "gdn"
            else "attention"
        ),
        output_layout=(
            "gdn"
            if getattr(attention_bias, "gdn_hidden_layout", "attention") == "gdn"
            else "attention"
        ),
        require_prebuilt_plan=False,
    )


@torch.compiler.disable
def gdn_shared_prefix_forward(
    gdn: Any,
    hidden_states: Tensor,
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
    execution_spec: GdnPackedExecutionSpec | None = None,
    execution_plan: GdnRankExecutionPlan | None = None,
    cp_group: Any | None = None,
    require_prebuilt_plan: bool = False,
    input_layout: Literal["attention", "gdn"] = "attention",
    output_layout: Literal["attention", "gdn"] = "attention",
) -> tuple[Tensor, Tensor | None]:
    """Run one GDN layer over ART shared-prefix packed rows."""

    return run_gdn_layer(
        gdn,
        hidden_states,
        group_ids=group_ids,
        parent_ids=parent_ids,
        execution_spec=execution_spec,
        execution_plan=execution_plan,
        cp_group=cp_group,
        require_prebuilt_plan=require_prebuilt_plan,
        input_layout=input_layout,
        output_layout=output_layout,
    )


@torch.compiler.disable
def run_gdn_layer(
    gdn: Any,
    hidden_states: Tensor,
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
    execution_spec: GdnPackedExecutionSpec | None = None,
    execution_plan: GdnRankExecutionPlan | None = None,
    cp_group: Any | None = None,
    require_prebuilt_plan: bool = False,
    input_layout: Literal["attention", "gdn"] = "attention",
    output_layout: Literal["attention", "gdn"] = "attention",
) -> tuple[Tensor, Tensor | None]:
    """Run one production shared-prefix GDN layer."""

    _disable_reentrant_te_linear_transpose_cache(gdn)
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must be [S, B, D], got {tuple(hidden_states.shape)}"
        )
    seq_len, batch_size, _ = hidden_states.shape
    requested_cp_size = (
        execution_plan.cp_size if execution_plan is not None else _default_cp_size()
    )
    cp_rank = (
        execution_plan.cp_rank
        if execution_plan is not None
        else _default_cp_rank(requested_cp_size)
    )
    full_shape_required = requested_cp_size == 1
    expected_group_seq_len = seq_len
    if full_shape_required and _gdn_uses_sequence_parallel(gdn):
        expected_group_seq_len *= int(getattr(gdn, "sp_size", 1))
    if full_shape_required and (
        int(group_ids.shape[0]) != batch_size
        or int(group_ids.shape[1]) != expected_group_seq_len
    ):
        raise ValueError(
            "shared-prefix GDN group_ids shape must match the logical sequence "
            "processed by Megatron GDN after sequence-parallel input gather, got "
            f"hidden={tuple(hidden_states.shape)} group_ids={tuple(group_ids.shape)} "
            f"expected_group_shape={(batch_size, expected_group_seq_len)}"
        )

    if require_prebuilt_plan and execution_plan is None:
        raise ValueError(
            "ART shared-prefix GDN production path requires a prebuilt "
            "GDN execution plan on SharedPrefixAttentionState. Build it once "
            "per packed sequence via create_shared_prefix_state(..., "
            "build_gdn_execution_spec=True)."
        )

    if execution_spec is None and execution_plan is None:
        with _nvtx_range("art_gdn_parse_shared_prefix_layout", hidden_states):
            execution_spec = parse_gdn_shared_prefix_segments(
                group_ids, parent_ids, min_completions_per_family=0
            )
    if (
        execution_spec is not None
        and requested_cp_size == 1
        and (
            execution_spec.batch_size != batch_size
            or execution_spec.sequence_length != expected_group_seq_len
        )
    ):
        raise ValueError(
            "GDN execution spec shape must match hidden_states, got "
            f"spec={(execution_spec.batch_size, execution_spec.sequence_length)} "
            f"expected={(batch_size, expected_group_seq_len)} "
            f"hidden={(batch_size, seq_len)}"
        )
    if execution_plan is None:
        if execution_spec is None:
            raise ValueError("GDN execution spec is required to build a missing plan")
        with _nvtx_range("art_gdn_plan_shared_prefix_layout", hidden_states):
            execution_plan = build_gdn_rank_execution_plan(
                execution_spec,
                device=hidden_states.device,
                cp_rank=cp_rank,
                cp_size=requested_cp_size,
            )
    elif execution_plan.cp_size == 1 and (
        execution_plan.batch_size != batch_size
        or execution_plan.sequence_length != seq_len
    ):
        raise ValueError(
            "GDN execution plan shape must match hidden_states, got "
            f"plan={(execution_plan.batch_size, execution_plan.sequence_length)} "
            f"hidden={(batch_size, seq_len)}"
        )
    if execution_plan.cp_size != 1:
        return _run_cp_planned_prefixes_and_completions(
            gdn,
            hidden_states,
            execution_plan,
            group=cp_group or _default_cp_group(execution_plan.cp_size),
            input_layout=input_layout,
            output_layout=output_layout,
        )
    if input_layout != "attention" or output_layout != "attention":
        raise ValueError("GDN layout controls require a CP execution plan")
    return _run_planned_prefixes_and_completions(gdn, hidden_states, execution_plan)


def _run_planned_prefixes_and_completions(
    gdn: Any,
    hidden_states: Tensor,
    plan: GdnRankExecutionPlan,
) -> tuple[Tensor, Tensor | None]:
    if _has_chunk_aligned_local_plan(plan):
        return _run_chunk_aligned_prefixes_and_completions(gdn, hidden_states, plan)
    raise ValueError(
        "shared-prefix GDN requires a chunk-aligned execution plan; "
        "prefix/completion bucket execution has been removed"
    )


def _has_chunk_aligned_local_plan(plan: GdnRankExecutionPlan) -> bool:
    return bool(
        plan.prefix_boundary_buckets
        or plan.prefix_tail_buckets
        or plan.completion_warmup_buckets
    )


def _run_chunk_aligned_prefixes_and_completions(
    gdn: Any,
    hidden_states: Tensor,
    plan: GdnRankExecutionPlan,
) -> tuple[Tensor, Tensor | None]:
    with _nvtx_range("art_gdn_in_proj", hidden_states):
        qkv, gate, beta, recurrent_g = _project_gdn_inputs(gdn, hidden_states)
    gate = gate.clone()
    recurrent_output = torch.zeros_like(gate)
    boundary_family_chunks: list[Tensor] = []
    boundary_conv_chunks: list[Tensor] = []
    boundary_rec_chunks: list[Tensor] = []

    for bucket in plan.prefix_boundary_buckets:
        with _nvtx_range("art_gdn_input_layout_gather_reorder", qkv):
            prefix_qkv, prefix_beta, prefix_g = _gather_compact_bucket_streams(
                qkv, beta, recurrent_g, bucket
            )
        zero_conv = _zero_conv_state(
            gdn, hidden_states, batch_size=bucket.segment_count
        )
        zero_rec = _zero_recurrent_state(
            gdn, hidden_states, batch_size=bucket.segment_count
        )
        with _nvtx_range("art_gdn_prefix_boundary_segment", prefix_qkv):
            prefix_out, prefix_conv, prefix_rec = run_gdn_bucket(
                bucket,
                (prefix_qkv, prefix_beta, prefix_g),
                (zero_conv, zero_rec),
                gdn=gdn,
                output_final_state=True,
            )
        if prefix_conv is None or prefix_rec is None:
            raise RuntimeError("prefix boundary GDN execution must return final states")
        recurrent_output = _scatter_bucket_recurrent_output(
            recurrent_output, bucket, prefix_out
        )
        boundary_family_chunks.append(bucket.family_indices)
        boundary_conv_chunks.append(prefix_conv)
        boundary_rec_chunks.append(prefix_rec)

    boundary_conv_table = _materialize_indexed_family_state_table(
        plan=plan,
        family_chunks=boundary_family_chunks,
        state_chunks=boundary_conv_chunks,
        zero_state=_zero_conv_state(gdn, hidden_states, batch_size=plan.family_count),
    )
    boundary_rec_table = _materialize_indexed_family_state_table(
        plan=plan,
        family_chunks=boundary_family_chunks,
        state_chunks=boundary_rec_chunks,
        zero_state=_zero_recurrent_state(
            gdn, hidden_states, batch_size=plan.family_count
        ),
    )

    tail_family_chunks: list[Tensor] = []
    tail_conv_chunks: list[Tensor] = []
    tail_rec_chunks: list[Tensor] = []
    for bucket in plan.prefix_tail_buckets:
        with _nvtx_range("art_gdn_input_layout_gather_reorder", qkv):
            tail_qkv, tail_beta, tail_g = _gather_compact_bucket_streams(
                qkv, beta, recurrent_g, bucket
            )
        with _nvtx_range("art_gdn_state_fanout", tail_qkv):
            tail_conv = boundary_conv_table.index_select(0, bucket.family_indices)
            tail_rec = boundary_rec_table.index_select(0, bucket.family_indices)
        with _nvtx_range("art_gdn_prefix_tail_segment", tail_qkv):
            tail_out, tail_conv, tail_rec = run_gdn_bucket(
                bucket,
                (tail_qkv, tail_beta, tail_g),
                (tail_conv, tail_rec),
                gdn=gdn,
                output_final_state=True,
            )
        if tail_conv is None or tail_rec is None:
            raise RuntimeError("prefix tail GDN execution must return final states")
        recurrent_output = _scatter_bucket_recurrent_output(
            recurrent_output, bucket, tail_out
        )
        tail_family_chunks.append(bucket.family_indices)
        tail_conv_chunks.append(tail_conv)
        tail_rec_chunks.append(tail_rec)

    prefix_conv_table = _replace_indexed_family_states(
        boundary_conv_table,
        family_chunks=tail_family_chunks,
        state_chunks=tail_conv_chunks,
    )
    prefix_rec_table = _replace_indexed_family_states(
        boundary_rec_table,
        family_chunks=tail_family_chunks,
        state_chunks=tail_rec_chunks,
    )

    for bucket in plan.completion_warmup_buckets:
        with _nvtx_range("art_gdn_state_fanout", hidden_states):
            completion_conv = prefix_conv_table.index_select(0, bucket.family_indices)
            completion_rec = prefix_rec_table.index_select(0, bucket.family_indices)
        with _nvtx_range("art_gdn_input_layout_gather_reorder", qkv):
            completion_qkv, completion_beta, completion_g = (
                _gather_compact_bucket_streams(qkv, beta, recurrent_g, bucket)
            )
        with _nvtx_range("art_gdn_completion_warmup_segment", completion_qkv):
            completion_out, _, _ = run_gdn_bucket(
                bucket,
                (completion_qkv, completion_beta, completion_g),
                (completion_conv, completion_rec),
                gdn=gdn,
                output_final_state=False,
            )
        recurrent_output = _scatter_bucket_recurrent_output(
            recurrent_output, bucket, completion_out
        )

    return _project_gdn_output(gdn, recurrent_output, gate, plan)


def _iter_prepared_bucket_columns(
    bucket: GdnSegmentBucketPlan,
    qkv: Tensor,
    beta: Tensor,
    recurrent_g: Tensor,
    conv_initial: Tensor,
    recurrent_initial: Tensor,
) -> Iterator[tuple[GdnSegmentBucketPlan, Tensor, Tensor, Tensor, Tensor, Tensor]]:
    for column in range(int(bucket.lengths.numel())):
        length = int(bucket.lengths[column].item())
        if length == 0:
            continue
        column_bucket = _slice_bucket_column(bucket, column=column, length=length)
        yield (
            column_bucket,
            qkv[column : column + 1, :, :length],
            beta[column : column + 1, :length],
            recurrent_g[column : column + 1, :length],
            conv_initial[column : column + 1],
            recurrent_initial[column : column + 1],
        )


def _slice_bucket_column(
    bucket: GdnSegmentBucketPlan, *, column: int, length: int
) -> GdnSegmentBucketPlan:
    lengths = bucket.lengths[column : column + 1]
    cu_seqlens = torch.stack((lengths.new_zeros(()), lengths[0]))
    output_mask = (
        None
        if bucket.output_mask is None
        else bucket.output_mask[:length, column : column + 1]
    )
    return GdnSegmentBucketPlan.model_construct(
        length=length,
        lengths=lengths,
        real_mask=bucket.real_mask[:length, column : column + 1],
        cu_seqlens=cu_seqlens,
        row_indices=bucket.row_indices[:length, column : column + 1],
        position_indices=bucket.position_indices[:length, column : column + 1],
        family_indices=bucket.family_indices[column : column + 1],
        real_token_count_static=length,
        output_mask=output_mask,
    )


def _run_cp_planned_prefixes_and_completions(
    gdn: Any,
    hidden_states: Tensor,
    plan: GdnRankExecutionPlan,
    *,
    group: Any,
    input_layout: Literal["attention", "gdn"],
    output_layout: Literal["attention", "gdn"],
) -> tuple[Tensor, Tensor | None]:
    if plan.attention_to_gdn is None or plan.gdn_to_attention is None:
        raise ValueError("CP GDN execution requires prebuilt exchange plans")
    if input_layout not in ("attention", "gdn") or output_layout not in (
        "attention",
        "gdn",
    ):
        raise ValueError(
            f"unsupported GDN CP layouts: {input_layout=} {output_layout=}"
        )
    from .cp_runtime import run_gdn_prepared_varlen_native_fla_cp

    if input_layout == "attention":
        gdn_hidden, original_shape = gdn_cp_attention_to_gdn_layout(
            hidden_states, plan, group
        )
    else:
        gdn_hidden = _validate_gdn_hidden_for_cp_plan(hidden_states, plan)
        original_shape = _attention_original_shape_from_plan(hidden_states, plan)
    with _nvtx_range("art_gdn_in_proj", gdn_hidden):
        qkv, gate, beta, recurrent_g = _project_gdn_inputs(gdn, gdn_hidden)
    gate = gate.clone()
    recurrent_output = torch.zeros_like(gate)
    prefix_family_chunks: list[Tensor] = []
    prefix_conv_chunks: list[Tensor] = []
    prefix_rec_chunks: list[Tensor] = []
    cp_dependency = _empty_autograd_dependency(qkv)

    for bucket in plan.chain_prefix_buckets:
        with _nvtx_range("art_gdn_input_layout_gather_reorder", qkv):
            prefix_qkv, prefix_beta, prefix_g = _gather_bucket_streams(
                qkv, beta, recurrent_g, bucket
            )
        zero_conv = _zero_conv_state(gdn, gdn_hidden, batch_size=prefix_qkv.shape[0])
        zero_rec = _zero_recurrent_state(
            gdn, gdn_hidden, batch_size=prefix_qkv.shape[0]
        )
        with _nvtx_range("art_gdn_cp_prefix_segment", prefix_qkv):
            prefix_out, prefix_conv, prefix_rec = run_gdn_prepared_varlen_native_fla_cp(
                gdn,
                prefix_qkv,
                beta=prefix_beta,
                recurrent_g=prefix_g,
                lengths=bucket.lengths,
                cu_seqlens=bucket.cu_seqlens,
                conv_initial=zero_conv,
                recurrent_initial=zero_rec,
                group=group,
                output_final_state=True,
            )
        if prefix_conv is None or prefix_rec is None:
            raise RuntimeError("CP prefix GDN execution must return final states")
        prefix_out = _add_autograd_dependency(prefix_out, cp_dependency)
        prefix_conv = _add_autograd_dependency(prefix_conv, cp_dependency)
        prefix_rec = _add_autograd_dependency(prefix_rec, cp_dependency)
        cp_dependency = _make_autograd_dependency(prefix_out, prefix_conv, prefix_rec)
        recurrent_output = _scatter_bucket_recurrent_output(
            recurrent_output, bucket, prefix_out
        )
        prefix_family_chunks.append(bucket.family_indices)
        prefix_conv_chunks.append(prefix_conv)
        prefix_rec_chunks.append(prefix_rec)

    boundary_family_chunks: list[Tensor] = []
    boundary_conv_chunks: list[Tensor] = []
    boundary_rec_chunks: list[Tensor] = []
    for bucket in plan.prefix_boundary_buckets:
        with _nvtx_range("art_gdn_input_layout_gather_reorder", qkv):
            prefix_qkv, prefix_beta, prefix_g = _gather_bucket_streams(
                qkv, beta, recurrent_g, bucket
            )
        zero_conv = _zero_conv_state(gdn, gdn_hidden, batch_size=prefix_qkv.shape[0])
        zero_rec = _zero_recurrent_state(
            gdn, gdn_hidden, batch_size=prefix_qkv.shape[0]
        )
        with _nvtx_range("art_gdn_local_prefix_segment", prefix_qkv):
            prefix_out, prefix_conv, prefix_rec = _run_gdn_prepared_varlen_batch(
                gdn,
                prefix_qkv,
                beta=prefix_beta,
                recurrent_g=prefix_g,
                bucket=bucket,
                conv_initial=zero_conv,
                recurrent_initial=zero_rec,
                output_final_state=True,
            )
        if prefix_conv is None or prefix_rec is None:
            raise RuntimeError("local prefix GDN execution must return final states")
        prefix_out = _add_autograd_dependency(prefix_out, cp_dependency)
        prefix_conv = _add_autograd_dependency(prefix_conv, cp_dependency)
        prefix_rec = _add_autograd_dependency(prefix_rec, cp_dependency)
        recurrent_output = _scatter_bucket_recurrent_output(
            recurrent_output, bucket, prefix_out
        )
        boundary_family_chunks.append(bucket.family_indices)
        boundary_conv_chunks.append(prefix_conv)
        boundary_rec_chunks.append(prefix_rec)
        prefix_family_chunks.append(bucket.family_indices)
        prefix_conv_chunks.append(prefix_conv)
        prefix_rec_chunks.append(prefix_rec)

    if plan.prefix_tail_buckets or plan.completion_warmup_buckets:
        boundary_conv_table = _materialize_indexed_family_state_table(
            plan=plan,
            family_chunks=boundary_family_chunks,
            state_chunks=boundary_conv_chunks,
            zero_state=_zero_conv_state(gdn, gdn_hidden, batch_size=plan.family_count),
        )
        boundary_rec_table = _materialize_indexed_family_state_table(
            plan=plan,
            family_chunks=boundary_family_chunks,
            state_chunks=boundary_rec_chunks,
            zero_state=_zero_recurrent_state(
                gdn, gdn_hidden, batch_size=plan.family_count
            ),
        )
        tail_family_chunks: list[Tensor] = []
        tail_conv_chunks: list[Tensor] = []
        tail_rec_chunks: list[Tensor] = []
        for bucket in plan.prefix_tail_buckets:
            with _nvtx_range("art_gdn_input_layout_gather_reorder", qkv):
                tail_qkv, tail_beta, tail_g = _gather_bucket_streams(
                    qkv, beta, recurrent_g, bucket
                )
            tail_conv = boundary_conv_table.index_select(0, bucket.family_indices)
            tail_rec = boundary_rec_table.index_select(0, bucket.family_indices)
            with _nvtx_range("art_gdn_local_prefix_segment", tail_qkv):
                tail_out, tail_conv, tail_rec = _run_gdn_prepared_varlen_batch(
                    gdn,
                    tail_qkv,
                    beta=tail_beta,
                    recurrent_g=tail_g,
                    bucket=bucket,
                    conv_initial=tail_conv,
                    recurrent_initial=tail_rec,
                    output_final_state=True,
                )
            if tail_conv is None or tail_rec is None:
                raise RuntimeError("local prefix tail GDN execution must return states")
            tail_out = _add_autograd_dependency(tail_out, cp_dependency)
            tail_conv = _add_autograd_dependency(tail_conv, cp_dependency)
            tail_rec = _add_autograd_dependency(tail_rec, cp_dependency)
            recurrent_output = _scatter_bucket_recurrent_output(
                recurrent_output, bucket, tail_out
            )
            tail_family_chunks.append(bucket.family_indices)
            tail_conv_chunks.append(tail_conv)
            tail_rec_chunks.append(tail_rec)
            prefix_family_chunks.append(bucket.family_indices)
            prefix_conv_chunks.append(tail_conv)
            prefix_rec_chunks.append(tail_rec)
        prefix_conv_table = _replace_indexed_family_states(
            boundary_conv_table,
            family_chunks=tail_family_chunks,
            state_chunks=tail_conv_chunks,
        )
        prefix_rec_table = _replace_indexed_family_states(
            boundary_rec_table,
            family_chunks=tail_family_chunks,
            state_chunks=tail_rec_chunks,
        )
        for bucket in plan.completion_warmup_buckets:
            completion_conv = prefix_conv_table.index_select(0, bucket.family_indices)
            completion_rec = prefix_rec_table.index_select(0, bucket.family_indices)
            completion_conv, completion_rec = _couple_parent_states(
                completion_conv, completion_rec
            )
            with _nvtx_range("art_gdn_input_layout_gather_reorder", qkv):
                completion_qkv, completion_beta, completion_g = _gather_bucket_streams(
                    qkv, beta, recurrent_g, bucket
                )
            for (
                column_bucket,
                qkv_col,
                beta_col,
                g_col,
                conv_col,
                rec_col,
            ) in _iter_prepared_bucket_columns(
                bucket,
                completion_qkv,
                completion_beta,
                completion_g,
                completion_conv,
                completion_rec,
            ):
                with _nvtx_range("art_gdn_local_completion_segment", qkv_col):
                    completion_out, _, _ = _run_gdn_prepared_varlen_batch(
                        gdn,
                        qkv_col,
                        beta=beta_col,
                        recurrent_g=g_col,
                        bucket=column_bucket,
                        conv_initial=conv_col,
                        recurrent_initial=rec_col,
                        output_final_state=False,
                    )
                completion_out = _add_autograd_dependency(completion_out, cp_dependency)
                recurrent_output = _scatter_bucket_recurrent_output(
                    recurrent_output, column_bucket, completion_out
                )

    for bucket in plan.local_prefix_buckets:
        with _nvtx_range("art_gdn_input_layout_gather_reorder", qkv):
            prefix_qkv, prefix_beta, prefix_g = _gather_bucket_streams(
                qkv, beta, recurrent_g, bucket
            )
        zero_conv = _zero_conv_state(gdn, gdn_hidden, batch_size=prefix_qkv.shape[0])
        zero_rec = _zero_recurrent_state(
            gdn, gdn_hidden, batch_size=prefix_qkv.shape[0]
        )
        with _nvtx_range("art_gdn_local_prefix_segment", prefix_qkv):
            prefix_out, prefix_conv, prefix_rec = _run_gdn_prepared_varlen_batch(
                gdn,
                prefix_qkv,
                beta=prefix_beta,
                recurrent_g=prefix_g,
                bucket=bucket,
                conv_initial=zero_conv,
                recurrent_initial=zero_rec,
                output_final_state=True,
            )
        if prefix_conv is None or prefix_rec is None:
            raise RuntimeError("local prefix GDN execution must return final states")
        prefix_out = _add_autograd_dependency(prefix_out, cp_dependency)
        prefix_conv = _add_autograd_dependency(prefix_conv, cp_dependency)
        prefix_rec = _add_autograd_dependency(prefix_rec, cp_dependency)
        recurrent_output = _scatter_bucket_recurrent_output(
            recurrent_output, bucket, prefix_out
        )
        prefix_family_chunks.append(bucket.family_indices)
        prefix_conv_chunks.append(prefix_conv)
        prefix_rec_chunks.append(prefix_rec)

    if not prefix_conv_chunks and not plan.parent_state_exchange_family_indices:
        projected, out_bias = _project_gdn_output(gdn, recurrent_output, gate, plan)
        if output_layout == "gdn":
            return projected, out_bias
        return _cp_output_to_attention(projected, plan, original_shape, group), out_bias

    prefix_conv_table = _materialize_ordered_family_state_table(
        family_chunks=prefix_family_chunks,
        state_chunks=prefix_conv_chunks,
        zero_state=_zero_conv_state(gdn, gdn_hidden, batch_size=plan.family_count),
    )
    prefix_rec_table = _materialize_ordered_family_state_table(
        family_chunks=prefix_family_chunks,
        state_chunks=prefix_rec_chunks,
        zero_state=_zero_recurrent_state(gdn, gdn_hidden, batch_size=plan.family_count),
    )
    for bucket in plan.chain_completion_buckets:
        with _nvtx_range("art_gdn_input_layout_gather_reorder", qkv):
            completion_qkv, completion_beta, completion_g = _gather_bucket_streams(
                qkv, beta, recurrent_g, bucket
            )
        completion_conv = prefix_conv_table.index_select(0, bucket.family_indices)
        completion_rec = prefix_rec_table.index_select(0, bucket.family_indices)
        completion_conv, completion_rec = _couple_parent_states(
            completion_conv, completion_rec
        )
        completion_conv = _scale_state_gradient(completion_conv, 1.0 / plan.cp_size)
        completion_rec = _scale_state_gradient(completion_rec, 1.0 / plan.cp_size)
        with _nvtx_range("art_gdn_cp_completion_segment", completion_qkv):
            completion_out, _, _ = run_gdn_prepared_varlen_native_fla_cp(
                gdn,
                completion_qkv,
                beta=completion_beta,
                recurrent_g=completion_g,
                lengths=bucket.lengths,
                cu_seqlens=bucket.cu_seqlens,
                conv_initial=completion_conv,
                recurrent_initial=completion_rec,
                group=group,
                output_final_state=False,
            )
        completion_out = _add_autograd_dependency(completion_out, cp_dependency)
        cp_dependency = _make_autograd_dependency(completion_out)
        recurrent_output = _scatter_bucket_recurrent_output(
            recurrent_output, bucket, completion_out
        )

    ready_completion_buckets = (
        plan.ready_local_completion_buckets
        if plan.ready_local_completion_buckets or plan.remote_local_completion_buckets
        else plan.local_completion_buckets
    )
    for bucket in ready_completion_buckets:
        with _nvtx_range("art_gdn_input_layout_gather_reorder", qkv):
            completion_qkv, completion_beta, completion_g = _gather_bucket_streams(
                qkv, beta, recurrent_g, bucket
            )
        completion_conv = prefix_conv_table.index_select(0, bucket.family_indices)
        completion_rec = prefix_rec_table.index_select(0, bucket.family_indices)
        completion_conv, completion_rec = _couple_parent_states(
            completion_conv, completion_rec
        )
        with _nvtx_range("art_gdn_local_completion_segment", completion_qkv):
            completion_out, _, _ = _run_gdn_prepared_varlen_batch(
                gdn,
                completion_qkv,
                beta=completion_beta,
                recurrent_g=completion_g,
                bucket=bucket,
                conv_initial=completion_conv,
                recurrent_initial=completion_rec,
                output_final_state=False,
            )
        completion_out = _add_autograd_dependency(completion_out, cp_dependency)
        recurrent_output = _scatter_bucket_recurrent_output(
            recurrent_output, bucket, completion_out
        )

    if plan.parent_state_exchange_family_indices:
        if not plan.parent_state_transfers:
            raise ValueError("CP parent-state exchange requires planned transfers")
        with _nvtx_range("art_gdn_cp_parent_state_exchange", prefix_conv_table):
            prefix_conv_table, prefix_rec_table, exchange_dependency = (
                _exchange_parent_state_rows(
                    prefix_conv_table,
                    prefix_rec_table,
                    transfers=plan.parent_state_transfers,
                    group=group,
                )
            )
        cp_dependency = cp_dependency + exchange_dependency

    for bucket in plan.remote_local_completion_buckets:
        with _nvtx_range("art_gdn_input_layout_gather_reorder", qkv):
            completion_qkv, completion_beta, completion_g = _gather_bucket_streams(
                qkv, beta, recurrent_g, bucket
            )
        completion_conv = prefix_conv_table.index_select(0, bucket.family_indices)
        completion_rec = prefix_rec_table.index_select(0, bucket.family_indices)
        completion_conv, completion_rec = _couple_parent_states(
            completion_conv, completion_rec
        )
        with _nvtx_range("art_gdn_local_completion_segment", completion_qkv):
            completion_out, _, _ = _run_gdn_prepared_varlen_batch(
                gdn,
                completion_qkv,
                beta=completion_beta,
                recurrent_g=completion_g,
                bucket=bucket,
                conv_initial=completion_conv,
                recurrent_initial=completion_rec,
                output_final_state=False,
            )
        completion_out = _add_autograd_dependency(completion_out, cp_dependency)
        recurrent_output = _scatter_bucket_recurrent_output(
            recurrent_output, bucket, completion_out
        )

    projected, out_bias = _project_gdn_output(gdn, recurrent_output, gate, plan)
    projected = _add_autograd_dependency(projected, cp_dependency)
    if output_layout == "gdn":
        return projected, out_bias
    return _cp_output_to_attention(projected, plan, original_shape, group), out_bias


@torch.compiler.disable
def gdn_cp_attention_to_gdn_layout(
    hidden_states: Tensor,
    plan: GdnRankExecutionPlan,
    group: Any,
) -> tuple[Tensor, tuple[int, int, int]]:
    from .layout import exchange_rank_tensor_all_to_all

    if plan.attention_to_gdn is None or plan.gdn_to_attention is None:
        raise ValueError("CP GDN layout conversion requires prebuilt exchange plans")
    attention_flat, original_shape = _flatten_hidden_for_cp_plan(hidden_states, plan)
    with _nvtx_range("art_gdn_cp_attention_to_gdn_exchange", attention_flat):
        gdn_flat = exchange_rank_tensor_all_to_all(
            attention_flat,
            plan.attention_to_gdn,
            rank=plan.cp_rank,
            group=group,
            backward_plan=plan.gdn_to_attention,
        )
    return gdn_flat.unsqueeze(1).contiguous(), original_shape


@torch.compiler.disable
def gdn_cp_gdn_to_attention_layout(
    gdn_hidden: Tensor,
    plan: GdnRankExecutionPlan,
    original_shape: tuple[int, int, int] | None,
    group: Any,
) -> Tensor:
    original_shape = original_shape or _attention_original_shape_from_plan(
        gdn_hidden, plan
    )
    return _cp_output_to_attention(gdn_hidden, plan, original_shape, group)


def _enter_gdn_island_layout(
    hidden_states: Tensor, attention_bias: Any, *, force: bool = False
) -> Tensor:
    plan = _require_gdn_cp_plan(attention_bias)
    if not force and getattr(attention_bias, "gdn_hidden_layout", "attention") == "gdn":
        return _validate_gdn_hidden_for_cp_plan(hidden_states, plan)
    gdn_hidden, original_shape = gdn_cp_attention_to_gdn_layout(
        hidden_states,
        plan,
        _default_cp_group(plan.cp_size),
    )
    attention_bias.gdn_hidden_layout = "gdn"
    attention_bias.gdn_attention_original_shape = original_shape
    return gdn_hidden


def _mark_attention_layout_active(attention_bias: Any) -> None:
    attention_bias.gdn_hidden_layout = "attention"
    attention_bias.gdn_attention_original_shape = None


def _leave_gdn_island_layout(hidden_states: Tensor, attention_bias: Any) -> Tensor:
    plan = _require_gdn_cp_plan(attention_bias)
    gdn_hidden = _validate_gdn_hidden_for_cp_plan(hidden_states, plan)
    attention_hidden = gdn_cp_gdn_to_attention_layout(
        gdn_hidden,
        plan,
        getattr(attention_bias, "gdn_attention_original_shape", None),
        _default_cp_group(plan.cp_size),
    )
    _mark_attention_layout_active(attention_bias)
    return attention_hidden


def _mark_gdn_layout_active(attention_bias: Any, hidden_states: Tensor) -> None:
    plan = _require_gdn_cp_plan(attention_bias)
    _validate_gdn_hidden_for_cp_plan(hidden_states, plan)
    attention_bias.gdn_hidden_layout = "gdn"
    if getattr(attention_bias, "gdn_attention_original_shape", None) is None:
        attention_bias.gdn_attention_original_shape = (
            _attention_original_shape_from_plan(hidden_states, plan)
        )


def _require_gdn_cp_plan(attention_bias: Any) -> GdnRankExecutionPlan:
    plan = getattr(attention_bias, "gdn_execution_plan", None)
    if plan is None or int(getattr(plan, "cp_size", 1)) <= 1:
        raise ValueError("GDN island layout conversion requires a CP execution plan")
    return cast(GdnRankExecutionPlan, plan)


def _cp_output_to_attention(
    gdn_output: Tensor,
    plan: GdnRankExecutionPlan,
    original_shape: tuple[int, int, int],
    group: Any,
) -> Tensor:
    from .layout import exchange_rank_tensor_all_to_all

    if plan.gdn_to_attention is None:
        raise ValueError("CP GDN execution requires a GDN-to-attention exchange plan")
    gdn_flat = gdn_output.squeeze(1).contiguous()
    with _nvtx_range("art_gdn_cp_gdn_to_attention_exchange", gdn_flat):
        attention_flat = exchange_rank_tensor_all_to_all(
            gdn_flat,
            plan.gdn_to_attention,
            rank=plan.cp_rank,
            group=group,
            backward_plan=plan.attention_to_gdn,
        )
    return _restore_hidden_from_cp_flat(attention_flat, original_shape)


def _flatten_hidden_for_cp_plan(
    hidden_states: Tensor, plan: GdnRankExecutionPlan
) -> tuple[Tensor, tuple[int, int, int]]:
    seq_len, batch_size, hidden_size = hidden_states.shape
    flat = hidden_states.transpose(0, 1).reshape(seq_len * batch_size, hidden_size)
    expected = int(plan.attention_token_count)
    if int(flat.shape[0]) != expected:
        raise ValueError(
            "CP GDN hidden token count must match the rank-local attention plan, "
            f"got {int(flat.shape[0])} tokens and expected {expected}"
        )
    return flat.contiguous(), (seq_len, batch_size, hidden_size)


def _validate_gdn_hidden_for_cp_plan(
    hidden_states: Tensor, plan: GdnRankExecutionPlan
) -> Tensor:
    expected = int(plan.gdn_token_count)
    if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != expected:
        raise ValueError(
            "CP GDN-layout hidden_states must be [rank_gdn_tokens, 1, D], "
            f"got {tuple(hidden_states.shape)} for {expected} planned tokens"
        )
    if int(hidden_states.shape[1]) != 1:
        raise ValueError(
            "CP GDN-layout hidden_states must use a flattened local batch, "
            f"got batch dimension {int(hidden_states.shape[1])}"
        )
    return hidden_states.contiguous()


def _attention_original_shape_from_plan(
    hidden_states: Tensor, plan: GdnRankExecutionPlan
) -> tuple[int, int, int]:
    return (int(plan.attention_token_count), 1, int(hidden_states.shape[-1]))


def _restore_hidden_from_cp_flat(
    flat: Tensor, original_shape: tuple[int, int, int]
) -> Tensor:
    seq_len, batch_size, hidden_size = original_shape
    if int(flat.shape[0]) != seq_len * batch_size:
        raise ValueError(
            "CP GDN output token count changed across layout exchange, got "
            f"{int(flat.shape[0])} for original shape {original_shape}"
        )
    return flat.reshape(batch_size, seq_len, hidden_size).transpose(0, 1).contiguous()


def _empty_autograd_dependency(reference: Tensor) -> Tensor:
    return reference.new_zeros(())


def _make_autograd_dependency(*tensors: Tensor | None) -> Tensor:
    dependency: Tensor | None = None
    for tensor in tensors:
        if tensor is None or int(tensor.numel()) == 0:
            continue
        piece = tensor.reshape(-1)[:1].sum() * 0
        dependency = piece if dependency is None else dependency + piece
    if dependency is None:
        raise ValueError("at least one non-empty tensor is required")
    return dependency


def _add_autograd_dependency(tensor: Tensor, dependency: Tensor) -> Tensor:
    return tensor + dependency.to(dtype=tensor.dtype)


def _couple_parent_states(
    conv_state: Tensor, recurrent_state: Tensor
) -> tuple[Tensor, Tensor]:
    return _CoupledParentStates.apply(conv_state, recurrent_state)


class _CoupledParentStates(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, conv_state: Tensor, recurrent_state: Tensor
    ) -> tuple[Tensor, Tensor]:
        del ctx
        return conv_state, recurrent_state

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None]:
        del ctx
        grad_conv, grad_recurrent = grad_outputs
        return grad_conv, grad_recurrent


def _scale_state_gradient(tensor: Tensor, scale: float) -> Tensor:
    return _ScaleStateGradient.apply(tensor, scale)


class _ScaleStateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor | None) -> tuple[Tensor | None, None]:
        (grad_output,) = grad_outputs
        if grad_output is None:
            return None, None
        return grad_output * ctx.scale, None


def _gather_flat_bucket_streams(
    qkv_flat: Tensor,
    beta_flat: Tensor,
    recurrent_g_flat: Tensor,
    *,
    layout: _BucketFlatLayout,
    length: int,
    segment_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return _FlatBucketStreamGather.apply(
        qkv_flat,
        beta_flat,
        recurrent_g_flat,
        layout.padded_indices,
        layout.padded_mask,
        length,
        segment_count,
    )


def _gather_compact_bucket_streams(
    qkv: Tensor,
    beta: Tensor,
    recurrent_g: Tensor,
    bucket: GdnSegmentBucketPlan,
) -> tuple[Tensor, Tensor, Tensor]:
    return _gather_bucket_streams_compact_fused(
        qkv.reshape(-1, int(qkv.shape[-1])),
        beta.reshape(-1, int(beta.shape[-1])),
        recurrent_g.reshape(-1, int(recurrent_g.shape[-1])),
        bucket.row_indices,
        bucket.position_indices,
        bucket.cu_seqlens,
        token_count=int(bucket.real_token_count),
        segment_count=int(bucket.segment_count),
        sequence_length=int(qkv.shape[1]),
    )


class _FlatBucketStreamGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        qkv_flat: Tensor,
        beta_flat: Tensor,
        recurrent_g_flat: Tensor,
        padded_indices: Tensor,
        padded_mask: Tensor,
        length: int,
        segment_count: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        flat_indices = padded_indices.reshape(-1)
        flat_mask = padded_mask.reshape(-1)
        safe_indices = torch.where(
            flat_mask,
            flat_indices,
            torch.zeros((), device=flat_indices.device, dtype=flat_indices.dtype),
        )
        qkv = qkv_flat.index_select(0, safe_indices).reshape(
            length, segment_count, int(qkv_flat.shape[-1])
        )
        beta = beta_flat.index_select(0, safe_indices).reshape(
            length, segment_count, int(beta_flat.shape[-1])
        )
        recurrent_g = recurrent_g_flat.index_select(0, safe_indices).reshape(
            length, segment_count, int(recurrent_g_flat.shape[-1])
        )
        qkv = qkv.masked_fill(~padded_mask.unsqueeze(-1), 0)
        beta = beta.masked_fill(~padded_mask.unsqueeze(-1), 0)
        recurrent_g = recurrent_g.masked_fill(~padded_mask.unsqueeze(-1), 0)
        ctx.save_for_backward(safe_indices, flat_mask)
        ctx.qkv_flat_count = int(qkv_flat.shape[0])
        ctx.beta_flat_count = int(beta_flat.shape[0])
        ctx.recurrent_g_flat_count = int(recurrent_g_flat.shape[0])
        return (
            qkv.permute(1, 2, 0).contiguous(),
            beta.transpose(0, 1).contiguous(),
            recurrent_g.transpose(0, 1).contiguous(),
        )

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, None, None, None, None]:
        grad_qkv_bucket, grad_beta_bucket, grad_g_bucket = grad_outputs
        safe_indices, flat_mask = ctx.saved_tensors
        grad_qkv = (
            _bucket_stream_grad_to_flat(
                grad_qkv_bucket.permute(2, 0, 1).contiguous()
                if grad_qkv_bucket is not None
                else None,
                safe_indices,
                flat_mask,
                ctx.qkv_flat_count,
            )
            if ctx.needs_input_grad[0]
            else None
        )
        grad_beta = (
            _bucket_stream_grad_to_flat(
                grad_beta_bucket.transpose(0, 1).contiguous()
                if grad_beta_bucket is not None
                else None,
                safe_indices,
                flat_mask,
                ctx.beta_flat_count,
            )
            if ctx.needs_input_grad[1]
            else None
        )
        grad_g = (
            _bucket_stream_grad_to_flat(
                grad_g_bucket.transpose(0, 1).contiguous()
                if grad_g_bucket is not None
                else None,
                safe_indices,
                flat_mask,
                ctx.recurrent_g_flat_count,
            )
            if ctx.needs_input_grad[2]
            else None
        )
        return grad_qkv, grad_beta, grad_g, None, None, None, None


def _bucket_stream_grad_to_flat(
    grad: Tensor | None,
    safe_indices: Tensor,
    flat_mask: Tensor,
    flat_count: int,
) -> Tensor | None:
    if grad is None:
        return None
    grad_flat_values = grad.reshape(int(safe_indices.numel()), int(grad.shape[-1]))
    grad_flat_values = grad_flat_values.masked_fill(~flat_mask.unsqueeze(-1), 0)
    grad_flat = grad.new_zeros(flat_count, int(grad.shape[-1]))
    return grad_flat.index_add(0, safe_indices, grad_flat_values)


def _scatter_compact_hidden(
    compact: Tensor,
    indices: Tensor,
    *,
    batch_size: int,
    sequence_length: int,
) -> Tensor:
    return _CompactHiddenScatter.apply(compact, indices, batch_size, sequence_length)


class _CompactHiddenScatter(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        compact: Tensor,
        indices: Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> Tensor:
        hidden_size = int(compact.shape[-1])
        flat = compact.new_zeros(batch_size * sequence_length, hidden_size)
        if int(indices.numel()):
            flat = flat.index_copy(0, indices, compact.reshape(-1, hidden_size))
        ctx.save_for_backward(indices)
        ctx.batch_size = batch_size
        ctx.sequence_length = sequence_length
        return (
            flat.reshape(batch_size, sequence_length, hidden_size)
            .transpose(0, 1)
            .contiguous()
        )

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor | None
    ) -> tuple[Tensor | None, None, None, None]:
        if grad_output is None:
            return None, None, None, None
        (indices,) = ctx.saved_tensors
        flat_grad = grad_output.transpose(0, 1).reshape(
            ctx.batch_size * ctx.sequence_length, int(grad_output.shape[-1])
        )
        return flat_grad.index_select(0, indices), None, None, None


def _project_gdn_inputs(
    gdn: Any, hidden_states: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    seq_len, batch_size, _ = hidden_states.shape
    seq_len *= int(getattr(gdn, "sp_size", 1))
    qkvzba, _ = _in_proj(gdn, hidden_states)
    qkvzba = qkvzba.transpose(0, 1)
    qkv, gate, beta, alpha = torch.split(
        qkvzba,
        [
            (gdn.qk_dim * 2 + gdn.v_dim) // gdn.tp_size,
            gdn.v_dim // gdn.tp_size,
            gdn.num_value_heads // gdn.tp_size,
            gdn.num_value_heads // gdn.tp_size,
        ],
        dim=-1,
    )
    value_heads = _local_value_heads(gdn)
    gate = gate.reshape(
        batch_size, seq_len, value_heads, gdn.value_head_dim
    ).contiguous()
    beta = beta.reshape(batch_size, seq_len, value_heads).sigmoid().contiguous()
    alpha = alpha.reshape(batch_size, seq_len, value_heads)
    recurrent_g = (
        -gdn.A_log.exp() * F.softplus(alpha.float() + gdn.dt_bias)
    ).contiguous()
    return qkv.contiguous(), gate, beta, recurrent_g


def _in_proj(gdn: Any, hidden_states: Tensor) -> tuple[Tensor, Tensor | None]:
    return gdn.in_proj(hidden_states)


def _gather_bucket_streams(
    qkv: Tensor,
    beta: Tensor,
    recurrent_g: Tensor,
    bucket: GdnSegmentBucketPlan,
) -> tuple[Tensor, Tensor, Tensor]:
    layout = _bucket_flat_layout(
        bucket,
        sequence_length=int(qkv.shape[1]),
    )
    return _gather_flat_bucket_streams(
        qkv.reshape(-1, int(qkv.shape[-1])),
        beta.reshape(-1, int(beta.shape[-1])),
        recurrent_g.reshape(-1, int(recurrent_g.shape[-1])),
        layout=layout,
        length=int(bucket.length),
        segment_count=int(bucket.segment_count),
    )


def _bucket_flat_layout(
    bucket: GdnSegmentBucketPlan, *, sequence_length: int
) -> _BucketFlatLayout:
    positions = bucket.position_indices.clamp_max(sequence_length - 1)
    padded_indices = (bucket.row_indices * sequence_length + positions).contiguous()
    padded_mask = bucket.real_mask.contiguous()
    segment_major_indices = padded_indices.transpose(0, 1).contiguous()
    segment_major_mask = padded_mask.transpose(0, 1).contiguous()
    real_indices = segment_major_indices[segment_major_mask].contiguous()
    output_mask = _bucket_output_mask(bucket).transpose(0, 1).contiguous()
    output_indices = segment_major_indices[output_mask].contiguous()
    output_selector = None
    if bucket.output_mask is not None:
        output_selector = output_mask[segment_major_mask].contiguous()
    return _BucketFlatLayout(
        padded_indices=padded_indices,
        padded_mask=padded_mask,
        real_indices=real_indices,
        output_indices=output_indices,
        output_selector=output_selector,
    )


def _project_gdn_output(
    gdn: Any,
    recurrent_output: Tensor,
    gate: Tensor,
    plan: GdnRankExecutionPlan,
) -> tuple[Tensor, Tensor | None]:
    batch_size, seq_len, _, _ = recurrent_output.shape
    with _nvtx_range("art_gdn_output_norm_gate", recurrent_output):
        norm_out = _apply_gated_rms_norm(gdn, recurrent_output, gate)
        norm_out = norm_out.reshape(batch_size, seq_len, _local_value_dim(gdn))
        norm_out = norm_out.transpose(0, 1).contiguous()
    with _nvtx_range("art_gdn_out_proj", norm_out):
        if plan.cp_size > 1:
            out, out_bias = _out_proj_cp_full_shape(gdn, norm_out, plan)
        else:
            out, out_bias = _out_proj(gdn, norm_out)
    return _mask_gdn_output(gdn, out, plan), out_bias


def _mask_gdn_output(gdn: Any, out: Tensor, plan: GdnRankExecutionPlan) -> Tensor:
    real_mask = plan.real_token_mask.transpose(0, 1).unsqueeze(-1)
    if tuple(real_mask.shape[:2]) == tuple(out.shape[:2]):
        return out.masked_fill(~real_mask, 0)
    full_batch = int(plan.packed_batch_size or plan.batch_size)
    full_seq = int(plan.packed_sequence_length or plan.sequence_length)
    full_count = full_batch * full_seq
    local_indices = torch.tensor(
        plan.gdn_token_indices, device=out.device, dtype=torch.long
    )
    full_flat = torch.zeros(full_count, device=out.device, dtype=torch.bool)
    if int(local_indices.numel()):
        full_flat = full_flat.index_fill(0, local_indices, True)
    full_mask = full_flat.reshape(full_batch, full_seq).transpose(0, 1).unsqueeze(-1)
    if tuple(full_mask.shape[:2]) == tuple(out.shape[:2]):
        return out.masked_fill(~full_mask, 0)
    rank = _tp_rank(getattr(gdn.out_proj, "linear_proj", gdn.out_proj))
    start = rank * int(out.shape[0])
    end = start + int(out.shape[0])
    if end <= int(full_mask.shape[0]) and int(full_mask.shape[1]) == int(out.shape[1]):
        return out.masked_fill(~full_mask[start:end], 0)
    raise ValueError(
        "GDN output mask shape must match projected output, got "
        f"mask={tuple(real_mask.shape)} full_mask={tuple(full_mask.shape)} "
        f"out={tuple(out.shape)}"
    )


def _out_proj_cp_full_shape(
    gdn: Any, hidden_states: Tensor, plan: GdnRankExecutionPlan
) -> tuple[Tensor, Tensor | None]:
    full_batch = int(plan.packed_batch_size or plan.batch_size)
    full_seq = int(plan.packed_sequence_length or plan.sequence_length)
    full_count = full_batch * full_seq
    if full_count == int(hidden_states.shape[0]):
        return _out_proj(gdn, hidden_states)
    if int(hidden_states.shape[1]) != 1:
        raise ValueError(
            "CP GDN full-shape output projection expects flattened local batch, got "
            f"{tuple(hidden_states.shape)}"
        )
    local_indices = torch.tensor(
        plan.gdn_token_indices, device=hidden_states.device, dtype=torch.long
    )
    if int(local_indices.numel()) != int(hidden_states.shape[0]):
        raise ValueError(
            "CP GDN token index count must match local projection input, got "
            f"{int(local_indices.numel())} indices for {tuple(hidden_states.shape)}"
        )
    if int(local_indices.numel()) and int(local_indices.max().item()) >= full_count:
        raise ValueError(
            "CP GDN token index exceeds packed output shape, got "
            f"max_index={int(local_indices.max().item())} full_count={full_count}"
        )
    full_flat = hidden_states.new_zeros(full_count, int(hidden_states.shape[-1]))
    if int(local_indices.numel()):
        full_flat = full_flat.index_copy(0, local_indices, hidden_states.squeeze(1))
    full_hidden = (
        full_flat.reshape(full_batch, full_seq, int(hidden_states.shape[-1]))
        .transpose(0, 1)
        .contiguous()
    )
    full_out, out_bias = _out_proj(gdn, full_hidden)
    local_out = (
        full_out.transpose(0, 1)
        .reshape(full_count, int(full_out.shape[-1]))
        .index_select(0, local_indices)
        .unsqueeze(1)
        .contiguous()
    )
    return local_out, out_bias


def _apply_gated_rms_norm(gdn: Any, x: Tensor, gate: Tensor) -> Tensor:
    x_dtype = x.dtype
    hidden = gdn.out_norm(x.reshape(-1, int(x.shape[-1])))
    gate = gate.reshape(-1, int(gate.shape[-1]))
    return (hidden * gdn.act_fn(gate.float())).to(x_dtype)


def _out_proj(gdn: Any, hidden_states: Tensor) -> tuple[Tensor, Tensor | None]:
    return gdn.out_proj(hidden_states)


def _apply_explicit_norm(
    module: Any,
    x: Tensor,
    *,
    config: Any,
    weight_name: str,
    bias_name: str,
) -> Tensor:
    del config
    x_dtype = x.dtype
    x_float = x.float()
    normalization = str(module.normalization)
    if normalization == "RMSNorm":
        normed = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + float(module.eps)
        )
        bias = None
    elif normalization == "LayerNorm":
        centered = x_float - x_float.mean(dim=-1, keepdim=True)
        normed = centered * torch.rsqrt(
            centered.square().mean(dim=-1, keepdim=True) + float(module.eps)
        )
        bias = getattr(module, bias_name)
    else:
        raise ValueError(f"unsupported GDN normalization '{normalization}'")

    scale = getattr(module, weight_name).float()
    if bool(module.zero_centered_gamma):
        scale = scale + 1.0
    normed = normed * scale
    if isinstance(bias, Tensor):
        normed = normed + bias.float()
    return normed.to(dtype=x_dtype)


def _uses_sequence_parallel(projection: Any) -> bool:
    return bool(getattr(projection, "sequence_parallel", False)) and (
        _tp_world_size(projection) > 1
    )


def _gdn_uses_sequence_parallel(gdn: Any) -> bool:
    projection = getattr(gdn, "in_proj", None)
    base_projection = getattr(projection, "in_proj", projection)
    return _uses_sequence_parallel(base_projection)


def _tp_world_size(projection: Any) -> int:
    del projection
    from megatron.core import parallel_state as ps

    return int(ps.get_tensor_model_parallel_world_size())


def _tp_rank(projection: Any) -> int:
    del projection
    from megatron.core import parallel_state as ps

    return int(ps.get_tensor_model_parallel_rank())


def _local_key_heads(gdn: Any) -> int:
    return int(gdn.num_key_heads // gdn.tp_size)


def _local_value_heads(gdn: Any) -> int:
    return int(gdn.num_value_heads // gdn.tp_size)


def _local_value_dim(gdn: Any) -> int:
    return _local_value_heads(gdn) * int(gdn.value_head_dim)


def _scatter_bucket_recurrent_output(
    output: Tensor, bucket: GdnSegmentBucketPlan, bucket_output: Tensor
) -> Tensor:
    return _scatter_bucket_output_fused(
        output,
        bucket_output,
        bucket.row_indices,
        bucket.position_indices,
        _bucket_output_mask(bucket),
        bucket.cu_seqlens,
    )


def _bucket_output_mask(bucket: GdnSegmentBucketPlan) -> Tensor:
    output_mask = bucket.output_mask
    return bucket.real_mask if output_mask is None else output_mask


def _materialize_indexed_family_state_table(
    *,
    plan: GdnRankExecutionPlan,
    family_chunks: list[Tensor],
    state_chunks: list[Tensor],
    zero_state: Tensor,
) -> Tensor:
    table = zero_state.detach()
    if not state_chunks:
        return table.requires_grad_(True)
    values = torch.cat(state_chunks, dim=0)
    family_indices = torch.cat(family_chunks, dim=0)
    return table.index_copy(0, family_indices, values)


def _materialize_ordered_family_state_table(
    *,
    family_chunks: list[Tensor],
    state_chunks: list[Tensor],
    zero_state: Tensor,
) -> Tensor:
    if len(family_chunks) != len(state_chunks):
        raise RuntimeError("family and state chunk counts must match")
    table = zero_state.detach().requires_grad_(True)
    for family_indices, states in zip(family_chunks, state_chunks, strict=True):
        table = table.index_copy(0, family_indices, states)
    return table


def _replace_indexed_family_states(
    table: Tensor,
    *,
    family_chunks: list[Tensor],
    state_chunks: list[Tensor],
) -> Tensor:
    if not state_chunks:
        return table
    return table.index_copy(
        0,
        torch.cat(family_chunks, dim=0),
        torch.cat(state_chunks, dim=0),
    )


def _exchange_parent_state_rows(
    conv_table: Tensor,
    rec_table: Tensor,
    *,
    transfers: tuple[GdnParentStateTransferPlan, ...],
    group: Any,
) -> tuple[Tensor, Tensor, Tensor]:
    if not transfers:
        return conv_table, rec_table, _empty_autograd_dependency(conv_table)
    conv_table, rec_table = _ParentStateExchange.apply(
        conv_table, rec_table, transfers, group
    )
    return conv_table, rec_table, _make_autograd_dependency(conv_table, rec_table)


class _ParentStateExchange(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        conv_table: Tensor,
        rec_table: Tensor,
        transfers: tuple[GdnParentStateTransferPlan, ...],
        group: Any,
    ) -> tuple[Tensor, Tensor]:
        ctx.group = group
        ctx.transfers = transfers
        ctx.save_for_backward(conv_table, rec_table)
        return (
            _exchange_parent_state_tensor_forward(
                conv_table,
                transfers,
                group=group,
            ),
            _exchange_parent_state_tensor_forward(
                rec_table,
                transfers,
                group=group,
            ),
        )

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None, None, None]:
        grad_conv, grad_rec = grad_outputs
        conv_ref, rec_ref = ctx.saved_tensors
        return (
            _exchange_parent_state_tensor_backward(
                _zero_if_none(grad_conv, conv_ref),
                ctx.transfers,
                group=ctx.group,
            ),
            _exchange_parent_state_tensor_backward(
                _zero_if_none(grad_rec, rec_ref),
                ctx.transfers,
                group=ctx.group,
            ),
            None,
            None,
        )


def _exchange_parent_state_tensor_forward(
    table: Tensor,
    transfers: tuple[GdnParentStateTransferPlan, ...],
    *,
    group: Any,
) -> Tensor:
    rank = torch.distributed.get_rank(group)  # ty: ignore[possibly-missing-attribute]
    output = table.clone()
    recvs = _exchange_parent_state_rows_all_to_all(
        table, transfers, rank=rank, reverse=False, group=group
    )
    for transfer, rows in recvs:
        index = _parent_state_index_tensor(transfer, device=table.device)
        output.index_copy_(0, index, rows)
    return output


def _exchange_parent_state_tensor_backward(
    grad_output: Tensor,
    transfers: tuple[GdnParentStateTransferPlan, ...],
    *,
    group: Any,
) -> Tensor:
    rank = torch.distributed.get_rank(group)  # ty: ignore[possibly-missing-attribute]
    grad_input = grad_output.clone()
    for transfer in transfers:
        if transfer.dest_rank != rank:
            continue
        index = _parent_state_index_tensor(transfer, device=grad_output.device)
        grad_input.index_fill_(0, index, 0)
    recvs = _exchange_parent_state_rows_all_to_all(
        grad_output, transfers, rank=rank, reverse=True, group=group
    )
    for transfer, rows in recvs:
        index = _parent_state_index_tensor(transfer, device=grad_output.device)
        grad_input.index_add_(0, index, rows)
    return grad_input


def _zero_if_none(grad: Tensor | None, reference: Tensor) -> Tensor:
    if grad is None:
        return reference.new_zeros(reference.shape)
    return grad.contiguous()


def _exchange_parent_state_rows_all_to_all(
    table: Tensor,
    transfers: tuple[GdnParentStateTransferPlan, ...],
    *,
    rank: int,
    reverse: bool,
    group: Any,
) -> list[tuple[GdnParentStateTransferPlan, Tensor]]:
    world_size = torch.distributed.get_world_size(group)  # ty: ignore[possibly-missing-attribute]
    send_counts = [0 for _ in range(world_size)]
    recv_counts = [0 for _ in range(world_size)]
    send_pieces: list[Tensor] = []
    for peer_rank in range(world_size):
        for transfer in transfers:
            send_rank = transfer.dest_rank if reverse else transfer.source_rank
            recv_rank = transfer.source_rank if reverse else transfer.dest_rank
            if send_rank == recv_rank:
                continue
            row_count = len(transfer.family_indices)
            if rank == send_rank and peer_rank == recv_rank:
                index = _parent_state_index_tensor(transfer, device=table.device)
                send_pieces.append(table.index_select(0, index).contiguous())
                send_counts[peer_rank] += row_count
            if rank == recv_rank and peer_rank == send_rank:
                recv_counts[peer_rank] += row_count

    trailing_shape = tuple(table.shape[1:])
    send_buffer = (
        torch.cat(send_pieces, dim=0)
        if send_pieces
        else table.new_empty((0, *trailing_shape))
    )
    recv_buffer = table.new_empty((sum(recv_counts), *trailing_shape))
    work = torch.distributed.all_to_all_single(  # ty: ignore[possibly-missing-attribute]
        recv_buffer,
        send_buffer,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
        group=group,
        async_op=True,
    )
    work.wait()

    recvs: list[tuple[GdnParentStateTransferPlan, Tensor]] = []
    offset = 0
    for peer_rank, count in enumerate(recv_counts):
        peer_end = offset + count
        for transfer in transfers:
            send_rank = transfer.dest_rank if reverse else transfer.source_rank
            recv_rank = transfer.source_rank if reverse else transfer.dest_rank
            if send_rank == recv_rank:
                continue
            if rank != recv_rank or peer_rank != send_rank:
                continue
            rows = len(transfer.family_indices)
            recvs.append((transfer, recv_buffer[offset : offset + rows]))
            offset += rows
        if offset != peer_end:
            raise RuntimeError(
                "parent-state exchange unpack mismatch: "
                f"rank={rank} peer={peer_rank} consumed={offset} expected={peer_end}"
            )
    return recvs


def _parent_state_index_tensor(
    transfer: GdnParentStateTransferPlan,
    *,
    device: torch.device,
) -> Tensor:
    if (
        transfer.family_indices_tensor is not None
        and transfer.family_indices_tensor.device == device
    ):
        return transfer.family_indices_tensor
    return torch.tensor(transfer.family_indices, device=device, dtype=torch.long)


def _run_gdn_segment(
    gdn: Any,
    hidden_states: Tensor,
    *,
    conv_initial: Tensor,
    recurrent_initial: Tensor,
    output_final_state: bool = True,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    _disable_reentrant_te_linear_transpose_cache(gdn)
    seq_len, batch_size, _ = hidden_states.shape
    if int(conv_initial.shape[0]) != batch_size:
        raise ValueError(
            "conv_initial batch must match hidden_states batch, got "
            f"{tuple(conv_initial.shape)} for hidden {tuple(hidden_states.shape)}"
        )
    if int(recurrent_initial.shape[0]) != batch_size:
        raise ValueError(
            "recurrent_initial batch must match hidden_states batch, got "
            f"{tuple(recurrent_initial.shape)} for hidden {tuple(hidden_states.shape)}"
        )

    with _nvtx_range("art_gdn_in_proj", hidden_states):
        qkvzba, _ = _in_proj(gdn, hidden_states)
        qkvzba = qkvzba.transpose(0, 1)

    with _nvtx_range("art_gdn_qkv_gate_beta_alpha_split_reshape", qkvzba):
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (gdn.qk_dim * 2 + gdn.v_dim) // gdn.tp_size,
                gdn.v_dim // gdn.tp_size,
                gdn.num_value_heads // gdn.tp_size,
                gdn.num_value_heads // gdn.tp_size,
            ],
            dim=-1,
        )
        key_heads = _local_key_heads(gdn)
        value_heads = _local_value_heads(gdn)
        gate = gate.reshape(batch_size, seq_len, value_heads, gdn.value_head_dim)
        beta = beta.reshape(batch_size, seq_len, value_heads)
        alpha = alpha.reshape(batch_size, seq_len, value_heads)

    with _nvtx_range("art_gdn_causal_conv_forward", qkv):
        qkv = qkv.transpose(1, 2)
        qkv, conv_final = _causal_conv1d_with_state(
            gdn,
            qkv,
            conv_initial,
            output_final_state=output_final_state,
        )
        qkv = qkv.transpose(1, 2)

    with _nvtx_range("art_gdn_qkv_head_prepare", qkv):
        query, key, value = torch.split(
            qkv,
            [
                gdn.qk_dim // gdn.tp_size,
                gdn.qk_dim // gdn.tp_size,
                gdn.v_dim // gdn.tp_size,
            ],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, key_heads, gdn.key_head_dim)
        key = key.reshape(batch_size, seq_len, key_heads, gdn.key_head_dim)
        value = value.reshape(batch_size, seq_len, value_heads, gdn.value_head_dim)
        if gdn.use_qk_l2norm:
            query = _l2norm(query.contiguous())
            key = _l2norm(key.contiguous())
        if gdn.num_value_heads // gdn.num_key_heads > 1:
            repeat = gdn.num_value_heads // gdn.num_key_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    gate = gate.contiguous()
    beta = beta.contiguous()
    alpha = alpha.contiguous()

    with _nvtx_range("art_gdn_recurrent_gate_prepare", alpha):
        g = -gdn.A_log.exp() * F.softplus(alpha.float() + gdn.dt_bias)
        beta = beta.sigmoid()

    with _nvtx_range("art_gdn_recurrent_forward", query):
        recurrent_out, recurrent_final = _chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_initial,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
        )

    with _nvtx_range("art_gdn_output_norm_gate", recurrent_out):
        norm_out = _apply_gated_rms_norm(gdn, recurrent_out, gate)
        norm_out = norm_out.reshape(batch_size, seq_len, _local_value_dim(gdn))
        norm_out = norm_out.transpose(0, 1).contiguous()
    with _nvtx_range("art_gdn_out_proj", norm_out):
        out, out_bias = _out_proj(gdn, norm_out)
    return out, out_bias, conv_final, recurrent_final


def _run_gdn_prepared_varlen_batch(
    gdn: Any,
    qkv: Tensor,
    *,
    beta: Tensor,
    recurrent_g: Tensor,
    bucket: GdnSegmentBucketPlan,
    conv_initial: Tensor,
    recurrent_initial: Tensor,
    output_final_state: bool = True,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    _disable_reentrant_te_linear_transpose_cache(gdn)
    batch_size, _, max_len = qkv.shape
    if int(bucket.length) != max_len or int(bucket.segment_count) != batch_size:
        raise ValueError(
            "GDN prepared varlen bucket shape mismatch, got "
            f"qkv={tuple(qkv.shape)} bucket_len={bucket.length} "
            f"segments={bucket.segment_count}"
        )
    if int(conv_initial.shape[0]) != batch_size:
        raise ValueError(
            "conv_initial batch must match bucket segment count, got "
            f"{tuple(conv_initial.shape)} for {batch_size} segments"
        )
    if int(recurrent_initial.shape[0]) != batch_size:
        raise ValueError(
            "recurrent_initial batch must match bucket segment count, got "
            f"{tuple(recurrent_initial.shape)} for {batch_size} segments"
        )

    with _nvtx_range("art_gdn_causal_conv_forward", qkv):
        qkv, conv_final = _causal_conv1d_varlen_with_state(
            gdn,
            qkv,
            conv_initial,
            bucket.lengths,
            output_final_state=output_final_state,
        )
        qkv = qkv.transpose(1, 2)

    with _nvtx_range("art_gdn_qkv_head_prepare", qkv):
        query, key, value = torch.split(
            qkv,
            [
                gdn.qk_dim // gdn.tp_size,
                gdn.qk_dim // gdn.tp_size,
                gdn.v_dim // gdn.tp_size,
            ],
            dim=-1,
        )
        key_heads = _local_key_heads(gdn)
        value_heads = _local_value_heads(gdn)
        query = query.reshape(batch_size, max_len, key_heads, gdn.key_head_dim)
        key = key.reshape(batch_size, max_len, key_heads, gdn.key_head_dim)
        value = value.reshape(batch_size, max_len, value_heads, gdn.value_head_dim)
        if gdn.use_qk_l2norm:
            query = _l2norm(query.contiguous())
            key = _l2norm(key.contiguous())
        if gdn.num_value_heads // gdn.num_key_heads > 1:
            repeat = gdn.num_value_heads // gdn.num_key_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

    real_mask = bucket.real_mask.transpose(0, 1)
    query = query[real_mask].unsqueeze(0).contiguous()
    key = key[real_mask].unsqueeze(0).contiguous()
    value = value[real_mask].unsqueeze(0).contiguous()
    beta = beta[real_mask].unsqueeze(0).contiguous()
    recurrent_g = recurrent_g[real_mask].unsqueeze(0).contiguous()

    with _nvtx_range("art_gdn_recurrent_forward", query):
        recurrent_out, recurrent_final = _chunk_gated_delta_rule(
            query,
            key,
            value,
            g=recurrent_g,
            beta=beta,
            initial_state=recurrent_initial,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=bucket.cu_seqlens,
        )
    return recurrent_out, conv_final, recurrent_final


def _run_gdn_varlen_batch(
    gdn: Any,
    hidden_states: Tensor,
    *,
    bucket: GdnSegmentBucketPlan,
    conv_initial: Tensor,
    recurrent_initial: Tensor,
    output_final_state: bool = True,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    _disable_reentrant_te_linear_transpose_cache(gdn)
    max_len, batch_size, _ = hidden_states.shape
    if int(bucket.length) != max_len or int(bucket.segment_count) != batch_size:
        raise ValueError(
            "GDN varlen bucket shape mismatch, got "
            f"hidden={tuple(hidden_states.shape)} bucket_len={bucket.length} "
            f"segments={bucket.segment_count}"
        )
    if int(conv_initial.shape[0]) != batch_size:
        raise ValueError(
            "conv_initial batch must match bucket segment count, got "
            f"{tuple(conv_initial.shape)} for {batch_size} segments"
        )
    if int(recurrent_initial.shape[0]) != batch_size:
        raise ValueError(
            "recurrent_initial batch must match bucket segment count, got "
            f"{tuple(recurrent_initial.shape)} for {batch_size} segments"
        )

    with _nvtx_range("art_gdn_in_proj", hidden_states):
        qkvzba, _ = _in_proj(gdn, hidden_states)
        qkvzba = qkvzba.transpose(0, 1)

    with _nvtx_range("art_gdn_qkv_gate_beta_alpha_split_reshape", qkvzba):
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (gdn.qk_dim * 2 + gdn.v_dim) // gdn.tp_size,
                gdn.v_dim // gdn.tp_size,
                gdn.num_value_heads // gdn.tp_size,
                gdn.num_value_heads // gdn.tp_size,
            ],
            dim=-1,
        )
        key_heads = _local_key_heads(gdn)
        value_heads = _local_value_heads(gdn)
        gate = gate.reshape(batch_size, max_len, value_heads, gdn.value_head_dim)
        beta = beta.reshape(batch_size, max_len, value_heads)
        alpha = alpha.reshape(batch_size, max_len, value_heads)

    with _nvtx_range("art_gdn_causal_conv_forward", qkv):
        qkv = qkv.transpose(1, 2).contiguous()
        qkv, conv_final = _causal_conv1d_varlen_with_state(
            gdn,
            qkv,
            conv_initial,
            bucket.lengths,
            output_final_state=output_final_state,
        )
        qkv = qkv.transpose(1, 2)

    with _nvtx_range("art_gdn_qkv_head_prepare", qkv):
        query, key, value = torch.split(
            qkv,
            [
                gdn.qk_dim // gdn.tp_size,
                gdn.qk_dim // gdn.tp_size,
                gdn.v_dim // gdn.tp_size,
            ],
            dim=-1,
        )
        query = query.reshape(batch_size, max_len, key_heads, gdn.key_head_dim)
        key = key.reshape(batch_size, max_len, key_heads, gdn.key_head_dim)
        value = value.reshape(batch_size, max_len, value_heads, gdn.value_head_dim)
        if gdn.use_qk_l2norm:
            query = _l2norm(query.contiguous())
            key = _l2norm(key.contiguous())
        if gdn.num_value_heads // gdn.num_key_heads > 1:
            repeat = gdn.num_value_heads // gdn.num_key_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

    with _nvtx_range("art_gdn_recurrent_gate_prepare", alpha):
        g = -gdn.A_log.exp() * F.softplus(alpha.float() + gdn.dt_bias)
        beta = beta.sigmoid()

    real_mask = bucket.real_mask.transpose(0, 1)
    query = query[real_mask].unsqueeze(0).contiguous()
    key = key[real_mask].unsqueeze(0).contiguous()
    value = value[real_mask].unsqueeze(0).contiguous()
    gate = gate[real_mask].unsqueeze(0).contiguous()
    beta = beta[real_mask].unsqueeze(0).contiguous()
    g = g[real_mask].unsqueeze(0).contiguous()

    with _nvtx_range("art_gdn_recurrent_forward", query):
        recurrent_out, recurrent_final = _chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_initial,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=bucket.cu_seqlens,
        )

    with _nvtx_range("art_gdn_output_norm_gate", recurrent_out):
        norm_out = _apply_gated_rms_norm(gdn, recurrent_out, gate)
        if norm_out.ndim == 4:
            norm_out = norm_out.flatten(2).transpose(0, 1).contiguous()
        elif norm_out.ndim == 3:
            norm_out = (
                norm_out.transpose(0, 1).contiguous()
                if int(norm_out.shape[0]) == 1
                else norm_out.reshape(
                    norm_out.shape[0], 1, _local_value_dim(gdn)
                ).contiguous()
            )
        elif norm_out.ndim == 2:
            norm_out = norm_out.reshape(
                1, recurrent_out.shape[1], _local_value_dim(gdn)
            )
            norm_out = norm_out.transpose(0, 1).contiguous()
        else:
            raise RuntimeError(
                f"unexpected GDN norm output shape {tuple(norm_out.shape)}"
            )
    with _nvtx_range("art_gdn_out_proj", norm_out):
        out, out_bias = _out_proj(gdn, norm_out)
    return out, out_bias, conv_final, recurrent_final


def _conv_final_from_varlen_qkv(
    qkv: Tensor, conv_initial: Tensor, lengths: Tensor
) -> Tensor:
    tail_width = int(conv_initial.shape[-1])
    if tail_width == 0:
        return conv_initial
    batch_size, channel_count, max_len = qkv.shape
    arange = torch.arange(batch_size, device=qkv.device)
    pieces = []
    for tail_offset in range(tail_width):
        source = lengths - tail_width + tail_offset
        from_qkv = source >= 0
        qkv_index = source.clamp(min=0, max=max_len - 1)
        init_index = (source + tail_width).clamp(min=0, max=tail_width - 1)
        qkv_piece = qkv[arange, :, qkv_index]
        init_piece = conv_initial[arange, :, init_index]
        pieces.append(torch.where(from_qkv.unsqueeze(1), qkv_piece, init_piece))
    return torch.stack(pieces, dim=-1).reshape(batch_size, channel_count, tail_width)


def _causal_conv1d_varlen_with_state(
    gdn: Any,
    qkv: Tensor,
    conv_initial: Tensor,
    lengths: Tensor,
    *,
    output_final_state: bool,
) -> tuple[Tensor, Tensor | None]:
    if str(getattr(gdn, "activation", "")) == "gelu":
        return gdn_varlen_causal_conv_gelu(
            gdn,
            qkv,
            conv_initial,
            lengths,
            output_final_state=output_final_state,
        )
    conv_final = (
        _conv_final_from_varlen_qkv(qkv, conv_initial, lengths)
        if output_final_state
        else None
    )
    out, _ = _causal_conv1d_with_state(
        gdn,
        qkv,
        conv_initial,
        output_final_state=False,
    )
    return out, conv_final


def _causal_conv1d_packed_varlen_with_state(
    gdn: Any,
    qkv: Tensor,
    conv_initial: Tensor,
    cu_seqlens: Tensor,
    *,
    output_final_state: bool,
) -> tuple[Tensor, Tensor | None]:
    return packed_varlen_causal_conv(
        qkv,
        cu_seqlens,
        conv_initial,
        gdn.conv1d.weight.squeeze(1),
        gdn.conv1d.bias,
        activation=str(getattr(gdn, "activation", "gelu")),
        output_final_state=output_final_state,
    )


def _causal_conv1d_with_state(
    gdn: Any,
    qkv: Tensor,
    conv_initial: Tensor,
    *,
    output_final_state: bool,
) -> tuple[Tensor, Tensor | None]:
    weight = gdn.conv1d.weight.squeeze(1)
    bias = gdn.conv1d.bias
    if not bool(
        getattr(gdn.config, "deterministic_mode", False)
    ) and gdn.activation in ("silu", "swish"):
        qkv_fast = _channel_last_conv1d_layout(qkv)
        conv_initial_fast = _channel_last_conv1d_layout(conv_initial)
        if qkv_fast is not None and conv_initial_fast is not None:
            conv_result = causal_conv1d_fn(
                x=qkv_fast,
                weight=weight,
                bias=bias,
                initial_states=conv_initial_fast,
                return_final_states=output_final_state,
                activation=gdn.activation,
            )
            if output_final_state:
                out, final = conv_result
            else:
                out, final = conv_result, None
            return out, final

    qkv_dtype = qkv.dtype
    if not bool(getattr(gdn.config, "deterministic_mode", False)):
        final = (
            _conv_final_from_dense_qkv(qkv, conv_initial, weight.shape[1])
            if output_final_state
            else None
        )
        qkv_fast = _channel_last_conv1d_layout(qkv)
        conv_initial_fast = _channel_last_conv1d_layout(conv_initial)
        if qkv_fast is not None and conv_initial_fast is not None:
            out = causal_conv1d_fn(
                x=qkv_fast,
                weight=weight,
                bias=bias,
                initial_states=conv_initial_fast,
                return_final_states=False,
                activation=None,
            )
            out = gdn.act_fn(out).to(dtype=qkv_dtype)
            return out, final

    extended = torch.cat([conv_initial, qkv], dim=-1)
    out = F.conv1d(
        extended, weight.unsqueeze(1), bias, padding=0, groups=extended.shape[1]
    )
    out = out[..., : qkv.shape[-1]]
    out = gdn.act_fn(out).to(dtype=qkv_dtype)
    final = (
        extended[..., -(weight.shape[1] - 1) :].to(dtype=qkv_dtype)
        if output_final_state
        else None
    )
    return out, final


def _conv_final_from_dense_qkv(
    qkv: Tensor, conv_initial: Tensor, kernel_width: int
) -> Tensor:
    tail_width = int(kernel_width) - 1
    if tail_width <= 0:
        return conv_initial[..., :0].to(dtype=qkv.dtype)
    if int(qkv.shape[-1]) >= tail_width:
        return qkv[..., -tail_width:].to(dtype=qkv.dtype)
    initial_width = tail_width - int(qkv.shape[-1])
    return torch.cat([conv_initial[..., -initial_width:], qkv], dim=-1).to(
        dtype=qkv.dtype
    )


def _channel_last_conv1d_layout(tensor: Tensor) -> Tensor | None:
    if _causal_conv1d_layout_supported(tensor):
        return tensor
    channel_last = tensor.transpose(1, 2).contiguous().transpose(1, 2)
    if _causal_conv1d_layout_supported(channel_last):
        return channel_last
    return None


def _causal_conv1d_layout_supported(tensor: Tensor) -> bool:
    return (
        int(tensor.shape[-1]) >= 8
        and int(tensor.stride(1)) == 1
        and all(int(tensor.stride(dim)) % 8 == 0 for dim in (0, 2))
    )


def _disable_reentrant_te_linear_transpose_cache(gdn: Any) -> None:
    if getattr(gdn, "_art_reentrant_te_linear_transpose_cache_disabled", False):
        return
    for root in (getattr(gdn, "in_proj", None), getattr(gdn, "out_proj", None)):
        if isinstance(root, torch.nn.Module):
            linears = root.modules()
        else:
            linears = (root,)
        for linear in linears:
            if hasattr(linear, "disable_parameter_transpose_cache"):
                linear.disable_parameter_transpose_cache = True
    gdn._art_reentrant_te_linear_transpose_cache_disabled = True


def run_gdn_bucket(
    bucket: GdnSegmentBucketPlan,
    projected_streams: tuple[Tensor, Tensor, Tensor],
    parent_states: tuple[Tensor, Tensor],
    *,
    gdn: Any,
    output_final_state: bool = True,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    _disable_reentrant_te_linear_transpose_cache(gdn)
    qkv, beta, recurrent_g = projected_streams
    conv_initial, recurrent_initial = parent_states
    token_count = int(qkv.shape[0]) if qkv.ndim == 2 else -1
    segment_count = int(bucket.segment_count)
    if qkv.ndim != 2:
        raise ValueError(
            "GDN bucket execution requires compact projected streams; "
            f"got qkv shape {tuple(qkv.shape)}"
        )
    if token_count != int(bucket.real_token_count):
        raise ValueError(
            "GDN packed varlen token count mismatch, got "
            f"qkv={tuple(qkv.shape)} and bucket tokens={bucket.real_token_count}"
        )
    if tuple(beta.shape[:1]) != (token_count,) or tuple(recurrent_g.shape) != tuple(
        beta.shape
    ):
        raise ValueError(
            "packed beta/recurrent_g must be [tokens, heads], got "
            f"{tuple(beta.shape)} and {tuple(recurrent_g.shape)}"
        )
    if int(conv_initial.shape[0]) != segment_count:
        raise ValueError(
            "conv_initial batch must match bucket segment count, got "
            f"{tuple(conv_initial.shape)} for {segment_count} segments"
        )
    if int(recurrent_initial.shape[0]) != segment_count:
        raise ValueError(
            "recurrent_initial batch must match bucket segment count, got "
            f"{tuple(recurrent_initial.shape)} for {segment_count} segments"
        )

    with _nvtx_range("art_gdn_causal_conv_forward", qkv):
        qkv, conv_final = _causal_conv1d_packed_varlen_with_state(
            gdn,
            qkv,
            conv_initial,
            bucket.cu_seqlens,
            output_final_state=output_final_state,
        )

    with _nvtx_range("art_gdn_qkv_head_prepare", qkv):
        query, key, value, beta, recurrent_g = _prepare_packed_recurrent_inputs_fused(
            qkv,
            beta,
            recurrent_g,
            key_heads=_local_key_heads(gdn),
            value_heads=_local_value_heads(gdn),
            key_dim=int(gdn.key_head_dim),
            value_dim=int(gdn.value_head_dim),
        )
        if gdn.use_qk_l2norm:
            query = l2norm(query.contiguous())
            key = l2norm(key.contiguous())

    with _nvtx_range("art_gdn_recurrent_forward", query):
        recurrent_out, recurrent_final = _chunk_gated_delta_rule(
            query,
            key,
            value,
            g=recurrent_g,
            beta=beta,
            initial_state=recurrent_initial,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=bucket.cu_seqlens,
        )
    return recurrent_out, conv_final, recurrent_final


def _zero_conv_state(
    gdn: Any,
    hidden_states: Tensor,
    row: int | None = None,
    *,
    batch_size: int = 1,
) -> Tensor:
    del row
    return hidden_states.new_zeros(
        batch_size,
        gdn.conv_dim_local_tp,
        gdn.conv_kernel_dim - 1,
    )


def _zero_recurrent_state(
    gdn: Any,
    hidden_states: Tensor,
    row: int | None = None,
    *,
    batch_size: int = 1,
) -> Tensor:
    del row
    return hidden_states.new_zeros(
        batch_size,
        gdn.num_v_heads_local_tp,
        gdn.key_head_dim,
        gdn.value_head_dim,
        dtype=torch.float32,
    )


def _default_cp_rank(cp_size: int) -> int:
    del cp_size
    from megatron.core import parallel_state as ps

    return int(ps.get_context_parallel_rank())


def _default_cp_size() -> int:
    from megatron.core import parallel_state as ps

    return max(1, int(ps.get_context_parallel_world_size()))


def _default_cp_group(cp_size: int) -> Any:
    del cp_size
    from megatron.core import parallel_state as ps

    return ps.get_context_parallel_group()


def _l2norm(x: Tensor) -> Tensor:
    return l2norm(x)


def _chunk_gated_delta_rule(*args: Any, **kwargs: Any) -> tuple[Tensor, Tensor | None]:
    return chunk_gated_delta_rule(*args, **kwargs)


@contextmanager
def _nvtx_range(label: str, tensor: Tensor | None = None) -> Iterator[None]:
    if _NVTX_ENABLED.get() and tensor is not None and tensor.is_cuda:
        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
        return
    yield


@contextmanager
def gdn_nvtx_ranges(enabled: bool = True) -> Iterator[None]:
    token = _NVTX_ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _NVTX_ENABLED.reset(token)
