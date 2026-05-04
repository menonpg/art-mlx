from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, cast

import torch

CAPTURE_NAME_TOKENS = (
    ".self_attention",
    ".self_attention.in_proj",
    ".self_attention.in_proj.in_proj",
    ".self_attention.in_proj.qkv_lora",
    ".self_attention.in_proj.z_lora",
    ".self_attention.out_norm",
    ".self_attention.out_proj",
    ".self_attention.out_proj.lora",
    ".self_attention.linear_qkv",
    ".self_attention.linear_qkv.q_proj_lora",
    ".self_attention.linear_qkv.k_proj_lora",
    ".self_attention.linear_qkv.v_proj_lora",
    ".self_attention.linear_proj",
    ".self_attention.linear_proj.lora",
    ".mlp.router",
    ".mlp.experts.linear_fc1",
    ".mlp.experts.linear_fc1.gate_lora",
    ".mlp.experts.linear_fc1.up_lora",
    ".mlp.experts.linear_fc2",
    ".mlp.experts.linear_fc2.lora",
    ".mlp.linear_fc1",
    ".mlp.linear_fc1.gate_lora",
    ".mlp.linear_fc1.up_lora",
    ".mlp.linear_fc2",
    ".mlp.linear_fc2.row_parallel_lora",
    ".mlp.linear_fc2.row_parallel_lora.lora",
)
ROUTER_NAME_TOKEN = ".mlp.router"
PRIMARY_OUTPUT_CANONICAL_KEY = "primary_output__is_canonical"


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerces scalar values to int for trace metadata."""
    try:
        return int(value)
    except Exception:
        return default


def _safe_ps_stat(name: str, default: int) -> int:
    """Reads one Megatron parallel-state integer when available."""
    try:
        from megatron.core import parallel_state as ps

        getter = getattr(ps, name)
        return _safe_int(getter(), default)
    except Exception:
        return default


def _rank_metadata() -> dict[str, int]:
    """Builds lightweight distributed metadata for one trace call."""
    rank = 0
    world_size = 1
    if torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        rank = _safe_int(torch.distributed.get_rank(), 0)  # ty: ignore[possibly-missing-attribute]
        world_size = _safe_int(torch.distributed.get_world_size(), 1)  # ty: ignore[possibly-missing-attribute]
    return {
        "global_rank": rank,
        "world_size": world_size,
        "tp_rank": _safe_ps_stat("get_tensor_model_parallel_rank", 0),
        "tp_world_size": _safe_ps_stat("get_tensor_model_parallel_world_size", 1),
        "ep_rank": _safe_ps_stat("get_expert_model_parallel_rank", 0),
        "ep_world_size": _safe_ps_stat("get_expert_model_parallel_world_size", 1),
        "etp_rank": _safe_ps_stat("get_expert_tensor_parallel_rank", 0),
        "etp_world_size": _safe_ps_stat("get_expert_tensor_parallel_world_size", 1),
        "dp_rank": _safe_ps_stat("get_data_parallel_rank", 0),
        "dp_world_size": _safe_ps_stat("get_data_parallel_world_size", 1),
        "expert_dp_rank": _safe_ps_stat("get_expert_data_parallel_rank", 0),
        "expert_dp_world_size": _safe_ps_stat("get_expert_data_parallel_world_size", 1),
    }


def _extract_dp_slot_from_rank_meta(rank_meta: Any) -> tuple[int, int] | None:
    """Returns one stable `(dp_rank, dp_world_size)` pair from merged rank metadata."""
    if isinstance(rank_meta, dict):
        rank_meta = [rank_meta]
    if not isinstance(rank_meta, list) or not rank_meta:
        return None
    dp_ranks = {
        _safe_int(item.get("dp_rank"), 0)
        for item in rank_meta
        if isinstance(item, dict) and "dp_rank" in item
    }
    dp_world_sizes = {
        _safe_int(item.get("dp_world_size"), 1)
        for item in rank_meta
        if isinstance(item, dict) and "dp_world_size" in item
    }
    if len(dp_ranks) != 1 or len(dp_world_sizes) != 1:
        return None
    return next(iter(dp_ranks)), next(iter(dp_world_sizes))


def _trace_call_sort_key(call: dict[str, Any]) -> tuple[int, int]:
    """Builds a stable micro identity for merged trace ordering."""
    sample_index = call.get("micro_sample_index")
    if isinstance(sample_index, int):
        return 0, int(sample_index)
    micro_order = _safe_int(call.get("micro_order"), 0)
    dp_slot = _extract_dp_slot_from_rank_meta(call.get("rank_meta"))
    if dp_slot is None:
        return 1, micro_order
    dp_rank, dp_world_size = dp_slot
    return 1, micro_order * dp_world_size + dp_rank


def _local_dummy_micro_slot(micro_order: int) -> int:
    """Builds the stable dummy-micro slot used when one micro has no sample id."""
    dp_rank = _safe_ps_stat("get_data_parallel_rank", 0)
    dp_world_size = _safe_ps_stat("get_data_parallel_world_size", 1)
    return micro_order * dp_world_size + dp_rank


def _captured_output_sort_key(
    sample_index: int | None,
    micro_order: int,
    micro_slot: int | None,
) -> tuple[int, int, int]:
    """Builds the deterministic ordering used for captured root outputs."""
    if isinstance(sample_index, int):
        return 0, int(sample_index), micro_order
    return 1, _safe_int(micro_slot, micro_order), 0


def _shard_world_size_for_domain(domain: Any) -> int:
    """Returns shard-group world size for one LoRA shard domain."""
    if domain == "tp":
        return _safe_ps_stat("get_tensor_model_parallel_world_size", 1)
    if domain == "expert_tp":
        return _safe_ps_stat("get_expert_tensor_parallel_world_size", 1)
    return 1


def _world_size_key_for_domain(domain: Any) -> str | None:
    if domain == "tp":
        return "tp_world_size"
    if domain == "expert_tp":
        return "etp_world_size"
    return None


def _extract_primary_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_primary_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _extract_primary_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _materialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    full_tensor = getattr(tensor, "full_tensor", None)
    if callable(full_tensor):
        tensor = cast(torch.Tensor, full_tensor())
    else:
        to_local = getattr(tensor, "to_local", None)
        if callable(to_local):
            tensor = cast(torch.Tensor, to_local())
        else:
            local_tensor = getattr(tensor, "_local_tensor", None)
            if isinstance(local_tensor, torch.Tensor):
                tensor = local_tensor
    return tensor.detach().cpu()


def _materialize_trace_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _materialize_tensor(value)
    if isinstance(value, dict):
        return {key: _materialize_trace_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize_trace_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_trace_value(item) for item in value)
    return value


def _extract_tensor_attr(value: Any, attr_name: str) -> Any:
    if isinstance(value, torch.Tensor):
        return getattr(value, attr_name, None)
    if isinstance(value, dict):
        for item in value.values():
            attr_value = _extract_tensor_attr(item, attr_name)
            if attr_value is not None:
                return attr_value
    if isinstance(value, (list, tuple)):
        for item in value:
            attr_value = _extract_tensor_attr(item, attr_name)
            if attr_value is not None:
                return attr_value
    return None


@torch._dynamo.disable
def _extract_router_topk(output: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not isinstance(output, tuple) or len(output) < 2:
        return None
    probs = output[0]
    routing_map = output[1]
    if not isinstance(probs, torch.Tensor) or not isinstance(routing_map, torch.Tensor):
        return None
    probs = _materialize_tensor(probs.float())
    routing_map = _materialize_tensor(routing_map)
    topk = int(routing_map.sum(dim=-1).max().item())
    if topk < 0:
        raise RuntimeError(f"Invalid router topk={topk}")
    if topk == 0:
        topk_scores = probs.new_zeros((probs.shape[0], 0))
        topk_ids = torch.zeros((probs.shape[0], 0), dtype=torch.int64)
    else:
        topk_scores, topk_ids = torch.topk(probs, k=topk, dim=-1)
    return topk_ids.contiguous(), topk_scores.contiguous()


class ForwardTraceCapture:
    def __init__(
        self,
        model_chunks: list[Any],
        *,
        enabled: bool,
        capture_name_tokens: tuple[str, ...] = CAPTURE_NAME_TOKENS,
        micro_start_callback: Callable[[int | None, int], None] | None = None,
        strict_output_match: bool = True,
    ) -> None:
        self.enabled = enabled
        self.capture_name_tokens = capture_name_tokens
        self.micro_start_callback = micro_start_callback
        self.strict_output_match = strict_output_match
        self.current_step_index: int | None = None
        self.current_step_trace: dict[str, list[dict[str, Any]]] = {}
        self.current_micro_sample_index: int | None = None
        self.current_micro_order = 0
        self.current_micro_module_call_counts: dict[str, int] = {}
        self.current_step_sample_indices: list[int | None] = []
        self.current_step_outputs: list[
            tuple[int | None, int, int | None, torch.Tensor]
        ] = []
        self._trace_metadata_by_name: dict[str, dict[str, Any]] = {}
        self._next_micro_order = 0
        self._hook_handles: list[Any] = []
        if not enabled:
            return
        self._register_hooks(model_chunks)

    def _register_hooks(self, model_chunks: list[Any]) -> None:
        if not model_chunks:
            raise RuntimeError("Expected at least one model chunk for forward tracing")
        root_module = model_chunks[0]
        self._hook_handles.append(
            root_module.register_forward_pre_hook(self._root_pre_hook)
        )
        self._hook_handles.append(
            root_module.register_forward_hook(self._root_post_hook)
        )
        for chunk_index, chunk in enumerate(model_chunks):
            named_modules = list(chunk.named_modules())
            module_by_name = dict(named_modules)
            for module_name, module in named_modules:
                trace_module_name = f"chunk{chunk_index}.{module_name}"
                metadata = self._build_module_trace_metadata(
                    module_name=module_name,
                    module=module,
                    module_by_name=module_by_name,
                )
                if metadata:
                    self._trace_metadata_by_name[trace_module_name] = metadata
                is_layer_output = (
                    ".decoder.layers." in module_name
                    and module_name.rsplit(".", 1)[-1].isdigit()
                )
                if not is_layer_output and not any(
                    module_name.endswith(token) for token in self.capture_name_tokens
                ):
                    continue
                self._hook_handles.append(
                    module.register_forward_hook(
                        self._make_hook(trace_module_name, module)
                    )
                )

    @classmethod
    def _build_module_trace_metadata(
        cls,
        *,
        module_name: str,
        module: Any,
        module_by_name: dict[str, Any],
    ) -> dict[str, Any]:
        if module_name.endswith(".self_attention.in_proj"):
            return {
                "component_sizes": cls._gdn_in_proj_component_sizes(module),
            }
        if module_name.endswith(".self_attention.in_proj.in_proj"):
            parent_module = module_by_name[module_name.rsplit(".", 1)[0]]
            return {
                "component_sizes": cls._gdn_in_proj_component_sizes(parent_module),
            }
        if module_name.endswith(".self_attention.out_norm"):
            gdn_module = module_by_name[module_name.removesuffix(".out_norm")]
            return {
                "local_heads": int(gdn_module.num_value_heads // gdn_module.tp_size),
            }
        return {}

    @staticmethod
    def _gdn_in_proj_component_sizes(module: Any) -> tuple[int, ...]:
        qkv_sizes = tuple(
            int(size)
            for size in getattr(module.qkv_lora.B_T, "lora_tp_component_sizes")
        )
        z_world_size = _shard_world_size_for_domain(module.z_lora.B_T.lora_shard_domain)
        tp_world_size = _safe_ps_stat("get_tensor_model_parallel_world_size", 1)
        return (
            *qkv_sizes,
            int(module.z_lora.B_T.shape[-1]) * z_world_size,
            int(module.num_value_heads_per_partition) * tp_world_size,
            int(module.num_value_heads_per_partition) * tp_world_size,
        )

    @staticmethod
    def _sequence_parallel_enabled(module: Any) -> bool:
        """Returns sequence-parallel flag from module/provider/config when present."""
        for owner in (
            module,
            getattr(module, "provider", None),
            getattr(module, "config", None),
        ):
            if owner is None:
                continue
            value = getattr(owner, "sequence_parallel", None)
            if isinstance(value, bool):
                return value
        return False

    @staticmethod
    def _lora_primary_output_merge_hint(module: Any) -> dict[str, Any] | None:
        """Infers the correct output merge op for LoRA modules."""
        if module.__class__.__name__ != "LoRA":
            return None
        lora_module = module
        b_param = getattr(lora_module, "B_T", None)
        if b_param is None:
            return None
        b_domain = getattr(b_param, "lora_shard_domain", None)
        b_world_size = _shard_world_size_for_domain(b_domain)
        if bool(getattr(b_param, "lora_tp_sharded", False)) and b_world_size > 1:
            shard_dim = getattr(b_param, "lora_tp_shard_dim", None)
            if isinstance(shard_dim, int):
                hint: dict[str, Any] = {"op": "concat", "dim": shard_dim}
                component_sizes = tuple(
                    int(size)
                    for size in getattr(b_param, "lora_tp_component_sizes", ())
                )
                world_size_key = _world_size_key_for_domain(b_domain)
                if component_sizes and world_size_key is not None:
                    hint.update(
                        {
                            "layout": "componentwise",
                            "component_sizes": component_sizes,
                            "world_size_key": world_size_key,
                        }
                    )
                return hint
        a_param = getattr(lora_module, "A_T", None)
        if a_param is None:
            return None
        a_domain = getattr(a_param, "lora_shard_domain", None)
        a_world_size = _shard_world_size_for_domain(a_domain)
        if bool(getattr(a_param, "lora_tp_sharded", False)) and a_world_size > 1:
            return {"op": "sum"}
        return None

    def _infer_primary_output_merge_hint(
        self, name: str, module: Any
    ) -> dict[str, Any] | None:
        """Chooses canonical cross-rank concat axis for one module output."""
        if ROUTER_NAME_TOKEN in name:
            return {"op": "concat", "dim": 0}

        lora_hint = self._lora_primary_output_merge_hint(module)
        if lora_hint is not None:
            return lora_hint

        trace_metadata = self._trace_metadata_by_name.get(name, {})
        component_sizes = trace_metadata.get("component_sizes")
        if isinstance(component_sizes, tuple) and component_sizes:
            return {
                "op": "concat",
                "dim": -1,
                "layout": "componentwise",
                "component_sizes": component_sizes,
                "world_size_key": "tp_world_size",
            }

        local_heads = trace_metadata.get("local_heads")
        if isinstance(local_heads, int) and local_heads > 0:
            return {
                "op": "concat",
                "dim": 0,
                "layout": "rank_blocked_token_heads",
                "local_heads": local_heads,
                "world_size_key": "tp_world_size",
            }

        # Base MoE expert linears need expert-TP aware merge semantics.
        # With etp>1:
        # - FC1 (column-parallel) shards output features -> concat on feature dim.
        # - FC2 (row-parallel) emits partial output contributions -> sum across ranks.
        # With etp==1, keep the existing token-row concat behavior.
        etp_world_size = _safe_ps_stat("get_expert_tensor_parallel_world_size", 1)
        if ".mlp.experts.linear_fc1" in name and ".lora" not in name:
            if etp_world_size > 1:
                return {
                    "op": "concat",
                    "dim": -1,
                    "layout": "gate_up_rank_interleaved",
                    "world_size_key": "etp_world_size",
                }
            return {"op": "concat", "dim": 0}
        if ".mlp.experts.linear_fc2" in name and ".lora" not in name:
            if etp_world_size > 1:
                return {"op": "sum"}
            return {"op": "concat", "dim": 0}

        if ".mlp.linear_fc1" in name and ".lora" not in name:
            tp_world_size = _safe_ps_stat("get_tensor_model_parallel_world_size", 1)
            if tp_world_size > 1:
                return {
                    "op": "concat",
                    "dim": -1,
                    "layout": "gate_up_rank_interleaved",
                    "world_size_key": "tp_world_size",
                }
            return {"op": "concat", "dim": -1}
        if ".mlp.linear_fc2.row_parallel_lora" in name and ".lora" not in name:
            if self._sequence_parallel_enabled(module):
                return {"op": "concat", "dim": 0}
            return None
        if ".mlp.linear_fc2" in name and ".lora" not in name:
            row_parallel_lora = getattr(module, "row_parallel_lora", None)
            if row_parallel_lora is not None and self._sequence_parallel_enabled(
                row_parallel_lora
            ):
                return {"op": "concat", "dim": 0}
            return None

        gather_output = getattr(module, "gather_output", None)
        if isinstance(gather_output, bool) and not gather_output:
            return {"op": "concat", "dim": -1}

        if ".self_attention.linear_qkv" in name:
            return {"op": "concat", "dim": -1}
        if name.endswith(".self_attention.in_proj"):
            return {"op": "concat", "dim": -1}
        if name.endswith(
            ".self_attention.out_proj"
        ) and self._sequence_parallel_enabled(module):
            return {"op": "concat", "dim": 0}
        if name.endswith(".self_attention") and self._sequence_parallel_enabled(module):
            return {"op": "concat", "dim": 0}

        if ".mlp.experts." in name:
            return {"op": "concat", "dim": 0}

        if bool(
            getattr(module, "input_is_parallel", False)
        ) and self._sequence_parallel_enabled(module):
            return {"op": "concat", "dim": 0}

        return None

    def _build_merge_hints(self, name: str, module: Any) -> dict[str, dict[str, Any]]:
        """Builds field-level tensor merge hints for one call record."""
        hints: dict[str, dict[str, Any]] = {}
        primary_output_hint = self._infer_primary_output_merge_hint(name, module)
        if primary_output_hint is not None:
            hints["primary_output"] = primary_output_hint
        if ROUTER_NAME_TOKEN in name:
            concat_dim0 = {"op": "concat", "dim": 0}
            hints["output"] = concat_dim0
            hints["router_topk_ids"] = concat_dim0
            hints["router_topk_scores"] = concat_dim0
        return hints

    @torch._dynamo.disable
    def _record_module_hook(
        self, name: str, module: Any, inputs: Any, output: Any
    ) -> None:
        if self.current_step_index is None:
            return
        micro_call_index = self.current_micro_module_call_counts.get(name, 0)
        self.current_micro_module_call_counts[name] = micro_call_index + 1
        trace_item: dict[str, Any] = {
            "micro_call_index": micro_call_index,
            "micro_order": self.current_micro_order,
            "micro_sample_index": self.current_micro_sample_index,
            "module_type": module.__class__.__name__,
            "rank_meta": _rank_metadata(),
            "merge_hints": self._build_merge_hints(name, module),
            "inputs": _materialize_trace_value(inputs),
            "output": _materialize_trace_value(output),
            "primary_input": self.guess_primary_tensor(inputs),
            "primary_output": self.guess_primary_tensor(output),
        }
        if ROUTER_NAME_TOKEN in name:
            router_topk = _extract_router_topk(output)
            if router_topk is not None:
                topk_ids, topk_scores = router_topk
                trace_item["router_topk_ids"] = topk_ids
                trace_item["router_topk_scores"] = topk_scores
        trace_items = self._split_expert_trace_items(
            module_name=name,
            module=module,
            inputs=inputs,
            trace_item=trace_item,
        )
        trace_calls = self.current_step_trace.setdefault(name, [])
        for split_item in trace_items:
            split_item["call_index"] = len(trace_calls)
            trace_calls.append(split_item)

    def _make_hook(self, name: str, module: Any):
        def _hook(_module: Any, inputs: Any, output: Any) -> None:
            self._record_module_hook(name, module, inputs, output)

        return _hook

    @staticmethod
    def guess_primary_tensor(value: Any) -> torch.Tensor | None:
        tensor = _extract_primary_tensor(value)
        if tensor is None:
            return None
        return _materialize_tensor(tensor)

    def _sample_index_for_micro(self, micro_order: int) -> int | None:
        if micro_order < len(self.current_step_sample_indices):
            return self.current_step_sample_indices[micro_order]
        return None

    @torch._dynamo.disable
    def _root_pre_hook(self, _module: Any, _args: Any) -> None:
        if self.current_step_index is None:
            return
        micro_order = self._next_micro_order
        sample_index = self._sample_index_for_micro(micro_order)
        self.begin_micro(sample_index=sample_index, micro_order=micro_order)

    @torch._dynamo.disable
    def _root_post_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        if self.current_step_index is None:
            return
        output_tensor = self.guess_primary_tensor(output)
        if output_tensor is None:
            raise RuntimeError(
                f"Expected root forward output to contain a tensor, got {type(output)}"
            )
        sample_index = self.current_micro_sample_index
        micro_order = self.current_micro_order
        self.current_step_outputs.append(
            (
                sample_index,
                micro_order,
                None
                if sample_index is not None
                else _local_dummy_micro_slot(micro_order),
                output_tensor.float(),
            )
        )
        self._next_micro_order = micro_order + 1

    def set_step(
        self,
        step_index: int,
        sample_indices: list[int | None] | None = None,
    ) -> None:
        self.current_step_index = step_index
        self.current_step_trace = {}
        self.current_step_sample_indices = list(sample_indices or [])
        self.current_step_outputs = []
        self.current_micro_sample_index = None
        self.current_micro_order = 0
        self.current_micro_module_call_counts = {}
        self._next_micro_order = 0

    def begin_micro(self, sample_index: int | None, micro_order: int) -> None:
        self.current_micro_sample_index = sample_index
        self.current_micro_order = micro_order
        self.current_micro_module_call_counts = {}
        if self.micro_start_callback is not None:
            self.micro_start_callback(sample_index, micro_order)

    @staticmethod
    def _row_token_uids_for_trace(
        *,
        inputs: Any,
        module: Any,
    ) -> tuple[torch.Tensor | None, int | None]:
        row_token_uids = _extract_tensor_attr(inputs, "_art_trace_row_token_uids")
        if row_token_uids is None:
            row_token_uids = getattr(module, "_art_trace_row_token_uids", None)
        if not isinstance(row_token_uids, torch.Tensor):
            return None, None

        uid_span = _extract_tensor_attr(inputs, "_art_trace_uid_span")
        if uid_span is None:
            uid_span = getattr(module, "_art_trace_uid_span", None)
        uid_span_int = uid_span if isinstance(uid_span, int) and uid_span > 0 else None
        return (
            row_token_uids.detach().to(device="cpu", dtype=torch.int64).reshape(-1),
            uid_span_int,
        )

    @classmethod
    def _slice_row_aligned_value(
        cls,
        value: Any,
        *,
        row_indices: torch.Tensor,
        total_rows: int,
    ) -> Any:
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and int(value.shape[0]) == total_rows:
                return value.index_select(0, row_indices)
            return value
        if isinstance(value, dict):
            return {
                key: cls._slice_row_aligned_value(
                    item,
                    row_indices=row_indices,
                    total_rows=total_rows,
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                cls._slice_row_aligned_value(
                    item,
                    row_indices=row_indices,
                    total_rows=total_rows,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                cls._slice_row_aligned_value(
                    item,
                    row_indices=row_indices,
                    total_rows=total_rows,
                )
                for item in value
            )
        return value

    @classmethod
    def _split_expert_trace_items(
        cls,
        *,
        module_name: str,
        module: Any,
        inputs: Any,
        trace_item: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not cls._is_moe_expert_forward_module(module_name):
            return [trace_item]

        primary_output = trace_item.get("primary_output")
        if not isinstance(primary_output, torch.Tensor) or primary_output.ndim == 0:
            return [trace_item]

        row_token_uids, uid_span = cls._row_token_uids_for_trace(
            inputs=inputs,
            module=module,
        )
        if row_token_uids is None:
            return [trace_item]

        total_rows = int(row_token_uids.numel())
        if total_rows == 0 or int(primary_output.shape[0]) != total_rows:
            return [trace_item]

        trace_item["row_token_uids"] = row_token_uids
        if uid_span is None:
            return [trace_item]

        sample_ids = torch.div(row_token_uids, uid_span, rounding_mode="floor")
        ordered_sample_ids: list[int] = []
        seen_sample_ids: set[int] = set()
        for sample_id in sample_ids.tolist():
            sample_id_int = int(sample_id)
            if sample_id_int in seen_sample_ids:
                continue
            seen_sample_ids.add(sample_id_int)
            ordered_sample_ids.append(sample_id_int)

        if len(ordered_sample_ids) <= 1:
            if ordered_sample_ids:
                trace_item["micro_sample_index"] = ordered_sample_ids[0]
            return [trace_item]

        split_items: list[dict[str, Any]] = []
        for sample_id in ordered_sample_ids:
            row_indices = (sample_ids == sample_id).nonzero(as_tuple=False).reshape(-1)
            split_item = {
                key: cls._slice_row_aligned_value(
                    value,
                    row_indices=row_indices,
                    total_rows=total_rows,
                )
                for key, value in trace_item.items()
                if key not in {"call_index", "micro_sample_index", "row_token_uids"}
            }
            split_item["micro_sample_index"] = sample_id
            split_item["row_token_uids"] = row_token_uids.index_select(0, row_indices)
            split_items.append(split_item)
        return split_items

    @staticmethod
    def _is_moe_expert_forward_module(module_name: str) -> bool:
        """Returns whether one module emits MoE expert forward outputs."""
        if ".mlp.experts." not in module_name:
            return False
        if ".mlp.router" in module_name:
            return False
        return ".linear_fc1" in module_name or ".linear_fc2" in module_name

    @staticmethod
    def _primary_output_merge_hint(call: dict[str, Any]) -> dict[str, Any] | None:
        """Reads primary-output merge metadata from one call payload."""
        merge_hints = call.get("merge_hints")
        if not isinstance(merge_hints, dict):
            return None
        primary_hint = merge_hints.get("primary_output")
        if not isinstance(primary_hint, dict):
            return None
        return primary_hint

    @classmethod
    def _canonicalize_gate_up_rank_interleaved_feature_layout(
        cls,
        *,
        module_name: str,
        tensor: torch.Tensor,
        call: dict[str, Any],
    ) -> torch.Tensor:
        """Normalizes TP/ETP fused gate-up fc1 output feature order."""
        del module_name
        primary_hint = cls._primary_output_merge_hint(call)
        if not isinstance(primary_hint, dict):
            return tensor
        if primary_hint.get("layout") != "gate_up_rank_interleaved":
            return tensor
        world_size_key = primary_hint.get("world_size_key")
        if not isinstance(world_size_key, str):
            raise RuntimeError("gate_up_rank_interleaved hint requires world_size_key")
        rank_meta = call.get("rank_meta")
        rank_world_size = None
        if isinstance(rank_meta, list) and rank_meta:
            first_meta = rank_meta[0]
            if isinstance(first_meta, dict):
                rank_world_size = first_meta.get(world_size_key)
        elif isinstance(rank_meta, dict):
            rank_world_size = rank_meta.get(world_size_key)
        if not isinstance(rank_world_size, int) or rank_world_size <= 1:
            return tensor
        block_count = 2 * rank_world_size
        if tensor.ndim < 1 or tensor.shape[-1] % block_count != 0:
            raise RuntimeError(
                "gate_up_rank_interleaved tensor feature size must divide by "
                f"{block_count}, got shape={tuple(tensor.shape)}"
            )
        blocks = torch.chunk(tensor, block_count, dim=-1)
        reordered = [blocks[index] for index in range(0, block_count, 2)] + [
            blocks[index] for index in range(1, block_count, 2)
        ]
        return torch.cat(reordered, dim=-1).contiguous()

    @classmethod
    def _canonicalize_componentwise_feature_layout(
        cls,
        *,
        module_name: str,
        tensor: torch.Tensor,
        call: dict[str, Any],
    ) -> torch.Tensor:
        """Normalizes fused componentwise TP output order, e.g. GDN q/k/v."""
        del module_name
        primary_hint = cls._primary_output_merge_hint(call)
        if not isinstance(primary_hint, dict):
            return tensor
        if primary_hint.get("layout") != "componentwise":
            return tensor
        dim = primary_hint.get("dim")
        component_sizes = primary_hint.get("component_sizes")
        world_size_key = primary_hint.get("world_size_key")
        if not isinstance(dim, int) or not isinstance(world_size_key, str):
            raise RuntimeError("componentwise hint requires dim and world_size_key")
        if not isinstance(component_sizes, tuple) or not all(
            isinstance(size, int) and size > 0 for size in component_sizes
        ):
            raise RuntimeError("componentwise hint requires positive component sizes")
        rank_meta = call.get("rank_meta")
        rank_world_size = None
        if isinstance(rank_meta, list) and rank_meta:
            first_meta = rank_meta[0]
            if isinstance(first_meta, dict):
                rank_world_size = first_meta.get(world_size_key)
        elif isinstance(rank_meta, dict):
            rank_world_size = rank_meta.get(world_size_key)
        if not isinstance(rank_world_size, int) or rank_world_size <= 1:
            return tensor
        axis = dim if dim >= 0 else tensor.ndim + dim
        if axis < 0 or axis >= tensor.ndim:
            raise RuntimeError(
                f"Invalid componentwise axis {dim} for {tensor.ndim}D tensor"
            )
        if sum(component_sizes) != tensor.shape[axis]:
            raise RuntimeError(
                "componentwise component sizes must match tensor extent, got "
                f"sizes={component_sizes} shape={tuple(tensor.shape)} axis={axis}"
            )
        if any(size % rank_world_size != 0 for size in component_sizes):
            raise RuntimeError(
                "componentwise component sizes must divide rank world size, got "
                f"sizes={component_sizes} world_size={rank_world_size}"
            )
        local_sizes = [size // rank_world_size for size in component_sizes]
        rank_chunks: list[list[torch.Tensor]] = []
        cursor = 0
        for _rank in range(rank_world_size):
            rank_components = []
            for local_size in local_sizes:
                rank_components.append(tensor.narrow(axis, cursor, local_size))
                cursor += local_size
            rank_chunks.append(rank_components)
        ordered = [
            rank_chunks[rank][component_index]
            for component_index in range(len(component_sizes))
            for rank in range(rank_world_size)
        ]
        return torch.cat(ordered, dim=axis).contiguous()

    @classmethod
    def _canonicalize_rank_blocked_token_heads(
        cls,
        *,
        module_name: str,
        tensor: torch.Tensor,
        call: dict[str, Any],
    ) -> torch.Tensor:
        del module_name
        primary_hint = cls._primary_output_merge_hint(call)
        if not isinstance(primary_hint, dict):
            return tensor
        if primary_hint.get("layout") != "rank_blocked_token_heads":
            return tensor
        local_heads = primary_hint.get("local_heads")
        world_size_key = primary_hint.get("world_size_key")
        if not isinstance(local_heads, int) or local_heads <= 0:
            raise RuntimeError("rank_blocked_token_heads hint requires local_heads")
        if not isinstance(world_size_key, str):
            raise RuntimeError("rank_blocked_token_heads hint requires world_size_key")
        rank_meta = call.get("rank_meta")
        rank_world_size = None
        if isinstance(rank_meta, list) and rank_meta:
            first_meta = rank_meta[0]
            if isinstance(first_meta, dict):
                rank_world_size = first_meta.get(world_size_key)
        elif isinstance(rank_meta, dict):
            rank_world_size = rank_meta.get(world_size_key)
        if not isinstance(rank_world_size, int) or rank_world_size <= 1:
            return tensor
        if tensor.ndim != 2:
            raise RuntimeError(
                "rank_blocked_token_heads expects a 2D [rows, head_dim] tensor, "
                f"got shape={tuple(tensor.shape)}"
            )
        rows_per_rank, remainder = divmod(int(tensor.shape[0]), rank_world_size)
        if remainder != 0:
            raise RuntimeError(
                "rank_blocked_token_heads rows must divide rank world size, got "
                f"shape={tuple(tensor.shape)} world_size={rank_world_size}"
            )
        token_count, head_remainder = divmod(rows_per_rank, local_heads)
        if head_remainder != 0:
            raise RuntimeError(
                "rank_blocked_token_heads rows per rank must divide local_heads, got "
                f"rows_per_rank={rows_per_rank} local_heads={local_heads}"
            )
        return (
            tensor.reshape(rank_world_size, token_count, local_heads, tensor.shape[-1])
            .permute(1, 0, 2, 3)
            .reshape(tensor.shape)
            .contiguous()
        )

    @classmethod
    def _canonicalize_moe_expert_row_order(
        cls,
        *,
        module_name: str,
        tensor: torch.Tensor,
        call: dict[str, Any],
    ) -> torch.Tensor:
        """Canonicalizes MoE expert rows using dispatch-time UID identities."""
        if not cls._is_moe_expert_forward_module(module_name):
            return tensor
        if tensor.ndim != 2:
            return tensor
        primary_hint = cls._primary_output_merge_hint(call)
        if isinstance(primary_hint, dict) and (
            primary_hint.get("op") != "concat" or primary_hint.get("dim") != 0
        ):
            return tensor
        row_token_uids = call.get("row_token_uids")
        if not isinstance(row_token_uids, torch.Tensor):
            return tensor
        if int(row_token_uids.numel()) != int(tensor.shape[0]):
            return tensor
        order = torch.argsort(row_token_uids, stable=True)
        return tensor.index_select(0, order)

    @classmethod
    def _canonicalize_primary_output_tensor(
        cls,
        *,
        module_name: str,
        tensor: torch.Tensor,
        call: dict[str, Any],
    ) -> torch.Tensor:
        """Runs all remaining primary-output canonicalization passes for one call."""
        tensor = cls._canonicalize_gate_up_rank_interleaved_feature_layout(
            module_name=module_name,
            tensor=tensor,
            call=call,
        )
        tensor = cls._canonicalize_componentwise_feature_layout(
            module_name=module_name,
            tensor=tensor,
            call=call,
        )
        tensor = cls._canonicalize_rank_blocked_token_heads(
            module_name=module_name,
            tensor=tensor,
            call=call,
        )
        return cls._canonicalize_moe_expert_row_order(
            module_name=module_name,
            tensor=tensor,
            call=call,
        )

    @classmethod
    def canonicalize_trace(
        cls,
        trace: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Canonicalizes topology-dependent trace outputs in place."""
        for module_name in sorted(trace.keys()):
            calls = trace[module_name]
            for call_offset, call in enumerate(calls):
                if bool(call.get(PRIMARY_OUTPUT_CANONICAL_KEY)):
                    continue
                call_index = int(call.get("call_index", call_offset))
                tensor = call.get("primary_output")
                if isinstance(tensor, torch.Tensor):
                    call["primary_output"] = cls._canonicalize_primary_output_tensor(
                        module_name=module_name,
                        tensor=tensor,
                        call=call,
                    )
                call[PRIMARY_OUTPUT_CANONICAL_KEY] = True
        return trace

    @classmethod
    def flatten_trace_tensors(
        cls,
        trace: dict[str, list[dict[str, Any]]],
        *,
        value_key: str,
    ) -> dict[str, Any]:
        """Flattens trace calls into deterministic key->value tensor maps."""
        if value_key == "primary_output":
            cls.canonicalize_trace(trace)
        flattened: dict[str, Any] = {}
        for module_name in sorted(trace.keys()):
            for call_offset, call in enumerate(trace[module_name]):
                tensor = call.get(value_key)
                if tensor is None:
                    continue
                call_index = call.get("call_index", call_offset)
                flattened[f"{module_name}.call_{call_index}"] = tensor
        return flattened

    @classmethod
    def _merge_rank_values(
        cls,
        values_by_rank: list[Any],
        *,
        preferred_cat_dim: int | None = None,
        preferred_reduce: str | None = None,
    ) -> Any:
        if not values_by_rank:
            raise RuntimeError("Cannot merge empty rank value list")
        if all(isinstance(value, torch.Tensor) for value in values_by_rank):
            tensors = cast(list[torch.Tensor], values_by_rank)
            if preferred_reduce == "sum" and all(
                tensors[0].shape == tensor.shape for tensor in tensors[1:]
            ):
                return torch.stack(tensors, dim=0).sum(dim=0)
            if (
                preferred_cat_dim is not None
                and all(tensor.ndim > 0 for tensor in tensors)
                and cls._can_cat_along_dim(tensors, dim=preferred_cat_dim)
            ):
                return torch.cat(tensors, dim=preferred_cat_dim)
            if all(tensor.ndim > 0 for tensor in tensors):
                if cls._can_cat_along_dim(tensors, dim=0):
                    return torch.cat(tensors, dim=0)
                if cls._can_cat_along_dim(tensors, dim=-1):
                    return torch.cat(tensors, dim=-1)
            if all(tensors[0].shape == tensor.shape for tensor in tensors[1:]):
                return torch.stack(tensors, dim=0)
            return tensors
        if all(isinstance(value, dict) for value in values_by_rank):
            dicts = cast(list[dict[str, Any]], values_by_rank)
            keys = sorted(set().union(*(value.keys() for value in dicts)))
            return {
                key: cls._merge_rank_values(
                    [value[key] for value in dicts if key in value],
                    preferred_cat_dim=preferred_cat_dim,
                    preferred_reduce=preferred_reduce,
                )
                for key in keys
            }
        if all(isinstance(value, list) for value in values_by_rank):
            lists = cast(list[list[Any]], values_by_rank)
            if any(len(values) != len(lists[0]) for values in lists[1:]):
                return lists
            return [
                cls._merge_rank_values(
                    [value[index] for value in lists],
                    preferred_cat_dim=preferred_cat_dim,
                    preferred_reduce=preferred_reduce,
                )
                for index in range(len(lists[0]))
            ]
        if all(isinstance(value, tuple) for value in values_by_rank):
            tuples = cast(list[tuple[Any, ...]], values_by_rank)
            if any(len(values) != len(tuples[0]) for values in tuples[1:]):
                return tuples
            return tuple(
                cls._merge_rank_values(
                    [value[index] for value in tuples],
                    preferred_cat_dim=preferred_cat_dim,
                    preferred_reduce=preferred_reduce,
                )
                for index in range(len(tuples[0]))
            )
        if all(value == values_by_rank[0] for value in values_by_rank[1:]):
            return values_by_rank[0]
        return values_by_rank

    @classmethod
    def _merge_rank_call_entries(
        cls,
        rank_call_entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Merges one module call across ranks using per-field merge hints."""
        merged_call: dict[str, Any] = {}
        keys = sorted(set().union(*(entry.keys() for entry in rank_call_entries)))
        for key in keys:
            values = [entry[key] for entry in rank_call_entries if key in entry]
            if key == "rank_meta":
                merged_call[key] = values
                continue
            preferred_cat_dim: int | None = None
            preferred_reduce: str | None = None
            if values and key not in {"merge_hints", "call_index", "module_type"}:
                hint_values = [
                    cast(dict[str, Any], entry["merge_hints"]).get(key)
                    for entry in rank_call_entries
                    if isinstance(entry.get("merge_hints"), dict)
                ]
                op_hints = [
                    hint
                    for hint in hint_values
                    if isinstance(hint, dict) and isinstance(hint.get("op"), str)
                ]
                if op_hints:
                    selected_hint = op_hints[0]
                    op = selected_hint.get("op")
                    if op == "concat":
                        dim = selected_hint.get("dim")
                        if isinstance(dim, int):
                            preferred_cat_dim = dim
                    elif op == "sum":
                        preferred_reduce = "sum"
                if (
                    preferred_reduce is None
                    and preferred_cat_dim == 0
                    and all(isinstance(value, torch.Tensor) for value in values)
                ):
                    merged_call[f"{key}__row_splits"] = [
                        int(cast(torch.Tensor, value).shape[0]) for value in values
                    ]
            merged_call[key] = cls._merge_rank_values(
                values,
                preferred_cat_dim=preferred_cat_dim,
                preferred_reduce=preferred_reduce,
            )
        return merged_call

    @staticmethod
    def _can_cat_along_dim(tensors: list[torch.Tensor], dim: int) -> bool:
        if not tensors:
            return False
        if tensors[0].ndim == 0:
            return False
        ndim = tensors[0].ndim
        axis = dim if dim >= 0 else ndim + dim
        if axis < 0 or axis >= ndim:
            return False
        if any(tensor.ndim != ndim for tensor in tensors[1:]):
            return False
        for dim_index in range(ndim):
            if dim_index == axis:
                continue
            dim_size = tensors[0].shape[dim_index]
            if any(tensor.shape[dim_index] != dim_size for tensor in tensors[1:]):
                return False
        return True

    @classmethod
    def _merge_rank_traces(
        cls,
        rank_traces: list[dict[str, list[dict[str, Any]]]],
    ) -> dict[str, list[dict[str, Any]]]:
        if len(rank_traces) == 1:
            return rank_traces[0]
        merged: dict[str, list[dict[str, Any]]] = {}
        module_names = sorted(set().union(*(trace.keys() for trace in rank_traces)))
        for module_name in module_names:
            module_calls: list[dict[str, Any]] = []
            grouped_calls: dict[
                tuple[int, int, int, int],
                list[dict[str, Any]],
            ] = {}
            for trace in rank_traces:
                for call in trace.get(module_name, []):
                    sample_kind, sample_sort_index = _trace_call_sort_key(call)
                    merge_key = (
                        sample_kind,
                        sample_sort_index,
                        int(call.get("micro_order", 0)),
                        int(call.get("micro_call_index", call.get("call_index", 0))),
                    )
                    grouped_calls.setdefault(merge_key, []).append(call)
            for merged_index, merge_key in enumerate(sorted(grouped_calls)):
                merged_call = cls._merge_rank_call_entries(grouped_calls[merge_key])
                merged_call["call_index"] = merged_index
                module_calls.append(merged_call)
            merged[module_name] = module_calls
        return merged

    @staticmethod
    def _gather_rank_traces(
        local_trace: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, list[dict[str, Any]]]] | None:
        if (
            not torch.distributed.is_initialized()  # ty: ignore[possibly-missing-attribute]
            or torch.distributed.get_world_size() == 1  # ty: ignore[possibly-missing-attribute]
        ):
            return [local_trace]
        gathered: list[dict[str, list[dict[str, Any]]] | None] = [
            None
        ] * torch.distributed.get_world_size()  # ty: ignore[possibly-missing-attribute]
        torch.distributed.all_gather_object(gathered, local_trace)  # ty: ignore[possibly-missing-attribute]
        if torch.distributed.get_rank() != 0:  # ty: ignore[possibly-missing-attribute]
            return None
        return cast(list[dict[str, list[dict[str, Any]]]], gathered)

    @staticmethod
    def _merge_group_tensor(
        tensors: list[torch.Tensor], *, strict: bool = True
    ) -> torch.Tensor:
        if len(tensors) == 1:
            return tensors[0]
        first = tensors[0]
        if all(tensor.shape == first.shape for tensor in tensors[1:]) and all(
            torch.equal(first, tensor) for tensor in tensors[1:]
        ):
            return first
        if not strict:
            return first
        raise RuntimeError(
            "Mismatched output captures for the same micro output across non-DP ranks"
        )

    @staticmethod
    def _gather_rank_outputs(
        local_outputs: list[tuple[int | None, int, int | None, torch.Tensor]],
    ) -> list[list[tuple[int | None, int, int | None, torch.Tensor]]] | None:
        if (
            not torch.distributed.is_initialized()  # ty: ignore[possibly-missing-attribute]
            or torch.distributed.get_world_size() == 1  # ty: ignore[possibly-missing-attribute]
        ):
            return [local_outputs]
        gathered: list[
            list[tuple[int | None, int, int | None, torch.Tensor]] | None
        ] = [None] * torch.distributed.get_world_size()  # ty: ignore[possibly-missing-attribute]
        torch.distributed.all_gather_object(gathered, local_outputs)  # ty: ignore[possibly-missing-attribute]
        if torch.distributed.get_rank() != 0:  # ty: ignore[possibly-missing-attribute]
            return None
        return cast(
            list[list[tuple[int | None, int, int | None, torch.Tensor]]],
            gathered,
        )

    def ordered_step_outputs(self) -> list[torch.Tensor] | None:
        if not self.enabled:
            return None
        gathered_outputs = self._gather_rank_outputs(self.current_step_outputs)
        if gathered_outputs is None:
            return None
        grouped: dict[tuple[int | None, int | None, int], list[torch.Tensor]] = {}
        for rank_outputs in gathered_outputs:
            for sample_index, micro_order, micro_slot, tensor in rank_outputs:
                group_key = (sample_index, micro_slot, micro_order)
                grouped.setdefault(group_key, []).append(tensor)
        ordered_group_keys = sorted(
            grouped,
            key=lambda item: _captured_output_sort_key(item[0], item[2], item[1]),
        )
        return [
            self._merge_group_tensor(
                grouped[group_key],
                strict=self.strict_output_match,
            )
            for group_key in ordered_group_keys
        ]

    def save_current_step(self, traces_dir: Path) -> Path | None:
        if not self.enabled or self.current_step_index is None:
            return None
        gathered_traces = self._gather_rank_traces(self.current_step_trace)
        if gathered_traces is None:
            return None
        merged_trace = self.canonicalize_trace(self._merge_rank_traces(gathered_traces))
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_path = traces_dir / f"forward_trace_step_{self.current_step_index:03d}.pt"
        tmp_trace_path = trace_path.with_suffix(f"{trace_path.suffix}.tmp")
        torch.save(merged_trace, tmp_trace_path)
        os.replace(tmp_trace_path, trace_path)
        return trace_path

    @classmethod
    def load_trace(cls, trace_path: Path) -> dict[str, list[dict[str, Any]]]:
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        return cls.canonicalize_trace(trace)

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
