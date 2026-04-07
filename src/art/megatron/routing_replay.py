from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import types
from typing import Any, Protocol

from megatron.core.tensor_parallel import (
    all_to_all,
    gather_from_sequence_parallel_region,
)
from megatron.core.transformer.moe.moe_utils import permute, sort_chunks_by_idxs
from pydantic import BaseModel, ConfigDict, model_validator
from safetensors.torch import load_file, save_file
import torch

ROUTER_NAME_TOKEN = ".mlp.router"
ROUTER_KEY_FORMAT_VERSION = "moe_routing_replay_v1"
GLOBAL_TOKEN_UIDS_KEY = "global_token_uids"
TRACE_ROW_TOKEN_UIDS_ATTR = "_art_trace_row_token_uids"
TRACE_UID_SPAN_ATTR = "_art_trace_uid_span"

_ROUTER_LAYER_PATTERN = re.compile(r"decoder\.layers\.(?P<layer>\d+)\.mlp\.router$")
_TRACE_CHUNK_PREFIX_PATTERN = re.compile(r"^chunk(?P<chunk>\d+)\.(?P<name>.+)$")


def _to_tensor_cpu_contiguous(
    tensor: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    return tensor.detach().to(device="cpu", dtype=dtype).contiguous()


def _normalize_step_index(step_index: int) -> str:
    if step_index < 0:
        raise ValueError(f"step_index must be non-negative, got {step_index}")
    return f"{step_index:06d}"


def _build_tensor_key(router_key: str, call_index: int, field_name: str) -> str:
    return f"{router_key}/call_{call_index}/{field_name}"


def _flatten_router_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        raise RuntimeError(
            f"Router tensor must have rank >=2, got shape={tuple(tensor.shape)}"
        )
    num_experts = int(tensor.shape[-1])
    return tensor.reshape(-1, num_experts).contiguous()


def _extract_router_output_tensors(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, (list, tuple)) and len(output) >= 2:
        probs, routing_map = output[0], output[1]
    elif isinstance(output, dict):
        probs = output.get("probs")
        routing_map = output.get("routing_map")
    else:
        raise RuntimeError(f"Unsupported router output type: {type(output)}")

    if not isinstance(probs, torch.Tensor):
        raise RuntimeError(f"Expected probs tensor, got {type(probs)}")
    if not isinstance(routing_map, torch.Tensor):
        raise RuntimeError(f"Expected routing_map tensor, got {type(routing_map)}")

    probs_2d = _flatten_router_tensor(probs.to(torch.float32))
    routing_map_2d = _flatten_router_tensor(routing_map.bool())
    if probs_2d.shape != routing_map_2d.shape:
        raise RuntimeError(
            "Router output shape mismatch: "
            f"probs={tuple(probs_2d.shape)} routing_map={tuple(routing_map_2d.shape)}"
        )
    return probs_2d, routing_map_2d


def _extract_dp_slot_from_rank_meta(rank_meta: Any) -> tuple[int, int] | None:
    if isinstance(rank_meta, dict):
        rank_meta = [rank_meta]
    if not isinstance(rank_meta, list) or not rank_meta:
        return None
    dp_ranks = {
        int(item["dp_rank"])
        for item in rank_meta
        if isinstance(item, dict) and "dp_rank" in item
    }
    dp_world_sizes = {
        int(item["dp_world_size"])
        for item in rank_meta
        if isinstance(item, dict) and "dp_world_size" in item
    }
    if len(dp_ranks) != 1 or len(dp_world_sizes) != 1:
        return None
    return next(iter(dp_ranks)), next(iter(dp_world_sizes))


def _trace_call_route_metadata(
    call_entry: dict[str, Any],
) -> tuple[int | None, int | None]:
    sample_index = call_entry.get("micro_sample_index")
    if isinstance(sample_index, int):
        return int(sample_index), None
    dp_slot = _extract_dp_slot_from_rank_meta(call_entry.get("rank_meta"))
    micro_order = int(call_entry.get("micro_order", 0))
    if dp_slot is None:
        return None, micro_order
    dp_rank, dp_world_size = dp_slot
    return None, micro_order * dp_world_size + dp_rank


def build_router_key_from_module_name(*, chunk_index: int, module_name: str) -> str:
    match = _ROUTER_LAYER_PATTERN.search(module_name)
    if match is None:
        raise RuntimeError(
            f"Unable to derive router key from module name '{module_name}'. "
            f"Expected suffix matching '{_ROUTER_LAYER_PATTERN.pattern}'."
        )
    layer_index = int(match.group("layer"))
    return f"chunk_{chunk_index:02d}.layer_{layer_index:04d}.mlp.router"


def build_router_key_from_trace_name(trace_module_name: str) -> str:
    chunk_match = _TRACE_CHUNK_PREFIX_PATTERN.match(trace_module_name)
    if chunk_match is None:
        raise RuntimeError(
            "Forward trace router module name must start with 'chunk<idx>.'; "
            f"got '{trace_module_name}'"
        )
    chunk_index = int(chunk_match.group("chunk"))
    module_name = chunk_match.group("name")
    return build_router_key_from_module_name(
        chunk_index=chunk_index,
        module_name=module_name,
    )


class ParallelTopology(BaseModel):
    tp: int
    ep: int
    etp: int = 1
    dp: int = 1
    sp: bool = False
    cp: int = 1
    pp: int = 1
    vpp: int = 1


class RouterCallRoute(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    expert_indices: torch.Tensor
    expert_probs: torch.Tensor
    expert_mask: torch.Tensor
    routing_map: torch.Tensor | None = None
    num_experts: int
    sample_index: int | None = None
    micro_slot: int | None = None

    @model_validator(mode="after")
    def _validate(self) -> "RouterCallRoute":
        self.expert_indices = _to_tensor_cpu_contiguous(
            self.expert_indices, dtype=torch.int32
        )
        self.expert_probs = _to_tensor_cpu_contiguous(
            self.expert_probs, dtype=torch.float32
        )
        self.expert_mask = _to_tensor_cpu_contiguous(self.expert_mask, dtype=torch.bool)
        if self.routing_map is not None:
            self.routing_map = _to_tensor_cpu_contiguous(
                self.routing_map, dtype=torch.bool
            )

        if self.expert_indices.ndim != 2:
            raise RuntimeError(
                "expert_indices must have shape [num_tokens, max_topk], got "
                f"{tuple(self.expert_indices.shape)}"
            )
        if self.expert_probs.shape != self.expert_indices.shape:
            raise RuntimeError(
                "expert_probs shape must match expert_indices shape, got "
                f"{tuple(self.expert_probs.shape)} vs {tuple(self.expert_indices.shape)}"
            )
        if self.expert_mask.shape != self.expert_indices.shape:
            raise RuntimeError(
                "expert_mask shape must match expert_indices shape, got "
                f"{tuple(self.expert_mask.shape)} vs {tuple(self.expert_indices.shape)}"
            )
        if self.num_experts <= 0:
            raise RuntimeError(f"num_experts must be >0, got {self.num_experts}")
        if self.sample_index is not None:
            self.sample_index = int(self.sample_index)
        if self.micro_slot is not None:
            self.micro_slot = int(self.micro_slot)
        if self.routing_map is not None:
            expected = (self.expert_indices.shape[0], self.num_experts)
            if tuple(self.routing_map.shape) != expected:
                raise RuntimeError(
                    "routing_map shape mismatch: "
                    f"expected={expected}, got={tuple(self.routing_map.shape)}"
                )
        return self

    @property
    def num_global_tokens(self) -> int:
        return int(self.expert_indices.shape[0])

    @property
    def max_topk(self) -> int:
        return int(self.expert_indices.shape[1])


class StepRouterRoutes(BaseModel):
    calls: dict[int, RouterCallRoute]

    @model_validator(mode="after")
    def _validate_calls(self) -> "StepRouterRoutes":
        if not self.calls:
            raise RuntimeError("StepRouterRoutes.calls cannot be empty")
        for call_index in self.calls:
            if call_index < 0:
                raise RuntimeError(f"call_index must be >=0, got {call_index}")
        return self


class StepRoutes(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    routers: dict[str, StepRouterRoutes]
    global_token_uids: torch.Tensor

    @model_validator(mode="after")
    def _validate(self) -> "StepRoutes":
        if not self.routers:
            raise RuntimeError("StepRoutes.routers cannot be empty")
        self.global_token_uids = _to_tensor_cpu_contiguous(
            self.global_token_uids, dtype=torch.int64
        )
        if self.global_token_uids.ndim != 1:
            raise RuntimeError(
                "global_token_uids must have shape [num_global_tokens], got "
                f"{tuple(self.global_token_uids.shape)}"
            )
        if int(torch.unique(self.global_token_uids).numel()) != int(
            self.global_token_uids.numel()
        ):
            raise RuntimeError("global_token_uids must be unique per step")
        expected_tokens = int(self.global_token_uids.numel())
        for router_key, step_router in self.routers.items():
            for call_index, route in step_router.calls.items():
                if route.num_global_tokens != expected_tokens:
                    raise RuntimeError(
                        "Route token count mismatch for "
                        f"router='{router_key}' call={call_index}: "
                        f"route_tokens={route.num_global_tokens}, "
                        f"expected_tokens={expected_tokens}"
                    )
        return self


class MoeRoutingReplayBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    format_version: str = ROUTER_KEY_FORMAT_VERSION
    topology: ParallelTopology
    num_steps: int
    max_topk: int
    router_keys: list[str]
    steps: dict[int, StepRoutes]

    @model_validator(mode="after")
    def _validate(self) -> "MoeRoutingReplayBundle":
        if self.format_version != ROUTER_KEY_FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported format_version={self.format_version}; "
                f"expected={ROUTER_KEY_FORMAT_VERSION}"
            )
        if self.num_steps <= 0:
            raise RuntimeError(f"num_steps must be >0, got {self.num_steps}")
        if self.max_topk < 0:
            raise RuntimeError(f"max_topk must be >=0, got {self.max_topk}")
        if set(self.steps.keys()) != set(range(self.num_steps)):
            raise RuntimeError(
                "steps must be indexed from 0..num_steps-1 without gaps: "
                f"num_steps={self.num_steps}, step_keys={sorted(self.steps.keys())}"
            )
        if not self.router_keys:
            raise RuntimeError("router_keys cannot be empty")
        router_key_set = set(self.router_keys)
        for step_index, step_routes in self.steps.items():
            step_router_keys = set(step_routes.routers.keys())
            if step_router_keys != router_key_set:
                raise RuntimeError(
                    f"Step {step_index} router set mismatch. "
                    f"expected={sorted(router_key_set)}, got={sorted(step_router_keys)}"
                )
        return self

    @classmethod
    def from_dir(cls, bundle_dir: str | Path) -> "MoeRoutingReplayBundle":
        base_dir = Path(bundle_dir)
        manifest_path = base_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing routing replay manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        if manifest.get("format_version") != ROUTER_KEY_FORMAT_VERSION:
            raise RuntimeError(
                "Unsupported routing replay manifest version: "
                f"{manifest.get('format_version')}"
            )

        topology = ParallelTopology.model_validate(manifest["topology"])
        num_steps = int(manifest["num_steps"])
        max_topk = int(manifest["max_topk"])
        router_keys = [str(key) for key in manifest["router_keys"]]
        manifest_steps = manifest["steps"]

        steps: dict[int, StepRoutes] = {}
        for step_index in range(num_steps):
            step_manifest = manifest_steps[str(step_index)]
            step_file = base_dir / step_manifest["file"]
            if not step_file.exists():
                raise FileNotFoundError(
                    f"Missing routing replay step file for step={step_index}: {step_file}"
                )
            step_tensors = load_file(str(step_file))
            if GLOBAL_TOKEN_UIDS_KEY not in step_tensors:
                raise RuntimeError(
                    f"Step file missing '{GLOBAL_TOKEN_UIDS_KEY}': {step_file}"
                )
            global_token_uids = step_tensors[GLOBAL_TOKEN_UIDS_KEY]

            routers: dict[str, StepRouterRoutes] = {}
            for router_key in router_keys:
                router_step_manifest = step_manifest["routers"].get(router_key)
                if router_step_manifest is None:
                    raise RuntimeError(
                        f"Step manifest missing router_key='{router_key}' for step={step_index}"
                    )
                calls: dict[int, RouterCallRoute] = {}
                for call_index_raw, call_manifest in router_step_manifest.items():
                    call_index = int(call_index_raw)
                    expert_indices_key = _build_tensor_key(
                        router_key, call_index, "expert_indices"
                    )
                    expert_probs_key = _build_tensor_key(
                        router_key, call_index, "expert_probs"
                    )
                    expert_mask_key = _build_tensor_key(
                        router_key, call_index, "expert_mask"
                    )
                    routing_map_key = _build_tensor_key(
                        router_key, call_index, "routing_map"
                    )
                    if expert_indices_key not in step_tensors:
                        raise RuntimeError(
                            f"Missing tensor key '{expert_indices_key}' in {step_file}"
                        )
                    if expert_probs_key not in step_tensors:
                        raise RuntimeError(
                            f"Missing tensor key '{expert_probs_key}' in {step_file}"
                        )
                    if expert_mask_key not in step_tensors:
                        raise RuntimeError(
                            f"Missing tensor key '{expert_mask_key}' in {step_file}"
                        )
                    routing_map = (
                        step_tensors[routing_map_key]
                        if routing_map_key in step_tensors
                        else None
                    )
                    calls[call_index] = RouterCallRoute(
                        expert_indices=step_tensors[expert_indices_key],
                        expert_probs=step_tensors[expert_probs_key],
                        expert_mask=step_tensors[expert_mask_key],
                        routing_map=routing_map,
                        num_experts=int(call_manifest["num_experts"]),
                        sample_index=call_manifest.get("sample_index"),
                        micro_slot=call_manifest.get("micro_slot"),
                    )
                routers[router_key] = StepRouterRoutes(calls=calls)
            steps[step_index] = StepRoutes(
                routers=routers,
                global_token_uids=global_token_uids,
            )

        return cls(
            format_version=ROUTER_KEY_FORMAT_VERSION,
            topology=topology,
            num_steps=num_steps,
            max_topk=max_topk,
            router_keys=router_keys,
            steps=steps,
        )

    def to_dir(self, bundle_dir: str | Path) -> None:
        base_dir = Path(bundle_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        manifest_steps: dict[str, dict[str, Any]] = {}
        for step_index in range(self.num_steps):
            step_routes = self.steps[step_index]
            step_file_name = f"step_{_normalize_step_index(step_index)}.safetensors"
            step_file_path = base_dir / step_file_name
            step_tensors: dict[str, torch.Tensor] = {
                GLOBAL_TOKEN_UIDS_KEY: _to_tensor_cpu_contiguous(
                    step_routes.global_token_uids, dtype=torch.int64
                )
            }
            step_manifest_routers: dict[str, dict[str, dict[str, int]]] = {}
            for router_key in self.router_keys:
                router_routes = step_routes.routers[router_key]
                call_manifest: dict[str, dict[str, int]] = {}
                for call_index, route in sorted(router_routes.calls.items()):
                    step_tensors[
                        _build_tensor_key(router_key, call_index, "expert_indices")
                    ] = _to_tensor_cpu_contiguous(
                        route.expert_indices, dtype=torch.int32
                    )
                    step_tensors[
                        _build_tensor_key(router_key, call_index, "expert_probs")
                    ] = _to_tensor_cpu_contiguous(
                        route.expert_probs, dtype=torch.float32
                    )
                    step_tensors[
                        _build_tensor_key(router_key, call_index, "expert_mask")
                    ] = _to_tensor_cpu_contiguous(route.expert_mask, dtype=torch.bool)
                    if route.routing_map is not None:
                        step_tensors[
                            _build_tensor_key(router_key, call_index, "routing_map")
                        ] = _to_tensor_cpu_contiguous(
                            route.routing_map, dtype=torch.bool
                        )
                    call_entry: dict[str, int] = {"num_experts": route.num_experts}
                    if route.sample_index is not None:
                        call_entry["sample_index"] = int(route.sample_index)
                    if route.micro_slot is not None:
                        call_entry["micro_slot"] = int(route.micro_slot)
                    call_manifest[str(call_index)] = call_entry
                step_manifest_routers[router_key] = call_manifest
            save_file(step_tensors, str(step_file_path))
            manifest_steps[str(step_index)] = {
                "file": step_file_name,
                "routers": step_manifest_routers,
            }

        manifest = {
            "format_version": ROUTER_KEY_FORMAT_VERSION,
            "topology": self.topology.model_dump(mode="json"),
            "num_steps": self.num_steps,
            "max_topk": self.max_topk,
            "router_keys": self.router_keys,
            "steps": manifest_steps,
        }
        with (base_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)


class LocalTokenIndexer(Protocol):
    def build_local_token_uids(
        self,
        *,
        global_token_uids: torch.Tensor,
        num_local_tokens: int,
        sequence_parallel: bool,
        context_parallel_size: int,
    ) -> torch.Tensor:
        """Build local token uid order for current rank."""


class TopologyAwareLocalTokenIndexer:
    def __init__(self, parallel_state_module: Any | None = None) -> None:
        self._parallel_state = parallel_state_module

    def _ps(self) -> Any:
        if self._parallel_state is not None:
            return self._parallel_state
        from megatron.core import parallel_state as ps

        self._parallel_state = ps
        return ps

    def build_local_token_uids(
        self,
        *,
        global_token_uids: torch.Tensor,
        num_local_tokens: int,
        sequence_parallel: bool,
        context_parallel_size: int,
    ) -> torch.Tensor:
        ps = self._ps()

        local_uids = global_token_uids.to(dtype=torch.int64, device="cpu").view(1, -1)

        cp_size = int(ps.get_context_parallel_world_size())
        if context_parallel_size > 1 and cp_size > 1:
            from megatron.core.utils import get_batch_on_this_cp_rank

            local_uids = get_batch_on_this_cp_rank({"tokens": local_uids})["tokens"]

        tp_size = int(ps.get_tensor_model_parallel_world_size())
        tp_rank = int(ps.get_tensor_model_parallel_rank()) if tp_size > 1 else 0
        if sequence_parallel and tp_size > 1:
            tokens_per_tp_rank = local_uids.shape[1] // tp_size
            start = tp_rank * tokens_per_tp_rank
            local_uids = local_uids[:, start : start + tokens_per_tp_rank]

        return local_uids.reshape(-1).contiguous()


_ACTIVE_ROUTING_REPLAY_CONTROLLER: MoeRoutingReplayController | None = None


def _active_routing_replay_controller() -> MoeRoutingReplayController | None:
    return _ACTIVE_ROUTING_REPLAY_CONTROLLER


def _dispatcher_local_token_uids(
    controller: MoeRoutingReplayController,
    dispatcher: Any,
    *,
    num_local_tokens: int,
) -> torch.Tensor:
    step_routes = controller._active_step_routes
    if step_routes is None:
        raise RuntimeError("Routing replay dispatcher used without an active step")
    local_uids = controller.local_token_indexer.build_local_token_uids(
        global_token_uids=step_routes.global_token_uids,
        num_local_tokens=num_local_tokens,
        sequence_parallel=bool(
            getattr(getattr(dispatcher, "config", None), "sequence_parallel", False)
        ),
        context_parallel_size=int(
            getattr(getattr(dispatcher, "config", None), "context_parallel_size", 1)
        ),
    )
    if int(local_uids.numel()) != num_local_tokens:
        raise RuntimeError(
            "Local routing replay uid count mismatch: "
            f"expected={num_local_tokens}, got={int(local_uids.numel())}"
        )
    sample_index = getattr(controller, "_active_sample_index", None)
    uid_span = int(step_routes.global_token_uids.numel())
    if isinstance(sample_index, int) and sample_index >= 0 and uid_span > 0:
        local_uids = local_uids + sample_index * uid_span
    return local_uids


def _trace_row_uids_from_source(source: Any) -> tuple[torch.Tensor | None, int | None]:
    row_token_uids = getattr(source, TRACE_ROW_TOKEN_UIDS_ATTR, None)
    if not isinstance(row_token_uids, torch.Tensor):
        return None, None
    uid_span = getattr(source, TRACE_UID_SPAN_ATTR, None)
    uid_span_int = uid_span if isinstance(uid_span, int) and uid_span > 0 else None
    return row_token_uids, uid_span_int


def _attach_trace_row_uids(
    target: Any,
    *,
    row_token_uids: torch.Tensor,
    uid_span: int | None,
) -> None:
    setattr(
        target,
        TRACE_ROW_TOKEN_UIDS_ATTR,
        row_token_uids.detach().to(device="cpu", dtype=torch.int64).reshape(-1),
    )
    setattr(target, TRACE_UID_SPAN_ATTR, uid_span)


def _canonicalize_expert_token_order(
    expert_inputs: torch.Tensor,
    expert_probs: torch.Tensor,
    expert_token_uids: torch.Tensor,
    *,
    tokens_per_expert: torch.Tensor | list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(tokens_per_expert, torch.Tensor):
        counts = [int(count) for count in tokens_per_expert.tolist()]
    else:
        counts = [int(count) for count in tokens_per_expert]

    if sum(counts) != int(expert_token_uids.numel()):
        raise RuntimeError(
            "Expert token uid count mismatch after dispatch: "
            f"uids={int(expert_token_uids.numel())}, "
            f"tokens_per_expert_sum={sum(counts)}"
        )

    order_segments: list[torch.Tensor] = []
    cursor = 0
    for count in counts:
        if count <= 1:
            order_segments.append(
                torch.arange(cursor, cursor + count, dtype=torch.long)
            )
            cursor += count
            continue
        segment_uids = expert_token_uids[cursor : cursor + count].to(device="cpu")
        segment_order = torch.argsort(segment_uids, stable=True) + cursor
        order_segments.append(segment_order)
        cursor += count

    if not order_segments:
        empty = torch.empty(0, dtype=torch.long)
        return expert_inputs, expert_probs, expert_token_uids, empty

    canonical_order_cpu = torch.cat(order_segments, dim=0)
    inverse_order_cpu = torch.empty_like(canonical_order_cpu)
    inverse_order_cpu[canonical_order_cpu] = torch.arange(
        canonical_order_cpu.numel(), dtype=torch.long
    )

    canonical_order = canonical_order_cpu.to(
        device=expert_inputs.device, dtype=torch.long
    )
    reordered_inputs = expert_inputs.index_select(0, canonical_order)
    reordered_probs = expert_probs.index_select(0, canonical_order)
    reordered_uids = expert_token_uids.index_select(
        0,
        canonical_order_cpu.to(device=expert_token_uids.device, dtype=torch.long),
    )
    return (
        reordered_inputs,
        reordered_probs,
        reordered_uids,
        inverse_order_cpu,
    )


def _canonical_trace_row_uids(
    expert_token_uids: torch.Tensor,
    *,
    tokens_per_expert: torch.Tensor | list[int],
    local_expert_indices: list[int] | tuple[int, ...] | None,
    sample_uid_span: int,
    num_experts: int,
) -> tuple[torch.Tensor, int]:
    if isinstance(tokens_per_expert, torch.Tensor):
        counts = [int(count) for count in tokens_per_expert.tolist()]
    else:
        counts = [int(count) for count in tokens_per_expert]

    expert_indices = (
        [int(expert_index) for expert_index in local_expert_indices]
        if local_expert_indices is not None
        else list(range(len(counts)))
    )
    if len(expert_indices) != len(counts):
        raise RuntimeError(
            "Local expert index metadata mismatch: "
            f"num_expert_indices={len(expert_indices)}, num_counts={len(counts)}"
        )
    row_uid_span = sample_uid_span * max(int(num_experts), 1)
    row_uid_chunks: list[torch.Tensor] = []
    cursor = 0
    for global_expert_id, count in zip(expert_indices, counts):
        count_int = int(count)
        segment = expert_token_uids[cursor : cursor + count_int].to(dtype=torch.int64)
        sample_ids = torch.div(segment, sample_uid_span, rounding_mode="floor")
        local_token_ids = torch.remainder(segment, sample_uid_span)
        row_uid_chunks.append(
            sample_ids * row_uid_span
            + int(global_expert_id) * sample_uid_span
            + local_token_ids
        )
        cursor += count_int
    if cursor != int(expert_token_uids.numel()):
        raise RuntimeError(
            "Canonical trace row uid construction did not consume all expert rows: "
            f"consumed={cursor}, total={int(expert_token_uids.numel())}"
        )
    if not row_uid_chunks:
        return expert_token_uids.new_empty((0,), dtype=torch.int64), row_uid_span
    return torch.cat(row_uid_chunks, dim=0).contiguous(), row_uid_span


def _patch_alltoall_dispatcher_preprocess() -> None:
    try:
        from megatron.core.transformer.moe.experts import TEGroupedMLP
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAlltoAllTokenDispatcher,
        )

        from art.megatron.lora import MLPExpertsLinearFC2LoRA
    except Exception:
        return

    if hasattr(MoEAlltoAllTokenDispatcher, "_art_router_replay_preprocess_patched"):
        return

    original_preprocess = MoEAlltoAllTokenDispatcher.preprocess
    original_dispatch_preprocess = MoEAlltoAllTokenDispatcher.dispatch_preprocess
    original_token_dispatch = MoEAlltoAllTokenDispatcher.token_dispatch
    original_dispatch_postprocess = MoEAlltoAllTokenDispatcher.dispatch_postprocess
    original_combine_preprocess = MoEAlltoAllTokenDispatcher.combine_preprocess
    original_te_grouped_mlp_forward = TEGroupedMLP.forward
    original_fc2_forward = MLPExpertsLinearFC2LoRA.forward

    def patched_preprocess(
        self: Any, routing_map: torch.Tensor, *args: Any, **kwargs: Any
    ):
        result = original_preprocess(self, routing_map, *args, **kwargs)
        if (
            not getattr(self, "drop_and_pad", False)
            and getattr(self.config, "moe_expert_capacity_factor", None) is None
            and not (
                getattr(self.config, "moe_router_padding_for_quantization", None)
                or getattr(self.config, "moe_router_padding_for_fp8", None)
            )
        ):
            self.num_out_tokens = int(routing_map.sum().item())
        return result

    def patched_dispatch_preprocess(
        self: Any,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ):
        result = original_dispatch_preprocess(self, hidden_states, routing_map, probs)
        self._art_replay_permuted_local_token_uids = None
        self._art_replay_global_input_token_uids = None
        self._art_replay_expert_input_inverse_permutation = None

        controller = _active_routing_replay_controller()
        if controller is None:
            return result

        local_token_uids = _dispatcher_local_token_uids(
            controller,
            self,
            num_local_tokens=int(routing_map.shape[0]),
        )
        permuted_local_uids = permute(
            local_token_uids.to(
                device=hidden_states.device, dtype=torch.int64
            ).unsqueeze(-1),
            self.routing_map,
            num_out_tokens=self.num_out_tokens,
            fused=False,
            drop_and_pad=self.drop_and_pad,
        )[0]
        self._art_replay_permuted_local_token_uids = permuted_local_uids.reshape(
            -1
        ).contiguous()
        return result

    def patched_token_dispatch(
        self: Any,
        permutated_local_input_tokens: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        result = original_token_dispatch(
            self,
            permutated_local_input_tokens,
            permuted_probs,
        )
        controller = _active_routing_replay_controller()
        permuted_local_token_uids = getattr(
            self, "_art_replay_permuted_local_token_uids", None
        )
        if controller is None or permuted_local_token_uids is None:
            return result

        global_token_uids = permuted_local_token_uids.to(
            device=permutated_local_input_tokens.device, dtype=torch.int64
        ).unsqueeze(-1)
        if self.ep_size > 1:
            global_token_uids = all_to_all(
                self.ep_group,
                global_token_uids,
                self.output_splits,
                self.input_splits,
            )
        if self.tp_size > 1:
            output_split_sizes = (
                None
                if self.output_splits_tp is None
                else self.output_splits_tp.tolist()
            )
            global_token_uids = gather_from_sequence_parallel_region(
                global_token_uids,
                group=self.tp_group,
                output_split_sizes=output_split_sizes,
            )
        self._art_replay_global_input_token_uids = global_token_uids.reshape(
            -1
        ).contiguous()
        return result

    def patched_dispatch_postprocess(
        self: Any,
        global_input_tokens: torch.Tensor,
        global_probs: torch.Tensor,
    ):
        expert_inputs, tokens_per_expert, expert_probs = original_dispatch_postprocess(
            self,
            global_input_tokens,
            global_probs,
        )
        controller = _active_routing_replay_controller()
        global_input_token_uids = getattr(
            self, "_art_replay_global_input_token_uids", None
        )
        if controller is None or global_input_token_uids is None or self.drop_and_pad:
            return expert_inputs, tokens_per_expert, expert_probs

        expert_token_uids = global_input_token_uids
        if self.num_local_experts > 1:
            sorted_token_uids = sort_chunks_by_idxs(
                expert_token_uids.unsqueeze(-1),
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
                fused=False,
            )[0]
            expert_token_uids = sorted_token_uids.reshape(-1).contiguous()

        (
            expert_inputs,
            expert_probs,
            canonical_expert_token_uids,
            inverse_order_cpu,
        ) = _canonicalize_expert_token_order(
            expert_inputs,
            expert_probs,
            expert_token_uids,
            tokens_per_expert=tokens_per_expert,
        )
        self._art_replay_expert_input_inverse_permutation = inverse_order_cpu
        active_step_routes = controller._active_step_routes
        if active_step_routes is None:
            raise RuntimeError(
                "MoE replay dispatcher preprocess called before set_step"
            )
        trace_row_uids, trace_uid_span = _canonical_trace_row_uids(
            canonical_expert_token_uids,
            tokens_per_expert=tokens_per_expert,
            local_expert_indices=getattr(self, "local_expert_indices", None),
            sample_uid_span=int(active_step_routes.global_token_uids.numel()),
            num_experts=int(getattr(self, "num_experts", 1)),
        )
        _attach_trace_row_uids(
            expert_inputs,
            row_token_uids=trace_row_uids,
            uid_span=trace_uid_span,
        )
        return expert_inputs, tokens_per_expert, expert_probs

    def patched_combine_preprocess(self: Any, hidden_states: torch.Tensor):
        inverse_order_cpu = getattr(
            self, "_art_replay_expert_input_inverse_permutation", None
        )
        if inverse_order_cpu is not None and inverse_order_cpu.numel() > 0:
            hidden_states = hidden_states.index_select(
                0,
                inverse_order_cpu.to(device=hidden_states.device, dtype=torch.long),
            )
        self._art_replay_expert_input_inverse_permutation = None
        return original_combine_preprocess(self, hidden_states)

    def patched_te_grouped_mlp_forward(
        self: Any,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        row_token_uids, uid_span = _trace_row_uids_from_source(
            permuted_local_hidden_states
        )
        if row_token_uids is not None:
            _attach_trace_row_uids(
                self.linear_fc2,
                row_token_uids=row_token_uids,
                uid_span=uid_span,
            )
        return original_te_grouped_mlp_forward(
            self,
            permuted_local_hidden_states,
            tokens_per_expert,
            permuted_probs,
        )

    def patched_fc2_forward(
        self: Any,
        x: torch.Tensor,
        tokens_per_expert: list[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        row_token_uids, uid_span = _trace_row_uids_from_source(x)
        if row_token_uids is None:
            row_token_uids, uid_span = _trace_row_uids_from_source(self)
        if row_token_uids is not None:
            _attach_trace_row_uids(
                self.linear_fc2,
                row_token_uids=row_token_uids,
                uid_span=uid_span,
            )
            _attach_trace_row_uids(
                self.lora,
                row_token_uids=row_token_uids,
                uid_span=uid_span,
            )
        return original_fc2_forward(self, x, tokens_per_expert)

    setattr(MoEAlltoAllTokenDispatcher, "preprocess", patched_preprocess)
    setattr(
        MoEAlltoAllTokenDispatcher,
        "dispatch_preprocess",
        patched_dispatch_preprocess,
    )
    setattr(MoEAlltoAllTokenDispatcher, "token_dispatch", patched_token_dispatch)
    setattr(
        MoEAlltoAllTokenDispatcher,
        "dispatch_postprocess",
        patched_dispatch_postprocess,
    )
    setattr(
        MoEAlltoAllTokenDispatcher,
        "combine_preprocess",
        patched_combine_preprocess,
    )
    setattr(TEGroupedMLP, "forward", patched_te_grouped_mlp_forward)
    setattr(MLPExpertsLinearFC2LoRA, "forward", patched_fc2_forward)
    setattr(MoEAlltoAllTokenDispatcher, "_art_router_replay_preprocess_patched", True)


class MoeRoutingReplayController:
    def __init__(
        self,
        *,
        bundle: MoeRoutingReplayBundle,
        strict: bool,
        local_token_indexer: LocalTokenIndexer | None = None,
    ) -> None:
        self.bundle = bundle
        self.strict = strict
        self.local_token_indexer = (
            local_token_indexer or TopologyAwareLocalTokenIndexer()
        )

        self._active_step_index: int | None = None
        self._active_sample_index: int | None = None
        self._active_step_routes: StepRoutes | None = None
        self._router_call_cursors: dict[str, int] = {}
        self._router_call_sequences: dict[str, list[int]] = {}
        self._global_uid_to_row_index: dict[int, int] = {}
        self._local_router_keys: set[str] = set()
        self._active_micro_order: int | None = None

        self._patched_router_modules: list[dict[str, Any]] = []

    def install_router_patches(self, model_chunks: list[Any]) -> None:
        if self._patched_router_modules:
            return
        _patch_alltoall_dispatcher_preprocess()

        for chunk_index, chunk in enumerate(model_chunks):
            for module_name, module in chunk.named_modules():
                if ROUTER_NAME_TOKEN not in module_name:
                    continue
                if not hasattr(module, "routing"):
                    continue
                router_key = build_router_key_from_module_name(
                    chunk_index=chunk_index,
                    module_name=module_name,
                )
                if self.strict and router_key not in self.bundle.router_keys:
                    raise RuntimeError(
                        "Router key from model is missing in replay bundle: "
                        f"router_key='{router_key}'"
                    )

                original_routing = module.routing
                if getattr(module, "_art_router_replay_patched", False):
                    continue

                sequence_parallel = bool(
                    getattr(getattr(module, "config", None), "sequence_parallel", False)
                )
                context_parallel_size = int(
                    getattr(getattr(module, "config", None), "context_parallel_size", 1)
                )

                def routing_wrapper(
                    _module: Any,
                    logits: torch.Tensor,
                    *args: Any,
                    _router_key: str = router_key,
                    _sequence_parallel: bool = sequence_parallel,
                    _context_parallel_size: int = context_parallel_size,
                    **kwargs: Any,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    live_probs, live_routing_map = original_routing(
                        logits, *args, **kwargs
                    )
                    replay_probs, replay_routing_map = self.get_route_for_router(
                        router_key=_router_key,
                        logits=live_probs,
                        sequence_parallel=_sequence_parallel,
                        context_parallel_size=_context_parallel_size,
                    )
                    # same result, but autograd goes through
                    probs = (
                        live_probs
                        + (
                            replay_probs.to(
                                device=live_probs.device,
                                dtype=live_probs.dtype,
                            )
                            - live_probs
                        ).detach()
                    )
                    routing_map = replay_routing_map.to(
                        device=live_routing_map.device,
                        dtype=live_routing_map.dtype,
                    )
                    return probs, routing_map

                module.routing = types.MethodType(routing_wrapper, module)
                module._art_router_replay_patched = True
                self._local_router_keys.add(router_key)
                self._patched_router_modules.append(
                    {
                        "module": module,
                        "router_key": router_key,
                        "original_routing": original_routing,
                    }
                )

    def remove_router_patches(self) -> None:
        global _ACTIVE_ROUTING_REPLAY_CONTROLLER
        for item in self._patched_router_modules:
            module = item["module"]
            module.routing = item["original_routing"]
            if hasattr(module, "_art_router_replay_patched"):
                delattr(module, "_art_router_replay_patched")
        self._patched_router_modules.clear()
        self._local_router_keys.clear()
        if _ACTIVE_ROUTING_REPLAY_CONTROLLER is self:
            _ACTIVE_ROUTING_REPLAY_CONTROLLER = None

    def begin_micro(self, sample_index: int | None, micro_order: int) -> None:
        self._active_sample_index = sample_index
        self._active_micro_order = micro_order

    def set_step(
        self,
        *,
        step_index: int,
        sample_index: int | list[int | None],
        global_grad_accumulation_sequences: int | None = None,
    ) -> None:
        global _ACTIVE_ROUTING_REPLAY_CONTROLLER

        if step_index not in self.bundle.steps:
            raise RuntimeError(
                f"Replay bundle missing step_index={step_index}. "
                f"Available steps={sorted(self.bundle.steps.keys())}"
            )
        step_routes = self.bundle.steps[step_index]
        self._active_step_index = step_index
        if isinstance(sample_index, list):
            self._active_sample_index = next(
                (index for index in sample_index if index is not None),
                None,
            )
        else:
            self._active_sample_index = sample_index
        self._active_micro_order = None
        self._active_step_routes = step_routes
        for local_router_key in sorted(self._local_router_keys):
            if local_router_key not in step_routes.routers:
                raise RuntimeError(
                    "Replay bundle step is missing local router key: "
                    f"step={step_index}, router='{local_router_key}'"
                )
        self._router_call_cursors = {}
        self._router_call_sequences = {}
        local_call_keys = self._build_local_call_keys(
            sample_index=sample_index,
        )
        for router_key in sorted(self._local_router_keys):
            router_calls = step_routes.routers[router_key].calls
            if all(
                self._router_call_key(route) is not None
                for route in router_calls.values()
            ):
                calls_by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
                for call_index, route in sorted(router_calls.items()):
                    call_key = self._router_call_key(route)
                    assert call_key is not None
                    calls_by_key[call_key].append(call_index)
                call_sequence = []
                for call_key in local_call_keys:
                    if call_key is None:
                        continue
                    matching_call_indices = calls_by_key.get(call_key)
                    if not matching_call_indices:
                        raise RuntimeError(
                            "Replay router call sequence is missing local micro metadata: "
                            f"step={step_index}, router='{router_key}', call_key={call_key}"
                        )
                    call_sequence.extend(matching_call_indices)
            else:
                call_sequence = self._legacy_router_call_sequence(
                    step_index=step_index,
                    router_key=router_key,
                    sample_index=sample_index,
                    global_grad_accumulation_sequences=global_grad_accumulation_sequences,
                    total_calls=len(router_calls),
                )
            self._router_call_cursors[router_key] = 0
            self._router_call_sequences[router_key] = call_sequence
        self._global_uid_to_row_index = {
            int(uid.item()): row_index
            for row_index, uid in enumerate(step_routes.global_token_uids)
        }
        _ACTIVE_ROUTING_REPLAY_CONTROLLER = self

    def _build_local_call_keys(
        self,
        *,
        sample_index: int | list[int | None],
    ) -> list[tuple[str, int] | None]:
        if not isinstance(sample_index, list):
            if sample_index is None:
                return [self._dummy_micro_call_key(local_micro_index=0)]
            return [("sample", int(sample_index))]
        return [
            self._sample_or_dummy_call_key(
                global_sample_index=global_sample_index,
                local_micro_index=local_micro_index,
            )
            for local_micro_index, global_sample_index in enumerate(sample_index)
        ]

    def _sample_or_dummy_call_key(
        self,
        *,
        global_sample_index: int | None,
        local_micro_index: int,
    ) -> tuple[str, int] | None:
        if global_sample_index is not None:
            return ("sample", int(global_sample_index))
        return self._dummy_micro_call_key(local_micro_index=local_micro_index)

    def _dummy_micro_call_key(
        self,
        *,
        local_micro_index: int,
    ) -> tuple[str, int]:
        from megatron.core import parallel_state as ps

        dp_rank = int(ps.get_data_parallel_rank())
        dp_world_size = int(ps.get_data_parallel_world_size())
        micro_slot = local_micro_index * dp_world_size + dp_rank
        return ("dummy_micro_slot", micro_slot)

    @staticmethod
    def _router_call_key(route: RouterCallRoute) -> tuple[str, int] | None:
        if route.sample_index is not None:
            return ("sample", int(route.sample_index))
        if route.micro_slot is not None:
            return ("dummy_micro_slot", int(route.micro_slot))
        return None

    @staticmethod
    def _legacy_router_call_sequence(
        *,
        step_index: int,
        router_key: str,
        sample_index: int | list[int | None],
        global_grad_accumulation_sequences: int | None,
        total_calls: int,
    ) -> list[int]:
        step_sample_count = global_grad_accumulation_sequences
        if step_sample_count is None:
            if isinstance(sample_index, list):
                step_sample_count = len(
                    [index for index in sample_index if index is not None]
                )
            else:
                step_sample_count = 1
        if step_sample_count <= 0 or total_calls % step_sample_count != 0:
            raise RuntimeError(
                "Replay router call count is not divisible by step sample count: "
                f"step={step_index}, router='{router_key}', "
                f"total_calls={total_calls}, step_sample_count={step_sample_count}"
            )
        calls_per_sample = total_calls // step_sample_count
        step_base_sample_index = step_index * step_sample_count
        if isinstance(sample_index, list):
            call_sequence: list[int] = []
            for global_sample_index in sample_index:
                if global_sample_index is None:
                    continue
                sample_offset = int(global_sample_index) - step_base_sample_index
                if sample_offset < 0 or sample_offset >= step_sample_count:
                    raise RuntimeError(
                        "Replay router call index is outside the step-local range: "
                        f"step={step_index}, router='{router_key}', "
                        f"global_sample_index={global_sample_index}, "
                        f"step_base_sample_index={step_base_sample_index}, "
                        f"step_sample_count={step_sample_count}"
                    )
                call_start = sample_offset * calls_per_sample
                call_sequence.extend(range(call_start, call_start + calls_per_sample))
            return call_sequence

        sample_offset = int(sample_index) - step_base_sample_index
        if sample_offset < 0 or sample_offset >= step_sample_count:
            raise RuntimeError(
                "Replay router call index is outside the step-local range: "
                f"step={step_index}, router='{router_key}', "
                f"sample_index={sample_index}, "
                f"step_sample_count={step_sample_count}"
            )
        call_start = sample_offset * calls_per_sample
        return list(range(call_start, call_start + calls_per_sample))

    def finalize_step(self) -> None:
        global _ACTIVE_ROUTING_REPLAY_CONTROLLER
        if self._active_step_routes is None:
            raise RuntimeError("finalize_step called before set_step")
        for router_key in sorted(self._local_router_keys):
            consumed = self._router_call_cursors.get(router_key, 0)
            call_sequence = self._router_call_sequences.get(router_key)
            if call_sequence is None:
                raise RuntimeError(
                    "Routing replay call sequence missing for router key: "
                    f"step={self._active_step_index}, router='{router_key}'"
                )
            if consumed != len(call_sequence):
                raise RuntimeError(
                    "Routing replay step consumption mismatch: "
                    f"step={self._active_step_index}, router='{router_key}', "
                    f"consumed={consumed}, expected={len(call_sequence)}"
                )
        self._active_step_index = None
        self._active_sample_index = None
        self._active_step_routes = None
        self._router_call_cursors = {}
        self._router_call_sequences = {}
        self._global_uid_to_row_index = {}
        self._active_micro_order = None
        if _ACTIVE_ROUTING_REPLAY_CONTROLLER is self:
            _ACTIVE_ROUTING_REPLAY_CONTROLLER = None

    def get_route_for_router(
        self,
        *,
        router_key: str,
        logits: torch.Tensor,
        sequence_parallel: bool,
        context_parallel_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_routes = self._active_step_routes
        if step_routes is None:
            raise RuntimeError(
                "Routing replay get_route_for_router called before set_step"
            )
        call_cursor = self._router_call_cursors.get(router_key, 0)
        call_sequence = self._router_call_sequences.get(router_key)
        if call_sequence is None:
            raise RuntimeError(
                "Routing replay call sequence missing for router key: "
                f"step={self._active_step_index}, router='{router_key}'"
            )
        router_calls = step_routes.routers[router_key].calls
        if call_cursor >= len(call_sequence):
            raise RuntimeError(
                "Routing replay call cursor exceeded local call sequence: "
                f"step={self._active_step_index}, router='{router_key}', "
                f"call_cursor={call_cursor}, sequence_length={len(call_sequence)}"
            )
        route = router_calls[call_sequence[call_cursor]]
        self._router_call_cursors[router_key] = call_cursor + 1

        num_local_tokens = int(logits.shape[0])
        num_experts = int(logits.shape[1])

        local_uids = self.local_token_indexer.build_local_token_uids(
            global_token_uids=step_routes.global_token_uids,
            num_local_tokens=num_local_tokens,
            sequence_parallel=sequence_parallel,
            context_parallel_size=context_parallel_size,
        )
        row_index_tensor = torch.tensor(
            [self._global_uid_to_row_index[int(uid)] for uid in local_uids.tolist()],
            dtype=torch.int64,
        )

        local_indices = route.expert_indices.index_select(0, row_index_tensor)
        local_probs = route.expert_probs.index_select(0, row_index_tensor)
        local_mask = route.expert_mask.index_select(0, row_index_tensor)

        probs = torch.zeros(
            (num_local_tokens, num_experts),
            dtype=logits.dtype,
            device=logits.device,
        )
        routing_map = torch.zeros(
            (num_local_tokens, num_experts),
            dtype=torch.bool,
            device=logits.device,
        )

        if local_indices.numel() > 0:
            indices_device = local_indices.to(device=logits.device, dtype=torch.long)
            probs_device = local_probs.to(device=logits.device, dtype=logits.dtype)
            mask_device = local_mask.to(device=logits.device, dtype=torch.bool)
            row_index_device = (
                torch.arange(num_local_tokens, device=logits.device)
                .unsqueeze(1)
                .expand_as(indices_device)
            )

            selected_rows = row_index_device[mask_device]
            selected_cols = indices_device[mask_device]
            selected_probs = probs_device[mask_device]

            if selected_rows.numel() > 0:
                probs[selected_rows, selected_cols] = selected_probs
                routing_map[selected_rows, selected_cols] = True

        return probs, routing_map


def _compact_route_from_dense(
    probs_2d: torch.Tensor,
    routing_map_2d: torch.Tensor,
) -> RouterCallRoute:
    num_tokens, num_experts = probs_2d.shape
    if num_tokens == 0:
        return RouterCallRoute(
            expert_indices=torch.zeros((0, 0), dtype=torch.int32),
            expert_probs=torch.zeros((0, 0), dtype=torch.float32),
            expert_mask=torch.zeros((0, 0), dtype=torch.bool),
            num_experts=num_experts,
        )

    max_topk = int(routing_map_2d.sum(dim=1).max().item())
    expert_indices = torch.zeros((num_tokens, max_topk), dtype=torch.int32)
    expert_probs = torch.zeros((num_tokens, max_topk), dtype=torch.float32)
    expert_mask = torch.zeros((num_tokens, max_topk), dtype=torch.bool)
    for token_index in range(num_tokens):
        expert_ids = torch.nonzero(
            routing_map_2d[token_index], as_tuple=False
        ).flatten()
        slot_count = int(expert_ids.numel())
        if slot_count == 0:
            continue
        expert_indices[token_index, :slot_count] = expert_ids.to(torch.int32)
        expert_probs[token_index, :slot_count] = probs_2d[token_index, expert_ids].to(
            torch.float32
        )
        expert_mask[token_index, :slot_count] = True

    return RouterCallRoute(
        expert_indices=expert_indices,
        expert_probs=expert_probs,
        expert_mask=expert_mask,
        num_experts=num_experts,
    )


def build_bundle_from_forward_trace_dir(
    *,
    traces_dir: str | Path,
    num_steps: int,
    topology: ParallelTopology,
) -> MoeRoutingReplayBundle:
    """Build a replay bundle from saved forward traces for the correctness harness.

    This helper is intended for testing/oracle routing replay workflows and is not
    part of inference routing capture/export.
    """
    trace_dir = Path(traces_dir)
    steps: dict[int, StepRoutes] = {}
    router_keys_union: set[str] = set()
    max_topk = 0

    for step_index in range(num_steps):
        trace_path = trace_dir / f"forward_trace_step_{step_index:03d}.pt"
        if not trace_path.exists():
            raise FileNotFoundError(
                f"Missing forward trace for step={step_index}: {trace_path}"
            )
        step_trace: dict[str, list[dict[str, Any]]] = torch.load(
            trace_path, map_location="cpu", weights_only=False
        )

        step_routers: dict[str, StepRouterRoutes] = {}
        step_global_tokens: int | None = None
        for module_name in sorted(step_trace.keys()):
            if ROUTER_NAME_TOKEN not in module_name:
                continue
            router_key = build_router_key_from_trace_name(module_name)
            router_calls: dict[int, RouterCallRoute] = {}
            for call_index, call_entry in enumerate(step_trace[module_name]):
                output = call_entry.get("output")
                probs_2d, routing_map_2d = _extract_router_output_tensors(output)
                compact_route = _compact_route_from_dense(probs_2d, routing_map_2d)
                sample_index, micro_slot = _trace_call_route_metadata(call_entry)
                compact_route.sample_index = sample_index
                compact_route.micro_slot = micro_slot
                router_calls[call_index] = compact_route
                max_topk = max(max_topk, compact_route.max_topk)
                token_count = compact_route.num_global_tokens
                if step_global_tokens is None:
                    step_global_tokens = token_count
                elif step_global_tokens != token_count:
                    raise RuntimeError(
                        "Inconsistent token count across routers within step: "
                        f"step={step_index}, expected={step_global_tokens}, got={token_count}, "
                        f"router='{router_key}', call={call_index}"
                    )

            if not router_calls:
                raise RuntimeError(
                    f"Router trace has no calls for module '{module_name}' at step={step_index}"
                )
            step_routers[router_key] = StepRouterRoutes(calls=router_calls)
            router_keys_union.add(router_key)

        if not step_routers:
            raise RuntimeError(
                f"No router traces found for step={step_index} in {trace_path}"
            )
        if step_global_tokens is None:
            raise RuntimeError(
                f"Could not infer token count for step={step_index} from router traces"
            )
        global_token_uids = torch.arange(step_global_tokens, dtype=torch.int64)
        steps[step_index] = StepRoutes(
            routers=step_routers,
            global_token_uids=global_token_uids,
        )

    router_keys = sorted(router_keys_union)
    for step_index, step_routes in steps.items():
        if set(step_routes.routers.keys()) != set(router_keys):
            raise RuntimeError(
                f"Step {step_index} router keys differ from global set: "
                f"step_keys={sorted(step_routes.routers.keys())}, router_keys={router_keys}"
            )

    return MoeRoutingReplayBundle(
        format_version=ROUTER_KEY_FORMAT_VERSION,
        topology=topology,
        num_steps=num_steps,
        max_topk=max_topk,
        router_keys=router_keys,
        steps=steps,
    )
