from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from art.megatron.routing_replay import (
    ROUTER_NAME_TOKEN,
    MoeRoutingReplayBundle,
    ParallelTopology,
    RouterCallRoute,
    StepRouterRoutes,
    StepRoutes,
    build_router_key_from_trace_name,
)


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


def _compact_route_from_dense(
    _probs_2d: torch.Tensor,
    routing_map_2d: torch.Tensor,
) -> RouterCallRoute:
    num_tokens, num_experts = routing_map_2d.shape
    if num_tokens == 0:
        return RouterCallRoute(
            expert_indices=torch.zeros((0, 0), dtype=torch.int32),
            expert_mask=torch.zeros((0, 0), dtype=torch.bool),
            num_experts=num_experts,
        )
    topk_by_row = routing_map_2d.sum(dim=1)
    if not bool((topk_by_row == topk_by_row[0]).all().item()):
        raise RuntimeError(
            "Megatron Core RouterReplay requires a fixed topk for every token row; "
            f"observed row counts={torch.unique(topk_by_row).tolist()}"
        )
    topk = int(topk_by_row[0].item())
    expert_indices = torch.zeros((num_tokens, topk), dtype=torch.int32)
    for token_index in range(num_tokens):
        expert_ids = torch.nonzero(
            routing_map_2d[token_index], as_tuple=False
        ).flatten()
        if int(expert_ids.numel()) != topk:
            raise RuntimeError(
                f"Unexpected route topk for token={token_index}: "
                f"expected={topk}, got={int(expert_ids.numel())}"
            )
        expert_indices[token_index] = expert_ids.to(torch.int32)
    return RouterCallRoute(
        expert_indices=expert_indices,
        expert_mask=torch.ones_like(expert_indices, dtype=torch.bool),
        num_experts=num_experts,
    )


def build_bundle_from_forward_trace_dir(
    *,
    traces_dir: str | Path,
    num_steps: int,
    topology: ParallelTopology,
) -> MoeRoutingReplayBundle:
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
                probs_2d, routing_map_2d = _extract_router_output_tensors(
                    call_entry.get("output")
                )
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
                        f"step={step_index}, expected={step_global_tokens}, "
                        f"got={token_count}, router='{router_key}', call={call_index}"
                    )
            if not router_calls:
                raise RuntimeError(
                    f"Router trace has no calls for module '{module_name}' "
                    f"at step={step_index}"
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
        steps[step_index] = StepRoutes(
            routers=step_routers,
            global_token_uids=torch.arange(step_global_tokens, dtype=torch.int64),
        )

    router_keys = sorted(router_keys_union)
    for step_index, step_routes in steps.items():
        if set(step_routes.routers) != set(router_keys):
            raise RuntimeError(
                f"Step {step_index} router keys differ from global set: "
                f"step_keys={sorted(step_routes.routers)}, router_keys={router_keys}"
            )

    return MoeRoutingReplayBundle(
        topology=topology,
        num_steps=num_steps,
        max_topk=max_topk,
        router_keys=router_keys,
        steps=steps,
    )
