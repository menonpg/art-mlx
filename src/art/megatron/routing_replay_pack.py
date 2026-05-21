from __future__ import annotations

import math
import os

import torch

from art.preprocessing.pack import PackedTensors

from .routing_replay import (
    MoeRoutingReplayBundle,
    ParallelTopology,
    RouterCallRoute,
    StepRouterRoutes,
    StepRoutes,
)


def build_moe_routing_replay_bundle_from_packed_tensors(
    *,
    packed_tensors: PackedTensors,
    global_grad_accumulation_sequences: int,
    topology: ParallelTopology | None = None,
) -> MoeRoutingReplayBundle:
    if "moe_routing_expert_indices" not in packed_tensors:
        raise RuntimeError("Packed tensors do not contain MoE routing expert indices")
    if "moe_routing_token_mask" not in packed_tensors:
        raise RuntimeError("Packed tensors do not contain MoE routing token mask")
    if global_grad_accumulation_sequences <= 0:
        raise RuntimeError(
            "global_grad_accumulation_sequences must be positive when building "
            f"MoE routing replay bundles, got {global_grad_accumulation_sequences}"
        )
    expert_indices = packed_tensors["moe_routing_expert_indices"]
    token_mask = packed_tensors["moe_routing_token_mask"]
    if expert_indices.ndim != 4:
        raise RuntimeError(
            "moe_routing_expert_indices must have shape "
            f"[num_sequences, sequence_length, num_layers, topk], got "
            f"{tuple(expert_indices.shape)}"
        )
    if token_mask.shape != expert_indices.shape[:2]:
        raise RuntimeError(
            "moe_routing_token_mask shape must match packed route tokens, got "
            f"{tuple(token_mask.shape)} vs {tuple(expert_indices.shape[:2])}"
        )
    num_sequences = int(expert_indices.shape[0])
    sequence_length = int(expert_indices.shape[1])
    num_layers = int(expert_indices.shape[2])
    topk = int(expert_indices.shape[3])
    num_experts = int(
        packed_tensors.get("moe_routing_num_experts", 0)
        or int(expert_indices.max().item()) + 1
    )
    if topk > num_experts:
        raise RuntimeError(
            f"MoE routing topk cannot exceed num_experts: topk={topk}, "
            f"num_experts={num_experts}"
        )
    replay_padding_row = torch.arange(topk, dtype=expert_indices.dtype)
    group_ids = packed_tensors["group_ids"]
    parent_ids = packed_tensors["parent_ids"]
    non_padding = group_ids != -1
    next_group_ids = torch.nn.functional.pad(group_ids[:, 1:], (0, 1), value=-1)
    terminal_completion = (
        non_padding & (group_ids != parent_ids) & (group_ids != next_group_ids)
    )
    unexpected_missing = non_padding & ~token_mask & ~terminal_completion
    if bool(unexpected_missing.any().item()):
        raise RuntimeError(
            "Packed tensors are missing MoE routes outside terminal completion "
            f"tokens: missing_rows={int(unexpected_missing.sum().item())}"
        )

    router_keys = [
        f"chunk_00.layer_{layer_index:04d}.mlp.router"
        for layer_index in range(num_layers)
    ]
    steps: dict[int, StepRoutes] = {}
    num_steps = math.ceil(num_sequences / global_grad_accumulation_sequences)
    for step_index in range(num_steps):
        start = step_index * global_grad_accumulation_sequences
        end = start + global_grad_accumulation_sequences
        routers: dict[str, StepRouterRoutes] = {}
        for layer_index, router_key in enumerate(router_keys):
            calls: dict[int, RouterCallRoute] = {}
            for offset, sample_index in enumerate(range(start, end)):
                if sample_index < num_sequences:
                    route_indices = expert_indices[
                        sample_index, :, layer_index, :
                    ].clone()
                    missing_rows = ~token_mask[sample_index]
                    if bool(missing_rows.any().item()):
                        # Megatron Core RouterReplay replays only top-k ids and does
                        # not consume expert_mask. Rows without vLLM routes are
                        # allowed only for terminal completion tokens, which are not
                        # scored, but they still flow through Megatron's forward.
                        # Use valid unique fallback ids so Megatron's dense
                        # routing_map keeps exactly topk entries per token.
                        route_indices[missing_rows] = replay_padding_row
                    route_mask = token_mask[sample_index, :, None].expand_as(
                        route_indices
                    )
                    calls[offset] = RouterCallRoute(
                        expert_indices=route_indices,
                        expert_mask=route_mask,
                        num_experts=num_experts,
                        sample_index=sample_index,
                    )
                else:
                    route_indices = replay_padding_row.expand(
                        sequence_length, topk
                    ).clone()
                    calls[offset] = RouterCallRoute(
                        expert_indices=route_indices,
                        expert_mask=torch.ones_like(route_indices, dtype=torch.bool),
                        num_experts=max(num_experts, 1),
                        micro_slot=offset,
                    )
            routers[router_key] = StepRouterRoutes(calls=calls)
        steps[step_index] = StepRoutes(
            routers=routers,
            global_token_uids=torch.arange(sequence_length, dtype=torch.int64),
        )
    return MoeRoutingReplayBundle(
        topology=topology or parallel_topology_from_env(),
        num_steps=num_steps,
        max_topk=topk,
        router_keys=router_keys,
        steps=steps,
    )


def parallel_topology_from_env() -> ParallelTopology:
    tp = _env_int("ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE", 1)
    ep = _env_int("ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE", 1)
    etp = _env_int(
        "ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE",
        _env_int("ART_MEGATRON_EXPERT_TENSOR_MODEL_PARALLEL_SIZE", 1),
    )
    cp = _env_int("ART_MEGATRON_CONTEXT_PARALLEL_SIZE", 1)
    pp = _env_int("ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE", 1)
    return ParallelTopology(tp=tp, ep=ep, etp=etp, dp=1, sp=tp > 1, cp=cp, pp=pp)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)
