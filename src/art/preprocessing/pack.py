import os
import random
import time
from typing import Any, cast

import numpy as np
import torch
from typing_extensions import NotRequired, TypedDict, Unpack

from ..types import Verbosity
from .moe_routing import (
    MISSING_EXPERT_ID,
    MoeRouteArray,
    MoeRouteSegments,
    MoeRoutingPackStats,
    PackedMoeRoutingReplay,
)
from .tokenize import TokenizedResult


class PackedTensors(TypedDict):
    tokens: torch.Tensor
    group_ids: torch.Tensor
    parent_ids: torch.Tensor
    input_pos: torch.Tensor
    assistant_mask: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    weights: torch.Tensor
    pixel_values: list[torch.Tensor | None]
    image_grid_thw: list[torch.Tensor | None]
    moe_routing_replay: PackedMoeRoutingReplay | None


class DiskPackedTensors(TypedDict):
    dir: str
    num_sequences: int
    sequence_length: int
    pixel_values: NotRequired[tuple[int, list[int]]]
    image_grid_thw: NotRequired[tuple[int, list[int]]]


def packed_tensors_from_tokenized_results(
    tokenized_results: list[TokenizedResult],
    seq_len: int,
    pad_token_id: int = -100,
    truncate_long_results: bool = True,
    advantage_balance: float = 0.0,
    verbosity: Verbosity = 1,
    pack_results: bool = True,
    include_moe_routing: bool = False,
) -> PackedTensors:
    sequences: list[list[tuple[TokenizedResult, int, int, int, int]]] = [[]]
    sequence_lengths = [0]
    sequence_prompt_ids: list[set[int]] = [set()]
    moe_routing_pack_stats = MoeRoutingPackStats()

    for result in tokenized_results:
        if len(result.token_ids) > seq_len and not truncate_long_results:
            if verbosity > 1:
                print("Result is too long, skipping")
            continue
        if include_moe_routing and result.moe_routed_experts is None:
            raise RuntimeError(
                "MoE routing replay from trajectories was requested, but a "
                "tokenized result has no aligned routed experts"
            )
        if sum(result.assistant_mask[result.prompt_length :]) == 0:
            if verbosity > 1:
                print("Result has no unique completion tokens, skipping")
            continue
        prompt_seen = result.prompt_id in sequence_prompt_ids[-1]
        src_start = result.prompt_length if prompt_seen else 0
        result_len = len(result.token_ids) - src_start
        if sequence_lengths[-1] and (
            not pack_results or sequence_lengths[-1] + result_len > seq_len
        ):
            sequences.append([])
            sequence_lengths.append(0)
            sequence_prompt_ids.append(set())
            prompt_seen = False
            src_start = 0
        group_id = random.randint(-(2**63), 2**63 - 1)
        dst_start = sequence_lengths[-1]
        src_end = len(result.token_ids)
        if truncate_long_results:
            src_end = min(src_end, src_start + seq_len - dst_start)
        if src_end <= src_start:
            continue
        sequences[-1].append((result, src_start, src_end, dst_start, group_id))
        sequence_lengths[-1] += src_end - src_start
        if not prompt_seen and result.prompt_length > 0:
            sequence_prompt_ids[-1].add(result.prompt_id)

    if not any(sequences):
        raise RuntimeError("No tokenized results were packable")
    permutation = list(range(len(sequences)))
    random.shuffle(permutation)
    num_sequences = len(permutation)
    tokens_np = np.full((num_sequences, seq_len), pad_token_id, dtype=np.int64)
    group_ids_np = np.full((num_sequences, seq_len), -1, dtype=np.int64)
    parent_ids_np = np.full((num_sequences, seq_len), -1, dtype=np.int64)
    input_pos_np = np.zeros((num_sequences, seq_len), dtype=np.int64)
    assistant_mask_np = np.zeros((num_sequences, seq_len), dtype=np.bool_)
    logprobs_np = np.full((num_sequences, seq_len), np.nan, dtype=np.float32)
    advantages_np = np.zeros((num_sequences, seq_len), dtype=np.float32)
    weights_np = np.zeros((num_sequences, seq_len), dtype=np.float32)
    pixel_values: list[list[torch.Tensor]] = [[] for _ in permutation]
    image_grid_thw: list[list[torch.Tensor]] = [[] for _ in permutation]
    route_tensor_np: np.ndarray | None = None
    route_mask_np: np.ndarray | None = None
    route_shape = _first_moe_route_shape(sequences) if include_moe_routing else None
    max_expert_id = 0
    if route_shape is not None:
        num_layers, topk = route_shape
        route_tensor_np = np.zeros(
            (num_sequences, seq_len, num_layers, topk), dtype=np.int32
        )
        route_mask_np = np.zeros((num_sequences, seq_len), dtype=np.bool_)

    for dst_seq, src_seq in enumerate(permutation):
        for result, src_start, src_end, dst_start, group_id in sequences[src_seq]:
            dst_end = dst_start + src_end - src_start
            tokens_np[dst_seq, dst_start:dst_end] = result.token_ids[src_start:src_end]
            parent_ids_np[dst_seq, dst_start:dst_end] = result.prompt_id
            prompt_end = min(result.prompt_length, src_end)
            if src_start < prompt_end:
                end = dst_start + prompt_end - src_start
                group_ids_np[dst_seq, dst_start:end] = result.prompt_id
            if prompt_end < src_end:
                start = dst_start + max(prompt_end - src_start, 0)
                group_ids_np[dst_seq, start:dst_end] = group_id
            input_pos_np[dst_seq, dst_start:dst_end] = result.input_pos[
                src_start:src_end
            ]
            assistant_mask_np[dst_seq, dst_start:dst_end] = result.assistant_mask[
                src_start:src_end
            ]
            logprobs_np[dst_seq, dst_start:dst_end] = result.logprobs[src_start:src_end]
            advantages_np[dst_seq, dst_start:dst_end] = result.advantage
            weights_np[dst_seq, dst_start:dst_end] = result.weight
            if src_start == 0:
                if result.pixel_values is not None:
                    pixel_values[dst_seq].append(result.pixel_values)
                if result.image_grid_thw is not None:
                    image_grid_thw[dst_seq].append(result.image_grid_thw)
            if include_moe_routing:
                assert route_tensor_np is not None and route_mask_np is not None
                assert route_shape is not None
                max_expert_id = max(
                    max_expert_id,
                    _copy_moe_routes(
                        route_tensor_np=route_tensor_np,
                        route_mask_np=route_mask_np,
                        dst_seq=dst_seq,
                        dst_start=dst_start,
                        src_start=src_start,
                        src_end=src_end,
                        raw_routes=result.moe_routed_experts,
                        route_shape=route_shape,
                    ),
                )

    assistant_mask_tensor = torch.from_numpy(assistant_mask_np)
    weights_tensor = torch.from_numpy(weights_np)
    weights_tensor = torch.where(
        assistant_mask_tensor, weights_tensor, torch.zeros_like(weights_tensor)
    )
    if bool(assistant_mask_tensor.any()):
        weights_tensor[assistant_mask_tensor] /= weights_tensor[
            assistant_mask_tensor
        ].mean()
    advantages_tensor = torch.from_numpy(advantages_np)
    advantages_tensor = torch.where(
        assistant_mask_tensor, advantages_tensor, torch.zeros_like(advantages_tensor)
    )
    if advantage_balance > 0.0:
        advantages_tensor = torch.where(
            advantages_tensor > 0,
            advantages_tensor,
            advantages_tensor * (1 - advantage_balance),
        )
    elif advantage_balance < 0.0:
        advantages_tensor = torch.where(
            advantages_tensor < 0,
            advantages_tensor,
            advantages_tensor * (1 + advantage_balance),
        )
    if bool(assistant_mask_tensor.any()):
        advantages_tensor[assistant_mask_tensor] /= (
            advantages_tensor[assistant_mask_tensor].abs()
            * weights_tensor[assistant_mask_tensor]
        ).mean()

    packed_tensors: PackedTensors = {
        "tokens": torch.from_numpy(tokens_np),
        "group_ids": torch.from_numpy(group_ids_np),
        "parent_ids": torch.from_numpy(parent_ids_np),
        "input_pos": torch.from_numpy(input_pos_np),
        "assistant_mask": assistant_mask_tensor,
        "logprobs": torch.from_numpy(logprobs_np),
        "advantages": advantages_tensor,
        "weights": weights_tensor,
        "pixel_values": [
            torch.concat(tensors) if tensors else None for tensors in pixel_values
        ],
        "image_grid_thw": [
            torch.concat(tensors) if tensors else None for tensors in image_grid_thw
        ],
        "moe_routing_replay": None,
    }
    if include_moe_routing:
        assert route_tensor_np is not None and route_mask_np is not None
        assert route_shape is not None
        num_layers, topk = route_shape
        if not bool(route_mask_np.any()):
            raise RuntimeError("No MoE routes were packed")
        moe_routing_pack_stats.packed_tokens = int(route_mask_np.sum())
        packed_tensors["moe_routing_replay"] = PackedMoeRoutingReplay(
            expert_indices=torch.from_numpy(route_tensor_np),
            token_mask=torch.from_numpy(route_mask_np),
            num_layers=num_layers,
            topk=topk,
            num_experts=max_expert_id + 1,
            pack_stats=moe_routing_pack_stats,
        )
    return packed_tensors


def _first_moe_route_shape(
    sequences: list[list[tuple[TokenizedResult, int, int, int, int]]],
) -> tuple[int, int]:
    for sequence in sequences:
        for result, *_ in sequence:
            shape = _moe_route_shape(result.moe_routed_experts)
            if shape is not None:
                return shape
    raise RuntimeError("No MoE routes were packed")


def _moe_route_shape(raw: Any) -> tuple[int, int] | None:
    if isinstance(raw, MoeRouteSegments):
        return int(raw.shape[1]), int(raw.shape[2])
    routes = _coerce_moe_routes(raw)
    if routes.shape[0] == 0:
        return None
    return int(routes.shape[1]), int(routes.shape[2])


def _coerce_moe_routes(raw: Any) -> MoeRouteArray:
    if isinstance(raw, np.ndarray):
        routes = raw.astype(np.int32, copy=False)
    elif isinstance(raw, list):
        first = next((route for route in raw if route is not None), None)
        if first is None:
            raise RuntimeError("No MoE routes were packed")
        routes = np.full(
            (len(raw), len(first), len(first[0])), MISSING_EXPERT_ID, dtype=np.int32
        )
        for index, route in enumerate(raw):
            if route is not None:
                routes[index] = route
    else:
        raise RuntimeError(f"Expected MoE routes array, got {type(raw)}")
    if routes.ndim != 3 or routes.shape[1] <= 0 or routes.shape[2] <= 0:
        raise RuntimeError(f"Packed MoE routes must be rank 3, got {routes.shape}")
    return routes


def _copy_moe_routes(
    *,
    route_tensor_np: np.ndarray,
    route_mask_np: np.ndarray,
    dst_seq: int,
    dst_start: int,
    src_start: int,
    src_end: int,
    raw_routes: Any,
    route_shape: tuple[int, int],
) -> int:
    if isinstance(raw_routes, MoeRouteSegments):
        max_expert_id = 0
        copied = np.zeros(src_end - src_start, dtype=np.bool_)
        for segment_start, segment in raw_routes.iter_slices(src_start, src_end):
            if tuple(segment.shape[1:]) != route_shape:
                raise RuntimeError("Packed MoE routes must have one rectangular shape")
            rel_start = segment_start - src_start
            rel_end = rel_start + segment.shape[0]
            dst_slice_start = dst_start + rel_start
            dst_slice_end = dst_start + rel_end
            route_tensor_np[dst_seq, dst_slice_start:dst_slice_end] = segment
            route_mask_np[dst_seq, dst_slice_start:dst_slice_end] = True
            copied[rel_start:rel_end] = True
            if segment.size:
                max_expert_id = max(max_expert_id, int(segment.max()))
        if not bool(copied.all()):
            missing = np.flatnonzero(~copied)[:8].tolist()
            raise RuntimeError(f"Segmented MoE routes did not cover rows {missing}")
        return max_expert_id

    routes = _coerce_moe_routes(raw_routes)
    route_values = routes[src_start:src_end]
    if tuple(route_values.shape[1:]) != route_shape:
        raise RuntimeError("Packed MoE routes must have one rectangular shape")
    valid_routes = _moe_route_mask(route_values)
    if not bool(valid_routes.all()):
        route_values = route_values.copy()
        route_values[~valid_routes] = 0
    route_tensor_np[dst_seq, dst_start : dst_start + src_end - src_start] = route_values
    route_mask_np[dst_seq, dst_start : dst_start + src_end - src_start] = valid_routes
    return int(route_values[valid_routes].max()) if bool(valid_routes.any()) else 0


def _moe_route_mask(routes: MoeRouteArray) -> np.ndarray:
    return np.all(routes != MISSING_EXPERT_ID, axis=(1, 2))


def packed_tensors_from_dir(**kwargs: Unpack[DiskPackedTensors]) -> PackedTensors:
    os.makedirs(kwargs["dir"], exist_ok=True)
    packed_tensors = {
        key: torch.from_file(
            f"{kwargs['dir']}/{key}.pt",
            shared=True,
            size=kwargs["num_sequences"] * kwargs["sequence_length"],
            dtype=dtype,
        ).view(kwargs["num_sequences"], kwargs["sequence_length"])
        for key, dtype in {
            "tokens": torch.long,
            "group_ids": torch.long,
            "parent_ids": torch.long,
            "input_pos": torch.long,
            "assistant_mask": torch.bool,
            "logprobs": torch.float32,
            "advantages": torch.float32,
            "weights": torch.float32,
        }.items()
    }
    _add_tensor_list(packed_tensors, kwargs, "pixel_values", torch.float32)  # ty:ignore[invalid-argument-type]
    _add_tensor_list(packed_tensors, kwargs, "image_grid_thw", torch.long)  # ty:ignore[invalid-argument-type]
    return cast(PackedTensors, packed_tensors)


def _add_tensor_list(
    packed_tensors: dict[str, Any],
    disk_packed_tensors: DiskPackedTensors,
    key: str,
    dtype: torch.dtype,
) -> None:
    if info := disk_packed_tensors.get(key):
        packed_tensors[key] = []
        inner_dim, offsets = cast(tuple[int, list[int]], info)
        packed_pixel_values = torch.from_file(
            f"{disk_packed_tensors['dir']}/{key}.pt",
            shared=True,
            size=offsets[-1] * inner_dim,
            dtype=dtype,
        ).view(-1, inner_dim)
        for start, end in zip(offsets[:-1], offsets[1:]):
            packed_tensors[key].append(
                packed_pixel_values[start:end] if start < end else None
            )
    else:
        packed_tensors[key] = [None] * disk_packed_tensors["num_sequences"]


def packed_tensors_to_dir(tensors: PackedTensors, dir: str) -> DiskPackedTensors:
    os.makedirs(dir, exist_ok=True)
    disk_packed_tensors: DiskPackedTensors = {
        "dir": dir,
        "num_sequences": tensors["tokens"].shape[0],
        "sequence_length": tensors["tokens"].shape[1],
    }
    if info := _get_tensor_list_info(tensors["pixel_values"]):
        disk_packed_tensors["pixel_values"] = info
    if info := _get_tensor_list_info(tensors["image_grid_thw"]):
        disk_packed_tensors["image_grid_thw"] = info
    for key, tensor in packed_tensors_from_dir(**disk_packed_tensors).items():
        if isinstance(tensor, list):
            for i, t in enumerate(tensor):
                if t is not None:
                    t.copy_(tensors[key][i])  # ty:ignore[invalid-key, unresolved-attribute]
        else:
            tensor.copy_(tensors[key])  # type: ignore
    return disk_packed_tensors


def _get_tensor_list_info(
    tensors: list[torch.Tensor | None],
) -> tuple[int, list[int]] | None:
    inner_dims = {tensor.shape[1] for tensor in tensors if tensor is not None}
    if len(inner_dims) == 0:
        return None
    assert len(inner_dims) == 1, f"Inner dimensions of {tensors} are not the same"
    offsets = [0]
    for tensor in tensors:
        if tensor is not None:
            offsets.append(offsets[-1] + tensor.shape[0])
        else:
            offsets.append(offsets[-1])
    return inner_dims.pop(), offsets


def plot_packed_tensors(
    packed_tensors: PackedTensors, output_dir: str | None = None
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "Plotting dependencies are not installed. Please install them with: "
            "pip install openpipe-art[plotting]"
        )

    plt.figure(figsize=(15, 24))

    for tensor, label, title, subplot_idx in (
        (packed_tensors["tokens"], "Token IDs", "Token IDs", 1),
        (packed_tensors["logprobs"], "Log Probabilities", "Token Log Probs", 2),
        (packed_tensors["group_ids"], "Group IDs", "Token Groups", 3),
        (packed_tensors["parent_ids"], "Parent IDs", "Parent IDs", 4),
        (packed_tensors["input_pos"], "Position", "Input Position", 5),
        (packed_tensors["assistant_mask"], "Assistant Mask", "Assistant Mask", 6),
        (packed_tensors["advantages"], "Advantages", "Token Advantages", 7),
        (packed_tensors["weights"], "Weights", "Token Weights", 8),
    ):
        plt.subplot(4, 2, subplot_idx)
        sns.heatmap(
            tensor.numpy(),
            cmap="viridis",
            cbar_kws={"label": label},
            xticklabels=False,
        )
        plt.title(title)
        plt.xlabel("Sequence Position")
        plt.ylabel("Batch")

    plt.tight_layout()
    plt.show()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = f"{output_dir}/packed_tensors_plot_{int(time.time())}.png"
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
    else:
        print("No output directory specified, plot not saved")
