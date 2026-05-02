from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch
from torch import Tensor
from torch.distributed import (
    all_to_all_single,
    get_world_size,
)
from torch.distributed import (
    is_available as dist_is_available,
)
from torch.distributed import (
    is_initialized as dist_is_initialized,
)

from art.megatron.context_parallel.layout_index import TokenLayoutIndex


class GdnCpPeerTransfer(BaseModel):
    """Token rows sent from one source rank to one destination rank."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source_rank: int = Field(ge=0)
    dest_rank: int = Field(ge=0)
    token_count: int = Field(ge=0)
    source_positions_tensor: Tensor | None = None
    dest_positions_tensor: Tensor | None = None

    @model_validator(mode="after")
    def _same_lengths(self) -> "GdnCpPeerTransfer":
        lengths = {int(self.token_count)}
        if self.source_positions_tensor is not None:
            lengths.add(int(self.source_positions_tensor.numel()))
        if self.dest_positions_tensor is not None:
            lengths.add(int(self.dest_positions_tensor.numel()))
        if len(lengths) != 1:
            raise ValueError("token, source, and destination position counts differ")
        return self


class GdnCpExchangePlan(BaseModel):
    """Permutation/all-to-all metadata between two distributed token layouts."""

    model_config = ConfigDict(frozen=True)

    cp_size: int = Field(ge=1)
    source_token_counts_by_rank: tuple[int, ...]
    dest_token_counts_by_rank: tuple[int, ...]
    transfers: tuple[GdnCpPeerTransfer, ...]
    cross_rank_token_count_override: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _rank_counts(self) -> "GdnCpExchangePlan":
        if len(self.source_token_counts_by_rank) != self.cp_size:
            raise ValueError("source token count length must equal cp_size")
        if len(self.dest_token_counts_by_rank) != self.cp_size:
            raise ValueError("destination token count length must equal cp_size")
        return self

    @property
    def cross_rank_token_count(self) -> int:
        if self.cross_rank_token_count_override is not None:
            return int(self.cross_rank_token_count_override)
        return sum(
            _transfer_token_count(transfer)
            for transfer in self.transfers
            if transfer.source_rank != transfer.dest_rank
        )


def _layout_cp_size(layout: TokenLayoutIndex) -> int:
    return len(layout.token_counts_by_rank)


def _token_layout_from_rank_ranges(
    ranges_by_rank: Sequence[Sequence[tuple[int, int, int]]],
) -> TokenLayoutIndex:
    ranges = _normalize_rank_ranges(
        "ranges_by_rank",
        ranges_by_rank,
        cp_size=len(ranges_by_rank),
    )
    return TokenLayoutIndex(
        ownership_ranges_by_rank=ranges,
        token_counts_by_rank=tuple(
            _rank_range_count(rank_ranges) for rank_ranges in ranges
        ),
    )


def _normalize_rank_ranges(
    name: str,
    values: Sequence[Sequence[tuple[int, int, int]]],
    *,
    cp_size: int,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    if len(values) != cp_size:
        raise ValueError(f"{name} must have {cp_size} ranks, got {len(values)}")
    normalized = []
    for rank, rank_ranges in enumerate(values):
        cursor = 0
        normalized_rank = []
        for start, end, position in rank_ranges:
            start = int(start)
            end = int(end)
            position = int(position)
            if start < 0 or end < start:
                raise ValueError(f"{name}[{rank}] has invalid range {(start, end)}")
            if position != cursor:
                raise ValueError(
                    f"{name}[{rank}] positions must be contiguous; "
                    f"expected {cursor}, got {position}"
                )
            normalized_rank.append((start, end, position))
            cursor += end - start
        normalized.append(tuple(normalized_rank))
    return tuple(normalized)


def _rank_range_count(ranges: Sequence[tuple[int, int, int]]) -> int:
    return sum(int(end) - int(start) for start, end, _ in ranges)


def _intersection_position_tensors(
    source_ranges: Sequence[tuple[int, int, int]],
    dest_ranges: Sequence[tuple[int, int, int]],
) -> tuple[Tensor, Tensor]:
    source_sorted = sorted(source_ranges, key=lambda item: (item[0], item[1]))
    dest_sorted = sorted(dest_ranges, key=lambda item: (item[0], item[1]))
    source_starts: list[int] = []
    dest_starts: list[int] = []
    lengths: list[int] = []
    source_index = 0
    dest_index = 0
    while source_index < len(source_sorted) and dest_index < len(dest_sorted):
        source_start, source_end, source_pos = source_sorted[source_index]
        dest_start, dest_end, dest_pos = dest_sorted[dest_index]
        overlap_start = max(source_start, dest_start)
        overlap_end = min(source_end, dest_end)
        if overlap_start < overlap_end:
            source_starts.append(source_pos + overlap_start - source_start)
            dest_starts.append(dest_pos + overlap_start - dest_start)
            lengths.append(overlap_end - overlap_start)
        if source_end <= dest_end:
            source_index += 1
        else:
            dest_index += 1
    if not lengths:
        empty = torch.empty((0,), dtype=torch.long)
        return empty, empty
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    total = int(lengths_tensor.sum().item())
    range_offsets = torch.cumsum(lengths_tensor, dim=0) - lengths_tensor
    item_offsets = torch.arange(total, dtype=torch.long) - torch.repeat_interleave(
        range_offsets,
        lengths_tensor,
    )
    return (
        torch.repeat_interleave(
            torch.tensor(source_starts, dtype=torch.long),
            lengths_tensor,
        )
        + item_offsets,
        torch.repeat_interleave(
            torch.tensor(dest_starts, dtype=torch.long),
            lengths_tensor,
        )
        + item_offsets,
    )


def _merged_token_ranges(
    ranges_by_rank: Sequence[Sequence[tuple[int, int, int]]],
) -> tuple[tuple[int, int], ...]:
    ranges = sorted(
        (int(start), int(end))
        for rank_ranges in ranges_by_rank
        for start, end, _ in rank_ranges
        if int(start) < int(end)
    )
    if not ranges:
        return ()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _range_list_count(ranges: Sequence[tuple[int, int]]) -> int:
    return sum(int(end) - int(start) for start, end in ranges)


def build_cp_exchange_plan_from_layout_index(
    *,
    source_layout: TokenLayoutIndex,
    dest_layout: TokenLayoutIndex,
    device: torch.device | str | None,
    validate: bool = True,
    local_rank: int | None = None,
) -> GdnCpExchangePlan:
    cp_size = _layout_cp_size(source_layout)
    if _layout_cp_size(dest_layout) != cp_size:
        raise ValueError(
            "source and destination cp_size differ: "
            f"{cp_size} and {_layout_cp_size(dest_layout)}"
        )
    if local_rank is not None and (local_rank < 0 or local_rank >= cp_size):
        raise ValueError(f"local_rank must be in [0, {cp_size}), got {local_rank}")
    if validate:
        _validate_layout_token_sets_match(source_layout, dest_layout)
    source_counts = source_layout.token_counts_by_rank
    dest_counts = dest_layout.token_counts_by_rank
    transfers: list[GdnCpPeerTransfer] = []
    cross_rank_token_count = 0
    for source_rank, source_ranges in enumerate(source_layout.ownership_ranges_by_rank):
        for dest_rank, dest_ranges in enumerate(dest_layout.ownership_ranges_by_rank):
            source_positions, dest_positions = _intersection_position_tensors(
                source_ranges,
                dest_ranges,
            )
            token_count = int(source_positions.numel())
            if token_count == 0:
                continue
            if source_rank != dest_rank:
                cross_rank_token_count += token_count
            if (
                local_rank is not None
                and source_rank != local_rank
                and dest_rank != local_rank
            ):
                continue
            transfers.append(
                _make_peer_transfer(
                    source_rank=source_rank,
                    dest_rank=dest_rank,
                    source_positions=source_positions,
                    dest_positions=dest_positions,
                    source_count=source_counts[source_rank],
                    dest_count=dest_counts[dest_rank],
                    device=device,
                )
            )
    return GdnCpExchangePlan.model_construct(
        cp_size=cp_size,
        source_token_counts_by_rank=source_counts,
        dest_token_counts_by_rank=dest_counts,
        transfers=tuple(
            sorted(transfers, key=lambda item: (item.source_rank, item.dest_rank))
        ),
        cross_rank_token_count_override=cross_rank_token_count,
    )


def build_local_rank_cp_exchange_plan_from_dest_ranges(
    *,
    source_layout: TokenLayoutIndex,
    dest_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...],
    device: torch.device | str | None,
    local_rank: int,
    cross_rank_token_count: int,
) -> GdnCpExchangePlan:
    cp_size = _layout_cp_size(source_layout)
    if len(dest_ranges_by_rank) != cp_size:
        raise ValueError("destination range rank count must equal cp_size")
    if local_rank < 0 or local_rank >= cp_size:
        raise ValueError(f"local_rank must be in [0, {cp_size}), got {local_rank}")
    dest_ranges_by_rank = _normalize_rank_ranges(
        "dest_ranges_by_rank",
        dest_ranges_by_rank,
        cp_size=cp_size,
    )
    dest_counts = tuple(
        sum(int(end) - int(start) for start, end, _ in ranges)
        for ranges in dest_ranges_by_rank
    )
    transfers = []
    for dest_rank, ranges in enumerate(dest_ranges_by_rank):
        source_ranks = range(cp_size) if dest_rank == local_rank else (local_rank,)
        for source_rank in source_ranks:
            source_positions, dest_positions = _intersection_position_tensors(
                source_layout.ownership_ranges_by_rank[source_rank],
                ranges,
            )
            if not int(source_positions.numel()):
                continue
            transfers.append(
                _make_peer_transfer(
                    source_rank=source_rank,
                    dest_rank=dest_rank,
                    source_positions=source_positions,
                    dest_positions=dest_positions,
                    source_count=source_layout.token_counts_by_rank[source_rank],
                    dest_count=dest_counts[dest_rank],
                    device=device,
                )
            )
    return GdnCpExchangePlan.model_construct(
        cp_size=cp_size,
        source_token_counts_by_rank=source_layout.token_counts_by_rank,
        dest_token_counts_by_rank=dest_counts,
        transfers=tuple(
            sorted(transfers, key=lambda item: (item.source_rank, item.dest_rank))
        ),
        cross_rank_token_count_override=int(cross_rank_token_count),
    )


def _validate_layout_token_sets_match(
    source_layout: TokenLayoutIndex,
    dest_layout: TokenLayoutIndex,
) -> None:
    source_ranges = _merged_token_ranges(source_layout.ownership_ranges_by_rank)
    dest_ranges = _merged_token_ranges(dest_layout.ownership_ranges_by_rank)
    if (
        source_ranges != dest_ranges
        or sum(source_layout.token_counts_by_rank) != _range_list_count(source_ranges)
        or sum(dest_layout.token_counts_by_rank) != _range_list_count(dest_ranges)
    ):
        raise ValueError(
            "source and destination token layouts must cover the same tokens"
        )


def _make_peer_transfer(
    *,
    source_rank: int,
    dest_rank: int,
    source_positions: Tensor,
    dest_positions: Tensor,
    source_count: int,
    dest_count: int,
    device: torch.device | str | None,
) -> GdnCpPeerTransfer:
    token_count = int(source_positions.numel())
    if token_count != int(dest_positions.numel()):
        raise ValueError("source and destination position counts differ")
    if _is_full_identity_transfer(
        source_rank=source_rank,
        dest_rank=dest_rank,
        source_positions=source_positions,
        dest_positions=dest_positions,
        source_count=source_count,
        dest_count=dest_count,
    ):
        source_tensor = None
        dest_tensor = None
    else:
        target = torch.device(device) if device is not None else torch.device("cpu")
        source_tensor = source_positions.to(
            device=target, dtype=torch.long
        ).contiguous()
        dest_tensor = dest_positions.to(device=target, dtype=torch.long).contiguous()
    return GdnCpPeerTransfer.model_construct(
        source_rank=source_rank,
        dest_rank=dest_rank,
        token_count=token_count,
        source_positions_tensor=source_tensor,
        dest_positions_tensor=dest_tensor,
    )


def _is_full_identity_transfer(
    *,
    source_rank: int,
    dest_rank: int,
    source_positions: Tensor,
    dest_positions: Tensor,
    source_count: int,
    dest_count: int,
) -> bool:
    if source_rank != dest_rank or source_count != dest_count:
        return False
    if int(source_positions.numel()) != int(source_count):
        return False
    if int(dest_positions.numel()) != int(dest_count):
        return False
    expected = torch.arange(int(source_count), dtype=torch.long)
    return bool(torch.equal(source_positions.cpu(), expected)) and bool(
        torch.equal(dest_positions.cpu(), expected)
    )


def _reverse_exchange_plan(plan: GdnCpExchangePlan) -> GdnCpExchangePlan:
    return GdnCpExchangePlan.model_construct(
        cp_size=plan.cp_size,
        source_token_counts_by_rank=_dest_counts_by_rank(plan),
        dest_token_counts_by_rank=_source_counts_by_rank(plan),
        cross_rank_token_count_override=plan.cross_rank_token_count_override,
        transfers=tuple(
            GdnCpPeerTransfer.model_construct(
                source_rank=transfer.dest_rank,
                dest_rank=transfer.source_rank,
                token_count=_transfer_token_count(transfer),
                source_positions_tensor=transfer.dest_positions_tensor,
                dest_positions_tensor=transfer.source_positions_tensor,
            )
            for transfer in sorted(
                plan.transfers, key=lambda item: (item.dest_rank, item.source_rank)
            )
        ),
    )


def move_cp_exchange_plan_to_device(
    plan: GdnCpExchangePlan | None,
    device: torch.device | str,
) -> GdnCpExchangePlan | None:
    if plan is None:
        return None
    target = torch.device(device)
    return GdnCpExchangePlan.model_construct(
        cp_size=plan.cp_size,
        source_token_counts_by_rank=_source_counts_by_rank(plan),
        dest_token_counts_by_rank=_dest_counts_by_rank(plan),
        transfers=tuple(
            GdnCpPeerTransfer.model_construct(
                source_rank=transfer.source_rank,
                dest_rank=transfer.dest_rank,
                token_count=transfer.token_count,
                source_positions_tensor=_move_optional_index_tensor(
                    transfer.source_positions_tensor, target
                ),
                dest_positions_tensor=_move_optional_index_tensor(
                    transfer.dest_positions_tensor, target
                ),
            )
            for transfer in plan.transfers
        ),
        cross_rank_token_count_override=plan.cross_rank_token_count_override,
    )


def _move_optional_index_tensor(
    tensor: Tensor | None, device: torch.device
) -> Tensor | None:
    if tensor is None or tensor.device == device:
        return tensor
    return tensor.to(device=device)


def send_split_sizes_for_rank(plan: GdnCpExchangePlan, rank: int) -> tuple[int, ...]:
    _check_rank(plan, rank)
    return tuple(
        _transfer_token_count(_transfer(plan, source_rank=rank, dest_rank=dest_rank))
        for dest_rank in range(plan.cp_size)
    )


def recv_split_sizes_for_rank(plan: GdnCpExchangePlan, rank: int) -> tuple[int, ...]:
    _check_rank(plan, rank)
    return tuple(
        _transfer_token_count(_transfer(plan, source_rank=source_rank, dest_rank=rank))
        for source_rank in range(plan.cp_size)
    )


def pack_rank_send_tensor(
    local_tensor: Tensor,
    plan: GdnCpExchangePlan,
    *,
    source_rank: int,
) -> Tensor:
    """Pack one rank's local tensor in peer order for `all_to_all_single`."""

    _check_rank(plan, source_rank)
    expected_rows = _source_count_for_rank(plan, source_rank)
    if int(local_tensor.shape[0]) != expected_rows:
        raise ValueError(
            f"rank {source_rank} tensor has {int(local_tensor.shape[0])} rows, "
            f"expected {expected_rows}"
        )
    pieces = []
    for dest_rank in range(plan.cp_size):
        transfer = _transfer(plan, source_rank=source_rank, dest_rank=dest_rank)
        if _transfer_token_count(transfer):
            if _is_implicit_full_identity_transfer(
                transfer,
                source_count=_source_count_for_rank(plan, source_rank),
                dest_count=_dest_count_for_rank(plan, dest_rank),
            ):
                pieces.append(local_tensor)
            else:
                index = _transfer_index_tensor(
                    transfer.source_positions_tensor,
                    device=local_tensor.device,
                )
                pieces.append(local_tensor.index_select(0, index))
    if not pieces:
        return local_tensor.new_empty((0, *local_tensor.shape[1:]))
    return torch.cat(pieces, dim=0)


def unpack_rank_recv_tensor(
    recv_buffer: Tensor,
    plan: GdnCpExchangePlan,
    *,
    dest_rank: int,
) -> Tensor:
    """Unpack one rank's `all_to_all_single` receive buffer into destination order."""

    _check_rank(plan, dest_rank)
    expected_rows = sum(recv_split_sizes_for_rank(plan, dest_rank))
    if int(recv_buffer.shape[0]) != expected_rows:
        raise ValueError(
            f"rank {dest_rank} recv buffer has {int(recv_buffer.shape[0])} rows, "
            f"expected {expected_rows}"
        )
    dest_rows = _dest_count_for_rank(plan, dest_rank)
    output = recv_buffer.new_empty((dest_rows, *recv_buffer.shape[1:]))
    offset = 0
    for source_rank in range(plan.cp_size):
        transfer = _transfer(plan, source_rank=source_rank, dest_rank=dest_rank)
        rows = _transfer_token_count(transfer)
        peer_rows = recv_buffer[offset : offset + rows]
        offset += rows
        if rows == 0:
            continue
        if _is_implicit_full_identity_transfer(
            transfer,
            source_count=_source_count_for_rank(plan, source_rank),
            dest_count=dest_rows,
        ):
            output.copy_(peer_rows)
            continue
        dest_index = _transfer_index_tensor(
            transfer.dest_positions_tensor,
            device=recv_buffer.device,
        )
        output.index_copy_(0, dest_index, peer_rows)
    if dest_rows == 0:
        return recv_buffer.new_empty((0, *recv_buffer.shape[1:]))
    return output


@torch.compiler.disable
def exchange_rank_tensor_all_to_all(
    local_tensor: Tensor,
    plan: GdnCpExchangePlan,
    *,
    rank: int,
    group: Any | None = None,
    backward_plan: GdnCpExchangePlan | None = None,
) -> Tensor:
    """Redistribute one rank tensor with real `dist.all_to_all_single`.

    This is the eager distributed/autograd boundary for attention-layout to
    GDN-layout token exchange. Backward applies the inverse exchange plan.
    """

    _check_rank(plan, rank)
    if plan.cross_rank_token_count == 0:
        return _exchange_rank_tensor_local(local_tensor, plan, rank=rank)
    if not dist_is_available() or not dist_is_initialized():
        raise RuntimeError("torch.distributed must be initialized for GDN CP exchange")
    world_size = get_world_size(group)
    if world_size != plan.cp_size:
        raise ValueError(
            f"process group world size {world_size} must match plan cp_size "
            f"{plan.cp_size}"
        )
    if backward_plan is None:
        raise ValueError("cross-rank GDN CP exchange requires a prebuilt backward_plan")
    return _GdnCpExchangeFunction.apply(local_tensor, plan, backward_plan, rank, group)


def _transfer_token_count(transfer: GdnCpPeerTransfer) -> int:
    return int(transfer.token_count)


def _is_implicit_full_identity_transfer(
    transfer: GdnCpPeerTransfer,
    *,
    source_count: int,
    dest_count: int,
) -> bool:
    return (
        transfer.source_rank == transfer.dest_rank
        and _transfer_token_count(transfer) == int(source_count) == int(dest_count)
        and transfer.source_positions_tensor is None
        and transfer.dest_positions_tensor is None
    )


def _transfer_positions_tuple(tensor: Tensor | None) -> tuple[int, ...]:
    if tensor is None:
        return ()
    return tuple(int(value) for value in tensor.detach().cpu().tolist())


def _transfer_index_tensor(
    tensor: Tensor | None,
    *,
    device: torch.device,
) -> Tensor:
    if tensor is None:
        raise ValueError("non-identity GDN CP transfer requires an index tensor")
    if tensor.device == device:
        return tensor
    return tensor.to(device=device, non_blocking=True)


def _source_counts_by_rank(plan: GdnCpExchangePlan) -> tuple[int, ...]:
    return plan.source_token_counts_by_rank


def _dest_counts_by_rank(plan: GdnCpExchangePlan) -> tuple[int, ...]:
    return plan.dest_token_counts_by_rank


def _source_count_for_rank(plan: GdnCpExchangePlan, rank: int) -> int:
    return _source_counts_by_rank(plan)[rank]


def _dest_count_for_rank(plan: GdnCpExchangePlan, rank: int) -> int:
    return _dest_counts_by_rank(plan)[rank]


def _check_rank(plan: GdnCpExchangePlan, rank: int) -> None:
    if rank < 0 or rank >= plan.cp_size:
        raise ValueError(f"rank must be in [0, {plan.cp_size}), got {rank}")


def _transfer(
    plan: GdnCpExchangePlan,
    *,
    source_rank: int,
    dest_rank: int,
) -> GdnCpPeerTransfer:
    for transfer in plan.transfers:
        if transfer.source_rank == source_rank and transfer.dest_rank == dest_rank:
            return transfer
    return GdnCpPeerTransfer(
        source_rank=source_rank,
        dest_rank=dest_rank,
        token_count=0,
    )


class _GdnCpExchangeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        local_tensor: Tensor,
        plan: GdnCpExchangePlan,
        backward_plan: GdnCpExchangePlan,
        rank: int,
        group: Any | None,
    ) -> Tensor:
        ctx.rank = rank
        ctx.group = group
        ctx.reverse_plan = backward_plan
        return _exchange_rank_tensor_all_to_all_forward(
            local_tensor,
            plan,
            rank=rank,
            group=group,
        )

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Any:
        (grad_output,) = grad_outputs
        grad_input = _exchange_rank_tensor_all_to_all_forward(
            grad_output.contiguous(),
            ctx.reverse_plan,
            rank=ctx.rank,
            group=ctx.group,
        )
        return grad_input, None, None, None, None


def _exchange_rank_tensor_all_to_all_forward(
    local_tensor: Tensor,
    plan: GdnCpExchangePlan,
    *,
    rank: int,
    group: Any | None,
) -> Tensor:
    if plan.cross_rank_token_count == 0:
        return _exchange_rank_tensor_local(local_tensor, plan, rank=rank)
    accumulate = _rank_recv_requires_accumulation(plan, rank)
    output = _init_rank_exchange_output(
        local_tensor, plan, rank=rank, accumulate=accumulate
    )
    send_buffer = _pack_rank_cross_send_tensor(local_tensor, plan, source_rank=rank)
    send_buffer = send_buffer.contiguous()
    recv_rows = sum(_cross_recv_split_sizes_for_rank(plan, rank))
    recv_buffer = local_tensor.new_empty((recv_rows, *local_tensor.shape[1:]))
    all_to_all_single(
        recv_buffer,
        send_buffer,
        output_split_sizes=list(_cross_recv_split_sizes_for_rank(plan, rank)),
        input_split_sizes=list(_cross_send_split_sizes_for_rank(plan, rank)),
        group=group,
    )
    _unpack_rank_cross_recv_tensor_into(
        output, recv_buffer, plan, dest_rank=rank, accumulate=accumulate
    )
    return output


def _exchange_rank_tensor_local(
    local_tensor: Tensor,
    plan: GdnCpExchangePlan,
    *,
    rank: int,
) -> Tensor:
    transfer = _transfer(plan, source_rank=rank, dest_rank=rank)
    if _is_implicit_full_identity_transfer(
        transfer,
        source_count=_source_count_for_rank(plan, rank),
        dest_count=_dest_count_for_rank(plan, rank),
    ):
        return local_tensor
    return unpack_rank_recv_tensor(
        pack_rank_send_tensor(local_tensor, plan, source_rank=rank),
        plan,
        dest_rank=rank,
    )


def _init_rank_exchange_output(
    local_tensor: Tensor,
    plan: GdnCpExchangePlan,
    *,
    rank: int,
    accumulate: bool,
) -> Tensor:
    dest_rows = _dest_count_for_rank(plan, rank)
    output_shape = (dest_rows, *local_tensor.shape[1:])
    output = (
        local_tensor.new_zeros(output_shape)
        if accumulate
        else local_tensor.new_empty(output_shape)
    )
    transfer = _transfer(plan, source_rank=rank, dest_rank=rank)
    if not _transfer_token_count(transfer):
        return output
    if _is_implicit_full_identity_transfer(
        transfer,
        source_count=_source_count_for_rank(plan, rank),
        dest_count=dest_rows,
    ):
        if accumulate:
            output.add_(local_tensor)
        else:
            output.copy_(local_tensor)
        return output
    source_index = _transfer_index_tensor(
        transfer.source_positions_tensor,
        device=local_tensor.device,
    )
    dest_index = _transfer_index_tensor(
        transfer.dest_positions_tensor,
        device=local_tensor.device,
    )
    values = local_tensor.index_select(0, source_index)
    if accumulate:
        output.index_add_(0, dest_index, values)
    else:
        output.index_copy_(0, dest_index, values)
    return output


def _pack_rank_cross_send_tensor(
    local_tensor: Tensor,
    plan: GdnCpExchangePlan,
    *,
    source_rank: int,
) -> Tensor:
    pieces = []
    for dest_rank in range(plan.cp_size):
        if dest_rank == source_rank:
            continue
        transfer = _transfer(plan, source_rank=source_rank, dest_rank=dest_rank)
        if _transfer_token_count(transfer):
            index = _transfer_index_tensor(
                transfer.source_positions_tensor,
                device=local_tensor.device,
            )
            pieces.append(local_tensor.index_select(0, index))
    if not pieces:
        return local_tensor.new_empty((0, *local_tensor.shape[1:]))
    return torch.cat(pieces, dim=0)


def _unpack_rank_cross_recv_tensor_into(
    output: Tensor,
    recv_buffer: Tensor,
    plan: GdnCpExchangePlan,
    *,
    dest_rank: int,
    accumulate: bool,
) -> None:
    expected_rows = sum(_cross_recv_split_sizes_for_rank(plan, dest_rank))
    if int(recv_buffer.shape[0]) != expected_rows:
        raise ValueError(
            f"recv buffer for rank {dest_rank} has {int(recv_buffer.shape[0])} rows; "
            f"expected {expected_rows}"
        )
    offset = 0
    for source_rank in range(plan.cp_size):
        if source_rank == dest_rank:
            continue
        transfer = _transfer(plan, source_rank=source_rank, dest_rank=dest_rank)
        rows = _transfer_token_count(transfer)
        peer_rows = recv_buffer[offset : offset + rows]
        offset += rows
        if rows == 0:
            continue
        dest_index = _transfer_index_tensor(
            transfer.dest_positions_tensor,
            device=recv_buffer.device,
        )
        if accumulate:
            output.index_add_(0, dest_index, peer_rows)
        else:
            output.index_copy_(0, dest_index, peer_rows)


def _rank_recv_requires_accumulation(plan: GdnCpExchangePlan, rank: int) -> bool:
    positions: list[int] = []
    for source_rank in range(plan.cp_size):
        transfer = _transfer(plan, source_rank=source_rank, dest_rank=rank)
        if not _transfer_token_count(transfer):
            continue
        positions.extend(_transfer_dest_positions_for_duplicate_check(plan, transfer))
    return len(positions) != len(set(positions))


def _transfer_dest_positions_for_duplicate_check(
    plan: GdnCpExchangePlan, transfer: GdnCpPeerTransfer
) -> tuple[int, ...]:
    token_count = _transfer_token_count(transfer)
    if token_count == 0:
        return ()
    if _is_implicit_full_identity_transfer(
        transfer,
        source_count=_source_count_for_rank(plan, transfer.source_rank),
        dest_count=_dest_count_for_rank(plan, transfer.dest_rank),
    ):
        return tuple(range(token_count))
    positions = _transfer_positions_tuple(transfer.dest_positions_tensor)
    if len(positions) != token_count:
        raise ValueError("GDN CP transfer destination positions must match token_count")
    return positions


def _cross_send_split_sizes_for_rank(
    plan: GdnCpExchangePlan,
    rank: int,
) -> tuple[int, ...]:
    return tuple(
        0
        if dest_rank == rank
        else _transfer_token_count(
            _transfer(plan, source_rank=rank, dest_rank=dest_rank)
        )
        for dest_rank in range(plan.cp_size)
    )


def _cross_recv_split_sizes_for_rank(
    plan: GdnCpExchangePlan,
    rank: int,
) -> tuple[int, ...]:
    return tuple(
        0
        if source_rank == rank
        else _transfer_token_count(
            _transfer(plan, source_rank=source_rank, dest_rank=rank)
        )
        for source_rank in range(plan.cp_size)
    )
