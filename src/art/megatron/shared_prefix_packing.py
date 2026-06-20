from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SharedPrefixPack:
    tokens: torch.Tensor
    group_ids: torch.Tensor
    parent_ids: torch.Tensor
    position_ids: torch.Tensor
    positions_by_sequence: tuple[torch.Tensor, ...]


def pack_shared_prefixes(
    sequences: Iterable[torch.Tensor],
    *,
    max_depth: int,
) -> SharedPrefixPack:
    """Pack token sequences by storing shared prefixes once.

    This is the small packing step that lets `TrainerRank.forward()` run one
    model pass over a compact prefix tree instead of replaying the same prompt
    tokens for every request. Think of each input sequence as a path through a
    tree: when several paths start with the same tokens, this function writes
    that shared segment once, then writes each branch after it.

    Args:
        sequences: 1-D token tensors to pack.
        max_depth: How many nested shared-prefix levels to emit. `0` disables
            prefix sharing and writes each sequence as its own root segment. `1`
            shares the first common segment in each branch; larger values allow
            branches to contain shared sub-branches.

    Returns:
        `tokens` is the compact model input, shaped `[1, packed_length]`.
        `group_ids` and `parent_ids` describe the prefix tree to shared-prefix
        attention. Positions in the same emitted segment share a group, and each
        group points at the parent segment it continues from. Root groups point
        to themselves.
        `position_ids` keeps each token's original sequence position for
        positional embeddings/rotary attention.
        `positions_by_sequence` is the reverse index used after the model call
        to unpack logits, logprobs, or hidden states back into one tensor per
        original request.

    The implementation is a tiny radix-tree walk. It finds the longest prefix
    shared by the active sequences, emits that segment once, then partitions the
    remaining sequences by their next token while preserving first-seen order.
    Single sequences, empty branches, and branches past `max_depth` are emitted
    as ordinary unshared tails.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    tensors = tuple(_sequence_tensor(sequence) for sequence in sequences)
    if not tensors:
        return _empty_pack()

    device = tensors[0].device
    lengths = torch.tensor([len(tensor) for tensor in tensors], device=device)
    if int(lengths.max().item()) == 0:
        return _empty_pack(len(tensors), device=device)

    padded = torch.nn.utils.rnn.pad_sequence(list(tensors), batch_first=True)
    token_chunks: list[torch.Tensor] = []
    group_chunks: list[torch.Tensor] = []
    parent_chunks: list[torch.Tensor] = []
    position_chunks: list[torch.Tensor] = []
    positions_by_sequence: list[list[torch.Tensor]] = [[] for _ in tensors]
    cursor = 0
    next_group_id = 1

    def emit(
        indices: torch.Tensor,
        start: int,
        end: int,
        parent_group_id: int | None,
    ) -> int:
        nonlocal cursor, next_group_id
        segment = tensors[int(indices[0].item())][start:end]
        group_id = next_group_id
        next_group_id += 1
        parent_id = group_id if parent_group_id is None else parent_group_id
        packed_positions = torch.arange(cursor, cursor + len(segment), device=device)

        token_chunks.append(segment)
        group_chunks.append(torch.full_like(segment, group_id))
        parent_chunks.append(torch.full_like(segment, parent_id))
        position_chunks.append(torch.arange(start, end, device=device))
        for sequence_index in indices.tolist():
            positions_by_sequence[sequence_index].append(packed_positions)
        cursor += len(segment)
        return group_id

    def shared_end(indices: torch.Tensor, start: int) -> int:
        end = int(lengths.index_select(0, indices).min().item())
        if start >= end:
            return start
        shared = (
            padded.index_select(0, indices)[:, start:end]
            == padded[indices[0], start:end]
        ).all(dim=0)
        return (
            end
            if bool(shared.all().item())
            else start + int(shared.logical_not().nonzero()[0])
        )

    def branch_groups(indices: torch.Tensor, start: int) -> list[torch.Tensor]:
        groups: dict[int, list[int]] = {}
        order: list[int] = []
        symbols = padded.index_select(0, indices)[:, start].tolist()
        for symbol, index in zip(symbols, indices.tolist(), strict=True):
            if symbol not in groups:
                groups[symbol] = []
                order.append(symbol)
            groups[symbol].append(index)
        return [
            torch.tensor(groups[symbol], dtype=torch.long, device=device)
            for symbol in order
        ]

    def walk(
        indices: torch.Tensor,
        start: int,
        parent_group_id: int | None,
        depth: int,
    ) -> None:
        active = indices[lengths.index_select(0, indices) > start]
        if int(active.numel()) == 0:
            return
        if max_depth == 0 or int(active.numel()) == 1 or (
            parent_group_id is not None and depth >= max_depth
        ):
            for sequence_index in active:
                emit(
                    sequence_index[None],
                    start,
                    int(lengths[sequence_index].item()),
                    parent_group_id,
                )
            return

        end = shared_end(active, start)
        if end > start:
            group_id = emit(active, start, end, parent_group_id)
            walk(active, end, group_id, depth + 1)
            return

        for group in branch_groups(active, start):
            walk(group, start, parent_group_id, depth)

    walk(torch.arange(len(tensors), device=device), 0, None, 0)

    return SharedPrefixPack(
        tokens=torch.cat(token_chunks).unsqueeze(0),
        group_ids=torch.cat(group_chunks).unsqueeze(0),
        parent_ids=torch.cat(parent_chunks).unsqueeze(0),
        position_ids=torch.cat(position_chunks).unsqueeze(0),
        positions_by_sequence=tuple(
            torch.cat(chunks)
            if chunks
            else torch.empty(0, dtype=torch.long, device=device)
            for chunks in positions_by_sequence
        ),
    )


def visualize_shared_prefix_pack(pack: SharedPrefixPack) -> str:
    rows = ["pos token group parent source_pos"]
    for position, (token, group, parent, source_pos) in enumerate(
        zip(
            pack.tokens.reshape(-1).detach().cpu().tolist(),
            pack.group_ids.reshape(-1).detach().cpu().tolist(),
            pack.parent_ids.reshape(-1).detach().cpu().tolist(),
            pack.position_ids.reshape(-1).detach().cpu().tolist(),
            strict=True,
        )
    ):
        rows.append(
            f"{position:>3} {token:>5} {group:>5} {parent:>6} {source_pos:>10}"
        )
    for index, positions in enumerate(pack.positions_by_sequence):
        rows.append(f"seq {index}: {positions.detach().cpu().tolist()}")
    return "\n".join(rows)


def _empty_pack(
    sequence_count: int = 0,
    *,
    device: torch.device | None = None,
) -> SharedPrefixPack:
    flat = torch.empty(0, dtype=torch.long, device=device)
    row = flat.unsqueeze(0)
    return SharedPrefixPack(
        tokens=row,
        group_ids=row,
        parent_ids=row,
        position_ids=row,
        positions_by_sequence=tuple(flat for _ in range(sequence_count)),
    )


def _sequence_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 1:
        raise ValueError(
            f"pack_shared_prefixes expects 1-D tensors, got {tuple(tensor.shape)}"
        )
    return tensor.detach().to(dtype=torch.long).contiguous()
