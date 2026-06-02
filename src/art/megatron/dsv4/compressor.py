from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Protocol

import torch

from .types import (
    Dsv4BranchView,
    Dsv4CompressedEntry,
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    Dsv4HaloTransfer,
    Dsv4StreamKind,
    Dsv4StreamSpec,
    Dsv4TokenInView,
)

if TYPE_CHECKING:
    from art.megatron.context_parallel.types import ArtContextParallelState
else:
    ArtContextParallelState = Any

_PADDING_GROUP_ID = -1


class TokenLayoutIndexLike(Protocol):
    ownership_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    token_counts_by_rank: tuple[int, ...]


def build_dsv4_compressed_layout(
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    token_layout_index: TokenLayoutIndexLike,
    spec: Dsv4CompressionSpec,
) -> Dsv4CompressedLayout:
    """Build host-ahead DSV4 compression metadata from shared-prefix layout.

    This function is CPU metadata planning only. It does not inspect activations
    and must not read CUDA tensors in the production lookahead path. Compressed
    entries are deduplicated only when their full dependency is inside the
    shared prefix; closure-token ownership follows the ART CP token layout.
    """
    if group_ids.device.type != "cpu" or parent_ids.device.type != "cpu":
        raise RuntimeError("DSV4 compression planning requires CPU metadata tensors")
    if int(spec.ratio) <= 0:
        raise RuntimeError(f"DSV4 compression ratio must be positive, got {spec.ratio}")
    group_row, parent_row = _validate_metadata(group_ids, parent_ids)
    streams = _build_streams(group_row=group_row, parent_row=parent_row)
    branch_views = _build_branch_views(streams)
    token_owner = _build_token_ownership(token_layout_index)
    entries, entry_ids_by_branch_stream = _build_entries(
        branch_views=branch_views,
        token_owner=token_owner,
        spec=spec,
    )
    return Dsv4CompressedLayout(
        spec=spec,
        streams=streams,
        branch_views=branch_views,
        entries=entries,
        halo_transfers=_build_halo_transfers(entries, token_owner),
        entry_ids_by_owner_rank=_entry_ids_by_owner(entries, token_layout_index),
        entry_ids_by_branch_stream=entry_ids_by_branch_stream,
    )


def build_dsv4_compressed_layout_from_cp_state(
    *,
    state: ArtContextParallelState,
    spec: Dsv4CompressionSpec,
) -> Dsv4CompressedLayout:
    return build_dsv4_compressed_layout(
        group_ids=state.group_ids.unsqueeze(0)
        if state.group_ids.ndim == 1
        else state.group_ids,
        parent_ids=state.parent_ids.unsqueeze(0)
        if state.parent_ids.ndim == 1
        else state.parent_ids,
        token_layout_index=state.rank_plan.token_layout_index,
        spec=spec,
    )


def position_in_query_view(
    *,
    branch_view: Dsv4BranchView,
    candidate_token_id: int,
) -> int | None:
    for token in branch_view.tokens:
        if int(token.packed_token_id) == int(candidate_token_id):
            return int(token.view_pos)
    return None


def _validate_metadata(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(group_ids.shape) != tuple(parent_ids.shape):
        raise RuntimeError(
            "DSV4 group_ids and parent_ids must share shape, got "
            f"{tuple(group_ids.shape)} vs {tuple(parent_ids.shape)}"
        )
    if group_ids.ndim != 2:
        raise RuntimeError(
            f"DSV4 shared-prefix metadata must be rank-2, got {group_ids.ndim}"
        )
    if int(group_ids.shape[0]) != 1:
        raise RuntimeError(
            "DSV4 CP currently supports one packed row, got "
            f"batch={int(group_ids.shape[0])}"
        )
    valid_count = _valid_token_count(group_ids[0])
    return (
        group_ids[0, :valid_count].contiguous(),
        parent_ids[0, :valid_count].contiguous(),
    )


def _valid_token_count(group_row: torch.Tensor) -> int:
    valid = group_row != _PADDING_GROUP_ID
    count = int(valid.sum().item())
    if count == 0:
        return 0
    if not bool(valid[:count].all().item()):
        raise RuntimeError("DSV4 shared-prefix padding must be a contiguous tail")
    return count


def _build_streams(
    *,
    group_row: torch.Tensor,
    parent_row: torch.Tensor,
) -> tuple[Dsv4StreamSpec, ...]:
    streams: list[Dsv4StreamSpec] = []
    for start, end, group_id, parent_id in _scan_runs(group_row, parent_row):
        if group_id == _PADDING_GROUP_ID:
            continue
        kind = (
            Dsv4StreamKind.PREFIX
            if _is_prefix_run(start=start, group_id=group_id, parent_id=parent_id)
            else Dsv4StreamKind.COMPLETION
        )
        streams.append(
            Dsv4StreamSpec(
                stream_id=group_id,
                kind=kind,
                parent_stream_id=None if kind is Dsv4StreamKind.PREFIX else parent_id,
                start=start,
                end=end,
            )
        )
    _validate_stream_graph(streams)
    return tuple(streams)


def _scan_runs(
    group_row: torch.Tensor,
    parent_row: torch.Tensor,
) -> list[tuple[int, int, int, int]]:
    if int(group_row.numel()) == 0:
        return []
    runs: list[tuple[int, int, int, int]] = []
    start = 0
    prev_group = int(group_row[0].item())
    prev_parent = int(parent_row[0].item())
    for idx in range(1, int(group_row.numel())):
        group_id = int(group_row[idx].item())
        parent_id = int(parent_row[idx].item())
        if group_id == prev_group and parent_id != prev_parent:
            raise RuntimeError(
                "DSV4 found a contiguous group run with inconsistent parent ids: "
                f"group_id={group_id}, index={idx}"
            )
        if group_id == prev_group:
            continue
        runs.append((start, idx, prev_group, prev_parent))
        start = idx
        prev_group = group_id
        prev_parent = parent_id
    runs.append((start, int(group_row.numel()), prev_group, prev_parent))
    return runs


def _is_prefix_run(*, start: int, group_id: int, parent_id: int) -> bool:
    return int(group_id) == int(parent_id) or (
        int(start) == 0 and int(parent_id) == _PADDING_GROUP_ID
    )


def _validate_stream_graph(streams: list[Dsv4StreamSpec]) -> None:
    seen: set[int] = set()
    prefixes: set[int] = set()
    for stream in streams:
        if stream.stream_id in seen:
            raise RuntimeError(f"DSV4 stream {stream.stream_id} appears more than once")
        seen.add(stream.stream_id)
        if stream.kind is Dsv4StreamKind.PREFIX:
            prefixes.add(stream.stream_id)
    for stream in streams:
        if stream.kind is Dsv4StreamKind.PREFIX:
            continue
        if stream.parent_stream_id not in prefixes:
            raise RuntimeError(
                "DSV4 completion stream points to missing prefix: "
                f"stream_id={stream.stream_id}, parent={stream.parent_stream_id}"
            )


def _build_branch_views(
    streams: tuple[Dsv4StreamSpec, ...],
) -> tuple[Dsv4BranchView, ...]:
    by_id = {stream.stream_id: stream for stream in streams}
    branch_views: list[Dsv4BranchView] = []
    for stream in streams:
        if stream.kind is Dsv4StreamKind.PREFIX:
            branch_views.append(_make_branch_view(prefix=stream, suffix=None))
    for stream in streams:
        if stream.kind is Dsv4StreamKind.PREFIX:
            continue
        parent = by_id.get(stream.parent_stream_id)
        if parent is None:
            raise RuntimeError(
                f"DSV4 completion {stream.stream_id} missing parent {stream.parent_stream_id}"
            )
        branch_views.append(_make_branch_view(prefix=parent, suffix=stream))
    return tuple(branch_views)


def _make_branch_view(
    *,
    prefix: Dsv4StreamSpec,
    suffix: Dsv4StreamSpec | None,
) -> Dsv4BranchView:
    tokens: list[Dsv4TokenInView] = []
    cursor = 0
    for offset, packed_id in enumerate(range(prefix.start, prefix.end)):
        tokens.append(
            Dsv4TokenInView(
                packed_token_id=packed_id,
                stream_id=prefix.stream_id,
                view_pos=cursor,
                stream_pos=offset,
            )
        )
        cursor += 1
    if suffix is not None:
        for offset, packed_id in enumerate(range(suffix.start, suffix.end)):
            tokens.append(
                Dsv4TokenInView(
                    packed_token_id=packed_id,
                    stream_id=suffix.stream_id,
                    view_pos=cursor,
                    stream_pos=offset,
                )
            )
            cursor += 1
    return Dsv4BranchView(
        branch_stream_id=prefix.stream_id if suffix is None else suffix.stream_id,
        prefix_stream_id=prefix.stream_id,
        suffix_stream_id=None if suffix is None else suffix.stream_id,
        tokens=tuple(tokens),
        prefix_token_count=prefix.size(),
    )


def _build_token_ownership(
    token_layout_index: TokenLayoutIndexLike,
) -> dict[int, tuple[int, int]]:
    owners: dict[int, tuple[int, int]] = {}
    for rank, ranges in enumerate(token_layout_index.ownership_ranges_by_rank):
        for start, end, local_start in ranges:
            for token_id in range(int(start), int(end)):
                owners[token_id] = (rank, int(local_start) + token_id - int(start))
    return owners


def _build_entries(
    *,
    branch_views: tuple[Dsv4BranchView, ...],
    token_owner: dict[int, tuple[int, int]],
    spec: Dsv4CompressionSpec,
) -> tuple[tuple[Dsv4CompressedEntry, ...], dict[int, tuple[int, ...]]]:
    entries: list[Dsv4CompressedEntry] = []
    entry_ids_by_branch: dict[int, list[int]] = defaultdict(list)
    dedupe: dict[tuple[Dsv4CompressionKind, int, tuple[int, ...], int], int] = {}
    for branch_view in branch_views:
        branch_entry_index = 0
        for dependency_positions in _dependency_positions(
            branch_len=branch_view.size(),
            spec=spec,
        ):
            dependency_tokens = tuple(
                int(branch_view.tokens[position].packed_token_id)
                for position in dependency_positions
            )
            closure_view_pos = dependency_positions[-1]
            closure_token_id = int(branch_view.tokens[closure_view_pos].packed_token_id)
            shared_prefix_entry = all(
                int(branch_view.tokens[position].view_pos)
                < int(branch_view.prefix_token_count)
                for position in dependency_positions
            )
            dedupe_key = (
                spec.kind,
                int(spec.ratio),
                dependency_tokens,
                closure_token_id,
            )
            if shared_prefix_entry and dedupe_key in dedupe:
                entry_ids_by_branch[int(branch_view.branch_stream_id)].append(
                    dedupe[dedupe_key]
                )
                branch_entry_index += 1
                continue
            owner_rank, owner_local_offset = _owner_for_token(
                token_owner=token_owner,
                token_id=closure_token_id,
            )
            remote_dependency_token_ids = tuple(
                token_id
                for token_id in dependency_tokens
                if _owner_for_token(token_owner=token_owner, token_id=token_id)[0]
                != owner_rank
            )
            entry = Dsv4CompressedEntry(
                entry_id=len(entries),
                kind=spec.kind,
                ratio=spec.ratio,
                branch_stream_id=branch_view.branch_stream_id,
                prefix_stream_id=branch_view.prefix_stream_id,
                closure_token_id=closure_token_id,
                closure_view_pos=closure_view_pos,
                owner_rank=owner_rank,
                owner_local_offset=owner_local_offset,
                dependency_token_ids=dependency_tokens,
                remote_dependency_token_ids=remote_dependency_token_ids,
                shared_prefix_entry=shared_prefix_entry,
                branch_entry_index=branch_entry_index,
            )
            if shared_prefix_entry:
                dedupe[dedupe_key] = int(entry.entry_id)
            entries.append(entry)
            entry_ids_by_branch[int(branch_view.branch_stream_id)].append(
                int(entry.entry_id)
            )
            branch_entry_index += 1
    return tuple(entries), {
        branch_id: tuple(entry_ids)
        for branch_id, entry_ids in sorted(entry_ids_by_branch.items())
    }


def _dependency_positions(
    *,
    branch_len: int,
    spec: Dsv4CompressionSpec,
) -> list[tuple[int, ...]]:
    if spec.kind == Dsv4CompressionKind.HCA:
        return [
            tuple(range(start, start + spec.ratio))
            for start in range(0, branch_len - spec.ratio + 1, spec.ratio)
        ]
    if spec.kind != Dsv4CompressionKind.CSA:
        raise RuntimeError(f"Unsupported DSV4 compression kind: {spec.kind}")
    positions: list[tuple[int, ...]] = []
    for start in range(0, branch_len - spec.ratio + 1, spec.ratio):
        current = tuple(range(start, start + spec.ratio))
        previous = tuple(range(start - spec.ratio, start)) if start > 0 else tuple()
        positions.append(previous + current)
    return positions


def _owner_for_token(
    *,
    token_owner: dict[int, tuple[int, int]],
    token_id: int,
) -> tuple[int, int]:
    owner = token_owner.get(int(token_id))
    if owner is None:
        raise RuntimeError(f"DSV4 token {token_id} has no CP owner")
    return owner


def _build_halo_transfers(
    entries: tuple[Dsv4CompressedEntry, ...],
    token_owner: dict[int, tuple[int, int]],
) -> tuple[Dsv4HaloTransfer, ...]:
    by_peer: dict[tuple[int, int], dict[int, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for entry in entries:
        for token_id in entry.remote_dependency_token_ids:
            source_rank, _ = _owner_for_token(
                token_owner=token_owner,
                token_id=token_id,
            )
            by_peer[(source_rank, entry.owner_rank)][token_id].add(entry.entry_id)

    transfers: list[Dsv4HaloTransfer] = []
    for (source_rank, target_rank), token_to_entries in sorted(by_peer.items()):
        token_ids = tuple(sorted(token_to_entries))
        entry_ids = tuple(
            sorted(
                {
                    entry_id
                    for entries_ in token_to_entries.values()
                    for entry_id in entries_
                }
            )
        )
        transfers.append(
            Dsv4HaloTransfer(
                source_rank=source_rank,
                target_rank=target_rank,
                token_ids=token_ids,
                entry_ids=entry_ids,
            )
        )
    return tuple(transfers)


def _entry_ids_by_owner(
    entries: tuple[Dsv4CompressedEntry, ...],
    token_layout_index: TokenLayoutIndexLike,
) -> tuple[tuple[int, ...], ...]:
    by_rank: list[list[int]] = [
        [] for _ in range(len(token_layout_index.ownership_ranges_by_rank))
    ]
    for entry in entries:
        by_rank[int(entry.owner_rank)].append(int(entry.entry_id))
    return tuple(tuple(entry_ids) for entry_ids in by_rank)
