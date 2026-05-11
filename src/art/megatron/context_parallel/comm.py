from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
import torch.distributed as dist

from .range_ops import (
    range_gather,
    range_gather_head_major,
    range_reduce_sum_,
    range_reduce_sum_head_major_,
)
from .types import DkvReducePlan, KvFetchPlan, TokenRange

_DIST = cast(Any, dist)
class _Waitable(Protocol):
    def wait(self) -> Any: ...


def _active_peer_ranks(
    *,
    send_splits: tuple[int, ...],
    recv_splits: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        peer_rank
        for peer_rank, (send_split, recv_split) in enumerate(
            zip(send_splits, recv_splits, strict=True)
        )
        if int(send_split) > 0 or int(recv_split) > 0
    )


def _collective_mode(
    *,
    send_splits: tuple[int, ...],
    recv_splits: tuple[int, ...],
) -> str:
    active_peers = _active_peer_ranks(
        send_splits=send_splits,
        recv_splits=recv_splits,
    )
    if not active_peers:
        return "none"
    # Every rank participating in one peer exchange must choose the same collective.
    # Local heuristics can disagree across edge and middle ranks for the same wave.
    return "a2a"


def _launch_peer_exchange(
    *,
    recv_buffer: torch.Tensor,
    send_buffer: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group: Any,
    async_op: bool,
) -> _Waitable | None:
    collective_mode = _collective_mode(
        send_splits=tuple(int(split // 2) for split in input_split_sizes),
        recv_splits=tuple(int(split // 2) for split in output_split_sizes),
    )
    if collective_mode == "a2a":
        return cast(
            _Waitable | None,
            _DIST.all_to_all_single(
                recv_buffer,
                send_buffer,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=async_op,
            ),
        )
    if collective_mode == "none":
        return None
    raise RuntimeError(f"Unsupported peer-exchange mode: {collective_mode}")


@dataclass
class KvFetchWork:
    packed_buffer: torch.Tensor
    recv_splits: tuple[int, ...]
    handle: _Waitable | None
    send_buffer: torch.Tensor | None = None
    stream: torch.cuda.Stream | None = None
    label: str = "kv_fetch"
    output_layout: str = "head_major"
    _wait_complete: bool = False

    def is_completed(self) -> bool:
        if self._wait_complete:
            return True
        handle_complete = True
        if self.handle is not None:
            is_completed = getattr(self.handle, "is_completed", None)
            if callable(is_completed):
                handle_complete = bool(is_completed())
        if self.stream is not None:
            return handle_complete and bool(self.stream.query())
        return handle_complete

    def wait(self) -> None:
        if self._wait_complete:
            return
        if self.handle is not None:
            self.handle.wait()
        if self.stream is not None:
            current_stream = torch.cuda.current_stream(self.packed_buffer.device)
            current_stream.wait_stream(self.stream)
        self._wait_complete = True

    def wait_post_process(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.wait()
        return _unpack_packed_tensor_per_peer(
            self.packed_buffer,
            self.recv_splits,
            output_layout=self.output_layout,
        )


@dataclass
class DkvReduceWork:
    packed_buffer: torch.Tensor | None
    handle: _Waitable | None
    send_buffer: torch.Tensor | None
    stream: torch.cuda.Stream | None
    plan: DkvReducePlan
    dk_local: torch.Tensor
    dv_local: torch.Tensor
    range_meta_cache: dict[Any, Any] | None = None
    label: str = "dkv_reduce"
    input_layout: str = "token_major"
    _wait_complete: bool = False

    def is_completed(self) -> bool:
        if self._wait_complete:
            return True
        handle_complete = True
        if self.handle is not None:
            is_completed = getattr(self.handle, "is_completed", None)
            if callable(is_completed):
                handle_complete = bool(is_completed())
        if self.stream is not None:
            return handle_complete and bool(self.stream.query())
        return handle_complete

    def wait(self) -> None:
        if self._wait_complete:
            return
        if self.handle is not None:
            self.handle.wait()
        if self.stream is not None and self.packed_buffer is not None:
            current_stream = torch.cuda.current_stream(self.packed_buffer.device)
            current_stream.wait_stream(self.stream)
        self._wait_complete = True

    def wait_post_process(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.wait()
        if self.packed_buffer is not None and int(self.packed_buffer.shape[0]) > 0:
            dk_remote, dv_remote = _unpack_packed_tensor_per_peer(
                self.packed_buffer,
                self.plan.recv_splits,
                output_layout=(
                    "head_major" if self.input_layout == "head_major" else "token_major"
                ),
            )
            flattened_ranges = tuple(
                range_
                for peer_ranges in self.plan.recv_ranges_by_peer
                for range_ in peer_ranges
                if range_.size() > 0
            )

            def _apply_reduce() -> None:
                dk_reduce = (
                    dk_remote
                    if dk_remote.dtype == self.dk_local.dtype
                    else dk_remote.to(dtype=self.dk_local.dtype)
                )
                dv_reduce = (
                    dv_remote
                    if dv_remote.dtype == self.dv_local.dtype
                    else dv_remote.to(dtype=self.dv_local.dtype)
                )
                reduce_fn = (
                    range_reduce_sum_head_major_
                    if self.input_layout == "head_major"
                    else range_reduce_sum_
                )
                reduce_fn(
                    dk_reduce,
                    output_tensor=self.dk_local,
                    ranges=flattened_ranges,
                    range_meta_cache=self.range_meta_cache,
                )
                reduce_fn(
                    dv_reduce,
                    output_tensor=self.dv_local,
                    ranges=flattened_ranges,
                    range_meta_cache=self.range_meta_cache,
                )
                return

            _apply_reduce()
        return self.dk_local, self.dv_local


class A2AVCommunicator:
    def __init__(self) -> None:
        self._streams: dict[int, torch.cuda.Stream] = {}

    def _get_stream(self, tensor: torch.Tensor) -> torch.cuda.Stream | None:
        if not tensor.is_cuda:
            return None
        device_index = tensor.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        stream = self._streams.get(device_index)
        if stream is None:
            stream = torch.cuda.Stream(device=tensor.device)
            self._streams[device_index] = stream
        return stream

    def launch_kv_fetch(
        self,
        *,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        plan: KvFetchPlan,
        group: Any,
        async_op: bool,
        range_meta_cache: dict[Any, Any] | None = None,
        label: str = "kv_fetch",
        input_layout: str = "token_major",
        output_layout: str = "head_major",
    ) -> KvFetchWork:
        if group is None or _DIST.get_world_size(group) == 1:
            return KvFetchWork(
                packed_buffer=k_local.new_empty(
                    _packed_peer_tensor_shape(
                        tensor=k_local,
                        total_rows=0,
                        input_layout=input_layout,
                    )
                ),
                recv_splits=plan.recv_splits,
                handle=None,
                label=label,
                output_layout=output_layout,
            )

        total_send_rows = int(sum(plan.send_splits))
        total_recv_rows = int(sum(plan.recv_splits))
        recv_packed = k_local.new_empty(
            _packed_peer_tensor_shape(
                tensor=k_local,
                total_rows=total_recv_rows,
                input_layout=input_layout,
            )
        )
        input_split_sizes = [split * 2 for split in plan.send_splits]
        output_split_sizes = [split * 2 for split in plan.recv_splits]
        stream = self._get_stream(k_local) if async_op else None
        if stream is not None:
            current_stream = torch.cuda.current_stream(k_local.device)
            if total_send_rows <= 0:
                send_buffer = k_local.new_empty(
                    _packed_peer_tensor_shape(
                        tensor=k_local,
                        total_rows=0,
                        input_layout=input_layout,
                    )
                )
            else:
                send_buffer = _pack_gathered_tensors_per_peer(
                    left_tensor=k_local,
                    right_tensor=v_local,
                    ranges_by_peer=plan.send_ranges_by_peer,
                    range_meta_cache=range_meta_cache,
                    input_layout=input_layout,
                )
            stream.wait_stream(current_stream)
            send_buffer.record_stream(stream)
            recv_packed.record_stream(stream)
            with torch.cuda.stream(stream):
                handle = _launch_peer_exchange(
                    recv_buffer=recv_packed,
                    send_buffer=send_buffer,
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group,
                    async_op=True,
                )
        else:
            if total_send_rows <= 0:
                send_buffer = k_local.new_empty(
                    _packed_peer_tensor_shape(
                        tensor=k_local,
                        total_rows=0,
                        input_layout=input_layout,
                    )
                )
                handle = _launch_peer_exchange(
                    recv_buffer=recv_packed,
                    send_buffer=send_buffer,
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group,
                    async_op=async_op,
                )
            else:
                send_buffer = _pack_gathered_tensors_per_peer(
                    left_tensor=k_local,
                    right_tensor=v_local,
                    ranges_by_peer=plan.send_ranges_by_peer,
                    range_meta_cache=range_meta_cache,
                    input_layout=input_layout,
                )
                handle = _launch_peer_exchange(
                    recv_buffer=recv_packed,
                    send_buffer=send_buffer,
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group,
                    async_op=async_op,
                )
        return KvFetchWork(
            packed_buffer=recv_packed,
            recv_splits=plan.recv_splits,
            handle=handle,
            send_buffer=send_buffer,
            stream=stream,
            label=label,
            output_layout=output_layout,
        )

    def launch_dkv_reduce(
        self,
        *,
        dk_remote: torch.Tensor,
        dv_remote: torch.Tensor,
        plan: DkvReducePlan,
        group: Any,
        async_op: bool,
        dk_local: torch.Tensor,
        dv_local: torch.Tensor,
        range_meta_cache: dict[Any, Any] | None = None,
        label: str = "dkv_reduce",
        input_layout: str = "token_major",
    ) -> DkvReduceWork:
        if group is None or _DIST.get_world_size(group) == 1:
            return DkvReduceWork(
                packed_buffer=None,
                handle=None,
                send_buffer=None,
                stream=None,
                plan=plan,
                dk_local=dk_local,
                dv_local=dv_local,
                range_meta_cache=range_meta_cache,
                label=label,
            )

        total_send_rows = int(sum(plan.send_splits))
        recv_total = int(sum(plan.recv_splits))
        recv_packed = (
            dk_remote.new_empty(
                _packed_peer_tensor_shape(
                    tensor=dk_remote,
                    total_rows=recv_total,
                    input_layout=input_layout,
                )
            )
            if recv_total > 0
            else dk_remote.new_empty(
                _packed_peer_tensor_shape(
                    tensor=dk_remote,
                    total_rows=0,
                    input_layout=input_layout,
                )
            )
        )
        input_split_sizes = [split * 2 for split in plan.send_splits]
        output_split_sizes = [split * 2 for split in plan.recv_splits]
        stream = self._get_stream(dk_remote) if async_op else None
        if stream is not None:
            current_stream = torch.cuda.current_stream(dk_remote.device)
            if total_send_rows <= 0:
                send_buffer = dk_remote.new_empty(
                    _packed_peer_tensor_shape(
                        tensor=dk_remote,
                        total_rows=0,
                        input_layout=input_layout,
                    )
                )
            else:
                send_buffer = _pack_split_tensors_by_peer(
                    left_tensor=dk_remote,
                    right_tensor=dv_remote,
                    splits=plan.send_splits,
                    input_layout=input_layout,
                )
            stream.wait_stream(current_stream)
            send_buffer.record_stream(stream)
            recv_packed.record_stream(stream)
            with torch.cuda.stream(stream):
                handle = _launch_peer_exchange(
                    recv_buffer=recv_packed,
                    send_buffer=send_buffer,
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group,
                    async_op=True,
                )
        else:
            if total_send_rows <= 0:
                send_buffer = dk_remote.new_empty(
                    _packed_peer_tensor_shape(
                        tensor=dk_remote,
                        total_rows=0,
                        input_layout=input_layout,
                    )
                )
                handle = _launch_peer_exchange(
                    recv_buffer=recv_packed,
                    send_buffer=send_buffer,
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group,
                    async_op=async_op,
                )
            else:
                send_buffer = _pack_split_tensors_by_peer(
                    left_tensor=dk_remote,
                    right_tensor=dv_remote,
                    splits=plan.send_splits,
                    input_layout=input_layout,
                )
                handle = _launch_peer_exchange(
                    recv_buffer=recv_packed,
                    send_buffer=send_buffer,
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group,
                    async_op=async_op,
                )
        return DkvReduceWork(
            packed_buffer=recv_packed if recv_total > 0 else None,
            handle=handle,
            send_buffer=send_buffer,
            stream=stream,
            plan=plan,
            dk_local=dk_local,
            dv_local=dv_local,
            range_meta_cache=range_meta_cache,
            label=label,
            input_layout=input_layout,
        )


def range_gather_per_peer(
    input_tensor: torch.Tensor,
    ranges_by_peer: tuple[tuple[TokenRange, ...], ...],
    range_meta_cache: dict[Any, Any] | None = None,
) -> torch.Tensor:
    chunks = [
        range_gather(
            input_tensor,
            peer_ranges,
            range_meta_cache=range_meta_cache,
        )
        for peer_ranges in ranges_by_peer
    ]
    if not chunks:
        return input_tensor.new_empty((0, *input_tensor.shape[1:]))
    nonempty = [chunk for chunk in chunks if int(chunk.shape[0]) > 0]
    if not nonempty:
        return input_tensor.new_empty((0, *input_tensor.shape[1:]))
    return torch.cat(chunks, dim=0).contiguous()


def _split_tensor_to_peer(
    input_tensor: torch.Tensor,
    splits: tuple[int, ...],
) -> torch.Tensor:
    if int(sum(splits)) == 0:
        return input_tensor.new_empty((0, *input_tensor.shape[1:]))
    if int(input_tensor.shape[0]) == int(sum(splits)):
        return input_tensor.contiguous()
    if len([split for split in splits if split > 0]) > 1:
        raise RuntimeError(
            f"Expected at most one non-zero send split for dKV reduce, got {splits}"
        )
    pieces: list[torch.Tensor] = []
    cursor = 0
    for split in splits:
        if split == 0:
            pieces.append(input_tensor.new_empty((0, *input_tensor.shape[1:])))
            continue
        pieces.append(input_tensor[cursor : cursor + split])
        cursor += split
    return torch.cat(pieces, dim=0).contiguous()


def _pack_gathered_tensors_per_peer(
    *,
    left_tensor: torch.Tensor,
    right_tensor: torch.Tensor,
    ranges_by_peer: tuple[tuple[TokenRange, ...], ...],
    range_meta_cache: dict[Any, Any] | None = None,
    input_layout: str = "token_major",
) -> torch.Tensor:
    if input_layout == "head_major":
        return _pack_gathered_tensors_per_peer_head_major(
            left_tensor=left_tensor,
            right_tensor=right_tensor,
            ranges_by_peer=ranges_by_peer,
            range_meta_cache=range_meta_cache,
        )
    if input_layout != "token_major":
        raise ValueError(f"Unsupported gathered-pack input layout: {input_layout}")
    total_rows = sum(
        range_.size() for peer_ranges in ranges_by_peer for range_ in peer_ranges
    )
    if total_rows == 0:
        return left_tensor.new_empty((0, *left_tensor.shape[1:]))
    packed = left_tensor.new_empty((total_rows * 2, *left_tensor.shape[1:]))
    cursor = 0
    for peer_ranges in ranges_by_peer:
        split = sum(range_.size() for range_ in peer_ranges)
        if split <= 0:
            continue
        range_gather(
            left_tensor,
            peer_ranges,
            output=packed[cursor : cursor + split],
            range_meta_cache=range_meta_cache,
        )
        range_gather(
            right_tensor,
            peer_ranges,
            output=packed[cursor + split : cursor + split * 2],
            range_meta_cache=range_meta_cache,
        )
        cursor += split * 2
    return packed


def _pack_gathered_tensors_per_peer_head_major(
    *,
    left_tensor: torch.Tensor,
    right_tensor: torch.Tensor,
    ranges_by_peer: tuple[tuple[TokenRange, ...], ...],
    range_meta_cache: dict[Any, Any] | None = None,
) -> torch.Tensor:
    total_rows = sum(
        range_.size() for peer_ranges in ranges_by_peer for range_ in peer_ranges
    )
    if total_rows == 0:
        return left_tensor.new_empty((0, left_tensor.shape[0], left_tensor.shape[2]))
    packed = left_tensor.new_empty(
        (total_rows * 2, left_tensor.shape[0], left_tensor.shape[2])
    )
    cursor = 0
    for peer_ranges in ranges_by_peer:
        split = sum(range_.size() for range_ in peer_ranges)
        if split <= 0:
            continue
        packed[cursor : cursor + split].copy_(
            range_gather_head_major(
                left_tensor,
                peer_ranges,
                range_meta_cache=range_meta_cache,
            ).permute(1, 0, 2)
        )
        packed[cursor + split : cursor + split * 2].copy_(
            range_gather_head_major(
                right_tensor,
                peer_ranges,
                range_meta_cache=range_meta_cache,
            ).permute(1, 0, 2)
        )
        cursor += split * 2
    return packed


def _pack_split_tensors_by_peer(
    *,
    left_tensor: torch.Tensor,
    right_tensor: torch.Tensor,
    splits: tuple[int, ...],
    input_layout: str = "token_major",
) -> torch.Tensor:
    if input_layout == "head_major":
        return _pack_split_tensors_by_peer_head_major(
            left_tensor=left_tensor,
            right_tensor=right_tensor,
            splits=splits,
        )
    if input_layout != "token_major":
        raise ValueError(f"Unsupported split-pack input layout: {input_layout}")
    total_rows = int(sum(splits))
    if total_rows == 0:
        return left_tensor.new_empty((0, *left_tensor.shape[1:]))
    packed = left_tensor.new_empty((total_rows * 2, *left_tensor.shape[1:]))
    cursor = 0
    for split in splits:
        if split <= 0:
            continue
        packed[cursor * 2 : cursor * 2 + split].copy_(
            left_tensor[cursor : cursor + split]
        )
        packed[cursor * 2 + split : cursor * 2 + split * 2].copy_(
            right_tensor[cursor : cursor + split]
        )
        cursor += split
    if cursor != int(left_tensor.shape[0]) or cursor != int(right_tensor.shape[0]):
        raise RuntimeError(
            "Packed split consumed the wrong number of rows: "
            f"consumed={cursor}, left={int(left_tensor.shape[0])}, right={int(right_tensor.shape[0])}"
        )
    return packed


def _packed_peer_tensor_shape(
    *,
    tensor: torch.Tensor,
    total_rows: int,
    input_layout: str,
) -> tuple[int, ...]:
    if input_layout == "head_major":
        return (total_rows * 2, int(tensor.shape[0]), int(tensor.shape[2]))
    if input_layout != "token_major":
        raise ValueError(f"Unsupported split-pack input layout: {input_layout}")
    return (total_rows * 2, *tuple(int(dim) for dim in tensor.shape[1:]))


def _pack_split_tensors_by_peer_head_major(
    *,
    left_tensor: torch.Tensor,
    right_tensor: torch.Tensor,
    splits: tuple[int, ...],
) -> torch.Tensor:
    total_rows = int(sum(splits))
    if total_rows == 0:
        return left_tensor.new_empty((0, left_tensor.shape[0], left_tensor.shape[2]))
    packed = left_tensor.new_empty(
        (total_rows * 2, left_tensor.shape[0], left_tensor.shape[2])
    )
    cursor = 0
    for split in splits:
        if split <= 0:
            continue
        packed[cursor * 2 : cursor * 2 + split].copy_(
            left_tensor[:, cursor : cursor + split].permute(1, 0, 2)
        )
        packed[cursor * 2 + split : cursor * 2 + split * 2].copy_(
            right_tensor[:, cursor : cursor + split].permute(1, 0, 2)
        )
        cursor += split
    if cursor != int(left_tensor.shape[1]) or cursor != int(right_tensor.shape[1]):
        raise RuntimeError(
            "Head-major split pack consumed the wrong number of rows: "
            f"consumed={cursor}, left={int(left_tensor.shape[1])}, right={int(right_tensor.shape[1])}"
        )
    return packed


def _unpack_packed_tensor_per_peer(
    packed_tensor: torch.Tensor,
    splits: tuple[int, ...],
    *,
    output_layout: str = "token_major",
) -> tuple[torch.Tensor, torch.Tensor]:
    if output_layout == "head_major":
        return _unpack_packed_tensor_per_peer_head_major(
            packed_tensor,
            splits,
        )
    if output_layout != "token_major":
        raise ValueError(f"Unsupported packed-tensor output layout: {output_layout}")
    if int(packed_tensor.shape[0]) == 0:
        empty = packed_tensor.new_empty((0, *packed_tensor.shape[1:]))
        return empty, empty
    total_rows = 0
    cursor = 0
    for split in splits:
        if split <= 0:
            continue
        cursor += split * 2
        total_rows += split
    if cursor != int(packed_tensor.shape[0]):
        raise RuntimeError(
            "Packed tensor unpack consumed the wrong number of rows: "
            f"consumed={cursor}, input={int(packed_tensor.shape[0])}"
        )
    left = packed_tensor.new_empty((total_rows, *packed_tensor.shape[1:]))
    right = packed_tensor.new_empty((total_rows, *packed_tensor.shape[1:]))
    in_cursor = 0
    out_cursor = 0
    for split in splits:
        if split <= 0:
            continue
        left[out_cursor : out_cursor + split].copy_(
            packed_tensor[in_cursor : in_cursor + split]
        )
        right[out_cursor : out_cursor + split].copy_(
            packed_tensor[in_cursor + split : in_cursor + split * 2]
        )
        in_cursor += split * 2
        out_cursor += split
    return left, right


def _unpack_packed_tensor_per_peer_head_major(
    packed_tensor: torch.Tensor,
    splits: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(packed_tensor.shape[0]) == 0:
        empty = packed_tensor.new_empty(
            (packed_tensor.shape[1], 0, packed_tensor.shape[2])
        )
        return empty, empty
    total_rows = 0
    cursor = 0
    for split in splits:
        if split <= 0:
            continue
        cursor += split * 2
        total_rows += split
    if cursor != int(packed_tensor.shape[0]):
        raise RuntimeError(
            "Packed tensor unpack consumed the wrong number of rows: "
            f"consumed={cursor}, input={int(packed_tensor.shape[0])}"
        )
    left = packed_tensor.new_empty(
        (packed_tensor.shape[1], total_rows, packed_tensor.shape[2])
    )
    right = packed_tensor.new_empty(
        (packed_tensor.shape[1], total_rows, packed_tensor.shape[2])
    )
    in_cursor = 0
    out_cursor = 0
    for split in splits:
        if split <= 0:
            continue
        left[:, out_cursor : out_cursor + split].copy_(
            packed_tensor[in_cursor : in_cursor + split].permute(1, 0, 2)
        )
        right[:, out_cursor : out_cursor + split].copy_(
            packed_tensor[in_cursor + split : in_cursor + split * 2].permute(1, 0, 2)
        )
        in_cursor += split * 2
        out_cursor += split
    return left, right
