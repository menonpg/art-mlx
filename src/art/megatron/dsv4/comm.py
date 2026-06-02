from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, PrivateAttr
import torch
import torch.distributed as dist

from .types import Dsv4TensorExchangePlan, Dsv4TensorIdBuffer

_DIST = cast(Any, dist)


class Dsv4TensorExchangeWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    recv_buffer: torch.Tensor
    recv_ids_by_peer: tuple[tuple[int, ...], ...]
    handle: Any | None
    send_buffer: torch.Tensor | None = None
    stream: torch.cuda.Stream | None = None
    output_ndim: int
    label: str = "dsv4_tensor_exchange"
    _wait_complete: bool = PrivateAttr(default=False)

    def wait(self) -> None:
        if self._wait_complete:
            return
        if self.handle is not None:
            self.handle.wait()
        if self.stream is not None:
            torch.cuda.current_stream(self.recv_buffer.device).wait_stream(self.stream)
        self._wait_complete = True

    def wait_post_process(self) -> Dsv4TensorIdBuffer:
        self.wait()
        ids = _flatten_peer_ids(self.recv_ids_by_peer, name="recv_ids_by_peer")
        tensor = _unpack_wire_rows(
            packed=self.recv_buffer,
            output_ndim=int(self.output_ndim),
            total_rows=len(ids),
        )
        return Dsv4TensorIdBuffer(ids=ids, tensor=tensor)


@torch.compiler.disable
def launch_dsv4_tensor_exchange(
    *,
    tensor: torch.Tensor,
    tensor_ids: tuple[int, ...],
    plan: Dsv4TensorExchangePlan,
    group: Any,
    async_op: bool,
    label: str = "dsv4_tensor_exchange",
) -> Dsv4TensorExchangeWork:
    """Exchange DSV4 KV-like tensor rows by explicit ids.

    This is bespoke DSV4 communication and should remain eager. For CUDA async
    execution it uses a side-stream handoff with stable packed buffers:
    pack on the producer stream, make the comm stream wait on that stream,
    record packed-buffer lifetimes on the comm stream, and make the consumer
    stream wait during `wait()`.
    """
    _validate_exchange_tensor(tensor)
    world_size = _world_size(group)
    _validate_plan(plan=plan, world_size=world_size)
    tensor_ids = _normalize_ids(
        ids=tensor_ids,
        expected_count=int(tensor.shape[_row_dim(tensor)]),
        name="tensor_ids",
    )

    send_buffer = _pack_send_rows(
        tensor=tensor,
        tensor_ids=tensor_ids,
        send_ids_by_peer=plan.send_ids_by_peer,
    )
    total_recv_rows = sum(len(peer_ids) for peer_ids in plan.recv_ids_by_peer)
    recv_buffer = tensor.new_empty(
        _wire_shape(
            tensor=tensor,
            total_rows=total_recv_rows,
        )
    )

    if world_size == 1:
        local = _pack_send_rows(
            tensor=tensor,
            tensor_ids=tensor_ids,
            send_ids_by_peer=plan.recv_ids_by_peer,
        )
        if tuple(local.shape) != tuple(recv_buffer.shape):
            raise RuntimeError(
                "DSV4 local exchange packed shape mismatch: "
                f"{tuple(local.shape)} vs {tuple(recv_buffer.shape)}"
            )
        recv_buffer.copy_(local)
        return Dsv4TensorExchangeWork(
            recv_buffer=recv_buffer,
            recv_ids_by_peer=plan.recv_ids_by_peer,
            handle=None,
            send_buffer=send_buffer,
            stream=None,
            output_ndim=tensor.ndim,
            label=label,
        )

    input_split_sizes = [len(peer_ids) for peer_ids in plan.send_ids_by_peer]
    output_split_sizes = [len(peer_ids) for peer_ids in plan.recv_ids_by_peer]
    stream = _get_comm_stream(tensor) if async_op and tensor.is_cuda else None
    if stream is not None:
        current_stream = torch.cuda.current_stream(tensor.device)
        stream.wait_stream(current_stream)
        send_buffer.record_stream(stream)
        recv_buffer.record_stream(stream)
        with torch.cuda.stream(stream):
            handle = _all_to_all_single(
                recv_buffer=recv_buffer,
                send_buffer=send_buffer,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
    else:
        handle = _all_to_all_single(
            recv_buffer=recv_buffer,
            send_buffer=send_buffer,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )
    return Dsv4TensorExchangeWork(
        recv_buffer=recv_buffer,
        recv_ids_by_peer=plan.recv_ids_by_peer,
        handle=handle,
        send_buffer=send_buffer,
        stream=stream,
        output_ndim=tensor.ndim,
        label=label,
    )


_STREAMS: dict[int, torch.cuda.Stream] = {}


def _get_comm_stream(tensor: torch.Tensor) -> torch.cuda.Stream:
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _STREAMS.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=tensor.device)
        _STREAMS[device_index] = stream
    return stream


def _all_to_all_single(
    *,
    recv_buffer: torch.Tensor,
    send_buffer: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group: Any,
    async_op: bool,
) -> Any | None:
    return _DIST.all_to_all_single(
        recv_buffer,
        send_buffer,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=async_op,
    )


def _world_size(group: Any) -> int:
    if group is None:
        return 1
    return int(_DIST.get_world_size(group))


def _validate_exchange_tensor(tensor: torch.Tensor) -> None:
    if tensor.ndim not in (2, 3):
        raise RuntimeError(
            "DSV4 tensor exchange supports [N,D] or [B,N,D] tensors, got "
            f"{tuple(tensor.shape)}"
        )


def _validate_plan(*, plan: Dsv4TensorExchangePlan, world_size: int) -> None:
    if len(plan.send_ids_by_peer) != int(world_size):
        raise RuntimeError(
            "DSV4 exchange send peer count must match world size, got "
            f"{len(plan.send_ids_by_peer)} vs {world_size}"
        )
    if len(plan.recv_ids_by_peer) != int(world_size):
        raise RuntimeError(
            "DSV4 exchange recv peer count must match world size, got "
            f"{len(plan.recv_ids_by_peer)} vs {world_size}"
        )
    for peer, ids in enumerate(plan.send_ids_by_peer):
        _row_by_id(ids, name=f"send_ids_by_peer[{peer}]")
    _flatten_peer_ids(plan.recv_ids_by_peer, name="recv_ids_by_peer")


def _normalize_ids(
    *,
    ids: tuple[int, ...],
    expected_count: int,
    name: str,
) -> tuple[int, ...]:
    ids = tuple(int(id_) for id_ in ids)
    if len(ids) != int(expected_count):
        raise RuntimeError(
            f"DSV4 {name} length must match tensor rows, got "
            f"{len(ids)} vs {expected_count}"
        )
    _row_by_id(ids, name=name)
    return ids


def _pack_send_rows(
    *,
    tensor: torch.Tensor,
    tensor_ids: tuple[int, ...],
    send_ids_by_peer: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    row_by_id = _row_by_id(tensor_ids, name="tensor_ids")
    requested = tuple(id_ for peer_ids in send_ids_by_peer for id_ in peer_ids)
    missing = tuple(int(id_) for id_ in requested if int(id_) not in row_by_id)
    if missing:
        raise RuntimeError(f"DSV4 exchange tensor is missing ids: {missing}")
    total_rows = len(requested)
    packed = tensor.new_empty(_wire_shape(tensor=tensor, total_rows=total_rows))
    cursor = 0
    for peer_ids in send_ids_by_peer:
        if not peer_ids:
            continue
        indices = torch.tensor(
            tuple(row_by_id[int(id_)] for id_ in peer_ids),
            device=tensor.device,
            dtype=torch.long,
        )
        gathered = tensor.index_select(_row_dim(tensor), indices)
        rows = _to_wire_rows(gathered)
        split = int(rows.shape[0])
        packed[cursor : cursor + split].copy_(rows)
        cursor += split
    if cursor != total_rows:
        raise RuntimeError(
            f"DSV4 exchange packed {cursor} rows but expected {total_rows}"
        )
    return packed


def _row_dim(tensor: torch.Tensor) -> int:
    return 0 if tensor.ndim == 2 else 1


def _wire_shape(*, tensor: torch.Tensor, total_rows: int) -> tuple[int, ...]:
    if tensor.ndim == 2:
        return (int(total_rows), int(tensor.shape[1]))
    return (int(total_rows), int(tensor.shape[0]), int(tensor.shape[2]))


def _to_wire_rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.contiguous()
    return tensor.permute(1, 0, 2).contiguous()


def _unpack_wire_rows(
    *,
    packed: torch.Tensor,
    output_ndim: int,
    total_rows: int,
) -> torch.Tensor:
    if output_ndim == 2:
        if int(packed.shape[0]) != int(total_rows):
            raise RuntimeError(
                "DSV4 exchange received row count mismatch: "
                f"{int(packed.shape[0])} vs {total_rows}"
            )
        return packed.contiguous()
    if output_ndim != 3:
        raise RuntimeError(f"Unsupported DSV4 exchange output rank {output_ndim}")
    if int(packed.shape[0]) != int(total_rows):
        raise RuntimeError(
            "DSV4 exchange received row count mismatch: "
            f"{int(packed.shape[0])} vs {total_rows}"
        )
    return packed.permute(1, 0, 2).contiguous()


def _flatten_peer_ids(
    ids_by_peer: tuple[tuple[int, ...], ...],
    *,
    name: str,
) -> tuple[int, ...]:
    ids = tuple(int(id_) for peer_ids in ids_by_peer for id_ in peer_ids)
    _row_by_id(ids, name=name)
    return ids


def _row_by_id(ids: tuple[int, ...], *, name: str) -> dict[int, int]:
    row_by_id: dict[int, int] = {}
    for row, id_ in enumerate(ids):
        id_int = int(id_)
        if id_int in row_by_id:
            raise RuntimeError(f"DSV4 {name} contains duplicate id {id_int}")
        row_by_id[id_int] = row
    return row_by_id
