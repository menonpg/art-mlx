# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side NCCL transport subset extracted from vLLM."""

import ctypes
from datetime import timedelta
import os
import pickle
import socket
from typing import Any

from pydantic import BaseModel, ConfigDict
import torch
from torch.distributed import TCPStore

from .packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    packed_broadcast_producer,
)


class TrainerNcclSendWeightsArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    group: Any
    src: int = 0
    post_iter_func: Any = None
    packed: bool = False
    stream: Any = None
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


_nccl_result_t = ctypes.c_int
_nccl_comm_t = ctypes.c_void_p
_cuda_stream_t = ctypes.c_void_p
_buffer_type = ctypes.c_void_p


class _NcclDataType:
    INT8 = 0
    UINT8 = 1
    INT32 = 2
    INT64 = 4
    FLOAT16 = 6
    FLOAT32 = 7
    FLOAT64 = 8
    BFLOAT16 = 9

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.INT8
        if dtype == torch.uint8:
            return cls.UINT8
        if dtype == torch.int32:
            return cls.INT32
        if dtype == torch.int64:
            return cls.INT64
        if dtype == torch.float16:
            return cls.FLOAT16
        if dtype == torch.float32:
            return cls.FLOAT32
        if dtype == torch.float64:
            return cls.FLOAT64
        if dtype == torch.bfloat16:
            return cls.BFLOAT16
        raise ValueError(f"Unsupported NCCL dtype: {dtype}")


class _NcclRedOp:
    SUM = 0


class _NcclLibrary:
    def __init__(self, so_file: str | None = None):
        self._lib = ctypes.CDLL(so_file or _find_nccl_library())
        self._configure("ncclGetErrorString", ctypes.c_char_p, [_nccl_result_t])
        self._configure("ncclGetUniqueId", _nccl_result_t, [ctypes.POINTER(_NcclUniqueId)])
        self._configure(
            "ncclCommInitRank",
            _nccl_result_t,
            [ctypes.POINTER(_nccl_comm_t), ctypes.c_int, _NcclUniqueId, ctypes.c_int],
        )
        self._configure(
            "ncclAllReduce",
            _nccl_result_t,
            [
                _buffer_type,
                _buffer_type,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                _nccl_comm_t,
                _cuda_stream_t,
            ],
        )
        self._configure(
            "ncclBroadcast",
            _nccl_result_t,
            [
                _buffer_type,
                _buffer_type,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                _nccl_comm_t,
                _cuda_stream_t,
            ],
        )

    def _configure(self, name: str, restype: Any, argtypes: list[Any]) -> None:
        function = getattr(self._lib, name)
        function.restype = restype
        function.argtypes = argtypes

    def _check(self, result: int) -> None:
        if result != 0:
            error = self._lib.ncclGetErrorString(result).decode("utf-8")
            raise RuntimeError(f"NCCL error: {error}")

    def get_unique_id(self) -> _NcclUniqueId:
        unique_id = _NcclUniqueId()
        self._check(self._lib.ncclGetUniqueId(ctypes.byref(unique_id)))
        return unique_id

    def init_rank(self, world_size: int, unique_id: _NcclUniqueId, rank: int) -> Any:
        comm = _nccl_comm_t()
        self._check(
            self._lib.ncclCommInitRank(
                ctypes.byref(comm), world_size, unique_id, rank
            )
        )
        return comm

    def all_reduce(
        self,
        tensor: torch.Tensor,
        comm: Any,
        stream: torch.cuda.Stream,
    ) -> None:
        self._check(
            self._lib.ncclAllReduce(
                _buffer_type(tensor.data_ptr()),
                _buffer_type(tensor.data_ptr()),
                tensor.numel(),
                _NcclDataType.from_torch(tensor.dtype),
                _NcclRedOp.SUM,
                comm,
                _cuda_stream_t(stream.cuda_stream),
            )
        )

    def broadcast(
        self,
        tensor: torch.Tensor,
        comm: Any,
        *,
        rank: int,
        src: int,
        stream: torch.cuda.Stream,
    ) -> None:
        send_buffer = _buffer_type(tensor.data_ptr()) if rank == src else _buffer_type()
        self._check(
            self._lib.ncclBroadcast(
                send_buffer,
                _buffer_type(tensor.data_ptr()),
                tensor.numel(),
                _NcclDataType.from_torch(tensor.dtype),
                src,
                comm,
                _cuda_stream_t(stream.cuda_stream),
            )
        )


class _BootstrapGroup:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        store_timeout: int = 300,
    ) -> None:
        launch_server = rank == 0
        listen_socket = None
        listen_fd = None
        if launch_server:
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_socket.bind((host, port))
            listen_socket.listen()
            listen_fd = listen_socket.fileno()
        self.rank = rank
        self.world_size = world_size
        self.socket = listen_socket
        self.store = TCPStore(
            host_name=host,
            port=port,
            world_size=world_size,
            is_master=launch_server,
            timeout=timedelta(seconds=store_timeout),
            use_libuv=False,
            master_listen_fd=listen_fd,
        )
        self._broadcast_send_counter = 0
        self._broadcast_recv_counter = {value: 0 for value in range(world_size)}

    def broadcast_obj(self, obj: Any | None, *, src: int) -> Any:
        if self.rank == src:
            key = f"broadcast_from/{src}/{self._broadcast_send_counter}"
            self.store.set(key, pickle.dumps(obj))
            self._broadcast_send_counter += 1
            return obj
        key = f"broadcast_from/{src}/{self._broadcast_recv_counter[src]}"
        received = pickle.loads(self.store.get(key))
        self._broadcast_recv_counter[src] += 1
        return received


class TrainerNcclCommunicator:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | torch.device,
    ) -> None:
        bootstrap_group = _BootstrapGroup(
            host=host,
            port=port,
            rank=rank,
            world_size=world_size,
        )
        self.rank = rank
        self.world_size = world_size
        self.device = (
            torch.device(f"cuda:{device}") if isinstance(device, int) else device
        )
        self._nccl = _NcclLibrary()
        unique_id = self._nccl.get_unique_id() if rank == 0 else _NcclUniqueId()
        unique_id = bootstrap_group.broadcast_obj(unique_id, src=0)
        with torch.cuda.device(self.device):
            self._comm = self._nccl.init_rank(world_size, unique_id, rank)
            stream = torch.cuda.current_stream(self.device)
            warmup = torch.zeros(1, device=self.device)
            self.all_reduce(warmup, stream=stream)
            stream.synchronize()

    def all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        assert tensor.device == self.device
        self._nccl.all_reduce(
            tensor,
            self._comm,
            stream=stream or torch.cuda.current_stream(self.device),
        )

    def broadcast(
        self,
        tensor: torch.Tensor,
        *,
        src: int,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        assert tensor.device == self.device
        self._nccl.broadcast(
            tensor,
            self._comm,
            rank=self.rank,
            src=src,
            stream=stream or torch.cuda.current_stream(self.device),
        )


def _find_nccl_library() -> str:
    if override := os.environ.get("VLLM_NCCL_SO_PATH"):
        return override
    if torch.version.cuda is not None:
        return "libnccl.so.2"
    if torch.version.hip is not None:
        return "librccl.so.1"
    raise ValueError("NCCL only supports CUDA and ROCm backends.")


def trainer_init(init_info: dict[str, object]) -> TrainerNcclCommunicator:
    return TrainerNcclCommunicator(
        host=str(init_info["master_address"]),
        port=int(init_info["master_port"]),
        rank=0,
        world_size=int(init_info["world_size"]),
        device=torch.cuda.current_device(),
    )


def trainer_send_weights(
    iterator: Any,
    trainer_args: dict[str, Any] | TrainerNcclSendWeightsArgs,
) -> None:
    args = (
        TrainerNcclSendWeightsArgs(**trainer_args)
        if isinstance(trainer_args, dict)
        else trainer_args
    )
    post_iter_func = args.post_iter_func or (lambda item: item[1])
    if args.packed:
        packed_broadcast_producer(
            iterator=iterator,
            group=args.group,
            src=args.src,
            post_iter_func=post_iter_func,
            buffer_size_bytes=args.packed_buffer_size_bytes,
            num_buffers=args.packed_num_buffers,
        )
        return
    for item in iterator:
        tensor = post_iter_func(item)
        args.group.broadcast(
            tensor,
            src=args.src,
            stream=args.stream or torch.cuda.current_stream(tensor.device),
        )
