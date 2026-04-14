from collections.abc import Iterator
from dataclasses import dataclass, field
import gc
from typing import Any, Sequence

import torch

_SYNC_DEALLOC_FAKE_REGISTERED = False


@dataclass
class OffloadState:
    pinned_buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    is_offloaded: bool = False


def _maybe_register_sync_dealloc_fake() -> None:
    global _SYNC_DEALLOC_FAKE_REGISTERED
    if _SYNC_DEALLOC_FAKE_REGISTERED:
        return
    streams_ops = getattr(torch.ops, "streams", None)
    if streams_ops is None or not hasattr(streams_ops, "sync_dealloc"):
        return
    try:

        @torch.library.register_fake("streams::sync_dealloc")
        def _sync_dealloc_fake(*args, **kwargs):
            del args, kwargs
            return None
    except RuntimeError as exc:
        if "already has a fake impl registered" not in str(exc):
            raise
    _SYNC_DEALLOC_FAKE_REGISTERED = True


def _iter_megatron_param_buffers(model: Sequence[torch.nn.Module]) -> Iterator[Any]:
    for chunk in model:
        chunk_buffers = getattr(chunk, "buffers", None)
        if callable(chunk_buffers):
            raise RuntimeError("Megatron chunk is missing distributed param buffers")
        if chunk_buffers is not None:
            yield from chunk_buffers
        expert_buffers = getattr(chunk, "expert_parallel_buffers", None)
        if expert_buffers is not None:
            yield from expert_buffers


def offload_to_cpu(
    model: Sequence[torch.nn.Module],
    rank: int,
    offload_state: OffloadState,
) -> None:
    """Offload model params to CPU pinned memory."""
    if offload_state.is_offloaded:
        return
    pinned_buffers = offload_state.pinned_buffers

    for param_buffer in _iter_megatron_param_buffers(model):
        param_buffer.offload_to_cpu(move_params=True, move_grads=True)
    _maybe_register_sync_dealloc_fake()

    # Megatron remaps trainable params into contiguous DDP buffers. Offload those via the
    # native buffer APIs above, and only manually offload frozen params here.
    for chunk in model:
        for param in chunk.parameters():
            if (
                not isinstance(param, torch.nn.Parameter)
                or param.requires_grad
                or param.device.type != "cuda"
            ):
                continue
            key = f"param_{id(param)}"
            if (
                key not in pinned_buffers
                or pinned_buffers[key].shape != param.shape
                or pinned_buffers[key].dtype != param.dtype
            ):
                pinned_buffers[key] = torch.empty(
                    param.shape, dtype=param.dtype, device="cpu", pin_memory=True
                )
            pinned_buffers[key].copy_(param.data, non_blocking=True)
            param.data = pinned_buffers[key]

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    offload_state.is_offloaded = True
    if rank == 0:
        print("Offloaded model params to CPU")


def reload_to_gpu(
    model: Sequence[torch.nn.Module],
    rank: int,
    offload_state: OffloadState,
    device: torch.device | str | None = None,
) -> None:
    """Reload model params to GPU."""
    if not offload_state.is_offloaded:
        return

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)

    for param_buffer in _iter_megatron_param_buffers(model):
        param_buffer.reload_from_cpu(move_params=True, move_grads=True)
    _maybe_register_sync_dealloc_fake()

    # Reload frozen params that were manually offloaded.
    for chunk in model:
        for param in chunk.parameters():
            if (
                not isinstance(param, torch.nn.Parameter)
                or param.requires_grad
                or param.device.type != "cpu"
            ):
                continue
            gpu_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)
            gpu_tensor.copy_(param.data, non_blocking=True)
            param.data = gpu_tensor

    torch.cuda.synchronize()
    offload_state.is_offloaded = False
    if rank == 0:
        print("Reloaded LoRA params to GPU")
