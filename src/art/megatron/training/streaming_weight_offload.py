from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
import os
from typing import Any

from megatron.core.models.gpt import GPTModel
from megatron.core.tensor_parallel.random import is_checkpointing
from pydantic import BaseModel, ConfigDict, Field
import torch

from .model_chunks import ModelChunks


class StreamingWeightOffloadConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = False
    num_layers: int = Field(default=0, ge=0)


class _LayerState:
    def __init__(
        self, index: int, layer: torch.nn.Module, params: list[torch.nn.Parameter]
    ):
        self.index = index
        self.layer = layer
        self.params = params
        self.cpu_tensors: list[torch.Tensor] = []
        self.gpu_tensors: list[torch.Tensor] = []
        self.status = "gpu"
        self.load_event: torch.cuda.Event | None = None
        self.offload_event: torch.cuda.Event | None = None


class StreamingWeightOffloader:
    def __init__(
        self,
        *,
        layers: list[torch.nn.Module],
        rank: int,
        config: StreamingWeightOffloadConfig,
    ) -> None:
        self.rank = rank
        self.config = config
        selected_layers = layers[: config.num_layers or len(layers)]
        self.layers = [
            _LayerState(i, layer, _frozen_cuda_parameters(layer))
            for i, layer in enumerate(selected_layers)
        ]
        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()
        self._hooks: list[Any] = []

    def install(self) -> None:
        if not self.layers:
            raise RuntimeError(
                "Streaming weight offload found no transformer layers to manage"
            )
        for layer_state in self.layers:
            for param in layer_state.params:
                layer_state.cpu_tensors.append(
                    torch.empty(
                        param.shape,
                        dtype=param.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                )
                layer_state.gpu_tensors.append(param.data)
            self._hooks.append(
                layer_state.layer.register_forward_pre_hook(
                    lambda module, inputs, state=layer_state: self._pre_forward(state)
                )
            )
            self._hooks.append(
                layer_state.layer.register_forward_hook(
                    lambda module, inputs, output, state=layer_state: (
                        self._post_forward(state)
                    )
                )
            )
        self.offload_all(wait=True)
        if self.rank == 0:
            param_count = sum(
                param.numel() for layer in self.layers for param in layer.params
            )
            print(
                "Installed streaming frozen weight offload for "
                f"{len(self.layers)} layers ({param_count} rank-local params)"
            )

    def begin_job(self) -> None:
        self._finish_completed_offloads()
        self._start_load(0)

    def finish_job(self) -> None:
        self.offload_all(wait=True)

    def remove(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def offload_all(self, *, wait: bool) -> None:
        for layer_state in self.layers:
            self._ensure_offloaded(layer_state, wait=wait)
        if wait:
            self.d2h_stream.synchronize()
            self._finish_completed_offloads()
            torch.cuda.empty_cache()

    def _pre_forward(self, layer_state: _LayerState) -> None:
        self._finish_completed_offloads()
        recompute_forward = _is_recompute_forward()
        if recompute_forward:
            self._offload_recomputed_successors(layer_state.index)
            self._finish_pending_offloads()
        self._finish_load(layer_state)
        if not recompute_forward:
            self._finish_neighbor_offload(layer_state.index - 1)

    def _post_forward(self, layer_state: _LayerState) -> None:
        if is_checkpointing() and not torch.is_grad_enabled():
            self._start_offload(layer_state)
            self._start_load(layer_state.index + 1)

    def _offload_recomputed_successors(self, index: int) -> None:
        for layer_state in self.layers[index + 1 :]:
            if layer_state.status in {"gpu", "loading"}:
                self._ensure_offloaded(layer_state, wait=False)

    def _start_load(self, index: int) -> None:
        if index < 0 or index >= len(self.layers):
            return
        layer_state = self.layers[index]
        if layer_state.status in {"gpu", "loading"}:
            return
        if layer_state.status == "offloading":
            self._finish_offload(layer_state, wait=True)
        if layer_state.status != "cpu":
            raise RuntimeError(f"Unexpected layer offload state {layer_state.status!r}")
        layer_state.gpu_tensors = [
            torch.empty_like(cpu_tensor, device=torch.cuda.current_device())
            for cpu_tensor in layer_state.cpu_tensors
        ]
        current_stream = torch.cuda.current_stream()
        self.h2d_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.h2d_stream):
            for gpu_tensor, cpu_tensor in zip(
                layer_state.gpu_tensors, layer_state.cpu_tensors, strict=True
            ):
                gpu_tensor.copy_(cpu_tensor, non_blocking=True)
                gpu_tensor.record_stream(self.h2d_stream)
            event = torch.cuda.Event()
            event.record(self.h2d_stream)
        layer_state.load_event = event
        layer_state.status = "loading"

    def _finish_load(self, layer_state: _LayerState) -> None:
        if layer_state.status == "gpu":
            return
        if layer_state.status == "cpu":
            self._start_load(layer_state.index)
        if layer_state.status != "loading" or layer_state.load_event is None:
            raise RuntimeError(f"Unexpected layer load state {layer_state.status!r}")
        # Transformer Engine can launch work on internal streams. Complete the H2D
        # copy before installing the parameter pointer so every downstream stream
        # observes initialized weights.
        layer_state.load_event.synchronize()
        for param, gpu_tensor in zip(
            layer_state.params, layer_state.gpu_tensors, strict=True
        ):
            param.data = gpu_tensor
        layer_state.load_event = None
        layer_state.status = "gpu"

    def _ensure_offloaded(self, layer_state: _LayerState, *, wait: bool) -> None:
        if layer_state.status == "cpu":
            return
        if layer_state.status == "loading":
            self._finish_load(layer_state)
        if layer_state.status == "gpu":
            self._start_offload(layer_state)
        if layer_state.status == "offloading":
            self._finish_offload(layer_state, wait=wait)

    def _start_offload(self, layer_state: _LayerState) -> None:
        if layer_state.status == "cpu":
            return
        if layer_state.status == "loading":
            self._finish_load(layer_state)
        if layer_state.status != "gpu":
            raise RuntimeError(f"Unexpected layer offload state {layer_state.status!r}")
        current_stream = torch.cuda.current_stream()
        self.d2h_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.d2h_stream):
            for cpu_tensor, gpu_tensor in zip(
                layer_state.cpu_tensors, layer_state.gpu_tensors, strict=True
            ):
                cpu_tensor.copy_(gpu_tensor, non_blocking=True)
                gpu_tensor.record_stream(self.d2h_stream)
            event = torch.cuda.Event()
            event.record(self.d2h_stream)
        layer_state.offload_event = event
        layer_state.status = "offloading"

    def _finish_completed_offloads(self) -> None:
        for layer_state in self.layers:
            if layer_state.status == "offloading":
                self._finish_offload(layer_state, wait=False)

    def _finish_pending_offloads(self) -> None:
        for layer_state in self.layers:
            if layer_state.status == "offloading":
                self._finish_offload(layer_state, wait=True)

    def _finish_neighbor_offload(self, index: int) -> None:
        if 0 <= index < len(self.layers):
            layer_state = self.layers[index]
            if layer_state.status == "offloading":
                self._finish_offload(layer_state, wait=True)

    def _finish_offload(self, layer_state: _LayerState, *, wait: bool) -> None:
        event = layer_state.offload_event
        if event is None:
            return
        if wait:
            event.synchronize()
        elif not event.query():
            return
        for param, cpu_tensor in zip(
            layer_state.params, layer_state.cpu_tensors, strict=True
        ):
            param.data = cpu_tensor
        layer_state.gpu_tensors = []
        layer_state.offload_event = None
        layer_state.status = "cpu"


def streaming_weight_offload_config_from_env() -> StreamingWeightOffloadConfig:
    return StreamingWeightOffloadConfig(
        enabled=_env_flag("ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD"),
        num_layers=_env_int("ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD_NUM_LAYERS", 0),
    )


def maybe_install_streaming_weight_offload(
    *,
    model: ModelChunks,
    rank: int,
    compile_enabled: bool,
) -> StreamingWeightOffloader | None:
    config = streaming_weight_offload_config_from_env()
    if not config.enabled:
        return None
    if compile_enabled:
        raise RuntimeError(
            "Streaming weight offload requires uncompiled transformer layers"
        )
    layers = _transformer_layers(model)
    if not layers:
        raise RuntimeError("Streaming weight offload could not find transformer layers")
    _validate_checkpoint_shape(layers[0])
    offloader = StreamingWeightOffloader(layers=layers, rank=rank, config=config)
    offloader.install()
    return offloader


def _validate_checkpoint_shape(layer: torch.nn.Module) -> None:
    config = getattr(layer, "config", None)
    if (
        getattr(config, "recompute_granularity", None) != "full"
        or getattr(config, "recompute_method", None) != "uniform"
        or int(getattr(config, "recompute_num_layers", 0) or 0) != 1
    ):
        raise RuntimeError(
            "Streaming weight offload requires full uniform activation recompute with "
            "recompute_num_layers=1"
        )


def _transformer_layers(model: Sequence[torch.nn.Module]) -> list[torch.nn.Module]:
    layers: list[torch.nn.Module] = []
    for chunk in model:
        module = _unwrap_module(chunk)
        gpt_module = (
            module
            if isinstance(module, GPTModel)
            else getattr(module, "language_model", None)
        )
        decoder = getattr(gpt_module, "decoder", None)
        chunk_layers = getattr(decoder, "layers", None)
        if chunk_layers is not None:
            layers.extend(list(chunk_layers))
    return layers


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    current = module
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        for attr_name in ("_orig_mod", "module"):
            child = getattr(current, attr_name, None)
            if isinstance(child, torch.nn.Module):
                current = child
                break
        else:
            return current
    return current


def _frozen_cuda_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [
        param
        for param in module.parameters()
        if isinstance(param, torch.nn.Parameter)
        and not param.requires_grad
        and param.device.type == "cuda"
    ]


def _is_recompute_forward() -> bool:
    return is_checkpointing() and torch.is_grad_enabled()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    with suppress(ValueError):
        return int(raw)
    raise RuntimeError(f"{name} must be an integer, got {raw!r}")
