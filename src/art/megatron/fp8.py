from __future__ import annotations

from typing import Any

import torch

_PARAM_ATTRS = (
    "allreduce",
    "partition_dim",
    "partition_stride",
    "tensor_model_parallel",
)


def _copy_param_attrs(source: torch.Tensor, target: torch.Tensor) -> None:
    for name in _PARAM_ATTRS:
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))


def quantize_loaded_fp8_base_weights(model: list[Any]) -> None:
    config = getattr(model[0], "config", None) if model else None
    if not bool(getattr(config, "fp8_param", False)):
        return

    from megatron.core.fp8_utils import is_float8tensor
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

    converted = 0
    already_quantized = 0
    for model_module in model:
        for module in model_module.modules():
            if not bool(getattr(module, "primary_weights_in_fp8", False)):
                continue
            init_meta = getattr(module, "param_init_meta", None)
            quantizers = getattr(module, "quantizers", None)
            if not init_meta or not quantizers:
                raise RuntimeError(
                    f"{type(module).__name__} requested FP8 params without TE metadata"
                )
            for name, meta in init_meta.items():
                fp8_meta_index = getattr(meta, "fp8_meta_index", None)
                if fp8_meta_index is None or not hasattr(module, name):
                    continue
                parameter = getattr(module, name)
                if is_float8tensor(parameter) or isinstance(parameter, QuantizedTensor):
                    already_quantized += 1
                    continue
                if parameter.device.type == "meta":
                    raise RuntimeError(
                        f"{type(module).__name__}.{name} is still on meta device"
                    )
                quantizer = quantizers["scaling_fwd"][fp8_meta_index]
                if quantizer is None:
                    raise RuntimeError(
                        f"{type(module).__name__}.{name} has no FP8 weight quantizer"
                    )
                quantizer.set_usage(rowwise=True, columnwise=torch.is_grad_enabled())
                quantizer.internal = False
                quantized = torch.nn.Parameter(
                    quantizer(parameter.detach()),
                    requires_grad=parameter.requires_grad,
                )
                _copy_param_attrs(parameter, quantized)
                setattr(module, name, quantized)
                converted += 1

    if converted == 0 and already_quantized == 0:
        raise RuntimeError(
            "fp8_param=True did not find any TE base weights to quantize"
        )
