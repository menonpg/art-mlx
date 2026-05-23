from __future__ import annotations

import os
from typing import Any, cast

from megatron.core.transformer.transformer_layer import TransformerLayer
import torch

from art.megatron.compile_workarounds import install_torch_compile_workarounds
from art.megatron.provider_common import ProviderBundle
from art.megatron.training.model_chunks import ModelChunks


def _frozen_linear_grad_input(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if grad_output.dim() <= 2 or weight.dim() != 2:
        return grad_output.matmul(weight)
    grad_output_2d = grad_output.reshape(-1, int(grad_output.shape[-1]))
    grad_input_2d = grad_output_2d.matmul(weight)
    return grad_input_2d.reshape(*grad_output.shape[:-1], int(weight.shape[-1]))


def install_fast_frozen_output_backward() -> None:
    from megatron.core.tensor_parallel.layers import LinearWithFrozenWeight

    if getattr(LinearWithFrozenWeight.backward, "__art_fast_output_backward__", False):
        return

    def _fast_backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        (weight,) = ctx.saved_tensors
        grad_input = _frozen_linear_grad_input(grad_output, weight)
        if ctx.allreduce_dgrad:
            torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
                grad_input,
                group=ctx.tp_group,
            )
        return grad_input, None, None, None, None

    setattr(_fast_backward, "__art_fast_output_backward__", True)
    LinearWithFrozenWeight.backward = staticmethod(_fast_backward)


def compile_enabled() -> bool:
    return os.environ.get("ART_DISABLE_MEGATRON_COMPILE", "0") in {
        "0",
        "false",
        "False",
    }


def _set_child_module(
    parent: torch.nn.Module,
    name: str,
    child: torch.nn.Module,
) -> None:
    if isinstance(parent, torch.nn.ModuleList | torch.nn.Sequential):
        parent[int(name)] = child
        return
    setattr(parent, name, child)


def _compile_transformer_layers(module: torch.nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, TransformerLayer):
            physical_forward = getattr(child, "_art_gdn_island_physical_forward", None)
            if callable(physical_forward):
                child._art_gdn_island_physical_forward = torch.compile(physical_forward)
                continue
            compiled_child = cast(torch.nn.Module, torch.compile(child))
            _set_child_module(parent=module, name=name, child=compiled_child)
            continue
        _compile_transformer_layers(child)


def configure_training_compile(
    *,
    model: ModelChunks,
    provider: Any,
    provider_bundle: ProviderBundle,
) -> bool:
    compile_workaround_config = provider_bundle.handler.compile_workaround_config(
        provider
    )
    enabled = compile_enabled()
    flags = (
        compile_workaround_config.flags
        if enabled and not compile_workaround_config.disable_compile
        else compile_workaround_config.unconditional_flags
    )
    if flags:
        install_torch_compile_workarounds(
            compile_workaround_config.model_copy(update={"flags": flags})
        )
    transformer_layers_compiled = (
        enabled and not compile_workaround_config.disable_compile
    )
    if transformer_layers_compiled:
        for chunk in model:
            _compile_transformer_layers(chunk)
    return transformer_layers_compiled
