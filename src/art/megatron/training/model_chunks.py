from collections.abc import Sequence
from typing import Any, cast

from megatron.core.transformer.module import MegatronModule
import torch

ModelChunk = torch.nn.Module
ModelChunks = list[ModelChunk]


def unwrap_megatron_chunk(module: ModelChunk) -> MegatronModule:
    current: Any = module
    seen: set[int] = set()
    while True:
        if isinstance(current, MegatronModule):
            return current
        if id(current) in seen:
            break
        seen.add(id(current))
        for attr_name in ("_orig_mod", "module"):
            next_module = getattr(current, attr_name, None)
            if isinstance(next_module, torch.nn.Module):
                current = next_module
                break
        else:
            break
    raise TypeError(
        f"Expected model chunk backed by MegatronModule, got {type(module).__name__}"
    )


def validate_model_chunks(model_chunks: Sequence[ModelChunk]) -> None:
    for chunk in model_chunks:
        try:
            unwrap_megatron_chunk(chunk)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc


def as_megatron_api_chunks(model_chunks: Sequence[ModelChunk]) -> list[MegatronModule]:
    validate_model_chunks(model_chunks)
    return cast(list[MegatronModule], list(model_chunks))
