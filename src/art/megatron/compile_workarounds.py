from __future__ import annotations

import torch

_INSTALLED = False


def _disable(fn):
    if getattr(fn, "__art_compile_disabled__", False):
        return fn
    wrapped = torch.compiler.disable(fn)
    setattr(wrapped, "__art_compile_disabled__", True)
    return wrapped


def install_torch_compile_workarounds() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from megatron.core.transformer.moe import moe_utils, token_dispatcher
    from megatron.core.transformer.moe.moe_layer import MoELayer

    moe_utils.maybe_move_tensor_to_cpu = _disable(moe_utils.maybe_move_tensor_to_cpu)
    token_dispatcher.MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize = _disable(
        token_dispatcher.MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize
    )
    MoELayer.preprocess = _disable(MoELayer.preprocess)
    deepep_manager = getattr(token_dispatcher, "_DeepepManager", None)
    if deepep_manager is not None:
        deepep_manager.dispatch = _disable(deepep_manager.dispatch)
        deepep_manager.combine = _disable(deepep_manager.combine)
        deepep_manager.get_permuted_hidden_states_by_experts = _disable(
            deepep_manager.get_permuted_hidden_states_by_experts
        )
        deepep_manager.get_restored_hidden_states_by_experts = _disable(
            deepep_manager.get_restored_hidden_states_by_experts
        )
    _INSTALLED = True
