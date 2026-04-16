from __future__ import annotations

import torch
import torch._dynamo.variables.streams  # noqa: F401

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

    from art.megatron.lora import MLPExpertsLinearFC1LoRA, MLPExpertsLinearFC2LoRA

    try:

        @torch.library.register_fake("streams::sync_dealloc")
        def _sync_dealloc_fake(
            wait_event_index: int,
            src_stream_index: int,
            to_dealloc: torch.Tensor,
        ) -> None:
            del wait_event_index, src_stream_index, to_dealloc
            return None
    except RuntimeError as exc:
        if "already has a fake impl registered" not in str(exc):
            raise

    moe_utils.permute = _disable(moe_utils.permute)
    moe_utils.unpermute = _disable(moe_utils.unpermute)
    moe_utils.sort_chunks_by_idxs = _disable(moe_utils.sort_chunks_by_idxs)
    moe_utils.maybe_move_tensor_to_cpu = _disable(moe_utils.maybe_move_tensor_to_cpu)
    token_dispatcher.permute = _disable(token_dispatcher.permute)
    token_dispatcher.unpermute = _disable(token_dispatcher.unpermute)
    token_dispatcher.sort_chunks_by_idxs = _disable(
        token_dispatcher.sort_chunks_by_idxs
    )
    token_dispatcher.MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize = _disable(
        token_dispatcher.MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize
    )
    MoELayer.preprocess = _disable(MoELayer.preprocess)
    MLPExpertsLinearFC1LoRA.forward = _disable(MLPExpertsLinearFC1LoRA.forward)
    MLPExpertsLinearFC2LoRA.forward = _disable(MLPExpertsLinearFC2LoRA.forward)
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
