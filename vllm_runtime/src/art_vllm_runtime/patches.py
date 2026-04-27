"""Monkey patches and bootstrap contract for the ART-owned vLLM runtime."""

import ctypes
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


def apply_vllm_runtime_patches() -> None:
    patch_transformers_v5_compat()
    subclass_chat_completion_request()
    patch_listen_for_disconnect()
    patch_tool_parser_manager()
    patch_nccl_unique_id_bootstrap()


def patch_transformers_v5_compat() -> None:
    _patch_rope_validation_ignore_keys()
    _patch_qwen3_vl_moe_tie_word_embeddings()
    _patch_qwen3_5_lora()


def _patch_rope_validation_ignore_keys() -> None:
    from transformers.configuration_utils import PretrainedConfig

    original = PretrainedConfig.convert_rope_params_to_dict
    if getattr(original, "__art_patched__", False):
        return

    def patched(self: Any, ignore_keys_at_rope_validation: Any = None, **kwargs: Any):
        if ignore_keys_at_rope_validation is not None:
            ignore_keys_at_rope_validation = set(ignore_keys_at_rope_validation)
        return original(
            self,
            ignore_keys_at_rope_validation=ignore_keys_at_rope_validation,
            **kwargs,
        )

    patched.__art_patched__ = True  # type: ignore[attr-defined]
    PretrainedConfig.convert_rope_params_to_dict = patched  # type: ignore[method-assign]


def _patch_qwen3_vl_moe_tie_word_embeddings() -> None:
    from transformers import Qwen3VLMoeTextConfig

    setattr(Qwen3VLMoeTextConfig, "tie_word_embeddings", False)


def _patch_qwen3_5_lora() -> None:
    from vllm.lora.layers.column_parallel_linear import (
        MergedColumnParallelLinearWithLoRA,
        MergedColumnParallelLinearWithShardedLoRA,
    )
    from vllm.lora.layers.utils import _not_fully_sharded_can_replace
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
    )

    projections = ["in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z"]
    Qwen3_5ForCausalLMBase.packed_modules_mapping["in_proj_qkvz"] = projections
    Qwen3_5ForConditionalGeneration.packed_modules_mapping["in_proj_qkvz"] = projections

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: Any,
        lora_config: Any,
        packed_modules_list: list[str],
        model_config: Any = None,
    ) -> bool:
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        return type(source_layer) is MergedColumnParallelLinear and len(
            packed_modules_list
        ) == len(source_layer.output_sizes)

    MergedColumnParallelLinearWithLoRA.can_replace_layer = can_replace_layer

    def slice_lora_a(
        self: Any,
        lora_a: "list[Tensor | None]",
    ) -> "list[Tensor | None]":
        output_shard_size = self.lora_a_stacked[0].shape[2]
        output_start_idx = self.tp_rank * output_shard_size
        return [
            a[output_start_idx : output_start_idx + output_shard_size, :]
            if a is not None
            else None
            for a in lora_a
        ]

    MergedColumnParallelLinearWithShardedLoRA.slice_lora_a = slice_lora_a  # ty:ignore[invalid-assignment]


def subclass_chat_completion_request() -> None:
    from vllm.entrypoints.openai.chat_completion import protocol

    if getattr(protocol, "_art_chat_completion_request_patched", False):
        return

    class ChatCompletionRequest(protocol.ChatCompletionRequest):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # ty:ignore[invalid-argument-type]
            self.logprobs = True
            if self.top_logprobs is None:
                self.top_logprobs = 0

    protocol.ChatCompletionRequest = ChatCompletionRequest  # ty:ignore[invalid-assignment]
    setattr(protocol, "_art_chat_completion_request_patched", True)


def patch_listen_for_disconnect() -> None:
    import vllm.entrypoints.utils

    if getattr(vllm.entrypoints.utils, "_art_listen_for_disconnect_patched", False):
        return

    async def patched_listen_for_disconnect(request: Any) -> None:
        try:
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    break
        except UnboundLocalError:
            pass

    vllm.entrypoints.utils.listen_for_disconnect = patched_listen_for_disconnect  # ty:ignore[invalid-assignment]
    setattr(vllm.entrypoints.utils, "_art_listen_for_disconnect_patched", True)


def patch_tool_parser_manager() -> None:
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    original = ToolParserManager.get_tool_parser
    if getattr(original, "__art_patched__", False):
        return

    def patched_get_tool_parser(name: str) -> type:
        tool_parser_class = original(name)
        current = tool_parser_class.extract_tool_calls_streaming
        if getattr(current, "__art_patched__", False):
            return tool_parser_class

        def patch(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return current(*args, **kwargs) or DeltaMessage()

        patch.__art_patched__ = True  # type: ignore[attr-defined]
        tool_parser_class.extract_tool_calls_streaming = patch  # ty:ignore[invalid-assignment]
        return tool_parser_class

    patched_get_tool_parser.__art_patched__ = True  # type: ignore[attr-defined]
    ToolParserManager.get_tool_parser = patched_get_tool_parser  # ty:ignore[invalid-assignment]


def _restore_nccl_unique_id_payload(
    payload: object,
    template: object | None,
) -> object:
    from vllm.distributed.device_communicators.pynccl_wrapper import ncclUniqueId

    if not isinstance(payload, (bytes, bytearray)) or not isinstance(
        template, ncclUniqueId
    ):
        return payload
    raw = bytes(payload)
    assert len(raw) == ctypes.sizeof(ncclUniqueId)
    unique_id = ncclUniqueId()
    ctypes.memmove(ctypes.byref(unique_id), raw, len(raw))
    return unique_id


def patch_nccl_unique_id_bootstrap() -> None:
    from vllm.distributed.utils import StatelessProcessGroup

    original = StatelessProcessGroup.broadcast_obj
    if getattr(original, "__art_patched__", False):
        return

    def patched(self: Any, obj: Any | None, src: int) -> Any:
        return _restore_nccl_unique_id_payload(original(self, obj, src), obj)

    patched.__art_patched__ = True  # type: ignore[attr-defined]
    StatelessProcessGroup.broadcast_obj = patched  # type: ignore[method-assign]
