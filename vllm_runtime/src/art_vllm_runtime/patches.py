"""Monkey patches and bootstrap contract for the ART-owned vLLM runtime."""

import ctypes
from typing import Any


def apply_vllm_runtime_patches() -> None:
    patch_transformers_v5_compat()
    subclass_chat_completion_request()
    patch_listen_for_disconnect()
    patch_tool_parser_manager()
    patch_nccl_unique_id_bootstrap()


def patch_transformers_v5_compat() -> None:
    _patch_rope_validation_ignore_keys()
    _patch_qwen3_vl_moe_tie_word_embeddings()


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


def _normalize_nccl_comm_init_rank_unique_id(library: Any, unique_id: object) -> object:
    if isinstance(unique_id, (bytes, bytearray)):
        return library.unique_id_from_bytes(bytes(unique_id))
    return unique_id


def patch_nccl_unique_id_bootstrap() -> None:
    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
    from vllm.distributed.utils import StatelessProcessGroup

    original_broadcast = StatelessProcessGroup.broadcast_obj
    if not getattr(original_broadcast, "__art_patched__", False):

        def patched_broadcast(self: Any, obj: Any | None, src: int) -> Any:
            return _restore_nccl_unique_id_payload(
                original_broadcast(self, obj, src), obj
            )

        patched_broadcast.__art_patched__ = True  # type: ignore[attr-defined]
        StatelessProcessGroup.broadcast_obj = patched_broadcast  # type: ignore[method-assign]

    original_comm_init_rank = NCCLLibrary.ncclCommInitRank
    if getattr(original_comm_init_rank, "__art_patched__", False):
        return

    def patched_comm_init_rank(
        self: Any,
        world_size: int,
        unique_id: object,
        rank: int,
    ) -> Any:
        unique_id = _normalize_nccl_comm_init_rank_unique_id(self, unique_id)
        return original_comm_init_rank(self, world_size, unique_id, rank)

    patched_comm_init_rank.__art_patched__ = True  # type: ignore[attr-defined]
    NCCLLibrary.ncclCommInitRank = patched_comm_init_rank  # type: ignore[method-assign]
