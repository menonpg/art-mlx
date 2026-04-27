"""Monkey patches and bootstrap contract for the ART-owned vLLM runtime."""

import ctypes
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


def apply_vllm_runtime_patches() -> None:
    patch_transformers_v5_compat()
    patch_punica_ep_moe_lora_alignment()
    patch_fused_moe_ep_lora_support()
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


def _ep_local_expert_global_indices(expert_map: "Tensor") -> "Tensor":
    import torch

    local_mask = expert_map >= 0
    global_indices = torch.nonzero(local_mask, as_tuple=False).flatten()
    local_indices = expert_map.index_select(0, global_indices).to(torch.int64)
    return global_indices.index_select(0, torch.argsort(local_indices))


def _slice_ep_local_experts(
    lora_tensor: "Tensor | None",
    expert_map: "Tensor",
    local_num_experts: int,
) -> "Tensor | None":
    if lora_tensor is None or lora_tensor.shape[0] == local_num_experts:
        return lora_tensor
    global_indices = _ep_local_expert_global_indices(expert_map)
    assert global_indices.numel() == local_num_experts, (
        f"Expected {local_num_experts} EP-local experts, found "
        f"{global_indices.numel()} in expert_map"
    )
    return lora_tensor.index_select(0, global_indices.to(lora_tensor.device))


def patch_punica_ep_moe_lora_alignment() -> None:
    from vllm.lora.punica_wrapper import punica_gpu

    original = punica_gpu.PunicaWrapperGPU.moe_lora_align_block_size
    if getattr(original, "__art_patched__", False):
        return

    def patched_moe_lora_align_block_size(
        self: Any,
        topk_ids: Any,
        num_tokens: int,
        block_size: int,
        num_experts: int,
        max_loras: int,
        adapter_enabled: Any,
        expert_map: Any = None,
        pad_sorted_ids: bool = False,
        naive_block_assignment: bool = False,
    ) -> tuple[Any, Any, Any, Any]:
        (token_lora_mapping, _, _, _, lora_ids, _, _) = (
            self.token_mapping_meta.meta_args(
                num_tokens, self.lora_config.specialize_active_lora
            )
        )
        if expert_map is not None:
            expert_map = expert_map.to(topk_ids.device)
            num_experts = int(expert_map.shape[0])
            naive_block_assignment = False

        if naive_block_assignment:
            expert_ids = topk_ids.reshape(-1)
            sorted_ids = None
            num_tokens_post_pad = None
        else:
            max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
            if pad_sorted_ids:
                max_num_tokens_padded = punica_gpu.round_up(
                    max_num_tokens_padded, block_size
                )
            if topk_ids.numel() < num_experts:
                max_num_tokens_padded = topk_ids.numel() * block_size
            sorted_ids = topk_ids.new_empty((max_loras * max_num_tokens_padded,))
            max_num_m_blocks = punica_gpu.triton.cdiv(
                max_num_tokens_padded, block_size
            )
            expert_ids = topk_ids.new_empty((max_loras * max_num_m_blocks,))
            num_tokens_post_pad = topk_ids.new_empty((max_loras,))

            punica_gpu.ops.moe_lora_align_block_size(
                topk_ids,
                token_lora_mapping,
                num_experts,
                block_size,
                max_loras,
                max_num_tokens_padded,
                max_num_m_blocks,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                adapter_enabled,
                lora_ids,
            )
            if expert_map is not None:
                expert_ids = expert_map[expert_ids]

        return None, sorted_ids, expert_ids, num_tokens_post_pad

    patched_moe_lora_align_block_size.__art_patched__ = True  # type: ignore[attr-defined]
    punica_gpu.PunicaWrapperGPU.moe_lora_align_block_size = patched_moe_lora_align_block_size  # type: ignore[method-assign]


def patch_fused_moe_ep_lora_support() -> None:
    from vllm.lora.layers import base
    from vllm.lora.layers import fused_moe

    original_init = fused_moe.FusedMoEWithLoRA.__init__
    if not getattr(original_init, "__art_patched__", False):

        def patched_init(self: Any, base_layer: Any) -> None:
            base.BaseLayerWithLoRA.__init__(self)
            self.base_layer = base_layer
            self.tp_size = fused_moe.get_tensor_model_parallel_world_size()
            self.tp_rank = fused_moe.get_tensor_model_parallel_rank()
            self.device = fused_moe._get_lora_device(base_layer)
            self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1
            self._inject_lora_into_fused_moe()

        patched_init.__art_patched__ = True  # type: ignore[attr-defined]
        fused_moe.FusedMoEWithLoRA.__init__ = patched_init  # type: ignore[method-assign]

    def localize_loras(self: Any, loras: object) -> object:
        if not self.base_layer.use_ep:
            return loras
        expert_map = getattr(self.base_layer, "_expert_map", None)
        assert expert_map is not None, "Expected _expert_map when EP LoRA is enabled"
        assert isinstance(loras, list)
        return [
            _slice_ep_local_experts(lora, expert_map, self.base_layer.local_num_experts)
            for lora in loras
        ]

    original_set_lora = fused_moe.FusedMoEWithLoRA.set_lora
    if not getattr(original_set_lora, "__art_patched__", False):

        def patched_set_lora(
            self: Any,
            index: int,
            lora_a: object,
            lora_b: object,
        ) -> None:
            return original_set_lora(
                self,
                index,
                localize_loras(self, lora_a),
                localize_loras(self, lora_b),
            )

        patched_set_lora.__art_patched__ = True  # type: ignore[attr-defined]
        fused_moe.FusedMoEWithLoRA.set_lora = patched_set_lora  # type: ignore[method-assign]

    original_3d_set_lora = fused_moe.FusedMoE3DWithLoRA.set_lora
    if not getattr(original_3d_set_lora, "__art_patched__", False):

        def patched_3d_set_lora(
            self: Any,
            index: int,
            lora_a: object,
            lora_b: object,
        ) -> None:
            return original_3d_set_lora(
                self,
                index,
                localize_loras(self, lora_a),
                localize_loras(self, lora_b),
            )

        patched_3d_set_lora.__art_patched__ = True  # type: ignore[attr-defined]
        fused_moe.FusedMoE3DWithLoRA.set_lora = patched_3d_set_lora  # type: ignore[method-assign]


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
            return _restore_nccl_unique_id_payload(original_broadcast(self, obj, src), obj)

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
