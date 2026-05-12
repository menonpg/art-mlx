"""Monkey patches and bootstrap contract for the ART-owned vLLM runtime."""

import ctypes
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


def apply_vllm_runtime_patches() -> None:
    patch_transformers_v5_compat()
    patch_punica_ep_moe_lora_alignment()
    patch_lora_duplicate_module_aliases()
    patch_fused_moe_ep_lora_support()
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
    if lora_tensor is None:
        return lora_tensor
    global_indices = _ep_local_expert_global_indices(expert_map)
    assert global_indices.numel() == local_num_experts, (
        f"Expected {local_num_experts} EP-local experts, found "
        f"{global_indices.numel()} in expert_map"
    )
    return lora_tensor.index_select(0, global_indices.to(lora_tensor.device))


def _ep_moe_lora_expert_count(
    *,
    flat_rank_dim: int,
    lora_rank: int,
    expert_map: "Tensor",
    local_num_experts: int,
) -> int:
    """Return the expert axis for vLLM's two EP MoE LoRA input formats."""
    num_global_experts = int(expert_map.numel())
    if flat_rank_dim == lora_rank:
        assert flat_rank_dim % local_num_experts == 0, (
            "Expected vLLM EP-local dummy LoRA rank dimension to be divisible by "
            f"local_num_experts={local_num_experts}, got {flat_rank_dim}"
        )
        return local_num_experts
    assert flat_rank_dim == lora_rank * num_global_experts, (
        "Expected global vLLM MoE LoRA rank dimension to equal "
        f"rank * num_global_experts = {lora_rank} * {num_global_experts}, "
        f"got {flat_rank_dim}"
    )
    return num_global_experts


def _localize_ep_moe_lora_tensor(
    lora_tensor: "Tensor",
    *,
    num_experts: int,
    expert_map: "Tensor",
    local_num_experts: int,
) -> "Tensor":
    if num_experts == local_num_experts:
        return lora_tensor
    localized = _slice_ep_local_experts(lora_tensor, expert_map, local_num_experts)
    assert localized is not None
    return localized


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
        import torch

        (token_lora_mapping, _, _, _, lora_ids, _, _) = (
            self.token_mapping_meta.meta_args(
                num_tokens, self.lora_config.specialize_active_lora
            )
        )
        if expert_map is not None:
            expert_map = expert_map.to(topk_ids.device)
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
            max_num_m_blocks = punica_gpu.triton.cdiv(max_num_tokens_padded, block_size)
            expert_ids = torch.full(
                (max_loras * max_num_m_blocks,),
                -1,
                dtype=torch.int32,
                device=topk_ids.device,
            )
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
                expert_map,
            )

        return None, sorted_ids, expert_ids, num_tokens_post_pad

    patched_moe_lora_align_block_size.__art_patched__ = True  # type: ignore[attr-defined]
    punica_gpu.PunicaWrapperGPU.moe_lora_align_block_size = (
        patched_moe_lora_align_block_size  # type: ignore[method-assign]
    )


def patch_lora_duplicate_module_aliases() -> None:
    from vllm.lora import model_manager

    manager_cls = model_manager.LoRAModelManager
    if getattr(manager_cls, "__art_lora_duplicate_alias_patch__", False):
        return

    def _parent_module(module_name: str) -> str:
        return module_name.rpartition(".")[0]

    def _refresh_shared_expert_gate_alias(
        self: Any,
        module_name: str,
        old_module: Any,
        new_module: Any,
    ) -> None:
        if not module_name.endswith(".shared_expert_gate"):
            return
        parent_module = self.model.get_submodule(_parent_module(module_name))
        shared_expert = getattr(parent_module, "shared_expert", None)
        if shared_expert is None:
            return
        if getattr(shared_expert, "expert_gate", None) is old_module:
            shared_expert.expert_gate = new_module

    def patched_create_lora_modules(self: Any) -> None:
        seen_modules: set[Any] = set()
        for module_name, module in self.model.named_modules(remove_duplicate=False):
            if module in seen_modules:
                continue
            seen_modules.add(module)

            if isinstance(module, model_manager.PPMissingLayer):
                continue

            if not self._match_target_modules(module_name):
                continue

            punica_wrapper = self._get_punica_wrapper(module_name)
            if punica_wrapper is None:
                model_manager.logger.warning(
                    "Regarding %s, no matching PunicaWrapper "
                    "is found; %s will be ignored.",
                    self.model.__class__.__name__,
                    module_name,
                )
                continue

            if self._is_non_gated_moe and module_name.endswith("mixer.gate"):
                model_manager.logger.debug_once(
                    "LoRA is not supported for non-gated MoE gate module."
                    " %s will be ignored.",
                    module_name,
                    scope="local",
                )
                continue

            parts = module_name.split(".")[-1]
            packed_moduled_lst = self.packed_modules_mapping.get(parts, [])
            if isinstance(module, model_manager.FusedMoE):
                packed_moduled_lst = ["w13"] if self._is_3d_moe_model else ["w1", "w3"]
            new_module = model_manager.replace_submodule(
                self.model,
                module_name,
                model_manager.from_layer(
                    module,
                    self.lora_slots,
                    self.lora_config,
                    packed_moduled_lst,
                    self.model.config,
                ),
            )
            seen_modules.add(new_module)
            _refresh_shared_expert_gate_alias(self, module_name, module, new_module)

            if "lm_head" in module_name:
                logits_processor_module_name = "logits_processor"
                parent_module = _parent_module(module_name)
                if parent_module:
                    logits_processor_module_name = (
                        f"{parent_module}.{logits_processor_module_name}"
                    )

                logits_processor_module = self.model.get_submodule(
                    logits_processor_module_name
                )

                new_module = model_manager.replace_submodule(
                    self.model,
                    logits_processor_module_name,
                    model_manager.from_layer_logits_processor(
                        logits_processor_module,
                        module,
                        self.lora_slots,
                        self.lora_config,
                        self.model.config,
                    ),
                )
                seen_modules.add(new_module)

            if self.supports_mm and not isinstance(
                new_module, model_manager.BaseLayerWithLoRA
            ):
                continue
            self.register_module(module_name, new_module)

            self._register_packed_modules(module_name)
            new_module.set_mapping(punica_wrapper)

    def patched_activate_adapter(self: Any, lora_id: int) -> bool:
        if lora_id in self._active_adapters:
            return False
        first_free_slot = next(
            (
                (i, active_lora_id)
                for i, active_lora_id in enumerate(self.lora_index_to_id)
                if active_lora_id is None
            ),
            None,
        )
        if first_free_slot is None:
            raise ValueError("No free lora slots")
        index, _ = first_free_slot
        self._active_adapters[lora_id] = None
        lora_model = self._registered_adapters[lora_id]
        model_manager.logger.debug(
            "Activating LoRA. int id: %d, slot index: %d", lora_model.id, index
        )
        self.lora_index_to_id[index] = lora_model.id

        module_aliases: dict[Any, list[str]] = {}
        for module_name, module in self.modules.items():
            module_aliases.setdefault(module, []).append(module_name)

        for module, aliases in module_aliases.items():
            matches = []
            for module_name in aliases:
                module_lora = self._get_lora_layer_weights(lora_model, module_name)
                if module_lora is not None:
                    matches.append((module_name, module_lora))
            if not matches:
                module.reset_lora(index)
                model_manager.logger.debug(
                    "No LoRA weights found for module %s, skipping.", aliases[0]
                )
                continue
            if len({id(module_lora) for _, module_lora in matches}) > 1:
                raise RuntimeError(
                    "Multiple LoRA weight entries matched aliases for the same "
                    f"live module: {[module_name for module_name, _ in matches]}"
                )

            module_name, module_lora = matches[0]
            module.set_lora(
                index,
                module_lora.lora_a,
                module_lora.lora_b,
            )
            model_manager.logger.debug(
                "Successfully loaded LoRA weights for module %s.", module_name
            )
        return True

    patched_create_lora_modules.__art_patched__ = True  # type: ignore[attr-defined]
    patched_activate_adapter.__art_patched__ = True  # type: ignore[attr-defined]
    manager_cls._create_lora_modules = (  # type: ignore[method-assign]
        patched_create_lora_modules
    )
    manager_cls.activate_adapter = patched_activate_adapter  # type: ignore[method-assign]
    setattr(manager_cls, "__art_lora_duplicate_alias_patch__", True)


def patch_fused_moe_ep_lora_support() -> None:
    import torch
    from vllm.lora import model_manager
    from vllm.lora.layers import base, fused_moe

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

    original_stack = model_manager.LoRAModelManager._stack_moe_lora_weights
    if not getattr(original_stack, "__art_patched__", False):

        def patched_stack_moe_lora_weights(
            self: Any,
            lora_model: Any,
            module: Any,
            module_name: str,
        ) -> None:
            if not isinstance(module, fused_moe.FusedMoE3DWithLoRA):
                return original_stack(self, lora_model, module, module_name)
            if not module.base_layer.use_ep:
                return original_stack(self, lora_model, module, module_name)
            module_lora = self._get_lora_layer_weights(lora_model, module_name)
            if not module_lora:
                return
            if not torch.is_tensor(module_lora.lora_a):
                return
            gate_up_lora = self._get_lora_layer_weights(
                lora_model,
                module_name + ".base_layer",
            )
            assert gate_up_lora is not None
            expert_map = module.base_layer._expert_map
            local_num_experts = int(module.base_layer.local_num_experts)
            num_experts = _ep_moe_lora_expert_count(
                flat_rank_dim=int(gate_up_lora.lora_a.shape[0]),
                lora_rank=int(gate_up_lora.rank),
                expert_map=expert_map,
                local_num_experts=local_num_experts,
            )

            def stack_a(tensor: "Tensor") -> "Tensor":
                return tensor.reshape(num_experts, -1, tensor.shape[-1])

            def stack_b(tensor: "Tensor") -> "Tensor":
                return (
                    tensor.reshape(tensor.shape[0], -1, num_experts)
                    .permute(
                        2,
                        0,
                        1,
                    )
                    .contiguous()
                )

            module_lora.lora_a = [
                _localize_ep_moe_lora_tensor(
                    stack_a(gate_up_lora.lora_a),
                    num_experts=num_experts,
                    expert_map=expert_map,
                    local_num_experts=local_num_experts,
                ),
                _localize_ep_moe_lora_tensor(
                    stack_a(module_lora.lora_a),
                    num_experts=num_experts,
                    expert_map=expert_map,
                    local_num_experts=local_num_experts,
                ),
            ]
            module_lora.lora_b = [
                _localize_ep_moe_lora_tensor(
                    stack_b(gate_up_lora.lora_b),
                    num_experts=num_experts,
                    expert_map=expert_map,
                    local_num_experts=local_num_experts,
                ),
                _localize_ep_moe_lora_tensor(
                    stack_b(module_lora.lora_b),
                    num_experts=num_experts,
                    expert_map=expert_map,
                    local_num_experts=local_num_experts,
                ),
            ]

        patched_stack_moe_lora_weights.__art_patched__ = True  # type: ignore[attr-defined]
        model_manager.LoRAModelManager._stack_moe_lora_weights = (
            patched_stack_moe_lora_weights  # type: ignore[method-assign]
        )


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
