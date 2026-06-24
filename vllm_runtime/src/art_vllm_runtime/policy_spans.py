"""Policy-token span tracking for ART's owned vLLM runtime.

The hot path intentionally uses plain dict/list payloads. ART validates them
with Pydantic after the OpenAI response crosses back into the training process.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
import sys
from typing import Any

import msgspec
import numpy as np
import torch

POLICY_TOKEN_SPANS_FIELD = "policy_token_spans"
ART_POLICY_TOKEN_SPANS_FIELD = "art_policy_token_spans"

_CURRENT_ENGINE_POLICY_SPANS: dict[str, list[dict[str, Any]]] = {}
_COMPLETION_POLICY_SPANS_BY_REQUEST: dict[str, dict[int, list[dict[str, Any]]]] = {}
_WORKER_LORA_POLICY_BY_ID: dict[int, dict[str, Any]] = {}
_WORKER_LORA_UPDATE_SEQ = 0


def patch_policy_token_spans() -> None:
    _patch_engine_core_output_type()
    _patch_worker_policy_span_capture()
    _patch_scheduler_policy_span_transport()
    _patch_output_processor_policy_span_accumulation()
    _patch_openai_response_policy_spans()
    _patch_lora_alias_resolution()
    _patch_load_inplace_storage()


def register_lora_alias(
    models: Any,
    *,
    public_model_name: str,
    lora_slot: str,
    policy_version: int | None,
) -> None:
    aliases = getattr(models, "_art_lora_aliases", None)
    if aliases is None:
        aliases = {}
        setattr(models, "_art_lora_aliases", aliases)
    aliases[public_model_name] = lora_slot

    versions = getattr(models, "_art_lora_alias_policy_versions", None)
    if versions is None:
        versions = {}
        setattr(models, "_art_lora_alias_policy_versions", versions)
    if policy_version is not None:
        versions[public_model_name] = policy_version


def _resolve_lora_alias(models: Any, model_name: str | None) -> Any | None:
    if not model_name:
        return None
    slot = getattr(models, "_art_lora_aliases", {}).get(model_name)
    if not slot:
        return None
    return models.lora_requests.get(slot)


def _set_pydantic_extra(model: Any, key: str, value: Any) -> None:
    extra = getattr(model, "model_extra", None)
    if isinstance(extra, dict):
        extra[key] = value
        return
    setattr(model, key, value)


def _patch_engine_core_output_type() -> None:
    import vllm.v1.engine as engine_mod
    from vllm.v1.metrics.stats import PrefillStats, SchedulerStats
    from vllm.v1.outputs import LogprobsLists, LogprobsTensors
    from vllm.v1.serial_utils import UtilityResult

    if getattr(engine_mod, "_art_policy_token_spans_patched", False):
        return

    FinishReason = engine_mod.FinishReason
    EngineCoreEvent = engine_mod.EngineCoreEvent

    class EngineCoreOutput(  # type: ignore[call-arg]
        msgspec.Struct,
        array_like=True,
        omit_defaults=True,
        gc=False,
    ):
        request_id: str
        new_token_ids: list[int]

        new_logprobs: LogprobsLists | None = None
        new_prompt_logprobs_tensors: LogprobsTensors | None = None

        pooling_output: torch.Tensor | None = None

        finish_reason: FinishReason | None = None
        stop_reason: int | str | None = None
        events: list[EngineCoreEvent] | None = None
        kv_transfer_params: dict[str, Any] | None = None

        trace_headers: Mapping[str, str] | None = None

        prefill_stats: PrefillStats | None = None

        routed_experts: np.ndarray | None = None
        num_nans_in_logits: int = 0
        art_policy_token_spans: list[dict[str, Any]] | None = None

        @property
        def finished(self) -> bool:
            return self.finish_reason is not None

    class UtilityOutput(  # type: ignore[call-arg]
        msgspec.Struct,
        array_like=True,
        gc=False,
    ):
        call_id: int
        failure_message: str | None = None
        result: UtilityResult | None = None

    class EngineCoreOutputs(  # type: ignore[call-arg]
        msgspec.Struct,
        array_like=True,
        omit_defaults=True,
        gc=False,
    ):
        engine_index: int = 0
        outputs: list[EngineCoreOutput] = []
        scheduler_stats: SchedulerStats | None = None
        timestamp: float = 0.0
        utility_output: UtilityOutput | None = None
        finished_requests: set[str] | None = None
        wave_complete: int | None = None
        start_wave: int | None = None

    EngineCoreOutput.__module__ = engine_mod.__name__
    UtilityOutput.__module__ = engine_mod.__name__
    EngineCoreOutputs.__module__ = engine_mod.__name__
    engine_mod.EngineCoreOutput = EngineCoreOutput
    engine_mod.UtilityOutput = UtilityOutput
    engine_mod.EngineCoreOutputs = EngineCoreOutputs
    for module_name in (
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.engine.core",
        "vllm.v1.engine.output_processor",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "EngineCoreOutput", EngineCoreOutput)
            setattr(module, "EngineCoreOutputs", EngineCoreOutputs)
            if hasattr(module, "UtilityOutput"):
                setattr(module, "UtilityOutput", UtilityOutput)
    setattr(engine_mod, "_art_policy_token_spans_patched", True)


def _patch_worker_policy_span_capture() -> None:
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    original_add_adapter = LRUCacheWorkerLoRAManager.add_adapter
    if not getattr(original_add_adapter, "__art_policy_spans_patched__", False):

        def add_adapter(self: Any, lora_request: Any) -> bool:
            already_loaded = lora_request.lora_int_id in self.list_adapters()
            loaded = original_add_adapter(self, lora_request)
            if lora_request.load_inplace or not already_loaded:
                _record_worker_lora_policy(lora_request)
            return loaded

        add_adapter.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
        LRUCacheWorkerLoRAManager.add_adapter = add_adapter  # type: ignore[method-assign]

    original_sample_tokens = GPUModelRunner.sample_tokens
    if not getattr(original_sample_tokens, "__art_policy_spans_patched__", False):

        def sample_tokens(self: Any, *args: Any, **kwargs: Any) -> Any:
            context = _policy_context_from_runner(self)
            output = original_sample_tokens(self, *args, **kwargs)
            if context and output is not None:
                if isinstance(output, AsyncOutput):
                    setattr(output, "_art_policy_span_context", context)
                else:
                    _attach_policy_spans_to_model_output(output, context)
            return output

        sample_tokens.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
        GPUModelRunner.sample_tokens = sample_tokens  # type: ignore[method-assign]

    original_get_output = AsyncOutput.get_output
    if getattr(original_get_output, "__art_policy_spans_patched__", False):
        return

    def get_output(self: Any) -> Any:
        output = original_get_output(self)
        context = getattr(self, "_art_policy_span_context", None)
        if context:
            _attach_policy_spans_to_model_output(output, context)
        return output

    get_output.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
    AsyncOutput.get_output = get_output  # type: ignore[method-assign]


def _patch_scheduler_policy_span_transport() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    original_update = Scheduler.update_from_output
    if getattr(original_update, "__art_policy_spans_patched__", False):
        return

    def update_from_output(self: Any, scheduler_output: Any, model_runner_output: Any):
        outputs_by_client = original_update(self, scheduler_output, model_runner_output)
        spans_by_req = getattr(model_runner_output, ART_POLICY_TOKEN_SPANS_FIELD, None)
        if not spans_by_req:
            return outputs_by_client
        for client_outputs in outputs_by_client.values():
            for output in client_outputs.outputs:
                spans = spans_by_req.get(output.request_id)
                if not spans:
                    continue
                output.art_policy_token_spans = _trim_step_spans(
                    spans, len(output.new_token_ids)
                )
        return outputs_by_client

    update_from_output.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
    Scheduler.update_from_output = update_from_output  # type: ignore[method-assign]


def _patch_output_processor_policy_span_accumulation() -> None:
    from vllm.v1.engine.output_processor import OutputProcessor, RequestState

    original_process_outputs = OutputProcessor.process_outputs
    if not getattr(original_process_outputs, "__art_policy_spans_patched__", False):

        def process_outputs(
            self: Any, engine_core_outputs: list[Any], *args: Any, **kwargs: Any
        ):
            global _CURRENT_ENGINE_POLICY_SPANS
            previous = _CURRENT_ENGINE_POLICY_SPANS
            _CURRENT_ENGINE_POLICY_SPANS = {
                output.request_id: output.art_policy_token_spans
                for output in engine_core_outputs
                if getattr(output, "art_policy_token_spans", None)
            }
            try:
                return original_process_outputs(
                    self, engine_core_outputs, *args, **kwargs
                )
            finally:
                _CURRENT_ENGINE_POLICY_SPANS = previous

        process_outputs.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
        OutputProcessor.process_outputs = process_outputs  # type: ignore[method-assign]

    original_make = RequestState.make_request_output
    if getattr(original_make, "__art_policy_spans_patched__", False):
        return

    def make_request_output(
        self: Any,
        new_token_ids: list[int],
        *args: Any,
        **kwargs: Any,
    ):
        _append_current_policy_spans(self, len(new_token_ids))
        request_output = original_make(self, new_token_ids, *args, **kwargs)
        if request_output is not None and hasattr(request_output, "outputs"):
            spans = getattr(self, ART_POLICY_TOKEN_SPANS_FIELD, None)
            if spans:
                _record_request_output_spans(
                    request_output,
                    request_index=getattr(self, "request_index", 0),
                    spans=spans,
                )
        return request_output

    make_request_output.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
    RequestState.make_request_output = make_request_output  # type: ignore[method-assign]


def _patch_openai_response_policy_spans() -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    original_full = OpenAIServingChat.chat_completion_full_generator
    if getattr(original_full, "__art_policy_spans_patched__", False):
        return

    async def chat_completion_full_generator(
        self: Any,
        request: Any,
        result_generator: Any,
        request_id: str,
        model_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        response = await original_full(
            self,
            request,
            result_generator,
            request_id,
            model_name,
            *args,
            **kwargs,
        )
        if isinstance(response, ChatCompletionResponse):
            spans_by_choice = _COMPLETION_POLICY_SPANS_BY_REQUEST.pop(request_id, {})
            for choice in response.choices:
                spans = spans_by_choice.get(choice.index)
                if spans:
                    _set_pydantic_extra(choice, POLICY_TOKEN_SPANS_FIELD, spans)
            if _resolve_lora_alias(self.models, getattr(request, "model", None)):
                response.model = request.model
        return response

    chat_completion_full_generator.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
    OpenAIServingChat.chat_completion_full_generator = chat_completion_full_generator  # type: ignore[method-assign]


def _patch_lora_alias_resolution() -> None:
    from vllm.entrypoints.openai.engine.serving import OpenAIServing

    original_check = OpenAIServing._check_model
    if not getattr(original_check, "__art_policy_spans_patched__", False):

        async def _check_model(self: Any, request: Any) -> Any:
            if _resolve_lora_alias(self.models, getattr(request, "model", None)):
                return None
            return await original_check(self, request)

        _check_model.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
        OpenAIServing._check_model = _check_model  # type: ignore[method-assign]

    original_maybe = OpenAIServing._maybe_get_adapters
    if getattr(original_maybe, "__art_policy_spans_patched__", False):
        return

    def _maybe_get_adapters(
        self: Any,
        request: Any,
        supports_default_mm_loras: bool = False,
    ) -> Any:
        lora_request = _resolve_lora_alias(self.models, getattr(request, "model", None))
        if lora_request is not None:
            return lora_request
        return original_maybe(
            self,
            request,
            supports_default_mm_loras=supports_default_mm_loras,
        )

    _maybe_get_adapters.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
    OpenAIServing._maybe_get_adapters = _maybe_get_adapters  # type: ignore[method-assign]


def _patch_load_inplace_storage() -> None:
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.lora.request import LoRARequest

    original = OpenAIServingModels.load_lora_adapter
    if getattr(original, "__art_policy_spans_patched__", False):
        return

    async def load_lora_adapter(
        self: Any,
        request: Any,
        base_model_name: str | None = None,
    ) -> Any:
        result = await original(self, request, base_model_name=base_model_name)
        lora_request = self.lora_requests.get(request.lora_name)
        if lora_request is not None and lora_request.load_inplace:
            normalized = LoRARequest(
                lora_name=lora_request.lora_name,
                lora_int_id=lora_request.lora_int_id,
                lora_path=lora_request.lora_path,
                base_model_name=lora_request.base_model_name,
                tensorizer_config_dict=lora_request.tensorizer_config_dict,
                load_inplace=False,
                is_3d_lora_weight=lora_request.is_3d_lora_weight,
            )
            self.lora_requests[request.lora_name] = normalized
        return result

    load_lora_adapter.__art_policy_spans_patched__ = True  # type: ignore[attr-defined]
    OpenAIServingModels.load_lora_adapter = load_lora_adapter  # type: ignore[method-assign]


def _policy_context_from_runner(runner: Any) -> dict[str, dict[str, Any]]:
    state = getattr(runner, "execute_model_state", None)
    if state is None:
        return {}
    input_batch = state.input_batch
    lora_state = getattr(runner, "lora_state", None)
    context: dict[str, dict[str, Any]] = {}
    for req_id in input_batch.req_ids:
        lora_request = None
        if lora_state is not None:
            lora_request = lora_state.lora_requests.get(req_id)
        context[req_id] = _policy_metadata_for_lora_request(lora_request)
    return context


def _policy_metadata_for_lora_request(lora_request: Any | None) -> dict[str, Any]:
    if lora_request is None:
        return {"policy_version": 0, "lora_slot": "base", "update_seq": 0}
    state = _WORKER_LORA_POLICY_BY_ID.get(lora_request.lora_int_id)
    if state is None:
        state = _record_worker_lora_policy(lora_request)
    return state


def _record_worker_lora_policy(lora_request: Any) -> dict[str, Any]:
    global _WORKER_LORA_UPDATE_SEQ
    _WORKER_LORA_UPDATE_SEQ += 1
    policy_version = _policy_version_from_lora_request(lora_request)
    state = {
        "policy_version": int(policy_version or 0),
        "lora_slot": str(lora_request.lora_name),
        "update_seq": _WORKER_LORA_UPDATE_SEQ,
    }
    _WORKER_LORA_POLICY_BY_ID[int(lora_request.lora_int_id)] = state
    return state


def _policy_version_from_lora_request(lora_request: Any) -> int | None:
    for pattern, value in (
        (r"@(\d+)$", getattr(lora_request, "lora_name", "")),
        (
            r"^(?:step[_-]?)?(\d+)$",
            getattr(lora_request, "lora_path", "").rstrip("/").split("/")[-1],
        ),
    ):
        match = re.search(pattern, value)
        if match:
            return int(match.group(1))
    return None


def _attach_policy_spans_to_model_output(
    output: Any, context: dict[str, dict[str, Any]]
) -> None:
    spans_by_req: dict[str, list[dict[str, Any]]] = {}
    for req_id, token_ids in zip(output.req_ids, output.sampled_token_ids or ()):
        num_tokens = len(token_ids)
        if num_tokens <= 0:
            continue
        metadata = context.get(req_id)
        if not metadata:
            continue
        spans_by_req[req_id] = [
            {
                "start_token": 0,
                "end_token": num_tokens,
                "policy_version": metadata["policy_version"],
                "lora_slot": metadata["lora_slot"],
                "update_seq": metadata["update_seq"],
            }
        ]
    if spans_by_req:
        setattr(output, ART_POLICY_TOKEN_SPANS_FIELD, spans_by_req)


def _trim_step_spans(
    spans: list[dict[str, Any]], token_count: int
) -> list[dict[str, Any]]:
    if token_count <= 0:
        return []
    trimmed: list[dict[str, Any]] = []
    for span in spans:
        start = min(max(int(span["start_token"]), 0), token_count)
        end = min(max(int(span["end_token"]), start), token_count)
        if end <= start:
            continue
        current = {**span, "start_token": start, "end_token": end}
        if trimmed and _can_merge_spans(trimmed[-1], current):
            trimmed[-1]["end_token"] = end
        else:
            trimmed.append(current)
    return trimmed


def _append_current_policy_spans(req_state: Any, token_count: int) -> None:
    step_spans = _CURRENT_ENGINE_POLICY_SPANS.get(req_state.request_id)
    if not step_spans or token_count <= 0:
        return
    detokenizer = getattr(req_state, "detokenizer", None)
    output_tokens = detokenizer.num_output_tokens() if detokenizer is not None else 0
    offset = max(output_tokens - token_count, 0)
    accumulated = getattr(req_state, ART_POLICY_TOKEN_SPANS_FIELD, None)
    if accumulated is None:
        accumulated = []
        setattr(req_state, ART_POLICY_TOKEN_SPANS_FIELD, accumulated)
    for span in _trim_step_spans(step_spans, token_count):
        current = {
            **span,
            "start_token": offset + int(span["start_token"]),
            "end_token": offset + int(span["end_token"]),
        }
        if accumulated and _can_merge_spans(accumulated[-1], current):
            accumulated[-1]["end_token"] = current["end_token"]
        else:
            accumulated.append(current)


def _can_merge_spans(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("end_token") == right.get("start_token")
        and left.get("policy_version") == right.get("policy_version")
        and left.get("lora_slot") == right.get("lora_slot")
        and left.get("update_seq") == right.get("update_seq")
    )


def _record_request_output_spans(
    request_output: Any,
    *,
    request_index: int,
    spans: list[dict[str, Any]],
) -> None:
    by_choice = _COMPLETION_POLICY_SPANS_BY_REQUEST.setdefault(
        request_output.request_id, {}
    )
    for output in request_output.outputs:
        if getattr(output, "index", request_index) != request_index:
            continue
        copied = [dict(span) for span in spans]
        setattr(output, ART_POLICY_TOKEN_SPANS_FIELD, copied)
        by_choice[output.index] = copied
