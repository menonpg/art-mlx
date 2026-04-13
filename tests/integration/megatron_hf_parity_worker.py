from __future__ import annotations

import argparse
import faulthandler
import os
from pathlib import Path
import sys
import time
from typing import Any, cast

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer.utils import get_default_causal_mask
import torch
import torch.nn.functional as F

from art.megatron import train as megatron_train
from art.megatron.merged_weight_export import build_art_conversion_tasks
from art.megatron.provider import get_provider_bundle
from art.preprocessing.pack import packed_tensors_from_dir

from .megatron_hf_parity import (
    HF_PARITY_REPORT_FILENAME,
    HfParityRunRequest,
    build_hf_parity_report,
    build_parity_sample_indices,
    build_tensor_map_metric_rows,
    set_hf_config_num_layers,
    summarize_tensor_pair,
    zero_hf_dropout_config,
)
from .megatron_oracle_harness import ORACLE_TOPOLOGY, _read_json, _write_json
from .megatron_oracle_worker import (
    _assert_runtime_configuration,
    _build_optimizer_config,
    _configure_cuda_precision,
    _configure_provider,
    _set_deterministic_seed,
)
from .megatron_test_inputs import build_sft_trajectory_tensors_from_packed_tensors

HF_PARITY_DEBUG_ENV = "ART_HF_PARITY_DEBUG"
_DEBUG_START_TIME = time.perf_counter()
_VISUAL_HF_PREFIXES = ("model.visual.", "visual.")


def _debug(message: str) -> None:
    if os.environ.get(HF_PARITY_DEBUG_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    elapsed = time.perf_counter() - _DEBUG_START_TIME
    print(f"[hf_parity +{elapsed:8.2f}s] {message}", flush=True)


def _enable_debug_traceback_dump() -> None:
    if os.environ.get(HF_PARITY_DEBUG_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)


def _debug_enabled() -> bool:
    return os.environ.get(HF_PARITY_DEBUG_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _install_bridge_timing_debug(provider_bundle: Any) -> None:
    if not _debug_enabled():
        return
    provider = provider_bundle.provider
    pre_wrap_hooks = list(getattr(provider, "_pre_wrap_hooks", []))
    _debug(
        "registered pre-wrap hooks: "
        + ", ".join(
            getattr(hook, "__qualname__", repr(hook)) for hook in pre_wrap_hooks
        )
    )
    timed_hooks = []
    for index, hook in enumerate(pre_wrap_hooks):
        label = f"pre_wrap_hook[{index}]"

        def _timed_hook(
            model: list[Any], _hook: Any = hook, _label: str = label
        ) -> list[Any]:
            start = time.perf_counter()
            _debug(f"{_label}: start")
            try:
                return _hook(model)
            finally:
                _debug(f"{_label}: done in {time.perf_counter() - start:.2f}s")

        timed_hooks.append(_timed_hook)
    if pre_wrap_hooks:
        provider._pre_wrap_hooks = timed_hooks

    model_bridge = getattr(provider_bundle.bridge, "_model_bridge", None)
    if model_bridge is None:
        return
    if getattr(model_bridge, "_art_hf_parity_timing_wrapped", False):
        return
    original = model_bridge.load_weights_hf_to_megatron

    def _timed_load_weights(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        _debug("bridge.load_weights_hf_to_megatron: start")
        try:
            return original(*args, **kwargs)
        finally:
            _debug(
                "bridge.load_weights_hf_to_megatron: done in "
                f"{time.perf_counter() - start:.2f}s"
            )

    model_bridge.load_weights_hf_to_megatron = _timed_load_weights
    model_bridge._art_hf_parity_timing_wrapped = True


def _load_hf_model(
    *,
    base_model: str,
    num_layers: int,
    device: torch.device,
) -> Any:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    set_hf_config_num_layers(config, num_layers)
    zero_hf_dropout_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.train()
    return cast(Any, model).to(device)


def _collect_hf_grads(model: Any) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            grad = torch.zeros_like(param)
        grads[name] = grad.detach().cpu().to(dtype=torch.float32)
    return grads


def _collect_hf_params(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().to(dtype=torch.float32).clone()
        for name, param in model.named_parameters()
    }


def _tensor_map_deltas(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    before_keys = set(before.keys())
    after_keys = set(after.keys())
    if before_keys != after_keys:
        missing = sorted(before_keys - after_keys)
        extra = sorted(after_keys - before_keys)
        raise KeyError(
            f"Tensor-map keys changed across optimizer step: missing={missing[:3]} extra={extra[:3]}"
        )
    return {
        key: (after[key] - before[key]).detach().cpu().to(dtype=torch.float32)
        for key in sorted(before_keys)
    }


def _bridge_compatible_hf_key(key: str, expected_keys: set[str]) -> str:
    if key in expected_keys:
        return key
    if key.startswith("model."):
        prefixed = f"model.language_model.{key.removeprefix('model.')}"
        if prefixed in expected_keys:
            return prefixed
    if key.startswith("model.language_model."):
        stripped = f"model.{key.removeprefix('model.language_model.')}"
        if stripped in expected_keys:
            return stripped
    return key


def _normalize_hf_tensor_map_for_bridge(
    hf_map: dict[str, torch.Tensor],
    expected_keys: set[str],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in hf_map.items():
        normalized_key = _bridge_compatible_hf_key(key, expected_keys)
        if normalized_key in normalized:
            raise RuntimeError(
                f"Duplicate normalized HF key '{normalized_key}' from '{key}'"
            )
        normalized[normalized_key] = value
    return normalized


def _run_hf_sft_step(
    *,
    base_model: str,
    num_layers: int,
    micro_inputs: list[dict[str, torch.Tensor]],
    optimizer_config: Any,
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    _debug("loading HF model")
    model = _load_hf_model(base_model=base_model, num_layers=num_layers, device=device)
    _debug("running HF forward/backward")
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(optimizer_config.lr),
        betas=(float(optimizer_config.adam_beta1), float(optimizer_config.adam_beta2)),
        eps=float(optimizer_config.adam_eps),
        weight_decay=float(optimizer_config.weight_decay),
    )
    loss_sum = torch.tensor(0.0, device=device)
    token_count = 0
    trainable_losses: list[torch.Tensor] = []
    total_token_count = max(
        sum(
            int(megatron_train._count_sft_trainable_tokens(micro))
            for micro in micro_inputs
        ),
        1,
    )
    for micro in micro_inputs:
        attention_mask = micro["attention_mask"].reshape(-1)
        actual_len = max(int(attention_mask.sum().item()), 1)
        input_ids = micro["input_ids"].reshape(-1)[:actual_len].unsqueeze(0).to(device)
        labels = micro["labels"].reshape(-1)[:actual_len].unsqueeze(0).to(device)
        hf_attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        logits = model(
            input_ids=input_ids,
            attention_mask=hf_attention_mask,
            use_cache=False,
        ).logits
        shifted_labels = megatron_train.shift_tensor(labels, -100)
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            shifted_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(shifted_labels.shape)
        mask = shifted_labels != -100
        masked_losses = per_token_loss[mask]
        trainable_losses.append(masked_losses.detach().cpu())
        loss_sum = loss_sum + masked_losses.sum()
        token_count += int(mask.sum().item())
        (masked_losses.sum() / total_token_count).backward()
    grads = _collect_hf_grads(model)
    params_before = _collect_hf_params(model)
    _clip_hf_grads_like_megatron(
        model,
        max_norm=float(optimizer_config.clip_grad),
    )
    optimizer.step()
    params_after = _collect_hf_params(model)
    deltas = _tensor_map_deltas(params_before, params_after)
    scalar_loss = (loss_sum / max(token_count, 1)).detach().cpu().reshape(1)
    output_vector = torch.cat(trainable_losses, dim=0).to(dtype=torch.float32)
    del optimizer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _debug("finished HF step")
    return output_vector, scalar_loss, grads, deltas


def _build_megatron_runtime(
    request: HfParityRunRequest,
) -> megatron_train.TrainingRuntime:
    _debug("building Megatron provider bundle")
    provider_bundle = get_provider_bundle(
        request.case_config.base_model,
        torch_dtype=torch.float32,
        runtime_profile="single_gpu_parity",
    )
    _debug("Megatron provider bundle built")
    _install_bridge_timing_debug(provider_bundle)
    provider = provider_bundle.provider
    _configure_provider(provider, ORACLE_TOPOLOGY, request.case_config)
    _debug("Megatron provider configured for oracle topology")
    model = cast(
        list[Any],
        provider.provide_distributed_model(
            ddp_config=DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                average_in_collective=False,
            ),
            data_parallel_random_init=False,
            mixed_precision_wrapper=None,
        ),
    )
    _debug("Megatron model instantiated")
    megatron_train._install_gpt_preprocess_hook(model)
    return megatron_train.TrainingRuntime(
        provider_bundle=provider_bundle,
        provider=provider,
        model=model,
        optimizer=megatron_train._build_optimizer(
            model, _build_optimizer_config(request.case_config)
        ),
        optimizer_config=_build_optimizer_config(request.case_config),
        rank=torch.distributed.get_rank(),  # ty: ignore[possibly-missing-attribute]
        world_size=torch.distributed.get_world_size(),  # ty: ignore[possibly-missing-attribute]
    )


def _megatron_task_tensor(
    task: Any,
    *,
    mode: str,
) -> torch.Tensor:
    param = cast(torch.nn.Parameter, task.param_weight)
    if mode == "grad":
        grad = param.grad
        if grad is None:
            grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = torch.zeros_like(param)
        if hasattr(grad, "_local_tensor"):
            grad = cast(torch.Tensor, grad._local_tensor)
        return cast(torch.Tensor, grad)
    if mode == "param":
        return param.detach()
    raise ValueError(f"Unsupported task-tensor mode: {mode}")


def _mapping_supports_derivative_parity(mapping: Any) -> bool:
    from megatron.bridge.models.conversion.param_mapping import (
        RMSNorm2ZeroCenteredRMSNormMapping,
    )

    return not isinstance(mapping, RMSNorm2ZeroCenteredRMSNormMapping)


def _is_language_hf_param_name(name: str) -> bool:
    return not name.startswith(_VISUAL_HF_PREFIXES)


def _language_hf_param_names(mapping: Any) -> list[str]:
    hf_param = mapping.hf_param
    if isinstance(hf_param, str):
        return [hf_param]
    if isinstance(hf_param, dict):
        return [value for value in hf_param.values() if isinstance(value, str)]
    return []


def _mapping_targets_language_only(mapping: Any) -> bool:
    names = _language_hf_param_names(mapping)
    if not names:
        return True
    return all(_is_language_hf_param_name(name) for name in names)


def _filter_language_only_tensor_map(
    tensor_map: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in tensor_map.items()
        if _is_language_hf_param_name(key)
    }


def _clip_hf_grads_like_megatron(model: Any, *, max_norm: float) -> float:
    params = [param for param in model.parameters() if param.grad is not None]
    if not params or max_norm <= 0:
        return 0.0
    total_norm_sq = torch.zeros((), device=params[0].grad.device, dtype=torch.float32)
    for param in params:
        grad = param.grad.detach().to(dtype=torch.float32)
        total_norm_sq += torch.sum(grad * grad)
    total_norm = float(torch.sqrt(total_norm_sq).item())
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff >= 1.0:
        return total_norm
    for param in params:
        param.grad.mul_(clip_coeff)
    return total_norm


def _convert_megatron_tasks_to_hf(
    runtime: megatron_train.TrainingRuntime,
    *,
    mode: str,
    tasks: list[Any] | None = None,
) -> dict[str, torch.Tensor]:
    if tasks is None:
        tasks = [
            task
            for task in build_art_conversion_tasks(
                bridge=runtime.bridge,
                model=runtime.model,
            )
            if isinstance(task.param_weight, torch.nn.Parameter)
        ]
    model_bridge = runtime.bridge._model_bridge
    hf_state_dict = runtime.bridge.hf_pretrained.state
    grouped_buffers: dict[str, dict[int, torch.Tensor]] = {}
    converted: dict[str, torch.Tensor] = {}
    for task in tasks:
        tensor = _megatron_task_tensor(task, mode=mode)
        converted_weights_dict = task.mapping.megatron_to_hf(
            tensor,
            task.megatron_module,
        )
        if getattr(task.mapping, "is_grouped_export", False):
            merged_result = model_bridge._accumulate_grouped_export(
                task,
                converted_weights_dict,
                runtime.model[0].config,
                grouped_buffers,
                hf_state_dict,
            )
            if merged_result is None:
                continue
            converted_weights_dict = merged_result
        else:
            converted_weights_dict = model_bridge.maybe_modify_converted_hf_weight(
                task,
                converted_weights_dict,
                hf_state_dict,
            )
        for hf_name, value in converted_weights_dict.items():
            if not _is_language_hf_param_name(hf_name):
                continue
            if hf_name in converted:
                raise RuntimeError(f"Duplicate converted HF key '{hf_name}' in {mode}")
            converted[hf_name] = value.detach().cpu().to(dtype=torch.float32)
    return converted


def _run_megatron_sft_step(
    *,
    request: HfParityRunRequest,
    micro_inputs: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    runtime = _build_megatron_runtime(request)
    _assert_runtime_configuration(runtime.model, request.case_config)
    assert runtime.optimizer is not None
    uses_standard_attention_path = (
        getattr(runtime.provider, "_art_runtime_profile", None) == "single_gpu_parity"
    )
    _debug("initializing Megatron optimizer state")
    megatron_train._eager_initialize_optimizer_state(runtime.optimizer)
    tasks = [
        task
        for task in build_art_conversion_tasks(
            bridge=runtime.bridge,
            model=runtime.model,
        )
        if isinstance(task.param_weight, torch.nn.Parameter)
    ]
    _debug(f"built {len(tasks)} Megatron conversion tasks")
    for chunk in runtime.model:
        if hasattr(chunk, "zero_grad_buffer"):
            chunk.zero_grad_buffer()  # ty: ignore[call-non-callable]
        for param in chunk.parameters():
            param.grad = None
    loss_sum = torch.tensor(0.0, device=device)
    token_count = 0
    trainable_losses: list[torch.Tensor] = []
    for micro in micro_inputs:
        input_ids, position_ids, shifted_labels, mask, seq_len = (
            megatron_train._prepare_sft_micro_inputs(micro, device)
        )
        attention_mask = megatron_train._placeholder_attention_mask(device)
        if uses_standard_attention_path:
            attention_mask = get_default_causal_mask(seq_len).view(
                1, 1, seq_len, seq_len
            )
            forward_kwargs = runtime.model_support_handler.get_forward_kwargs(
                runtime.model[0]
            )
        else:
            forward_kwargs = runtime.model_support_handler.get_forward_kwargs(
                runtime.model[0],
                attention_bias=megatron_train._causal_attention_state(seq_len, device),
            )
        per_token_loss = runtime.model[0](
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=shifted_labels,
            **forward_kwargs,
        )
        masked_losses = per_token_loss[mask]
        trainable_losses.append(masked_losses.detach().cpu())
        loss_sum = loss_sum + masked_losses.sum()
        token_count += int(mask.sum().item())
        masked_losses.sum().backward()
    _debug("finished Megatron forward/backward")
    num_tokens = megatron_train._local_trainable_sft_token_count_tensor(
        micro_inputs,
        device=device,
    )
    megatron_train._flush_param_grads_to_main_grads(runtime.model)
    megatron_train.finalize_model_grads_extended(
        megatron_train.as_megatron_api_chunks(runtime.model),
        num_tokens=num_tokens,
    )
    _debug("finalized Megatron grads")
    derivative_tasks = [
        task
        for task in tasks
        if _mapping_supports_derivative_parity(task.mapping)
        and _mapping_targets_language_only(task.mapping)
    ]
    _debug(f"retained {len(derivative_tasks)} derivative-safe conversion tasks")
    grads = _convert_megatron_tasks_to_hf(
        runtime,
        mode="grad",
        tasks=derivative_tasks,
    )
    _debug("exported Megatron grads")
    params_before = _convert_megatron_tasks_to_hf(
        runtime,
        mode="param",
        tasks=derivative_tasks,
    )
    _debug("exported Megatron params before step")
    megatron_train._optimizer_step(runtime.optimizer, request.case_config.learning_rate)
    _debug("completed Megatron optimizer step")
    params_after = _convert_megatron_tasks_to_hf(
        runtime,
        mode="param",
        tasks=derivative_tasks,
    )
    _debug("exported Megatron params after step")
    deltas = _tensor_map_deltas(params_before, params_after)
    scalar_loss = (loss_sum / max(token_count, 1)).detach().cpu().reshape(1)
    output_vector = torch.cat(trainable_losses, dim=0).to(dtype=torch.float32)
    _debug("finished Megatron step")
    return output_vector, scalar_loss, grads, deltas


def _normalize_hf_maps_for_bridge(
    hf_grads: dict[str, torch.Tensor],
    hf_deltas: dict[str, torch.Tensor],
    *,
    expected_grad_keys: set[str],
    expected_delta_keys: set[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    hf_grads = _filter_language_only_tensor_map(hf_grads)
    hf_deltas = _filter_language_only_tensor_map(hf_deltas)
    normalized_hf_grads = _normalize_hf_tensor_map_for_bridge(
        hf_grads,
        expected_grad_keys,
    )
    normalized_hf_deltas = _normalize_hf_tensor_map_for_bridge(
        hf_deltas,
        expected_delta_keys,
    )
    return (
        {
            key: normalized_hf_grads[key]
            for key in sorted(expected_grad_keys)
            if key in normalized_hf_grads
        },
        {
            key: normalized_hf_deltas[key]
            for key in sorted(expected_delta_keys)
            if key in normalized_hf_deltas
        },
    )


def _worker_run(request: HfParityRunRequest) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("HF parity requires at least one CUDA device")
    torch.cuda.set_device(0)
    _set_deterministic_seed(request.case_config.seed)
    _configure_cuda_precision(request.case_config)
    _enable_debug_traceback_dump()

    packed_tensors = packed_tensors_from_dir(
        **request.packed_tensors.model_dump(exclude_none=True)
    )
    trajectory_tensors = build_sft_trajectory_tensors_from_packed_tensors(
        packed_tensors
    )
    zero_template = megatron_train._zero_contribution_sft_inputs(trajectory_tensors[0])
    sample_indices = build_parity_sample_indices(
        num_sequences=len(trajectory_tensors),
        global_grad_accumulation_sequences=request.case_config.grad_accumulation_sequences,
    )
    micro_inputs = megatron_train.select_sft_micro_inputs(
        trajectory_tensors,
        sample_indices,
        zero_template,
    )
    device = torch.device("cuda", 0)
    try:
        optimizer_config = _build_optimizer_config(request.case_config)
        _debug("starting HF parity worker")
        hf_outputs, hf_loss, hf_grads, hf_deltas = _run_hf_sft_step(
            base_model=request.case_config.base_model,
            num_layers=request.case_config.num_layers,
            micro_inputs=micro_inputs,
            optimizer_config=optimizer_config,
            device=device,
        )
        megatron_outputs, megatron_loss, megatron_grads, megatron_deltas = (
            _run_megatron_sft_step(
                request=request,
                micro_inputs=micro_inputs,
                device=device,
            )
        )
        _debug("finished HF and Megatron steps, building report")
        normalized_hf_grads, normalized_hf_deltas = _normalize_hf_maps_for_bridge(
            hf_grads,
            hf_deltas,
            expected_grad_keys=set(megatron_grads.keys()),
            expected_delta_keys=set(megatron_deltas.keys()),
        )
        outputs_summary = summarize_tensor_pair(hf_outputs, megatron_outputs)
        loss_summary = summarize_tensor_pair(hf_loss, megatron_loss)
        grads_rows = build_tensor_map_metric_rows(
            phase="grads",
            reference=normalized_hf_grads,
            candidate=megatron_grads,
        )
        deltas_rows = build_tensor_map_metric_rows(
            phase="deltas",
            reference=normalized_hf_deltas,
            candidate=megatron_deltas,
        )
        report = build_hf_parity_report(
            request=request,
            outputs_summary=outputs_summary,
            loss_summary=loss_summary,
            grads_rows=grads_rows,
            deltas_rows=deltas_rows,
        )
        _write_json(
            Path(request.output_dir) / HF_PARITY_REPORT_FILENAME,
            report.model_dump(mode="json"),
        )
        _debug("wrote HF parity report")
    finally:
        if torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
            torch.distributed.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


def run_worker_cli(run_request_path: Path) -> None:
    request = HfParityRunRequest.model_validate(_read_json(run_request_path))
    _worker_run(request)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Megatron HF parity worker")
    parser.add_argument("--run-request", type=Path, required=True)
    return parser.parse_args(argv)


def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    run_worker_cli(args.run_request)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
