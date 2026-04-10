from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, cast

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer.utils import get_default_causal_mask
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from art.megatron import train as megatron_train
from art.megatron.merged_weight_export import build_art_conversion_tasks
from art.megatron.provider import get_provider_bundle
from art.preprocessing.pack import packed_tensors_from_dir

from .megatron_hf_parity import (
    HF_PARITY_REPORT_FILENAME,
    HfParityRunRequest,
    build_hf_parity_report,
    build_parity_sample_indices,
    set_hf_config_num_layers,
    summarize_tensor_maps,
    summarize_tensor_pair,
    zero_hf_dropout_config,
)
from .megatron_oracle_harness import ORACLE_TOPOLOGY, _read_json, _write_json
from .megatron_oracle_worker import (
    _build_optimizer_config,
    _configure_cuda_precision,
    _configure_provider,
    _set_deterministic_seed,
)
from .megatron_test_inputs import build_sft_trajectory_tensors_from_packed_tensors

HF_PARITY_DEBUG_ENV = "ART_HF_PARITY_DEBUG"


def _debug(message: str) -> None:
    if os.environ.get(HF_PARITY_DEBUG_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    print(f"[hf_parity] {message}", flush=True)


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
    clip_grad = float(optimizer_config.clip_grad)
    if clip_grad > 0:
        clip_grad_norm_(model.parameters(), max_norm=clip_grad)
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


def _task_has_nonzero_grad(task: Any) -> bool:
    grad = _megatron_task_tensor(task, mode="grad")
    return bool(torch.count_nonzero(grad).item() > 0)


def _mapping_supports_derivative_parity(mapping: Any) -> bool:
    from megatron.bridge.models.conversion.param_mapping import (
        RMSNorm2ZeroCenteredRMSNormMapping,
    )

    return not isinstance(mapping, RMSNorm2ZeroCenteredRMSNormMapping)


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
    signal_tasks = [task for task in tasks if _task_has_nonzero_grad(task)]
    _debug(f"retained {len(signal_tasks)} non-zero-grad conversion tasks")
    derivative_tasks = [
        task
        for task in signal_tasks
        if _mapping_supports_derivative_parity(task.mapping)
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


def _filter_hf_maps(
    hf_grads: dict[str, torch.Tensor],
    hf_deltas: dict[str, torch.Tensor],
    expected_keys: set[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    normalized_hf_grads = _normalize_hf_tensor_map_for_bridge(hf_grads, expected_keys)
    normalized_hf_deltas = _normalize_hf_tensor_map_for_bridge(
        hf_deltas,
        expected_keys,
    )
    return (
        {
            key: normalized_hf_grads[key]
            for key in sorted(expected_keys)
            if key in normalized_hf_grads
        },
        {
            key: normalized_hf_deltas[key]
            for key in sorted(expected_keys)
            if key in normalized_hf_deltas
        },
    )


def _worker_run(request: HfParityRunRequest) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("HF parity requires at least one CUDA device")
    torch.cuda.set_device(0)
    _set_deterministic_seed(request.case_config.seed)
    _configure_cuda_precision(request.case_config)

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
        expected_keys = set(megatron_grads.keys()) | set(megatron_deltas.keys())
        filtered_hf_grads, filtered_hf_deltas = _filter_hf_maps(
            hf_grads,
            hf_deltas,
            expected_keys,
        )
        outputs_summary = summarize_tensor_pair(hf_outputs, megatron_outputs)
        loss_summary = summarize_tensor_pair(hf_loss, megatron_loss)
        grads_summary, grads_failure = summarize_tensor_maps(
            filtered_hf_grads,
            megatron_grads,
        )
        deltas_summary, deltas_failure = summarize_tensor_maps(
            filtered_hf_deltas,
            megatron_deltas,
        )
        report = build_hf_parity_report(
            request=request,
            outputs_summary=outputs_summary,
            loss_summary=loss_summary,
            grads_summary=grads_summary,
            deltas_summary=deltas_summary,
            grads_structural_failure=grads_failure,
            deltas_structural_failure=deltas_failure,
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
