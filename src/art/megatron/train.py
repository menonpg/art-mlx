# isort: off
from art.megatron.runtime_env import configure_megatron_runtime_env

configure_megatron_runtime_env()
# isort: on

import gc
import importlib
import math
import os
import time
from typing import Any, Callable, cast

from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.module import MegatronModule
from pydantic import BaseModel, ConfigDict
import torch
from torch._inductor.runtime.cache_dir_utils import cache_dir as inductor_cache_dir

from art import dev, types
from art.loss import loss_fn, shift_tensor
from art.megatron.finalize_grads import finalize_model_grads_extended
from art.megatron.flex_attention import create_shared_prefix_attention_state
from art.megatron.jobs import (
    DEFAULT_VLLM_WAKE_LOCK_PATH,
)
from art.megatron.lora import apply_lora_adapters
from art.megatron.offload import (
    OffloadState,
    clear_optimizer_state,
    offload_to_cpu,
    reload_to_gpu,
)
from art.megatron.provider import get_provider
from art.megatron.routing_replay import (
    MoeRoutingReplayBundle,
    MoeRoutingReplayController,
)
from art.preprocessing.pack import (
    PackedTensors,
)

safetensors_torch = importlib.import_module("safetensors.torch")

DEFAULT_MODEL_IDENTIFIER = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class TrainingRuntime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Any
    model: list[MegatronModule]
    optimizer: Any
    rank: int
    world_size: int
    moe_routing_replay_controller: MoeRoutingReplayController | None = None


class TrainStepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    reduced_loss: torch.Tensor
    probs_corr: float
    new_logprobs: torch.Tensor
    update_successful: bool
    grad_norm: float
    num_zeros_in_grad: int | None


def print0(rank: int, *values: Any) -> None:
    if rank == 0:
        print(*values)


def freeze_model(model_chunks: list[MegatronModule]) -> list[MegatronModule]:
    for module in model_chunks:
        for param in module.parameters():
            param.requires_grad = False
    return model_chunks


def _frozen_linear_grad_input(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if grad_output.dim() <= 2 or weight.dim() != 2:
        return grad_output.matmul(weight)
    try:
        grad_output_2d = grad_output.view(-1, int(grad_output.shape[-1]))
    except RuntimeError:
        grad_output_2d = grad_output.reshape(-1, int(grad_output.shape[-1]))
    grad_input_2d = grad_output_2d.matmul(weight)
    return grad_input_2d.reshape(*grad_output.shape[:-1], int(weight.shape[-1]))


def _install_fast_frozen_output_backward() -> None:
    from megatron.core.tensor_parallel.layers import LinearWithFrozenWeight

    if getattr(LinearWithFrozenWeight.backward, "__art_fast_output_backward__", False):
        return

    def _fast_backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        (weight,) = ctx.saved_tensors
        grad_input = _frozen_linear_grad_input(grad_output, weight)
        if ctx.allreduce_dgrad:
            torch.distributed.all_reduce(grad_input, group=ctx.tp_group)
        return grad_input, None, None, None, None

    setattr(_fast_backward, "__art_fast_output_backward__", True)
    LinearWithFrozenWeight.backward = staticmethod(_fast_backward)


def _install_gpt_preprocess_hook(model_chunks: list[MegatronModule]) -> None:
    for chunk in model_chunks:
        module: Any = chunk
        while not isinstance(module, GPTModel) and hasattr(module, "module"):
            module = module.module
        if not isinstance(module, GPTModel):
            continue
        preprocess = module._preprocess

        def preprocess_hook(*args, _preprocess=preprocess, **kwargs):
            preproc_output = list(_preprocess(*args, **kwargs))
            preproc_output[0].requires_grad = True  # type: ignore[index]
            table = preproc_output[1]  # [S, B, 1, D]  # type: ignore[index]
            embedding_dim = table.size(-1)
            table_flat = table.view(table.size(0), embedding_dim)
            position_ids = kwargs["position_ids"]  # [B, S]
            batch_size, sequence_length = position_ids.shape
            gathered = table_flat.index_select(0, position_ids.reshape(-1))
            gathered = (
                gathered.view(batch_size, sequence_length, embedding_dim)
                .permute(1, 0, 2)
                .contiguous()
            )
            preproc_output[1] = gathered.unsqueeze(2)  # [S, B, 1, D]
            return tuple(preproc_output)

        module._preprocess = preprocess_hook  # type: ignore[attr-defined]


def _default_optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        bf16=True,
        lr=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        clip_grad=0.1,
        weight_decay=0.1,
        adam_eps=1e-13,
    )


def configure_moe_routing_replay(
    runtime: TrainingRuntime,
    *,
    replay_bundle_path: str | None = None,
    replay_bundle: MoeRoutingReplayBundle | None = None,
    strict: bool = True,
) -> None:
    if runtime.moe_routing_replay_controller is not None:
        runtime.moe_routing_replay_controller.remove_router_patches()
        runtime.moe_routing_replay_controller = None

    if replay_bundle is not None and replay_bundle_path is not None:
        raise RuntimeError(
            "Provide either replay_bundle_path or replay_bundle, not both"
        )
    if replay_bundle is None and replay_bundle_path is None:
        return

    if replay_bundle is None:
        if replay_bundle_path is None:
            raise RuntimeError(
                "replay_bundle_path is required when replay_bundle is None"
            )
        replay_bundle = MoeRoutingReplayBundle.from_dir(replay_bundle_path)

    controller = MoeRoutingReplayController(
        bundle=replay_bundle,
        strict=strict,
    )
    controller.install_router_patches(runtime.model)
    runtime.moe_routing_replay_controller = controller


def build_training_runtime(
    *,
    model_identifier: str | None = None,
    provider_torch_dtype: torch.dtype = torch.bfloat16,
    provider_configure: Callable[[Any], None] | None = None,
    optimizer_config: OptimizerConfig | None = None,
    moe_routing_replay_path: str | None = None,
    moe_routing_replay_bundle: MoeRoutingReplayBundle | None = None,
    moe_routing_replay_strict: bool = True,
    print_env: bool = True,
    print_optimizer_stats: bool = True,
) -> TrainingRuntime:
    _install_fast_frozen_output_backward()
    provider = get_provider(
        model_identifier
        or os.environ.get("MODEL_IDENTIFIER", DEFAULT_MODEL_IDENTIFIER),
        torch_dtype=provider_torch_dtype,
    )
    if provider_configure is not None:
        provider_configure(provider)
    provider.register_pre_wrap_hook(freeze_model)
    provider.register_pre_wrap_hook(
        lambda chunks: apply_lora_adapters(chunks, provider)
    )

    model = cast(
        list[MegatronModule],
        provider.provide_distributed_model(
            ddp_config=DistributedDataParallelConfig(
                # memory and comm for this should be small anyways cause lora
                grad_reduce_in_fp32=True,
                average_in_collective=False,
            ),
            data_parallel_random_init=False,
        ),
    )

    if not torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        raise RuntimeError(
            "torch.distributed must be initialized before building runtime"
        )
    rank = torch.distributed.get_rank()  # ty: ignore[possibly-missing-attribute]
    world_size = torch.distributed.get_world_size()  # ty: ignore[possibly-missing-attribute]

    if rank == 0 and print_env:
        print("TORCHINDUCTOR_CACHE_DIR:", os.environ["TORCHINDUCTOR_CACHE_DIR"])
        print("Resolved inductor cache_dir():", inductor_cache_dir())
        print("TRITON_CACHE_DIR:", os.environ["TRITON_CACHE_DIR"])

    _install_gpt_preprocess_hook(model)

    optimizer = get_megatron_optimizer(
        config=optimizer_config or _default_optimizer_config(),
        model_chunks=model,
    )

    if rank == 0 and print_optimizer_stats:
        num_params = sum(
            p.numel()
            for group in optimizer.param_groups
            if not group["is_decoupled_lr"]
            for p in group["params"]
        )
        print(f"Number of parameters in optimizer: {num_params:,}")
        total_params = sum(p.numel() for module in model for p in module.parameters())
        percent = (num_params / total_params) * 100 if total_params > 0 else 0
        print(f"Optimizer parameters as percent of total: {percent:0.2f}%")

    runtime = TrainingRuntime(
        provider=provider,
        model=model,
        optimizer=optimizer,
        rank=rank,
        world_size=world_size,
    )
    configure_moe_routing_replay(
        runtime,
        replay_bundle_path=moe_routing_replay_path,
        replay_bundle=moe_routing_replay_bundle,
        strict=moe_routing_replay_strict,
    )
    return runtime


def iter_modules(model_chunks: list[MegatronModule]) -> Any:
    for chunk in model_chunks:
        for module in chunk.modules():
            yield module


def load_adapter_into_model(
    model_chunks: list[MegatronModule],
    adapter_model: dict[str, torch.Tensor],
    optimizer: Any | None = None,
) -> None:
    with torch.no_grad():
        for module in iter_modules(model_chunks):
            if hasattr(module, "load_lora"):
                module.load_lora(adapter_model)  # type: ignore[attr-defined]

    if optimizer is None:
        return
    optimizer.reload_model_params()


def collect_sharded_lora_state(
    model_chunks: list[MegatronModule],
    adapter_model: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    sharded_state_dict: dict[str, torch.Tensor] = {}
    sharded_state_manifest: dict[str, dict[str, Any]] = {}
    for module in iter_modules(model_chunks):
        if hasattr(module, "sharded_lora_state_dict"):
            module_sharded_lora_state_dict: dict[str, torch.Tensor] = (
                module.sharded_lora_state_dict()  # type: ignore[attr-defined]
            )
            for key, value in module_sharded_lora_state_dict.items():
                target_dtype = (
                    adapter_model[key].dtype if key in adapter_model else value.dtype
                )
                sharded_state_dict[key] = value.to(target_dtype)
        if hasattr(module, "sharded_lora_manifest"):
            module_sharded_lora_manifest: dict[str, dict[str, Any]] = (
                module.sharded_lora_manifest()  # type: ignore[attr-defined]
            )
            sharded_state_manifest.update(module_sharded_lora_manifest)
    return sharded_state_dict, sharded_state_manifest


@torch.no_grad()
def select_indexed_inputs(packed_tensors: PackedTensors, index: int) -> PackedTensors:
    return PackedTensors(  # type: ignore[call-arg]
        **{
            key: value[index : index + 1]
            for key, value in packed_tensors.items()
            if isinstance(value, torch.Tensor)
        },
        pixel_values=[None],
        image_grid_thw=[None],
    )


@torch.no_grad()
def _clone_packed_tensors(inputs: PackedTensors) -> PackedTensors:
    return PackedTensors(  # type: ignore[call-arg]
        **{
            key: value.clone()
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        },
        pixel_values=[None],
        image_grid_thw=[None],
    )


@torch.no_grad()
def _zero_contribution_inputs(template: PackedTensors) -> PackedTensors:
    dummy = _clone_packed_tensors(template)
    dummy["assistant_mask"].zero_()
    return dummy


def resolve_global_grad_accumulation_sequences(
    global_grad_accumulation_sequences: int | None,
) -> int:
    dp_world_size = ps.get_data_parallel_world_size()
    if global_grad_accumulation_sequences is None:
        return dp_world_size
    return global_grad_accumulation_sequences


def resolve_local_grad_accumulation_sequences(
    global_grad_accumulation_sequences: int | None,
) -> int:
    resolved_global_grad_accumulation_sequences = (
        resolve_global_grad_accumulation_sequences(
            global_grad_accumulation_sequences=global_grad_accumulation_sequences
        )
    )
    dp_world_size = ps.get_data_parallel_world_size()
    if (
        resolved_global_grad_accumulation_sequences <= 0
        or resolved_global_grad_accumulation_sequences % dp_world_size != 0
    ):
        raise RuntimeError(
            "Invalid global grad accumulation / DP world size combination: "
            f"global_grad_accumulation_sequences={resolved_global_grad_accumulation_sequences}, "
            f"dp_world_size={dp_world_size}"
        )
    return resolved_global_grad_accumulation_sequences // dp_world_size


def build_micro_sample_indices(
    step_index: int,
    num_sequences: int,
    global_grad_accumulation_sequences: int | None,
) -> list[int | None]:
    dp_rank = ps.get_data_parallel_rank()
    resolved_global_grad_accumulation_sequences = (
        resolve_global_grad_accumulation_sequences(
            global_grad_accumulation_sequences=global_grad_accumulation_sequences
        )
    )
    dp_world_size = ps.get_data_parallel_world_size()
    local_grad_accumulation_sequences = resolve_local_grad_accumulation_sequences(
        global_grad_accumulation_sequences=resolved_global_grad_accumulation_sequences,
    )
    base_global_sample_index = step_index * resolved_global_grad_accumulation_sequences
    global_step_indices: list[int | None] = []
    for offset in range(resolved_global_grad_accumulation_sequences):
        global_sample_index = base_global_sample_index + offset
        global_step_indices.append(
            global_sample_index if global_sample_index < num_sequences else None
        )
    return [
        global_step_indices[offset * dp_world_size + dp_rank]
        for offset in range(local_grad_accumulation_sequences)
    ]


def select_micro_inputs(
    packed_tensors: PackedTensors,
    sample_indices: list[int | None],
    zero_template: PackedTensors,
) -> list[PackedTensors]:
    return [
        _clone_packed_tensors(zero_template)
        if sample_index is None
        else select_indexed_inputs(packed_tensors, sample_index)
        for sample_index in sample_indices
    ]


def _move_inputs_to_device(inputs: PackedTensors, device: torch.device) -> None:
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)  # type: ignore[index]


def _optimizer_step(
    optimizer: Any,
    learning_rate: float,
) -> tuple[bool, float, int | None]:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    update_successful, grad_norm, num_zeros_in_grad = cast(
        tuple[bool, float, int | None], optimizer.step()
    )
    optimizer.zero_grad()
    return update_successful, grad_norm, num_zeros_in_grad


def _reduce_loss(
    loss: torch.Tensor,
    op: Any = torch.distributed.ReduceOp.AVG,  # ty: ignore[possibly-missing-attribute]
    group: Any | None = None,
) -> torch.Tensor:
    reduced_loss = loss.detach().clone()
    torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
        reduced_loss,
        op=op,
        group=group,
    )
    return reduced_loss


def _count_trainable_tokens(inputs: PackedTensors) -> float:
    assistant_mask = shift_tensor(inputs["assistant_mask"], False)
    return float(assistant_mask.sum().item())


def _local_trainable_token_count_tensor(
    micro_inputs: list[PackedTensors],
    device: torch.device,
) -> torch.Tensor:
    local_token_total = sum(_count_trainable_tokens(micro) for micro in micro_inputs)
    return torch.tensor([local_token_total], device=device, dtype=torch.float32)


def run_training_step(
    *,
    model_chunks: list[MegatronModule],
    optimizer: Any,
    learning_rate: float,
    inputs: PackedTensors | list[PackedTensors],
    config: types.TrainConfig,
    experimental_config: dev.TrainConfig,
    step_index: int,
    sample_index: int | list[int | None],
    ref_logprobs: torch.Tensor | None = None,
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
) -> TrainStepResult:
    micro_inputs = inputs if isinstance(inputs, list) else [inputs]
    if not micro_inputs:
        raise ValueError("run_training_step requires at least one packed sequence")

    if isinstance(sample_index, list):
        if len(sample_index) != len(micro_inputs):
            raise ValueError(
                "sample_index list length must match number of micro inputs: "
                f"{len(sample_index)} != {len(micro_inputs)}"
            )
        micro_sample_indices = sample_index
    else:
        assert len(micro_inputs) == 1
        micro_sample_indices = [sample_index]

    if moe_routing_replay_controller is not None:
        resolved_global_grad_accumulation_sequences = (
            resolve_global_grad_accumulation_sequences(
                config.grad_accumulation_sequences
            )
        )
        moe_routing_replay_controller.set_step(
            step_index=step_index,
            sample_index=micro_sample_indices,
            global_grad_accumulation_sequences=resolved_global_grad_accumulation_sequences,
        )

    device = next(model_chunks[0].parameters()).device

    for chunk in model_chunks:
        chunk.zero_grad_buffer()  # ty: ignore[call-non-callable]

    micro_count = len(micro_inputs)
    raw_loss_sum: torch.Tensor | None = None
    num_tokens = _local_trainable_token_count_tensor(micro_inputs, device=device)
    probs_corr_sum = 0.0
    new_logprobs: torch.Tensor | None = None

    for micro in micro_inputs:
        _move_inputs_to_device(micro, device)
        attention_state = create_shared_prefix_attention_state(
            group_ids=micro["group_ids"],
            parent_ids=micro["parent_ids"],
        )
        attention_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=device)

        new_logprobs = -model_chunks[0](
            input_ids=micro["tokens"],
            position_ids=micro["input_pos"],
            attention_mask=attention_mask,
            labels=shift_tensor(micro["tokens"], 0),
            extra_block_kwargs={"attention_bias": attention_state},
        )

        loss_info = loss_fn(
            micro,  # ty: ignore[invalid-argument-type]
            new_logprobs,
            ref_logprobs,
            None,
            experimental_config,
            reduction="sum",
        )
        micro_loss = loss_info.policy_loss
        micro_loss.backward()
        probs_corr_sum += float(loss_info.probs_corr.item())
        detached_micro_loss = micro_loss.detach()
        if raw_loss_sum is None:
            raw_loss_sum = detached_micro_loss
        else:
            raw_loss_sum = raw_loss_sum + detached_micro_loss

    if new_logprobs is None or raw_loss_sum is None:
        raise RuntimeError("run_training_step did not produce outputs")

    # num_tokens is reduced in place across ranks by finalize_model_grads().
    finalize_model_grads_extended(model_chunks, num_tokens=num_tokens)
    update_successful, grad_norm, num_zeros_in_grad = _optimizer_step(
        optimizer,
        learning_rate,
    )
    global_num_tokens = max(num_tokens.item(), 1.0)
    reduced_loss = _reduce_loss(
        raw_loss_sum / global_num_tokens,
        op=torch.distributed.ReduceOp.SUM,  # ty: ignore[possibly-missing-attribute]
        group=ps.get_data_parallel_group(with_context_parallel=True),
    )

    if moe_routing_replay_controller is not None:
        moe_routing_replay_controller.finalize_step()

    return TrainStepResult(
        reduced_loss=reduced_loss,
        probs_corr=probs_corr_sum / micro_count,
        new_logprobs=new_logprobs,
        update_successful=update_successful,
        grad_norm=grad_norm,
        num_zeros_in_grad=num_zeros_in_grad,
    )


def _run_service_loop(runtime: TrainingRuntime) -> None:
    offload_state = OffloadState()
    offload_to_cpu(runtime.model, runtime.optimizer, runtime.rank, offload_state)
    from .shared import run_megatron_worker_loop

    def wait_until_ready() -> None:
        while os.path.exists(DEFAULT_VLLM_WAKE_LOCK_PATH):
            time.sleep(0.2)

    run_megatron_worker_loop(
        runtime,
        supports_sft=False,
        wait_until_ready=wait_until_ready,
        before_job=lambda: reload_to_gpu(
            runtime.model, runtime.optimizer, runtime.rank, offload_state
        ),
        after_job=lambda: offload_to_cpu(
            runtime.model, runtime.optimizer, runtime.rank, offload_state
        ),
    )


def main() -> None:
    runtime = build_training_runtime(
        model_identifier=os.environ.get("MODEL_IDENTIFIER", DEFAULT_MODEL_IDENTIFIER)
    )
    _run_service_loop(runtime)


if __name__ == "__main__":
    main()
