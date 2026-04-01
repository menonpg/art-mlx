import gc
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable

from megatron.core import parallel_state as ps
import torch

from ..loss import shift_tensor
from ..preprocessing.pack import PackedTensors, packed_tensors_from_dir
from .finalize_grads import finalize_model_grads_extended
from .flex_attention import create_shared_prefix_attention_state
from .jobs import DEFAULT_JOBS_DIR, MegatronSFTTrainingJob, MegatronTrainingJob
from .offload import clear_optimizer_state
from .train import (
    DEFAULT_MODEL_IDENTIFIER,
    TrainingRuntime,
    _clone_packed_tensors,
    _zero_contribution_inputs,
    build_micro_sample_indices,
    build_training_runtime,
    collect_sharded_lora_state,
    configure_moe_routing_replay,
    load_adapter_into_model,
    print0,
    resolve_global_grad_accumulation_sequences,
    run_training_step,
    select_indexed_inputs,
    select_micro_inputs,
)

safetensors = importlib.import_module("safetensors")
safetensors_torch = importlib.import_module("safetensors.torch")
safe_open = safetensors.safe_open
load_file = safetensors_torch.load_file
save_file = safetensors_torch.save_file

MegatronTrainContext = TrainingRuntime
MegatronJob = MegatronTrainingJob | MegatronSFTTrainingJob


def create_megatron_train_context(
    model_identifier: str = DEFAULT_MODEL_IDENTIFIER,
) -> MegatronTrainContext:
    return build_training_runtime(model_identifier=model_identifier)


def run_megatron_worker_loop(
    ctx: MegatronTrainContext,
    *,
    supports_sft: bool,
    wait_until_ready: Callable[[], None] | None = None,
    before_job: Callable[[], None] | None = None,
    after_job: Callable[[], None] | None = None,
) -> None:
    while True:
        torch.distributed.barrier()  # type: ignore[possibly-missing-attribute]
        os.makedirs(DEFAULT_JOBS_DIR, exist_ok=True)
        job_names = sorted(
            job_name
            for job_name in os.listdir(DEFAULT_JOBS_DIR)
            if job_name.endswith(".json")
        )
        if not job_names:
            time.sleep(1)
            continue

        if wait_until_ready is not None:
            wait_until_ready()
        if before_job is not None:
            before_job()

        job_path = os.path.join(DEFAULT_JOBS_DIR, job_names[0])
        job = _load_megatron_job(job_path, supports_sft=supports_sft)
        print0(ctx.rank, "Loaded job from", job_path)
        print0(ctx.rank, "Job:", job)

        try:
            _run_megatron_job(ctx, job)
        finally:
            if after_job is not None:
                after_job()

        finalize_megatron_job(
            ctx,
            job_path=job_path,
            log_path=job.log_path,
            cleanup_path=_job_cleanup_path(job),
        )


def run_megatron_rl_job(
    ctx: MegatronTrainContext,
    job: MegatronTrainingJob,
) -> None:
    packed_tensors = None
    adapter_model = None
    template = None
    zero_template = None

    try:
        configure_moe_routing_replay(
            ctx,
            replay_bundle_path=job.moe_routing_replay_path,
            strict=job.moe_routing_replay_strict,
        )
        adapter_model = _load_lora_and_optimizer(
            ctx,
            lora_path=job.lora_path,
            optimizer_state_path=job.optimizer_state_path,
        )

        print0(ctx.rank, "Loading packed tensors from", job.disk_packed_tensors["dir"])
        packed_tensors = packed_tensors_from_dir(**job.disk_packed_tensors)
        template = _clone_packed_tensors(select_indexed_inputs(packed_tensors, 0))
        zero_template = _zero_contribution_inputs(template)
        num_sequences = job.disk_packed_tensors["num_sequences"]
        global_grad_accumulation_sequences = resolve_global_grad_accumulation_sequences(
            job.config.grad_accumulation_sequences
        )
        num_steps = math.ceil(num_sequences / global_grad_accumulation_sequences)
        for step_index in range(num_steps):
            micro_indices = build_micro_sample_indices(
                step_index=step_index,
                num_sequences=num_sequences,
                global_grad_accumulation_sequences=global_grad_accumulation_sequences,
            )
            micro_inputs = select_micro_inputs(
                packed_tensors,
                micro_indices,
                zero_template,
            )
            step_result = run_training_step(
                model_chunks=ctx.model,
                optimizer=ctx.optimizer,
                learning_rate=job.config.learning_rate,
                inputs=micro_inputs,
                config=job.config,
                experimental_config=job.experimental_config,
                ref_logprobs=None,
                step_index=step_index,
                sample_index=micro_indices,
                moe_routing_replay_controller=ctx.moe_routing_replay_controller,
            )
            print0(
                ctx.rank,
                "Correlation between old and new probabilities:",
                step_result.probs_corr,
            )

            if ctx.rank == 0:
                with open(job.log_path, "a+", encoding="utf-8") as log_file:
                    log_msg = json.dumps(
                        {
                            "loss": step_result.reduced_loss.item(),
                            "grad_norm": step_result.grad_norm,
                            "probs_corr": step_result.probs_corr,
                        }
                    )
                    print("Logging", log_msg)
                    log_file.write(log_msg + "\n")

        _save_lora_and_optimizer(
            ctx,
            adapter_model=adapter_model,
            lora_path=job.lora_path,
            optimizer_state_path=job.optimizer_state_path,
        )
    finally:
        if packed_tensors is not None:
            del packed_tensors
        if adapter_model is not None:
            del adapter_model
        if template is not None:
            del template
        if zero_template is not None:
            del zero_template
        if "micro_inputs" in locals():
            del micro_inputs
        gc.collect()
        torch.cuda.empty_cache()


def run_megatron_sft_job(
    ctx: MegatronTrainContext,
    job: MegatronSFTTrainingJob,
) -> None:
    adapter_model = None

    try:
        configure_moe_routing_replay(ctx)
        adapter_model = _load_lora_and_optimizer(
            ctx,
            lora_path=job.lora_path,
            optimizer_state_path=job.optimizer_state_path,
        )

        ctx.optimizer.config.clip_grad = job.max_grad_norm
        for param_group in ctx.optimizer.param_groups:
            param_group["weight_decay"] = job.weight_decay

        device = next(ctx.model[0].parameters()).device
        dp_rank = ps.get_data_parallel_rank()
        dp_world_size = ps.get_data_parallel_world_size()

        for batch_idx in range(job.num_batches):
            batch_start_time = time.perf_counter()
            batch_dir = os.path.join(job.sft_data_dir, f"batch_{batch_idx:06d}")
            batch_metadata, trajectory_tensors = _load_sft_batch_from_disk(batch_dir)
            global_trainable_tokens = max(
                int(batch_metadata["num_trainable_tokens"]),
                1,
            )
            local_trajectory_tensors = trajectory_tensors[dp_rank::dp_world_size]

            for chunk in ctx.model:
                chunk.zero_grad_buffer()  # type: ignore[call-non-callable]

            batch_loss = torch.tensor(0.0, device=device)
            local_trainable_tokens = 0.0
            for param_group in ctx.optimizer.param_groups:
                param_group["lr"] = job.learning_rates[batch_idx]

            for traj_tensors in local_trajectory_tensors:
                attention_mask_1d = traj_tensors["attention_mask"]
                actual_len = int(attention_mask_1d.sum().item())
                input_ids = (
                    traj_tensors["input_ids"][:actual_len].unsqueeze(0).to(device)
                )
                labels = traj_tensors["labels"][:actual_len].unsqueeze(0).to(device)
                seq_len = input_ids.shape[1]
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                shifted_labels = shift_tensor(labels, -100)
                mask = shifted_labels != -100
                local_trainable_tokens += float(mask.sum().item())

                per_token_loss: torch.Tensor = ctx.model[0](
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=_placeholder_attention_mask(device),
                    labels=shifted_labels,
                    extra_block_kwargs={
                        "attention_bias": _causal_attention_state(seq_len, device),
                    },
                )
                masked_loss = per_token_loss[mask].sum()
                masked_loss.backward()
                batch_loss += masked_loss.detach()

            num_tokens = torch.tensor(
                [local_trainable_tokens],
                device=device,
                dtype=torch.float32,
            )
            finalize_model_grads_extended(ctx.model, num_tokens=num_tokens)
            update_successful, grad_norm, num_zeros_in_grad = ctx.optimizer.step()
            ctx.optimizer.zero_grad()

            torch.distributed.all_reduce(
                batch_loss,
                op=torch.distributed.ReduceOp.SUM,
                group=ps.get_data_parallel_group(with_context_parallel=True),
            )
            avg_loss = batch_loss / float(global_trainable_tokens)
            batch_time = time.perf_counter() - batch_start_time
            tokens_per_second = (
                global_trainable_tokens / batch_time if batch_time > 0 else 0.0
            )

            if ctx.rank == 0:
                with open(job.log_path, "a+", encoding="utf-8") as log_file:
                    log_msg = json.dumps(
                        {
                            "loss": avg_loss.item(),
                            "learning_rate": job.learning_rates[batch_idx],
                            "grad_norm": float(grad_norm),
                            "num_trajectories": float(
                                batch_metadata["num_trajectories"]
                            ),
                            "num_trainable_tokens": float(global_trainable_tokens),
                            "tokens_per_second": tokens_per_second,
                        }
                    )
                    print("Logging SFT", log_msg)
                    log_file.write(log_msg + "\n")

        _save_lora_and_optimizer(
            ctx,
            adapter_model=adapter_model,
            lora_path=job.lora_path,
            optimizer_state_path=job.optimizer_state_path,
        )
    finally:
        if adapter_model is not None:
            del adapter_model
        gc.collect()
        torch.cuda.empty_cache()


def _load_megatron_job(job_path: str, *, supports_sft: bool) -> MegatronJob:
    with open(job_path, "rb") as handle:
        job_data = json.loads(handle.read())
    if job_data.get("job_type") == "sft":
        if not supports_sft:
            raise NotImplementedError("SFT jobs are not supported in this worker loop")
        return MegatronSFTTrainingJob.model_validate(job_data)
    return MegatronTrainingJob.model_validate(job_data)


def _run_megatron_job(ctx: MegatronTrainContext, job: MegatronJob) -> None:
    if isinstance(job, MegatronSFTTrainingJob):
        run_megatron_sft_job(ctx, job)
        return
    run_megatron_rl_job(ctx, job)


def _job_cleanup_path(job: MegatronJob) -> str:
    if isinstance(job, MegatronSFTTrainingJob):
        return job.sft_data_dir
    return job.disk_packed_tensors["dir"]


def merge_lora_adapter(lora_path: str) -> None:
    base_dir = Path(lora_path)
    shard_filenames = sorted(base_dir.glob("adapter_model-*-of-*.safetensors"))
    if not shard_filenames:
        return

    shard_files_by_suffix = {
        path.name.removeprefix("adapter_model-").removesuffix(".safetensors"): path
        for path in shard_filenames
    }
    manifest_filenames = sorted(base_dir.glob("adapter_manifest-*-of-*.json"))
    manifest_files_by_suffix = {
        path.name.removeprefix("adapter_manifest-").removesuffix(".json"): path
        for path in manifest_filenames
    }

    if set(shard_files_by_suffix) != set(manifest_files_by_suffix):
        raise RuntimeError(
            "Shard/manifest coverage mismatch: "
            f"shards={sorted(shard_files_by_suffix)}, "
            f"manifests={sorted(manifest_files_by_suffix)}"
        )

    entries_by_key: dict[str, list[tuple[dict[str, Any], torch.Tensor]]] = {}
    for suffix in sorted(shard_files_by_suffix):
        shard_path = shard_files_by_suffix[suffix]
        manifest_path = manifest_files_by_suffix[suffix]
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            shard_manifest: dict[str, dict[str, Any]] = json.load(manifest_file)
        with safe_open(shard_path, framework="pt") as file:
            shard_tensors = {key: file.get_tensor(key) for key in file.keys()}

        if set(shard_tensors) != set(shard_manifest):
            raise RuntimeError(
                f"Tensor/manifest key mismatch for shard suffix={suffix}: "
                f"tensor_keys={sorted(shard_tensors)}, "
                f"manifest_keys={sorted(shard_manifest)}"
            )
        for key, tensor in shard_tensors.items():
            entries_by_key.setdefault(key, []).append((shard_manifest[key], tensor))

    adapter_model: dict[str, torch.Tensor] = {}
    for key, key_entries in entries_by_key.items():
        first_manifest = key_entries[0][0]
        sharded = bool(first_manifest["sharded"])
        shard_world_size = int(first_manifest["shard_world_size"])
        for manifest_entry, _tensor in key_entries:
            if bool(manifest_entry["sharded"]) != sharded:
                raise RuntimeError(f"Inconsistent sharded flag for key={key}")
            if int(manifest_entry["shard_world_size"]) != shard_world_size:
                raise RuntimeError(f"Inconsistent shard world size for key={key}")

        if not sharded:
            if len(key_entries) != 1:
                raise RuntimeError(
                    f"Replicated key={key} expected 1 shard, got {len(key_entries)}"
                )
            tensor = key_entries[0][1]
        else:
            shard_rank_to_tensor: dict[int, torch.Tensor] = {}
            for manifest_entry, shard_tensor in key_entries:
                shard_rank = int(manifest_entry["shard_rank"])
                if shard_rank in shard_rank_to_tensor:
                    raise RuntimeError(
                        f"Duplicate shard_rank={shard_rank} for key={key}"
                    )
                shard_rank_to_tensor[shard_rank] = shard_tensor

            expected_shard_ranks = set(range(shard_world_size))
            if set(shard_rank_to_tensor) != expected_shard_ranks:
                raise RuntimeError(
                    f"Shard rank coverage mismatch for key={key}: "
                    f"expected {sorted(expected_shard_ranks)}, got {sorted(shard_rank_to_tensor)}"
                )

            ordered_shards = [
                shard_rank_to_tensor[shard_rank]
                for shard_rank in range(shard_world_size)
            ]
            concat_dim = 1 if "lora_A" in key else 0
            tensor = torch.cat(ordered_shards, dim=concat_dim)
        adapter_model[key] = tensor

    adapter_model_path = base_dir / "adapter_model.safetensors"
    save_file(adapter_model, adapter_model_path)
    for filename in shard_filenames:
        filename.unlink()
    for filename in manifest_filenames:
        filename.unlink()


def _load_sft_batch_from_disk(
    batch_dir: str,
) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]]]:
    with open(os.path.join(batch_dir, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)

    trajectory_tensors = []
    for index in range(metadata["num_trajectory_tensors"]):
        tensors = load_file(os.path.join(batch_dir, f"trajectory_{index}.safetensors"))
        trajectory_tensors.append(tensors)
    return metadata, trajectory_tensors


def _load_lora_and_optimizer(
    ctx: MegatronTrainContext,
    *,
    lora_path: str,
    optimizer_state_path: str,
) -> dict[str, torch.Tensor]:
    adapter_model_path = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_model_path):
        raise FileNotFoundError(f"No adapter model found at {adapter_model_path}")
    print0(ctx.rank, "Loading adapter model from", adapter_model_path)
    adapter_model = load_file(adapter_model_path)
    load_adapter_into_model(ctx.model, adapter_model, ctx.optimizer)

    optimizer_shard_path = os.path.join(
        optimizer_state_path,
        f"{ctx.rank + 1:02d}-of-{ctx.world_size:02d}.pt",
    )
    if os.path.exists(optimizer_shard_path):
        print0(ctx.rank, "Loading optimizer state from", optimizer_shard_path)
        ctx.optimizer.load_state_dict(torch.load(optimizer_shard_path))
    else:
        print0(
            ctx.rank,
            "No optimizer state found at",
            optimizer_shard_path,
            "- resetting optimizer for new run",
        )
        clear_optimizer_state(ctx.optimizer)
        ctx.optimizer.reload_model_params()
    return adapter_model


def _save_lora_and_optimizer(
    ctx: MegatronTrainContext,
    *,
    adapter_model: dict[str, torch.Tensor],
    lora_path: str,
    optimizer_state_path: str,
) -> None:
    sharded_state_dict, sharded_state_manifest = collect_sharded_lora_state(
        ctx.model,
        adapter_model,
    )
    shard_path = os.path.join(
        lora_path,
        f"adapter_model-{ctx.rank + 1:02d}-of-{ctx.world_size:02d}.safetensors",
    )
    manifest_path = os.path.join(
        lora_path,
        f"adapter_manifest-{ctx.rank + 1:02d}-of-{ctx.world_size:02d}.json",
    )
    print("Saving adapter shard to", shard_path)
    os.makedirs(lora_path, exist_ok=True)
    save_file(sharded_state_dict, shard_path)
    print("Saving adapter shard manifest to", manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(sharded_state_manifest, manifest_file, sort_keys=True)

    optimizer_shard_path = os.path.join(
        optimizer_state_path,
        f"{ctx.rank + 1:02d}-of-{ctx.world_size:02d}.pt",
    )
    print("Saving optimizer shard to", optimizer_shard_path)
    os.makedirs(optimizer_state_path, exist_ok=True)
    torch.save(ctx.optimizer.state_dict(), optimizer_shard_path)


def finalize_megatron_job(
    ctx: MegatronTrainContext,
    *,
    job_path: str | None,
    log_path: str,
    cleanup_path: str,
) -> None:
    torch.distributed.barrier()  # type: ignore[possibly-missing-attribute]
    if ctx.rank != 0:
        return

    if job_path is not None and os.path.exists(job_path):
        os.remove(job_path)
    if os.path.exists(cleanup_path):
        shutil.rmtree(cleanup_path)
    with open(log_path, "a+", encoding="utf-8") as log_file:
        log_file.write("all done\n")


def _placeholder_attention_mask(device: torch.device) -> torch.Tensor:
    return torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=device)


def _causal_attention_state(seq_len: int, device: torch.device) -> Any:
    group_ids = torch.zeros((1, seq_len), dtype=torch.int64, device=device)
    parent_ids = torch.zeros_like(group_ids)
    return create_shared_prefix_attention_state(
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
