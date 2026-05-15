from __future__ import annotations

from contextlib import asynccontextmanager
import math
import os
from pathlib import Path
import socket
import subprocess
from typing import AsyncIterator, Literal
import uuid

from pydantic import BaseModel, Field
import pytest

import art
from art import dev
from art.dev.get_model_config import default_target_modules
from art.megatron.runtime.backend import MegatronBackend
from art.pipeline_trainer import PipelineTrainer
import art.vllm_runtime as vllm_runtime

from ..model_support.oracle_harness import Topology
from ..model_support.oracle_worker import provider_topology_env
from ..trainability.yes_no_trainability import _wandb_disabled

torch = pytest.importorskip("torch")

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-35B-A3B"
LIVE_ENV = "ART_RUN_LIVE_MEGATRON_EXTERNAL_VLLM_LENGTH_SMOKE"
TRAINING_TOPOLOGY = Topology(tp=1, cp=2, ep=2, etp=1, dp=1, sp=False)


class LengthScenario(BaseModel):
    scenario_index: int
    target_tokens: int
    max_tokens: int
    metadata: dict[str, int] = Field(default_factory=dict)


class LengthSampleReport(BaseModel):
    split: Literal["train", "eval", "baseline"]
    step: int | None
    scenario_index: int
    target_tokens: int
    max_tokens: int
    generated_tokens: int
    abs_error: int
    reward: float
    text: str


class LengthTrainabilityReport(BaseModel):
    base_model: str
    max_steps_off_policy: int
    max_steps: int
    latest_step: int
    trainer_gpu_ids: list[int]
    inference_gpu_ids: list[int]
    training_topology: dict[str, int | bool | None]
    inference_engine_args: dict[str, object]
    runtime_command: list[str]
    runtime_log_path: str
    baseline_final_target_reward: float
    final_eval_reward: float | None
    model_ids_after: list[str]
    samples: list[LengthSampleReport]


def _require_opt_in() -> None:
    if os.environ.get(LIVE_ENV) != "1":
        pytest.skip(f"set {LIVE_ENV}=1 to run external vLLM length trainability")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _base_model() -> str:
    return os.environ.get(
        "ART_LIVE_EXTERNAL_VLLM_LENGTH_BASE_MODEL",
        os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL),
    )


def _parse_gpu_ids(name: str) -> list[int] | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _nvidia_smi_lines(*query_fields: str) -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(query_fields)}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"nvidia-smi is required for live CUDA GPU discovery: {exc}")
    return [line.strip() for line in output.splitlines() if line.strip()]


def _gpu_memory_info_by_index() -> dict[int, tuple[int, int]]:
    info: dict[int, tuple[int, int]] = {}
    for line in _nvidia_smi_lines("index", "memory.free", "memory.total"):
        raw_index, raw_free_mib, raw_total_mib = [
            part.strip() for part in line.split(",")
        ]
        info[int(raw_index)] = (
            int(raw_free_mib) * 1024**2,
            int(raw_total_mib) * 1024**2,
        )
    return info


def _resolve_gpu_ids() -> tuple[list[int], list[int]]:
    trainer_ids = _parse_gpu_ids("ART_EXTERNAL_VLLM_LENGTH_TRAINER_GPU_IDS")
    inference_ids = _parse_gpu_ids("ART_EXTERNAL_VLLM_LENGTH_INFERENCE_GPU_IDS")
    if trainer_ids is not None or inference_ids is not None:
        if trainer_ids is None or inference_ids is None:
            raise RuntimeError(
                "ART_EXTERNAL_VLLM_LENGTH_TRAINER_GPU_IDS and "
                "ART_EXTERNAL_VLLM_LENGTH_INFERENCE_GPU_IDS must both be set"
            )
    else:
        gpu_count = len(_gpu_memory_info_by_index())
        if gpu_count < 4:
            pytest.skip(
                "Need at least 4 visible CUDA GPUs for local external-vLLM "
                "length trainability: 2 training GPUs and 2 inference GPUs."
            )
        trainer_ids = [0, 1]
        inference_ids = [2, 3]

    assert trainer_ids is not None
    assert inference_ids is not None
    if len(trainer_ids) != 2 or len(inference_ids) != 2:
        raise RuntimeError(
            "External length trainability requires exactly 2 trainer GPUs and "
            "exactly 2 inference GPUs for CP2/EP2."
        )
    overlap = set(trainer_ids) & set(inference_ids)
    if overlap:
        raise RuntimeError(f"Trainer and inference GPU IDs overlap: {sorted(overlap)}")
    visible = len(_gpu_memory_info_by_index())
    invalid = [gpu_id for gpu_id in [*trainer_ids, *inference_ids] if gpu_id >= visible]
    if invalid:
        raise RuntimeError(
            f"GPU IDs {invalid} are outside the visible CUDA range 0..{visible - 1}"
        )
    return trainer_ids, inference_ids


def _check_inference_gpu_memory(device_ids: list[int]) -> None:
    min_free_gib = _env_float("ART_EXTERNAL_VLLM_LENGTH_MIN_FREE_GPU_GIB", 20.0)
    memory_info = _gpu_memory_info_by_index()
    for device_id in device_ids:
        free_bytes, _total_bytes = memory_info[device_id]
        free_gib = free_bytes / (1024**3)
        if free_gib < min_free_gib:
            pytest.skip(
                "Insufficient free GPU memory for external vLLM length smoke: "
                f"GPU {device_id} has {free_gib:.1f} GiB free < "
                f"{min_free_gib:.1f} GiB required."
            )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _target_tokens_for_index(index: int) -> int:
    return _env_int("ART_EXTERNAL_VLLM_LENGTH_TARGET_START", 16) + index * _env_int(
        "ART_EXTERNAL_VLLM_LENGTH_TARGET_INCREMENT",
        4,
    )


def _scenario(index: int) -> LengthScenario:
    target_tokens = _target_tokens_for_index(index)
    max_tokens = max(
        target_tokens + 1,
        math.ceil(
            target_tokens
            * _env_float("ART_EXTERNAL_VLLM_LENGTH_MAX_TOKENS_MULTIPLIER", 1.4)
        ),
    )
    return LengthScenario(
        scenario_index=index,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        metadata={
            "scenario_index": index,
            "target_tokens": target_tokens,
            "max_tokens": max_tokens,
        },
    )


async def _scenario_iter(count: int) -> AsyncIterator[LengthScenario]:
    for index in range(count):
        yield _scenario(index)


def _messages(scenario: LengthScenario) -> art.Messages:
    return [
        {
            "role": "user",
            "content": (
                "Write a plain answer about a quiet harbor. "
                f"Make the answer exactly {scenario.target_tokens} generated tokens. "
                "Do not use bullets, numbering, code, or a preface."
            ),
        }
    ]


def _extra_body() -> dict[str, object]:
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _generated_token_count(choice: object) -> int:
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None)
    if content is not None:
        return len(content)
    message = getattr(choice, "message", None)
    text = getattr(message, "content", "") or ""
    return len(text.split())


def _reward(generated_tokens: int, target_tokens: int) -> float:
    abs_error = abs(generated_tokens - target_tokens)
    return max(0.0, 1.0 - abs_error / max(1, target_tokens))


def _sample_report(
    *,
    split: Literal["train", "eval", "baseline"],
    step: int | None,
    scenario: LengthScenario,
    choice: object,
) -> LengthSampleReport:
    generated_tokens = _generated_token_count(choice)
    message = getattr(choice, "message", None)
    text = getattr(message, "content", "") or ""
    return LengthSampleReport(
        split=split,
        step=step,
        scenario_index=scenario.scenario_index,
        target_tokens=scenario.target_tokens,
        max_tokens=scenario.max_tokens,
        generated_tokens=generated_tokens,
        abs_error=abs(generated_tokens - scenario.target_tokens),
        reward=_reward(generated_tokens, scenario.target_tokens),
        text=text,
    )


async def _length_group(
    model: art.TrainableModel,
    *,
    scenario: LengthScenario,
    model_name: str,
    split: Literal["train", "eval", "baseline"],
    step: int | None,
    n: int,
    temperature: float,
    samples: list[LengthSampleReport],
) -> art.TrajectoryGroup:
    client = model.openai_client()
    messages = _messages(scenario)
    completion = await client.chat.completions.create(
        messages=messages,
        model=model_name,
        max_tokens=scenario.max_tokens,
        n=n,
        temperature=temperature,
        extra_body=_extra_body(),
        logprobs=True,
        top_logprobs=0,
        timeout=_env_float("ART_EXTERNAL_VLLM_LENGTH_REQUEST_TIMEOUT", 900.0),
    )
    trajectories: list[art.Trajectory] = []
    for choice in completion.choices:
        report = _sample_report(
            split=split,
            step=step,
            scenario=scenario,
            choice=choice,
        )
        samples.append(report)
        trajectories.append(
            art.Trajectory(
                messages_and_choices=[*messages, choice],
                reward=report.reward,
                metrics={
                    "length/generated_tokens": report.generated_tokens,
                    "length/target_tokens": scenario.target_tokens,
                    "length/max_tokens": scenario.max_tokens,
                    "length/abs_error": report.abs_error,
                },
                metadata={
                    "target_tokens": scenario.target_tokens,
                    "max_tokens": scenario.max_tokens,
                    "scenario_index": scenario.scenario_index,
                },
            )
        )
    return art.TrajectoryGroup(trajectories)


def _mean_reward(samples: list[LengthSampleReport]) -> float:
    return sum(sample.reward for sample in samples) / max(1, len(samples))


def _inference_engine_args(base_model: str, inference_gpu_ids: list[int]) -> dict:
    _check_inference_gpu_memory(inference_gpu_ids)
    engine_args: dict[str, object] = {
        "tensor_parallel_size": 2,
        "enable_expert_parallel": True,
        "dtype": "bfloat16",
        "max_model_len": _env_int("ART_EXTERNAL_VLLM_LENGTH_MAX_MODEL_LEN", 1024),
        "max_num_seqs": _env_int("ART_EXTERNAL_VLLM_LENGTH_MAX_NUM_SEQS", 8),
        "enforce_eager": True,
        "lora_dtype": "bfloat16",
        "max_loras": _env_int("ART_EXTERNAL_VLLM_LENGTH_MAX_LORAS", 16),
        "max_cpu_loras": _env_int("ART_EXTERNAL_VLLM_LENGTH_MAX_CPU_LORAS", 16),
        "lora_target_modules": list(default_target_modules(base_model)),
    }
    requested_gpu_memory_utilization = os.environ.get(
        "ART_EXTERNAL_VLLM_LENGTH_GPU_MEMORY_UTILIZATION"
    )
    if requested_gpu_memory_utilization is not None:
        engine_args["gpu_memory_utilization"] = float(requested_gpu_memory_utilization)
    return engine_args


@asynccontextmanager
async def _external_vllm_runtime(
    *,
    artifact_dir: Path,
    base_model: str,
    inference_gpu_ids: list[int],
) -> AsyncIterator[tuple[str, list[str], Path, dict[str, object]]]:
    port = _free_port()
    log_path = artifact_dir / "external_vllm_runtime.log"
    engine_args = _inference_engine_args(base_model, inference_gpu_ids)
    launch_config = vllm_runtime.VllmRuntimeLaunchConfig(
        base_model=base_model,
        port=port,
        host="127.0.0.1",
        cuda_visible_devices=",".join(str(gpu_id) for gpu_id in inference_gpu_ids),
        served_model_name=f"external-vllm-base-{uuid.uuid4().hex[:8]}",
        rollout_weights_mode="lora",
        engine_args=engine_args,
        server_args={"return_tokens_as_token_ids": True},
    )
    command = vllm_runtime.build_vllm_runtime_server_cmd(launch_config)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = "offline"
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(vllm_runtime.get_vllm_runtime_working_dir()),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        await vllm_runtime.wait_for_vllm_runtime(
            process=process,
            host=launch_config.host,
            port=launch_config.port,
            timeout=_env_float("ART_EXTERNAL_VLLM_LENGTH_STARTUP_TIMEOUT", 1800.0),
        )
        yield (
            f"http://{launch_config.host}:{launch_config.port}",
            command,
            log_path,
            engine_args,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        log_file.close()


async def _list_model_ids(model: art.TrainableModel) -> list[str]:
    client = model.openai_client()
    return [model_info.id async for model_info in client.models.list()]


def _internal_config(
    *,
    trainer_gpu_ids: list[int],
    server_url: str,
    backend_root: Path,
) -> dev.InternalModelConfig:
    topology = art.MegatronTopologyConfig(tp=1, cp=2, ep=2, etp=1)
    return {
        "trainer_gpu_ids": trainer_gpu_ids,
        "rollout_weights_mode": "lora",
        "init_args": {
            "max_seq_length": _env_int(
                "ART_EXTERNAL_VLLM_LENGTH_MAX_SEQ_LENGTH",
                1024,
            )
        },
        "megatron_topology": topology.model_dump(),
        "vllm_runtime": {
            "mode": "external",
            "server_url": server_url,
            "local_checkpoint_root": str(backend_root),
            "server_checkpoint_root": str(backend_root),
            "health_timeout_s": _env_float(
                "ART_EXTERNAL_VLLM_LENGTH_HEALTH_TIMEOUT",
                120.0,
            ),
        },
    }


@pytest.mark.parametrize(
    "max_steps_off_policy", [0, 3], ids=["off_policy_0", "off_policy_3"]
)
@pytest.mark.asyncio
async def test_megatron_pipeline_external_vllm_length_trainability_live(
    artifact_dir: Path,
    max_steps_off_policy: int,
) -> None:
    _require_opt_in()
    base_model = _base_model()
    trainer_gpu_ids, inference_gpu_ids = _resolve_gpu_ids()
    max_steps = _env_int("ART_EXTERNAL_VLLM_LENGTH_MAX_STEPS", 10)
    rollouts_per_prompt = _env_int("ART_EXTERNAL_VLLM_LENGTH_ROLLOUTS_PER_PROMPT", 8)
    if rollouts_per_prompt < 2:
        raise RuntimeError(
            "ART_EXTERNAL_VLLM_LENGTH_ROLLOUTS_PER_PROMPT must be at least 2 "
            "so reward groups can have non-zero variance."
        )
    eval_rollouts = _env_int("ART_EXTERNAL_VLLM_LENGTH_EVAL_ROLLOUTS", 1)
    rollout_workers = _env_int(
        "ART_EXTERNAL_VLLM_LENGTH_ROLLOUT_WORKERS",
        max(2, max_steps_off_policy + 1),
    )
    scenario_count = _env_int(
        "ART_EXTERNAL_VLLM_LENGTH_SCENARIOS",
        max_steps * max(rollouts_per_prompt, 2) + rollout_workers + 4,
    )
    backend_root = artifact_dir / f"workspace_off_policy_{max_steps_off_policy}"
    backend_root.mkdir(parents=True, exist_ok=True)
    samples: list[LengthSampleReport] = []

    async with _external_vllm_runtime(
        artifact_dir=artifact_dir,
        base_model=base_model,
        inference_gpu_ids=inference_gpu_ids,
    ) as (server_url, command, runtime_log_path, engine_args):
        with _wandb_disabled(), provider_topology_env(TRAINING_TOPOLOGY):
            async with MegatronBackend(
                path=str(backend_root), in_process=False
            ) as backend:
                model = art.TrainableModel(
                    name=f"length-external-vllm-{uuid.uuid4().hex[:8]}",
                    project="integration-tests",
                    base_model=base_model,
                    _internal_config=_internal_config(
                        trainer_gpu_ids=trainer_gpu_ids,
                        server_url=server_url,
                        backend_root=backend_root,
                    ),
                    report_metrics=[],
                )
                await model.register(backend)

                final_target_scenario = _scenario(max_steps)
                baseline_samples_before = len(samples)
                await _length_group(
                    model,
                    scenario=final_target_scenario,
                    model_name=model.get_inference_name(step=0),
                    split="baseline",
                    step=0,
                    n=eval_rollouts,
                    temperature=0.0,
                    samples=samples,
                )
                baseline_final_target_reward = _mean_reward(
                    samples[baseline_samples_before:]
                )

                async def rollout_fn(
                    rollout_model: art.TrainableModel,
                    scenario: LengthScenario,
                    _config: None,
                ) -> art.TrajectoryGroup:
                    return await _length_group(
                        rollout_model,
                        scenario=scenario,
                        model_name=rollout_model.get_inference_name(),
                        split="train",
                        step=None,
                        n=rollouts_per_prompt,
                        temperature=_env_float(
                            "ART_EXTERNAL_VLLM_LENGTH_ROLLOUT_TEMPERATURE",
                            1.1,
                        ),
                        samples=samples,
                    )

                async def eval_fn(
                    eval_model: art.TrainableModel,
                    step: int,
                    _config: None,
                ) -> list[art.TrajectoryGroup]:
                    return [
                        await _length_group(
                            eval_model,
                            scenario=_scenario(step),
                            model_name=eval_model.get_inference_name(step=step),
                            split="eval",
                            step=step,
                            n=eval_rollouts,
                            temperature=0.0,
                            samples=samples,
                        )
                    ]

                trainer = PipelineTrainer(
                    model=model,
                    backend=backend,
                    rollout_fn=rollout_fn,
                    scenarios=_scenario_iter(scenario_count),
                    config=None,
                    num_rollout_workers=rollout_workers,
                    min_batch_size=1,
                    max_batch_size=1,
                    max_steps_off_policy=max_steps_off_policy,
                    learning_rate=_env_float(
                        "ART_EXTERNAL_VLLM_LENGTH_LEARNING_RATE",
                        1e-4,
                    ),
                    loss_fn="cispo",
                    packed_sequence_length=_env_int(
                        "ART_EXTERNAL_VLLM_LENGTH_PACKED_SEQUENCE_LENGTH",
                        1024,
                    ),
                    megatron_topology=art.MegatronTopologyConfig(
                        tp=1,
                        cp=2,
                        ep=2,
                        etp=1,
                    ),
                    max_steps=max_steps,
                    eval_fn=eval_fn,
                    eval_every_n_steps=1,
                    eval_at_start=True,
                    total_scenarios=scenario_count,
                    log_interval_seconds=30.0,
                    discard_queue_multiplier=1000,
                )
                await trainer.train(handle_signals=False)

                latest_step = await model.get_step()
                final_eval_samples = [
                    sample
                    for sample in samples
                    if sample.split == "eval" and sample.step == latest_step
                ]
                final_eval_reward = (
                    _mean_reward(final_eval_samples) if final_eval_samples else None
                )
                if final_eval_reward is None:
                    final_eval_start = len(samples)
                    await _length_group(
                        model,
                        scenario=_scenario(latest_step),
                        model_name=model.get_inference_name(step=latest_step),
                        split="eval",
                        step=latest_step,
                        n=eval_rollouts,
                        temperature=0.0,
                        samples=samples,
                    )
                    final_eval_reward = _mean_reward(samples[final_eval_start:])
                model_ids_after = await _list_model_ids(model)

    train_samples = [sample for sample in samples if sample.split == "train"]
    report = LengthTrainabilityReport(
        base_model=base_model,
        max_steps_off_policy=max_steps_off_policy,
        max_steps=max_steps,
        latest_step=latest_step,
        trainer_gpu_ids=trainer_gpu_ids,
        inference_gpu_ids=inference_gpu_ids,
        training_topology=TRAINING_TOPOLOGY.model_dump(),
        inference_engine_args=engine_args,
        runtime_command=command,
        runtime_log_path=str(runtime_log_path),
        baseline_final_target_reward=baseline_final_target_reward,
        final_eval_reward=final_eval_reward,
        model_ids_after=model_ids_after,
        samples=samples,
    )
    (artifact_dir / "external_vllm_length_trainability.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    assert latest_step >= max_steps
    assert train_samples
    assert all(sample.max_tokens > sample.target_tokens for sample in train_samples)
    assert any(sample.generated_tokens < sample.max_tokens for sample in train_samples)
    assert final_eval_reward is not None
    assert f"{model.name}@{latest_step}" in model_ids_after
