from __future__ import annotations

import asyncio
from contextlib import contextmanager
from itertools import permutations
import os
from pathlib import Path
import re
from typing import Iterator, cast
import uuid

from pydantic import BaseModel, Field
import torch

import art
from art import dev
from art.megatron.backend import MegatronBackend
from art.megatron.model_support.registry import get_model_support_spec

from .megatron_oracle_harness import ORACLE_TOPOLOGY
from .megatron_oracle_worker import provider_topology_env

_TRAINER_GPU_IDS_ENV = "ART_MODEL_SUPPORT_TRAINER_GPU_IDS"
_INFERENCE_GPU_IDS_ENV = "ART_MODEL_SUPPORT_INFERENCE_GPU_IDS"


def build_prompts() -> list[str]:
    prompt = os.environ.get("ART_MODEL_SUPPORT_YES_NO_PROMPT", "").strip()
    prompt_count = _get_env_int("ART_MODEL_SUPPORT_YES_NO_PROMPT_COUNT", 8)
    if prompt:
        return [prompt] * max(1, prompt_count)
    prompts = [
        f"{prefix} exactly one of {body}"
        for prefix in ("respond with", "just respond with")
        for use_quotes in (True, False)
        for length in (3, 2)
        for words in permutations(("yes", "no", "maybe"), length)
        for body in [
            ", ".join(f"'{word}'" if use_quotes else word for word in words)
            if length == 3
            else " or ".join(f"'{word}'" if use_quotes else word for word in words)
        ]
    ]
    if prompt_count <= len(prompts):
        return prompts[: max(1, prompt_count)]
    return [prompts[index % len(prompts)] for index in range(prompt_count)]


def _slugify(value: str) -> str:
    return value.lower().replace("/", "_").replace(".", "_").replace("-", "_")


def _artifact_dir(base_model: str) -> Path:
    root = Path(__file__).resolve().parents[2] / ".local" / "model_support_validation"
    path = root / _slugify(base_model) / "yes_no_trainability" / uuid.uuid4().hex[:8]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_gpu_id_env(name: str) -> list[int] | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _resolve_dedicated_gpu_ids() -> tuple[list[int], list[int]]:
    trainer_gpu_ids = _parse_gpu_id_env(_TRAINER_GPU_IDS_ENV)
    inference_gpu_ids = _parse_gpu_id_env(_INFERENCE_GPU_IDS_ENV)
    if trainer_gpu_ids is not None or inference_gpu_ids is not None:
        if trainer_gpu_ids is None or inference_gpu_ids is None:
            raise RuntimeError(
                f"{_TRAINER_GPU_IDS_ENV} and {_INFERENCE_GPU_IDS_ENV} must both be set"
            )
        return trainer_gpu_ids, inference_gpu_ids
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Need at least 2 visible CUDA GPUs for yes/no trainability")
    return [0], [1]


def _safe_gpu_memory_utilization(device_ids: list[int]) -> float:
    requested = float(
        os.environ.get("ART_MODEL_SUPPORT_YES_NO_GPU_MEMORY_UTILIZATION", "0.85")
    )
    min_free_gib = float(
        os.environ.get("ART_MODEL_SUPPORT_YES_NO_MIN_FREE_GPU_GIB", "8")
    )
    free_ratios: list[float] = []
    for device in sorted(set(device_ids)):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gib = free_bytes / (1024**3)
        if free_gib < min_free_gib:
            raise RuntimeError(
                f"GPU {device} has only {free_gib:.1f} GiB free < {min_free_gib:.1f} GiB required"
            )
        free_ratios.append(free_bytes / total_bytes)
    return max(0.02, min(requested, min(free_ratios) * 0.95))


def reward_for_answer(text: str) -> float:
    return {
        "yes": 0.5,
        "no": 0.75,
        "maybe": 1.0,
    }.get(first_word_for_answer(text).lower(), 0.0)


def first_word_for_answer(text: str | None) -> str:
    if not text:
        return ""
    stripped = re.sub(
        r"<think>.*?</think>\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    first_word = stripped.strip().split(maxsplit=1)
    if not first_word:
        return ""
    return first_word[0].strip(".,!?:;\"'()[]{}")


def _get_env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _get_env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _max_tokens() -> int:
    return _get_env_int("ART_MODEL_SUPPORT_YES_NO_MAX_TOKENS", 5)


def _render_chat_messages(base_model: str, prompt: str) -> art.Messages:
    del base_model
    return [{"role": "user", "content": prompt}]


def _enable_thinking() -> bool:
    return os.environ.get(
        "ART_MODEL_SUPPORT_YES_NO_ENABLE_THINKING", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _extra_body() -> dict[str, object]:
    return {"chat_template_kwargs": {"enable_thinking": _enable_thinking()}}


def _request_timeout(name: str, default: float) -> float:
    return _get_env_float(name, default)


def _engine_args_for_yes_no_trainability(
    *,
    inference_gpu_ids: list[int],
) -> dev.EngineArgs:
    return cast(
        dev.EngineArgs,
        {
            "gpu_memory_utilization": _safe_gpu_memory_utilization(inference_gpu_ids),
            "max_model_len": _get_env_int(
                "ART_MODEL_SUPPORT_YES_NO_MAX_MODEL_LEN", 128
            ),
            "max_num_seqs": _get_env_int("ART_MODEL_SUPPORT_YES_NO_MAX_NUM_SEQS", 4),
            "enforce_eager": True,
        },
    )


class TrainabilityStepReport(BaseModel):
    step: int
    eval_reward: float
    train_reward: float
    train_metrics: dict[str, float] = Field(default_factory=dict)


class YesNoTrainabilityReport(BaseModel):
    base_model: str
    output_dir: str
    trainer_gpu_ids: list[int]
    inference_gpu_ids: list[int]
    rollout_weights_mode: str
    reward_threshold: float
    max_steps: int
    prompt_count: int
    eval_prompt_count: int
    rollouts_per_prompt: int
    latest_step: int
    initial_eval_reward: float
    final_eval_reward: float | None = None
    saturated_step: int | None = None
    steps: list[TrainabilityStepReport] = Field(default_factory=list)


@contextmanager
def _wandb_disabled() -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in ("WANDB_API_KEY", "WANDB_MODE")}
    os.environ.pop("WANDB_API_KEY", None)
    os.environ["WANDB_MODE"] = "disabled"
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


async def _evaluate_model(
    model: art.TrainableModel,
    *,
    base_model: str,
    prompts: list[str],
    step: int,
) -> float:
    client = model.openai_client()
    rewards: list[float] = []
    for prompt in prompts:
        completion = await client.chat.completions.create(
            messages=_render_chat_messages(base_model, prompt),
            model=model.get_inference_name(step=step),
            max_tokens=_max_tokens(),
            extra_body=_extra_body(),
            temperature=_get_env_float(
                "ART_MODEL_SUPPORT_YES_NO_EVAL_TEMPERATURE",
                0.0,
            ),
            timeout=_request_timeout(
                "ART_MODEL_SUPPORT_YES_NO_EVAL_TIMEOUT",
                180.0,
            ),
        )
        rewards.append(reward_for_answer(completion.choices[0].message.content or ""))
    return sum(rewards) / len(rewards)


async def _build_training_groups(
    model: art.TrainableModel,
    *,
    base_model: str,
    prompts: list[str],
    rollouts_per_prompt: int,
) -> list[art.TrajectoryGroup]:
    client = model.openai_client()

    async def _group_for_prompt(prompt: str) -> art.TrajectoryGroup:
        messages = _render_chat_messages(base_model, prompt)
        completion = await client.chat.completions.create(
            messages=messages,
            model=model.get_inference_name(),
            max_tokens=_max_tokens(),
            n=rollouts_per_prompt,
            extra_body=_extra_body(),
            temperature=_get_env_float(
                "ART_MODEL_SUPPORT_YES_NO_ROLLOUT_TEMPERATURE",
                1.2,
            ),
            timeout=_request_timeout(
                "ART_MODEL_SUPPORT_YES_NO_ROLLOUT_TIMEOUT",
                180.0,
            ),
        )
        return art.TrajectoryGroup(
            [
                art.Trajectory(
                    messages_and_choices=[
                        *messages,
                        {
                            "role": "assistant",
                            "content": choice.message.content or "",
                        },
                    ],
                    reward=reward_for_answer(choice.message.content or ""),
                )
                for choice in completion.choices
            ]
        )

    return await art.gather_trajectory_groups(
        [_group_for_prompt(prompt) for prompt in prompts]  # ty: ignore[invalid-argument-type]
    )


def _group_has_reward_variance(group: art.TrajectoryGroup) -> bool:
    return len({trajectory.reward for trajectory in group.trajectories}) > 1


async def _build_trainable_groups(
    model: art.TrainableModel,
    *,
    base_model: str,
    prompts: list[str],
    rollouts_per_prompt: int,
) -> list[art.TrajectoryGroup]:
    max_attempts = _get_env_int("ART_MODEL_SUPPORT_YES_NO_MAX_ROLLOUT_ATTEMPTS", 4)
    for _ in range(max_attempts):
        groups = await _build_training_groups(
            model,
            base_model=base_model,
            prompts=prompts,
            rollouts_per_prompt=rollouts_per_prompt,
        )
        trainable_groups = [
            group for group in groups if _group_has_reward_variance(group)
        ]
        if trainable_groups:
            return trainable_groups
    raise RuntimeError(
        "No reward-variant trajectory groups were produced for yes/no trainability"
    )


async def _warmup_model(
    model: art.TrainableModel,
    *,
    base_model: str,
    prompt: str,
) -> None:
    client = model.openai_client()
    await client.chat.completions.create(
        messages=_render_chat_messages(base_model, prompt),
        model=model.get_inference_name(step=0),
        max_tokens=1,
        extra_body=_extra_body(),
        temperature=0.0,
        timeout=_request_timeout(
            "ART_MODEL_SUPPORT_YES_NO_WARMUP_TIMEOUT",
            900.0,
        ),
    )


async def _run_yes_no_trainability(base_model: str) -> YesNoTrainabilityReport:
    output_dir = _artifact_dir(base_model)
    trainer_gpu_ids, inference_gpu_ids = _resolve_dedicated_gpu_ids()
    reward_threshold = _get_env_float("ART_MODEL_SUPPORT_YES_NO_REWARD_THRESHOLD", 0.95)
    max_steps = _get_env_int("ART_MODEL_SUPPORT_YES_NO_MAX_STEPS", 4)
    rollouts_per_prompt = _get_env_int(
        "ART_MODEL_SUPPORT_YES_NO_ROLLOUTS_PER_PROMPT",
        4,
    )
    eval_prompt_count = _get_env_int("ART_MODEL_SUPPORT_YES_NO_EVAL_PROMPTS", 8)
    prompts = build_prompts()
    eval_prompts = prompts[:eval_prompt_count]
    spec = get_model_support_spec(base_model)
    packed_sequence_length = _get_env_int(
        "ART_MODEL_SUPPORT_YES_NO_PACKED_SEQUENCE_LENGTH",
        128,
    )
    internal_config = dev.InternalModelConfig(
        trainer_gpu_ids=trainer_gpu_ids,
        inference_gpu_ids=inference_gpu_ids,
        rollout_weights_mode=spec.default_rollout_weights_mode,
        engine_args=_engine_args_for_yes_no_trainability(
            inference_gpu_ids=inference_gpu_ids
        ),
        init_args={"max_seq_length": packed_sequence_length},
    )
    dev.validate_dedicated_config(internal_config)
    model = art.TrainableModel(
        name=f"model-support-trainability-{uuid.uuid4().hex[:8]}",
        project="model-support-validation",
        base_model=base_model,
        _internal_config=internal_config,
        report_metrics=[],
    )

    with _wandb_disabled():
        with provider_topology_env(ORACLE_TOPOLOGY):
            async with MegatronBackend(path=str(output_dir), in_process=True) as backend:
                print(
                    f"[yes_no_trainability] registering model in {output_dir}",
                    flush=True,
                )
                await model.register(backend)
                print("[yes_no_trainability] model registered", flush=True)
                print("[yes_no_trainability] warming inference path", flush=True)
                await _warmup_model(
                    model,
                    base_model=base_model,
                    prompt=prompts[0],
                )
                print("[yes_no_trainability] warmup complete", flush=True)
                initial_eval_reward = await _evaluate_model(
                    model,
                    base_model=base_model,
                    prompts=eval_prompts,
                    step=0,
                )
                print(
                    f"[yes_no_trainability] initial_eval_reward={initial_eval_reward:.4f}",
                    flush=True,
                )
                report = YesNoTrainabilityReport(
                    base_model=base_model,
                    output_dir=str(output_dir),
                    trainer_gpu_ids=trainer_gpu_ids,
                    inference_gpu_ids=inference_gpu_ids,
                    rollout_weights_mode=spec.default_rollout_weights_mode,
                    reward_threshold=reward_threshold,
                    max_steps=max_steps,
                    prompt_count=len(prompts),
                    eval_prompt_count=len(eval_prompts),
                    rollouts_per_prompt=rollouts_per_prompt,
                    latest_step=0,
                    initial_eval_reward=initial_eval_reward,
                )

                for _ in range(max_steps):
                    print("[yes_no_trainability] building train groups", flush=True)
                    train_groups = await _build_trainable_groups(
                        model,
                        base_model=base_model,
                        prompts=prompts,
                        rollouts_per_prompt=rollouts_per_prompt,
                    )
                    print("[yes_no_trainability] starting train step", flush=True)
                    result = await backend.train(
                        model,
                        train_groups,
                        learning_rate=_get_env_float(
                            "ART_MODEL_SUPPORT_YES_NO_LEARNING_RATE", 1e-4
                        ),
                        loss_fn="cispo",
                        allow_training_without_logprobs=True,
                        packed_sequence_length=packed_sequence_length,
                    )
                    print(
                        f"[yes_no_trainability] train step complete step={result.step}",
                        flush=True,
                    )
                    eval_reward = await _evaluate_model(
                        model,
                        base_model=base_model,
                        prompts=eval_prompts,
                        step=result.step,
                    )
                    print(
                        f"[yes_no_trainability] eval_reward={eval_reward:.4f} step={result.step}",
                        flush=True,
                    )
                    report.latest_step = int(result.step)
                    report.final_eval_reward = float(eval_reward)
                    report.steps.append(
                        TrainabilityStepReport(
                            step=int(result.step),
                            eval_reward=float(eval_reward),
                            train_reward=sum(
                                trajectory.reward
                                for group in train_groups
                                for trajectory in group.trajectories
                            )
                            / max(
                                1,
                                sum(len(group.trajectories) for group in train_groups),
                            ),
                            train_metrics={
                                key: float(value)
                                for key, value in result.metrics.items()
                                if isinstance(value, int | float)
                            },
                        )
                    )
                    if eval_reward >= reward_threshold:
                        report.saturated_step = int(result.step)
                        break
                return report


def run_yes_no_trainability(base_model: str) -> YesNoTrainabilityReport:
    report = asyncio.run(_run_yes_no_trainability(base_model))
    output_dir = Path(report.output_dir)
    (output_dir / "report.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return report
