from __future__ import annotations

from contextlib import asynccontextmanager
import math
import os
from pathlib import Path
import random
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
BASE_PROMPT = (
    "Write a plain answer about a quiet harbor. Use the unrelated notes below "
    "only as background texture. Use one sentence. Do not use bullets, numbering, "
    "code, or a preface."
)
FILLER_SENTENCES = (
    "The morning ledger mentioned a bicycle bell near the old customs window.",
    "A folded receipt waited beside three dull pencils and a chipped mug.",
    "Someone had drawn a small square around Thursday on the calendar.",
    "The storage room smelled faintly of rope, dust, and yesterday's rain.",
    "A green notebook listed errands that no one seemed eager to finish.",
    "The clock above the doorway ticked with a patient mechanical rhythm.",
    "Two mismatched gloves rested under the bench near the umbrella stand.",
    "A paper tag fluttered from a crate of spare brass hinges.",
    "The shop radio murmured about traffic far from the waterfront.",
    "A narrow envelope contained a map with several coffee stains.",
    "The caretaker had stacked clean towels beside a basket of loose keys.",
    "A faded poster advertised a lecture about practical knot repairs.",
    "Someone left a blue scarf draped over the back of a wooden chair.",
    "The rain gauge showed a modest line from a storm before dawn.",
    "A quiet clerk sorted stamps into a tin marked for later use.",
    "The window latch clicked softly whenever a colder breeze arrived.",
    "A jar of buttons sat near the lamp with no label attached.",
    "The floorboards held a faint shine where people usually turned left.",
    "A postcard showed a bridge, though no bridge could be seen nearby.",
    "The supply shelf included chalk, twine, soap, and several blank cards.",
    "A small toolbox waited open with every socket arranged by size.",
    "The notice board carried old schedules with careful handwritten corrections.",
    "A kettle cooled on the counter beside a plate of plain biscuits.",
    "The narrow hallway displayed framed photographs of ordinary cloudy afternoons.",
    "A stack of forms leaned against a vase holding one dry reed.",
    "The back office kept a spare lantern wrapped in brown paper.",
    "A silver whistle hung from a nail beside the maintenance checklist.",
    "The cupboard door closed unevenly unless pressed near the lower hinge.",
    "A receipt book recorded purchases of candles, nails, and black ink.",
    "The stair rail felt smooth where many hands had passed over it.",
    "A shallow drawer contained string, labels, and a forgotten measuring tape.",
    "The wall map used faded pins to mark unimportant delivery stops.",
    "A wool cap lay on a crate beside a coil of clean line.",
    "The afternoon light made the dust above the desk look almost orderly.",
    "A clipboard noted that the north window should be painted soon.",
    "The brass hook near the door held only an empty canvas bag.",
    "A stack of newspapers waited under a stone used as a weight.",
    "The broom leaned in a corner beside a cardboard box of washers.",
    "A shallow bowl held wrapped peppermints for visitors who rarely arrived.",
    "The gray filing cabinet opened with a scrape and a small sigh.",
    "A pencil sharpener was screwed to the wall beside a crooked shelf.",
    "The old ledger contained careful columns and very little useful drama.",
    "A canvas cover protected the spare chair from dust and sunlight.",
    "The side table held a ruler, a thimble, and a sealed jar.",
    "A neat row of jars preserved screws sorted by uncertain categories.",
    "The calendar showed local holidays in red and market days in blue.",
    "A small bell above the entrance moved only when the door stuck.",
    "The envelope tray was empty except for a note about lamp oil.",
    "The desk drawer included a spare button and two brittle rubber bands.",
    "A plain brown box carried the words archive later in pencil.",
)


class LengthScenario(BaseModel):
    scenario_index: int
    target_step: int
    target_tokens: int
    max_tokens: int
    prompt: str
    prompt_word_count: int
    metadata: dict[str, int] = Field(default_factory=dict)


class LengthSampleReport(BaseModel):
    split: Literal["train", "eval", "baseline"]
    step: int | None
    scenario_index: int
    target_step: int
    target_tokens: int
    max_tokens: int
    prompt_word_count: int
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
    summary_log_path: str
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


def _target_tokens_for_step(step: int) -> int:
    return _env_int("ART_EXTERNAL_VLLM_LENGTH_TARGET_START", 16) + step * _env_int(
        "ART_EXTERNAL_VLLM_LENGTH_TARGET_INCREMENT",
        4,
    )


def _word_count(text: str) -> int:
    return len(text.split())


def _check_prompt_hides_target(prompt: str) -> None:
    lowered = prompt.lower()
    forbidden = ("generated tokens", "target tokens", "target length", "exactly")
    leaked = [phrase for phrase in forbidden if phrase in lowered]
    if leaked:
        raise RuntimeError(f"Length prompt leaks target wording: {leaked}")


def _prompt_for_index(index: int) -> tuple[str, int]:
    target_words = _env_int("ART_EXTERNAL_VLLM_LENGTH_PROMPT_WORDS", 300)
    rng = random.Random(index)
    sentences = list(FILLER_SENTENCES)
    rng.shuffle(sentences)
    selected: list[str] = []
    prompt = BASE_PROMPT
    for sentence in sentences:
        if _word_count(prompt) >= target_words:
            break
        selected.append(sentence)
        prompt = f"{BASE_PROMPT}\n\nNotes: {' '.join(selected)}"
    _check_prompt_hides_target(prompt)
    return prompt, _word_count(prompt)


def _scenario(index: int, *, target_step: int | None = None) -> LengthScenario:
    resolved_target_step = index if target_step is None else target_step
    target_tokens = _target_tokens_for_step(resolved_target_step)
    max_tokens = max(
        target_tokens + 1,
        math.ceil(
            target_tokens
            * _env_float("ART_EXTERNAL_VLLM_LENGTH_MAX_TOKENS_MULTIPLIER", 1.4)
        )
        + 128,
    )
    prompt, prompt_word_count = _prompt_for_index(index)
    return LengthScenario(
        scenario_index=index,
        target_step=resolved_target_step,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        prompt=prompt,
        prompt_word_count=prompt_word_count,
        metadata={
            "scenario_index": index,
            "target_step": resolved_target_step,
            "target_tokens": target_tokens,
            "max_tokens": max_tokens,
            "prompt_word_count": prompt_word_count,
        },
    )


def _scenario_input(index: int, *, target_step: int = 0) -> dict[str, object]:
    return _scenario(index, target_step=target_step).model_dump()


async def _scenario_iter(count: int) -> AsyncIterator[dict[str, object]]:
    for index in range(count):
        yield _scenario_input(index)


def _scenario_for_training_step(
    scenario: LengthScenario | dict[str, object],
    step: int,
) -> LengthScenario:
    parsed = LengthScenario.model_validate(scenario)
    return _scenario(parsed.scenario_index, target_step=step)


def _step_from_model_name(model_name: str) -> int | None:
    if "@" not in model_name:
        return None
    try:
        return int(model_name.rsplit("@", 1)[1])
    except ValueError:
        return None


def _messages(scenario: LengthScenario) -> art.Messages:
    return [
        {
            "role": "user",
            "content": scenario.prompt,
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
        target_step=scenario.target_step,
        target_tokens=scenario.target_tokens,
        max_tokens=scenario.max_tokens,
        prompt_word_count=scenario.prompt_word_count,
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
    summary_log_path: Path | None = None,
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
                    "length/prompt_word_count": scenario.prompt_word_count,
                    "length/abs_error": report.abs_error,
                },
                metadata={
                    "target_step": scenario.target_step,
                    "target_tokens": scenario.target_tokens,
                    "max_tokens": scenario.max_tokens,
                    "prompt_word_count": scenario.prompt_word_count,
                    "scenario_index": scenario.scenario_index,
                },
            )
        )
    _append_step_summary(summary_log_path, samples, split=split, step=step)
    return art.TrajectoryGroup(trajectories)


def _mean_reward(samples: list[LengthSampleReport]) -> float:
    return sum(sample.reward for sample in samples) / max(1, len(samples))


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _init_summary_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "# external vLLM length trainability summary",
                "# rows append when a rollout/eval group completes; n is cumulative for split+step",
                (
                    "split      step target max_tok prompt_w     n reward_mean "
                    "gen_mean abs_err_mean gen_min gen_max reward_min reward_max"
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _append_step_summary(
    path: Path | None,
    samples: list[LengthSampleReport],
    *,
    split: Literal["train", "eval", "baseline"],
    step: int | None,
) -> None:
    if path is None:
        return
    matching = [
        sample for sample in samples if sample.split == split and sample.step == step
    ]
    if not matching:
        return
    generated = [float(sample.generated_tokens) for sample in matching]
    abs_errors = [float(sample.abs_error) for sample in matching]
    rewards = [sample.reward for sample in matching]
    latest = matching[-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{split:<9} {step if step is not None else '-':>4} "
            f"{latest.target_tokens:>6} {latest.max_tokens:>7} "
            f"{latest.prompt_word_count:>8} {len(matching):>5} "
            f"{_mean(rewards):>11.4f} {_mean(generated):>8.1f} "
            f"{_mean(abs_errors):>12.1f} {int(min(generated)):>7} "
            f"{int(max(generated)):>7} {min(rewards):>10.4f} "
            f"{max(rewards):>10.4f}\n"
        )


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
    rollouts_per_prompt = _env_int("ART_EXTERNAL_VLLM_LENGTH_ROLLOUTS_PER_PROMPT", 4)
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
    summary_log_path = artifact_dir / "external_vllm_length_trainability.log"
    _init_summary_log(summary_log_path)
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
                    summary_log_path=summary_log_path,
                )
                baseline_final_target_reward = _mean_reward(
                    samples[baseline_samples_before:]
                )

                async def rollout_fn(
                    rollout_model: art.TrainableModel,
                    scenario: dict[str, object],
                    _config: None,
                ) -> art.TrajectoryGroup:
                    model_name = rollout_model.get_inference_name()
                    target_step = _step_from_model_name(model_name)
                    if target_step is None:
                        target_step = await rollout_model.get_step()
                    step_scenario = _scenario_for_training_step(
                        scenario,
                        target_step,
                    )
                    return await _length_group(
                        rollout_model,
                        scenario=step_scenario,
                        model_name=model_name,
                        split="train",
                        step=target_step,
                        n=rollouts_per_prompt,
                        temperature=_env_float(
                            "ART_EXTERNAL_VLLM_LENGTH_ROLLOUT_TEMPERATURE",
                            1.1,
                        ),
                        samples=samples,
                        summary_log_path=summary_log_path,
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
                            summary_log_path=summary_log_path,
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
                        summary_log_path=summary_log_path,
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
        summary_log_path=str(summary_log_path),
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
