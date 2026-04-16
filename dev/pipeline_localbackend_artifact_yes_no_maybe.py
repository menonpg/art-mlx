"""Resume a dedicated LocalBackend PipelineTrainer run from a W&B LoRA artifact.

This script is intended to validate the `PipelineTrainer + LocalBackend`
infrastructure on two GPUs:

- GPU 0: trainer
- GPU 1: inference (vLLM)

It stages a W&B LoRA artifact into ART's local checkpoint layout as step 0,
registers a dedicated `LocalBackend`, verifies step-0 inference supports
logprobs, and then runs a small yes-no-maybe RL loop.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from functools import partial
from itertools import cycle, permutations
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any

from dotenv import load_dotenv
import torch
import wandb

import art
from art.local import LocalBackend
from art.pipeline_trainer import PipelineTrainer
from art.utils.output_dirs import get_model_dir, get_step_checkpoint_dir

WEIGHT_FILE_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
)
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value!r}")


def _get_env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if value is None:
        return default
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid GPU list for {name}: {value!r}")
    return [int(part) for part in parts]


def _with_quotes(word: str) -> str:
    return f"'{word}'"


def build_prompts() -> list[str]:
    prompts: list[str] = []
    for prefix in ["respond", "just respond"]:
        for use_quotes in [True, False]:
            for length in [3, 2]:
                for words in permutations(["yes", "no", "maybe"], length):
                    rendered = [_with_quotes(word) if use_quotes else word for word in words]
                    if length == 3:
                        suffix = ", ".join(rendered)
                    else:
                        suffix = f"{rendered[0]} or {rendered[1]}"
                    prompts.append(f"{prefix} with {suffix}")
    return prompts


def first_word_for_answer(content: str | None) -> str:
    if not content:
        return ""
    words = content.strip().lower().split(maxsplit=1)
    if not words:
        return ""
    return re.sub(r"^[^a-z]+|[^a-z]+$", "", words[0])


def reward_for_answer(content: str | None) -> float:
    first_word = first_word_for_answer(content)
    if first_word == "yes":
        return 0.5
    if first_word == "no":
        return 0.75
    if first_word == "maybe":
        return 1.0
    return 0.0


def scenario_id_for_prompt(prompt: str) -> str:
    return prompt.replace(" ", "_").replace("'", "")


def build_scenarios(prompts: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": prompt,
            "metadata": {"scenario_id": scenario_id_for_prompt(prompt)},
        }
        for prompt in prompts
    ]


def build_internal_config() -> art.dev.InternalModelConfig:
    trainer_gpu_ids = _get_env_int_list("TRAINER_GPU_IDS", [0])
    inference_gpu_ids = _get_env_int_list("INFERENCE_GPU_IDS", [1])
    if not trainer_gpu_ids:
        raise ValueError("TRAINER_GPU_IDS must not be empty")
    if not inference_gpu_ids:
        raise ValueError("INFERENCE_GPU_IDS must not be empty")
    if set(trainer_gpu_ids) & set(inference_gpu_ids):
        raise ValueError("TRAINER_GPU_IDS and INFERENCE_GPU_IDS must be disjoint")

    init_args = art.dev.InitArgs(
        max_seq_length=int(os.environ.get("MAX_SEQ_LENGTH", "2048")),
        load_in_4bit=_get_env_bool("LOAD_IN_4BIT", False),
        load_in_16bit=_get_env_bool("LOAD_IN_16BIT", True),
    )
    engine_args = art.dev.EngineArgs(
        gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.4")),
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "2048")),
        max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "16")),
        enforce_eager=_get_env_bool("ENFORCE_EAGER", True),
    )
    config = art.dev.InternalModelConfig(
        trainer_gpu_ids=trainer_gpu_ids,
        inference_gpu_ids=inference_gpu_ids,
        engine_args=engine_args,
        init_args=init_args,
        rollout_weights_mode=os.environ.get("ROLLOUT_WEIGHTS_MODE", "lora"),
    )
    return config


def _find_adapter_root(download_root: Path) -> Path:
    candidates: list[Path] = []
    for config_path in download_root.rglob("adapter_config.json"):
        candidate = config_path.parent
        if any((candidate / name).exists() for name in WEIGHT_FILE_NAMES):
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"No PEFT adapter root found under {download_root}. "
            f"Expected adapter_config.json plus one of {WEIGHT_FILE_NAMES!r}."
        )

    return min(candidates, key=lambda path: (len(path.relative_to(download_root).parts), str(path)))


def _summarize_tensor_dtypes(checkpoint_dir: Path) -> dict[str, int] | None:
    safetensors_path = checkpoint_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        from safetensors import safe_open

        counts: Counter[str] = Counter()
        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                counts[str(handle.get_tensor(key).dtype)] += 1
        return dict(sorted(counts.items()))

    bin_path = checkpoint_dir / "adapter_model.bin"
    if bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")  # type: ignore[call-arg]
        counts = Counter(str(tensor.dtype) for tensor in state_dict.values())
        return dict(sorted(counts.items()))

    return None


def summarize_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    adapter_config = (
        json.loads(adapter_config_path.read_text()) if adapter_config_path.exists() else None
    )
    files = sorted(
        str(path.relative_to(checkpoint_dir))
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
    )
    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_base_model": (
            adapter_config.get("base_model_name_or_path")
            if isinstance(adapter_config, dict)
            else None
        ),
        "weight_files": [name for name in WEIGHT_FILE_NAMES if (checkpoint_dir / name).exists()],
        "tensor_dtypes": _summarize_tensor_dtypes(checkpoint_dir),
        "files": files,
        "adapter_config": adapter_config,
    }
    return summary


def stage_wandb_artifact_as_step_zero(
    *,
    model: art.TrainableModel,
    art_path: str,
    artifact_name: str,
    artifact_type: str = "lora",
    force_download: bool = False,
) -> Path:
    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError(
            "WANDB_API_KEY is required to download the starting LoRA artifact."
        )

    model_dir = Path(get_model_dir(model, art_path=art_path))
    checkpoint_dir = Path(get_step_checkpoint_dir(str(model_dir), 0))
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) and not force_download:
        print(f"Using existing local checkpoint at {checkpoint_dir}")
        return checkpoint_dir

    api = wandb.Api(api_key=os.environ["WANDB_API_KEY"])
    artifact = api.artifact(artifact_name, type=artifact_type)

    temp_root = Path(tempfile.mkdtemp(prefix="art_resume_artifact_"))
    download_root = Path(artifact.download(root=str(temp_root)))
    adapter_root = _find_adapter_root(download_root)

    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    shutil.copytree(adapter_root, checkpoint_dir)

    summary = summarize_checkpoint(checkpoint_dir)
    print(
        json.dumps(
            {
                "artifact": artifact_name,
                "artifact_type": artifact_type,
                "aliases": list(getattr(artifact, "aliases", []) or []),
                "metadata": artifact.metadata,
                "download_root": str(download_root),
                "adapter_root": str(adapter_root),
                "staged_checkpoint": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return checkpoint_dir


async def assert_chat_logprobs(
    client: Any,
    model_name: str,
    *,
    timeout: float,
) -> None:
    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Say hello."}],
        model=model_name,
        max_tokens=8,
        timeout=timeout,
        logprobs=True,
        top_logprobs=0,
    )
    if completion.choices[0].logprobs is None:
        raise RuntimeError(f"logprobs were not returned for {model_name}")


async def rollout_fn(
    model: art.TrainableModel,
    scenario: dict[str, Any],
    _config: None,
    *,
    rollouts_per_prompt: int,
    max_tokens: int,
    timeout: float,
    temperature: float,
) -> art.TrajectoryGroup:
    messages: art.Messages = [{"role": "user", "content": scenario["prompt"]}]
    response = await model.openai_client().chat.completions.create(
        messages=messages,
        model=model.get_inference_name(),
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
        n=rollouts_per_prompt,
        logprobs=True,
        top_logprobs=0,
    )
    return art.TrajectoryGroup(
        [
            art.Trajectory(
                messages_and_choices=[*messages, choice],
                reward=reward_for_answer(choice.message.content),
            )
            for choice in response.choices
        ]
    )


async def eval_fn(
    model: art.TrainableModel,
    step: int,
    _config: None,
    *,
    scenarios: list[dict[str, Any]],
    max_tokens: int,
    timeout: float,
) -> list[art.TrajectoryGroup]:
    groups: list[art.TrajectoryGroup] = []
    for scenario in scenarios:
        messages: art.Messages = [{"role": "user", "content": scenario["prompt"]}]
        response = await model.openai_client().chat.completions.create(
            messages=messages,
            model=model.get_inference_name(step=step),
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=0,
            n=1,
        )
        choice = response.choices[0]
        group = art.TrajectoryGroup(
            [
                art.Trajectory(
                    messages_and_choices=[*messages, choice],
                    reward=reward_for_answer(choice.message.content),
                )
            ]
        )
        group.metadata["scenario_id"] = scenario["metadata"]["scenario_id"]
        groups.append(group)
    return groups


def history_path_for(model: art.TrainableModel, art_path: str) -> Path:
    return Path(get_model_dir(model, art_path=art_path)) / "history.jsonl"


async def main() -> None:
    load_dotenv()

    art_path = os.environ.get("ART_PATH", ".art")
    base_model = os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL)
    project = os.environ.get("PROJECT", "pipeline-localbackend-artifact-yes-no-maybe")
    model_name = os.environ.get(
        "MODEL_NAME",
        f"pipeline-localbackend-artifact-ynm-{int(time.time())}",
    )
    artifact_name = os.environ.get("WANDB_ARTIFACT")
    if not artifact_name:
        raise RuntimeError("WANDB_ARTIFACT must be set to a W&B LoRA artifact path.")
    force_download = _get_env_bool("FORCE_DOWNLOAD_ARTIFACT", False)

    backend = LocalBackend(path=art_path)
    model = art.TrainableModel(
        name=model_name,
        project=project,
        base_model=base_model,
        report_metrics=["wandb"],
        _internal_config=build_internal_config(),
    )

    client: Any | None = None
    try:
        checkpoint_dir = stage_wandb_artifact_as_step_zero(
            model=model,
            art_path=art_path,
            artifact_name=artifact_name,
            force_download=force_download,
        )
        checkpoint_summary = summarize_checkpoint(checkpoint_dir)
        adapter_base_model = checkpoint_summary.get("adapter_base_model")
        if adapter_base_model and adapter_base_model != base_model:
            print(
                "Warning: artifact adapter base model differs from requested base model: "
                f"{adapter_base_model!r} != {base_model!r}"
            )

        await model.register(backend)
        if wandb_run := model._get_wandb_run():
            print(f"W&B run: {wandb_run.url}")

        client = model.openai_client()
        timeout = float(os.environ.get("TIMEOUT", "120"))
        await assert_chat_logprobs(
            client,
            model.get_inference_name(step=0),
            timeout=timeout,
        )
        print(f"Verified step-0 logprobs for {model.get_inference_name(step=0)}")

        prompts = build_prompts()
        scenarios = build_scenarios(prompts)
        eval_scenarios = scenarios[: int(os.environ.get("EVAL_PROMPTS", "12"))]
        trainer = PipelineTrainer(
            model=model,
            backend=backend,
            rollout_fn=partial(
                rollout_fn,
                rollouts_per_prompt=int(os.environ.get("ROLLOUTS_PER_PROMPT", "8")),
                max_tokens=int(os.environ.get("MAX_TOKENS", "5")),
                timeout=timeout,
                temperature=float(os.environ.get("TEMPERATURE", "1.0")),
            ),
            scenarios=cycle(scenarios),
            config=None,
            eval_fn=partial(
                eval_fn,
                scenarios=eval_scenarios,
                max_tokens=int(os.environ.get("EVAL_MAX_TOKENS", "5")),
                timeout=timeout,
            ),
            num_rollout_workers=int(os.environ.get("NUM_ROLLOUT_WORKERS", "4")),
            min_batch_size=int(os.environ.get("MIN_BATCH_SIZE", "4")),
            max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "4")),
            max_steps=int(os.environ.get("NUM_STEPS", "20")),
            max_steps_off_policy=int(os.environ.get("MAX_STEPS_OFF_POLICY", "4")),
            learning_rate=float(os.environ.get("LEARNING_RATE", "5e-5")),
            loss_fn=os.environ.get("LOSS_FN", "cispo"),
            eval_every_n_steps=int(os.environ.get("EVAL_EVERY_N_STEPS", "2")),
            eval_step_0=_get_env_bool("EVAL_STEP_0", True),
            total_scenarios=None,
        )

        print(
            json.dumps(
                {
                    "project": project,
                    "model_name": model_name,
                    "base_model": base_model,
                    "artifact_name": artifact_name,
                    "checkpoint_dir": str(checkpoint_dir),
                    "num_scenarios": len(scenarios),
                    "num_eval_scenarios": len(eval_scenarios),
                    "num_steps": int(os.environ.get("NUM_STEPS", "20")),
                    "rollouts_per_prompt": int(os.environ.get("ROLLOUTS_PER_PROMPT", "8")),
                    "num_rollout_workers": int(os.environ.get("NUM_ROLLOUT_WORKERS", "4")),
                    "min_batch_size": int(os.environ.get("MIN_BATCH_SIZE", "4")),
                    "max_batch_size": int(os.environ.get("MAX_BATCH_SIZE", "4")),
                    "learning_rate": float(os.environ.get("LEARNING_RATE", "5e-5")),
                    "history_path": str(history_path_for(model, art_path)),
                },
                indent=2,
                sort_keys=True,
            )
        )

        await trainer.train()
        final_step = await model.get_step()
        print(f"Training finished at step {final_step}")
        print(f"History: {history_path_for(model, art_path)}")
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                maybe_awaitable = close()
                if asyncio.iscoroutine(maybe_awaitable):
                    await maybe_awaitable
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
