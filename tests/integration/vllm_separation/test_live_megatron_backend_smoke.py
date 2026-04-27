import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import AsyncIterator, cast
import uuid

import httpx
import pytest

import art
from art import dev
from art.megatron.backend import MegatronBackend
from art.megatron.service import MegatronService

from tests.integration.megatron_oracle_harness import ORACLE_TOPOLOGY, Topology
from tests.integration.megatron_oracle_worker import provider_topology_env
from tests.integration.megatron_yes_no_trainability import (
    _build_trainable_groups,
    _engine_args_for_yes_no_trainability,
    _evaluate_model,
    _wandb_disabled,
    _warmup_model,
    build_prompts,
)

torch = pytest.importorskip("torch")

DEFAULT_BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_MAX_SEQ_LENGTH = 128
DEFAULT_PACKED_SEQUENCE_LENGTH = 128
DEDICATED_MERGED_ENV = "ART_RUN_LIVE_MEGATRON_MERGED_SMOKE"
SHARED_LORA_ENV = "ART_RUN_LIVE_MEGATRON_SHARED_SMOKE"
SHARED_TOPOLOGY = Topology(tp=2, ep=1, etp=1, dp=1, sp=True)


def _base_model() -> str:
    return os.environ.get(
        "ART_LIVE_MEGATRON_BASE_MODEL",
        os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL),
    )


def _max_seq_length() -> int:
    return int(os.environ.get("ART_TEST_MAX_SEQ_LENGTH", str(DEFAULT_MAX_SEQ_LENGTH)))


def _packed_sequence_length() -> int:
    return int(
        os.environ.get(
            "ART_TEST_PACKED_SEQUENCE_LENGTH",
            str(DEFAULT_PACKED_SEQUENCE_LENGTH),
        )
    )


def _train_group_prompts() -> list[str]:
    prompt_count = int(os.environ.get("ART_TEST_MEGATRON_PROMPT_COUNT", "2"))
    return build_prompts()[: max(1, prompt_count)]


def _rollouts_per_prompt() -> int:
    return int(os.environ.get("ART_TEST_MEGATRON_ROLLOUTS_PER_PROMPT", "2"))


def _trainer_gpu_ids() -> list[int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Need at least 2 visible CUDA GPUs for Megatron live smokes")
    return [0]


def _inference_gpu_ids() -> list[int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Need at least 2 visible CUDA GPUs for Megatron live smokes")
    return [1]


def _require_opt_in(env_name: str) -> None:
    if os.environ.get(env_name) != "1":
        pytest.skip(f"set {env_name}=1 to run this live Megatron smoke")


def _shared_live_config() -> dev.InternalModelConfig:
    return {
        "rollout_weights_mode": "lora",
        "engine_args": {
            **_engine_args_for_yes_no_trainability(inference_gpu_ids=[0, 1]),
            "enable_sleep_mode": True,
        },
        "init_args": {"max_seq_length": _max_seq_length()},
    }


def _dedicated_merged_config() -> dev.InternalModelConfig:
    return {
        "trainer_gpu_ids": _trainer_gpu_ids(),
        "inference_gpu_ids": _inference_gpu_ids(),
        "rollout_weights_mode": "merged",
        "engine_args": {
            **_engine_args_for_yes_no_trainability(
                inference_gpu_ids=_inference_gpu_ids()
            ),
        },
        "init_args": {"max_seq_length": _max_seq_length()},
    }


async def _list_model_ids(model: art.TrainableModel) -> list[str]:
    client = model.openai_client()
    return [model_info.id async for model_info in client.models.list()]


async def _chat_snapshot(model: art.TrainableModel, *, step: int) -> dict[str, object]:
    client = model.openai_client()
    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Say hello."}],
        model=model.get_inference_name(step=step),
        max_tokens=8,
        timeout=180.0,
        logprobs=True,
        top_logprobs=0,
    )
    return {
        "text": completion.choices[0].message.content,
        "has_logprobs": completion.choices[0].logprobs is not None,
    }


async def _runtime_is_sleeping(service: MegatronService) -> bool:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{service._vllm_base_url}/is_sleeping")
        response.raise_for_status()
        return bool(response.json()["is_sleeping"])


async def _wait_until_runtime_sleeping(
    service: MegatronService,
    *,
    timeout_s: float = 300.0,
    poll_s: float = 0.5,
) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if await _runtime_is_sleeping(service):
            return True
        await asyncio.sleep(poll_s)
    return False


@asynccontextmanager
async def _megatron_backend_context(
    *,
    backend_root: Path,
    topology: Topology,
) -> AsyncIterator[MegatronBackend]:
    with _wandb_disabled():
        with provider_topology_env(topology):
            async with MegatronBackend(path=str(backend_root), in_process=True) as backend:
                yield backend


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs for Megatron live smokes",
)
@pytest.mark.asyncio
async def test_megatron_backend_shared_lora_runtime_sleep_wake_live_smoke(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_opt_in(SHARED_LORA_ENV)
    monkeypatch.setenv("ART_DISABLE_SERVER_MONITOR", "1")
    backend_root = artifact_dir / "art_workspace"
    backend_root.mkdir(parents=True, exist_ok=True)

    async with _megatron_backend_context(
        backend_root=backend_root,
        topology=SHARED_TOPOLOGY,
    ) as backend:
        model = art.TrainableModel(
            name=f"megatron-shared-live-{uuid.uuid4().hex[:8]}",
            project="integration-tests",
            base_model=_base_model(),
            _internal_config=_shared_live_config(),
            report_metrics=[],
        )
        await model.register(backend)
        service = cast(MegatronService, await backend._get_service(model))
        prompts = _train_group_prompts()
        await _warmup_model(model, base_model=model.base_model, prompt=prompts[0])
        step0_name = model.get_inference_name(step=0)
        model_ids_before = await _list_model_ids(model)
        train_groups = await _build_trainable_groups(
            model,
            base_model=model.base_model,
            prompts=prompts,
            rollouts_per_prompt=_rollouts_per_prompt(),
        )
        train_task = asyncio.create_task(
            backend.train(
                model,
                train_groups,
                learning_rate=float(os.environ.get("ART_TEST_MEGATRON_LR", "1e-4")),
                loss_fn="cispo",
                allow_training_without_logprobs=True,
                packed_sequence_length=_packed_sequence_length(),
            )
        )
        observed_sleep = False
        try:
            while not train_task.done():
                if await _runtime_is_sleeping(service):
                    observed_sleep = True
                    break
                await asyncio.sleep(0.5)
            assert observed_sleep or train_task.done()
            result = await train_task
        finally:
            if not train_task.done():
                await train_task

        latest_step = int(result.step)
        latest_name = model.get_inference_name(step=latest_step)
        model_ids_after = await _list_model_ids(model)
        eval_reward = await _evaluate_model(
            model,
            base_model=model.base_model,
            prompts=prompts,
            step=latest_step,
        )
        latest_snapshot = await _chat_snapshot(model, step=latest_step)
        runtime_sleep_after = await _runtime_is_sleeping(service)
        payload = {
            "base_model": model.base_model,
            "output_dir": service.output_dir,
            "step0_name": step0_name,
            "latest_name": latest_name,
            "latest_step": latest_step,
            "model_ids_before": model_ids_before,
            "model_ids_after": model_ids_after,
            "observed_sleep": observed_sleep,
            "runtime_sleep_after": runtime_sleep_after,
            "eval_reward": eval_reward,
            "latest_snapshot": latest_snapshot,
        }
        (artifact_dir / "shared_megatron_live_result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        assert observed_sleep
        assert runtime_sleep_after is False
        assert latest_step > 0
        assert step0_name in model_ids_before
        assert step0_name in model_ids_after
        assert latest_name in model_ids_after
        assert latest_snapshot["has_logprobs"] is True


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs for Megatron live smokes",
)
@pytest.mark.asyncio
async def test_megatron_backend_dedicated_merged_live_smoke(
    artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_opt_in(DEDICATED_MERGED_ENV)
    monkeypatch.setenv("ART_DISABLE_SERVER_MONITOR", "1")
    backend_root = artifact_dir / "art_workspace"
    backend_root.mkdir(parents=True, exist_ok=True)

    async with _megatron_backend_context(
        backend_root=backend_root,
        topology=ORACLE_TOPOLOGY,
    ) as backend:
        model = art.TrainableModel(
            name=f"megatron-merged-live-{uuid.uuid4().hex[:8]}",
            project="integration-tests",
            base_model=_base_model(),
            _internal_config=_dedicated_merged_config(),
            report_metrics=[],
        )
        await model.register(backend)
        service = cast(MegatronService, await backend._get_service(model))
        prompts = _train_group_prompts()
        await _warmup_model(model, base_model=model.base_model, prompt=prompts[0])
        step0_name = model.get_inference_name(step=0)
        model_ids_before = await _list_model_ids(model)
        train_groups = await _build_trainable_groups(
            model,
            base_model=model.base_model,
            prompts=prompts,
            rollouts_per_prompt=_rollouts_per_prompt(),
        )
        result = await backend.train(
            model,
            train_groups,
            learning_rate=float(os.environ.get("ART_TEST_MEGATRON_LR", "1e-4")),
            loss_fn="cispo",
            allow_training_without_logprobs=True,
            packed_sequence_length=_packed_sequence_length(),
        )
        latest_step = int(result.step)
        latest_name = model.get_inference_name(step=latest_step)
        model_ids_after = await _list_model_ids(model)
        eval_reward = await _evaluate_model(
            model,
            base_model=model.base_model,
            prompts=prompts,
            step=latest_step,
        )
        latest_snapshot = await _chat_snapshot(model, step=latest_step)
        payload = {
            "base_model": model.base_model,
            "output_dir": service.output_dir,
            "step0_name": step0_name,
            "latest_name": latest_name,
            "latest_step": latest_step,
            "model_ids_before": model_ids_before,
            "model_ids_after": model_ids_after,
            "eval_reward": eval_reward,
            "latest_snapshot": latest_snapshot,
        }
        (artifact_dir / "dedicated_megatron_merged_live_result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        assert latest_step > 0
        assert step0_name in model_ids_before
        assert latest_name in model_ids_after
        assert step0_name not in model_ids_after
        assert latest_snapshot["has_logprobs"] is True
