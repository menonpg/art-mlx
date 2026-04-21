from collections.abc import AsyncIterator
from pathlib import Path
import signal
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from art.megatron.jobs import MergedWeightTransferInitInfo, MergedWeightTransferSpec
from art.megatron.service import MegatronService
from art.types import TrainConfig


async def _empty_stream(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
    del args, kwargs
    if False:
        yield {}


@pytest.mark.asyncio
async def test_start_openai_server_syncs_initial_merged_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "trainer_gpu_ids": [0],
            "inference_gpu_ids": [1],
            "rollout_weights_mode": "merged",
        },
        output_dir=str(tmp_path),
    )
    start_vllm = AsyncMock(return_value=("127.0.0.1", 8000))
    sync_merged = AsyncMock()
    monkeypatch.setattr(service, "_resolve_active_lora_path", lambda: "/tmp/lora")
    monkeypatch.setattr(service, "_start_vllm_subprocess", start_vllm)
    monkeypatch.setattr(service, "_sync_dedicated_merged_weights", sync_merged)

    location = await service.start_openai_server(None)

    assert location == ("127.0.0.1", 8000)
    start_vllm.assert_awaited_once()
    sync_merged.assert_awaited_once_with(lora_path="/tmp/lora", step=0)


def test_resolve_active_lora_path_materializes_identity_adapter_for_merged_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "trainer_gpu_ids": [0],
            "inference_gpu_ids": [1],
            "rollout_weights_mode": "merged",
        },
        output_dir=str(tmp_path),
    )
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "art.megatron.service.get_last_checkpoint_dir",
        lambda _output_dir: None,
    )
    monkeypatch.setattr(
        service,
        "_ensure_identity_lora",
        lambda path: calls.append(("identity", path)),
    )
    monkeypatch.setattr(
        service,
        "_ensure_lora_adapter_config",
        lambda path, source_path=None: calls.append(("config", path)),
    )

    path = service._resolve_active_lora_path()

    assert path == str(tmp_path / "checkpoints" / "0000")
    assert calls == [("identity", path), ("config", path)]


@pytest.mark.asyncio
async def test_dedicated_train_uses_merged_job_and_updates_latest_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "trainer_gpu_ids": [0],
            "inference_gpu_ids": [1],
            "rollout_weights_mode": "merged",
        },
        output_dir=str(tmp_path),
    )
    seen_job: dict[str, Any] = {}

    async def _stream_job(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        del args, kwargs
        if False:
            yield {}

    monkeypatch.setattr(service, "_ensure_megatron_running", AsyncMock())
    monkeypatch.setattr(service, "_resolve_active_lora_path", lambda: "/tmp/lora")
    monkeypatch.setattr(service, "_clear_pending_jobs", lambda: None)
    monkeypatch.setattr(
        service,
        "_create_megatron_job_paths",
        lambda: ("/tmp/job.json", "/tmp/log.jsonl"),
    )
    monkeypatch.setattr(service, "_init_merged_weight_transfer", AsyncMock())
    monkeypatch.setattr(
        service,
        "_build_merged_weight_transfer_spec",
        lambda step: MergedWeightTransferSpec(
            init_info=MergedWeightTransferInitInfo(
                master_address="127.0.0.1",
                master_port=2345,
                rank_offset=1,
                world_size=2,
            ),
            vllm_base_url="http://127.0.0.1:8000",
            served_model_name=f"test-model@{step}",
        ),
    )
    monkeypatch.setattr(
        "art.megatron.service.write_megatron_job",
        lambda job, *, job_path: seen_job.update({"job": job, "job_path": job_path}),
    )
    monkeypatch.setattr("art.megatron.service.stream_megatron_job", _stream_job)
    monkeypatch.setattr("art.megatron.service.shutil.copy", lambda src, dst: None)
    monkeypatch.setattr(
        service,
        "_ensure_lora_adapter_config",
        lambda lora_path, source_path=None: None,
    )

    results = [
        result
        async for result in service.train(
            {"dir": "/tmp/packed", "num_sequences": 2, "sequence_length": 128},
            TrainConfig(
                learning_rate=1e-5,
                grad_accumulation_sequences=1,
            ),
            {},
        )
    ]

    assert results == []
    assert seen_job["job"].kind == "train_merged"
    assert service._latest_step == 1


def test_stop_megatron_process_kills_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "trainer_gpu_ids": [0],
            "inference_gpu_ids": [1],
            "rollout_weights_mode": "merged",
        },
        output_dir=str(tmp_path),
    )

    class _Process:
        pid = 4321
        returncode = None

    seen: dict[str, int] = {}
    monkeypatch.setattr("art.megatron.service.os.getpgid", lambda pid: pid + 1)
    monkeypatch.setattr(
        "art.megatron.service.os.killpg",
        lambda pgid, sig: seen.update({"pgid": pgid, "sig": int(sig)}),
    )
    service._megatron_process = cast(Any, _Process())

    service._stop_megatron_process()

    assert seen == {"pgid": 4322, "sig": int(signal.SIGTERM)}
    assert service._megatron_process is None


def test_stop_megatron_process_ignores_missing_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "trainer_gpu_ids": [0],
            "inference_gpu_ids": [1],
            "rollout_weights_mode": "merged",
        },
        output_dir=str(tmp_path),
    )

    class _Process:
        pid = 4321
        returncode = None

    monkeypatch.setattr("art.megatron.service.os.getpgid", lambda pid: pid)

    def _raise_process_lookup(pgid: int, sig: int) -> None:
        del pgid, sig
        raise ProcessLookupError

    monkeypatch.setattr("art.megatron.service.os.killpg", _raise_process_lookup)
    service._megatron_process = cast(Any, _Process())

    service._stop_megatron_process()

    assert service._megatron_process is None
