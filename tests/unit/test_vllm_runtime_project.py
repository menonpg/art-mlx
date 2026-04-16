from pathlib import Path
from typing import Any, cast

import pytest

import art.vllm.runtime_project as runtime_project
from art.vllm.runtime_project import (
    build_dedicated_vllm_server_cmd,
    get_vllm_runtime_project_root,
    wait_for_dedicated_vllm_server,
)


def test_get_vllm_runtime_project_root_defaults_to_repo_subdir(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ART_VLLM_RUNTIME_PROJECT_ROOT", raising=False)
    runtime_root = get_vllm_runtime_project_root()
    assert runtime_root.name == "vllm_runtime"
    assert runtime_root == Path(__file__).resolve().parents[2] / "vllm_runtime"


def test_get_vllm_runtime_project_root_honors_override(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ART_VLLM_RUNTIME_PROJECT_ROOT", "/tmp/custom-runtime")
    assert get_vllm_runtime_project_root() == Path("/tmp/custom-runtime")


def test_build_dedicated_vllm_server_cmd_uses_runtime_project(monkeypatch) -> None:
    monkeypatch.setenv("ART_VLLM_RUNTIME_PROJECT_ROOT", "/tmp/custom-runtime")
    cmd = build_dedicated_vllm_server_cmd(
        base_model="Qwen/Qwen3-14B",
        port=8000,
        host="127.0.0.1",
        cuda_visible_devices="1",
        lora_path="/tmp/lora",
        served_model_name="test@0",
        rollout_weights_mode="merged",
        engine_args={"weight_transfer_config": {"backend": "nccl"}},
        server_args={"tool_call_parser": "hermes"},
    )
    assert cmd[:5] == [
        "uv",
        "run",
        "--project",
        "/tmp/custom-runtime",
        "art-vllm-dedicated-server",
    ]
    assert "--model=Qwen/Qwen3-14B" in cmd
    assert '--engine-args-json={"weight_transfer_config": {"backend": "nccl"}}' in cmd
    assert '--server-args-json={"tool_call_parser": "hermes"}' in cmd


@pytest.mark.asyncio
async def test_wait_for_dedicated_vllm_server_uses_vllm_server_process(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    class FakeServerProcess:
        _server_process: object

        def __init__(
            self,
            server_cmd: list[str],
            after_bench_cmd: list[str],
            *,
            show_stdout: bool,
        ) -> None:
            seen["server_cmd"] = server_cmd
            seen["after_bench_cmd"] = after_bench_cmd
            seen["show_stdout"] = show_stdout

        def wait_until_ready(self, timeout: int) -> None:
            seen["timeout"] = timeout
            seen["process"] = self._server_process

    async def fake_to_thread(func, *args):
        return func(*args)

    process = cast(Any, object())
    monkeypatch.setattr(
        runtime_project,
        "_get_server_process_class",
        lambda: FakeServerProcess,
    )
    monkeypatch.setattr(runtime_project.asyncio, "to_thread", fake_to_thread)

    await wait_for_dedicated_vllm_server(
        process=process,
        host="127.0.0.1",
        port=8123,
        timeout=1200.1,
    )

    assert seen == {
        "server_cmd": [
            "vllm",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "8123",
        ],
        "after_bench_cmd": [],
        "show_stdout": False,
        "timeout": 1201,
        "process": process,
    }
