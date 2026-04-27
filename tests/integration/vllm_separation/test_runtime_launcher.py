from pathlib import Path

import pytest

import art.vllm_runtime as runtime


ROOT = Path(__file__).resolve().parents[3]


def test_get_vllm_runtime_project_root_defaults_to_repo_subdir(monkeypatch) -> None:
    monkeypatch.delenv("ART_VLLM_RUNTIME_PROJECT_ROOT", raising=False)
    runtime_root = runtime.get_vllm_runtime_project_root()
    assert runtime_root == ROOT / "vllm_runtime"


def test_get_vllm_runtime_project_root_honors_override(monkeypatch) -> None:
    monkeypatch.setenv("ART_VLLM_RUNTIME_PROJECT_ROOT", "/tmp/custom-runtime")
    assert runtime.get_vllm_runtime_project_root() == Path("/tmp/custom-runtime")


def test_build_runtime_server_cmd_uses_runtime_project(monkeypatch) -> None:
    monkeypatch.delenv("ART_VLLM_RUNTIME_BIN", raising=False)
    monkeypatch.setenv("ART_VLLM_RUNTIME_PROJECT_ROOT", "/tmp/custom-runtime")
    command = runtime.build_vllm_runtime_server_cmd(
        runtime.VllmRuntimeLaunchConfig(
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
    )
    assert command[:5] == [
        "uv",
        "run",
        "--project",
        "/tmp/custom-runtime",
        "art-vllm-runtime-server",
    ]
    assert "--model=Qwen/Qwen3-14B" in command
    assert '--engine-args-json={"weight_transfer_config": {"backend": "nccl"}}' in command
    assert '--server-args-json={"tool_call_parser": "hermes"}' in command


@pytest.mark.asyncio
async def test_wait_for_vllm_runtime_polls_http_health(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class FakeProcess:
        def poll(self):
            return None

    class FakeResponse:
        status_code = 200

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, timeout: float):
            seen["url"] = url
            seen["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setattr(runtime.httpx, "AsyncClient", lambda: FakeClient())
    await runtime.wait_for_vllm_runtime(
        process=FakeProcess(),
        host="127.0.0.1",
        port=8123,
        timeout=12.0,
    )
    assert seen == {
        "url": "http://127.0.0.1:8123/health",
        "timeout": 5.0,
    }
