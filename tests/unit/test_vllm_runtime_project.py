from pathlib import Path

from art.vllm.runtime_project import (
    build_dedicated_vllm_server_cmd,
    get_vllm_runtime_project_root,
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
