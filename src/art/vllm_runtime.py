import asyncio
import httpx
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VllmRuntimeLaunchConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_model: str
    port: int
    host: str = "127.0.0.1"
    cuda_visible_devices: str
    lora_path: str
    served_model_name: str
    rollout_weights_mode: Literal["lora", "merged"]
    engine_args: dict[str, object] = Field(default_factory=dict)
    server_args: dict[str, object] = Field(default_factory=dict)


def get_vllm_runtime_project_root() -> Path:
    override = os.environ.get("ART_VLLM_RUNTIME_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2] / "vllm_runtime"


def _runtime_command_prefix() -> list[str]:
    override = os.environ.get("ART_VLLM_RUNTIME_BIN")
    if override:
        return shlex.split(override)
    return [
        "uv",
        "run",
        "--project",
        str(get_vllm_runtime_project_root()),
        "art-vllm-runtime-server",
    ]


def build_vllm_runtime_server_cmd(config: VllmRuntimeLaunchConfig) -> list[str]:
    return [
        *_runtime_command_prefix(),
        f"--model={config.base_model}",
        f"--port={config.port}",
        f"--host={config.host}",
        f"--cuda-visible-devices={config.cuda_visible_devices}",
        f"--lora-path={config.lora_path}",
        f"--served-model-name={config.served_model_name}",
        f"--rollout-weights-mode={config.rollout_weights_mode}",
        f"--engine-args-json={json.dumps(config.engine_args)}",
        f"--server-args-json={json.dumps(config.server_args)}",
    ]


async def wait_for_vllm_runtime(
    *,
    process: subprocess.Popen[object],
    host: str,
    port: int,
    timeout: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    url = f"http://{host}:{port}/health"
    async with httpx.AsyncClient() as client:
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"vLLM runtime exited with code {process.returncode}"
                )
            try:
                response = await client.get(url, timeout=5.0)
                if response.status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"vLLM runtime did not become ready within {math.ceil(timeout)}s"
                )
            await asyncio.sleep(0.5)
