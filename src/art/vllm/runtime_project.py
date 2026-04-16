import asyncio
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Literal


def get_vllm_runtime_project_root() -> Path:
    override = os.environ.get("ART_VLLM_RUNTIME_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3] / "vllm_runtime"


def build_dedicated_vllm_server_cmd(
    *,
    base_model: str,
    port: int,
    host: str,
    cuda_visible_devices: str,
    lora_path: str,
    served_model_name: str,
    rollout_weights_mode: Literal["lora", "merged"],
    engine_args: dict[str, object],
    server_args: dict[str, object],
) -> list[str]:
    runtime_project_root = get_vllm_runtime_project_root()
    return [
        "uv",
        "run",
        "--project",
        str(runtime_project_root),
        "art-vllm-dedicated-server",
        f"--model={base_model}",
        f"--port={port}",
        f"--host={host}",
        f"--cuda-visible-devices={cuda_visible_devices}",
        f"--lora-path={lora_path}",
        f"--served-model-name={served_model_name}",
        f"--rollout-weights-mode={rollout_weights_mode}",
        f"--engine-args-json={json.dumps(engine_args)}",
        f"--server-args-json={json.dumps(server_args)}",
    ]


def _get_server_process_class() -> type[Any]:
    from vllm.benchmarks.sweep.server import ServerProcess

    return ServerProcess


async def wait_for_dedicated_vllm_server(
    *,
    process: subprocess.Popen[Any],
    host: str,
    port: int,
    timeout: float,
) -> None:
    server_process_class = _get_server_process_class()
    waiter = server_process_class(
        server_cmd=["vllm", "serve", "--host", host, "--port", str(port)],
        after_bench_cmd=[],
        show_stdout=False,
    )
    # wait_until_ready() only needs the process handle and host/port metadata.
    setattr(waiter, "_server_process", process)
    await asyncio.to_thread(waiter.wait_until_ready, max(1, math.ceil(timeout)))
