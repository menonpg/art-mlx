import json
import os
from pathlib import Path
from typing import Literal


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
