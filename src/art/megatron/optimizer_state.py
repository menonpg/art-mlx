from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir

ALLOW_UNPAIRED_MEGATRON_RESUME_ENV = "ART_ALLOW_UNPAIRED_MEGATRON_RESUME"


class MegatronResumeStep(BaseModel):
    step: int
    latest_lora_step: int
    optimizer_step: int | None
    used_unpaired_override: bool = False


def write_optimizer_step_marker(optimizer_state_path: str, step: int) -> None:
    path = Path(optimizer_state_path)
    path.mkdir(parents=True, exist_ok=True)
    for marker in path.iterdir():
        if marker.is_file() and marker.name.isdigit():
            marker.unlink()
    (path / f"{step:04d}").touch()


def read_optimizer_step_marker(optimizer_state_path: str) -> int | None:
    path = Path(optimizer_state_path)
    if not path.exists():
        return None
    return max(
        (
            int(marker.name)
            for marker in path.iterdir()
            if marker.is_file() and marker.name.isdigit()
        ),
        default=None,
    )


def _allow_unpaired_resume() -> bool:
    return os.environ.get(ALLOW_UNPAIRED_MEGATRON_RESUME_ENV, "").lower() in {
        "1",
        "true",
        "yes",
    }


def resolve_megatron_resume_step(
    *,
    output_dir: str,
    optimizer_state_path: str,
) -> MegatronResumeStep:
    latest_lora_step = get_step_from_dir(output_dir)
    optimizer_step = read_optimizer_step_marker(optimizer_state_path)
    if latest_lora_step == 0:
        return MegatronResumeStep(
            step=0,
            latest_lora_step=latest_lora_step,
            optimizer_step=optimizer_step,
        )
    if optimizer_step is not None and os.path.isdir(
        get_step_checkpoint_dir(output_dir, optimizer_step)
    ):
        return MegatronResumeStep(
            step=optimizer_step,
            latest_lora_step=latest_lora_step,
            optimizer_step=optimizer_step,
        )
    if _allow_unpaired_resume():
        return MegatronResumeStep(
            step=latest_lora_step,
            latest_lora_step=latest_lora_step,
            optimizer_step=optimizer_step,
            used_unpaired_override=True,
        )
    marker = (
        "no optimizer step marker"
        if optimizer_step is None
        else f"optimizer marker step {optimizer_step:04d} has no matching LoRA checkpoint"
    )
    raise RuntimeError(
        "Cannot resume Megatron training from an unpaired LoRA/optimizer state: "
        f"latest LoRA checkpoint is {latest_lora_step:04d}, {marker}. "
        f"Set {ALLOW_UNPAIRED_MEGATRON_RESUME_ENV}=1 to override."
    )
