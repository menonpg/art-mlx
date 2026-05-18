from __future__ import annotations

import os
from pathlib import Path

import pytest

from .real_path import (
    BF16_FWD_MEAN_ABS_PCT_LIMIT,
    config_from_env,
    run_real_path_train_inf_mismatch,
)

torch = pytest.importorskip("torch")

LIVE_ENV = "ART_RUN_TRAIN_INF_MISMATCH_REAL_PATH_LIVE"


def _require_live_opt_in() -> None:
    if os.environ.get(LIVE_ENV) != "1":
        pytest.skip(f"set {LIVE_ENV}=1 to run real-path train/inf mismatch")


def _require_visible_gpus(gpu_ids: list[int]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real-path train/inf mismatch")
    visible_count = int(torch.cuda.device_count())
    required = max(gpu_ids) + 1 if gpu_ids else 0
    if visible_count < required:
        pytest.skip(
            f"Need visible CUDA device ids through {required - 1}, "
            f"but torch sees {visible_count} devices"
        )


@pytest.mark.asyncio
async def test_real_path_train_inf_mismatch_live(artifact_dir: Path) -> None:
    _require_live_opt_in()
    config = config_from_env()
    parity_config = config.output_parity
    _require_visible_gpus(
        parity_config.trainer_gpu_ids + parity_config.inference_gpu_ids
    )

    report = await run_real_path_train_inf_mismatch(
        config=config,
        artifact_dir=artifact_dir,
    )

    assert report.logical_prompt_count > 0
    assert report.logical_token_count > 0
    assert report.moe_routing_packed_tokens > 0
    assert report.passed, report.model_dump_json(indent=2)
    assert report.lora.mean_abs_pct <= BF16_FWD_MEAN_ABS_PCT_LIMIT
