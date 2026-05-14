from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import pytest

from .oracle_harness import (
    LIVE_TRAINING_LOG_PATH,
    TEST_DEFAULT_FLEX_BACKEND,
    available_gpu_count,
    case_config,
    oracle_topology,
    run_fp8_base_weight_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
FP8_CORRECTNESS_LOG_PATH = REPO_ROOT / ".local" / "fp8_base_weight_correctness.log"
TEST_FLEX_BACKEND = TEST_DEFAULT_FLEX_BACKEND
QWEN35_MOE_BASE_MODEL = "Qwen/Qwen3.5-35B-A3B"


def _run_suite_with_log(
    *,
    log_path: Path,
    run: Callable[[], object],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LIVE_TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_TRAINING_LOG_PATH.write_text("", encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            run()


def _announce_report_log(
    *,
    log_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with capsys.disabled():
        print(f"\nMegatron FP8 base-weight oracle report log: {log_path}", flush=True)
        print(
            f"Megatron FP8 base-weight live training log: {LIVE_TRAINING_LOG_PATH}",
            flush=True,
        )


def test_megatron_qwen35_fp8_base_weights_single_rank_oracle(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compares single-rank FP8 base weights + BF16 LoRA against the BF16 oracle."""
    _announce_report_log(log_path=FP8_CORRECTNESS_LOG_PATH, capsys=capsys)
    config = case_config(base_model=QWEN35_MOE_BASE_MODEL).model_copy(
        update={"precision": "bf16"}
    )
    topology = oracle_topology(is_moe=config.is_moe)
    gpu_count = available_gpu_count()
    if gpu_count < topology.world_size():
        FP8_CORRECTNESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        FP8_CORRECTNESS_LOG_PATH.write_text(
            (
                "FP8 base-weight oracle suite skipped. "
                f"Need {topology.world_size()} GPUs, found {gpu_count}.\n"
            ),
            encoding="utf-8",
        )
        pytest.skip(
            f"Need {topology.world_size()} GPUs for FP8 base-weight oracle, only found {gpu_count}"
        )
    _run_suite_with_log(
        log_path=FP8_CORRECTNESS_LOG_PATH,
        run=lambda: run_fp8_base_weight_suite(
            case_config=config,
            oracle_flex_backend=TEST_FLEX_BACKEND,
            variant_flex_backend=TEST_FLEX_BACKEND,
        ),
    )
