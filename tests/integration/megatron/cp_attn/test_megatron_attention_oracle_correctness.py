from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import pytest

from ..model_support.oracle_harness import LIVE_TRAINING_LOG_PATH, available_gpu_count
from .megatron_attention_oracle_harness import (
    ATTN_SENSITIVITY_MUTATION_ENV,
    attention_case_config,
    attention_required_world_size,
    attention_sensitivity_enabled,
    attention_sensitivity_mutations,
    run_attention_sensitivity_suite,
    run_attention_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
ATTN_CORRECTNESS_LOG_PATH = REPO_ROOT / ".local" / "attention_correctness.log"
ATTN_SENSITIVITY_LOG_PATH = REPO_ROOT / ".local" / "attention_sensitivity.log"


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
        print(f"\nMegatron attention oracle report log: {log_path}", flush=True)
        print(
            f"Megatron attention live training log: {LIVE_TRAINING_LOG_PATH}",
            flush=True,
        )


def _require_gpus_for(topology_world_size: int) -> None:
    gpu_count = available_gpu_count()
    if gpu_count < topology_world_size:
        pytest.skip(
            f"Need {topology_world_size} GPUs for attention topology run, only found {gpu_count}"
        )


def test_megatron_attention_diff_sensitivity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _announce_report_log(log_path=ATTN_SENSITIVITY_LOG_PATH, capsys=capsys)
    if not attention_sensitivity_enabled():
        ATTN_SENSITIVITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ATTN_SENSITIVITY_LOG_PATH.write_text(
            (
                "Attention sensitivity suite skipped. "
                f"Set {ATTN_SENSITIVITY_MUTATION_ENV}=all (or one mutation / CSV).\n"
            ),
            encoding="utf-8",
        )
        pytest.skip(
            f"Set {ATTN_SENSITIVITY_MUTATION_ENV}=all (or one mutation / CSV) to enable attention sensitivity check."
        )
    mutations = attention_sensitivity_mutations()
    sensitivity_world_size = attention_required_world_size(mutations)
    _require_gpus_for(sensitivity_world_size)
    _run_suite_with_log(
        log_path=ATTN_SENSITIVITY_LOG_PATH,
        run=lambda: run_attention_sensitivity_suite(
            case_config=attention_case_config(),
            mutations=mutations,
        ),
    )


def test_megatron_attention_topology_suite(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _announce_report_log(log_path=ATTN_CORRECTNESS_LOG_PATH, capsys=capsys)
    gpu_count = available_gpu_count()
    if gpu_count < 2:
        ATTN_CORRECTNESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ATTN_CORRECTNESS_LOG_PATH.write_text(
            (
                "Attention topology suite skipped. "
                f"Need at least 2 GPUs, found {gpu_count}.\n"
            ),
            encoding="utf-8",
        )
    _require_gpus_for(2)
    _run_suite_with_log(
        log_path=ATTN_CORRECTNESS_LOG_PATH,
        run=lambda: run_attention_suite(
            case_config=attention_case_config(),
            max_world_size=gpu_count,
        ),
    )
