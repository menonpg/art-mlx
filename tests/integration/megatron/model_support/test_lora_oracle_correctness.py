from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import pytest

from .oracle_harness import (
    LIVE_TRAINING_LOG_PATH,
    SENSITIVITY_MUTATION_ENV,
    TEST_DEFAULT_FLEX_BACKEND,
    available_gpu_count,
    case_config,
    oracle_topology,
    run_sensitivity_suite,
    run_suite,
    sensitivity_enabled,
    sensitivity_mutations,
    sensitivity_required_world_size,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CORRECTNESS_LOG_PATH = REPO_ROOT / ".local" / "correctness.log"
SENSITIVITY_LOG_PATH = REPO_ROOT / ".local" / "sensitivity.log"
TEST_FLEX_BACKEND = TEST_DEFAULT_FLEX_BACKEND


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
        print(f"\nMegatron LoRA oracle report log: {log_path}", flush=True)
        print(
            f"Megatron LoRA live training log: {LIVE_TRAINING_LOG_PATH}",
            flush=True,
        )


def test_megatron_lora_topology_suite(capsys: pytest.CaptureFixture[str]) -> None:
    """
    Runs the suite of topologies and expects each to pass (numerical differences within our thresholds)
    """
    _announce_report_log(log_path=CORRECTNESS_LOG_PATH, capsys=capsys)
    config = case_config()
    topology = oracle_topology(is_moe=config.is_moe)
    gpu_count = available_gpu_count()
    if gpu_count < topology.world_size():
        CORRECTNESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CORRECTNESS_LOG_PATH.write_text(
            (
                "Topology suite skipped. "
                f"Need {topology.world_size()} GPUs, found {gpu_count}.\n"
            ),
            encoding="utf-8",
        )
        pytest.skip(
            f"Need {topology.world_size()} GPUs for topology run, only found {gpu_count}"
        )
    _run_suite_with_log(
        log_path=CORRECTNESS_LOG_PATH,
        run=lambda: run_suite(
            case_config=config,
            max_world_size=gpu_count,
            oracle_flex_backend=TEST_FLEX_BACKEND,
            variant_flex_backend=TEST_FLEX_BACKEND,
        ),
    )


def test_megatron_lora_diff_sensitivity(capsys: pytest.CaptureFixture[str]) -> None:
    """
    Runs a each of the sensitivity mutations (e.g. drop megatron finalize grads)
    and expects each to fail (numerical differences larger than our thresholds)

    This test ensures we can catch errors we know of (implying we will be able to catch unknown errors as well)
    """
    _announce_report_log(log_path=SENSITIVITY_LOG_PATH, capsys=capsys)
    if not sensitivity_enabled():
        SENSITIVITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        SENSITIVITY_LOG_PATH.write_text(
            (
                "Sensitivity suite skipped. "
                f"Set {SENSITIVITY_MUTATION_ENV}=all (or one mutation / CSV).\n"
            ),
            encoding="utf-8",
        )
        pytest.skip(
            f"Set {SENSITIVITY_MUTATION_ENV}=all (or one mutation / CSV) to enable sensitivity check."
        )
    mutations = sensitivity_mutations()
    assert mutations
    config = case_config()
    sensitivity_world_size = sensitivity_required_world_size(
        mutations,
        is_moe=config.is_moe,
    )
    gpu_count = available_gpu_count()
    if gpu_count < sensitivity_world_size:
        SENSITIVITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        SENSITIVITY_LOG_PATH.write_text(
            (
                "Sensitivity suite skipped. "
                f"Need {sensitivity_world_size} GPUs, found {gpu_count}.\n"
            ),
            encoding="utf-8",
        )
        pytest.skip(
            f"Need {sensitivity_world_size} GPUs for topology run, only found {gpu_count}"
        )
    _run_suite_with_log(
        log_path=SENSITIVITY_LOG_PATH,
        run=lambda: run_sensitivity_suite(
            case_config=config,
            mutations=mutations,
            oracle_flex_backend=TEST_FLEX_BACKEND,
            variant_flex_backend=TEST_FLEX_BACKEND,
        ),
    )
