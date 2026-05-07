from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import pytest

from .oracle_harness import (
    ORACLE_TOPOLOGY,
    SENSITIVITY_MUTATION_ENV,
    available_gpu_count,
    case_config,
    run_sensitivity_suite,
    run_suite,
    sensitivity_enabled,
    sensitivity_mutations,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CORRECTNESS_LOG_PATH = REPO_ROOT / ".local" / "correctness.log"
SENSITIVITY_LOG_PATH = REPO_ROOT / ".local" / "sensitivity.log"


def _run_suite_with_log(
    *,
    log_path: Path,
    run: Callable[[], object],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def _require_gpus_for(topology_world_size: int) -> None:
    gpu_count = available_gpu_count()
    if gpu_count < topology_world_size:
        pytest.skip(
            f"Need {topology_world_size} GPUs for topology run, only found {gpu_count}"
        )


def test_megatron_lora_topology_suite(capsys: pytest.CaptureFixture[str]) -> None:
    """
    Runs the suite of topologies and expects each to pass (numerical differences within our thresholds)
    """
    _announce_report_log(log_path=CORRECTNESS_LOG_PATH, capsys=capsys)
    gpu_count = available_gpu_count()
    if gpu_count < ORACLE_TOPOLOGY.world_size():
        CORRECTNESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CORRECTNESS_LOG_PATH.write_text(
            (
                "Topology suite skipped. "
                f"Need {ORACLE_TOPOLOGY.world_size()} GPUs, found {gpu_count}.\n"
            ),
            encoding="utf-8",
        )
    _require_gpus_for(ORACLE_TOPOLOGY.world_size())
    _run_suite_with_log(
        log_path=CORRECTNESS_LOG_PATH,
        run=lambda: run_suite(
            case_config=case_config(),
            max_world_size=gpu_count,
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
    gpu_count = available_gpu_count()
    if gpu_count < ORACLE_TOPOLOGY.world_size():
        SENSITIVITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        SENSITIVITY_LOG_PATH.write_text(
            (
                "Sensitivity suite skipped. "
                f"Need {ORACLE_TOPOLOGY.world_size()} GPUs, found {gpu_count}.\n"
            ),
            encoding="utf-8",
        )
    _require_gpus_for(ORACLE_TOPOLOGY.world_size())
    _run_suite_with_log(
        log_path=SENSITIVITY_LOG_PATH,
        run=lambda: run_sensitivity_suite(
            case_config=case_config(),
            mutations=mutations,
            max_world_size=gpu_count,
        ),
    )
