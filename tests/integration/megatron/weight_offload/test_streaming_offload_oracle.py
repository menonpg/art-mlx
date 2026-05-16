from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import pytest

from art.megatron.training.streaming_weight_offload import StreamingWeightOffloadConfig

from ..model_support.oracle_harness import (
    LIVE_TRAINING_LOG_PATH,
    MetricThresholdRule,
    PhasePassFn,
    Topology,
    VariantRunner,
    VariantSpec,
    available_gpu_count,
    case_config,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
STREAMING_OFFLOAD_LOG_PATH = REPO_ROOT / ".local" / "streaming_weight_offload.log"
STREAMING_OFFLOAD_TOPOLOGY = Topology(tp=1, ep=2, etp=1, dp=1, cp=2, sp=False)


def _run_with_log(*, log_path: Path, run: Callable[[], object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LIVE_TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_TRAINING_LOG_PATH.write_text("", encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            run()


def _exact_phase_pass_fns() -> dict[str, PhasePassFn]:
    exact_tensor = MetricThresholdRule(
        limits={"mean_abs_diff": 0.0, "relative_l2": 0.0, "mean_abs_pct": 0.0}
    )
    exact_topk = MetricThresholdRule(
        limits={"topk_mismatch_fraction": 0.0, "top1_mismatch_fraction": 0.0}
    )
    return {
        "forward": exact_tensor,
        "outputs": exact_tensor,
        "losses": exact_tensor,
        "grads": exact_tensor,
        "deltas": exact_tensor,
        "router_scores": exact_tensor,
        "router_topk_ids": exact_topk,
    }


def test_streaming_weight_offload_matches_no_offload_oracle(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with capsys.disabled():
        print(f"\nStreaming weight offload oracle log: {STREAMING_OFFLOAD_LOG_PATH}")
        print(f"Megatron live training log: {LIVE_TRAINING_LOG_PATH}")

    gpu_count = available_gpu_count()
    required_gpus = STREAMING_OFFLOAD_TOPOLOGY.world_size()
    if gpu_count < required_gpus:
        STREAMING_OFFLOAD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        STREAMING_OFFLOAD_LOG_PATH.write_text(
            (
                "Streaming weight offload oracle skipped. "
                f"Need {required_gpus} GPUs, found {gpu_count}.\n"
            ),
            encoding="utf-8",
        )
        pytest.skip(
            "Need "
            f"{required_gpus} GPUs for streaming weight offload oracle, found {gpu_count}"
        )

    config = case_config().model_copy(update={"precision": "bf16", "num_layers": 8})
    runner = VariantRunner(
        case_config=config,
        oracle_topology_override=STREAMING_OFFLOAD_TOPOLOGY,
        oracle_slug_override="rl__cp2_ep2_no_streaming_weight_offload",
        oracle_offload_between_jobs=True,
        oracle_streaming_weight_offload=StreamingWeightOffloadConfig(enabled=False),
        oracle_flex_backend=None,
        variant_flex_backend=None,
    )
    variant = VariantSpec(
        name="streaming_weight_offload_resident2_slots4",
        topology=STREAMING_OFFLOAD_TOPOLOGY,
        output_slug="rl__cp2_ep2_streaming_weight_offload_resident2_slots4",
        reference_slug=runner.oracle_slug,
        pass_fn_by_phase=_exact_phase_pass_fns(),
        offload_between_jobs=True,
        streaming_weight_offload=StreamingWeightOffloadConfig(
            enabled=True,
            num_layers=8,
            resident_layers=2,
            num_slots=4,
        ),
    )

    _run_with_log(
        log_path=STREAMING_OFFLOAD_LOG_PATH,
        run=lambda: runner.run_suite([variant]),
    )
