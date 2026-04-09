from pathlib import Path

import pytest

from .megatron_hf_parity import HF_PARITY_ENABLE_ENV, hf_parity_enabled, run_hf_parity
from .megatron_oracle_harness import available_gpu_count, case_config

HF_PARITY_LOG_PATH = Path(__file__).resolve().parents[2] / ".local" / "hf_parity.log"


def test_megatron_hf_sft_parity() -> None:
    if not hf_parity_enabled():
        HF_PARITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        HF_PARITY_LOG_PATH.write_text(
            f"HF parity skipped. Set {HF_PARITY_ENABLE_ENV}=1 to enable.\n",
            encoding="utf-8",
        )
        pytest.skip(f"Set {HF_PARITY_ENABLE_ENV}=1 to enable HF parity.")
    if available_gpu_count() < 1:
        HF_PARITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        HF_PARITY_LOG_PATH.write_text(
            "HF parity skipped. Need at least 1 GPU.\n",
            encoding="utf-8",
        )
        pytest.skip("Need at least 1 GPU for HF parity.")
    report = run_hf_parity(
        case_config=case_config(base_model="Qwen/Qwen3.5-35B-A3B"),
    )
    HF_PARITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    HF_PARITY_LOG_PATH.write_text(
        f"HF parity report: {report.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    assert report.signal == "pass"
