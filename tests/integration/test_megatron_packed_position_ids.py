from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")

from .megatron_packed_position_ids import run_packed_position_ids


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for packed position id validation",
)
def test_run_packed_position_ids_qwen35() -> None:
    report = run_packed_position_ids(
        base_model="Qwen/Qwen3.5-35B-A3B",
    )

    assert len(report.scenarios) == 2
    assert all(scenario.matched for scenario in report.scenarios)
    assert all(scenario.checked_token_count > 0 for scenario in report.scenarios)
    assert all(scenario.prompt_family_count >= 2 for scenario in report.scenarios)
    assert all(scenario.rotary_grouping_checked for scenario in report.scenarios)
    assert all(
        scenario.repeated_position_key_count > 0 for scenario in report.scenarios
    )
    assert all(scenario.completion_pair_count > 0 for scenario in report.scenarios)
    assert all(scenario.logits_mean_abs_pct <= 0.1 for scenario in report.scenarios)
