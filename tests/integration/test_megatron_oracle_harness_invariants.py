import torch

from .megatron_oracle_harness import (
    DENSE_ORACLE_TOPOLOGY,
    MAX_WORLD_SIZE_ENV,
    ORACLE_TOPOLOGY,
    DiffAccumulator,
    MetricThresholdRule,
    _default_phase_pass_fns,
    _suite_variants,
    selected_suite_topologies,
)


def test_metric_threshold_rule_can_require_strictly_positive_values() -> None:
    rule = MetricThresholdRule(minimums={"candidate_abs_scale": 0.0})

    summary = {"candidate_abs_scale": 0.0}

    assert not rule(summary)
    assert rule.failure_reasons(summary) == ["candidate_abs_scale=0<=0"]


def test_diff_accumulator_summary_tracks_candidate_abs_scale() -> None:
    accumulator = DiffAccumulator()

    accumulator.update(
        torch.tensor([1.0, -2.0], dtype=torch.float32),
        torch.tensor([0.5, 0.0], dtype=torch.float32),
    )

    summary = accumulator.as_summary()

    assert summary["typical_abs_scale"] == 1.5
    assert summary["candidate_abs_scale"] == 0.25


def test_default_phase_rules_require_non_zero_forward_outputs_grads_and_deltas() -> (
    None
):
    phase_pass = _default_phase_pass_fns()
    zero_signal_summary = {
        "relative_l2": 0.0,
        "mean_abs_pct": 0.0,
        "typical_abs_scale": 0.0,
        "candidate_abs_scale": 0.0,
    }

    assert not phase_pass["forward"](zero_signal_summary)
    assert not phase_pass["outputs"](zero_signal_summary)
    assert not phase_pass["grads"](zero_signal_summary)
    assert not phase_pass["deltas"](zero_signal_summary)
    assert phase_pass["losses"](zero_signal_summary)


def test_suite_variants_skip_duplicate_oracle_replay_variant() -> None:
    variants = _suite_variants("rl", is_moe=True)

    assert variants
    assert all(variant.topology != ORACLE_TOPOLOGY for variant in variants)
    assert all("oracle_replay" not in variant.name for variant in variants)


def test_dense_suite_variants_include_tp2_dp2_without_oracle_duplicate() -> None:
    variants = _suite_variants("rl", is_moe=False)

    assert variants
    assert all(variant.topology != DENSE_ORACLE_TOPOLOGY for variant in variants)
    assert any(
        variant.topology.tp == 2 and variant.topology.dp == 2 for variant in variants
    )


def test_max_world_size_env_filters_dense_topologies(monkeypatch) -> None:
    monkeypatch.setenv(MAX_WORLD_SIZE_ENV, "2")

    topologies = selected_suite_topologies(is_moe=False)

    assert topologies
    assert all(topology.world_size() <= 2 for topology in topologies)
    assert not any(topology.tp == 2 and topology.dp == 2 for topology in topologies)
