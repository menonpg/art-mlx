import torch

from .megatron_oracle_harness import (
    ORACLE_TOPOLOGY,
    DiffAccumulator,
    MetricThresholdRule,
    _default_phase_pass_fns,
    _suite_variants,
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
    variants = _suite_variants("rl")

    assert variants
    assert all(variant.topology != ORACLE_TOPOLOGY for variant in variants)
    assert all("oracle_replay" not in variant.name for variant in variants)
