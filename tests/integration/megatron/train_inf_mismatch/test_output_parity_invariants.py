from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from .output_parity import (
    TOP_K,
    EngineSide,
    ScoreBundle,
    TokenTopK,
    WeightState,
    build_logical_token_map,
    compare_rollout,
    sequence_mean_abs_pct,
)


def test_logical_map_flattens_shared_prefix_branches() -> None:
    packed = {
        "tokens": torch.tensor([[10, 11, 12, 13, 14, 12, 15, 16]]),
        "group_ids": torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]]),
        "parent_ids": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]]),
    }

    logical_map = build_logical_token_map(packed)

    assert [prompt.token_ids for prompt in logical_map.prompts] == [
        [10, 11, 12, 13, 14],
        [10, 11, 12, 15, 16],
    ]
    assert [token.token_id for token in logical_map.tokens] == [13, 14, 15, 16]
    assert [token.art_logit_index for token in logical_map.tokens] == [2, 3, 5, 6]
    assert [token.vllm_prompt_token_index for token in logical_map.tokens] == [
        3,
        4,
        3,
        4,
    ]


def test_sequence_mean_abs_pct_uses_elementwise_support_branch_formula() -> None:
    summary = sequence_mean_abs_pct(
        candidate=torch.tensor([0.5, 0.0]),
        target=torch.tensor([1.0, -2.0]),
        sequence_ids=[0, 0],
    )

    assert summary.source_numel == 2
    assert summary.trimmed_numel == 0
    assert summary.mean_abs_pct == pytest.approx(
        ((0.5 / 1.0) + (2.0 / 2.0)) / 2 * 100.0
    )


def test_sequence_mean_abs_pct_trims_top_three_per_sequence() -> None:
    target = torch.ones(40)
    candidate = target.clone()
    candidate[0] = 101.0
    candidate[1] = 51.0
    candidate[2] = 26.0
    candidate[3] = 2.0

    summary = sequence_mean_abs_pct(
        candidate=candidate,
        target=target,
        sequence_ids=[0] * 40,
    )

    assert summary.source_numel == 40
    assert summary.trimmed_numel == 3
    assert summary.mean_abs_pct == pytest.approx((1.0 / 37) * 100.0)


def test_sequence_mean_abs_pct_averages_sequence_summaries() -> None:
    target = torch.ones(80)
    candidate = target.clone()
    candidate[0] = 101.0
    candidate[1] = 51.0
    candidate[2] = 26.0
    candidate[3] = 2.0

    summary = sequence_mean_abs_pct(
        candidate=candidate,
        target=target,
        sequence_ids=[0] * 40 + [1] * 40,
    )

    assert summary.source_numel == 80
    assert summary.trimmed_numel == 6
    assert summary.mean_abs_pct == pytest.approx(((1.0 / 37) * 100.0) / 2)


def _score(
    values: list[float],
    *,
    side: EngineSide,
    state: WeightState,
) -> ScoreBundle:
    return ScoreBundle(
        side=side,
        weight_state=state,
        target_logprobs=values,
        topk=[
            TokenTopK(
                token_ids=list(range(TOP_K)),
                logprobs=[-float(index) for index in range(TOP_K)],
            )
            for _ in values
        ],
    )


def test_compare_rollout_reports_base_lora_and_delta_separately() -> None:
    packed = {
        "tokens": torch.tensor([[10, 11, 12, 13, 14]]),
        "group_ids": torch.tensor([[0, 0, 1, 1, 1]]),
        "parent_ids": torch.tensor([[0, 0, 0, 0, 0]]),
    }
    logical_map = build_logical_token_map(packed)

    report = compare_rollout(
        rollout_mode="native_lora",
        megatron_base=_score([-1.0, -2.0], side="megatron", state="base"),
        megatron_lora=_score([-1.5, -2.5], side="megatron", state="lora"),
        vllm_base=_score([-1.1, -2.2], side="vllm", state="base"),
        vllm_lora=_score([-1.7, -2.8], side="vllm", state="lora"),
        logical_map=logical_map,
    )

    assert report.base.mean_abs_pct > 0
    assert report.lora.mean_abs_pct > 0
    assert report.delta.mean_abs_pct > 0
