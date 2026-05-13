from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from .output_parity import (
    TOP_K,
    EngineSide,
    ScoreBundle,
    TokenTopK,
    WeightState,
    aggregate_mean_abs_pct,
    build_logical_token_map,
    compare_rollout,
    compare_topk,
    config_from_env,
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


def test_aggregate_mean_abs_pct_uses_vllm_merge_formula() -> None:
    summary = aggregate_mean_abs_pct(
        candidate=torch.tensor([2.0, 4.0]),
        target=torch.tensor([1.0, 3.0]),
        sequence_ids=[0, 0],
    )

    assert summary.source_numel == 2
    assert summary.trimmed_numel == 0
    assert summary.mean_abs_pct == pytest.approx((2.0 / 4.0) * 100.0)


def test_aggregate_mean_abs_pct_does_not_trim_or_average_sequence_summaries() -> None:
    target = torch.ones(80)
    candidate = target.clone()
    candidate[0] = 101.0
    candidate[1] = 51.0
    candidate[2] = 26.0
    candidate[3] = 2.0

    summary = aggregate_mean_abs_pct(
        candidate=candidate,
        target=target,
        sequence_ids=[0] * 40 + [1] * 40,
    )

    assert summary.source_numel == 80
    assert summary.sequence_count == 2
    assert summary.trimmed_numel == 0
    assert summary.mean_abs_pct == pytest.approx((176.0 / 80.0) * 100.0)


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


def test_compare_topk_reports_restricted_intersection_kl() -> None:
    target = ScoreBundle(
        side="megatron",
        weight_state="base",
        target_logprobs=[0.0],
        topk=[
            TokenTopK(
                token_ids=[10, 11],
                logprobs=[math.log(0.75), math.log(0.25)],
            )
        ],
    )
    candidate = ScoreBundle(
        side="vllm",
        weight_state="base",
        target_logprobs=[0.0],
        topk=[
            TokenTopK(
                token_ids=[10, 11],
                logprobs=[math.log(0.5), math.log(0.5)],
            )
        ],
    )

    report = compare_topk(candidate, target)

    assert report.top20_intersection_kl_target_to_candidate == pytest.approx(
        0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    )
    assert report.top20_intersection_kl_candidate_to_target == pytest.approx(
        0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(0.5 / 0.25)
    )


def test_config_from_env_accepts_lora_target_module_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ART_TRAIN_INF_MISMATCH_LORA_TARGET_MODULES",
        "experts,in_proj_qkv,in_proj_z",
    )

    config = config_from_env()

    assert config.lora_target_modules == ["experts", "in_proj_qkv", "in_proj_z"]
