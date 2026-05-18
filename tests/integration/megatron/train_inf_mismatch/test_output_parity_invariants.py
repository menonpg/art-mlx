from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from . import workflow_stage
from .output_parity import (
    TOP_K,
    EngineSide,
    ScoreBundle,
    TokenTopK,
    Topology,
    TrainInfOutputParityConfig,
    WeightState,
    aggregate_mean_abs_pct,
    build_logical_token_map,
    build_vllm_routing_replay_bundle,
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


def test_vllm_routing_replay_bundle_maps_unpacked_routes_to_packed_rows() -> None:
    packed = {
        "tokens": torch.tensor([[10, 11, 12, 13, 14, 12, 15, 16]]),
        "group_ids": torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]]),
        "parent_ids": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]]),
    }
    logical_map = build_logical_token_map(packed)
    responses = {
        0: {
            "prompt_routed_experts": [
                [[1, 2], [11, 12]],
                [[3, 4], [13, 14]],
                [[5, 6], [15, 16]],
                [[7, 8], [17, 18]],
                [[9, 10], [19, 20]],
            ]
        },
        1: {
            "prompt_routed_experts": [
                [[1, 2], [11, 12]],
                [[3, 4], [13, 14]],
                [[21, 22], [31, 32]],
                [[23, 24], [33, 34]],
                [[25, 26], [35, 36]],
            ]
        },
    }

    bundle = build_vllm_routing_replay_bundle(
        packed_tensors=packed,
        logical_map=logical_map,
        responses_by_prompt=responses,
        topology=Topology(tp=2, ep=2),
    )

    layer0 = bundle.steps[0].routers["chunk_00.layer_0000.mlp.router"].calls[0]
    layer1 = bundle.steps[0].routers["chunk_00.layer_0001.mlp.router"].calls[0]
    assert layer0.expert_indices.tolist() == [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [21, 22],
        [23, 24],
        [25, 26],
    ]
    assert layer1.expert_indices.tolist() == [
        [11, 12],
        [13, 14],
        [15, 16],
        [17, 18],
        [19, 20],
        [31, 32],
        [33, 34],
        [35, 36],
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


def test_default_rollout_modes_follow_model_support_native_lora_status() -> None:
    assert TrainInfOutputParityConfig(
        base_model="Qwen/Qwen3.5-35B-A3B"
    ).rollout_modes == ["native_lora", "merged"]
    assert TrainInfOutputParityConfig(
        base_model="unvalidated/native-disabled",
        allow_unvalidated_arch=True,
    ).rollout_modes == ["merged"]


def test_config_from_env_rollout_modes_override_handler_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ART_TRAIN_INF_MISMATCH_BASE_MODEL",
        "unvalidated/native-disabled",
    )
    monkeypatch.setenv("ART_TRAIN_INF_MISMATCH_ALLOW_UNVALIDATED_ARCH", "1")
    monkeypatch.setenv("ART_TRAIN_INF_MISMATCH_ROLLOUT_MODES", "native_lora")

    config = config_from_env()

    assert config.rollout_modes == ["native_lora"]


def test_workflow_stage_enables_live_train_inf_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import subprocess

    captured_env = {}

    def fake_run(*args, **kwargs):
        captured_env.update(kwargs["env"])
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="1 passed\n",
            stderr="",
        )

    monkeypatch.setattr(workflow_stage, "create_artifact_dir", lambda _nodeid: tmp_path)
    monkeypatch.setattr(workflow_stage.subprocess, "run", fake_run)

    report = workflow_stage.run_train_inf_mismatch(base_model="Qwen/Qwen3.5-35B-A3B")

    assert report.passed is True
    assert captured_env["ART_RUN_TRAIN_INF_MISMATCH_LIVE"] == "1"
