from __future__ import annotations

import random
from typing import Any, cast

from pydantic import BaseModel
import pytest
import torch

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.gdn.operator import (
    _attach_cp_layout,
    _gdn_island_layer_forward,
    _infer_cp_hidden_layout,
    run_gdn_layer,
)
from art.preprocessing.pack import packed_tensors_from_tokenized_results
from art.preprocessing.tokenize import TokenizedResult

from .cases import default_phase0_cases
from .metrics import GDN_CORRECTNESS_DTYPE
from .packed_layout import build_phase0_packed_tensors, summarize_case
from .parser_import import (
    build_gdn_chain_only_rank_execution_plan,
    build_gdn_cp_segment_schedule,
    build_gdn_rank_execution_plan,
    parse_gdn_shared_prefix_segments,
)


class _FakeCpPlan(BaseModel):
    cp_size: int = 2
    attention_token_count: int
    gdn_token_count: int
    attention_token_indices: tuple[int, ...]
    gdn_token_indices: tuple[int, ...]


class _FakeAttentionBias:
    def __init__(self, plan: Any) -> None:
        self.gdn_execution_plan = plan
        self.gdn_hidden_layout = "gdn"
        self.gdn_active_module = object()


class _FakeNonGdnLayer:
    _art_gdn_island_is_gdn = False

    def __init__(self) -> None:
        self._art_gdn_island_physical_forward = self._forward

    def _forward(self, hidden_states: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        return hidden_states + 1


class _FakeGdnLayer:
    _art_gdn_island_is_gdn = True
    _art_gdn_island_prev_is_gdn = True
    _art_gdn_island_next_is_gdn = True

    def __init__(self) -> None:
        self.self_attention = object()
        self.forward_calls = 0
        self._art_gdn_island_physical_forward = self._forward

    def _forward(self, hidden_states: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        self.forward_calls += 1
        return hidden_states


def test_default_phase0_cases_parse_and_cover_required_shapes() -> None:
    summaries = []
    for case in default_phase0_cases(conv_width=4):
        tensors = build_phase0_packed_tensors(case)
        spec = parse_gdn_shared_prefix_segments(
            tensors["group_ids"], tensors["parent_ids"], min_completions_per_family=1
        )
        assert spec.family_count >= 1
        assert spec.completion_count >= spec.family_count
        assert spec.real_token_count == int((tensors["group_ids"] != -1).sum().item())
        assert len({segment.group_id for segment in spec.segments()}) == len(
            spec.segments()
        )
        _assert_segments_cover_valid_tokens_once(spec)
        summaries.append(summarize_case(case, tensors, conv_width=4))

    by_name = {summary.name: summary for summary in summaries}
    assert by_name["multi_family_repeated"].family_count >= 3
    assert by_name["conv_tail_boundary"].suffix_shorter_than_conv
    assert by_name["conv_tail_boundary"].suffix_equal_to_conv
    assert by_name["conv_tail_boundary"].suffix_longer_than_conv
    assert by_name["padding_tail"].valid_lengths[0] < 80
    assert any(summary.cp_boundary_prefix for summary in summaries)
    assert any(summary.cp_boundary_suffix for summary in summaries)
    assert by_name["long_sibling"].max_segment_length >= 96
    assert by_name["many_branches_wave"].completion_count >= 12
    assert by_name["family_boundary_at_partition"].family_boundary_at_partition
    assert by_name["empty_trailing_rank"].empty_trailing_rank


def test_parser_accepts_real_art_without_prompt_packing_semantics() -> None:
    random.seed(20260426)
    packed = packed_tensors_from_tokenized_results(
        [
            _tokenized_result(
                prompt_id=101,
                token_ids=(11, 12, 13, 21, 22, 23),
                logprobs=(
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    -1.1,
                    -1.2,
                ),
            ),
            _tokenized_result(
                prompt_id=101,
                token_ids=(11, 12, 13, 31, 32, 33),
                logprobs=(
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    -1.3,
                    -1.4,
                ),
            ),
            _tokenized_result(
                prompt_id=202,
                token_ids=(41, 42, 43, 51, 52, 53),
                logprobs=(
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    -1.5,
                    -1.6,
                ),
            ),
            _tokenized_result(
                prompt_id=202,
                token_ids=(41, 42, 43, 61, 62, 63),
                logprobs=(
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    -1.7,
                    -1.8,
                ),
            ),
        ],
        seq_len=18,
        pad_token_id=-100,
        truncate_long_results=False,
        verbosity=0,
    )

    spec = parse_gdn_shared_prefix_segments(
        packed["group_ids"], packed["parent_ids"], min_completions_per_family=2
    )

    assert spec.family_count == 2
    assert spec.completion_count == 4
    for family in spec.families:
        assert family.prefix.length == 3
        assert tuple(completion.length for completion in family.completions) == (3, 3)
        for completion in family.completions:
            assert not bool(
                packed["assistant_mask"][family.row_index, completion.start]
            )
            assert packed["input_pos"][family.row_index, completion.start].item() == 3
            assert bool(
                packed["assistant_mask"][family.row_index, completion.start + 1]
            )


def test_production_gdn_call_requires_prebuilt_plan() -> None:
    hidden = torch.zeros((4, 1, 8), dtype=GDN_CORRECTNESS_DTYPE)
    group_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    parent_ids = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)

    with pytest.raises(ValueError, match="requires a prebuilt"):
        run_gdn_layer(
            _DummyGdn(),
            hidden,
            group_ids=group_ids,
            parent_ids=parent_ids,
            require_prebuilt_plan=True,
        )


def test_parser_rejects_rank_mismatch() -> None:
    group_ids = torch.zeros((4,), dtype=torch.long)
    parent_ids = torch.zeros((1, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="rank 2"):
        parse_gdn_shared_prefix_segments(group_ids, parent_ids)


def test_parser_rejects_shape_mismatch() -> None:
    group_ids = torch.zeros((1, 4), dtype=torch.long)
    parent_ids = torch.zeros((1, 5), dtype=torch.long)
    with pytest.raises(ValueError, match="same shape"):
        parse_gdn_shared_prefix_segments(group_ids, parent_ids)


def test_parser_rejects_non_contiguous_padding() -> None:
    group_ids = torch.tensor([[0, -1, 1]], dtype=torch.long)
    parent_ids = torch.tensor([[0, -1, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match="contiguous"):
        parse_gdn_shared_prefix_segments(group_ids, parent_ids)


def test_parser_rejects_completion_before_prefix() -> None:
    group_ids = torch.tensor([[1, 1, -1]], dtype=torch.long)
    parent_ids = torch.tensor([[0, 0, -1]], dtype=torch.long)
    with pytest.raises(ValueError, match="before its prefix"):
        parse_gdn_shared_prefix_segments(group_ids, parent_ids)


def test_parser_rejects_wrong_active_parent() -> None:
    group_ids = torch.tensor([[0, 0, 1, 1, -1]], dtype=torch.long)
    parent_ids = torch.tensor([[0, 0, 9, 9, -1]], dtype=torch.long)
    with pytest.raises(ValueError, match="expected active prefix"):
        parse_gdn_shared_prefix_segments(group_ids, parent_ids)


def test_parser_rejects_interleaved_unrelated_family() -> None:
    group_ids = torch.tensor([[0, 0, 1, 1, 2, 2, -1]], dtype=torch.long)
    parent_ids = torch.tensor([[0, 0, 1, 1, 0, 0, -1]], dtype=torch.long)
    with pytest.raises(ValueError, match="before its prefix|expected active prefix"):
        parse_gdn_shared_prefix_segments(group_ids, parent_ids)


def test_parser_rejects_group_parent_change() -> None:
    group_ids = torch.tensor([[0, 0, 1, 1, -1]], dtype=torch.long)
    parent_ids = torch.tensor([[0, 0, 0, 2, -1]], dtype=torch.long)
    with pytest.raises(ValueError, match="changes parent"):
        parse_gdn_shared_prefix_segments(group_ids, parent_ids)


def test_parser_rejects_reused_group_id() -> None:
    group_ids = torch.tensor([[0, 0, 1, 1, 0, -1]], dtype=torch.long)
    parent_ids = torch.tensor([[0, 0, 0, 0, 0, -1]], dtype=torch.long)
    with pytest.raises(ValueError, match="non-contiguous"):
        parse_gdn_shared_prefix_segments(group_ids, parent_ids)


def test_min_completions_gate() -> None:
    group_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    parent_ids = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="expected at least 2"):
        parse_gdn_shared_prefix_segments(
            group_ids, parent_ids, min_completions_per_family=2
        )


def test_cp_rank_plan_builds_native_fla_cp_metadata() -> None:
    tensors = build_phase0_packed_tensors(default_phase0_cases(conv_width=4)[0])
    spec = parse_gdn_shared_prefix_segments(
        tensors["group_ids"], tensors["parent_ids"], min_completions_per_family=1
    )
    plan = build_gdn_rank_execution_plan(spec, device="cpu", cp_rank=0, cp_size=2)
    assert plan.cp_size == 2
    assert plan.attention_to_gdn is not None
    assert plan.gdn_to_attention is not None


def test_cp_hidden_layout_inference_rejects_stale_attention_bias_layout() -> None:
    chosen_plan = cast(
        Any,
        _FakeCpPlan(
            attention_token_count=5,
            gdn_token_count=7,
            attention_token_indices=tuple(range(5)),
            gdn_token_indices=tuple(range(7)),
        ),
    )

    attention_hidden = torch.empty((chosen_plan.attention_token_count, 1, 8))
    gdn_hidden = torch.empty((chosen_plan.gdn_token_count, 1, 8))

    assert (
        _infer_cp_hidden_layout(attention_hidden, chosen_plan, gdn=None) == "attention"
    )
    assert _infer_cp_hidden_layout(gdn_hidden, chosen_plan, gdn=None) == "gdn"
    assert (
        _infer_cp_hidden_layout(
            _attach_cp_layout(attention_hidden, "gdn"), chosen_plan, gdn=None
        )
        == "gdn"
    )


def test_gdn_island_recompute_repairs_stale_attention_layout_marker() -> None:
    plan = cast(
        Any,
        _FakeCpPlan(
            attention_token_count=5,
            gdn_token_count=7,
            attention_token_indices=tuple(range(5)),
            gdn_token_indices=tuple(range(7)),
        ),
    )
    attention_bias = _FakeAttentionBias(plan)
    hidden_states = torch.zeros((plan.attention_token_count, 1, 8))

    output = _gdn_island_layer_forward(
        _FakeNonGdnLayer(), hidden_states, attention_bias=attention_bias
    )

    assert torch.equal(output, hidden_states + 1)
    assert attention_bias.gdn_hidden_layout == "attention"
    assert attention_bias.gdn_active_module is None


def test_gdn_island_empty_rank_still_runs_transformer_layer_forward() -> None:
    plan = cast(
        Any,
        _FakeCpPlan(
            attention_token_count=5,
            gdn_token_count=0,
            attention_token_indices=tuple(range(5)),
            gdn_token_indices=(),
        ),
    )
    attention_bias = _FakeAttentionBias(plan)
    hidden_states = torch.zeros((0, 1, 8))
    layer = _FakeGdnLayer()

    output = _gdn_island_layer_forward(
        layer, hidden_states, attention_bias=attention_bias
    )

    assert output.shape == hidden_states.shape
    assert layer.forward_calls == 1
    assert attention_bias.gdn_hidden_layout == "gdn"


def test_cp_rank_plan_rejects_invalid_external_attention_layout() -> None:
    group_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    parent_ids = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    spec = parse_gdn_shared_prefix_segments(
        group_ids, parent_ids, min_completions_per_family=1
    )

    with pytest.raises(ValueError, match="missing a real token"):
        build_gdn_rank_execution_plan(
            spec,
            device="cpu",
            cp_rank=0,
            cp_size=2,
            attention_token_layout_index=TokenLayoutIndex(
                ownership_ranges_by_rank=(((0, 2, 0),), ((1, 3, 0),)),
                token_counts_by_rank=(2, 2),
            ),
        )

    with pytest.raises(ValueError, match="token count must match"):
        build_gdn_rank_execution_plan(
            spec,
            device="cpu",
            cp_rank=0,
            cp_size=2,
            attention_token_layout_index=TokenLayoutIndex(
                ownership_ranges_by_rank=(((0, 2, 0),), ()),
                token_counts_by_rank=(2, 0),
            ),
        )


def test_cp_rank_plan_accepts_attention_token_layout_index_without_tuple_layout() -> (
    None
):
    tensors = build_phase0_packed_tensors(default_phase0_cases(conv_width=4)[0])
    spec = parse_gdn_shared_prefix_segments(
        tensors["group_ids"], tensors["parent_ids"], min_completions_per_family=1
    )
    real_tokens = tuple(
        reversed(
            tuple(
                token
                for segment in spec.segments()
                for token in segment.linear_indices(spec.sequence_length)
            )
        )
    )
    attention_by_rank = (
        tuple(real_tokens[0::2]),
        tuple(real_tokens[1::2]),
    )
    layout_index = _layout_from_tokens_by_rank(attention_by_rank)
    index_plan = build_gdn_rank_execution_plan(
        spec,
        device="cpu",
        cp_rank=0,
        cp_size=2,
        attention_token_layout_index=layout_index,
    )

    assert index_plan.attention_token_indices == attention_by_rank[0]
    assert index_plan.attention_to_gdn.cross_rank_token_count >= 0


def test_many_small_default_plan_uses_chunk_native_local_buckets() -> None:
    group_ids, parent_ids = _many_small_group_tensors(
        family_count=304,
        completions_per_family=4,
        prefix_base=64,
        completion_base=16,
    )
    spec = parse_gdn_shared_prefix_segments(
        group_ids,
        parent_ids,
        min_completions_per_family=1,
    )

    local_plan = build_gdn_rank_execution_plan(spec, device="cpu")
    assert len(local_plan.prefix_boundary_buckets) == 1
    assert not local_plan.prefix_tail_buckets
    assert local_plan.completion_with_prefix_tail_buckets

    schedule = build_gdn_cp_segment_schedule(spec, cp_size=2)
    for rank in range(2):
        rank_plan = build_gdn_rank_execution_plan(
            spec,
            device="cpu",
            cp_rank=rank,
            cp_size=2,
            cp_segment_schedule=schedule,
        )
        assert len(rank_plan.prefix_boundary_buckets) <= 1
        assert not rank_plan.prefix_tail_buckets
        assert rank_plan.completion_with_prefix_tail_buckets


def test_cp_local_schedule_uses_family_cohesion_with_matching_attention_layout() -> (
    None
):
    group_ids, parent_ids = _many_small_group_tensors(
        family_count=12,
        completions_per_family=4,
        prefix_base=96,
        completion_base=24,
    )
    spec = parse_gdn_shared_prefix_segments(
        group_ids,
        parent_ids,
        min_completions_per_family=1,
    )

    schedule = build_gdn_cp_segment_schedule(
        spec,
        cp_size=2,
        attention_token_layout_index=_layout_from_tokens_by_rank(
            _whole_family_rank_indices(spec, cp_size=2)
        ),
    )
    rank_loads = list(schedule.gdn_token_counts_by_rank)

    assert schedule.cross_rank_token_count == 0
    assert schedule.parent_state_exchange_family_indices == ()
    assert max(rank_loads) - min(rank_loads) <= 256


def test_cp_rank_plan_splits_oversized_non_chain_family_by_segments() -> None:
    group_ids, parent_ids = _dominant_with_background_group_tensors()
    spec = parse_gdn_shared_prefix_segments(
        group_ids,
        parent_ids,
        min_completions_per_family=1,
    )

    rank_plans = tuple(
        build_gdn_rank_execution_plan(
            spec,
            device="cpu",
            cp_rank=rank,
            cp_size=4,
        )
        for rank in range(4)
    )
    rank_loads = [plan.gdn_token_count for plan in rank_plans]

    assert min(rank_loads) > 0
    assert max(rank_loads) < spec.real_token_count
    assert any(plan.remote_prefix_tail_state_transfers for plan in rank_plans)
    assert all(
        transfer.family_indices_tensor is not None
        for plan in rank_plans
        for transfer in plan.remote_prefix_tail_state_transfers
    )
    assert any(plan.remote_completion_with_prefix_tail_buckets for plan in rank_plans)
    assert all(
        0 not in bucket.family_indices.tolist()
        for plan in rank_plans
        for bucket in plan.ready_local_completion_buckets
    )
    assert any(
        0 in bucket.family_indices.tolist()
        for plan in rank_plans
        for bucket in plan.remote_completion_with_prefix_tail_buckets
    )
    for plan in rank_plans:
        _assert_plan_outputs_each_local_position_once(plan)


def test_cp_local_family_plan_rebalances_skewed_completion_segments() -> None:
    group_ids, parent_ids = _weak_scaled_dominant_group_tensors()
    spec = parse_gdn_shared_prefix_segments(
        group_ids,
        parent_ids,
        min_completions_per_family=1,
    )

    rank_plans = tuple(
        build_gdn_rank_execution_plan(spec, device="cpu", cp_rank=rank, cp_size=2)
        for rank in range(2)
    )
    rank_loads = [plan.gdn_token_count for plan in rank_plans]

    assert min(rank_loads) > 0
    assert max(rank_loads) <= 1.05 * (sum(rank_loads) / len(rank_loads))
    assert any(plan.remote_prefix_tail_state_transfers for plan in rank_plans)
    assert any(plan.remote_completion_with_prefix_tail_buckets for plan in rank_plans)


def test_cp_explicit_attention_layout_rebalances_skewed_local_work() -> None:
    group_ids, parent_ids = _group_tensors_from_families(
        [
            (15825, tuple(921 for _ in range(16))),
            *((512, (64, 65, 66, 67)) for _ in range(171)),
        ]
    )
    spec = parse_gdn_shared_prefix_segments(
        group_ids,
        parent_ids,
        min_completions_per_family=1,
    )
    token_count = int(spec.real_token_count)
    attention_layout = _layout_from_tokens_by_rank(
        (
            tuple(range(0, 4096)),
            tuple(range(4096, 8192)),
            tuple(range(8192, 12288)),
            tuple(range(12288, token_count)),
        )
    )

    rank_plans = tuple(
        build_gdn_rank_execution_plan(
            spec,
            device="cpu",
            cp_rank=rank,
            cp_size=4,
            attention_token_layout_index=attention_layout,
        )
        for rank in range(4)
    )
    rank_loads = [plan.gdn_token_count for plan in rank_plans]

    assert min(rank_loads) > 0
    assert max(rank_loads) <= 1.10 * (sum(rank_loads) / len(rank_loads))
    assert any(plan.attention_to_gdn.cross_rank_token_count > 0 for plan in rank_plans)


def test_cp_remote_prefix_tail_plan_does_not_duplicate_legacy_work() -> None:
    group_ids, parent_ids = _many_small_group_tensors(
        family_count=49,
        completions_per_family=16,
        prefix_base=5000,
        completion_base=1000,
    )
    spec = parse_gdn_shared_prefix_segments(
        group_ids,
        parent_ids,
        min_completions_per_family=1,
    )
    plans = tuple(
        build_gdn_rank_execution_plan(
            spec,
            device="cpu",
            cp_rank=rank,
            cp_size=8,
        )
        for rank in range(8)
    )

    assert any(plan.remote_completion_with_prefix_tail_buckets for plan in plans)
    for plan in plans:
        assert plan.local_prefix_buckets == ()
        _assert_plan_outputs_each_local_position_once(plan)


@pytest.mark.parametrize("cp_size", (2, 4, 8))
def test_cp_default_plan_routes_64k_prefix_to_native_chain(cp_size: int) -> None:
    group_ids, parent_ids = _single_long_family_group_tensors(
        prefix_len=65_536,
        suffix_len=65_536,
        completion_count=8,
    )
    spec = parse_gdn_shared_prefix_segments(
        group_ids,
        parent_ids,
        min_completions_per_family=1,
    )

    plans = tuple(
        build_gdn_rank_execution_plan(
            spec,
            device="cpu",
            cp_rank=rank,
            cp_size=cp_size,
        )
        for rank in range(cp_size)
    )
    rank_loads = [plan.gdn_token_count for plan in plans]

    assert all(plan.chain_prefix_buckets for plan in plans)
    assert all(
        plan.chain_completion_buckets or plan.local_completion_buckets for plan in plans
    )
    assert max(rank_loads) == min(rank_loads)
    assert all(
        int(bucket.lengths.min().item()) > 0
        for plan in plans
        for bucket in (
            *plan.chain_prefix_buckets,
            *plan.chain_completion_buckets,
            *plan.local_completion_buckets,
        )
    )
    for plan in plans:
        _assert_plan_outputs_each_local_position_once(plan)


@pytest.mark.parametrize("cp_size", (2, 4, 8))
def test_cp_chain_only_fast_plan_avoids_global_schedule(cp_size: int) -> None:
    group_ids, parent_ids = _single_long_family_group_tensors(
        prefix_len=65_536,
        suffix_len=65_536,
        completion_count=8,
    )
    spec = parse_gdn_shared_prefix_segments(
        group_ids,
        parent_ids,
        min_completions_per_family=1,
    )

    plans = tuple(
        build_gdn_chain_only_rank_execution_plan(
            spec,
            device="cpu",
            cp_rank=rank,
            cp_size=cp_size,
        )
        for rank in range(cp_size)
    )

    assert all(plan is not None for plan in plans)
    rank_plans = tuple(plan for plan in plans if plan is not None)
    assert sum(plan.gdn_token_count for plan in rank_plans) == (spec.real_token_count)
    assert all(
        plan.attention_token_ranges == plan.gdn_token_ranges for plan in rank_plans
    )
    assert all(plan.chain_prefix_buckets for plan in rank_plans)
    assert all(plan.chain_completion_buckets for plan in rank_plans)
    assert all(
        plan.attention_to_gdn is not None
        and plan.attention_to_gdn.cross_rank_token_count == 0
        for plan in rank_plans
    )


def _assert_segments_cover_valid_tokens_once(spec: Any) -> None:
    seen: set[tuple[int, int]] = set()
    for segment in spec.segments():
        for position in range(segment.start, segment.end):
            key = (segment.row_index, position)
            assert key not in seen
            seen.add(key)
    expected = {
        (row_index, position)
        for row_index, valid_length in enumerate(spec.valid_lengths)
        for position in range(valid_length)
    }
    assert seen == expected


def _assert_plan_outputs_each_local_position_once(plan: Any) -> None:
    positions: list[int] = []
    ready_completion_buckets = (
        plan.ready_local_completion_buckets
        if plan.ready_local_completion_buckets or plan.remote_local_completion_buckets
        else plan.local_completion_buckets
    )
    for bucket in (
        *plan.chain_prefix_buckets,
        *plan.prefix_boundary_buckets,
        *plan.prefix_tail_buckets,
        *plan.completion_with_prefix_tail_buckets,
        *plan.remote_prefix_tail_buckets,
        *plan.remote_completion_with_prefix_tail_buckets,
        *plan.local_prefix_buckets,
        *plan.chain_completion_buckets,
        *ready_completion_buckets,
        *plan.remote_local_completion_buckets,
    ):
        output_mask = bucket.real_mask
        if bucket.output_mask is not None:
            output_mask = output_mask & bucket.output_mask
        positions.extend(
            int(position) for position in bucket.position_indices[output_mask]
        )
    assert sorted(positions) == list(range(plan.gdn_token_count))


def _layout_from_tokens_by_rank(
    tokens_by_rank: tuple[tuple[int, ...], ...],
) -> TokenLayoutIndex:
    return TokenLayoutIndex(
        ownership_ranges_by_rank=tuple(
            _rank_ranges_from_tokens(tokens) for tokens in tokens_by_rank
        ),
        token_counts_by_rank=tuple(len(tokens) for tokens in tokens_by_rank),
    )


def _rank_ranges_from_tokens(
    tokens: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    if not tokens:
        return ()
    ranges = []
    start = tokens[0]
    end = start + 1
    position = 0
    for local_position, token in enumerate(tokens[1:], start=1):
        if token == end:
            end += 1
            continue
        ranges.append((start, end, position))
        start = token
        end = token + 1
        position = local_position
    ranges.append((start, end, position))
    return tuple(ranges)


def _many_small_group_tensors(
    *,
    family_count: int,
    completions_per_family: int,
    prefix_base: int,
    completion_base: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_ids: list[int] = []
    parent_ids: list[int] = []
    group = 0
    for family_index in range(family_count):
        prefix_group = group
        prefix_len = prefix_base + (family_index % 9) - 4
        group_ids.extend([prefix_group] * prefix_len)
        parent_ids.extend([prefix_group] * prefix_len)
        group += 1
        for completion_index in range(completions_per_family):
            completion_group = group
            completion_len = (
                completion_base + ((family_index + completion_index) % 7) - 3
            )
            group_ids.extend([completion_group] * completion_len)
            parent_ids.extend([prefix_group] * completion_len)
            group += 1
    return torch.tensor([group_ids], dtype=torch.long), torch.tensor(
        [parent_ids], dtype=torch.long
    )


def _single_long_family_group_tensors(
    *, prefix_len: int, suffix_len: int, completion_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    group_ids = [0] * prefix_len
    parent_ids = [0] * prefix_len
    for group in range(1, completion_count + 1):
        group_ids.extend([group] * suffix_len)
        parent_ids.extend([0] * suffix_len)
    return torch.tensor([group_ids], dtype=torch.long), torch.tensor(
        [parent_ids], dtype=torch.long
    )


def _tokenized_result(
    *,
    prompt_id: int,
    token_ids: tuple[int, ...],
    logprobs: tuple[float, ...],
) -> TokenizedResult:
    return TokenizedResult(
        advantage=1.0,
        chat="",
        token_ids=list(token_ids),
        input_pos=list(range(len(token_ids))),
        assistant_mask=[0, 0, 0, 0, 1, 1],
        logprobs=list(logprobs),
        pixel_values=None,
        image_grid_thw=None,
        trajectory=None,  # type: ignore[arg-type]
        choice_offsets=[],
        extra_logprobs={},
        _tokenizer=cast(Any, _DummyTokenizer()),
        weight=1.0,
        prompt_id=prompt_id,
        prompt_length=3,
    )


class _DummyTokenizer:
    def decode(self, token_id: int) -> str:
        return str(token_id)


class _DummyGdn:
    pass


def _dominant_with_background_group_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    families: list[tuple[int, tuple[int, ...]]] = [
        (14745, tuple(921 for _ in range(16)))
    ]
    families.extend((256, (64, 65, 66, 67)) for _ in range(21))
    return _group_tensors_from_families(families)


def _weak_scaled_dominant_group_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    families: list[tuple[int, tuple[int, ...]]] = [
        (29491, tuple(1843 for _ in range(16)))
    ]
    for family_index in range(43):
        prefix = 256 + (family_index % 5) * 3
        suffixes = tuple(64 + ((family_index + child) % 4) for child in range(4))
        families.append((prefix, suffixes))
    return _group_tensors_from_families(families)


def _group_tensors_from_families(
    families: list[tuple[int, tuple[int, ...]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    group_ids: list[int] = []
    parent_ids: list[int] = []
    group = 0
    for prefix_len, suffix_lengths in families:
        prefix_group = group
        group_ids.extend([prefix_group] * prefix_len)
        parent_ids.extend([prefix_group] * prefix_len)
        group += 1
        for suffix_len in suffix_lengths:
            group_ids.extend([group] * suffix_len)
            parent_ids.extend([prefix_group] * suffix_len)
            group += 1
    return torch.tensor([group_ids], dtype=torch.long), torch.tensor(
        [parent_ids], dtype=torch.long
    )


def _whole_family_rank_indices(
    spec: Any, *, cp_size: int
) -> tuple[tuple[int, ...], ...]:
    ranks: list[list[int]] = [[] for _ in range(cp_size)]
    loads = [0] * cp_size
    for family in spec.families:
        rank = min(range(cp_size), key=lambda index: (loads[index], index))
        tokens = [
            token
            for segment in (family.prefix, *family.completions)
            for token in segment.linear_indices(spec.sequence_length)
        ]
        ranks[rank].extend(tokens)
        loads[rank] += len(tokens)
    return tuple(tuple(tokens) for tokens in ranks)
