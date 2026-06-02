from __future__ import annotations

import pytest
import torch

pytest.importorskip("megatron.core")

from art.megatron.context_parallel import ContextParallelConfig, ParallelTopology
from art.megatron.context_parallel.runtime import (
    prepare_megatron_context_parallel_state,
)
from art.megatron.dsv4 import prepare_dsv4_context_parallel_state
from art.preprocessing.pack import PackedTensors


def test_prepare_dsv4_context_parallel_state_wraps_real_cp_plan() -> None:
    cp_state, rank_plan, _spec, _pad = prepare_megatron_context_parallel_state(
        micro=_shared_prefix_micro(),
        topology=ParallelTopology(cp=2),
        config=ContextParallelConfig(block_size=4, planner_chunk_size=4),
        cp_group=None,
        cp_rank=1,
    )

    dsv4_state = prepare_dsv4_context_parallel_state(
        cp_state=cp_state,
        hca_ratio=4,
        extra={"source": "test"},
    )
    plan = dsv4_state.dsv4_plan

    assert dsv4_state.cp_state is cp_state
    assert dsv4_state.extra == {"source": "test"}
    assert plan.csa_layout is not None
    assert plan.hca_layout is not None
    assert len(plan.csa_layout.entry_ids_by_owner_rank) == 2
    assert len(plan.hca_layout.entry_ids_by_owner_rank) == 2
    assert tuple(slot.stage_index for slot in plan.stage_plan_slots) == tuple(
        stage.stage_index for stage in rank_plan.stage_plans
    )
    assert all(len(slot.stage_plans_by_rank) == 2 for slot in plan.stage_plan_slots)
    assert tuple(stage.stage_index for stage in plan.csa_indexer_stage_plans) == tuple(
        slot.stage_index for slot in plan.stage_plan_slots
    )
    for stage_plan in plan.csa_indexer_stage_plans:
        assert len(stage_plan.query_token_ids_by_rank) == 2
        assert len(stage_plan.candidate_entry_ids_by_rank) == 2
    assert len(plan.csa_indexer_kv_peer_plans_by_stage) == len(
        plan.csa_indexer_stage_plans
    )
    assert len(plan.csa_stage_kv_peer_plans_by_slot) == len(plan.stage_plan_slots)
    assert len(plan.hca_stage_kv_peer_plans_by_slot) == len(plan.stage_plan_slots)
    assert all(
        len(peer_plans) == 2 for peer_plans in plan.csa_stage_kv_peer_plans_by_slot
    )
    assert all(
        len(peer_plans) == 2 for peer_plans in plan.hca_stage_kv_peer_plans_by_slot
    )
    assert plan.csa_attention_backward_plan is not None
    assert plan.hca_attention_backward_plan is not None
    assert len(plan.csa_attention_backward_plan.rank_plans) == 2
    assert len(plan.hca_attention_backward_plan.rank_plans) == 2


def test_prepare_dsv4_context_parallel_state_allows_hca_only() -> None:
    cp_state, _rank_plan, _spec, _pad = prepare_megatron_context_parallel_state(
        micro=_shared_prefix_micro(),
        topology=ParallelTopology(cp=2),
        config=ContextParallelConfig(block_size=4, planner_chunk_size=4),
        cp_group=None,
        cp_rank=0,
    )

    dsv4_state = prepare_dsv4_context_parallel_state(
        cp_state=cp_state,
        include_csa=False,
        hca_ratio=4,
    )

    assert dsv4_state.dsv4_plan.csa_layout is None
    assert dsv4_state.dsv4_plan.hca_layout is not None
    assert dsv4_state.dsv4_plan.csa_indexer_stage_plans == ()
    assert dsv4_state.dsv4_plan.csa_indexer_kv_peer_plans_by_stage == ()
    assert dsv4_state.dsv4_plan.csa_stage_kv_peer_plans_by_slot == ()
    assert dsv4_state.dsv4_plan.hca_stage_kv_peer_plans_by_slot
    assert dsv4_state.dsv4_plan.csa_attention_backward_plan is None
    assert dsv4_state.dsv4_plan.hca_attention_backward_plan is not None
    assert dsv4_state.dsv4_plan.stage_plan_slots


def test_prepare_dsv4_context_parallel_state_requires_a_layer_family() -> None:
    cp_state, _rank_plan, _spec, _pad = prepare_megatron_context_parallel_state(
        micro=_shared_prefix_micro(),
        topology=ParallelTopology(cp=2),
        config=ContextParallelConfig(block_size=4, planner_chunk_size=4),
        cp_group=None,
        cp_rank=0,
    )

    with pytest.raises(RuntimeError, match="requires CSA, HCA, or both"):
        prepare_dsv4_context_parallel_state(
            cp_state=cp_state,
            include_csa=False,
            include_hca=False,
        )


def _shared_prefix_micro() -> PackedTensors:
    group_ids = torch.tensor([[0] * 8 + [1] * 8 + [2] * 8], dtype=torch.long)
    parent_ids = torch.tensor([[0] * 8 + [0] * 8 + [0] * 8], dtype=torch.long)
    seq_len = int(group_ids.shape[1])
    return PackedTensors(
        tokens=torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
        group_ids=group_ids,
        parent_ids=parent_ids,
        input_pos=torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
        assistant_mask=torch.ones(1, seq_len, dtype=torch.bool),
        logprobs=torch.zeros(1, seq_len, dtype=torch.float32),
        advantages=torch.ones(1, seq_len, dtype=torch.float32),
        weights=torch.ones(1, seq_len, dtype=torch.float32),
        pixel_values=[None],
        image_grid_thw=[None],
    )
