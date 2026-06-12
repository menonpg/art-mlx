from __future__ import annotations

from typing import cast

import pytest

torch = pytest.importorskip("torch")
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from art.megatron.context_parallel.executor import (
    _build_stage_block_mask,
    _get_prepared_stage_block_mask,
    _get_prepared_stage_execution_spec,
    _resolve_stage_execution_spec,
)
from art.megatron.context_parallel.runtime import (
    prepare_megatron_context_parallel_state,
)
from art.megatron.context_parallel.types import (
    ArtContextParallelState,
    ContextParallelConfig,
    ParallelTopology,
    StageExecutionSpec,
    StagePlan,
)
from art.preprocessing.pack import PackedTensors
from tests.integration.megatron.gdn_shared_prefix.cases import default_phase0_cases
from tests.integration.megatron.gdn_shared_prefix.packed_layout import (
    build_phase0_packed_tensors,
)


def test_cp_stage_block_masks_match_slice_aware_reference_metadata() -> None:
    for case_name in ("long_sibling", "cp_boundary_suffix"):
        case = next(item for item in default_phase0_cases() if item.name == case_name)
        micro = cast(PackedTensors, build_phase0_packed_tensors(case))
        for topology in _metadata_topologies():
            for cp_rank in range(int(topology.cp)):
                state, _rank_plan, _spec, _pad_multiple = (
                    prepare_megatron_context_parallel_state(
                        micro=micro,
                        topology=topology,
                        config=ContextParallelConfig(),
                        cp_group=None,
                        cp_rank=cp_rank,
                        target_device=torch.device("cpu"),
                    )
                )
                for stage_plan in state.rank_plan.stage_plans:
                    if (
                        stage_plan.q_len <= 0
                        or stage_plan.k_len <= 0
                        or not stage_plan.slices
                    ):
                        continue
                    execution_spec = _resolve_stage_execution_spec(
                        stage_plan=stage_plan,
                        state=state,
                        block_size=state.config.block_size,
                    )
                    block_mask = _build_stage_block_mask(
                        stage_plan=stage_plan,
                        state=state,
                        device=torch.device("cpu"),
                        execution_spec=execution_spec,
                        block_size=state.config.block_size,
                    )
                    assert block_mask is not None
                    reference = _stage_reference_block_mask(
                        stage_plan=stage_plan,
                        execution_spec=execution_spec,
                        state=state,
                        block_size=block_mask.BLOCK_SIZE,
                    )
                    _assert_block_mask_metadata_equal(
                        f"{case_name} {topology} rank={cp_rank} stage={stage_plan.stage_index}",
                        block_mask,
                        reference,
                    )


def test_cp_forward_requires_prepared_stage_caches() -> None:
    case = next(item for item in default_phase0_cases() if item.name == "long_sibling")
    micro = cast(PackedTensors, build_phase0_packed_tensors(case))
    state, _rank_plan, _spec, _pad_multiple = prepare_megatron_context_parallel_state(
        micro=micro,
        topology=ParallelTopology(tp=1, cp=2, dp=1, sp=False),
        config=ContextParallelConfig(),
        cp_group=None,
        cp_rank=0,
        target_device=torch.device("cpu"),
    )
    stage_plan = next(
        stage
        for stage in state.rank_plan.stage_plans
        if stage.q_len > 0 and stage.k_len > 0 and stage.slices
    )
    with pytest.raises(RuntimeError, match="unprepared stage execution-spec"):
        _get_prepared_stage_execution_spec(
            stage_plan=stage_plan,
            state=state,
            block_size=state.config.block_size,
        )
    execution_spec = _resolve_stage_execution_spec(
        stage_plan=stage_plan,
        state=state,
        block_size=state.config.block_size,
    )
    with pytest.raises(RuntimeError, match="unprepared stage block-mask"):
        _get_prepared_stage_block_mask(
            stage_plan=stage_plan,
            state=state,
            device=torch.device("cpu"),
            execution_spec=execution_spec,
            block_size=state.config.block_size,
        )


def _metadata_topologies() -> tuple[ParallelTopology, ...]:
    return (
        ParallelTopology(tp=1, cp=2, dp=1, sp=False),
        ParallelTopology(tp=2, cp=2, dp=1, sp=True),
        ParallelTopology(tp=1, cp=4, dp=1, sp=False),
    )


def _stage_reference_block_mask(
    *,
    stage_plan: StagePlan,
    execution_spec: StageExecutionSpec,
    state: ArtContextParallelState,
    block_size: tuple[int, int],
) -> BlockMask:
    mask_metadata = execution_spec.mask_metadata
    assert mask_metadata is not None
    q_abs = mask_metadata.q_token_indices
    k_abs = mask_metadata.k_token_indices
    group_ids = state.group_ids
    parent_ids = state.parent_ids

    def mask_mod(
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        query_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del batch_idx, head_idx
        q_abs_local = q_abs[query_idx]
        k_abs_local = k_abs[kv_idx]
        valid = (q_abs_local >= 0) & (k_abs_local >= 0)
        q_safe = q_abs_local.clamp_min(0)
        k_safe = k_abs_local.clamp_min(0)
        same_group = group_ids[q_safe] == group_ids[k_safe]
        parent_prefix = parent_ids[q_safe] == group_ids[k_safe]
        semantic = (q_abs_local >= k_abs_local) & (same_group | parent_prefix) & valid
        in_slice = torch.zeros_like(semantic)
        for slice_ in stage_plan.slices:
            in_slice |= (
                (query_idx >= slice_.q_range.start)
                & (query_idx < slice_.q_range.end)
                & (kv_idx >= slice_.k_range.start)
                & (kv_idx < slice_.k_range.end)
            )
        return semantic & in_slice

    return create_block_mask(
        mask_mod,
        1,
        None,
        int(execution_spec.q_len),
        int(execution_spec.k_len),
        device="cpu",
        BLOCK_SIZE=block_size,
    )


def _assert_block_mask_metadata_equal(
    case_name: str,
    new_mask: BlockMask,
    old_mask: BlockMask,
) -> None:
    assert new_mask.seq_lengths == old_mask.seq_lengths, case_name
    assert new_mask.BLOCK_SIZE == old_mask.BLOCK_SIZE, case_name
    for count_attr, index_attr in (
        ("kv_num_blocks", "kv_indices"),
        ("full_kv_num_blocks", "full_kv_indices"),
        ("q_num_blocks", "q_indices"),
        ("full_q_num_blocks", "full_q_indices"),
    ):
        assert _ordered_sparse_rows(
            getattr(new_mask, count_attr),
            getattr(new_mask, index_attr),
        ) == _ordered_sparse_rows(
            getattr(old_mask, count_attr),
            getattr(old_mask, index_attr),
        ), f"{case_name}: {count_attr}/{index_attr} mismatch"


def _ordered_sparse_rows(
    num_blocks: torch.Tensor | None,
    indices: torch.Tensor | None,
) -> tuple[tuple[int, ...], ...] | None:
    if num_blocks is None or indices is None:
        return None
    counts = num_blocks.cpu().reshape(-1, num_blocks.shape[-1])
    rows = indices.cpu().reshape(-1, indices.shape[-2], indices.shape[-1])
    output: list[tuple[int, ...]] = []
    for batch_head_idx in range(int(rows.shape[0])):
        for row_idx in range(int(rows.shape[1])):
            count = int(counts[batch_head_idx, row_idx])
            output.append(
                tuple(
                    sorted(
                        int(value) for value in rows[batch_head_idx, row_idx, :count]
                    )
                )
            )
    return tuple(output)
