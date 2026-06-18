from __future__ import annotations

import torch

from art.megatron.context_parallel.builder import (
    build_dense_reference_mask,
    build_shared_prefix_attention_spec,
)
from art.megatron.context_parallel.block_mask import build_block_mask
from art.megatron.context_parallel.runtime import build_context_parallel_token_layout_index
from art.megatron.context_parallel.types import (
    AttnMaskKind,
    AttnSlice,
    ContextParallelConfig,
    ExactMaskMetadata,
    FlexMaskSpec,
    ParallelTopology,
    TokenRange,
)
from art.megatron.shared_prefix_packing import pack_shared_prefixes


def test_shared_prefix_attention_spec_supports_depth_two() -> None:
    pack = pack_shared_prefixes(
        (
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1, 2, 3, 5]),
            torch.tensor([1, 6, 7]),
        ),
        max_depth=2,
    )

    spec = build_shared_prefix_attention_spec(
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
    )
    dense = build_dense_reference_mask(row_spec=spec.rows[0])

    assert dense.int().tolist() == [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 1],
    ]


def test_shared_prefix_attention_spec_supports_arbitrary_depth() -> None:
    pack = pack_shared_prefixes(
        (
            torch.tensor([1, 2, 3, 4, 8]),
            torch.tensor([1, 2, 3, 4, 9]),
            torch.tensor([1, 2, 3, 5]),
            torch.tensor([1, 6]),
        ),
        max_depth=3,
    )

    spec = build_shared_prefix_attention_spec(
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
    )
    dense = build_dense_reference_mask(row_spec=spec.rows[0])

    assert dense.equal(_reference_tree_mask(pack.group_ids[0], pack.parent_ids[0]))


def test_depth_two_shared_prefix_can_build_context_parallel_layout() -> None:
    pack = pack_shared_prefixes(
        (
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1, 2, 3, 5]),
            torch.tensor([1, 6, 7]),
        ),
        max_depth=2,
    )

    layout = build_context_parallel_token_layout_index(
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
        topology=ParallelTopology(cp=2),
        config=ContextParallelConfig(planner_chunk_size=2, planner_max_search_steps=1),
        original_seq_len=int(pack.tokens.numel()),
    )

    assert sum(layout.token_counts_by_rank) == int(pack.tokens.numel())


def test_depth_two_sparse_block_mask_exact_predicate_matches_dense_reference() -> None:
    pack = pack_shared_prefixes(
        (
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1, 2, 3, 5]),
            torch.tensor([1, 6, 7]),
        ),
        max_depth=2,
    )
    spec = build_shared_prefix_attention_spec(
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
    )
    row = spec.rows[0]
    token_indices = torch.arange(row.valid_tokens, dtype=torch.long)
    block_mask = build_block_mask(
        FlexMaskSpec(
            q_len=row.valid_tokens,
            k_len=row.valid_tokens,
            block_size=(2, 2),
            slices=row.slices,
            exact_mask=ExactMaskMetadata(
                q_token_indices=token_indices,
                k_token_indices=token_indices,
                cache_key="depth-two",
            ),
        ),
        group_ids=pack.group_ids[0],
        parent_ids=pack.parent_ids[0],
        device=torch.device("cpu"),
    )

    assert block_mask is not None
    q_indices = torch.arange(row.valid_tokens)[:, None]
    k_indices = torch.arange(row.valid_tokens)[None, :]
    actual = block_mask.mask_mod(
        torch.zeros_like(q_indices),
        torch.zeros_like(q_indices),
        q_indices,
        k_indices,
    )

    assert actual.equal(build_dense_reference_mask(row_spec=row))


def test_sparse_block_mask_supports_non_monotonic_remote_k_indices() -> None:
    q_token_indices = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    k_token_indices = torch.tensor([0, 1, 6, 2, 3, 4], dtype=torch.long)
    block_mask = build_block_mask(
        FlexMaskSpec(
            q_len=int(q_token_indices.numel()),
            k_len=int(k_token_indices.numel()),
            block_size=(2, 2),
            slices=(
                AttnSlice(
                    q_range=TokenRange(start=0, end=int(q_token_indices.numel())),
                    k_range=TokenRange(start=0, end=int(k_token_indices.numel())),
                    mask_kind=AttnMaskKind.CAUSAL,
                    row_index=0,
                ),
            ),
            exact_mask=ExactMaskMetadata(
                q_token_indices=q_token_indices,
                k_token_indices=k_token_indices,
                cache_key="non-monotonic-k",
            ),
        ),
        group_ids=torch.ones(8, dtype=torch.long),
        parent_ids=torch.ones(8, dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert block_mask is not None
    q_indices = torch.arange(q_token_indices.numel())[:, None]
    k_indices = torch.arange(k_token_indices.numel())[None, :]

    actual = block_mask.mask_mod(
        torch.zeros_like(q_indices),
        torch.zeros_like(q_indices),
        q_indices,
        k_indices,
    )

    assert actual.equal(q_token_indices[:, None] >= k_token_indices[None, :])


def _reference_tree_mask(group_ids: torch.Tensor, parent_ids: torch.Tensor) -> torch.Tensor:
    group_list = [int(value) for value in group_ids.tolist()]
    parent_by_group: dict[int, int | None] = {}
    for group_id, parent_id in zip(group_list, parent_ids.tolist(), strict=True):
        group_id = int(group_id)
        parent_id = int(parent_id)
        if group_id not in parent_by_group:
            parent_by_group[group_id] = None if parent_id == group_id else parent_id

    ancestors_by_group = {
        group_id: _ancestors(group_id, parent_by_group) for group_id in parent_by_group
    }
    dense = torch.zeros((len(group_list), len(group_list)), dtype=torch.bool)
    for q_pos, q_group in enumerate(group_list):
        allowed_groups = ancestors_by_group[q_group] | {q_group}
        for k_pos, k_group in enumerate(group_list):
            dense[q_pos, k_pos] = k_pos <= q_pos and k_group in allowed_groups
    return dense


def _ancestors(
    group_id: int,
    parent_by_group: dict[int, int | None],
) -> set[int]:
    ancestors: set[int] = set()
    cursor = parent_by_group[group_id]
    while cursor is not None and cursor not in ancestors:
        ancestors.add(cursor)
        cursor = parent_by_group.get(cursor)
    return ancestors
