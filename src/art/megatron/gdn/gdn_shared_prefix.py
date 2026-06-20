from __future__ import annotations

from bisect import bisect_left
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field
import torch

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.shared_prefix_tree import parse_shared_prefix_tree

GdnSegmentKind = Literal["prefix", "completion"]
# FLA's public chunk_gated_delta_rule hard-codes 64-token WY chunks.
FLA_CHUNK_SIZE = 64
_PydanticModelT = TypeVar("_PydanticModelT", bound=BaseModel)


class GdnSegmentSpec(BaseModel):
    """Contiguous logical GDN segment in one packed row."""

    model_config = ConfigDict(frozen=True)

    row_index: int = Field(ge=0)
    family_index: int = Field(ge=0)
    group_id: int
    parent_id: int
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    kind: GdnSegmentKind
    child_index: int | None = Field(default=None, ge=0)

    @property
    def length(self) -> int:
        return self.end - self.start

    def linear_indices(self, sequence_length: int) -> tuple[int, ...]:
        base = self.row_index * sequence_length
        return tuple(range(base + self.start, base + self.end))


class GdnPackedExecutionSpec(BaseModel):
    """Parsed shared-prefix GDN execution metadata for a packed batch."""

    model_config = ConfigDict(frozen=True)

    batch_size: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    valid_lengths: tuple[int, ...]
    tree_segments: tuple[GdnSegmentSpec, ...]
    tree_parent_indices: tuple[int, ...]
    tree_depths: tuple[int, ...]

    @property
    def family_count(self) -> int:
        return len(self.tree_segments)

    @property
    def completion_count(self) -> int:
        return sum(1 for parent in self.tree_parent_indices if parent >= 0)

    @property
    def real_token_count(self) -> int:
        return sum(self.valid_lengths)

    @property
    def max_segment_length(self) -> int:
        return max((segment.length for segment in self.tree_segments), default=0)

    def segments(self) -> tuple[GdnSegmentSpec, ...]:
        return self.tree_segments


_GDN_SEGMENT_SPEC_FIELDS = frozenset(
    {
        "row_index",
        "family_index",
        "group_id",
        "parent_id",
        "start",
        "end",
        "kind",
        "child_index",
    }
)


def _trusted_pydantic_construct(
    model_type: type[_PydanticModelT],
    fields_set: frozenset[str],
    **values: Any,
) -> _PydanticModelT:
    model = model_type.__new__(model_type)
    object.__setattr__(model, "__dict__", values)
    object.__setattr__(model, "__pydantic_fields_set__", fields_set)
    object.__setattr__(model, "__pydantic_extra__", None)
    object.__setattr__(model, "__pydantic_private__", None)
    return model


class GdnSegmentBucketPlan(BaseModel):
    """Device-local index tensors for a variable-length GDN segment batch."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    length: int = Field(ge=1)
    lengths: torch.Tensor
    lengths_cpu: torch.Tensor
    lengths_by_rank_cpu: torch.Tensor | None = None
    real_mask: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor
    row_indices: torch.Tensor
    position_indices: torch.Tensor
    family_indices: torch.Tensor
    family_indices_cpu: torch.Tensor | None = None
    parent_indices: torch.Tensor | None = None
    parent_indices_cpu: torch.Tensor | None = None
    needs_final_state: bool = True
    real_token_count_static: int = Field(ge=0)
    output_mask: torch.Tensor | None = None

    @property
    def segment_count(self) -> int:
        return int(self.family_indices.numel())

    @property
    def real_token_count(self) -> int:
        return self.real_token_count_static


class GdnPlannerConfig(BaseModel):
    """Tunable cost coefficients for one packed-row GDN execution plan."""

    model_config = ConfigDict(frozen=True)

    max_padding_ratio: float = Field(default=2.0, gt=1.0)
    max_segments_per_batch: int = Field(default=4096, ge=1)
    cp_chain_min_tokens_per_rank: int = Field(default=32, ge=1)
    cp_chain_min_total_tokens: int = Field(default=32768, ge=1)
    cp_chain_min_prefix_only_tokens: int = Field(default=32768, ge=1)
    cp_tree_chain_min_total_tokens: int = Field(default=8192, ge=1)
    cp_tree_chain_min_prefix_only_tokens: int = Field(default=8192, ge=1)
    rank_idle_token_cost: float = Field(default=1.0, ge=0.0)
    max_zero_exchange_load_imbalance: float = Field(default=1.5, ge=1.0)
    planner_local_token_ms: float = Field(default=0.00065, ge=0.0)
    planner_layout_cross_rank_token_ms: float = Field(default=0.00008, ge=0.0)
    planner_empty_rank_ms: float = Field(default=32.0, ge=0.0)


class GdnRankExecutionPlan(BaseModel):
    """Rank-local planned execution metadata for shared-prefix GDN."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    cp_rank: int = Field(ge=0)
    cp_size: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    sequence_length: int = Field(ge=0)
    packed_batch_size: int | None = Field(default=None, ge=1)
    packed_sequence_length: int | None = Field(default=None, ge=1)
    real_token_mask: torch.Tensor
    family_count: int = Field(ge=0)
    completion_count: int = Field(ge=0)
    attention_to_gdn: Any | None = None
    gdn_to_attention: Any | None = None
    attention_token_ranges: tuple[tuple[int, int, int], ...] = ()
    gdn_token_ranges: tuple[tuple[int, int, int], ...] = ()
    attention_token_count: int = Field(default=0, ge=0)
    gdn_token_count: int = Field(default=0, ge=0)
    tree_segment_buckets_by_depth: tuple[tuple[GdnSegmentBucketPlan, ...], ...] = ()
    tree_chain_buckets_by_depth: tuple[tuple[GdnSegmentBucketPlan, ...], ...] = ()

    @property
    def attention_token_indices(self) -> tuple[int, ...]:
        return _tokens_from_rank_ranges(self.attention_token_ranges)

    @property
    def gdn_token_indices(self) -> tuple[int, ...]:
        return _tokens_from_rank_ranges(self.gdn_token_ranges)


class _AttentionLayoutIndex(BaseModel):
    """Counting index for CP attention token ownership."""

    model_config = ConfigDict(frozen=True)

    token_ranges_by_rank: tuple[tuple[tuple[int, int], ...], ...]
    token_range_ends_by_rank: tuple[tuple[int, ...], ...]
    range_count: int = Field(ge=0)


def _layout_cp_size(layout: TokenLayoutIndex) -> int:
    return len(layout.token_counts_by_rank)


def _layout_token_count(layout: TokenLayoutIndex) -> int:
    return sum(int(count) for count in layout.token_counts_by_rank)


def _tokens_from_rank_ranges(
    ranges: tuple[tuple[int, int, int], ...],
) -> tuple[int, ...]:
    return tuple(token for start, end, _ in ranges for token in range(start, end))


def _token_layout_from_rank_ranges(
    ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...],
) -> TokenLayoutIndex:
    return TokenLayoutIndex(
        ownership_ranges_by_rank=ranges_by_rank,
        token_counts_by_rank=tuple(
            _ranges_token_count(ranges) for ranges in ranges_by_rank
        ),
    )


def _ranges_token_count(ranges: tuple[tuple[int, int, int], ...]) -> int:
    return sum(int(end) - int(start) for start, end, _ in ranges)


def build_gdn_rank_execution_plan(
    spec: GdnPackedExecutionSpec,
    *,
    device: torch.device | str,
    cp_rank: int = 0,
    cp_size: int = 1,
    attention_token_layout_index: TokenLayoutIndex | None = None,
    planner_config: GdnPlannerConfig | None = None,
) -> GdnRankExecutionPlan:
    """Build rank-local tensor metadata from a parsed shared-prefix DAG.

    Planning is CPU-bound and must run once per packed training sequence. CP>1
    emits mixed work: native FLA CP chain buckets for long segments and local
    fork buckets for short work where CP collectives would be inefficient.
    """

    planner_config = planner_config or GdnPlannerConfig()
    target_device = torch.device(device)
    if target_device.type != "cpu":
        cpu_plan = build_gdn_rank_execution_plan(
            spec,
            device="cpu",
            cp_rank=cp_rank,
            cp_size=cp_size,
            attention_token_layout_index=attention_token_layout_index,
            planner_config=planner_config,
        )
        return move_gdn_rank_execution_plan_to_device(cpu_plan, target_device)
    return _build_tree_rank_execution_plan(
        spec,
        device=device,
        cp_rank=cp_rank,
        cp_size=cp_size,
        attention_token_layout_index=attention_token_layout_index,
        planner_config=planner_config,
    )


def _build_tree_rank_execution_plan(
    spec: GdnPackedExecutionSpec,
    *,
    device: torch.device | str,
    cp_rank: int,
    cp_size: int,
    attention_token_layout_index: TokenLayoutIndex | None,
    planner_config: GdnPlannerConfig,
) -> GdnRankExecutionPlan:
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
    if not spec.tree_segments:
        raise ValueError("tree GDN planning requires tree segments")
    if len(spec.tree_parent_indices) != len(spec.tree_segments):
        raise ValueError("tree parent metadata length must match tree segments")
    if len(spec.tree_depths) != len(spec.tree_segments):
        raise ValueError("tree depth metadata length must match tree segments")

    from art.megatron.gdn.layout import (
        _reverse_exchange_plan,
        build_local_rank_cp_exchange_plan_from_dest_ranges,
    )

    source_layout = _attention_source_layout(
        spec,
        cp_size=cp_size,
        attention_token_layout_index=attention_token_layout_index,
        planner_config=planner_config,
    )
    attention_layout_index = _build_attention_layout_index_from_token_layout(
        source_layout,
        max_ranges=max(1, 2 * spec.real_token_count // len(spec.tree_segments)),
    )
    segment_attention_counts = _segment_attention_rank_counts(
        spec,
        cp_size=cp_size,
        attention_layout_index=attention_layout_index,
    )

    depth_count = max(spec.tree_depths, default=0) + 1
    rank_loads = [0] * cp_size
    owner_by_node = [-1] * len(spec.tree_segments)
    chained_nodes = [False] * len(spec.tree_segments)
    tree_has_children = [False] * len(spec.tree_segments)
    for parent_index in spec.tree_parent_indices:
        if parent_index >= 0:
            tree_has_children[parent_index] = True
    gdn_ranges_by_rank: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
    segments_by_rank_depth: list[list[list[GdnSegmentSpec]]] = [
        [[] for _ in range(depth_count)] for _ in range(cp_size)
    ]
    chain_segments_by_depth: list[list[GdnSegmentSpec]] = [
        [] for _ in range(depth_count)
    ]
    cross_rank_token_count = 0

    tree_segments_by_depth: list[list[GdnSegmentSpec]] = [
        [] for _ in range(depth_count)
    ]
    for segment in spec.tree_segments:
        tree_segments_by_depth[spec.tree_depths[segment.family_index]].append(segment)

    for depth, depth_segments in enumerate(tree_segments_by_depth):
        local_groups: list[tuple[GdnSegmentSpec, ...]] = []
        siblings_by_parent: dict[int, list[GdnSegmentSpec]] = {}
        for segment in depth_segments:
            parent_index = spec.tree_parent_indices[segment.family_index]
            if (
                parent_index < 0
                and cp_size > 1
                and _can_chain_tree_segment(
                    segment,
                    cp_size=cp_size,
                    planner_config=planner_config,
                )
            ):
                chained_nodes[segment.family_index] = True
                chain_segments_by_depth[depth].append(segment)
                cross_rank_token_count += _append_chain_segment(
                    gdn_ranges_by_rank,
                    rank_loads,
                    segment,
                    spec,
                    attention_layout_index=attention_layout_index,
                )
                continue
            if parent_index < 0:
                local_groups.append((segment,))
            else:
                if depth_count <= 2:
                    siblings_by_parent.setdefault(parent_index, []).append(segment)
                else:
                    local_groups.append((segment,))
        local_groups.extend(tuple(group) for group in siblings_by_parent.values())

        for local_group in local_groups:
            parent_owner = _tree_group_parent_owner(
                local_group,
                tree_parent_indices=spec.tree_parent_indices,
                owner_by_node=owner_by_node,
                chained_nodes=chained_nodes,
            )
            owner = (
                parent_owner
                if parent_owner is not None
                else _best_segment_owner(
                    local_group,
                    rank_loads,
                    segment_attention_counts=segment_attention_counts,
                    planner_config=planner_config,
                )
            )
            for segment in local_group:
                owner_by_node[segment.family_index] = owner
                segments_by_rank_depth[owner][depth].append(segment)
                cross_rank_token_count += _append_local_segment(
                    gdn_ranges_by_rank,
                    rank_loads,
                    owner,
                    segment,
                    spec,
                    segment_attention_counts=segment_attention_counts,
                )

    gdn_ranges_by_rank_by_position = tuple(
        tuple(ranges) for ranges in gdn_ranges_by_rank
    )
    gdn_ranges_by_rank_by_source = tuple(
        tuple(sorted(ranges)) for ranges in gdn_ranges_by_rank
    )

    attention_to_gdn = build_local_rank_cp_exchange_plan_from_dest_ranges(
        source_layout=source_layout,
        device=device,
        local_rank=cp_rank,
        dest_ranges_by_rank=gdn_ranges_by_rank_by_position,
        cross_rank_token_count=cross_rank_token_count,
    )
    local_token_ranges = gdn_ranges_by_rank_by_source[cp_rank]
    tree_segment_buckets_by_depth = tuple(
        (
            _build_tree_segment_bucket_plans(
                tuple(segments_by_rank_depth[cp_rank][depth]),
                spec.tree_parent_indices,
                tuple(tree_has_children),
                device=device,
                planner_config=planner_config,
            )
            if cp_size == 1
            else _build_tree_position_bucket_plans(
                tuple(segments_by_rank_depth[cp_rank][depth]),
                spec.tree_parent_indices,
                tuple(tree_has_children),
                local_token_ranges,
                sequence_length=spec.sequence_length,
                device=device,
                planner_config=planner_config,
            )
        )
        for depth in range(depth_count)
    )
    tree_chain_buckets_by_depth = (
        tuple(
            _build_tree_position_bucket_plans(
                tuple(chain_segments_by_depth[depth]),
                spec.tree_parent_indices,
                tuple(tree_has_children),
                local_token_ranges,
                sequence_length=spec.sequence_length,
                device=device,
                planner_config=planner_config,
                token_ranges_by_rank=tuple(
                    tuple(ranges) for ranges in gdn_ranges_by_rank_by_source
                ),
                split_by_final_state=False,
            )
            for depth in range(depth_count)
        )
        if cp_size > 1
        else tuple(() for _ in range(depth_count))
    )
    if cp_size == 1:
        valid_lengths = torch.tensor(
            spec.valid_lengths, device=device, dtype=torch.long
        )
        positions = torch.arange(spec.sequence_length, device=device, dtype=torch.long)
        real_token_mask = positions.unsqueeze(0) < valid_lengths.unsqueeze(1)
    else:
        real_token_mask = torch.ones(
            1,
            rank_loads[cp_rank],
            device=device,
            dtype=torch.bool,
        )

    return GdnRankExecutionPlan.model_construct(
        cp_rank=cp_rank,
        cp_size=cp_size,
        batch_size=1 if cp_size > 1 else spec.batch_size,
        sequence_length=rank_loads[cp_rank] if cp_size > 1 else spec.sequence_length,
        packed_batch_size=spec.batch_size,
        packed_sequence_length=spec.sequence_length,
        real_token_mask=real_token_mask,
        family_count=spec.family_count,
        completion_count=spec.completion_count,
        attention_to_gdn=attention_to_gdn,
        gdn_to_attention=_reverse_exchange_plan(attention_to_gdn),
        attention_token_ranges=source_layout.ownership_ranges_by_rank[cp_rank],
        gdn_token_ranges=gdn_ranges_by_rank_by_position[cp_rank],
        attention_token_count=source_layout.token_counts_by_rank[cp_rank],
        gdn_token_count=rank_loads[cp_rank],
        tree_segment_buckets_by_depth=tree_segment_buckets_by_depth,
        tree_chain_buckets_by_depth=tree_chain_buckets_by_depth,
    )


def move_gdn_rank_execution_plan_to_device(
    plan: GdnRankExecutionPlan,
    device: torch.device | str,
) -> GdnRankExecutionPlan:
    """Move planner tensors to the execution device after CPU planning."""

    from art.megatron.gdn.layout import move_cp_exchange_plan_to_device

    return GdnRankExecutionPlan.model_construct(
        cp_rank=plan.cp_rank,
        cp_size=plan.cp_size,
        batch_size=plan.batch_size,
        sequence_length=plan.sequence_length,
        packed_batch_size=plan.packed_batch_size,
        packed_sequence_length=plan.packed_sequence_length,
        real_token_mask=_move_planner_tensor(plan.real_token_mask, device),
        family_count=plan.family_count,
        completion_count=plan.completion_count,
        attention_to_gdn=move_cp_exchange_plan_to_device(plan.attention_to_gdn, device),
        gdn_to_attention=move_cp_exchange_plan_to_device(plan.gdn_to_attention, device),
        attention_token_ranges=plan.attention_token_ranges,
        gdn_token_ranges=plan.gdn_token_ranges,
        attention_token_count=plan.attention_token_count,
        gdn_token_count=plan.gdn_token_count,
        tree_segment_buckets_by_depth=tuple(
            _move_bucket_plans(buckets, device)
            for buckets in plan.tree_segment_buckets_by_depth
        ),
        tree_chain_buckets_by_depth=tuple(
            _move_bucket_plans(buckets, device)
            for buckets in plan.tree_chain_buckets_by_depth
        ),
    )


def _move_bucket_plans(
    buckets: tuple[GdnSegmentBucketPlan, ...],
    device: torch.device | str,
) -> tuple[GdnSegmentBucketPlan, ...]:
    return tuple(
        GdnSegmentBucketPlan.model_construct(
            length=bucket.length,
            lengths=_move_planner_tensor(bucket.lengths, device),
            lengths_cpu=bucket.lengths_cpu,
            lengths_by_rank_cpu=bucket.lengths_by_rank_cpu,
            real_mask=_move_planner_tensor(bucket.real_mask, device),
            cu_seqlens=_move_planner_tensor(bucket.cu_seqlens, device),
            cu_seqlens_cpu=bucket.cu_seqlens_cpu,
            row_indices=_move_planner_tensor(bucket.row_indices, device),
            position_indices=_move_planner_tensor(bucket.position_indices, device),
            family_indices=_move_planner_tensor(bucket.family_indices, device),
            family_indices_cpu=bucket.family_indices_cpu,
            parent_indices=(
                _move_planner_tensor(bucket.parent_indices, device)
                if bucket.parent_indices is not None
                else None
            ),
            parent_indices_cpu=bucket.parent_indices_cpu,
            needs_final_state=bucket.needs_final_state,
            real_token_count_static=bucket.real_token_count,
            output_mask=(
                _move_planner_tensor(bucket.output_mask, device)
                if bucket.output_mask is not None
                else None
            ),
        )
        for bucket in buckets
    )


def parse_gdn_shared_prefix_segments(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    *,
    min_completions_per_family: int = 0,
) -> GdnPackedExecutionSpec:
    """Parse ART packed shared-prefix metadata into generic GDN tree nodes."""

    del min_completions_per_family
    groups = _rank2_long_cpu("group_ids", group_ids)
    parents = _rank2_long_cpu("parent_ids", parent_ids)
    if tuple(groups.shape) != tuple(parents.shape):
        raise ValueError(
            "group_ids and parent_ids must have the same shape, got "
            f"{tuple(groups.shape)} and {tuple(parents.shape)}"
        )

    batch_size, sequence_length = (int(groups.shape[0]), int(groups.shape[1]))
    rows = parse_shared_prefix_tree(group_ids=groups, parent_ids=parents)
    tree_segments: list[GdnSegmentSpec] = []
    tree_parent_indices: list[int] = []
    tree_depths: list[int] = []
    valid_lengths: list[int] = []
    node_by_row_group: dict[tuple[int, int], int] = {}
    child_counts_by_parent: dict[int, int] = {}

    for row in rows:
        valid_lengths.append(row.valid_tokens)
        for segment in row.segments:
            node_index = len(tree_segments)
            is_root = segment.depth == 0
            parent_node_index = (
                -1
                if is_root
                else node_by_row_group[(segment.row_index, segment.parent_id)]
            )
            child_index = None
            if not is_root:
                child_index = child_counts_by_parent.get(parent_node_index, 0)
                child_counts_by_parent[parent_node_index] = child_index + 1
            tree_segments.append(
                _trusted_pydantic_construct(
                    GdnSegmentSpec,
                    _GDN_SEGMENT_SPEC_FIELDS,
                    row_index=segment.row_index,
                    family_index=node_index,
                    group_id=segment.group_id,
                    parent_id=segment.parent_id,
                    start=segment.start,
                    end=segment.end,
                    kind="prefix" if is_root else "completion",
                    child_index=child_index,
                )
            )
            tree_parent_indices.append(parent_node_index)
            tree_depths.append(segment.depth)
            node_by_row_group[(segment.row_index, segment.group_id)] = node_index

    return GdnPackedExecutionSpec(
        batch_size=batch_size,
        sequence_length=sequence_length,
        valid_lengths=tuple(valid_lengths),
        tree_segments=tuple(tree_segments),
        tree_parent_indices=tuple(tree_parent_indices),
        tree_depths=tuple(tree_depths),
    )


def _build_segment_bucket_plans(
    segment_buckets: tuple[tuple[GdnSegmentSpec, ...], ...],
    *,
    device: torch.device | str,
) -> tuple[GdnSegmentBucketPlan, ...]:
    return tuple(
        _build_segment_bucket_plan(bucket[0].length, bucket, device=device)
        for bucket in segment_buckets
    )


def _attention_source_layout(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    attention_token_layout_index: TokenLayoutIndex | None,
    planner_config: GdnPlannerConfig,
) -> TokenLayoutIndex:
    if attention_token_layout_index is not None:
        if _layout_cp_size(attention_token_layout_index) != cp_size:
            raise ValueError(
                "attention token layout index cp_size must match GDN cp_size, got "
                f"{_layout_cp_size(attention_token_layout_index)} and {cp_size}"
            )
        if _layout_token_count(attention_token_layout_index) != spec.real_token_count:
            raise ValueError(
                "attention token layout index token count must match GDN real token "
                f"count, got {_layout_token_count(attention_token_layout_index)} and "
                f"{spec.real_token_count}"
            )
        return attention_token_layout_index
    return _token_layout_from_rank_ranges(
        _default_attention_layout_ranges(
            spec,
            cp_size=cp_size,
            planner_config=planner_config,
        )
    )


def _can_chain_segment(
    segment: GdnSegmentSpec,
    *,
    cp_size: int,
    planner_config: GdnPlannerConfig,
) -> bool:
    min_tokens = (
        planner_config.cp_chain_min_prefix_only_tokens
        if segment.kind == "prefix"
        else planner_config.cp_chain_min_total_tokens
    )
    return _can_chain_segment_with_min_tokens(
        segment,
        cp_size=cp_size,
        min_tokens=min_tokens,
        planner_config=planner_config,
    )


def _can_chain_tree_segment(
    segment: GdnSegmentSpec,
    *,
    cp_size: int,
    planner_config: GdnPlannerConfig,
) -> bool:
    min_tokens = (
        min(
            planner_config.cp_tree_chain_min_prefix_only_tokens,
            planner_config.cp_chain_min_prefix_only_tokens,
        )
        if segment.kind == "prefix"
        else min(
            planner_config.cp_tree_chain_min_total_tokens,
            planner_config.cp_chain_min_total_tokens,
        )
    )
    return _can_chain_segment_with_min_tokens(
        segment,
        cp_size=cp_size,
        min_tokens=min_tokens,
        planner_config=planner_config,
    )


def _can_chain_segment_with_min_tokens(
    segment: GdnSegmentSpec,
    *,
    cp_size: int,
    min_tokens: int,
    planner_config: GdnPlannerConfig,
) -> bool:
    if segment.length < min_tokens:
        return False
    if segment.length < cp_size:
        return False
    if segment.length // FLA_CHUNK_SIZE < cp_size:
        return False
    per_rank = segment.length / cp_size
    if per_rank < planner_config.cp_chain_min_tokens_per_rank:
        return False
    return True


def _best_segment_owner(
    segments: tuple[GdnSegmentSpec, ...],
    rank_loads: list[int],
    *,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
    planner_config: GdnPlannerConfig,
) -> int:
    segment_length = sum(segment.length for segment in segments)
    if len(segments) == 1:
        on_rank_tokens = segment_attention_counts[_segment_key(segments[0])]
    else:
        rank_count = len(rank_loads)
        counts_by_rank = [0] * rank_count
        for segment in segments:
            segment_counts = segment_attention_counts[_segment_key(segment)]
            for rank in range(rank_count):
                counts_by_rank[rank] += segment_counts[rank]
        on_rank_tokens = tuple(counts_by_rank)
    best: tuple[float, float, int, int, int, int] | None = None
    for rank, tokens in enumerate(on_rank_tokens):
        projected_loads = list(rank_loads)
        projected_loads[rank] += segment_length
        max_load = max(projected_loads, default=0)
        target_load = sum(projected_loads) / max(1, len(projected_loads))
        overload = max(
            0.0,
            max_load - planner_config.max_zero_exchange_load_imbalance * target_load,
        )
        idle_tokens = sum(max_load - load for load in projected_loads)
        cross_rank_tokens = segment_length - int(tokens)
        empty_rank_count = sum(1 for load in projected_loads if load == 0)
        score = (
            max_load * planner_config.planner_local_token_ms
            + idle_tokens
            * planner_config.rank_idle_token_cost
            * planner_config.planner_local_token_ms
            + cross_rank_tokens * planner_config.planner_layout_cross_rank_token_ms
            + empty_rank_count * planner_config.planner_empty_rank_ms
        )
        candidate = (
            overload,
            score,
            max_load,
            cross_rank_tokens,
            -int(tokens),
            rank,
        )
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return _least_loaded_rank(rank_loads)
    return best[-1]


def _tree_group_parent_owner(
    segments: tuple[GdnSegmentSpec, ...],
    *,
    tree_parent_indices: tuple[int, ...],
    owner_by_node: list[int],
    chained_nodes: list[bool],
) -> int | None:
    if not segments:
        return None
    segment = segments[0]
    parent_index = tree_parent_indices[segment.family_index]
    if parent_index < 0 or chained_nodes[parent_index]:
        return None
    parent_owner = owner_by_node[parent_index]
    return parent_owner if parent_owner >= 0 else None


def _build_attention_layout_index_from_token_layout(
    layout: TokenLayoutIndex,
    *,
    max_ranges: int,
) -> _AttentionLayoutIndex:
    del max_ranges
    ranges_by_rank = tuple(
        tuple(sorted((int(start), int(end)) for start, end, _ in rank_ranges))
        for rank_ranges in layout.ownership_ranges_by_rank
    )
    range_count = sum(len(ranges) for ranges in ranges_by_rank)
    return _AttentionLayoutIndex.model_construct(
        token_ranges_by_rank=ranges_by_rank,
        token_range_ends_by_rank=tuple(
            tuple(end for _, end in ranges) for ranges in ranges_by_rank
        ),
        range_count=range_count,
    )


def _segment_attention_rank_counts(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    attention_layout_index: _AttentionLayoutIndex,
) -> dict[tuple[int, int, int], tuple[int, ...]]:
    del cp_size
    segments = tuple(spec.segments())
    if not segments:
        return {}
    starts = torch.tensor(
        [_segment_token_start(segment, spec.sequence_length) for segment in segments],
        dtype=torch.long,
    )
    lengths = torch.tensor([segment.length for segment in segments], dtype=torch.long)
    ends = starts + lengths
    counts_by_rank = []
    for ranges in attention_layout_index.token_ranges_by_rank:
        counts_by_rank.append(_rank_range_overlap_counts(starts, ends, ranges))
    counts_tensor = torch.stack(counts_by_rank, dim=1)
    totals = counts_tensor.sum(dim=1)
    if not torch.equal(totals, lengths):
        bad_index = int(torch.nonzero(totals != lengths, as_tuple=False)[0].item())
        raise ValueError(
            "attention layout is missing a real token required by GDN; "
            f"segment={_segment_key(segments[bad_index])}"
        )
    counts = counts_tensor.tolist()
    return {
        _segment_key(segment): tuple(int(value) for value in counts[index])
        for index, segment in enumerate(segments)
    }


def _rank_range_overlap_counts(
    starts: torch.Tensor,
    ends: torch.Tensor,
    ranges: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    if not ranges:
        return torch.zeros_like(starts)
    range_starts = torch.tensor([start for start, _ in ranges], dtype=torch.long)
    range_ends = torch.tensor([end for _, end in ranges], dtype=torch.long)
    range_lengths = range_ends - range_starts
    prefix = torch.cat((range_lengths.new_zeros(1), torch.cumsum(range_lengths, dim=0)))

    def owned_before(points: torch.Tensor) -> torch.Tensor:
        indices = torch.searchsorted(range_ends, points, right=False)
        counts = prefix.index_select(0, indices)
        active = indices < int(range_starts.numel())
        if bool(active.any().item()):
            active_indices = indices[active]
            active_starts = range_starts.index_select(0, active_indices)
            active_ends = range_ends.index_select(0, active_indices)
            counts[active] += torch.minimum(
                torch.clamp(points[active] - active_starts, min=0),
                active_ends - active_starts,
            )
        return counts

    return owned_before(ends) - owned_before(starts)


def _segment_key(segment: GdnSegmentSpec) -> tuple[int, int, int]:
    return (segment.row_index, segment.start, segment.end)


def _default_attention_layout_ranges(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    planner_config: GdnPlannerConfig,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    ranks: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
    loads = [0] * cp_size
    target_rank_load = spec.real_token_count / cp_size

    def append_segment(rank: int, token_start: int, token_count: int) -> None:
        ranks[rank].append((token_start, token_start + token_count, loads[rank]))
        loads[rank] += token_count

    def should_split_segment(segment: GdnSegmentSpec) -> bool:
        if segment.length <= planner_config.max_zero_exchange_load_imbalance * (
            target_rank_load
        ):
            return False
        return _can_chain_tree_segment(
            segment, cp_size=cp_size, planner_config=planner_config
        )

    for segment in spec.tree_segments:
        token_start = _segment_token_start(segment, spec.sequence_length)
        if should_split_segment(segment):
            _append_split_default_attention_segment(
                ranks, loads, token_start, segment.length
            )
            continue
        owner = _least_loaded_rank(loads)
        append_segment(owner, token_start, segment.length)
    return tuple(tuple(ranges) for ranges in ranks)


def _append_split_default_attention_segment(
    ranks: list[list[tuple[int, int, int]]],
    loads: list[int],
    token_start: int,
    token_count: int,
) -> None:
    cp_size = len(ranks)
    for rank in range(cp_size):
        start = (token_count * rank) // cp_size
        end = (token_count * (rank + 1)) // cp_size
        ranks[rank].append((token_start + start, token_start + end, loads[rank]))
        loads[rank] += end - start


def _append_chain_segment(
    gdn_ranges_by_rank: list[list[tuple[int, int, int]]],
    rank_loads: list[int],
    segment: GdnSegmentSpec,
    spec: GdnPackedExecutionSpec,
    *,
    attention_layout_index: _AttentionLayoutIndex | None = None,
) -> int:
    token_start = _segment_token_start(segment, spec.sequence_length)
    cp_size = len(gdn_ranges_by_rank)
    attention_shards = _attention_contiguous_chain_shards(
        token_start,
        segment.length,
        cp_size=cp_size,
        attention_layout_index=attention_layout_index,
    )
    if attention_shards is not None:
        for rank, shard in enumerate(attention_shards):
            position_start = rank_loads[rank]
            gdn_ranges_by_rank[rank].append((shard.start, shard.stop, position_start))
            rank_loads[rank] += len(shard)
        return 0
    cross_rank_tokens = 0
    shard_lengths = _fla_aligned_chain_shard_lengths(segment.length, cp_size=cp_size)
    start = 0
    for rank, shard_length in enumerate(shard_lengths):
        end = start + shard_length
        if start >= end:
            raise ValueError(
                "CP chain planning requires non-empty shards; "
                f"segment={segment.kind}:{segment.family_index} "
                f"length={segment.length} cp_size={cp_size}"
            )
        shard_start = token_start + start
        position_start = rank_loads[rank]
        gdn_ranges_by_rank[rank].append(
            (shard_start, shard_start + shard_length, position_start)
        )
        rank_loads[rank] += shard_length
        if attention_layout_index is not None:
            cross_rank_tokens += shard_length - _attention_overlap_count(
                attention_layout_index,
                rank,
                shard_start,
                shard_start + shard_length,
            )
        start = end
    return cross_rank_tokens


def _fla_aligned_chain_shard_lengths(length: int, *, cp_size: int) -> tuple[int, ...]:
    full_chunks = int(length) // FLA_CHUNK_SIZE
    if full_chunks < int(cp_size):
        raise ValueError(
            "CP chain planning requires at least one full FLA chunk per rank; "
            f"length={length} cp_size={cp_size}"
        )
    base_chunks = full_chunks // int(cp_size)
    extra_chunks = full_chunks % int(cp_size)
    chunk_counts = tuple(
        base_chunks + (1 if rank < extra_chunks else 0) for rank in range(int(cp_size))
    )
    lengths = [count * FLA_CHUNK_SIZE for count in chunk_counts]
    lengths[-1] += int(length) - full_chunks * FLA_CHUNK_SIZE
    return tuple(lengths)


def _attention_contiguous_chain_shards(
    token_start: int,
    token_count: int,
    *,
    cp_size: int,
    attention_layout_index: _AttentionLayoutIndex | None,
) -> tuple[range, ...] | None:
    if attention_layout_index is None:
        return None
    segment_end = token_start + token_count
    shards: list[range] = []
    cursor = token_start
    for rank in range(cp_size):
        overlap = _attention_single_contiguous_overlap(
            attention_layout_index,
            rank,
            token_start,
            segment_end,
        )
        if overlap is None:
            return None
        start, end = overlap
        if start != cursor or end <= start:
            return None
        shards.append(range(start, end))
        cursor = end
    if cursor != segment_end:
        return None
    if any(len(shard) % FLA_CHUNK_SIZE != 0 for shard in shards[:-1]):
        return None
    return tuple(shards)


def _attention_single_contiguous_overlap(
    index: _AttentionLayoutIndex,
    rank: int,
    start: int,
    end: int,
) -> tuple[int, int] | None:
    overlaps = _range_overlaps(start, end, index.token_ranges_by_rank[rank])
    if len(overlaps) != 1:
        return None
    return overlaps[0]


def _append_local_segment(
    gdn_ranges_by_rank: list[list[tuple[int, int, int]]],
    rank_loads: list[int],
    rank: int,
    segment: GdnSegmentSpec,
    spec: GdnPackedExecutionSpec,
    *,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
) -> int:
    token_start = _segment_token_start(segment, spec.sequence_length)
    position_start = rank_loads[rank]
    gdn_ranges_by_rank[rank].append(
        (token_start, token_start + segment.length, position_start)
    )
    rank_loads[rank] += segment.length
    return segment.length - segment_attention_counts[_segment_key(segment)][rank]


def _least_loaded_rank(rank_loads: list[int]) -> int:
    return min(range(len(rank_loads)), key=lambda rank: (rank_loads[rank], rank))


def _build_tree_segment_bucket_plans(
    segments: tuple[GdnSegmentSpec, ...],
    tree_parent_indices: tuple[int, ...],
    tree_has_children: tuple[bool, ...],
    *,
    device: torch.device | str,
    planner_config: GdnPlannerConfig,
) -> tuple[GdnSegmentBucketPlan, ...]:
    segment_buckets = _batch_tree_segments_by_padded_work(
        segments,
        tree_has_children,
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    plans = _build_segment_bucket_plans(segment_buckets, device=device)
    return tuple(
        _bucket_with_tree_parent_indices(
            plan,
            bucket,
            tree_parent_indices,
            tree_has_children,
            device=device,
        )
        for plan, bucket in zip(plans, segment_buckets, strict=True)
    )


def _build_tree_position_bucket_plans(
    segments: tuple[GdnSegmentSpec, ...],
    tree_parent_indices: tuple[int, ...],
    tree_has_children: tuple[bool, ...],
    local_token_ranges: tuple[tuple[int, int, int], ...],
    *,
    sequence_length: int,
    device: torch.device | str,
    planner_config: GdnPlannerConfig,
    token_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...] | None = None,
    split_by_final_state: bool = True,
) -> tuple[GdnSegmentBucketPlan, ...]:
    segment_buckets = (
        _batch_tree_segments_by_padded_work(
            segments,
            tree_has_children,
            max_padding_ratio=planner_config.max_padding_ratio,
            max_segments_per_batch=planner_config.max_segments_per_batch,
        )
        if split_by_final_state
        else _batch_segments_by_padded_work(
            segments,
            max_padding_ratio=planner_config.max_padding_ratio,
            max_segments_per_batch=planner_config.max_segments_per_batch,
        )
    )
    plans = _build_position_bucket_plans(
        segment_buckets,
        local_token_ranges,
        sequence_length=sequence_length,
        device=device,
        token_ranges_by_rank=token_ranges_by_rank,
    )
    return tuple(
        _bucket_with_tree_parent_indices(
            plan,
            bucket,
            tree_parent_indices,
            tree_has_children,
            device=device,
        )
        for plan, bucket in zip(plans, segment_buckets, strict=True)
    )


def _bucket_with_tree_parent_indices(
    plan: GdnSegmentBucketPlan,
    segments: tuple[GdnSegmentSpec, ...],
    tree_parent_indices: tuple[int, ...],
    tree_has_children: tuple[bool, ...],
    *,
    device: torch.device | str,
) -> GdnSegmentBucketPlan:
    parent_indices = torch.tensor(
        [tree_parent_indices[segment.family_index] for segment in segments],
        dtype=torch.long,
    )
    return plan.model_copy(
        update={
            "parent_indices": _move_planner_tensor(parent_indices, device),
            "parent_indices_cpu": parent_indices,
            "needs_final_state": any(
                tree_has_children[segment.family_index] for segment in segments
            ),
        }
    )


def _build_position_bucket_plans(
    segment_buckets: tuple[tuple[GdnSegmentSpec, ...], ...],
    local_token_ranges: tuple[tuple[int, int, int], ...],
    *,
    sequence_length: int,
    device: torch.device | str,
    token_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...] | None = None,
) -> tuple[GdnSegmentBucketPlan, ...]:
    return tuple(
        _build_position_bucket_plan(
            bucket,
            local_token_ranges,
            sequence_length=sequence_length,
            device=device,
            token_ranges_by_rank=token_ranges_by_rank,
        )
        for bucket in segment_buckets
    )


def _build_position_bucket_plan(
    segments: tuple[GdnSegmentSpec, ...],
    local_token_ranges: tuple[tuple[int, int, int], ...],
    *,
    sequence_length: int,
    device: torch.device | str,
    token_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...] | None = None,
) -> GdnSegmentBucketPlan:
    exact_plan = _build_exact_range_position_bucket_plan(
        segments,
        local_token_ranges,
        sequence_length=sequence_length,
        device=device,
        token_ranges_by_rank=token_ranges_by_rank,
    )
    if exact_plan is not None:
        return exact_plan
    local_positions_by_segment = []
    lengths = []
    local_range_ends = tuple(token_end for _, token_end, _ in local_token_ranges)
    for segment in segments:
        positions = _local_positions_for_segment(
            segment,
            sequence_length=sequence_length,
            local_token_ranges=local_token_ranges,
            local_range_ends=local_range_ends,
        )
        length = int(positions.numel())
        if not length:
            raise ValueError(
                "planned GDN bucket contains a segment with no local tokens; "
                f"family={segment.family_index} kind={segment.kind}"
            )
        local_positions_by_segment.append(positions)
        lengths.append(length)
    max_length = max(lengths)
    lengths_cpu = torch.tensor(lengths, dtype=torch.long)
    offsets_cpu = torch.arange(max_length, dtype=torch.long).unsqueeze(1)
    real_mask_cpu = offsets_cpu < lengths_cpu.unsqueeze(0)
    position_indices_cpu = torch.zeros(max_length, len(segments), dtype=torch.long)
    for column, positions in enumerate(local_positions_by_segment):
        position_indices_cpu[: int(positions.numel()), column] = positions
    cu_seqlens_cpu = torch.cat(
        [lengths_cpu.new_zeros(1), torch.cumsum(lengths_cpu, dim=0)]
    )
    lengths_by_rank_cpu = _bucket_lengths_by_rank_cpu(
        segments,
        token_ranges_by_rank,
        sequence_length=sequence_length,
    )
    row_indices_cpu = torch.zeros(max_length, len(segments), dtype=torch.long)
    family_indices_cpu = torch.tensor(
        [segment.family_index for segment in segments],
        dtype=torch.long,
    )
    return GdnSegmentBucketPlan.model_construct(
        length=max_length,
        lengths=_move_planner_tensor(lengths_cpu, device),
        lengths_cpu=lengths_cpu,
        lengths_by_rank_cpu=lengths_by_rank_cpu,
        real_mask=_move_planner_tensor(real_mask_cpu, device),
        cu_seqlens=_move_planner_tensor(cu_seqlens_cpu, device),
        cu_seqlens_cpu=cu_seqlens_cpu,
        row_indices=_move_planner_tensor(row_indices_cpu, device),
        position_indices=_move_planner_tensor(position_indices_cpu, device),
        family_indices=_move_planner_tensor(family_indices_cpu, device),
        family_indices_cpu=family_indices_cpu,
        real_token_count_static=sum(lengths),
    )


def _build_exact_range_position_bucket_plan(
    segments: tuple[GdnSegmentSpec, ...],
    local_token_ranges: tuple[tuple[int, int, int], ...],
    *,
    sequence_length: int,
    device: torch.device | str,
    token_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...] | None = None,
) -> GdnSegmentBucketPlan | None:
    range_positions = {
        (start, end): position for start, end, position in local_token_ranges
    }
    starts = []
    lengths = []
    for segment in segments:
        token_start = _segment_token_start(segment, sequence_length)
        token_end = token_start + segment.length
        position_start = range_positions.get((token_start, token_end))
        if position_start is None:
            return None
        starts.append(position_start)
        lengths.append(segment.length)
    max_length = max(lengths)
    starts_cpu = torch.tensor(starts, dtype=torch.long)
    lengths_cpu = torch.tensor(lengths, dtype=torch.long)
    offsets_cpu = torch.arange(max_length, dtype=torch.long).unsqueeze(1)
    real_mask_cpu = offsets_cpu < lengths_cpu.unsqueeze(0)
    position_indices_cpu = torch.where(
        real_mask_cpu,
        starts_cpu.unsqueeze(0) + offsets_cpu,
        torch.zeros_like(offsets_cpu),
    )
    cu_seqlens_cpu = torch.cat(
        [lengths_cpu.new_zeros(1), torch.cumsum(lengths_cpu, dim=0)]
    )
    lengths_by_rank_cpu = _bucket_lengths_by_rank_cpu(
        segments,
        token_ranges_by_rank,
        sequence_length=sequence_length,
    )
    row_indices_cpu = torch.zeros(max_length, len(segments), dtype=torch.long)
    family_indices_cpu = torch.tensor(
        [segment.family_index for segment in segments],
        dtype=torch.long,
    )
    return GdnSegmentBucketPlan.model_construct(
        length=max_length,
        lengths=_move_planner_tensor(lengths_cpu, device),
        lengths_cpu=lengths_cpu,
        lengths_by_rank_cpu=lengths_by_rank_cpu,
        real_mask=_move_planner_tensor(real_mask_cpu, device),
        cu_seqlens=_move_planner_tensor(cu_seqlens_cpu, device),
        cu_seqlens_cpu=cu_seqlens_cpu,
        row_indices=_move_planner_tensor(row_indices_cpu, device),
        position_indices=_move_planner_tensor(position_indices_cpu, device),
        family_indices=_move_planner_tensor(family_indices_cpu, device),
        family_indices_cpu=family_indices_cpu,
        real_token_count_static=sum(lengths),
    )


def _bucket_lengths_by_rank_cpu(
    segments: tuple[GdnSegmentSpec, ...],
    token_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...] | None,
    *,
    sequence_length: int,
) -> torch.Tensor | None:
    if token_ranges_by_rank is None:
        return None
    lengths_by_rank = []
    for rank_ranges_with_positions in token_ranges_by_rank:
        rank_ranges = tuple(
            (start, end) for start, end, _position in rank_ranges_with_positions
        )
        rank_lengths = []
        for segment in segments:
            start = _segment_token_start(segment, sequence_length)
            end = start + segment.length
            rank_lengths.append(
                sum(
                    max(0, min(end, range_end) - max(start, range_start))
                    for range_start, range_end in rank_ranges
                )
            )
        lengths_by_rank.append(rank_lengths)
    return torch.tensor(lengths_by_rank, dtype=torch.long)


def _move_planner_tensor(
    tensor: torch.Tensor, device: torch.device | str
) -> torch.Tensor:
    target = torch.device(device)
    if target.type == "cpu":
        return tensor
    return tensor.to(device=target)


def _batch_segments_by_padded_work(
    segments: tuple[GdnSegmentSpec, ...],
    *,
    max_padding_ratio: float = 1.25,
    max_segments_per_batch: int = 128,
) -> tuple[tuple[GdnSegmentSpec, ...], ...]:
    if not segments:
        return ()
    ordered = sorted(
        segments, key=lambda segment: (segment.length, segment.family_index)
    )
    batches: list[list[GdnSegmentSpec]] = []
    current: list[GdnSegmentSpec] = []
    current_tokens = 0
    current_max = 0
    for segment in ordered:
        next_count = len(current) + 1
        next_tokens = current_tokens + segment.length
        next_max = max(current_max, segment.length)
        padded = next_max * next_count
        can_extend = not current or (
            next_count <= max_segments_per_batch
            and padded <= max_padding_ratio * next_tokens
        )
        if not can_extend:
            batches.append(current)
            current = []
            current_tokens = 0
            current_max = 0
        current.append(segment)
        current_tokens += segment.length
        current_max = max(current_max, segment.length)
    if current:
        batches.append(current)
    return tuple(tuple(batch) for batch in batches)


def _batch_tree_segments_by_padded_work(
    segments: tuple[GdnSegmentSpec, ...],
    tree_has_children: tuple[bool, ...],
    *,
    max_padding_ratio: float = 1.25,
    max_segments_per_batch: int = 128,
) -> tuple[tuple[GdnSegmentSpec, ...], ...]:
    stateful = tuple(
        segment for segment in segments if tree_has_children[segment.family_index]
    )
    stateless = tuple(
        segment for segment in segments if not tree_has_children[segment.family_index]
    )
    return (
        *_batch_segments_by_padded_work(
            stateful,
            max_padding_ratio=max_padding_ratio,
            max_segments_per_batch=max_segments_per_batch,
        ),
        *_batch_segments_by_padded_work(
            stateless,
            max_padding_ratio=max_padding_ratio,
            max_segments_per_batch=max_segments_per_batch,
        ),
    )


def _build_segment_bucket_plan(
    length: int, segments: tuple[GdnSegmentSpec, ...], *, device: torch.device | str
) -> GdnSegmentBucketPlan:
    max_length = max(segment.length for segment in segments)
    lengths_cpu = torch.tensor(
        [segment.length for segment in segments], dtype=torch.long
    )
    starts_cpu = torch.tensor([segment.start for segment in segments], dtype=torch.long)
    rows_cpu = torch.tensor(
        [segment.row_index for segment in segments], dtype=torch.long
    )
    offsets_cpu = torch.arange(max_length, dtype=torch.long).unsqueeze(1)
    real_mask_cpu = offsets_cpu < lengths_cpu.unsqueeze(0)
    positions_cpu = starts_cpu.unsqueeze(0) + offsets_cpu
    family_indices_cpu = torch.tensor(
        [segment.family_index for segment in segments],
        dtype=torch.long,
    )
    cu_seqlens_cpu = torch.cat(
        [lengths_cpu.new_zeros(1), torch.cumsum(lengths_cpu, dim=0)]
    )
    return GdnSegmentBucketPlan.model_construct(
        length=max_length,
        lengths=_move_planner_tensor(lengths_cpu, device),
        lengths_cpu=lengths_cpu,
        lengths_by_rank_cpu=None,
        real_mask=_move_planner_tensor(real_mask_cpu, device),
        cu_seqlens=_move_planner_tensor(cu_seqlens_cpu, device),
        cu_seqlens_cpu=cu_seqlens_cpu,
        row_indices=_move_planner_tensor(
            rows_cpu.unsqueeze(0).expand(max_length, -1).contiguous(), device
        ),
        position_indices=_move_planner_tensor(positions_cpu, device),
        family_indices=_move_planner_tensor(family_indices_cpu, device),
        family_indices_cpu=family_indices_cpu,
        real_token_count_static=sum(segment.length for segment in segments),
    )


def _segment_token_start(segment: GdnSegmentSpec, sequence_length: int) -> int:
    return segment.row_index * sequence_length + segment.start


def _attention_overlap_count(
    index: _AttentionLayoutIndex,
    rank: int,
    start: int,
    end: int,
) -> int:
    return _range_overlap_count(
        start,
        end,
        index.token_ranges_by_rank[rank],
        index.token_range_ends_by_rank[rank],
    )


def _range_overlap_count(
    start: int,
    end: int,
    ranges: tuple[tuple[int, int], ...],
    range_ends: tuple[int, ...],
) -> int:
    count = 0
    range_index = bisect_left(range_ends, start + 1)
    for range_start, range_end in ranges[range_index:]:
        if range_start >= end:
            break
        count += min(end, range_end) - max(start, range_start)
    return count


def _range_overlaps(
    start: int,
    end: int,
    ranges: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    overlaps = [
        (max(start, range_start), min(end, range_end))
        for range_start, range_end in ranges
        if max(start, range_start) < min(end, range_end)
    ]
    overlaps.sort()
    return overlaps


def _local_positions_for_segment(
    segment: GdnSegmentSpec,
    *,
    sequence_length: int,
    local_token_ranges: tuple[tuple[int, int, int], ...],
    local_range_ends: tuple[int, ...],
) -> torch.Tensor:
    segment_start = _segment_token_start(segment, sequence_length)
    segment_end = segment_start + segment.length
    pieces = []
    range_index = bisect_left(local_range_ends, segment_start + 1)
    for token_start, token_end, position_start in local_token_ranges[range_index:]:
        if token_start >= segment_end:
            break
        overlap_start = max(segment_start, token_start)
        overlap_end = min(segment_end, token_end)
        if overlap_start >= overlap_end:
            continue
        pieces.append(
            torch.arange(
                position_start + overlap_start - token_start,
                position_start + overlap_end - token_start,
                dtype=torch.long,
            )
        )
    if not pieces:
        return torch.empty((0,), dtype=torch.long)
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces)


def _rank2_long_cpu(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank 2 [batch, sequence], got {tensor.ndim}")
    if tensor.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
    ):
        raise TypeError(f"{name} must contain integer ids, got dtype={tensor.dtype}")
    return tensor.detach().to(device="cpu", dtype=torch.long)
