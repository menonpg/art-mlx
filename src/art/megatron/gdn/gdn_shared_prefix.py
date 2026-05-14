from __future__ import annotations

from bisect import bisect_left
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field
import torch

from art.megatron.context_parallel.layout_index import TokenLayoutIndex

GdnSegmentKind = Literal["prefix", "completion"]
GdnSegmentDecisionKey = tuple[int, int, int]
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


class GdnPackedFamilySpec(BaseModel):
    """One shared-prefix family plus child completion segments."""

    model_config = ConfigDict(frozen=True)

    row_index: int = Field(ge=0)
    family_index: int = Field(ge=0)
    prefix: GdnSegmentSpec
    completions: tuple[GdnSegmentSpec, ...]

    @property
    def completion_count(self) -> int:
        return len(self.completions)

    @property
    def token_count(self) -> int:
        return self.prefix.length + sum(segment.length for segment in self.completions)


class GdnPackedExecutionSpec(BaseModel):
    """Parsed shared-prefix GDN execution metadata for a packed batch."""

    model_config = ConfigDict(frozen=True)

    batch_size: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    valid_lengths: tuple[int, ...]
    families: tuple[GdnPackedFamilySpec, ...]

    @property
    def family_count(self) -> int:
        return len(self.families)

    @property
    def completion_count(self) -> int:
        return sum(family.completion_count for family in self.families)

    @property
    def real_token_count(self) -> int:
        return sum(self.valid_lengths)

    @property
    def max_segment_length(self) -> int:
        lengths = [
            segment.length
            for family in self.families
            for segment in (family.prefix, *family.completions)
        ]
        return max(lengths, default=0)

    def segments(self) -> tuple[GdnSegmentSpec, ...]:
        return tuple(
            segment
            for family in self.families
            for segment in (family.prefix, *family.completions)
        )


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
_GDN_PACKED_FAMILY_SPEC_FIELDS = frozenset(
    {
        "row_index",
        "family_index",
        "prefix",
        "completions",
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
    real_token_count_static: int = Field(ge=0)
    output_mask: torch.Tensor | None = None

    @property
    def segment_count(self) -> int:
        return int(self.family_indices.numel())

    @property
    def real_token_count(self) -> int:
        return self.real_token_count_static


class GdnParentStateTransferPlan(BaseModel):
    """Prefix-state rows transferred from one CP rank to another."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source_rank: int = Field(ge=0)
    dest_rank: int = Field(ge=0)
    family_indices: tuple[int, ...]
    family_indices_tensor: torch.Tensor | None = None


class GdnPlannerConfig(BaseModel):
    """Tunable cost coefficients for one packed-row GDN execution plan."""

    model_config = ConfigDict(frozen=True)

    max_padding_ratio: float = Field(default=2.0, gt=1.0)
    max_segments_per_batch: int = Field(default=4096, ge=1)
    cp_chain_min_tokens_per_rank: int = Field(default=32, ge=1)
    cp_chain_min_total_tokens: int = Field(default=32768, ge=1)
    cp_chain_min_prefix_only_tokens: int = Field(default=32768, ge=1)
    local_fork_launch_penalty_tokens: int = Field(default=256, ge=0)
    cp_collective_latency_tokens: int = Field(default=512, ge=0)
    parent_state_exchange_penalty_tokens: int = Field(default=16384, ge=0)
    layout_cross_rank_token_cost: float = Field(default=6.0, ge=0.0)
    rank_idle_token_cost: float = Field(default=1.0, ge=0.0)
    empty_rank_penalty_tokens: int = Field(default=65536, ge=0)
    max_zero_exchange_load_imbalance: float = Field(default=1.5, ge=1.0)
    local_completion_rebalance_min_imbalance: float = Field(default=1.08, ge=1.0)
    cp_chain_beam_width: int = Field(default=2, ge=1)
    cp_chain_beam_branch_factor: int = Field(default=4, ge=1)
    cp_chain_beam_candidate_limit: int = Field(default=16, ge=1)
    cp_chain_beam_max_steps: int = Field(default=4, ge=0)
    cp_chain_beam_min_score_delta_tokens: float = Field(default=512.0, ge=0.0)
    cp_chain_min_score_delta_ms: float = Field(default=0.25, ge=0.0)
    planner_local_token_ms: float = Field(default=0.00065, ge=0.0)
    planner_chain_token_ms: float = Field(default=0.00055, ge=0.0)
    planner_local_bucket_ms: float = Field(default=0.25, ge=0.0)
    planner_chain_bucket_ms: float = Field(default=22.0, ge=0.0)
    planner_local_segment_ms: float = Field(default=0.010, ge=0.0)
    planner_layout_cross_rank_token_ms: float = Field(default=0.00008, ge=0.0)
    planner_parent_state_exchange_base_ms: float = Field(default=40.0, ge=0.0)
    planner_parent_state_exchange_ms: float = Field(default=0.5, ge=0.0)
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
    local_prefix_buckets: tuple[GdnSegmentBucketPlan, ...] = ()
    local_completion_buckets: tuple[GdnSegmentBucketPlan, ...] = ()
    ready_local_completion_buckets: tuple[GdnSegmentBucketPlan, ...] = ()
    remote_local_completion_buckets: tuple[GdnSegmentBucketPlan, ...] = ()
    chain_prefix_buckets: tuple[GdnSegmentBucketPlan, ...] = ()
    chain_completion_buckets: tuple[GdnSegmentBucketPlan, ...] = ()
    prefix_table_is_dense_ordered: bool
    attention_to_gdn: Any | None = None
    gdn_to_attention: Any | None = None
    attention_token_ranges: tuple[tuple[int, int, int], ...] = ()
    gdn_token_ranges: tuple[tuple[int, int, int], ...] = ()
    attention_token_count: int = Field(default=0, ge=0)
    gdn_token_count: int = Field(default=0, ge=0)
    parent_state_exchange_family_indices: tuple[int, ...] = ()
    parent_state_transfers: tuple[GdnParentStateTransferPlan, ...] = ()
    prefix_boundary_buckets: tuple[GdnSegmentBucketPlan, ...] = ()
    prefix_tail_buckets: tuple[GdnSegmentBucketPlan, ...] = ()
    completion_with_prefix_tail_buckets: tuple[GdnSegmentBucketPlan, ...] = ()
    remote_prefix_tail_buckets: tuple[GdnSegmentBucketPlan, ...] = ()
    remote_completion_with_prefix_tail_buckets: tuple[GdnSegmentBucketPlan, ...] = ()
    remote_prefix_tail_exchange: Any | None = None
    remote_prefix_tail_backward_exchange: Any | None = None
    remote_prefix_tail_state_transfers: tuple[GdnParentStateTransferPlan, ...] = ()

    @property
    def attention_token_indices(self) -> tuple[int, ...]:
        return _tokens_from_rank_ranges(self.attention_token_ranges)

    @property
    def gdn_token_indices(self) -> tuple[int, ...]:
        return _tokens_from_rank_ranges(self.gdn_token_ranges)


class GdnCpSegmentSchedule(BaseModel):
    """CPU-side ownership and bucket schedule for one CP GDN plan."""

    model_config = ConfigDict(frozen=True)

    gdn_token_counts_by_rank: tuple[int, ...]
    gdn_token_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...] = ()
    cross_rank_token_count: int = Field(ge=0)
    chain_prefix_buckets: tuple[tuple[GdnSegmentSpec, ...], ...]
    chain_completion_buckets: tuple[tuple[GdnSegmentSpec, ...], ...]
    local_prefix_segments_by_rank: tuple[tuple[GdnSegmentSpec, ...], ...]
    local_completion_segments_by_rank: tuple[tuple[GdnSegmentSpec, ...], ...]
    parent_state_exchange_family_indices: tuple[int, ...] = ()
    parent_state_transfers: tuple[GdnParentStateTransferPlan, ...] = ()


class _GdnCpSegmentSearchDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    chain_segment_keys: frozenset[GdnSegmentDecisionKey]
    co_locate_local_families: bool
    score: float


class _ExplicitBucketColumn(BaseModel):
    model_config = ConfigDict(frozen=True)

    row_index: int
    family_index: int
    positions: tuple[int, ...]
    output_mask: tuple[bool, ...]

    @property
    def length(self) -> int:
        return len(self.positions)


def _explicit_bucket_column(
    *,
    row_index: int,
    family_index: int,
    positions: tuple[int, ...],
    output_mask: tuple[bool, ...],
) -> _ExplicitBucketColumn:
    return _ExplicitBucketColumn.model_construct(
        row_index=row_index,
        family_index=family_index,
        positions=positions,
        output_mask=output_mask,
    )


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
    cp_segment_schedule: GdnCpSegmentSchedule | None = None,
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
            cp_segment_schedule=cp_segment_schedule,
            planner_config=planner_config,
        )
        return move_gdn_rank_execution_plan_to_device(cpu_plan, target_device)
    if cp_size != 1 or cp_rank != 0:
        return _build_cp_rank_execution_plan(
            spec,
            device=device,
            cp_rank=cp_rank,
            cp_size=cp_size,
            attention_token_layout_index=attention_token_layout_index,
            cp_segment_schedule=cp_segment_schedule,
            planner_config=planner_config,
        )
    (
        prefix_boundary_buckets,
        prefix_tail_buckets,
        completion_with_prefix_tail_buckets,
    ) = _build_chunk_aligned_cp1_bucket_plans(
        spec,
        device=device,
        planner_config=planner_config,
    )
    valid_lengths = torch.tensor(
        spec.valid_lengths,
        device=device,
        dtype=torch.long,
    )
    positions = torch.arange(spec.sequence_length, device=device, dtype=torch.long)
    local_range_list: list[tuple[int, int, int]] = []
    local_position = 0
    for row_index, length in enumerate(spec.valid_lengths):
        if length:
            start = row_index * spec.sequence_length
            local_range_list.append((start, start + length, local_position))
            local_position += length
    local_ranges = tuple(local_range_list)
    return GdnRankExecutionPlan.model_construct(
        cp_rank=cp_rank,
        cp_size=cp_size,
        batch_size=spec.batch_size,
        sequence_length=spec.sequence_length,
        packed_batch_size=spec.batch_size,
        packed_sequence_length=spec.sequence_length,
        real_token_mask=positions.unsqueeze(0) < valid_lengths.unsqueeze(1),
        family_count=spec.family_count,
        completion_count=spec.completion_count,
        local_prefix_buckets=(),
        local_completion_buckets=(),
        ready_local_completion_buckets=(),
        remote_local_completion_buckets=(),
        chain_prefix_buckets=(),
        chain_completion_buckets=(),
        prefix_table_is_dense_ordered=False,
        attention_token_ranges=local_ranges,
        gdn_token_ranges=local_ranges,
        attention_token_count=spec.real_token_count,
        gdn_token_count=spec.real_token_count,
        prefix_boundary_buckets=prefix_boundary_buckets,
        prefix_tail_buckets=prefix_tail_buckets,
        completion_with_prefix_tail_buckets=completion_with_prefix_tail_buckets,
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
        local_prefix_buckets=_move_bucket_plans(plan.local_prefix_buckets, device),
        local_completion_buckets=_move_bucket_plans(
            plan.local_completion_buckets, device
        ),
        ready_local_completion_buckets=_move_bucket_plans(
            plan.ready_local_completion_buckets, device
        ),
        remote_local_completion_buckets=_move_bucket_plans(
            plan.remote_local_completion_buckets, device
        ),
        chain_prefix_buckets=_move_bucket_plans(plan.chain_prefix_buckets, device),
        chain_completion_buckets=_move_bucket_plans(
            plan.chain_completion_buckets, device
        ),
        prefix_table_is_dense_ordered=plan.prefix_table_is_dense_ordered,
        attention_to_gdn=move_cp_exchange_plan_to_device(plan.attention_to_gdn, device),
        gdn_to_attention=move_cp_exchange_plan_to_device(plan.gdn_to_attention, device),
        attention_token_ranges=plan.attention_token_ranges,
        gdn_token_ranges=plan.gdn_token_ranges,
        attention_token_count=plan.attention_token_count,
        gdn_token_count=plan.gdn_token_count,
        parent_state_exchange_family_indices=plan.parent_state_exchange_family_indices,
        parent_state_transfers=_move_parent_state_transfers(
            plan.parent_state_transfers, device
        ),
        prefix_boundary_buckets=_move_bucket_plans(
            plan.prefix_boundary_buckets, device
        ),
        prefix_tail_buckets=_move_bucket_plans(plan.prefix_tail_buckets, device),
        completion_with_prefix_tail_buckets=_move_bucket_plans(
            plan.completion_with_prefix_tail_buckets, device
        ),
        remote_prefix_tail_buckets=_move_bucket_plans(
            plan.remote_prefix_tail_buckets, device
        ),
        remote_completion_with_prefix_tail_buckets=_move_bucket_plans(
            plan.remote_completion_with_prefix_tail_buckets, device
        ),
        remote_prefix_tail_exchange=move_cp_exchange_plan_to_device(
            plan.remote_prefix_tail_exchange, device
        ),
        remote_prefix_tail_backward_exchange=move_cp_exchange_plan_to_device(
            plan.remote_prefix_tail_backward_exchange, device
        ),
        remote_prefix_tail_state_transfers=_move_parent_state_transfers(
            plan.remote_prefix_tail_state_transfers, device
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
            real_token_count_static=bucket.real_token_count,
            output_mask=(
                _move_planner_tensor(bucket.output_mask, device)
                if bucket.output_mask is not None
                else None
            ),
        )
        for bucket in buckets
    )


def _move_parent_state_transfers(
    transfers: tuple[GdnParentStateTransferPlan, ...],
    device: torch.device | str,
) -> tuple[GdnParentStateTransferPlan, ...]:
    return tuple(
        GdnParentStateTransferPlan.model_construct(
            source_rank=transfer.source_rank,
            dest_rank=transfer.dest_rank,
            family_indices=transfer.family_indices,
            family_indices_tensor=(
                _move_planner_tensor(transfer.family_indices_tensor, device)
                if transfer.family_indices_tensor is not None
                else None
            ),
        )
        for transfer in transfers
    )


def build_gdn_chain_only_rank_execution_plan(
    spec: GdnPackedExecutionSpec,
    *,
    device: torch.device | str,
    cp_rank: int,
    cp_size: int,
    planner_config: GdnPlannerConfig | None = None,
) -> GdnRankExecutionPlan | None:
    """Build the rank-local plan for rows that are entirely native CP chains.

    This avoids a large Python-object schedule broadcast for long pure-chain rows
    such as `64k + 8x64k`. Mixed local/chain rows still use the general planner.
    """

    planner_config = planner_config or GdnPlannerConfig()
    if cp_size <= 1:
        return None
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
    if not spec.families:
        return None
    for family in spec.families:
        if not _can_chain_prefix_segment(
            family.prefix, cp_size=cp_size, planner_config=planner_config
        ):
            return None
        if any(
            not _can_chain_segment(
                completion, cp_size=cp_size, planner_config=planner_config
            )
            for completion in family.completions
        ):
            return None

    from art.megatron.gdn.layout import GdnCpExchangePlan, GdnCpPeerTransfer

    local_tokens: list[int] = []
    prefix_segments: list[GdnSegmentSpec] = []
    completion_segments: list[GdnSegmentSpec] = []
    token_ranges_by_rank = []
    for rank in range(cp_size):
        rank_tokens = []
        for family in spec.families:
            rank_tokens.extend(
                _chain_rank_token_indices(
                    family.prefix,
                    spec,
                    cp_rank=rank,
                    cp_size=cp_size,
                )
            )
            for completion in family.completions:
                rank_tokens.extend(
                    _chain_rank_token_indices(
                        completion,
                        spec,
                        cp_rank=rank,
                        cp_size=cp_size,
                    )
                )
        token_ranges_by_rank.append(_local_token_ranges(tuple(rank_tokens)))
    for family in spec.families:
        prefix_segments.append(family.prefix)
        local_tokens.extend(
            _chain_rank_token_indices(
                family.prefix,
                spec,
                cp_rank=cp_rank,
                cp_size=cp_size,
            )
        )
        for completion in family.completions:
            completion_segments.append(completion)
            local_tokens.extend(
                _chain_rank_token_indices(
                    completion,
                    spec,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
            )
    local_token_tuple = tuple(local_tokens)
    local_token_ranges = _local_token_ranges(local_token_tuple)
    token_counts_by_rank = tuple(
        len(local_token_tuple) if rank == cp_rank else 0 for rank in range(cp_size)
    )
    identity_exchange = GdnCpExchangePlan.model_construct(
        cp_size=cp_size,
        source_token_counts_by_rank=token_counts_by_rank,
        dest_token_counts_by_rank=token_counts_by_rank,
        transfers=tuple(
            GdnCpPeerTransfer.model_construct(
                source_rank=rank,
                dest_rank=rank,
                token_count=count,
                source_positions_tensor=None,
                dest_positions_tensor=None,
            )
            for rank, count in enumerate(token_counts_by_rank)
            if count
        ),
    )
    chain_prefix_buckets = _batch_segments_by_padded_work(
        tuple(prefix_segments),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    chain_completion_buckets = _batch_segments_by_padded_work(
        tuple(completion_segments),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    prefix_family_order = tuple(
        segment.family_index for bucket in chain_prefix_buckets for segment in bucket
    )
    return GdnRankExecutionPlan.model_construct(
        cp_rank=cp_rank,
        cp_size=cp_size,
        batch_size=1,
        sequence_length=len(local_token_tuple),
        packed_batch_size=spec.batch_size,
        packed_sequence_length=spec.sequence_length,
        real_token_mask=torch.ones(
            1, len(local_token_tuple), device=device, dtype=torch.bool
        ),
        family_count=spec.family_count,
        completion_count=spec.completion_count,
        local_prefix_buckets=(),
        local_completion_buckets=(),
        ready_local_completion_buckets=(),
        remote_local_completion_buckets=(),
        chain_prefix_buckets=_build_position_bucket_plans(
            chain_prefix_buckets,
            local_token_ranges,
            sequence_length=spec.sequence_length,
            device=device,
            token_ranges_by_rank=tuple(token_ranges_by_rank),
        ),
        chain_completion_buckets=_build_position_bucket_plans(
            chain_completion_buckets,
            local_token_ranges,
            sequence_length=spec.sequence_length,
            device=device,
            token_ranges_by_rank=tuple(token_ranges_by_rank),
        ),
        prefix_table_is_dense_ordered=(
            prefix_family_order == tuple(range(spec.family_count))
        ),
        attention_to_gdn=identity_exchange,
        gdn_to_attention=identity_exchange,
        attention_token_ranges=local_token_ranges,
        gdn_token_ranges=local_token_ranges,
        attention_token_count=len(local_token_tuple),
        gdn_token_count=len(local_token_tuple),
        parent_state_exchange_family_indices=(),
        parent_state_transfers=(),
    )


def _build_chain_attention_layout_rank_execution_plan(
    spec: GdnPackedExecutionSpec,
    *,
    device: torch.device | str,
    cp_rank: int,
    cp_size: int,
    attention_token_layout_index: TokenLayoutIndex | None,
    planner_config: GdnPlannerConfig,
) -> GdnRankExecutionPlan | None:
    if cp_size <= 1 or not spec.families:
        return None
    for family in spec.families:
        if not _can_chain_prefix_segment(
            family.prefix, cp_size=cp_size, planner_config=planner_config
        ):
            return None
        if any(
            not _can_chain_segment(
                completion, cp_size=cp_size, planner_config=planner_config
            )
            for completion in family.completions
        ):
            return None

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
        max_ranges=max(1, 2 * spec.real_token_count // len(tuple(spec.segments()))),
    )
    rank_loads = [0] * cp_size
    gdn_ranges_by_rank: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
    prefix_segments: list[GdnSegmentSpec] = []
    completion_segments: list[GdnSegmentSpec] = []
    cross_rank_token_count = 0
    for family in spec.families:
        for segment in (family.prefix, *family.completions):
            if segment.kind == "prefix":
                prefix_segments.append(segment)
            else:
                completion_segments.append(segment)
            token_start = _segment_token_start(segment, spec.sequence_length)
            shards = _attention_contiguous_chain_shards(
                token_start,
                segment.length,
                cp_size=cp_size,
                attention_layout_index=attention_layout_index,
            )
            if shards is None:
                shards = tuple(
                    _chain_rank_token_indices(
                        segment,
                        spec,
                        cp_rank=rank,
                        cp_size=cp_size,
                    )
                    for rank in range(cp_size)
                )
            for rank, shard in enumerate(shards):
                position_start = rank_loads[rank]
                gdn_ranges_by_rank[rank].append(
                    (shard.start, shard.stop, position_start)
                )
                rank_loads[rank] += len(shard)
                cross_rank_token_count += len(shard) - _attention_overlap_count(
                    attention_layout_index,
                    rank,
                    shard.start,
                    shard.stop,
                )
    local_token_ranges = tuple(gdn_ranges_by_rank[cp_rank])
    local_token_count = rank_loads[cp_rank]
    attention_to_gdn = build_local_rank_cp_exchange_plan_from_dest_ranges(
        source_layout=source_layout,
        device=device,
        dest_ranges_by_rank=tuple(tuple(ranges) for ranges in gdn_ranges_by_rank),
        local_rank=cp_rank,
        cross_rank_token_count=cross_rank_token_count,
    )
    gdn_to_attention = _reverse_exchange_plan(attention_to_gdn)
    chain_prefix_buckets = _batch_segments_by_padded_work(
        tuple(prefix_segments),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    chain_completion_buckets = _batch_segments_by_padded_work(
        tuple(completion_segments),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    prefix_family_order = tuple(
        segment.family_index for bucket in chain_prefix_buckets for segment in bucket
    )
    return GdnRankExecutionPlan.model_construct(
        cp_rank=cp_rank,
        cp_size=cp_size,
        batch_size=1,
        sequence_length=local_token_count,
        packed_batch_size=spec.batch_size,
        packed_sequence_length=spec.sequence_length,
        real_token_mask=torch.ones(
            1, local_token_count, device=device, dtype=torch.bool
        ),
        family_count=spec.family_count,
        completion_count=spec.completion_count,
        local_prefix_buckets=(),
        local_completion_buckets=(),
        ready_local_completion_buckets=(),
        remote_local_completion_buckets=(),
        chain_prefix_buckets=_build_position_bucket_plans(
            chain_prefix_buckets,
            local_token_ranges,
            sequence_length=spec.sequence_length,
            device=device,
            token_ranges_by_rank=tuple(tuple(ranges) for ranges in gdn_ranges_by_rank),
        ),
        chain_completion_buckets=_build_position_bucket_plans(
            chain_completion_buckets,
            local_token_ranges,
            sequence_length=spec.sequence_length,
            device=device,
            token_ranges_by_rank=tuple(tuple(ranges) for ranges in gdn_ranges_by_rank),
        ),
        prefix_table_is_dense_ordered=(
            prefix_family_order == tuple(range(spec.family_count))
        ),
        attention_to_gdn=attention_to_gdn,
        gdn_to_attention=gdn_to_attention,
        attention_token_ranges=source_layout.ownership_ranges_by_rank[cp_rank],
        gdn_token_ranges=local_token_ranges,
        attention_token_count=source_layout.token_counts_by_rank[cp_rank],
        gdn_token_count=local_token_count,
        parent_state_exchange_family_indices=(),
        parent_state_transfers=(),
    )


def _build_local_attention_layout_rank_execution_plan(
    spec: GdnPackedExecutionSpec,
    *,
    device: torch.device | str,
    cp_rank: int,
    cp_size: int,
    attention_token_layout_index: TokenLayoutIndex | None,
    planner_config: GdnPlannerConfig,
) -> GdnRankExecutionPlan | None:
    if cp_size <= 1 or not spec.families:
        return None
    if any(
        _has_chainable_segment(family, cp_size=cp_size, planner_config=planner_config)
        for family in spec.families
    ):
        return None

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
        max_ranges=max(1, 2 * spec.real_token_count // len(tuple(spec.segments()))),
    )
    segment_attention_counts = _segment_attention_rank_counts(
        spec,
        cp_size=cp_size,
        attention_layout_index=attention_layout_index,
    )
    best = _assign_local_attention_segments(
        spec,
        cp_size=cp_size,
        segment_attention_counts=segment_attention_counts,
        co_locate_local_families=False,
        planner_config=planner_config,
    )
    co_located = _assign_local_attention_segments(
        spec,
        cp_size=cp_size,
        segment_attention_counts=segment_attention_counts,
        co_locate_local_families=True,
        planner_config=planner_config,
    )
    if co_located[4] < best[4]:
        best = co_located
    (
        prefix_owner_by_family,
        completion_owners_by_family,
        _,
        cross_rank_token_count,
        _,
    ) = best

    local_prefix_segments: list[GdnSegmentSpec] = []
    local_completion_segments: list[GdnSegmentSpec] = []
    prefix_segments_by_rank: list[list[GdnSegmentSpec]] = [[] for _ in range(cp_size)]
    completion_segments_by_rank: list[list[GdnSegmentSpec]] = [
        [] for _ in range(cp_size)
    ]
    gdn_ranges_by_rank: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
    rank_loads = [0] * cp_size
    parent_state_exchange_families: set[int] = set()
    parent_state_transfer_families: dict[tuple[int, int], set[int]] = {}

    def append_segment(rank: int, segment: GdnSegmentSpec) -> None:
        token_start = _segment_token_start(segment, spec.sequence_length)
        position_start = rank_loads[rank]
        gdn_ranges_by_rank[rank].append(
            (token_start, token_start + segment.length, position_start)
        )
        rank_loads[rank] += segment.length

    for family in spec.families:
        prefix_owner = prefix_owner_by_family[family.family_index]
        if prefix_owner == cp_rank:
            local_prefix_segments.append(family.prefix)
        prefix_segments_by_rank[prefix_owner].append(family.prefix)
        append_segment(prefix_owner, family.prefix)
        completion_owners = completion_owners_by_family[family.family_index]
        for completion, completion_owner in zip(
            family.completions, completion_owners, strict=True
        ):
            if completion_owner == cp_rank:
                local_completion_segments.append(completion)
            completion_segments_by_rank[completion_owner].append(completion)
            append_segment(completion_owner, completion)
            if completion_owner != prefix_owner:
                parent_state_exchange_families.add(family.family_index)
                parent_state_transfer_families.setdefault(
                    (prefix_owner, completion_owner), set()
                ).add(family.family_index)

    local_token_ranges = tuple(gdn_ranges_by_rank[cp_rank])
    local_token_count = rank_loads[cp_rank]
    schedule = GdnCpSegmentSchedule.model_construct(
        gdn_token_counts_by_rank=tuple(rank_loads),
        gdn_token_ranges_by_rank=tuple(tuple(ranges) for ranges in gdn_ranges_by_rank),
        cross_rank_token_count=cross_rank_token_count,
        chain_prefix_buckets=(),
        chain_completion_buckets=(),
        local_prefix_segments_by_rank=tuple(
            tuple(segments) for segments in prefix_segments_by_rank
        ),
        local_completion_segments_by_rank=tuple(
            tuple(segments) for segments in completion_segments_by_rank
        ),
        parent_state_exchange_family_indices=tuple(
            sorted(parent_state_exchange_families)
        ),
        parent_state_transfers=_build_parent_state_transfer_plans(
            parent_state_transfer_families
        ),
    )
    if parent_state_transfer_families:
        (
            remote_prefix_tail_buckets,
            remote_completion_with_prefix_tail_buckets,
            remote_prefix_tail_exchange,
            remote_prefix_tail_backward_exchange,
            remote_prefix_tail_state_transfers,
            remote_prefix_tail_families,
        ) = _build_remote_prefix_tail_plans(
            spec,
            schedule,
            cp_rank=cp_rank,
            device=device,
            planner_config=planner_config,
        )
    else:
        (
            remote_prefix_tail_buckets,
            remote_completion_with_prefix_tail_buckets,
            remote_prefix_tail_exchange,
            remote_prefix_tail_backward_exchange,
            remote_prefix_tail_state_transfers,
            remote_prefix_tail_families,
        ) = _empty_remote_prefix_tail_plans()
    attention_to_gdn = build_local_rank_cp_exchange_plan_from_dest_ranges(
        source_layout=source_layout,
        device=device,
        dest_ranges_by_rank=tuple(tuple(ranges) for ranges in gdn_ranges_by_rank),
        local_rank=cp_rank,
        cross_rank_token_count=cross_rank_token_count,
    )
    gdn_to_attention = _reverse_exchange_plan(attention_to_gdn)
    local_prefix_family_indices = {
        segment.family_index for segment in local_prefix_segments
    }
    local_prefix_buckets = _batch_segments_by_padded_work(
        (),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    chunk_local_completion_segments = tuple(
        segment
        for segment in local_completion_segments
        if segment.family_index in local_prefix_family_indices
    )
    plain_local_completion_segments = tuple(
        segment
        for segment in local_completion_segments
        if segment.family_index not in local_prefix_family_indices
        and segment.family_index not in remote_prefix_tail_families
    )
    ready_completion_segments, remote_completion_segments = (
        _split_ready_and_remote_completion_segments(
            plain_local_completion_segments,
            local_prefix_segments=(),
            chain_prefix_buckets=(),
        )
    )
    ready_completion_buckets = _batch_segments_by_padded_work(
        ready_completion_segments,
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    remote_completion_buckets = _batch_segments_by_padded_work(
        remote_completion_segments,
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    prefix_family_order = tuple(
        segment.family_index for bucket in local_prefix_buckets for segment in bucket
    )
    ready_completion_bucket_plans = _build_position_bucket_plans(
        ready_completion_buckets,
        local_token_ranges,
        sequence_length=spec.sequence_length,
        device=device,
    )
    remote_completion_bucket_plans = _build_position_bucket_plans(
        remote_completion_buckets,
        local_token_ranges,
        sequence_length=spec.sequence_length,
        device=device,
    )
    (
        prefix_boundary_buckets,
        prefix_tail_buckets,
        completion_with_prefix_tail_buckets,
    ) = _build_chunk_aligned_position_bucket_plans(
        tuple(local_prefix_segments),
        chunk_local_completion_segments,
        local_token_ranges,
        sequence_length=spec.sequence_length,
        device=device,
        planner_config=planner_config,
    )
    return GdnRankExecutionPlan.model_construct(
        cp_rank=cp_rank,
        cp_size=cp_size,
        batch_size=1,
        sequence_length=local_token_count,
        packed_batch_size=spec.batch_size,
        packed_sequence_length=spec.sequence_length,
        real_token_mask=torch.ones(
            1, local_token_count, device=device, dtype=torch.bool
        ),
        family_count=spec.family_count,
        completion_count=spec.completion_count,
        local_prefix_buckets=_build_position_bucket_plans(
            local_prefix_buckets,
            local_token_ranges,
            sequence_length=spec.sequence_length,
            device=device,
        ),
        local_completion_buckets=(
            ready_completion_bucket_plans + remote_completion_bucket_plans
        ),
        ready_local_completion_buckets=ready_completion_bucket_plans,
        remote_local_completion_buckets=remote_completion_bucket_plans,
        chain_prefix_buckets=(),
        chain_completion_buckets=(),
        prefix_table_is_dense_ordered=(
            not local_prefix_segments
            and prefix_family_order == tuple(range(spec.family_count))
        ),
        attention_to_gdn=attention_to_gdn,
        gdn_to_attention=gdn_to_attention,
        attention_token_ranges=source_layout.ownership_ranges_by_rank[cp_rank],
        gdn_token_ranges=local_token_ranges,
        attention_token_count=source_layout.token_counts_by_rank[cp_rank],
        gdn_token_count=local_token_count,
        parent_state_exchange_family_indices=tuple(
            sorted(parent_state_exchange_families - remote_prefix_tail_families)
        ),
        parent_state_transfers=_filter_parent_state_transfers(
            _build_parent_state_transfer_plans(parent_state_transfer_families),
            excluded_families=remote_prefix_tail_families,
            device=device,
        ),
        prefix_boundary_buckets=prefix_boundary_buckets,
        prefix_tail_buckets=prefix_tail_buckets,
        completion_with_prefix_tail_buckets=completion_with_prefix_tail_buckets,
        remote_prefix_tail_buckets=remote_prefix_tail_buckets,
        remote_completion_with_prefix_tail_buckets=remote_completion_with_prefix_tail_buckets,
        remote_prefix_tail_exchange=remote_prefix_tail_exchange,
        remote_prefix_tail_backward_exchange=remote_prefix_tail_backward_exchange,
        remote_prefix_tail_state_transfers=remote_prefix_tail_state_transfers,
    )


def _assign_local_attention_segments(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
    co_locate_local_families: bool,
    planner_config: GdnPlannerConfig,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    int,
    float,
]:
    rank_loads = [0] * cp_size
    has_prefix = [False] * cp_size
    has_completion = [False] * cp_size
    prefix_owner_by_family: list[int] = []
    completion_owners_by_family: list[tuple[int, ...]] = []
    parent_state_exchange_families: set[int] = set()
    cross_rank_token_count = 0

    def append_owner(rank: int, segment: GdnSegmentSpec) -> None:
        nonlocal cross_rank_token_count
        rank_loads[rank] += segment.length
        cross_rank_token_count += (
            segment.length - segment_attention_counts[_segment_key(segment)][rank]
        )

    for family in spec.families:
        if co_locate_local_families:
            owner = _best_segment_owner(
                (family.prefix, *family.completions),
                rank_loads,
                segment_attention_counts=segment_attention_counts,
                planner_config=planner_config,
            )
            prefix_owner_by_family.append(owner)
            completion_owners = tuple(owner for _ in family.completions)
            completion_owners_by_family.append(completion_owners)
            has_prefix[owner] = True
            for segment in (family.prefix, *family.completions):
                append_owner(owner, segment)
            if family.completions:
                has_completion[owner] = True
            continue

        prefix_owner = _best_segment_owner(
            (family.prefix,),
            rank_loads,
            segment_attention_counts=segment_attention_counts,
            planner_config=planner_config,
        )
        prefix_owner_by_family.append(prefix_owner)
        has_prefix[prefix_owner] = True
        append_owner(prefix_owner, family.prefix)
        completion_owners = []
        for completion in family.completions:
            owner = _best_segment_owner(
                (completion,),
                rank_loads,
                segment_attention_counts=segment_attention_counts,
                planner_config=planner_config,
            )
            completion_owners.append(owner)
            has_completion[owner] = True
            append_owner(owner, completion)
            if owner != prefix_owner:
                parent_state_exchange_families.add(family.family_index)
        completion_owners_by_family.append(tuple(completion_owners))

    del has_prefix, has_completion
    score = _score_local_segment_assignment(
        spec,
        cp_size=cp_size,
        prefix_owner_by_family=tuple(prefix_owner_by_family),
        completion_owners_by_family=tuple(completion_owners_by_family),
        rank_loads=tuple(rank_loads),
        cross_rank_token_count=cross_rank_token_count,
        parent_state_exchange_family_count=len(parent_state_exchange_families),
        planner_config=planner_config,
    )
    return (
        tuple(prefix_owner_by_family),
        tuple(completion_owners_by_family),
        tuple(sorted(parent_state_exchange_families)),
        cross_rank_token_count,
        score,
    )


def _score_local_segment_assignment(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    prefix_owner_by_family: tuple[int, ...],
    completion_owners_by_family: tuple[tuple[int, ...], ...],
    rank_loads: tuple[int, ...],
    cross_rank_token_count: int,
    parent_state_exchange_family_count: int,
    planner_config: GdnPlannerConfig,
) -> float:
    local_prefix_segments_by_rank: list[list[GdnSegmentSpec]] = [
        [] for _ in range(cp_size)
    ]
    local_completion_segments_by_rank: list[list[GdnSegmentSpec]] = [
        [] for _ in range(cp_size)
    ]
    for family in spec.families:
        prefix_owner = prefix_owner_by_family[family.family_index]
        local_prefix_segments_by_rank[prefix_owner].append(family.prefix)
        completion_owners = completion_owners_by_family[family.family_index]
        for completion, completion_owner in zip(
            family.completions, completion_owners, strict=True
        ):
            local_completion_segments_by_rank[completion_owner].append(completion)
    (
        local_work_by_rank,
        local_bucket_count,
        local_segment_count,
    ) = _estimate_local_rank_kernel_work(
        tuple(tuple(segments) for segments in local_prefix_segments_by_rank),
        tuple(tuple(segments) for segments in local_completion_segments_by_rank),
        planner_config=planner_config,
    )
    return _score_cp_segment_stats(
        rank_local_work=local_work_by_rank,
        rank_chain_work=tuple(0 for _ in range(cp_size)),
        rank_real_tokens=rank_loads,
        cross_rank_token_count=cross_rank_token_count,
        parent_state_exchange_family_count=parent_state_exchange_family_count,
        local_bucket_count=local_bucket_count,
        local_segment_count=local_segment_count,
        chain_bucket_count=0,
        planner_config=planner_config,
    )


def _can_zero_exchange_colocate_families(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
) -> bool:
    for family in spec.families:
        family_rank_counts = [0] * cp_size
        for segment in (family.prefix, *family.completions):
            segment_counts = segment_attention_counts[_segment_key(segment)]
            for rank in range(cp_size):
                family_rank_counts[rank] += segment_counts[rank]
        if max(family_rank_counts, default=0) != family.token_count:
            return False
    return True


def parse_gdn_shared_prefix_segments(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    *,
    min_completions_per_family: int = 0,
) -> GdnPackedExecutionSpec:
    """Parse ART packed shared-prefix metadata into a GDN segment DAG.

    The parser is intentionally strict: GDN state routing depends on prompt-family
    boundaries, so malformed metadata should fail before execution can silently
    leak recurrent or conv state across siblings or independent families.
    """

    groups = _rank2_long_cpu("group_ids", group_ids)
    parents = _rank2_long_cpu("parent_ids", parent_ids)
    if tuple(groups.shape) != tuple(parents.shape):
        raise ValueError(
            "group_ids and parent_ids must have the same shape, got "
            f"{tuple(groups.shape)} and {tuple(parents.shape)}"
        )

    batch_size, sequence_length = (int(groups.shape[0]), int(groups.shape[1]))
    valid_lengths: list[int] = []
    families: list[GdnPackedFamilySpec] = []
    for row_index in range(batch_size):
        row_group_ids = groups[row_index]
        row_parent_ids = parents[row_index]
        valid_length = _validate_padding_tensor(
            row_index, row_group_ids, row_parent_ids
        )
        valid_lengths.append(valid_length)
        if valid_length == 0:
            continue
        families.extend(
            _parse_row_tensor(
                row_index=row_index,
                group_ids=row_group_ids,
                parent_ids=row_parent_ids,
                valid_length=valid_length,
                first_family_index=len(families),
                min_completions_per_family=min_completions_per_family,
            )
        )

    return GdnPackedExecutionSpec(
        batch_size=batch_size,
        sequence_length=sequence_length,
        valid_lengths=tuple(valid_lengths),
        families=tuple(families),
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


def _build_chunk_aligned_cp1_bucket_plans(
    spec: GdnPackedExecutionSpec,
    *,
    device: torch.device | str,
    planner_config: GdnPlannerConfig,
) -> tuple[
    tuple[GdnSegmentBucketPlan, ...],
    tuple[GdnSegmentBucketPlan, ...],
    tuple[GdnSegmentBucketPlan, ...],
]:
    boundary_segments: list[GdnSegmentSpec] = []
    tail_segments: list[GdnSegmentSpec] = []
    completion_columns: list[_ExplicitBucketColumn] = []
    for family in spec.families:
        prefix = family.prefix
        boundary_end = _prefix_chunk_boundary_end(prefix)
        if boundary_end > prefix.start:
            boundary_segments.append(
                _segment_with_bounds(prefix, prefix.start, boundary_end)
            )
        prefix_tail_positions = tuple(range(boundary_end, prefix.end))
        if prefix_tail_positions and not family.completions:
            tail_segments.append(_segment_with_bounds(prefix, boundary_end, prefix.end))
        for child_offset, completion in enumerate(family.completions):
            completion_positions = prefix_tail_positions + tuple(
                range(completion.start, completion.end)
            )
            completion_columns.append(
                _explicit_bucket_column(
                    row_index=completion.row_index,
                    family_index=completion.family_index,
                    positions=completion_positions,
                    output_mask=(
                        ((child_offset == 0),) * len(prefix_tail_positions)
                        + (True,) * completion.length
                    ),
                )
            )
    boundary_buckets = _batch_segments_by_padded_work(
        tuple(boundary_segments),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    tail_buckets = _batch_segments_by_padded_work(
        tuple(tail_segments),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    completion_column_batches = _batch_explicit_bucket_columns(
        tuple(completion_columns),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    return (
        _build_segment_bucket_plans(boundary_buckets, device=device),
        _build_segment_bucket_plans(tail_buckets, device=device),
        _build_explicit_bucket_plans(completion_column_batches, device=device),
    )


def _build_chunk_aligned_position_bucket_plans(
    prefix_segments: tuple[GdnSegmentSpec, ...],
    completion_segments: tuple[GdnSegmentSpec, ...],
    local_token_ranges: tuple[tuple[int, int, int], ...],
    *,
    sequence_length: int,
    device: torch.device | str,
    planner_config: GdnPlannerConfig,
) -> tuple[
    tuple[GdnSegmentBucketPlan, ...],
    tuple[GdnSegmentBucketPlan, ...],
    tuple[GdnSegmentBucketPlan, ...],
]:
    local_range_ends = tuple(token_end for _, token_end, _ in local_token_ranges)
    local_range_positions = {
        (token_start, token_end): position_start
        for token_start, token_end, position_start in local_token_ranges
    }
    completions_by_family: dict[int, list[GdnSegmentSpec]] = {}
    for completion in completion_segments:
        completions_by_family.setdefault(completion.family_index, []).append(completion)
    boundary_segments: list[GdnSegmentSpec] = []
    tail_segments: list[GdnSegmentSpec] = []
    completion_columns: list[_ExplicitBucketColumn] = []
    for prefix in prefix_segments:
        boundary_end = _prefix_chunk_boundary_end(prefix)
        if boundary_end > prefix.start:
            boundary_segments.append(
                _segment_with_bounds(prefix, prefix.start, boundary_end)
            )
        family_completions = tuple(completions_by_family.get(prefix.family_index, ()))
        prefix_tail_positions = _local_positions_for_span(
            prefix.row_index,
            boundary_end,
            prefix.end,
            sequence_length=sequence_length,
            local_token_ranges=local_token_ranges,
            local_range_ends=local_range_ends,
            local_range_positions=local_range_positions,
        )
        if prefix_tail_positions and not family_completions:
            tail_segments.append(_segment_with_bounds(prefix, boundary_end, prefix.end))
        for child_offset, completion in enumerate(family_completions):
            completion_positions = _local_positions_for_span(
                completion.row_index,
                completion.start,
                completion.end,
                sequence_length=sequence_length,
                local_token_ranges=local_token_ranges,
                local_range_ends=local_range_ends,
                local_range_positions=local_range_positions,
            )
            positions = prefix_tail_positions + completion_positions
            completion_columns.append(
                _explicit_bucket_column(
                    row_index=0,
                    family_index=completion.family_index,
                    positions=positions,
                    output_mask=(
                        ((child_offset == 0),) * len(prefix_tail_positions)
                        + (True,) * len(completion_positions)
                    ),
                )
            )
    boundary_buckets = _batch_segments_by_padded_work(
        tuple(boundary_segments),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    tail_buckets = _batch_segments_by_padded_work(
        tuple(tail_segments),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    completion_column_batches = _batch_explicit_bucket_columns(
        tuple(completion_columns),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    return (
        _build_position_bucket_plans(
            boundary_buckets,
            local_token_ranges,
            sequence_length=sequence_length,
            device=device,
        ),
        _build_position_bucket_plans(
            tail_buckets,
            local_token_ranges,
            sequence_length=sequence_length,
            device=device,
        ),
        _build_explicit_bucket_plans(completion_column_batches, device=device),
    )


def _build_remote_prefix_tail_plans(
    spec: GdnPackedExecutionSpec,
    schedule: GdnCpSegmentSchedule,
    *,
    cp_rank: int,
    device: torch.device | str,
    planner_config: GdnPlannerConfig,
) -> tuple[
    tuple[GdnSegmentBucketPlan, ...],
    tuple[GdnSegmentBucketPlan, ...],
    Any | None,
    Any | None,
    tuple[GdnParentStateTransferPlan, ...],
    frozenset[int],
]:
    from art.megatron.gdn.layout import (
        GdnCpExchangePlan,
        GdnCpPeerTransfer,
        _reverse_exchange_plan,
    )

    family_by_index = {family.family_index: family for family in spec.families}
    prefix_owner_by_family = _prefix_owner_by_family(schedule)
    source_positions_by_pair: dict[tuple[int, int], list[int]] = {}
    dest_positions_by_pair: dict[tuple[int, int], list[int]] = {}
    dest_counts = [0 for _ in schedule.gdn_token_counts_by_rank]
    state_transfer_families: dict[tuple[int, int], set[int]] = {}
    remote_tail_family_indices: set[int] = set()
    local_tail_columns: list[_ExplicitBucketColumn] = []
    local_completion_columns: list[_ExplicitBucketColumn] = []
    tail_positions_by_dest_family: dict[tuple[int, int], tuple[int, ...]] = {}
    local_tail_column_families: set[int] = set()
    rank_ranges = schedule.gdn_token_ranges_by_rank
    rank_range_ends = tuple(
        tuple(end for _, end, _ in ranges) for ranges in rank_ranges
    )
    rank_range_positions = tuple(
        {
            (token_start, token_end): position_start
            for token_start, token_end, position_start in ranges
        }
        for ranges in rank_ranges
    )

    for dest_rank, completions in enumerate(schedule.local_completion_segments_by_rank):
        for completion in completions:
            source_rank = prefix_owner_by_family.get(completion.family_index)
            if source_rank is None or source_rank == dest_rank:
                continue
            family = family_by_index[completion.family_index]
            boundary_end = _prefix_chunk_boundary_end(family.prefix)
            if boundary_end == family.prefix.end:
                continue
            dest_family = (dest_rank, family.family_index)
            dest_positions = tail_positions_by_dest_family.get(dest_family)
            if dest_positions is None:
                source_positions = _local_positions_for_span(
                    family.prefix.row_index,
                    boundary_end,
                    family.prefix.end,
                    sequence_length=spec.sequence_length,
                    local_token_ranges=rank_ranges[source_rank],
                    local_range_ends=rank_range_ends[source_rank],
                    local_range_positions=rank_range_positions[source_rank],
                )
                if len(source_positions) != family.prefix.end - boundary_end:
                    raise ValueError(
                        "remote prefix-tail exchange could not locate all source tokens "
                        f"for family {family.family_index}"
                    )
                dest_start = dest_counts[dest_rank]
                dest_positions = tuple(
                    range(dest_start, dest_start + len(source_positions))
                )
                tail_positions_by_dest_family[dest_family] = dest_positions
                dest_counts[dest_rank] += len(source_positions)
                pair = (source_rank, dest_rank)
                source_positions_by_pair.setdefault(pair, []).extend(source_positions)
                dest_positions_by_pair.setdefault(pair, []).extend(dest_positions)
                state_transfer_families.setdefault(pair, set()).add(family.family_index)
                remote_tail_family_indices.add(family.family_index)

            if dest_rank != cp_rank:
                continue
            completion_positions = _local_positions_for_span(
                completion.row_index,
                completion.start,
                completion.end,
                sequence_length=spec.sequence_length,
                local_token_ranges=rank_ranges[dest_rank],
                local_range_ends=rank_range_ends[dest_rank],
                local_range_positions=rank_range_positions[dest_rank],
            )
            if len(completion_positions) != completion.length:
                raise ValueError(
                    "remote prefix-tail bucket could not locate all completion tokens "
                    f"for family {family.family_index}"
                )
            remote_base = int(schedule.gdn_token_counts_by_rank[dest_rank])
            if (
                len(dest_positions) > 0
                and family.family_index not in local_tail_column_families
            ):
                local_tail_column_families.add(family.family_index)
                local_tail_columns.append(
                    _explicit_bucket_column(
                        row_index=0,
                        family_index=family.family_index,
                        positions=tuple(remote_base + pos for pos in dest_positions),
                        output_mask=(False,) * len(dest_positions),
                    )
                )
            local_completion_columns.append(
                _explicit_bucket_column(
                    row_index=0,
                    family_index=family.family_index,
                    positions=completion_positions,
                    output_mask=(True,) * len(completion_positions),
                )
            )

    if not source_positions_by_pair:
        return (), (), None, None, (), frozenset()

    transfers = tuple(
        GdnCpPeerTransfer.model_construct(
            source_rank=source_rank,
            dest_rank=dest_rank,
            token_count=len(source_positions),
            source_positions_tensor=_move_planner_tensor(
                torch.tensor(source_positions, dtype=torch.long), device
            ),
            dest_positions_tensor=_move_planner_tensor(
                torch.tensor(
                    dest_positions_by_pair[(source_rank, dest_rank)],
                    dtype=torch.long,
                ),
                device,
            ),
        )
        for (source_rank, dest_rank), source_positions in sorted(
            source_positions_by_pair.items()
        )
    )
    exchange = GdnCpExchangePlan.model_construct(
        cp_size=len(schedule.gdn_token_counts_by_rank),
        source_token_counts_by_rank=schedule.gdn_token_counts_by_rank,
        dest_token_counts_by_rank=tuple(dest_counts),
        transfers=transfers,
        cross_rank_token_count_override=sum(dest_counts),
    )
    tail_column_batches = _batch_explicit_bucket_columns(
        tuple(local_tail_columns),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    completion_column_batches = _batch_explicit_bucket_columns(
        tuple(local_completion_columns),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    return (
        _build_explicit_bucket_plans(tail_column_batches, device=device),
        _build_explicit_bucket_plans(completion_column_batches, device=device),
        exchange,
        _reverse_exchange_plan(exchange),
        _transfer_plans_to_device(
            _build_parent_state_transfer_plans(state_transfer_families),
            device=device,
        ),
        frozenset(remote_tail_family_indices),
    )


def _empty_remote_prefix_tail_plans() -> tuple[
    tuple[GdnSegmentBucketPlan, ...],
    tuple[GdnSegmentBucketPlan, ...],
    Any | None,
    Any | None,
    tuple[GdnParentStateTransferPlan, ...],
    frozenset[int],
]:
    return (), (), None, None, (), frozenset()


def _prefix_owner_by_family(schedule: GdnCpSegmentSchedule) -> dict[int, int]:
    owners: dict[int, int] = {}
    for rank, segments in enumerate(schedule.local_prefix_segments_by_rank):
        for segment in segments:
            owners[segment.family_index] = rank
    return owners


def _filter_parent_state_transfers(
    transfers: tuple[GdnParentStateTransferPlan, ...],
    *,
    excluded_families: frozenset[int],
    device: torch.device | str,
) -> tuple[GdnParentStateTransferPlan, ...]:
    if not excluded_families:
        return _transfer_plans_to_device(transfers, device=device)
    kept: dict[tuple[int, int], set[int]] = {}
    for transfer in transfers:
        families = set(transfer.family_indices) - excluded_families
        if families:
            kept.setdefault((transfer.source_rank, transfer.dest_rank), set()).update(
                families
            )
    return _transfer_plans_to_device(
        _build_parent_state_transfer_plans(kept), device=device
    )


def _local_positions_for_span(
    row_index: int,
    start: int,
    end: int,
    *,
    sequence_length: int,
    local_token_ranges: tuple[tuple[int, int, int], ...],
    local_range_ends: tuple[int, ...],
    local_range_positions: dict[tuple[int, int], int] | None = None,
) -> tuple[int, ...]:
    if start == end:
        return ()
    token_start = row_index * sequence_length + start
    token_end = row_index * sequence_length + end
    if local_range_positions is not None:
        position_start = local_range_positions.get((token_start, token_end))
        if position_start is not None:
            return tuple(range(position_start, position_start + end - start))
    range_index = bisect_left(local_range_ends, token_start + 1)
    if range_index < len(local_token_ranges):
        range_start, range_end, position_start = local_token_ranges[range_index]
        if range_start <= token_start and token_end <= range_end:
            local_start = position_start + token_start - range_start
            return tuple(range(local_start, local_start + end - start))
    segment = _trusted_pydantic_construct(
        GdnSegmentSpec,
        _GDN_SEGMENT_SPEC_FIELDS,
        row_index=row_index,
        family_index=0,
        group_id=0,
        parent_id=0,
        start=start,
        end=end,
        kind="prefix",
        child_index=None,
    )
    return tuple(
        int(position)
        for position in _local_positions_for_segment(
            segment,
            sequence_length=sequence_length,
            local_token_ranges=local_token_ranges,
            local_range_ends=local_range_ends,
        ).tolist()
    )


def _prefix_chunk_boundary_end(prefix: GdnSegmentSpec) -> int:
    aligned_length = (prefix.length // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
    return prefix.start + aligned_length


def _segment_with_bounds(
    segment: GdnSegmentSpec, start: int, end: int
) -> GdnSegmentSpec:
    return _trusted_pydantic_construct(
        GdnSegmentSpec,
        _GDN_SEGMENT_SPEC_FIELDS,
        row_index=segment.row_index,
        family_index=segment.family_index,
        group_id=segment.group_id,
        parent_id=segment.parent_id,
        start=start,
        end=end,
        kind=segment.kind,
        child_index=segment.child_index,
    )


def _batch_explicit_bucket_columns(
    columns: tuple[_ExplicitBucketColumn, ...],
    *,
    max_padding_ratio: float = 1.25,
    max_segments_per_batch: int = 128,
) -> tuple[tuple[_ExplicitBucketColumn, ...], ...]:
    if not columns:
        return ()
    ordered = sorted(
        columns,
        key=lambda column: (column.length, column.family_index, column.row_index),
    )
    batches: list[list[_ExplicitBucketColumn]] = []
    current: list[_ExplicitBucketColumn] = []
    current_tokens = 0
    current_max = 0
    for column in ordered:
        next_count = len(current) + 1
        next_tokens = current_tokens + column.length
        next_max = max(current_max, column.length)
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
        current.append(column)
        current_tokens += column.length
        current_max = max(current_max, column.length)
    if current:
        batches.append(current)
    return tuple(tuple(batch) for batch in batches)


def _build_explicit_bucket_plans(
    bucket_columns: tuple[tuple[_ExplicitBucketColumn, ...], ...],
    *,
    device: torch.device | str,
) -> tuple[GdnSegmentBucketPlan, ...]:
    return tuple(
        _build_explicit_bucket_plan(columns, device=device)
        for columns in bucket_columns
    )


def _build_explicit_bucket_plan(
    columns: tuple[_ExplicitBucketColumn, ...],
    *,
    device: torch.device | str,
) -> GdnSegmentBucketPlan:
    max_length = max(column.length for column in columns)
    column_count = len(columns)
    lengths = [column.length for column in columns]
    lengths_cpu = torch.tensor(lengths, dtype=torch.long)
    offsets_cpu = torch.arange(max_length, dtype=torch.long).unsqueeze(1)
    real_mask_cpu = offsets_cpu < lengths_cpu.unsqueeze(0)
    padded_element_count = max_length * column_count
    row_indices = [0] * padded_element_count
    position_indices = [0] * padded_element_count
    output_mask = [False] * padded_element_count
    for column_index, column in enumerate(columns):
        length = column.length
        column_slice = slice(column_index, length * column_count, column_count)
        row_indices[column_slice] = [column.row_index] * length
        position_indices[column_slice] = column.positions
        output_mask[column_slice] = column.output_mask
    row_indices_cpu = torch.tensor(row_indices, dtype=torch.long).reshape(
        max_length, column_count
    )
    position_indices_cpu = torch.tensor(position_indices, dtype=torch.long).reshape(
        max_length, column_count
    )
    output_mask_cpu = torch.tensor(output_mask, dtype=torch.bool).reshape(
        max_length, column_count
    )
    family_indices_cpu = torch.tensor(
        [column.family_index for column in columns], dtype=torch.long
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
        row_indices=_move_planner_tensor(row_indices_cpu, device),
        position_indices=_move_planner_tensor(position_indices_cpu, device),
        family_indices=_move_planner_tensor(family_indices_cpu, device),
        real_token_count_static=int(lengths_cpu.sum().item()),
        output_mask=_move_planner_tensor(output_mask_cpu, device),
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


def _build_cp_rank_execution_plan(
    spec: GdnPackedExecutionSpec,
    *,
    device: torch.device | str,
    cp_rank: int,
    cp_size: int,
    attention_token_layout_index: TokenLayoutIndex | None,
    cp_segment_schedule: GdnCpSegmentSchedule | None,
    planner_config: GdnPlannerConfig,
) -> GdnRankExecutionPlan:
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
    if (
        attention_token_layout_index is not None
        and _layout_cp_size(attention_token_layout_index) != cp_size
    ):
        raise ValueError(
            "attention token layout index cp_size must match GDN cp_size, got "
            f"{_layout_cp_size(attention_token_layout_index)} and {cp_size}"
        )

    from art.megatron.gdn.layout import (
        _reverse_exchange_plan,
        build_local_rank_cp_exchange_plan_from_dest_ranges,
    )

    has_explicit_attention_layout = attention_token_layout_index is not None
    if cp_segment_schedule is None and not has_explicit_attention_layout:
        local_family_plan = _build_local_family_rank_execution_plan(
            spec,
            device=device,
            cp_rank=cp_rank,
            cp_size=cp_size,
            planner_config=planner_config,
        )
        if local_family_plan is not None:
            return local_family_plan
    if cp_segment_schedule is None and has_explicit_attention_layout:
        local_layout_plan = _build_local_attention_layout_rank_execution_plan(
            spec,
            device=device,
            cp_rank=cp_rank,
            cp_size=cp_size,
            attention_token_layout_index=attention_token_layout_index,
            planner_config=planner_config,
        )
        if local_layout_plan is not None:
            return local_layout_plan

    source_layout = _attention_source_layout(
        spec,
        cp_size=cp_size,
        attention_token_layout_index=attention_token_layout_index,
        planner_config=planner_config,
    )
    if cp_segment_schedule is None:
        schedule = _build_cp_segment_schedule(
            spec,
            cp_size=cp_size,
            attention_layout_index=_build_attention_layout_index_from_token_layout(
                source_layout,
                max_ranges=max(
                    1,
                    (2 * spec.real_token_count) // max(1, len(spec.segments())),
                ),
            ),
            planner_config=planner_config,
        )
    else:
        schedule = cp_segment_schedule
    if len(schedule.gdn_token_counts_by_rank) != cp_size:
        raise ValueError(f"CP GDN schedule must contain {cp_size} ranks")
    attention_to_gdn = build_local_rank_cp_exchange_plan_from_dest_ranges(
        source_layout=source_layout,
        device=device,
        local_rank=cp_rank,
        dest_ranges_by_rank=schedule.gdn_token_ranges_by_rank,
        cross_rank_token_count=schedule.cross_rank_token_count,
    )
    gdn_to_attention = _reverse_exchange_plan(attention_to_gdn)
    local_token_ranges = schedule.gdn_token_ranges_by_rank[cp_rank]
    local_gdn_token_count = schedule.gdn_token_counts_by_rank[cp_rank]
    if schedule.parent_state_exchange_family_indices:
        (
            remote_prefix_tail_buckets,
            remote_completion_with_prefix_tail_buckets,
            remote_prefix_tail_exchange,
            remote_prefix_tail_backward_exchange,
            remote_prefix_tail_state_transfers,
            remote_prefix_tail_families,
        ) = _build_remote_prefix_tail_plans(
            spec,
            schedule,
            cp_rank=cp_rank,
            device=device,
            planner_config=planner_config,
        )
    else:
        (
            remote_prefix_tail_buckets,
            remote_completion_with_prefix_tail_buckets,
            remote_prefix_tail_exchange,
            remote_prefix_tail_backward_exchange,
            remote_prefix_tail_state_transfers,
            remote_prefix_tail_families,
        ) = _empty_remote_prefix_tail_plans()

    chain_prefix_buckets = tuple(
        bucket for bucket in schedule.chain_prefix_buckets if bucket
    )
    chain_completion_buckets = tuple(
        bucket for bucket in schedule.chain_completion_buckets if bucket
    )
    local_prefix_segments = tuple(schedule.local_prefix_segments_by_rank[cp_rank])
    local_prefix_family_indices = {
        segment.family_index for segment in local_prefix_segments
    }
    local_prefix_buckets = _batch_segments_by_padded_work(
        () if local_prefix_segments else (),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    local_completion_segments = tuple(
        schedule.local_completion_segments_by_rank[cp_rank]
    )
    chunk_local_completion_segments = tuple(
        segment
        for segment in local_completion_segments
        if segment.family_index in local_prefix_family_indices
    )
    plain_local_completion_segments = tuple(
        segment
        for segment in local_completion_segments
        if segment.family_index not in local_prefix_family_indices
        and segment.family_index not in remote_prefix_tail_families
    )
    ready_completion_segments, remote_completion_segments = (
        _split_ready_and_remote_completion_segments(
            plain_local_completion_segments,
            local_prefix_segments=(),
            chain_prefix_buckets=chain_prefix_buckets,
        )
    )
    ready_local_completion_buckets = _batch_segments_by_padded_work(
        ready_completion_segments,
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    remote_local_completion_buckets = _batch_segments_by_padded_work(
        remote_completion_segments,
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    local_completion_buckets = (
        ready_local_completion_buckets + remote_local_completion_buckets
    )
    prefix_family_order = tuple(
        segment.family_index
        for bucket in (
            *chain_prefix_buckets,
            *local_prefix_buckets,
        )
        for segment in bucket
    )
    (
        prefix_boundary_buckets,
        prefix_tail_buckets,
        completion_with_prefix_tail_buckets,
    ) = _build_chunk_aligned_position_bucket_plans(
        local_prefix_segments,
        chunk_local_completion_segments,
        local_token_ranges,
        sequence_length=spec.sequence_length,
        device=device,
        planner_config=planner_config,
    )
    return GdnRankExecutionPlan.model_construct(
        cp_rank=cp_rank,
        cp_size=cp_size,
        batch_size=1,
        sequence_length=local_gdn_token_count,
        packed_batch_size=spec.batch_size,
        packed_sequence_length=spec.sequence_length,
        real_token_mask=torch.ones(
            1, local_gdn_token_count, device=device, dtype=torch.bool
        ),
        family_count=spec.family_count,
        completion_count=spec.completion_count,
        local_prefix_buckets=_build_position_bucket_plans(
            local_prefix_buckets,
            local_token_ranges,
            sequence_length=spec.sequence_length,
            device=device,
        ),
        local_completion_buckets=_build_position_bucket_plans(
            local_completion_buckets,
            local_token_ranges,
            sequence_length=spec.sequence_length,
            device=device,
        ),
        ready_local_completion_buckets=_build_position_bucket_plans(
            ready_local_completion_buckets,
            local_token_ranges,
            sequence_length=spec.sequence_length,
            device=device,
        ),
        remote_local_completion_buckets=_build_position_bucket_plans(
            remote_local_completion_buckets,
            local_token_ranges,
            sequence_length=spec.sequence_length,
            device=device,
        ),
        chain_prefix_buckets=_build_position_bucket_plans(
            chain_prefix_buckets,
            local_token_ranges,
            sequence_length=spec.sequence_length,
            device=device,
            token_ranges_by_rank=schedule.gdn_token_ranges_by_rank,
        ),
        chain_completion_buckets=_build_position_bucket_plans(
            chain_completion_buckets,
            local_token_ranges,
            sequence_length=spec.sequence_length,
            device=device,
            token_ranges_by_rank=schedule.gdn_token_ranges_by_rank,
        ),
        prefix_table_is_dense_ordered=(
            not local_prefix_segments
            and prefix_family_order == tuple(range(spec.family_count))
        ),
        attention_to_gdn=attention_to_gdn,
        gdn_to_attention=gdn_to_attention,
        attention_token_ranges=source_layout.ownership_ranges_by_rank[cp_rank],
        gdn_token_ranges=local_token_ranges,
        attention_token_count=source_layout.token_counts_by_rank[cp_rank],
        gdn_token_count=local_gdn_token_count,
        parent_state_exchange_family_indices=(
            tuple(
                family_index
                for family_index in schedule.parent_state_exchange_family_indices
                if family_index not in remote_prefix_tail_families
            )
        ),
        parent_state_transfers=_filter_parent_state_transfers(
            schedule.parent_state_transfers,
            excluded_families=remote_prefix_tail_families,
            device=device,
        ),
        prefix_boundary_buckets=prefix_boundary_buckets,
        prefix_tail_buckets=prefix_tail_buckets,
        completion_with_prefix_tail_buckets=completion_with_prefix_tail_buckets,
        remote_prefix_tail_buckets=remote_prefix_tail_buckets,
        remote_completion_with_prefix_tail_buckets=remote_completion_with_prefix_tail_buckets,
        remote_prefix_tail_exchange=remote_prefix_tail_exchange,
        remote_prefix_tail_backward_exchange=remote_prefix_tail_backward_exchange,
        remote_prefix_tail_state_transfers=remote_prefix_tail_state_transfers,
    )


def build_gdn_cp_segment_schedule(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    attention_token_layout_index: TokenLayoutIndex | None = None,
    planner_config: GdnPlannerConfig | None = None,
) -> GdnCpSegmentSchedule:
    planner_config = planner_config or GdnPlannerConfig()
    source_layout = _attention_source_layout(
        spec,
        cp_size=cp_size,
        attention_token_layout_index=attention_token_layout_index,
        planner_config=planner_config,
    )
    return _build_cp_segment_schedule(
        spec,
        cp_size=cp_size,
        attention_layout_index=_build_attention_layout_index_from_token_layout(
            source_layout,
            max_ranges=max(
                1, (2 * spec.real_token_count) // max(1, len(spec.segments()))
            ),
        ),
        planner_config=planner_config,
    )


def _build_cp_segment_schedule(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    attention_layout_index: _AttentionLayoutIndex,
    planner_config: GdnPlannerConfig,
) -> GdnCpSegmentSchedule:
    segment_attention_counts = _segment_attention_rank_counts(
        spec,
        cp_size=cp_size,
        attention_layout_index=attention_layout_index,
    )
    legal_chain_segments = tuple(
        segment
        for family in spec.families
        for segment in (family.prefix, *family.completions)
        if (
            _can_chain_prefix_segment(
                segment, cp_size=cp_size, planner_config=planner_config
            )
            if segment.kind == "prefix"
            else _can_chain_segment(
                segment, cp_size=cp_size, planner_config=planner_config
            )
        )
    )
    decision = _beam_search_cp_segment_schedule_decision(
        spec,
        cp_size=cp_size,
        attention_layout_index=attention_layout_index,
        segment_attention_counts=segment_attention_counts,
        legal_chain_segments=legal_chain_segments,
        planner_config=planner_config,
    )
    return _materialize_cp_segment_schedule(
        spec,
        cp_size=cp_size,
        attention_layout_index=attention_layout_index,
        segment_attention_counts=segment_attention_counts,
        chain_segment_keys=decision.chain_segment_keys,
        co_locate_local_families=decision.co_locate_local_families,
        planner_config=planner_config,
    )


def _beam_search_cp_segment_schedule_decision(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    attention_layout_index: _AttentionLayoutIndex,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
    legal_chain_segments: tuple[GdnSegmentSpec, ...],
    planner_config: GdnPlannerConfig,
) -> _GdnCpSegmentSearchDecision:
    legal_chain_keys = frozenset(
        _segment_key(segment) for segment in legal_chain_segments
    )
    chain_rank_counts_by_key: dict[GdnSegmentDecisionKey, tuple[int, ...]] = {}
    chain_cross_rank_tokens_by_key: dict[GdnSegmentDecisionKey, int] = {}
    for segment in legal_chain_segments:
        key = _segment_key(segment)
        (
            chain_rank_counts_by_key[key],
            chain_cross_rank_tokens_by_key[key],
        ) = _chain_segment_rank_counts_and_cross_rank_tokens(
            segment,
            spec,
            cp_size=cp_size,
            attention_layout_index=attention_layout_index,
        )

    score_cache: dict[
        frozenset[GdnSegmentDecisionKey], _GdnCpSegmentSearchDecision
    ] = {}

    def decision_for(
        chain_segment_keys: frozenset[GdnSegmentDecisionKey],
    ) -> _GdnCpSegmentSearchDecision:
        cached = score_cache.get(chain_segment_keys)
        if cached is not None:
            return cached
        non_colocated_score = _score_cp_segment_decisions(
            spec,
            cp_size=cp_size,
            segment_attention_counts=segment_attention_counts,
            chain_rank_counts_by_key=chain_rank_counts_by_key,
            chain_cross_rank_tokens_by_key=chain_cross_rank_tokens_by_key,
            chain_segment_keys=chain_segment_keys,
            co_locate_local_families=False,
            planner_config=planner_config,
        )
        colocated_score = _score_cp_segment_decisions(
            spec,
            cp_size=cp_size,
            segment_attention_counts=segment_attention_counts,
            chain_rank_counts_by_key=chain_rank_counts_by_key,
            chain_cross_rank_tokens_by_key=chain_cross_rank_tokens_by_key,
            chain_segment_keys=chain_segment_keys,
            co_locate_local_families=True,
            planner_config=planner_config,
        )
        co_locate = colocated_score < non_colocated_score
        decision = _GdnCpSegmentSearchDecision.model_construct(
            chain_segment_keys=chain_segment_keys,
            co_locate_local_families=co_locate,
            score=colocated_score if co_locate else non_colocated_score,
        )
        score_cache[chain_segment_keys] = decision
        return decision

    best = decision_for(frozenset())
    beam_by_keys = {best.chain_segment_keys: best}
    if legal_chain_keys:
        all_chain = decision_for(legal_chain_keys)
        beam_by_keys[all_chain.chain_segment_keys] = all_chain
        if best.score - all_chain.score > planner_config.cp_chain_min_score_delta_ms:
            best = all_chain
    candidate_groups = _bounded_chain_candidate_groups(
        spec,
        legal_chain_segments,
        segment_attention_counts=segment_attention_counts,
        chain_rank_counts_by_key=chain_rank_counts_by_key,
        planner_config=planner_config,
    )
    beam = _best_cp_segment_search_decisions(
        beam_by_keys.values(),
        limit=planner_config.cp_chain_beam_width,
    )
    stale_steps = 0
    for _ in range(planner_config.cp_chain_beam_max_steps):
        if not candidate_groups:
            break
        expanded: dict[
            frozenset[GdnSegmentDecisionKey], _GdnCpSegmentSearchDecision
        ] = {}
        for decision in beam:
            neighbors = []
            for segment_keys in _chain_beam_neighbor_groups(
                decision.chain_segment_keys,
                candidate_groups=candidate_groups,
                branch_factor=planner_config.cp_chain_beam_branch_factor,
            ):
                if segment_keys.issubset(decision.chain_segment_keys):
                    next_keys = decision.chain_segment_keys - segment_keys
                else:
                    next_keys = decision.chain_segment_keys | segment_keys
                neighbors.append(decision_for(frozenset(next_keys)))
            for neighbor in _best_cp_segment_search_decisions(
                neighbors,
                limit=planner_config.cp_chain_beam_branch_factor,
            ):
                expanded[neighbor.chain_segment_keys] = neighbor
        if not expanded:
            break
        beam = _best_cp_segment_search_decisions(
            (*beam, *expanded.values()),
            limit=planner_config.cp_chain_beam_width,
        )
        step_best = beam[0]
        if best.score - step_best.score > planner_config.cp_chain_min_score_delta_ms:
            best = step_best
            stale_steps = 0
        else:
            stale_steps += 1
            if stale_steps >= 2:
                break
    return best


def _chain_beam_neighbor_groups(
    chain_segment_keys: frozenset[GdnSegmentDecisionKey],
    *,
    candidate_groups: tuple[frozenset[GdnSegmentDecisionKey], ...],
    branch_factor: int,
) -> tuple[frozenset[GdnSegmentDecisionKey], ...]:
    selected: list[frozenset[GdnSegmentDecisionKey]] = []
    for group in candidate_groups:
        if group and not group.issubset(chain_segment_keys):
            selected.append(group)
            if len(selected) >= branch_factor:
                return tuple(selected)
    for group in reversed(candidate_groups):
        if group and group.intersection(chain_segment_keys) and group not in selected:
            selected.append(group)
            if len(selected) >= branch_factor:
                break
    return tuple(selected)


def _best_cp_segment_search_decisions(
    decisions: Any,
    *,
    limit: int,
) -> tuple[_GdnCpSegmentSearchDecision, ...]:
    return tuple(
        sorted(
            decisions,
            key=lambda decision: (
                decision.score,
                len(decision.chain_segment_keys),
                tuple(sorted(decision.chain_segment_keys)),
            ),
        )[:limit]
    )


def _bounded_chain_candidate_groups(
    spec: GdnPackedExecutionSpec,
    legal_chain_segments: tuple[GdnSegmentSpec, ...],
    *,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
    chain_rank_counts_by_key: dict[GdnSegmentDecisionKey, tuple[int, ...]],
    planner_config: GdnPlannerConfig,
) -> tuple[frozenset[GdnSegmentDecisionKey], ...]:
    legal_key_set = frozenset(_segment_key(segment) for segment in legal_chain_segments)
    if not legal_key_set:
        return ()
    prefix_keys = frozenset(
        _segment_key(family.prefix)
        for family in spec.families
        if _segment_key(family.prefix) in legal_key_set
    )
    completion_keys = legal_key_set - prefix_keys
    groups: list[frozenset[GdnSegmentDecisionKey]] = []
    for group in (legal_key_set, prefix_keys, completion_keys):
        if group and group not in groups:
            groups.append(group)
    for group in _ranked_chain_beam_groups(
        spec,
        legal_chain_segments,
        segment_attention_counts=segment_attention_counts,
        chain_rank_counts_by_key=chain_rank_counts_by_key,
        planner_config=planner_config,
    ):
        if group and group not in groups:
            groups.append(group)
    return tuple(groups[: planner_config.cp_chain_beam_candidate_limit])


def _ranked_chain_beam_groups(
    spec: GdnPackedExecutionSpec,
    legal_chain_segments: tuple[GdnSegmentSpec, ...],
    *,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
    chain_rank_counts_by_key: dict[GdnSegmentDecisionKey, tuple[int, ...]],
    planner_config: GdnPlannerConfig,
) -> tuple[frozenset[GdnSegmentDecisionKey], ...]:
    if not legal_chain_segments:
        return ()
    priority_by_key = {
        _segment_key(segment): _chain_beam_segment_priority(
            segment,
            segment_attention_counts=segment_attention_counts,
            chain_rank_counts_by_key=chain_rank_counts_by_key,
        )
        for segment in legal_chain_segments
    }
    legal_key_set = frozenset(priority_by_key)
    groups: set[frozenset[GdnSegmentDecisionKey]] = {
        frozenset((key,)) for key in legal_key_set
    }
    for family in spec.families:
        completion_keys = frozenset(
            _segment_key(completion)
            for completion in family.completions
            if _segment_key(completion) in legal_key_set
        )
        if len(completion_keys) > 1:
            groups.add(completion_keys)
        family_keys = completion_keys
        prefix_key = _segment_key(family.prefix)
        if prefix_key in legal_key_set:
            family_keys = family_keys | frozenset((prefix_key,))
        if len(family_keys) > 1:
            groups.add(family_keys)
    ranked = tuple(
        sorted(
            groups,
            key=lambda group: _chain_beam_group_priority(
                group, priority_by_key=priority_by_key
            ),
            reverse=True,
        )
    )
    limit = planner_config.cp_chain_beam_candidate_limit
    if len(ranked) <= limit:
        return ranked
    high_count = (limit + 1) // 2
    low_count = limit - high_count
    selected = [*ranked[:high_count]]
    for group in ranked[-low_count:]:
        if group not in selected:
            selected.append(group)
    return tuple(selected)


def _chain_beam_group_priority(
    group: frozenset[GdnSegmentDecisionKey],
    *,
    priority_by_key: dict[GdnSegmentDecisionKey, tuple[int, int, int, int]],
) -> tuple[int, int, int, int, int]:
    priorities = tuple(priority_by_key[key] for key in group)
    return (
        sum(priority[0] for priority in priorities),
        sum(priority[1] for priority in priorities),
        max((priority[2] for priority in priorities), default=0),
        sum(priority[3] for priority in priorities),
        len(group),
    )


def _chain_beam_segment_priority(
    segment: GdnSegmentSpec,
    *,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
    chain_rank_counts_by_key: dict[GdnSegmentDecisionKey, tuple[int, ...]],
) -> tuple[int, int, int, int]:
    key = _segment_key(segment)
    chain_max_load = max(chain_rank_counts_by_key[key], default=0)
    best_attention_locality = max(segment_attention_counts[key], default=0)
    chain_load_relief = segment.length - chain_max_load
    minimum_local_exchange = segment.length - best_attention_locality
    return (
        chain_load_relief,
        segment.length,
        best_attention_locality,
        -minimum_local_exchange,
    )


def _score_cp_segment_decisions(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
    chain_rank_counts_by_key: dict[GdnSegmentDecisionKey, tuple[int, ...]],
    chain_cross_rank_tokens_by_key: dict[GdnSegmentDecisionKey, int],
    chain_segment_keys: frozenset[GdnSegmentDecisionKey],
    co_locate_local_families: bool,
    planner_config: GdnPlannerConfig,
) -> float:
    rank_loads = [0] * cp_size
    local_prefix_segments_by_rank: list[list[GdnSegmentSpec]] = [
        [] for _ in range(cp_size)
    ]
    local_completion_segments_by_rank: list[list[GdnSegmentSpec]] = [
        [] for _ in range(cp_size)
    ]
    chain_prefix_segments: list[GdnSegmentSpec] = []
    chain_completion_segments: list[GdnSegmentSpec] = []
    parent_state_exchange_families: set[int] = set()
    cross_rank_token_count = 0

    for family in spec.families:
        prefix_key = _segment_key(family.prefix)
        chain_prefix = prefix_key in chain_segment_keys
        local_completions = tuple(
            completion
            for completion in family.completions
            if _segment_key(completion) not in chain_segment_keys
        )
        prefix_owner: int | None = None
        if chain_prefix:
            chain_prefix_segments.append(family.prefix)
            cross_rank_token_count += _add_chain_search_load(
                rank_loads,
                family.prefix,
                chain_rank_counts_by_key=chain_rank_counts_by_key,
                chain_cross_rank_tokens_by_key=chain_cross_rank_tokens_by_key,
            )
        else:
            owner_segments = (
                (family.prefix, *local_completions)
                if co_locate_local_families
                else (family.prefix,)
            )
            prefix_owner = _best_segment_owner(
                owner_segments,
                rank_loads,
                segment_attention_counts=segment_attention_counts,
                planner_config=planner_config,
            )
            local_prefix_segments_by_rank[prefix_owner].append(family.prefix)
            cross_rank_token_count += _add_local_search_load(
                rank_loads,
                prefix_owner,
                family.prefix,
                segment_attention_counts=segment_attention_counts,
            )
        for completion in family.completions:
            completion_key = _segment_key(completion)
            if completion_key in chain_segment_keys:
                chain_completion_segments.append(completion)
                cross_rank_token_count += _add_chain_search_load(
                    rank_loads,
                    completion,
                    chain_rank_counts_by_key=chain_rank_counts_by_key,
                    chain_cross_rank_tokens_by_key=chain_cross_rank_tokens_by_key,
                )
                if not chain_prefix:
                    parent_state_exchange_families.add(family.family_index)
                continue
            if co_locate_local_families and not chain_prefix:
                if prefix_owner is None:
                    raise RuntimeError(
                        "co-located local completion planning lost the prefix owner"
                    )
                owner = prefix_owner
            else:
                owner = _best_segment_owner(
                    (completion,),
                    rank_loads,
                    segment_attention_counts=segment_attention_counts,
                    planner_config=planner_config,
                )
            if not chain_prefix:
                if prefix_owner is None:
                    raise RuntimeError(
                        "local completion planning lost the prefix owner"
                    )
                if owner != prefix_owner:
                    parent_state_exchange_families.add(family.family_index)
            local_completion_segments_by_rank[owner].append(completion)
            cross_rank_token_count += _add_local_search_load(
                rank_loads,
                owner,
                completion,
                segment_attention_counts=segment_attention_counts,
            )
    (
        local_work_by_rank,
        local_bucket_count,
        local_segment_count,
    ) = _estimate_local_rank_kernel_work(
        tuple(tuple(segments) for segments in local_prefix_segments_by_rank),
        tuple(tuple(segments) for segments in local_completion_segments_by_rank),
        planner_config=planner_config,
    )
    chain_work_by_rank, chain_bucket_count = _estimate_chain_rank_kernel_work(
        cp_size=cp_size,
        chain_prefix_segments=tuple(chain_prefix_segments),
        chain_completion_segments=tuple(chain_completion_segments),
        chain_rank_counts_by_key=chain_rank_counts_by_key,
        planner_config=planner_config,
    )
    return _score_cp_segment_stats(
        rank_local_work=local_work_by_rank,
        rank_chain_work=chain_work_by_rank,
        rank_real_tokens=tuple(rank_loads),
        cross_rank_token_count=cross_rank_token_count,
        parent_state_exchange_family_count=len(parent_state_exchange_families),
        local_bucket_count=local_bucket_count,
        local_segment_count=local_segment_count,
        chain_bucket_count=chain_bucket_count,
        planner_config=planner_config,
    )


def _estimate_local_rank_kernel_work(
    local_prefix_segments_by_rank: tuple[tuple[GdnSegmentSpec, ...], ...],
    local_completion_segments_by_rank: tuple[tuple[GdnSegmentSpec, ...], ...],
    *,
    planner_config: GdnPlannerConfig,
) -> tuple[tuple[int, ...], int, int]:
    rank_work: list[int] = []
    rank_bucket_counts: list[int] = []
    rank_segment_counts: list[int] = []
    for prefix_segments, completion_segments in zip(
        local_prefix_segments_by_rank,
        local_completion_segments_by_rank,
        strict=True,
    ):
        prefix_family_indices = {segment.family_index for segment in prefix_segments}
        chunk_local_completion_segments = tuple(
            segment
            for segment in completion_segments
            if segment.family_index in prefix_family_indices
        )
        plain_local_completion_segments = tuple(
            segment
            for segment in completion_segments
            if segment.family_index not in prefix_family_indices
        )
        chunk_work, chunk_bucket_count = _estimate_chunk_aligned_local_work(
            prefix_segments,
            chunk_local_completion_segments,
            planner_config=planner_config,
        )
        completion_work, completion_bucket_count = _padded_work_from_lengths(
            tuple(segment.length for segment in plain_local_completion_segments),
            max_padding_ratio=planner_config.max_padding_ratio,
            max_segments_per_batch=planner_config.max_segments_per_batch,
        )
        rank_work.append(chunk_work + completion_work)
        rank_bucket_counts.append(chunk_bucket_count + completion_bucket_count)
        rank_segment_counts.append(len(prefix_segments) + len(completion_segments))
    return (
        tuple(rank_work),
        max(rank_bucket_counts, default=0),
        max(rank_segment_counts, default=0),
    )


def _estimate_chunk_aligned_local_work(
    prefix_segments: tuple[GdnSegmentSpec, ...],
    completion_segments: tuple[GdnSegmentSpec, ...],
    *,
    planner_config: GdnPlannerConfig,
) -> tuple[int, int]:
    completions_by_family: dict[int, list[GdnSegmentSpec]] = {}
    for completion in completion_segments:
        completions_by_family.setdefault(completion.family_index, []).append(completion)
    boundary_lengths: list[int] = []
    tail_lengths: list[int] = []
    completion_column_lengths: list[int] = []
    for prefix in prefix_segments:
        boundary_end = _prefix_chunk_boundary_end(prefix)
        boundary_length = boundary_end - prefix.start
        if boundary_length > 0:
            boundary_lengths.append(boundary_length)
        tail_length = prefix.end - boundary_end
        family_completions = tuple(completions_by_family.get(prefix.family_index, ()))
        if tail_length > 0 and not family_completions:
            tail_lengths.append(tail_length)
        for completion in family_completions:
            completion_column_lengths.append(tail_length + completion.length)
    boundary_work, boundary_bucket_count = _padded_work_from_lengths(
        tuple(boundary_lengths),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    tail_work, tail_bucket_count = _padded_work_from_lengths(
        tuple(tail_lengths),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    completion_work, completion_bucket_count = _padded_work_from_lengths(
        tuple(completion_column_lengths),
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    return (
        boundary_work + tail_work + completion_work,
        boundary_bucket_count + tail_bucket_count + completion_bucket_count,
    )


def _estimate_chain_rank_kernel_work(
    *,
    cp_size: int,
    chain_prefix_segments: tuple[GdnSegmentSpec, ...],
    chain_completion_segments: tuple[GdnSegmentSpec, ...],
    chain_rank_counts_by_key: dict[GdnSegmentDecisionKey, tuple[int, ...]],
    planner_config: GdnPlannerConfig,
) -> tuple[tuple[int, ...], int]:
    rank_work = [0] * cp_size
    bucket_count = 0
    for segments in (chain_prefix_segments, chain_completion_segments):
        buckets = _batch_segments_by_padded_work(
            segments,
            max_padding_ratio=planner_config.max_padding_ratio,
            max_segments_per_batch=planner_config.max_segments_per_batch,
        )
        bucket_count += len(buckets)
        for bucket in buckets:
            for rank in range(cp_size):
                lengths = tuple(
                    chain_rank_counts_by_key[_segment_key(segment)][rank]
                    for segment in bucket
                )
                rank_work[rank] += max(lengths, default=0) * len(lengths)
    return tuple(rank_work), bucket_count


def _padded_work_from_lengths(
    lengths: tuple[int, ...],
    *,
    max_padding_ratio: float,
    max_segments_per_batch: int,
) -> tuple[int, int]:
    if not lengths:
        return 0, 0
    ordered = sorted(length for length in lengths if length > 0)
    if not ordered:
        return 0, 0
    bucket_count = 0
    padded_work = 0
    current_count = 0
    current_tokens = 0
    current_max = 0
    for length in ordered:
        next_count = current_count + 1
        next_tokens = current_tokens + length
        next_max = max(current_max, length)
        next_padded = next_max * next_count
        can_extend = current_count == 0 or (
            next_count <= max_segments_per_batch
            and next_padded <= max_padding_ratio * next_tokens
        )
        if not can_extend:
            bucket_count += 1
            padded_work += current_max * current_count
            current_count = 0
            current_tokens = 0
            current_max = 0
        current_count += 1
        current_tokens += length
        current_max = max(current_max, length)
    if current_count:
        bucket_count += 1
        padded_work += current_max * current_count
    return padded_work, bucket_count


def _add_chain_search_load(
    rank_loads: list[int],
    segment: GdnSegmentSpec,
    *,
    chain_rank_counts_by_key: dict[GdnSegmentDecisionKey, tuple[int, ...]],
    chain_cross_rank_tokens_by_key: dict[GdnSegmentDecisionKey, int],
) -> int:
    key = _segment_key(segment)
    for rank, token_count in enumerate(chain_rank_counts_by_key[key]):
        rank_loads[rank] += token_count
    return chain_cross_rank_tokens_by_key[key]


def _add_local_search_load(
    rank_loads: list[int],
    rank: int,
    segment: GdnSegmentSpec,
    *,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
) -> int:
    rank_loads[rank] += segment.length
    return segment.length - segment_attention_counts[_segment_key(segment)][rank]


def _chain_segment_rank_counts_and_cross_rank_tokens(
    segment: GdnSegmentSpec,
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    attention_layout_index: _AttentionLayoutIndex,
) -> tuple[tuple[int, ...], int]:
    token_start = _segment_token_start(segment, spec.sequence_length)
    attention_shards = _attention_contiguous_chain_shards(
        token_start,
        segment.length,
        cp_size=cp_size,
        attention_layout_index=attention_layout_index,
    )
    if attention_shards is not None:
        return tuple(len(shard) for shard in attention_shards), 0
    shard_lengths = _fla_aligned_chain_shard_lengths(segment.length, cp_size=cp_size)
    cross_rank_tokens = 0
    start = 0
    for rank, shard_length in enumerate(shard_lengths):
        end = start + shard_length
        shard_start = token_start + start
        cross_rank_tokens += shard_length - _attention_overlap_count(
            attention_layout_index,
            rank,
            shard_start,
            shard_start + shard_length,
        )
        start = end
    return shard_lengths, cross_rank_tokens


def _materialize_cp_segment_schedule(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    attention_layout_index: _AttentionLayoutIndex,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
    chain_segment_keys: frozenset[GdnSegmentDecisionKey],
    co_locate_local_families: bool,
    planner_config: GdnPlannerConfig,
) -> GdnCpSegmentSchedule:
    gdn_ranges_by_rank: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
    rank_loads = [0] * cp_size
    local_prefix_segments_by_rank: list[list[GdnSegmentSpec]] = [
        [] for _ in range(cp_size)
    ]
    local_completion_segments_by_rank: list[list[GdnSegmentSpec]] = [
        [] for _ in range(cp_size)
    ]
    chain_prefix_segments: list[GdnSegmentSpec] = []
    chain_completion_segments: list[GdnSegmentSpec] = []
    parent_state_exchange_families: set[int] = set()
    parent_state_transfer_families: dict[tuple[int, int], set[int]] = {}
    cross_rank_token_count = 0

    for family in spec.families:
        prefix_key = _segment_key(family.prefix)
        chain_prefix = prefix_key in chain_segment_keys
        local_completions = tuple(
            completion
            for completion in family.completions
            if _segment_key(completion) not in chain_segment_keys
        )
        prefix_owner: int | None = None
        if chain_prefix:
            chain_prefix_segments.append(family.prefix)
            cross_rank_token_count += _append_chain_segment(
                gdn_ranges_by_rank,
                rank_loads,
                family.prefix,
                spec,
                attention_layout_index=attention_layout_index,
            )
        else:
            owner_segments = (
                (family.prefix, *local_completions)
                if co_locate_local_families
                else (family.prefix,)
            )
            prefix_owner = _best_segment_owner(
                owner_segments,
                rank_loads,
                segment_attention_counts=segment_attention_counts,
                planner_config=planner_config,
            )
            local_prefix_segments_by_rank[prefix_owner].append(family.prefix)
            cross_rank_token_count += _append_local_segment(
                gdn_ranges_by_rank,
                rank_loads,
                prefix_owner,
                family.prefix,
                spec,
                segment_attention_counts=segment_attention_counts,
            )
        for completion in family.completions:
            if _segment_key(completion) in chain_segment_keys:
                chain_completion_segments.append(completion)
                cross_rank_token_count += _append_chain_segment(
                    gdn_ranges_by_rank,
                    rank_loads,
                    completion,
                    spec,
                    attention_layout_index=attention_layout_index,
                )
                if not chain_prefix:
                    if prefix_owner is None:
                        raise RuntimeError(
                            "local-prefix/chained-completion planning lost the prefix owner"
                        )
                    parent_state_exchange_families.add(family.family_index)
                    for dest_rank in range(cp_size):
                        if dest_rank == prefix_owner:
                            continue
                        parent_state_transfer_families.setdefault(
                            (prefix_owner, dest_rank), set()
                        ).add(family.family_index)
                continue
            if co_locate_local_families and not chain_prefix:
                if prefix_owner is None:
                    raise RuntimeError(
                        "co-located local completion planning lost the prefix owner"
                    )
                owner = prefix_owner
            else:
                owner = _best_segment_owner(
                    (completion,),
                    rank_loads,
                    segment_attention_counts=segment_attention_counts,
                    planner_config=planner_config,
                )
            if not chain_prefix:
                if prefix_owner is None:
                    raise RuntimeError(
                        "local completion planning lost the prefix owner"
                    )
                if owner != prefix_owner:
                    parent_state_exchange_families.add(family.family_index)
                    parent_state_transfer_families.setdefault(
                        (prefix_owner, owner), set()
                    ).add(family.family_index)
            local_completion_segments_by_rank[owner].append(completion)
            cross_rank_token_count += _append_local_segment(
                gdn_ranges_by_rank,
                rank_loads,
                owner,
                completion,
                spec,
                segment_attention_counts=segment_attention_counts,
            )

    return GdnCpSegmentSchedule.model_construct(
        gdn_token_counts_by_rank=tuple(rank_loads),
        gdn_token_ranges_by_rank=tuple(tuple(ranges) for ranges in gdn_ranges_by_rank),
        cross_rank_token_count=cross_rank_token_count,
        chain_prefix_buckets=_batch_segments_by_padded_work(
            tuple(chain_prefix_segments),
            max_padding_ratio=planner_config.max_padding_ratio,
            max_segments_per_batch=planner_config.max_segments_per_batch,
        ),
        chain_completion_buckets=_batch_segments_by_padded_work(
            tuple(chain_completion_segments),
            max_padding_ratio=planner_config.max_padding_ratio,
            max_segments_per_batch=planner_config.max_segments_per_batch,
        ),
        local_prefix_segments_by_rank=tuple(
            tuple(segments) for segments in local_prefix_segments_by_rank
        ),
        local_completion_segments_by_rank=tuple(
            tuple(segments) for segments in local_completion_segments_by_rank
        ),
        parent_state_exchange_family_indices=tuple(
            sorted(parent_state_exchange_families)
        ),
        parent_state_transfers=_build_parent_state_transfer_plans(
            parent_state_transfer_families
        ),
    )


def _build_local_family_rank_execution_plan(
    spec: GdnPackedExecutionSpec,
    *,
    device: torch.device | str,
    cp_rank: int,
    cp_size: int,
    planner_config: GdnPlannerConfig,
) -> GdnRankExecutionPlan | None:
    if cp_size <= 1 or not spec.families:
        return None
    target_rank_load = spec.real_token_count / cp_size
    loads = [0] * cp_size
    prefix_owner_by_family: list[int] = []
    completion_owners_by_family: list[tuple[int, ...]] = []
    for family in spec.families:
        if _has_chainable_segment(
            family, cp_size=cp_size, planner_config=planner_config
        ):
            return None
        prefix_locality_limit = max(
            planner_config.max_zero_exchange_load_imbalance * target_rank_load,
            min(64.0, float(spec.real_token_count)),
        )
        if family.prefix.length > prefix_locality_limit:
            return None
        owner = _least_loaded_rank(loads)
        prefix_owner_by_family.append(owner)
        completion_owners_by_family.append(tuple(owner for _ in family.completions))
        loads[owner] += family.token_count

    if max(loads, default=0) > (
        planner_config.local_completion_rebalance_min_imbalance * target_rank_load
    ):
        completion_owners_by_family = list(
            _rebalance_local_completion_segments(
                spec,
                prefix_owner_by_family=tuple(prefix_owner_by_family),
                completion_owners_by_family=tuple(completion_owners_by_family),
                initial_loads=tuple(loads),
                planner_config=planner_config,
            )
        )
    rank_assignments = _materialize_local_family_rank_assignments(
        spec,
        cp_size=cp_size,
        prefix_owner_by_family=tuple(prefix_owner_by_family),
        completion_owners_by_family=tuple(completion_owners_by_family),
    )
    local_token_count, local_token_ranges, prefix_segments, completion_segments = (
        rank_assignments[cp_rank]
    )
    parent_state_transfer_families: dict[tuple[int, int], set[int]] = {}
    for family in spec.families:
        prefix_owner = prefix_owner_by_family[family.family_index]
        completion_owners = completion_owners_by_family[family.family_index]
        for completion_owner in sorted(set(completion_owners)):
            if completion_owner == prefix_owner:
                continue
            parent_state_transfer_families.setdefault(
                (prefix_owner, completion_owner), set()
            ).add(family.family_index)

    from art.megatron.gdn.layout import GdnCpExchangePlan, GdnCpPeerTransfer

    token_counts_by_rank = tuple(assignment[0] for assignment in rank_assignments)
    identity_exchange = GdnCpExchangePlan.model_construct(
        cp_size=cp_size,
        source_token_counts_by_rank=token_counts_by_rank,
        dest_token_counts_by_rank=token_counts_by_rank,
        transfers=tuple(
            GdnCpPeerTransfer.model_construct(
                source_rank=rank,
                dest_rank=rank,
                token_count=token_count,
                source_positions_tensor=None,
                dest_positions_tensor=None,
            )
            for rank, token_count in enumerate(token_counts_by_rank)
            if token_count
        ),
    )
    parent_state_exchange_family_indices = tuple(
        sorted(
            family_index
            for family_indices in parent_state_transfer_families.values()
            for family_index in family_indices
        )
    )
    schedule = GdnCpSegmentSchedule.model_construct(
        gdn_token_counts_by_rank=token_counts_by_rank,
        gdn_token_ranges_by_rank=tuple(
            assignment[1] for assignment in rank_assignments
        ),
        cross_rank_token_count=0,
        chain_prefix_buckets=(),
        chain_completion_buckets=(),
        local_prefix_segments_by_rank=tuple(
            assignment[2] for assignment in rank_assignments
        ),
        local_completion_segments_by_rank=tuple(
            assignment[3] for assignment in rank_assignments
        ),
        parent_state_exchange_family_indices=parent_state_exchange_family_indices,
        parent_state_transfers=_build_parent_state_transfer_plans(
            parent_state_transfer_families
        ),
    )
    if parent_state_exchange_family_indices:
        (
            remote_prefix_tail_buckets,
            remote_completion_with_prefix_tail_buckets,
            remote_prefix_tail_exchange,
            remote_prefix_tail_backward_exchange,
            remote_prefix_tail_state_transfers,
            remote_prefix_tail_families,
        ) = _build_remote_prefix_tail_plans(
            spec,
            schedule,
            cp_rank=cp_rank,
            device=device,
            planner_config=planner_config,
        )
    else:
        (
            remote_prefix_tail_buckets,
            remote_completion_with_prefix_tail_buckets,
            remote_prefix_tail_exchange,
            remote_prefix_tail_backward_exchange,
            remote_prefix_tail_state_transfers,
            remote_prefix_tail_families,
        ) = _empty_remote_prefix_tail_plans()
    local_prefix_family_indices = {segment.family_index for segment in prefix_segments}
    chunk_local_completion_segments = tuple(
        segment
        for segment in completion_segments
        if segment.family_index in local_prefix_family_indices
    )
    suffix_only_completion_segments = tuple(
        segment
        for segment in completion_segments
        if segment.family_index not in local_prefix_family_indices
        and segment.family_index not in remote_prefix_tail_families
    )
    ready_completion_segments, remote_completion_segments = (
        _split_ready_and_remote_completion_segments(
            suffix_only_completion_segments,
            local_prefix_segments=(),
            chain_prefix_buckets=(),
        )
    )
    ready_completion_buckets = _batch_segments_by_padded_work(
        ready_completion_segments,
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    remote_completion_buckets = _batch_segments_by_padded_work(
        remote_completion_segments,
        max_padding_ratio=planner_config.max_padding_ratio,
        max_segments_per_batch=planner_config.max_segments_per_batch,
    )
    ready_completion_bucket_plans = _build_position_bucket_plans(
        ready_completion_buckets,
        local_token_ranges,
        sequence_length=spec.sequence_length,
        device=device,
    )
    remote_completion_bucket_plans = _build_position_bucket_plans(
        remote_completion_buckets,
        local_token_ranges,
        sequence_length=spec.sequence_length,
        device=device,
    )
    local_completion_bucket_plans = (
        ready_completion_bucket_plans + remote_completion_bucket_plans
    )
    (
        prefix_boundary_buckets,
        prefix_tail_buckets,
        completion_with_prefix_tail_buckets,
    ) = _build_chunk_aligned_position_bucket_plans(
        prefix_segments,
        chunk_local_completion_segments,
        local_token_ranges,
        sequence_length=spec.sequence_length,
        device=device,
        planner_config=planner_config,
    )
    return GdnRankExecutionPlan.model_construct(
        cp_rank=cp_rank,
        cp_size=cp_size,
        batch_size=1,
        sequence_length=local_token_count,
        packed_batch_size=spec.batch_size,
        packed_sequence_length=spec.sequence_length,
        real_token_mask=torch.ones(
            1, local_token_count, device=device, dtype=torch.bool
        ),
        family_count=spec.family_count,
        completion_count=spec.completion_count,
        local_prefix_buckets=(),
        local_completion_buckets=local_completion_bucket_plans,
        ready_local_completion_buckets=ready_completion_bucket_plans,
        remote_local_completion_buckets=remote_completion_bucket_plans,
        chain_prefix_buckets=(),
        chain_completion_buckets=(),
        prefix_table_is_dense_ordered=(
            tuple(segment.family_index for segment in prefix_segments)
            == tuple(range(spec.family_count))
        ),
        attention_to_gdn=identity_exchange,
        gdn_to_attention=identity_exchange,
        attention_token_ranges=local_token_ranges,
        gdn_token_ranges=local_token_ranges,
        attention_token_count=local_token_count,
        gdn_token_count=local_token_count,
        parent_state_exchange_family_indices=tuple(
            family_index
            for family_index in parent_state_exchange_family_indices
            if family_index not in remote_prefix_tail_families
        ),
        parent_state_transfers=_filter_parent_state_transfers(
            _build_parent_state_transfer_plans(parent_state_transfer_families),
            excluded_families=remote_prefix_tail_families,
            device=device,
        ),
        prefix_boundary_buckets=prefix_boundary_buckets,
        prefix_tail_buckets=prefix_tail_buckets,
        completion_with_prefix_tail_buckets=completion_with_prefix_tail_buckets,
        remote_prefix_tail_buckets=remote_prefix_tail_buckets,
        remote_completion_with_prefix_tail_buckets=remote_completion_with_prefix_tail_buckets,
        remote_prefix_tail_exchange=remote_prefix_tail_exchange,
        remote_prefix_tail_backward_exchange=remote_prefix_tail_backward_exchange,
        remote_prefix_tail_state_transfers=remote_prefix_tail_state_transfers,
    )


def _rebalance_local_completion_segments(
    spec: GdnPackedExecutionSpec,
    *,
    prefix_owner_by_family: tuple[int, ...],
    completion_owners_by_family: tuple[tuple[int, ...], ...],
    initial_loads: tuple[int, ...],
    planner_config: GdnPlannerConfig,
) -> tuple[tuple[int, ...], ...]:
    owners = [list(family_owners) for family_owners in completion_owners_by_family]
    loads = list(initial_loads)
    remote_owners_by_family = [
        {
            owner
            for owner in family_owners
            if owner != prefix_owner_by_family[family_index]
        }
        for family_index, family_owners in enumerate(owners)
    ]
    transfer_count = sum(
        len(remote_owners) for remote_owners in remote_owners_by_family
    )

    def score(candidate_loads: list[int], candidate_transfer_count: int) -> float:
        max_load = max(candidate_loads, default=0)
        idle_tokens = sum(max_load - load for load in candidate_loads)
        return (
            max_load
            + planner_config.rank_idle_token_cost * idle_tokens
            + planner_config.parent_state_exchange_penalty_tokens
            * candidate_transfer_count
        )

    best_score = score(loads, transfer_count)
    while True:
        best_move: (
            tuple[int, int, int, tuple[int, ...], list[int], int, float] | None
        ) = None
        for family in spec.families:
            family_owners = owners[family.family_index]
            prefix_owner = prefix_owner_by_family[family.family_index]
            original_remote_owners = remote_owners_by_family[family.family_index]
            for source in sorted(set(family_owners)):
                source_children = [
                    child_index
                    for child_index, owner in enumerate(family_owners)
                    if owner == source
                ]
                ordered_children = sorted(
                    source_children,
                    key=lambda child_index: family.completions[child_index].length,
                    reverse=True,
                )
                for dest in range(len(loads)):
                    if dest == source:
                        continue
                    moved_tokens = 0
                    moved_children = []
                    for child_index in ordered_children:
                        moved_tokens += family.completions[child_index].length
                        moved_children.append(child_index)
                        candidate_loads = list(loads)
                        candidate_loads[source] -= moved_tokens
                        candidate_loads[dest] += moved_tokens
                        candidate_remote_owners = set(original_remote_owners)
                        if source != prefix_owner and len(moved_children) == len(
                            source_children
                        ):
                            candidate_remote_owners.discard(source)
                        if dest != prefix_owner:
                            candidate_remote_owners.add(dest)
                        candidate_transfer_count = (
                            transfer_count
                            - len(original_remote_owners)
                            + len(candidate_remote_owners)
                        )
                        candidate_score = score(
                            candidate_loads, candidate_transfer_count
                        )
                        if candidate_score >= best_score:
                            continue
                        if best_move is None or candidate_score < best_move[-1]:
                            best_move = (
                                family.family_index,
                                source,
                                dest,
                                tuple(moved_children),
                                candidate_loads,
                                candidate_transfer_count,
                                candidate_score,
                            )
        if best_move is None:
            return tuple(tuple(item) for item in owners)
        (
            family_index,
            _source,
            dest,
            moved_children,
            loads,
            transfer_count,
            best_score,
        ) = best_move
        for child_index in moved_children:
            owners[family_index][child_index] = dest
        prefix_owner = prefix_owner_by_family[family_index]
        remote_owners_by_family[family_index] = {
            owner for owner in set(owners[family_index]) if owner != prefix_owner
        }


def _materialize_local_family_rank_assignments(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    prefix_owner_by_family: tuple[int, ...],
    completion_owners_by_family: tuple[tuple[int, ...], ...],
) -> tuple[
    tuple[
        int,
        tuple[tuple[int, int, int], ...],
        tuple[GdnSegmentSpec, ...],
        tuple[GdnSegmentSpec, ...],
    ],
    ...,
]:
    token_ranges_by_rank: list[list[tuple[int, int, int]]] = [
        [] for _ in range(cp_size)
    ]
    token_counts_by_rank = [0] * cp_size
    prefix_segments_by_rank: list[list[GdnSegmentSpec]] = [[] for _ in range(cp_size)]
    completion_segments_by_rank: list[list[GdnSegmentSpec]] = [
        [] for _ in range(cp_size)
    ]
    sequence_length = spec.sequence_length
    for family in spec.families:
        prefix_owner = prefix_owner_by_family[family.family_index]
        prefix_segments_by_rank[prefix_owner].append(family.prefix)
        prefix_token_start = (
            family.prefix.row_index * sequence_length + family.prefix.start
        )
        prefix_position_start = token_counts_by_rank[prefix_owner]
        token_ranges_by_rank[prefix_owner].append(
            (
                prefix_token_start,
                prefix_token_start + family.prefix.length,
                prefix_position_start,
            )
        )
        token_counts_by_rank[prefix_owner] = (
            prefix_position_start + family.prefix.length
        )
        for completion, completion_owner in zip(
            family.completions,
            completion_owners_by_family[family.family_index],
            strict=True,
        ):
            completion_segments_by_rank[completion_owner].append(completion)
            completion_token_start = (
                completion.row_index * sequence_length + completion.start
            )
            completion_position_start = token_counts_by_rank[completion_owner]
            token_ranges_by_rank[completion_owner].append(
                (
                    completion_token_start,
                    completion_token_start + completion.length,
                    completion_position_start,
                )
            )
            token_counts_by_rank[completion_owner] = (
                completion_position_start + completion.length
            )
    return tuple(
        (
            token_counts_by_rank[rank],
            tuple(token_ranges_by_rank[rank]),
            tuple(prefix_segments_by_rank[rank]),
            tuple(completion_segments_by_rank[rank]),
        )
        for rank in range(cp_size)
    )


def _empty_local_family_rank_execution_plan(
    spec: GdnPackedExecutionSpec,
    *,
    device: torch.device | str,
    cp_rank: int,
    cp_size: int,
) -> GdnRankExecutionPlan:
    from art.megatron.gdn.layout import GdnCpExchangePlan

    identity_exchange = GdnCpExchangePlan.model_construct(
        cp_size=cp_size,
        source_token_counts_by_rank=tuple(0 for _ in range(cp_size)),
        dest_token_counts_by_rank=tuple(0 for _ in range(cp_size)),
        transfers=(),
    )
    return GdnRankExecutionPlan.model_construct(
        cp_rank=cp_rank,
        cp_size=cp_size,
        batch_size=1,
        sequence_length=0,
        packed_batch_size=spec.batch_size,
        packed_sequence_length=spec.sequence_length,
        real_token_mask=torch.ones(1, 0, device=device, dtype=torch.bool),
        family_count=spec.family_count,
        completion_count=spec.completion_count,
        local_prefix_buckets=(),
        local_completion_buckets=(),
        ready_local_completion_buckets=(),
        remote_local_completion_buckets=(),
        chain_prefix_buckets=(),
        chain_completion_buckets=(),
        prefix_table_is_dense_ordered=False,
        attention_to_gdn=identity_exchange,
        gdn_to_attention=identity_exchange,
        attention_token_ranges=(),
        gdn_token_ranges=(),
        attention_token_count=0,
        gdn_token_count=0,
        parent_state_exchange_family_indices=(),
        parent_state_transfers=(),
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


def _build_parent_state_transfer_plans(
    families_by_peer: dict[tuple[int, int], set[int]],
) -> tuple[GdnParentStateTransferPlan, ...]:
    return tuple(
        GdnParentStateTransferPlan(
            source_rank=source_rank,
            dest_rank=dest_rank,
            family_indices=tuple(sorted(family_indices)),
        )
        for (source_rank, dest_rank), family_indices in sorted(families_by_peer.items())
        if source_rank != dest_rank and family_indices
    )


def _split_ready_and_remote_completion_segments(
    completion_segments: tuple[GdnSegmentSpec, ...],
    *,
    local_prefix_segments: tuple[GdnSegmentSpec, ...],
    chain_prefix_buckets: tuple[tuple[GdnSegmentSpec, ...], ...],
) -> tuple[tuple[GdnSegmentSpec, ...], tuple[GdnSegmentSpec, ...]]:
    ready_family_indices = {
        segment.family_index for segment in local_prefix_segments
    } | {segment.family_index for bucket in chain_prefix_buckets for segment in bucket}
    ready = []
    remote = []
    for segment in completion_segments:
        if segment.family_index in ready_family_indices:
            ready.append(segment)
        else:
            remote.append(segment)
    return tuple(ready), tuple(remote)


def _transfer_plans_to_device(
    transfers: tuple[GdnParentStateTransferPlan, ...],
    *,
    device: torch.device | str,
) -> tuple[GdnParentStateTransferPlan, ...]:
    return tuple(
        transfer.model_copy(
            update={
                "family_indices_tensor": _move_planner_tensor(
                    torch.tensor(transfer.family_indices, dtype=torch.long),
                    device,
                )
            }
        )
        for transfer in transfers
    )


def _has_chainable_segment(
    family: GdnPackedFamilySpec,
    *,
    cp_size: int,
    planner_config: GdnPlannerConfig,
) -> bool:
    return _can_chain_prefix_segment(
        family.prefix, cp_size=cp_size, planner_config=planner_config
    ) or any(
        _can_chain_segment(completion, cp_size=cp_size, planner_config=planner_config)
        for completion in family.completions
    )


def _can_chain_prefix_segment(
    segment: GdnSegmentSpec,
    *,
    cp_size: int,
    planner_config: GdnPlannerConfig,
) -> bool:
    return _can_chain_segment(segment, cp_size=cp_size, planner_config=planner_config)


def _score_cp_segment_stats(
    *,
    rank_local_work: tuple[int, ...],
    rank_chain_work: tuple[int, ...],
    rank_real_tokens: tuple[int, ...],
    cross_rank_token_count: int,
    parent_state_exchange_family_count: int,
    local_bucket_count: int,
    local_segment_count: int,
    chain_bucket_count: int,
    planner_config: GdnPlannerConfig,
) -> float:
    empty_rank_count = sum(1 for token_count in rank_real_tokens if token_count == 0)
    return (
        _rank_kernel_ms(
            rank_local_work,
            rank_chain_work,
            local_token_ms=planner_config.planner_local_token_ms,
            chain_token_ms=planner_config.planner_chain_token_ms,
        )
        + planner_config.planner_local_bucket_ms * local_bucket_count
        + planner_config.planner_chain_bucket_ms * chain_bucket_count
        + planner_config.planner_local_segment_ms * local_segment_count
        + planner_config.planner_layout_cross_rank_token_ms * cross_rank_token_count
        + (
            planner_config.planner_parent_state_exchange_base_ms
            + planner_config.planner_parent_state_exchange_ms
            * parent_state_exchange_family_count
            if parent_state_exchange_family_count
            else 0.0
        )
        + planner_config.planner_empty_rank_ms * empty_rank_count
    )


def _rank_kernel_ms(
    rank_local_work: tuple[int, ...],
    rank_chain_work: tuple[int, ...],
    *,
    local_token_ms: float,
    chain_token_ms: float,
) -> float:
    return max(
        (
            local_work * local_token_ms + chain_work * chain_token_ms
            for local_work, chain_work in zip(
                rank_local_work, rank_chain_work, strict=True
            )
        ),
        default=0.0,
    )


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
    best: tuple[float, int, int, int, int] | None = None
    for rank, tokens in enumerate(on_rank_tokens):
        projected_loads = list(rank_loads)
        projected_loads[rank] += segment_length
        max_load = max(projected_loads, default=0)
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
        if segment.kind == "prefix":
            return _can_chain_prefix_segment(
                segment, cp_size=cp_size, planner_config=planner_config
            )
        return _can_chain_segment(
            segment, cp_size=cp_size, planner_config=planner_config
        )

    for family in spec.families:
        has_split_segment = any(
            should_split_segment(segment)
            for segment in (family.prefix, *family.completions)
        )
        if not has_split_segment:
            if _should_co_locate_non_chain_family(
                family,
                total_real_tokens=spec.real_token_count,
                cp_size=cp_size,
                planner_config=planner_config,
            ):
                owner = _least_loaded_rank(loads)
                for segment in (family.prefix, *family.completions):
                    token_start = _segment_token_start(segment, spec.sequence_length)
                    append_segment(owner, token_start, segment.length)
                continue
            for segment in (family.prefix, *family.completions):
                token_start = _segment_token_start(segment, spec.sequence_length)
                owner = _least_loaded_rank(loads)
                append_segment(owner, token_start, segment.length)
            continue
        for segment in (family.prefix, *family.completions):
            token_start = _segment_token_start(segment, spec.sequence_length)
            if should_split_segment(segment):
                _append_split_default_attention_segment(
                    ranks, loads, token_start, segment.length
                )
                continue
            owner = _least_loaded_rank(loads)
            append_segment(owner, token_start, segment.length)
    return tuple(tuple(ranges) for ranges in ranks)


def _should_co_locate_non_chain_family(
    family: GdnPackedFamilySpec,
    *,
    total_real_tokens: int,
    cp_size: int,
    planner_config: GdnPlannerConfig,
) -> bool:
    target_rank_load = total_real_tokens / cp_size
    return family.token_count <= (
        planner_config.max_zero_exchange_load_imbalance * target_rank_load
    )


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


def _chain_rank_token_indices(
    segment: GdnSegmentSpec,
    spec: GdnPackedExecutionSpec,
    *,
    cp_rank: int,
    cp_size: int,
) -> range:
    token_start = _segment_token_start(segment, spec.sequence_length)
    lengths = _fla_aligned_chain_shard_lengths(segment.length, cp_size=cp_size)
    start = sum(lengths[:cp_rank])
    end = start + lengths[cp_rank]
    if start >= end:
        raise ValueError(
            "CP chain planning requires non-empty shards; "
            f"segment={segment.kind}:{segment.family_index} "
            f"length={segment.length} cp_size={cp_size}"
        )
    return range(token_start + start, token_start + end)


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


def _owner_rank(
    local_prefix_segments_by_rank: list[list[GdnSegmentSpec]],
    prefix: GdnSegmentSpec,
) -> int:
    for rank, segments in enumerate(local_prefix_segments_by_rank):
        if prefix in segments:
            return rank
    raise RuntimeError("local prefix owner was not recorded")


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


def _local_token_ranges(
    local_gdn_tokens: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    if not local_gdn_tokens:
        return ()
    ranges = []
    token_start = local_gdn_tokens[0]
    token_end = token_start + 1
    position_start = 0
    for position, token in enumerate(local_gdn_tokens[1:], start=1):
        if token == token_end:
            token_end += 1
            continue
        ranges.append((token_start, token_end, position_start))
        token_start = token
        token_end = token + 1
        position_start = position
    ranges.append((token_start, token_end, position_start))
    return tuple(ranges)


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


def _validate_padding_tensor(
    row_index: int,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> int:
    padding_positions = torch.nonzero(group_ids == -1, as_tuple=False)
    valid_length = (
        int(padding_positions[0].item())
        if int(padding_positions.numel()) > 0
        else int(group_ids.numel())
    )
    if valid_length == 0:
        if bool(torch.any(parent_ids != -1).item()):
            raise ValueError(f"row {row_index}: padding parent_ids must be -1")
        return 0
    if bool(torch.any(group_ids[valid_length:] != -1).item()):
        raise ValueError(
            f"row {row_index}: valid tokens must be contiguous before padding"
        )
    if bool(torch.any(parent_ids[:valid_length] == -1).item()):
        raise ValueError(
            f"row {row_index}: valid tokens must have non-padding parent_ids"
        )
    if bool(torch.any(parent_ids[valid_length:] != -1).item()):
        raise ValueError(f"row {row_index}: padding parent_ids must be -1")
    return valid_length


def _validate_padding(
    row_index: int,
    group_ids: list[int],
    parent_ids: list[int],
) -> int:
    valid_length = 0
    for group_id in group_ids:
        if group_id == -1:
            break
        valid_length += 1
    if valid_length == 0:
        if any(parent_id != -1 for parent_id in parent_ids):
            raise ValueError(f"row {row_index}: padding parent_ids must be -1")
        return 0
    if any(group_id != -1 for group_id in group_ids[valid_length:]):
        raise ValueError(
            f"row {row_index}: valid tokens must be contiguous before padding"
        )
    if any(parent_id == -1 for parent_id in parent_ids[:valid_length]):
        raise ValueError(
            f"row {row_index}: valid tokens must have non-padding parent_ids"
        )
    if any(parent_id != -1 for parent_id in parent_ids[valid_length:]):
        raise ValueError(f"row {row_index}: padding parent_ids must be -1")
    return valid_length


def _parse_row_tensor(
    *,
    row_index: int,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    valid_length: int,
    first_family_index: int,
    min_completions_per_family: int,
) -> list[GdnPackedFamilySpec]:
    valid_groups = group_ids[:valid_length]
    valid_parents = parent_ids[:valid_length]
    if valid_length > 1:
        same_group = valid_groups[1:] == valid_groups[:-1]
        parent_changed = same_group & (valid_parents[1:] != valid_parents[:-1])
        if bool(torch.any(parent_changed).item()):
            position = int(torch.nonzero(parent_changed, as_tuple=False)[0].item()) + 1
            group_id = int(valid_groups[position].item())
            previous_parent = int(valid_parents[position - 1].item())
            current_parent = int(valid_parents[position].item())
            raise ValueError(
                f"row {row_index}: group {group_id} changes parent from "
                f"{previous_parent} to {current_parent}"
            )
        boundaries = torch.nonzero(~same_group, as_tuple=False).flatten() + 1
        starts_tensor = torch.cat(
            (valid_groups.new_zeros(1), boundaries.to(valid_groups.dtype))
        )
        ends_tensor = torch.cat(
            (
                boundaries.to(valid_groups.dtype),
                valid_groups.new_tensor([valid_length]),
            )
        )
    else:
        starts_tensor = valid_groups.new_zeros(1)
        ends_tensor = valid_groups.new_tensor([valid_length])

    starts = tuple(int(value) for value in starts_tensor.tolist())
    ends = tuple(int(value) for value in ends_tensor.tolist())
    segment_group_ids = tuple(int(valid_groups[start].item()) for start in starts)
    segment_parent_ids = tuple(int(valid_parents[start].item()) for start in starts)
    families: list[GdnPackedFamilySpec] = []
    seen_groups: set[int] = set()
    segment_cursor = 0
    while segment_cursor < len(starts):
        group_id = segment_group_ids[segment_cursor]
        parent_id = segment_parent_ids[segment_cursor]
        start = starts[segment_cursor]
        end = ends[segment_cursor]
        if group_id in seen_groups:
            raise ValueError(f"row {row_index}: group_id {group_id} is non-contiguous")
        if group_id != parent_id:
            raise ValueError(
                f"row {row_index}: completion group {group_id} appears before "
                f"its prefix parent {parent_id}"
            )
        seen_groups.add(group_id)
        family_index = first_family_index + len(families)
        prefix = _trusted_pydantic_construct(
            GdnSegmentSpec,
            _GDN_SEGMENT_SPEC_FIELDS,
            row_index=row_index,
            family_index=family_index,
            group_id=group_id,
            parent_id=parent_id,
            start=start,
            end=end,
            kind="prefix",
            child_index=None,
        )
        segment_cursor += 1
        completions: list[GdnSegmentSpec] = []
        while segment_cursor < len(starts):
            child_group_id = segment_group_ids[segment_cursor]
            child_parent_id = segment_parent_ids[segment_cursor]
            child_start = starts[segment_cursor]
            child_end = ends[segment_cursor]
            if child_group_id == child_parent_id:
                break
            if child_parent_id != group_id:
                raise ValueError(
                    f"row {row_index}: completion group {child_group_id} has "
                    f"parent {child_parent_id}, expected active prefix {group_id}"
                )
            if child_group_id in seen_groups:
                raise ValueError(
                    f"row {row_index}: group_id {child_group_id} is non-contiguous"
                )
            seen_groups.add(child_group_id)
            completions.append(
                _trusted_pydantic_construct(
                    GdnSegmentSpec,
                    _GDN_SEGMENT_SPEC_FIELDS,
                    row_index=row_index,
                    family_index=family_index,
                    group_id=child_group_id,
                    parent_id=child_parent_id,
                    start=child_start,
                    end=child_end,
                    kind="completion",
                    child_index=len(completions),
                )
            )
            segment_cursor += 1
        if len(completions) < min_completions_per_family:
            raise ValueError(
                f"row {row_index}: prefix group {group_id} has {len(completions)} "
                f"completion(s), expected at least {min_completions_per_family}"
            )
        families.append(
            _trusted_pydantic_construct(
                GdnPackedFamilySpec,
                _GDN_PACKED_FAMILY_SPEC_FIELDS,
                row_index=row_index,
                family_index=family_index,
                prefix=prefix,
                completions=tuple(completions),
            )
        )
    return families


def _parse_row(
    *,
    row_index: int,
    group_ids: list[int],
    parent_ids: list[int],
    valid_length: int,
    first_family_index: int,
    min_completions_per_family: int,
) -> list[GdnPackedFamilySpec]:
    families: list[GdnPackedFamilySpec] = []
    seen_groups: set[int] = set()
    cursor = 0
    while cursor < valid_length:
        group_id, parent_id, start, end = _read_segment(
            row_index, group_ids, parent_ids, valid_length, cursor
        )
        if group_id in seen_groups:
            raise ValueError(f"row {row_index}: group_id {group_id} is non-contiguous")
        if group_id != parent_id:
            raise ValueError(
                f"row {row_index}: completion group {group_id} appears before "
                f"its prefix parent {parent_id}"
            )
        seen_groups.add(group_id)
        family_index = first_family_index + len(families)
        prefix = GdnSegmentSpec(
            row_index=row_index,
            family_index=family_index,
            group_id=group_id,
            parent_id=parent_id,
            start=start,
            end=end,
            kind="prefix",
        )
        cursor = end
        completions: list[GdnSegmentSpec] = []
        while cursor < valid_length:
            child_group_id, child_parent_id, child_start, child_end = _read_segment(
                row_index, group_ids, parent_ids, valid_length, cursor
            )
            if child_group_id == child_parent_id:
                break
            if child_parent_id != group_id:
                raise ValueError(
                    f"row {row_index}: completion group {child_group_id} has "
                    f"parent {child_parent_id}, expected active prefix {group_id}"
                )
            if child_group_id in seen_groups:
                raise ValueError(
                    f"row {row_index}: group_id {child_group_id} is non-contiguous"
                )
            seen_groups.add(child_group_id)
            completions.append(
                GdnSegmentSpec(
                    row_index=row_index,
                    family_index=family_index,
                    group_id=child_group_id,
                    parent_id=child_parent_id,
                    start=child_start,
                    end=child_end,
                    kind="completion",
                    child_index=len(completions),
                )
            )
            cursor = child_end
        if len(completions) < min_completions_per_family:
            raise ValueError(
                f"row {row_index}: prefix group {group_id} has {len(completions)} "
                f"completion(s), expected at least {min_completions_per_family}"
            )
        families.append(
            GdnPackedFamilySpec(
                row_index=row_index,
                family_index=family_index,
                prefix=prefix,
                completions=tuple(completions),
            )
        )
    return families


def _read_segment(
    row_index: int,
    group_ids: list[int],
    parent_ids: list[int],
    valid_length: int,
    cursor: int,
) -> tuple[int, int, int, int]:
    group_id = int(group_ids[cursor])
    parent_id = int(parent_ids[cursor])
    if group_id < 0 or parent_id < 0:
        raise ValueError(f"row {row_index}: segment ids must be non-negative")
    start = cursor
    cursor += 1
    while cursor < valid_length and int(group_ids[cursor]) == group_id:
        current_parent = int(parent_ids[cursor])
        if current_parent != parent_id:
            raise ValueError(
                f"row {row_index}: group {group_id} changes parent from "
                f"{parent_id} to {current_parent}"
            )
        cursor += 1
    return group_id, parent_id, start, cursor
