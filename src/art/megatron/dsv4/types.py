from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field
import torch

if TYPE_CHECKING:
    from art.megatron.context_parallel.types import ArtContextParallelState
else:
    ArtContextParallelState = Any


class Dsv4CompressionKind(str, Enum):
    CSA = "csa"
    HCA = "hca"


class Dsv4StageKeyKind(str, Enum):
    RAW = "raw"
    COMPRESSED = "compressed"


class Dsv4StreamKind(str, Enum):
    PREFIX = "prefix"
    COMPLETION = "completion"


class Dsv4FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Dsv4TensorModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class Dsv4WorkModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Dsv4CompressionSpec(Dsv4FrozenModel):
    kind: Dsv4CompressionKind
    ratio: int


class Dsv4StreamSpec(Dsv4FrozenModel):
    stream_id: int
    kind: Dsv4StreamKind
    parent_stream_id: int | None
    start: int
    end: int


class Dsv4BranchView(Dsv4FrozenModel):
    branch_stream_id: int
    prefix_stream_id: int
    suffix_stream_id: int | None
    prefix_start: int
    prefix_end: int
    suffix_start: int | None = None
    suffix_end: int | None = None
    prefix_token_count: int

    def size(self) -> int:
        if self.suffix_start is None or self.suffix_end is None:
            return int(self.prefix_token_count)
        return (
            int(self.prefix_token_count) + int(self.suffix_end) - int(self.suffix_start)
        )

    def position_of_token(self, token_id: int) -> int | None:
        token = int(token_id)
        if int(self.prefix_start) <= token < int(self.prefix_end):
            return token - int(self.prefix_start)
        if (
            self.suffix_start is not None
            and self.suffix_end is not None
            and int(self.suffix_start) <= token < int(self.suffix_end)
        ):
            return int(self.prefix_token_count) + token - int(self.suffix_start)
        return None


class Dsv4HaloTransfer(Dsv4FrozenModel):
    source_rank: int
    target_rank: int
    token_ids: tuple[int, ...]
    entry_ids: tuple[int, ...]


class Dsv4ProjectedTokenBuffer(Dsv4TensorModel):
    token_ids: tuple[int, ...]
    projected_kv: torch.Tensor
    projected_gate: torch.Tensor


class Dsv4TensorExchangePlan(Dsv4FrozenModel):
    send_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_ids_by_peer: tuple[tuple[int, ...], ...]


class Dsv4TensorIdBuffer(Dsv4TensorModel):
    ids: tuple[int, ...]
    tensor: torch.Tensor


class Dsv4CompressionHaloPayload(Dsv4TensorModel):
    source_rank: int
    target_rank: int
    token_ids: tuple[int, ...]
    entry_ids: tuple[int, ...]
    projected_kv: torch.Tensor
    projected_gate: torch.Tensor


class Dsv4CompressionHaloGradientPayload(Dsv4TensorModel):
    source_rank: int
    target_rank: int
    token_ids: tuple[int, ...]
    entry_ids: tuple[int, ...]
    dprojected_kv: torch.Tensor
    dprojected_gate: torch.Tensor


class Dsv4CompressedKvForwardResult(Dsv4TensorModel):
    layout: Dsv4CompressedLayout
    owner_rank: int
    local_token_ids: tuple[int, ...]
    compressed_entry_ids: tuple[int, ...]
    token_buffer: Dsv4ProjectedTokenBuffer
    positional_bias: torch.Tensor
    compressed_kv: torch.Tensor


class Dsv4CompressedKvGradientResult(Dsv4TensorModel):
    token_ids: tuple[int, ...]
    dprojected_kv: torch.Tensor
    dprojected_gate: torch.Tensor
    dpositional_bias: torch.Tensor


class Dsv4TopkResult(Dsv4TensorModel):
    indices: torch.Tensor
    scores: torch.Tensor


class Dsv4StageInputs(Dsv4TensorModel):
    stage_index: int
    query_token_ids: tuple[int, ...]
    raw_token_ids: tuple[int, ...]
    compressed_entry_ids: tuple[int, ...]
    key_kinds: tuple[Dsv4StageKeyKind, ...]
    key_global_ids: tuple[int, ...]
    topk_stage_local: torch.Tensor


class Dsv4IndexerStagePlan(Dsv4FrozenModel):
    stage_index: int
    query_token_ids_by_rank: tuple[tuple[int, ...], ...]
    candidate_entry_ids_by_rank: tuple[tuple[int, ...], ...]


class Dsv4IndexerKvExchangePeerPlan(Dsv4FrozenModel):
    send_entry_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_entry_ids_by_peer: tuple[tuple[int, ...], ...]


class Dsv4StagePlanSlot(Dsv4TensorModel):
    stage_index: int
    stage_plans_by_rank: tuple[Any, ...]


class Dsv4StageKvExchangePeerPlan(Dsv4FrozenModel):
    send_raw_token_ids_by_peer: tuple[tuple[int, ...], ...]
    send_compressed_entry_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_raw_token_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_compressed_entry_ids_by_peer: tuple[tuple[int, ...], ...]


class Dsv4MaterializedStage(Dsv4TensorModel):
    stage_index: int
    query_token_ids: tuple[int, ...]
    q_stage: torch.Tensor
    kv_stage: torch.Tensor
    topk_stage_local: torch.Tensor
    raw_count: int
    compressed_count: int
    key_kinds: tuple[Dsv4StageKeyKind, ...]
    key_global_ids: tuple[int, ...]


class Dsv4SparseForwardResult(Dsv4TensorModel):
    out: torch.Tensor
    lse: torch.Tensor


class Dsv4SparseBackwardResult(Dsv4TensorModel):
    dq: torch.Tensor
    dkv: torch.Tensor
    d_attn_sink: torch.Tensor


class Dsv4StageForwardRecord(Dsv4TensorModel):
    materialized_stage: Dsv4MaterializedStage
    out: torch.Tensor
    lse: torch.Tensor


class Dsv4AttentionForwardResult(Dsv4TensorModel):
    out: torch.Tensor
    lse: torch.Tensor
    real_out: torch.Tensor
    real_lse: torch.Tensor
    query_token_ids: tuple[int, ...]
    attn_sink: torch.Tensor
    scale: float | None
    stage_records: tuple[Dsv4StageForwardRecord, ...]


class Dsv4StageBackwardRecord(Dsv4TensorModel):
    materialized_stage: Dsv4MaterializedStage
    dq_stage: torch.Tensor
    dkv_stage: torch.Tensor


class Dsv4AttentionBackwardReplayResult(Dsv4TensorModel):
    stage_records: tuple[Dsv4StageBackwardRecord, ...]
    d_attn_sink: torch.Tensor


class Dsv4AttentionGradientResult(Dsv4TensorModel):
    query_token_ids: tuple[int, ...]
    raw_token_ids: tuple[int, ...]
    compressed_entry_ids: tuple[int, ...]
    dq: torch.Tensor
    draw_kv: torch.Tensor
    dcompressed_kv: torch.Tensor
    d_attn_sink: torch.Tensor


class Dsv4AttentionBackwardRankPlan(Dsv4FrozenModel):
    query_token_ids: tuple[int, ...]
    raw_token_ids: tuple[int, ...]
    compressed_entry_ids: tuple[int, ...]
    query_owner_ranks: tuple[int, ...]
    raw_owner_ranks: tuple[int, ...]
    compressed_owner_ranks: tuple[int, ...]
    recv_query_token_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_raw_token_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_compressed_entry_ids_by_peer: tuple[tuple[int, ...], ...]
    owned_query_token_ids: tuple[int, ...]
    owned_raw_token_ids: tuple[int, ...]
    owned_compressed_entry_ids: tuple[int, ...]


class Dsv4AttentionBackwardPlan(Dsv4FrozenModel):
    compression_kind: Dsv4CompressionKind
    stage_indices: tuple[int, ...]
    local_rank: int
    local_rank_plan: Dsv4AttentionBackwardRankPlan


class Dsv4ProjectedAttentionForwardResult(Dsv4TensorModel):
    compression_kind: Dsv4CompressionKind
    attention: Dsv4AttentionForwardResult
    main_compressed: Dsv4CompressedKvForwardResult
    indexer_compressed: Dsv4CompressedKvForwardResult | None = None


class Dsv4ProjectedAttentionGradientResult(Dsv4TensorModel):
    attention: Dsv4AttentionGradientResult
    main_compressor: Dsv4CompressedKvGradientResult


class Dsv4GradientOwnerBucket(Dsv4TensorModel):
    owner_rank: int
    query_token_ids: tuple[int, ...]
    raw_token_ids: tuple[int, ...]
    compressed_entry_ids: tuple[int, ...]
    dq: torch.Tensor
    draw_kv: torch.Tensor
    dcompressed_kv: torch.Tensor


class Dsv4CompressedLayout(Dsv4FrozenModel):
    spec: Dsv4CompressionSpec
    streams: tuple[Dsv4StreamSpec, ...]
    branch_views: tuple[Dsv4BranchView, ...]
    compressed_entry_count: int
    halo_transfers: tuple[Dsv4HaloTransfer, ...]
    entry_ids_by_owner_rank: tuple[tuple[int, ...], ...]
    raw_token_owner_ranks: tuple[int, ...]
    raw_token_owner_change_positions: tuple[int, ...] = ()
    compressed_entry_owner_ranks: tuple[int, ...] = ()
    entry_branch_stream_ids: tuple[int, ...] = ()
    entry_prefix_stream_ids: tuple[int, ...] = ()
    entry_closure_view_positions: tuple[int, ...] = ()
    entry_shared_prefix_flags: tuple[bool, ...] = ()
    entry_dependency_start_view_positions: tuple[int, ...] = ()
    closure_token_ids: tuple[int, ...] = ()
    closure_entry_ids: tuple[int, ...] = ()

    def entry_count(self) -> int:
        return int(self.compressed_entry_count)


class Dsv4PreparedPlan(Dsv4FrozenModel):
    csa_layout: Dsv4CompressedLayout | None = None
    hca_layout: Dsv4CompressedLayout | None = None
    stage_plan_slots: tuple[Dsv4StagePlanSlot, ...] = ()
    csa_indexer_stage_plans: tuple[Dsv4IndexerStagePlan, ...] = ()
    csa_indexer_kv_peer_plans_by_stage: tuple[
        tuple[Dsv4IndexerKvExchangePeerPlan, ...], ...
    ] = ()
    csa_stage_kv_peer_plans_by_slot: tuple[
        tuple[Dsv4StageKvExchangePeerPlan, ...], ...
    ] = ()
    hca_stage_kv_peer_plans_by_slot: tuple[
        tuple[Dsv4StageKvExchangePeerPlan, ...], ...
    ] = ()
    csa_attention_backward_plan: Dsv4AttentionBackwardPlan | None = None
    hca_attention_backward_plan: Dsv4AttentionBackwardPlan | None = None


class Dsv4ContextParallelState(Dsv4TensorModel):
    cp_state: ArtContextParallelState
    dsv4_plan: Dsv4PreparedPlan
    extra: dict[str, Any] = Field(default_factory=dict)
