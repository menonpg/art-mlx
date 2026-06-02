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


class Dsv4CompressionSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Dsv4CompressionKind
    ratio: int


class Dsv4StreamSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    stream_id: int
    kind: Dsv4StreamKind
    parent_stream_id: int | None
    start: int
    end: int

    def size(self) -> int:
        return int(self.end) - int(self.start)


class Dsv4TokenInView(BaseModel):
    model_config = ConfigDict(frozen=True)

    packed_token_id: int
    stream_id: int
    view_pos: int
    stream_pos: int


class Dsv4BranchView(BaseModel):
    model_config = ConfigDict(frozen=True)

    branch_stream_id: int
    prefix_stream_id: int
    suffix_stream_id: int | None
    tokens: tuple[Dsv4TokenInView, ...]
    prefix_token_count: int

    def size(self) -> int:
        return len(self.tokens)


class Dsv4CompressedEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry_id: int
    kind: Dsv4CompressionKind
    ratio: int
    branch_stream_id: int
    prefix_stream_id: int
    closure_token_id: int
    closure_view_pos: int
    owner_rank: int
    owner_local_offset: int
    dependency_token_ids: tuple[int, ...]
    remote_dependency_token_ids: tuple[int, ...]
    shared_prefix_entry: bool
    branch_entry_index: int


class Dsv4HaloTransfer(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_rank: int
    target_rank: int
    token_ids: tuple[int, ...]
    entry_ids: tuple[int, ...]


class Dsv4ProjectedTokenBuffer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    token_ids: tuple[int, ...]
    projected_kv: torch.Tensor
    projected_gate: torch.Tensor


class Dsv4TensorExchangePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    send_ids_by_peer: tuple[tuple[int, ...], ...]
    recv_ids_by_peer: tuple[tuple[int, ...], ...]


class Dsv4TensorIdBuffer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    ids: tuple[int, ...]
    tensor: torch.Tensor


class Dsv4CompressionHaloPayload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source_rank: int
    target_rank: int
    token_ids: tuple[int, ...]
    entry_ids: tuple[int, ...]
    projected_kv: torch.Tensor
    projected_gate: torch.Tensor


class Dsv4CompressionHaloGradientPayload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source_rank: int
    target_rank: int
    token_ids: tuple[int, ...]
    entry_ids: tuple[int, ...]
    dprojected_kv: torch.Tensor
    dprojected_gate: torch.Tensor


class Dsv4TopkResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    indices: torch.Tensor
    scores: torch.Tensor


class Dsv4StageInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    stage_index: int
    query_token_ids: tuple[int, ...]
    raw_token_ids: tuple[int, ...]
    compressed_entry_ids: tuple[int, ...]
    key_kinds: tuple[Dsv4StageKeyKind, ...]
    key_global_ids: tuple[int, ...]
    raw_token_ids_by_query: tuple[tuple[int, ...], ...]
    compressed_entry_ids_by_query: tuple[tuple[int, ...], ...]
    topk_stage_local: torch.Tensor


class Dsv4IndexerStagePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage_index: int
    query_token_ids_by_rank: tuple[tuple[int, ...], ...]
    candidate_entry_ids_by_rank: tuple[tuple[int, ...], ...]


class Dsv4StagePlanGroup(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    stage_index: int
    stage_inputs_by_rank: tuple[Dsv4StageInputs, ...]


class Dsv4MaterializedStage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    stage_index: int
    query_token_ids: tuple[int, ...]
    q_stage: torch.Tensor
    kv_stage: torch.Tensor
    topk_stage_local: torch.Tensor
    raw_count: int
    compressed_count: int
    key_kinds: tuple[Dsv4StageKeyKind, ...]
    key_global_ids: tuple[int, ...]


class Dsv4SparseForwardResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    out: torch.Tensor
    lse: torch.Tensor


class Dsv4SparseBackwardResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    dq: torch.Tensor
    dkv: torch.Tensor
    d_attn_sink: torch.Tensor


class Dsv4StageForwardRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    materialized_stage: Dsv4MaterializedStage
    out: torch.Tensor
    lse: torch.Tensor


class Dsv4AttentionForwardResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    out: torch.Tensor
    lse: torch.Tensor
    real_out: torch.Tensor
    real_lse: torch.Tensor
    query_token_ids: tuple[int, ...]
    attn_sink: torch.Tensor
    scale: float | None
    stage_records: tuple[Dsv4StageForwardRecord, ...]


class Dsv4StageBackwardRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    materialized_stage: Dsv4MaterializedStage
    dq_stage: torch.Tensor
    dkv_stage: torch.Tensor


class Dsv4AttentionBackwardReplayResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    stage_records: tuple[Dsv4StageBackwardRecord, ...]
    d_attn_sink: torch.Tensor


class Dsv4AttentionGradientResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    query_token_ids: tuple[int, ...]
    raw_token_ids: tuple[int, ...]
    compressed_entry_ids: tuple[int, ...]
    dq: torch.Tensor
    draw_kv: torch.Tensor
    dcompressed_kv: torch.Tensor
    d_attn_sink: torch.Tensor


class Dsv4GradientOwnerBucket(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    owner_rank: int
    query_token_ids: tuple[int, ...]
    raw_token_ids: tuple[int, ...]
    compressed_entry_ids: tuple[int, ...]
    dq: torch.Tensor
    draw_kv: torch.Tensor
    dcompressed_kv: torch.Tensor


class Dsv4CompressedLayout(BaseModel):
    model_config = ConfigDict(frozen=True)

    spec: Dsv4CompressionSpec
    streams: tuple[Dsv4StreamSpec, ...]
    branch_views: tuple[Dsv4BranchView, ...]
    entries: tuple[Dsv4CompressedEntry, ...]
    halo_transfers: tuple[Dsv4HaloTransfer, ...]
    entry_ids_by_owner_rank: tuple[tuple[int, ...], ...]
    raw_token_owner_ranks: tuple[int, ...]
    entry_ids_by_branch_stream: dict[int, tuple[int, ...]] = Field(default_factory=dict)


class Dsv4PreparedPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    csa_layout: Dsv4CompressedLayout | None = None
    hca_layout: Dsv4CompressedLayout | None = None


class Dsv4ContextParallelState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    cp_state: ArtContextParallelState
    dsv4_plan: Dsv4PreparedPlan
    extra: dict[str, Any] = Field(default_factory=dict)


Dsv4TopkResult.model_rebuild()
Dsv4ProjectedTokenBuffer.model_rebuild()
Dsv4TensorExchangePlan.model_rebuild()
Dsv4TensorIdBuffer.model_rebuild()
Dsv4CompressionHaloPayload.model_rebuild()
Dsv4CompressionHaloGradientPayload.model_rebuild()
Dsv4StageInputs.model_rebuild()
Dsv4IndexerStagePlan.model_rebuild()
Dsv4StagePlanGroup.model_rebuild()
Dsv4MaterializedStage.model_rebuild()
Dsv4SparseForwardResult.model_rebuild()
Dsv4SparseBackwardResult.model_rebuild()
Dsv4StageForwardRecord.model_rebuild()
Dsv4AttentionForwardResult.model_rebuild()
Dsv4StageBackwardRecord.model_rebuild()
Dsv4AttentionBackwardReplayResult.model_rebuild()
Dsv4AttentionGradientResult.model_rebuild()
Dsv4GradientOwnerBucket.model_rebuild()
