from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, PrivateAttr
import torch
import torch.distributed as dist

from .comm import Dsv4TensorExchangeWork, launch_dsv4_tensor_exchange
from .types import (
    Dsv4BranchView,
    Dsv4CompressedEntry,
    Dsv4CompressedKvForwardResult,
    Dsv4CompressedKvGradientResult,
    Dsv4CompressedLayout,
    Dsv4CompressionHaloGradientPayload,
    Dsv4CompressionHaloPayload,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    Dsv4HaloTransfer,
    Dsv4ProjectedTokenBuffer,
    Dsv4StreamKind,
    Dsv4StreamSpec,
    Dsv4TensorExchangePlan,
)

_DIST = cast(Any, dist)

if TYPE_CHECKING:
    from art.megatron.context_parallel.types import ArtContextParallelState
else:
    ArtContextParallelState = Any

_PADDING_GROUP_ID = -1


class TokenLayoutIndexLike(Protocol):
    ownership_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    token_counts_by_rank: tuple[int, ...]


class Dsv4CompressionHaloExchangeWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rank: int
    projected_dim: int
    incoming_transfers: tuple[Dsv4HaloTransfer, ...]
    tensor_work: Dsv4TensorExchangeWork

    def wait(self) -> None:
        self.tensor_work.wait()

    def wait_post_process(self) -> tuple[Dsv4CompressionHaloPayload, ...]:
        result = self.tensor_work.wait_post_process()
        expected_ids = tuple(
            int(token_id)
            for transfer in self.incoming_transfers
            for token_id in transfer.token_ids
        )
        if result.ids != expected_ids:
            raise RuntimeError(
                "DSV4 compression halo exchange received unexpected token ids: "
                f"{result.ids} vs {expected_ids}"
            )
        expected_width = int(self.projected_dim) * 2
        if int(result.tensor.shape[-1]) != expected_width:
            raise RuntimeError(
                "DSV4 compression halo fused tensor width mismatch: "
                f"{int(result.tensor.shape[-1])} vs {expected_width}"
            )
        projected_kv, projected_gate = result.tensor.split(
            int(self.projected_dim),
            dim=-1,
        )
        payloads: list[Dsv4CompressionHaloPayload] = []
        cursor = 0
        for transfer in self.incoming_transfers:
            count = len(transfer.token_ids)
            payloads.append(
                Dsv4CompressionHaloPayload(
                    source_rank=int(transfer.source_rank),
                    target_rank=int(self.rank),
                    token_ids=transfer.token_ids,
                    entry_ids=transfer.entry_ids,
                    projected_kv=_narrow_token_dim(projected_kv, cursor, count),
                    projected_gate=_narrow_token_dim(projected_gate, cursor, count),
                )
            )
            cursor += count
        return tuple(payloads)


class Dsv4CompressionHaloGradientExchangeWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rank: int
    projected_dim: int
    incoming_transfers: tuple[Dsv4HaloTransfer, ...]
    tensor_work: Dsv4TensorExchangeWork

    def wait(self) -> None:
        self.tensor_work.wait()

    def wait_post_process(self) -> tuple[Dsv4CompressionHaloGradientPayload, ...]:
        result = self.tensor_work.wait_post_process()
        expected_ids = tuple(
            int(token_id)
            for transfer in self.incoming_transfers
            for token_id in transfer.token_ids
        )
        if result.ids != expected_ids:
            raise RuntimeError(
                "DSV4 compression halo gradient exchange received unexpected "
                f"token ids: {result.ids} vs {expected_ids}"
            )
        expected_width = int(self.projected_dim) * 2
        if int(result.tensor.shape[-1]) != expected_width:
            raise RuntimeError(
                "DSV4 compression halo gradient fused tensor width mismatch: "
                f"{int(result.tensor.shape[-1])} vs {expected_width}"
            )
        dprojected_kv, dprojected_gate = result.tensor.split(
            int(self.projected_dim),
            dim=-1,
        )
        payloads: list[Dsv4CompressionHaloGradientPayload] = []
        cursor = 0
        for transfer in self.incoming_transfers:
            count = len(transfer.token_ids)
            payloads.append(
                Dsv4CompressionHaloGradientPayload(
                    source_rank=int(transfer.target_rank),
                    target_rank=int(self.rank),
                    token_ids=transfer.token_ids,
                    entry_ids=transfer.entry_ids,
                    dprojected_kv=_narrow_token_dim(dprojected_kv, cursor, count),
                    dprojected_gate=_narrow_token_dim(
                        dprojected_gate,
                        cursor,
                        count,
                    ),
                )
            )
            cursor += count
        return tuple(payloads)


class Dsv4CompressedKvForwardWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    layout: Dsv4CompressedLayout
    rank: int
    projected_kv: torch.Tensor
    projected_gate: torch.Tensor
    token_ids: tuple[int, ...]
    positional_bias: torch.Tensor
    halo_work: Dsv4CompressionHaloExchangeWork

    def wait(self) -> None:
        self.halo_work.wait()

    def wait_post_process(self) -> Dsv4CompressedKvForwardResult:
        halo_payloads = self.halo_work.wait_post_process()
        buffer = materialize_dsv4_compression_token_buffer(
            layout=self.layout,
            owner_rank=int(self.rank),
            projected_kv=self.projected_kv,
            projected_gate=self.projected_gate,
            token_ids=self.token_ids,
            halo_payloads=halo_payloads,
        )
        compressed_entry_ids = self.layout.entry_ids_by_owner_rank[int(self.rank)]
        compressed = compress_owned_projected_kv(
            layout=self.layout,
            owner_rank=int(self.rank),
            projected_kv=buffer.projected_kv,
            projected_gate=buffer.projected_gate,
            positional_bias=self.positional_bias,
            token_ids=buffer.token_ids,
        )
        return Dsv4CompressedKvForwardResult(
            layout=self.layout,
            owner_rank=int(self.rank),
            local_token_ids=self.token_ids,
            compressed_entry_ids=compressed_entry_ids,
            token_buffer=buffer,
            positional_bias=self.positional_bias,
            compressed_kv=compressed,
        )


class Dsv4CompressedKvBackwardWork(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rank: int
    local_gradient: Dsv4ProjectedTokenBuffer
    dpositional_bias: torch.Tensor
    halo_gradient_work: Dsv4CompressionHaloGradientExchangeWork
    bias_handle: Any | None = None
    _wait_complete: bool = PrivateAttr(default=False)

    def wait(self) -> None:
        if self._wait_complete:
            return
        self.halo_gradient_work.wait()
        if self.bias_handle is not None:
            self.bias_handle.wait()
        self._wait_complete = True

    def wait_post_process(self) -> Dsv4CompressedKvGradientResult:
        self.wait()
        halo_payloads = self.halo_gradient_work.wait_post_process()
        accumulated = accumulate_dsv4_compression_halo_gradient_payloads(
            target_rank=int(self.rank),
            token_ids=self.local_gradient.token_ids,
            dprojected_kv=self.local_gradient.projected_kv,
            dprojected_gate=self.local_gradient.projected_gate,
            halo_gradient_payloads=halo_payloads,
        )
        return Dsv4CompressedKvGradientResult(
            token_ids=accumulated.token_ids,
            dprojected_kv=accumulated.projected_kv,
            dprojected_gate=accumulated.projected_gate,
            dpositional_bias=self.dpositional_bias,
        )


def build_dsv4_compressed_layout(
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    token_layout_index: TokenLayoutIndexLike,
    spec: Dsv4CompressionSpec,
) -> Dsv4CompressedLayout:
    """Build host-ahead DSV4 compression metadata from shared-prefix layout.

    This function is CPU metadata planning only. It does not inspect activations
    and must not read CUDA tensors in the production lookahead path. Compressed
    entries are deduplicated only when their full dependency is inside the
    shared prefix; closure-token ownership follows the ART CP token layout.
    """
    if group_ids.device.type != "cpu" or parent_ids.device.type != "cpu":
        raise RuntimeError("DSV4 compression planning requires CPU metadata tensors")
    if int(spec.ratio) <= 0:
        raise RuntimeError(f"DSV4 compression ratio must be positive, got {spec.ratio}")
    group_row, parent_row = _validate_metadata(group_ids, parent_ids)
    streams = _build_streams(group_row=group_row, parent_row=parent_row)
    branch_views = _build_branch_views(streams)
    token_owner = _build_token_ownership(token_layout_index)
    return _build_dsv4_compressed_layout_from_parts(
        spec=spec,
        streams=streams,
        branch_views=branch_views,
        token_owner=token_owner,
        token_layout_index=token_layout_index,
    )


def _build_dsv4_compressed_layout_from_parts(
    *,
    spec: Dsv4CompressionSpec,
    streams: tuple[Dsv4StreamSpec, ...],
    branch_views: tuple[Dsv4BranchView, ...],
    token_owner: dict[int, tuple[int, int]],
    token_layout_index: TokenLayoutIndexLike,
) -> Dsv4CompressedLayout:
    entries, entry_ids_by_branch_stream = _build_entries(
        branch_views=branch_views,
        token_owner=token_owner,
        spec=spec,
    )
    return Dsv4CompressedLayout(
        spec=spec,
        streams=streams,
        branch_views=branch_views,
        entries=entries,
        halo_transfers=_build_halo_transfers(entries, token_owner),
        entry_ids_by_owner_rank=_entry_ids_by_owner(entries, token_layout_index),
        raw_token_owner_ranks=_raw_token_owner_ranks(token_owner),
        entry_ids_by_branch_stream=entry_ids_by_branch_stream,
        entry_ids_by_closure_token=_entry_ids_by_closure_token(entries),
        closure_token_ids=_closure_token_ids(entries),
    )


def build_dsv4_compressed_layout_from_cp_state(
    *,
    state: ArtContextParallelState,
    spec: Dsv4CompressionSpec,
) -> Dsv4CompressedLayout:
    return build_dsv4_compressed_layout(
        group_ids=state.group_ids.unsqueeze(0)
        if state.group_ids.ndim == 1
        else state.group_ids,
        parent_ids=state.parent_ids.unsqueeze(0)
        if state.parent_ids.ndim == 1
        else state.parent_ids,
        token_layout_index=state.rank_plan.token_layout_index,
        spec=spec,
    )


def build_dsv4_compressed_layouts_from_cp_state(
    *,
    state: ArtContextParallelState,
    specs: Sequence[Dsv4CompressionSpec],
) -> tuple[Dsv4CompressedLayout, ...]:
    """Build multiple layouts while sharing stream and branch-view planning."""
    if not specs:
        return ()
    group_ids = (
        state.group_ids.unsqueeze(0) if state.group_ids.ndim == 1 else state.group_ids
    )
    parent_ids = (
        state.parent_ids.unsqueeze(0)
        if state.parent_ids.ndim == 1
        else state.parent_ids
    )
    if group_ids.device.type != "cpu" or parent_ids.device.type != "cpu":
        raise RuntimeError("DSV4 compression planning requires CPU metadata tensors")
    for spec in specs:
        if int(spec.ratio) <= 0:
            raise RuntimeError(
                f"DSV4 compression ratio must be positive, got {spec.ratio}"
            )
    group_row, parent_row = _validate_metadata(group_ids, parent_ids)
    streams = _build_streams(group_row=group_row, parent_row=parent_row)
    branch_views = _build_branch_views(streams)
    token_owner = _build_token_ownership(state.rank_plan.token_layout_index)
    return tuple(
        _build_dsv4_compressed_layout_from_parts(
            spec=spec,
            streams=streams,
            branch_views=branch_views,
            token_owner=token_owner,
            token_layout_index=state.rank_plan.token_layout_index,
        )
        for spec in specs
    )


def position_in_query_view(
    *,
    branch_view: Dsv4BranchView,
    candidate_token_id: int,
) -> int | None:
    return branch_view.position_of_token(candidate_token_id)


def compress_projected_kv(
    *,
    layout: Dsv4CompressedLayout,
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    positional_bias: torch.Tensor,
    entry_ids: Sequence[int] | None = None,
    token_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Compress projected DSV4 KV/gate tensors according to a branch layout.

    `projected_kv` and `projected_gate` are the outputs of the DSV4 compressor
    projections, before RMSNorm/RoPE. For HCA their last dim is `D`; for CSA it
    is `2 * D` and the A/B halves are selected according to the Miles overlap
    transform. `token_ids` maps rows in the projected tensors to packed token ids
    when the rank owns a compact local+halo buffer; if omitted, tensor row `t`
    is packed token `t`.
    """
    if projected_kv.shape != projected_gate.shape:
        raise RuntimeError(
            "DSV4 projected KV and gate tensors must share shape, got "
            f"{tuple(projected_kv.shape)} vs {tuple(projected_gate.shape)}"
        )
    if projected_kv.device != projected_gate.device:
        raise RuntimeError(
            "DSV4 projected KV and gate tensors must share device, got "
            f"{projected_kv.device} vs {projected_gate.device}"
        )
    if projected_kv.ndim not in (2, 3):
        raise RuntimeError(
            "DSV4 projected tensors must have shape [T, C] or [B, T, C], got "
            f"{tuple(projected_kv.shape)}"
        )
    if positional_bias.ndim != 2:
        raise RuntimeError(
            "DSV4 compressor positional bias must have shape [ratio, C], got "
            f"{tuple(positional_bias.shape)}"
        )
    if int(positional_bias.shape[0]) != int(layout.spec.ratio):
        raise RuntimeError(
            "DSV4 positional-bias ratio mismatch: "
            f"layout={layout.spec.ratio}, bias={int(positional_bias.shape[0])}"
        )

    selected_entries = _select_entries(layout=layout, entry_ids=entry_ids)
    head_dim = _compressed_head_dim(
        layout=layout,
        projected_dim=int(projected_kv.shape[-1]),
        positional_bias=positional_bias,
    )
    if not selected_entries:
        return _empty_compressed_output(projected_kv, head_dim)

    gather = _build_projected_compression_gather(
        entries=selected_entries,
        layout=layout,
        token_ids=token_ids,
        tensor_token_count=int(projected_kv.shape[-2]),
    )
    token_index = gather["token_index"].to(projected_kv.device)
    valid = gather["valid"].to(projected_kv.device)
    ape_row = gather["ape_row"].to(projected_kv.device)
    half = gather["half"].to(projected_kv.device)
    positional_bias = positional_bias.to(device=projected_gate.device)

    gathered_kv = _gather_tokens(projected_kv, token_index)
    gathered_gate = _gather_tokens(projected_gate, token_index)
    if layout.spec.kind == Dsv4CompressionKind.CSA:
        gathered_kv = _select_csa_halves(gathered_kv, half, head_dim)
        gathered_gate = _select_csa_halves(gathered_gate, half, head_dim)
        bias = _select_csa_positional_bias(positional_bias, ape_row, half, head_dim)
    elif layout.spec.kind == Dsv4CompressionKind.HCA:
        bias = positional_bias.index_select(0, ape_row.reshape(-1)).reshape(
            *ape_row.shape,
            head_dim,
        )
    else:
        raise RuntimeError(f"Unsupported DSV4 compression kind: {layout.spec.kind}")

    bias = bias.to(device=projected_gate.device, dtype=torch.float32)
    score = gathered_gate.float() + bias
    score = score.masked_fill(~valid.to(score.device).unsqueeze(-1), float("-inf"))
    weights = torch.softmax(score, dim=-2)
    weights = torch.where(
        valid.to(weights.device).unsqueeze(-1),
        weights,
        torch.zeros((), dtype=weights.dtype, device=weights.device),
    )
    return (gathered_kv * weights).sum(dim=-2).to(dtype=projected_kv.dtype)


def compress_owned_projected_kv(
    *,
    layout: Dsv4CompressedLayout,
    owner_rank: int,
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    positional_bias: torch.Tensor,
    token_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    return compress_projected_kv(
        layout=layout,
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        positional_bias=positional_bias,
        entry_ids=layout.entry_ids_by_owner_rank[int(owner_rank)],
        token_ids=token_ids,
    )


@torch.compiler.disable
def launch_dsv4_compressed_kv_forward(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    positional_bias: torch.Tensor,
    token_ids: Sequence[int],
    group: Any,
    async_op: bool,
) -> Dsv4CompressedKvForwardWork:
    """Launch DSV4 compressed-KV production with projected-token halo exchange.

    This is the production compression boundary the model handler can call after
    local compressor projections. It starts the fused projected KV/gate halo
    exchange immediately; `wait_post_process()` materializes the local+halo
    token buffer and computes this rank's owned compressed entries.
    """
    _validate_projected_pair(projected_kv=projected_kv, projected_gate=projected_gate)
    rank_int = _validate_layout_rank_value(
        rank=rank,
        rank_count=len(layout.entry_ids_by_owner_rank),
    )
    token_ids = _normalize_token_ids(
        token_ids=token_ids,
        tensor_token_count=int(projected_kv.shape[-2]),
        name="token_ids",
    )
    _validate_positional_bias(
        layout=layout,
        projected_kv=projected_kv,
        positional_bias=positional_bias,
    )
    halo_work = launch_dsv4_compression_halo_exchange(
        layout=layout,
        rank=rank_int,
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        token_ids=token_ids,
        group=group,
        async_op=async_op,
    )
    return Dsv4CompressedKvForwardWork(
        layout=layout,
        rank=rank_int,
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        token_ids=token_ids,
        positional_bias=positional_bias,
        halo_work=halo_work,
    )


@torch.compiler.disable
def launch_dsv4_compressed_kv_backward(
    *,
    forward_result: Dsv4CompressedKvForwardResult,
    dcompressed_kv: torch.Tensor,
    group: Any,
    async_op: bool,
) -> Dsv4CompressedKvBackwardWork:
    """Replay compressor backward and launch reverse halo-gradient exchange.

    The compressed-KV forward path can be used outside normal autograd because
    DSV4 sparse attention backward is replay-based. This function recomputes the
    small compression operation over the saved local+halo token buffer, scatters
    owner-local projected-token gradients, returns remote halo gradients, and
    CP-reduces the positional-bias gradient.
    """
    _validate_dcompressed_kv(
        forward_result=forward_result,
        dcompressed_kv=dcompressed_kv,
    )
    buffer = forward_result.token_buffer
    if len(forward_result.compressed_entry_ids) == 0:
        d_buffer_kv = torch.zeros_like(buffer.projected_kv)
        d_buffer_gate = torch.zeros_like(buffer.projected_gate)
        dpositional_bias = torch.zeros_like(forward_result.positional_bias)
    else:
        replay_kv = buffer.projected_kv.detach().requires_grad_(True)
        replay_gate = buffer.projected_gate.detach().requires_grad_(True)
        replay_bias = forward_result.positional_bias.detach().requires_grad_(True)
        replay_compressed = compress_owned_projected_kv(
            layout=forward_result.layout,
            owner_rank=int(forward_result.owner_rank),
            projected_kv=replay_kv,
            projected_gate=replay_gate,
            positional_bias=replay_bias,
            token_ids=buffer.token_ids,
        )
        d_buffer_kv, d_buffer_gate, dpositional_bias = torch.autograd.grad(
            replay_compressed,
            (replay_kv, replay_gate, replay_bias),
            grad_outputs=dcompressed_kv,
        )
    local_gradient = _scatter_buffer_gradient_to_local_tokens(
        local_token_ids=forward_result.local_token_ids,
        buffer_token_ids=buffer.token_ids,
        dprojected_kv=d_buffer_kv,
        dprojected_gate=d_buffer_gate,
    )
    halo_gradient_work = launch_dsv4_compression_halo_gradient_exchange(
        layout=forward_result.layout,
        rank=int(forward_result.owner_rank),
        token_ids=buffer.token_ids,
        dprojected_kv=d_buffer_kv,
        dprojected_gate=d_buffer_gate,
        group=group,
        async_op=async_op,
    )
    reduced_bias = dpositional_bias.contiguous().clone()
    bias_handle = None
    if len(forward_result.layout.entry_ids_by_owner_rank) > 1:
        bias_handle = _DIST.all_reduce(reduced_bias, group=group, async_op=async_op)
    return Dsv4CompressedKvBackwardWork(
        rank=int(forward_result.owner_rank),
        local_gradient=local_gradient,
        dpositional_bias=reduced_bias,
        halo_gradient_work=halo_gradient_work,
        bias_handle=bias_handle,
    )


def pack_dsv4_compression_halo_payloads(
    *,
    layout: Dsv4CompressedLayout,
    source_rank: int,
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    token_ids: Sequence[int],
) -> tuple[Dsv4CompressionHaloPayload, ...]:
    """Pack projected compressor tensors needed by remote closure owners.

    The returned payloads are the stable inputs for an eager DSV4 halo
    communication path. Future NCCL/P2P code should launch outside compiled
    regions and make producer-to-comm, comm-to-consumer, and tensor lifetime
    ordering explicit.
    """
    _validate_projected_pair(projected_kv=projected_kv, projected_gate=projected_gate)
    token_ids = _normalize_token_ids(
        token_ids=token_ids,
        tensor_token_count=int(projected_kv.shape[-2]),
        name="token_ids",
    )
    payloads: list[Dsv4CompressionHaloPayload] = []
    for transfer in layout.halo_transfers:
        if int(transfer.source_rank) != int(source_rank):
            continue
        positions = _positions_for_token_ids(
            available_token_ids=token_ids,
            requested_token_ids=transfer.token_ids,
            name="halo source token_ids",
        )
        payloads.append(
            Dsv4CompressionHaloPayload(
                source_rank=int(source_rank),
                target_rank=int(transfer.target_rank),
                token_ids=transfer.token_ids,
                entry_ids=transfer.entry_ids,
                projected_kv=_index_select_token_dim(projected_kv, positions),
                projected_gate=_index_select_token_dim(projected_gate, positions),
            )
        )
    return tuple(payloads)


@torch.compiler.disable
def launch_dsv4_compression_halo_exchange(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    token_ids: Sequence[int],
    group: Any,
    async_op: bool,
) -> Dsv4CompressionHaloExchangeWork:
    """Launch fused projected-KV/gate halo exchange for DSV4 compression.

    This is a DSV4-specific eager communication boundary. It fuses KV and gate
    rows into one explicit-id exchange so CSA/HCA closure owners receive exactly
    the remote projected rows needed for `materialize_dsv4_compression_token_buffer`.
    Callers must await the returned work before using payload tensors; otherwise
    async CUDA communication can race the compression consumer.
    """
    _validate_projected_pair(projected_kv=projected_kv, projected_gate=projected_gate)
    token_ids = _normalize_token_ids(
        token_ids=token_ids,
        tensor_token_count=int(projected_kv.shape[-2]),
        name="token_ids",
    )
    projected_dim = int(projected_kv.shape[-1])
    fused = torch.cat((projected_kv, projected_gate), dim=-1)
    plan = _compression_halo_exchange_plan(layout=layout, rank=rank)
    return Dsv4CompressionHaloExchangeWork(
        rank=int(rank),
        projected_dim=projected_dim,
        incoming_transfers=_incoming_halo_transfers(layout=layout, rank=rank),
        tensor_work=launch_dsv4_tensor_exchange(
            tensor=fused,
            tensor_ids=token_ids,
            plan=plan,
            group=group,
            async_op=async_op,
            label="dsv4_compression_halo_exchange",
        ),
    )


def materialize_dsv4_compression_token_buffer(
    *,
    layout: Dsv4CompressedLayout,
    owner_rank: int,
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    token_ids: Sequence[int],
    halo_payloads: Sequence[Dsv4CompressionHaloPayload] = (),
) -> Dsv4ProjectedTokenBuffer:
    """Build the compact local+halo token buffer for owned compression entries.

    Compression ownership is by closure token. This helper consumes local
    projected compressor tensors plus received halo payloads and returns exactly
    the packed-token rows required by `compress_owned_projected_kv`.
    """
    _validate_projected_pair(projected_kv=projected_kv, projected_gate=projected_gate)
    token_ids = _normalize_token_ids(
        token_ids=token_ids,
        tensor_token_count=int(projected_kv.shape[-2]),
        name="token_ids",
    )
    needed_token_ids = _needed_tokens_for_owner(layout=layout, owner_rank=owner_rank)
    all_kv = [projected_kv]
    all_gate = [projected_gate]
    all_token_ids = list(token_ids)
    for payload in halo_payloads:
        if int(payload.target_rank) != int(owner_rank):
            raise RuntimeError(
                "DSV4 compression halo payload target rank mismatch: "
                f"payload={payload.target_rank}, owner={owner_rank}"
            )
        _validate_payload_matches_plan(
            layout=layout,
            source_rank=int(payload.source_rank),
            target_rank=int(owner_rank),
            token_ids=payload.token_ids,
            entry_ids=payload.entry_ids,
        )
        _validate_halo_payload_tensors(payload)
        _validate_payload_compatible(
            reference=projected_kv,
            payload_tensor=payload.projected_kv,
            name="projected_kv",
        )
        _validate_payload_compatible(
            reference=projected_gate,
            payload_tensor=payload.projected_gate,
            name="projected_gate",
        )
        all_kv.append(payload.projected_kv)
        all_gate.append(payload.projected_gate)
        all_token_ids.extend(int(token_id) for token_id in payload.token_ids)
    _row_by_token_id(all_token_ids, name="local+halo token_ids")

    if not needed_token_ids:
        return Dsv4ProjectedTokenBuffer(
            token_ids=(),
            projected_kv=_empty_token_rows_like(projected_kv),
            projected_gate=_empty_token_rows_like(projected_gate),
        )

    merged_kv = torch.cat(all_kv, dim=_token_dim(projected_kv))
    merged_gate = torch.cat(all_gate, dim=_token_dim(projected_gate))
    positions = _positions_for_token_ids(
        available_token_ids=tuple(all_token_ids),
        requested_token_ids=needed_token_ids,
        name="local+halo token_ids",
    )
    return Dsv4ProjectedTokenBuffer(
        token_ids=needed_token_ids,
        projected_kv=_index_select_token_dim(merged_kv, positions),
        projected_gate=_index_select_token_dim(merged_gate, positions),
    )


def pack_dsv4_compression_halo_gradient_payloads(
    *,
    layout: Dsv4CompressedLayout,
    owner_rank: int,
    token_ids: Sequence[int],
    dprojected_kv: torch.Tensor,
    dprojected_gate: torch.Tensor,
) -> tuple[Dsv4CompressionHaloGradientPayload, ...]:
    """Pack gradients for remote projected tokens imported by this owner.

    This is the reverse payload of `pack_dsv4_compression_halo_payloads`: the
    compression owner sends gradients back to the raw-token owner. Actual
    communication should stay eager with explicit stream and lifetime ordering.
    """
    _validate_projected_pair(
        projected_kv=dprojected_kv,
        projected_gate=dprojected_gate,
    )
    token_ids = _normalize_token_ids(
        token_ids=token_ids,
        tensor_token_count=int(dprojected_kv.shape[-2]),
        name="token_ids",
    )
    payloads: list[Dsv4CompressionHaloGradientPayload] = []
    for transfer in layout.halo_transfers:
        if int(transfer.target_rank) != int(owner_rank):
            continue
        positions = _positions_for_token_ids(
            available_token_ids=token_ids,
            requested_token_ids=transfer.token_ids,
            name="owner buffer token_ids",
        )
        payloads.append(
            Dsv4CompressionHaloGradientPayload(
                source_rank=int(owner_rank),
                target_rank=int(transfer.source_rank),
                token_ids=transfer.token_ids,
                entry_ids=transfer.entry_ids,
                dprojected_kv=_index_select_token_dim(dprojected_kv, positions),
                dprojected_gate=_index_select_token_dim(dprojected_gate, positions),
            )
        )
    return tuple(payloads)


@torch.compiler.disable
def launch_dsv4_compression_halo_gradient_exchange(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
    token_ids: Sequence[int],
    dprojected_kv: torch.Tensor,
    dprojected_gate: torch.Tensor,
    group: Any,
    async_op: bool,
) -> Dsv4CompressionHaloGradientExchangeWork:
    """Launch fused reverse halo-gradient exchange for DSV4 compression.

    Compression owners call this with gradients for their local+halo token
    buffers. Raw-token owners receive projected KV/gate gradients for imported
    boundary rows and can pass the returned payloads to
    `accumulate_dsv4_compression_halo_gradient_payloads`.
    """
    _validate_projected_pair(
        projected_kv=dprojected_kv,
        projected_gate=dprojected_gate,
    )
    token_ids = _normalize_token_ids(
        token_ids=token_ids,
        tensor_token_count=int(dprojected_kv.shape[-2]),
        name="token_ids",
    )
    projected_dim = int(dprojected_kv.shape[-1])
    fused = torch.cat((dprojected_kv, dprojected_gate), dim=-1)
    plan = _compression_halo_gradient_exchange_plan(layout=layout, rank=rank)
    return Dsv4CompressionHaloGradientExchangeWork(
        rank=int(rank),
        projected_dim=projected_dim,
        incoming_transfers=_incoming_halo_gradient_transfers(layout=layout, rank=rank),
        tensor_work=launch_dsv4_tensor_exchange(
            tensor=fused,
            tensor_ids=token_ids,
            plan=plan,
            group=group,
            async_op=async_op,
            label="dsv4_compression_halo_gradient_exchange",
            allow_duplicate_recv_ids=True,
        ),
    )


def accumulate_dsv4_compression_halo_gradient_payloads(
    *,
    target_rank: int,
    token_ids: Sequence[int],
    dprojected_kv: torch.Tensor,
    dprojected_gate: torch.Tensor,
    halo_gradient_payloads: Sequence[Dsv4CompressionHaloGradientPayload],
) -> Dsv4ProjectedTokenBuffer:
    """Add returned halo gradients into this raw-token owner's grad space."""
    _validate_projected_pair(
        projected_kv=dprojected_kv,
        projected_gate=dprojected_gate,
    )
    token_ids = _normalize_token_ids(
        token_ids=token_ids,
        tensor_token_count=int(dprojected_kv.shape[-2]),
        name="token_ids",
    )
    kv = dprojected_kv.clone()
    gate = dprojected_gate.clone()
    token_index = _row_by_token_id(token_ids, name="token_ids")
    for payload in halo_gradient_payloads:
        if int(payload.target_rank) != int(target_rank):
            raise RuntimeError(
                "DSV4 compression halo gradient target rank mismatch: "
                f"payload={payload.target_rank}, target={target_rank}"
            )
        _validate_halo_gradient_payload_tensors(payload)
        _validate_payload_compatible(
            reference=kv,
            payload_tensor=payload.dprojected_kv,
            name="dprojected_kv",
        )
        _validate_payload_compatible(
            reference=gate,
            payload_tensor=payload.dprojected_gate,
            name="dprojected_gate",
        )
        positions = _positions_for_token_ids(
            available_token_ids=token_ids,
            requested_token_ids=payload.token_ids,
            name="gradient target token_ids",
            precomputed_index=token_index,
        )
        _index_add_token_dim(kv, positions, payload.dprojected_kv)
        _index_add_token_dim(gate, positions, payload.dprojected_gate)
    return Dsv4ProjectedTokenBuffer(
        token_ids=token_ids,
        projected_kv=kv,
        projected_gate=gate,
    )


def _validate_projected_pair(
    *,
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
) -> None:
    if projected_kv.shape != projected_gate.shape:
        raise RuntimeError(
            "DSV4 projected KV and gate tensors must share shape, got "
            f"{tuple(projected_kv.shape)} vs {tuple(projected_gate.shape)}"
        )
    if projected_kv.device != projected_gate.device:
        raise RuntimeError(
            "DSV4 projected KV and gate tensors must share device, got "
            f"{projected_kv.device} vs {projected_gate.device}"
        )
    if projected_kv.ndim not in (2, 3):
        raise RuntimeError(
            "DSV4 projected tensors must have shape [T, C] or [B, T, C], got "
            f"{tuple(projected_kv.shape)}"
        )


def _validate_positional_bias(
    *,
    layout: Dsv4CompressedLayout,
    projected_kv: torch.Tensor,
    positional_bias: torch.Tensor,
) -> None:
    if positional_bias.ndim != 2:
        raise RuntimeError(
            "DSV4 compressor positional bias must have shape [ratio, C], got "
            f"{tuple(positional_bias.shape)}"
        )
    if int(positional_bias.shape[0]) != int(layout.spec.ratio):
        raise RuntimeError(
            "DSV4 positional-bias ratio mismatch: "
            f"layout={layout.spec.ratio}, bias={int(positional_bias.shape[0])}"
        )
    _compressed_head_dim(
        layout=layout,
        projected_dim=int(projected_kv.shape[-1]),
        positional_bias=positional_bias,
    )


def _validate_dcompressed_kv(
    *,
    forward_result: Dsv4CompressedKvForwardResult,
    dcompressed_kv: torch.Tensor,
) -> None:
    if tuple(dcompressed_kv.shape) != tuple(forward_result.compressed_kv.shape):
        raise RuntimeError(
            "DSV4 compressed-KV grad shape mismatch: "
            f"{tuple(dcompressed_kv.shape)} vs "
            f"{tuple(forward_result.compressed_kv.shape)}"
        )
    if dcompressed_kv.device != forward_result.compressed_kv.device:
        raise RuntimeError(
            "DSV4 compressed-KV grad device mismatch: "
            f"{dcompressed_kv.device} vs {forward_result.compressed_kv.device}"
        )


def _scatter_buffer_gradient_to_local_tokens(
    *,
    local_token_ids: Sequence[int],
    buffer_token_ids: Sequence[int],
    dprojected_kv: torch.Tensor,
    dprojected_gate: torch.Tensor,
) -> Dsv4ProjectedTokenBuffer:
    _validate_projected_pair(
        projected_kv=dprojected_kv,
        projected_gate=dprojected_gate,
    )
    buffer_token_ids = _normalize_token_ids(
        token_ids=buffer_token_ids,
        tensor_token_count=int(dprojected_kv.shape[-2]),
        name="compression buffer token_ids",
    )
    local_token_ids = tuple(int(token_id) for token_id in local_token_ids)
    local_index = _row_by_token_id(local_token_ids, name="local token_ids")

    local_shape = list(dprojected_kv.shape)
    local_shape[_token_dim(dprojected_kv)] = len(local_token_ids)
    local_kv = dprojected_kv.new_zeros(tuple(local_shape))
    local_gate = dprojected_gate.new_zeros(tuple(local_shape))

    source_positions: list[int] = []
    target_positions: list[int] = []
    for source_position, token_id in enumerate(buffer_token_ids):
        target_position = local_index.get(int(token_id))
        if target_position is None:
            continue
        source_positions.append(source_position)
        target_positions.append(target_position)
    if source_positions:
        _index_add_token_dim(
            local_kv,
            target_positions,
            _index_select_token_dim(dprojected_kv, source_positions),
        )
        _index_add_token_dim(
            local_gate,
            target_positions,
            _index_select_token_dim(dprojected_gate, source_positions),
        )
    return Dsv4ProjectedTokenBuffer(
        token_ids=local_token_ids,
        projected_kv=local_kv,
        projected_gate=local_gate,
    )


def _normalize_token_ids(
    *,
    token_ids: Sequence[int],
    tensor_token_count: int,
    name: str,
) -> tuple[int, ...]:
    token_ids = tuple(int(token_id) for token_id in token_ids)
    if len(token_ids) != int(tensor_token_count):
        raise RuntimeError(
            f"DSV4 {name} length must match projected tensor token count, got "
            f"{len(token_ids)} vs {tensor_token_count}"
        )
    _row_by_token_id(token_ids, name=name)
    return token_ids


def _needed_tokens_for_owner(
    *,
    layout: Dsv4CompressedLayout,
    owner_rank: int,
) -> tuple[int, ...]:
    if int(owner_rank) < 0 or int(owner_rank) >= len(layout.entry_ids_by_owner_rank):
        raise RuntimeError(f"DSV4 owner rank {owner_rank} is outside layout rank count")
    return tuple(
        sorted(
            {
                int(token_id)
                for entry_id in layout.entry_ids_by_owner_rank[int(owner_rank)]
                for token_id in layout.entries[int(entry_id)].dependency_token_ids
            }
        )
    )


def _compression_halo_exchange_plan(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
) -> Dsv4TensorExchangePlan:
    rank = int(rank)
    rank_count = _layout_rank_count(layout)
    _validate_layout_rank(rank=rank, rank_count=rank_count)
    send_ids_by_peer: list[tuple[int, ...]] = [() for _ in range(rank_count)]
    recv_ids_by_peer: list[tuple[int, ...]] = [() for _ in range(rank_count)]
    for transfer in layout.halo_transfers:
        source_rank = int(transfer.source_rank)
        target_rank = int(transfer.target_rank)
        _validate_layout_rank(rank=source_rank, rank_count=rank_count)
        _validate_layout_rank(rank=target_rank, rank_count=rank_count)
        if source_rank == rank:
            if send_ids_by_peer[target_rank]:
                raise RuntimeError(
                    "DSV4 compression halo has duplicate source->target transfer: "
                    f"{source_rank}->{target_rank}"
                )
            send_ids_by_peer[target_rank] = transfer.token_ids
        if target_rank == rank:
            if recv_ids_by_peer[source_rank]:
                raise RuntimeError(
                    "DSV4 compression halo has duplicate source->target transfer: "
                    f"{source_rank}->{target_rank}"
                )
            recv_ids_by_peer[source_rank] = transfer.token_ids
    return Dsv4TensorExchangePlan(
        send_ids_by_peer=tuple(send_ids_by_peer),
        recv_ids_by_peer=tuple(recv_ids_by_peer),
    )


def _incoming_halo_transfers(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
) -> tuple[Dsv4HaloTransfer, ...]:
    rank = int(rank)
    rank_count = _layout_rank_count(layout)
    _validate_layout_rank(rank=rank, rank_count=rank_count)
    by_source: dict[int, Dsv4HaloTransfer] = {}
    for transfer in layout.halo_transfers:
        if int(transfer.target_rank) != rank:
            continue
        source_rank = int(transfer.source_rank)
        if source_rank in by_source:
            raise RuntimeError(
                "DSV4 compression halo has duplicate source->target transfer: "
                f"{source_rank}->{rank}"
            )
        by_source[source_rank] = transfer
    return tuple(
        by_source[source_rank]
        for source_rank in range(rank_count)
        if source_rank in by_source
    )


def _compression_halo_gradient_exchange_plan(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
) -> Dsv4TensorExchangePlan:
    rank = int(rank)
    rank_count = _layout_rank_count(layout)
    _validate_layout_rank(rank=rank, rank_count=rank_count)
    send_ids_by_peer: list[tuple[int, ...]] = [() for _ in range(rank_count)]
    recv_ids_by_peer: list[tuple[int, ...]] = [() for _ in range(rank_count)]
    for transfer in layout.halo_transfers:
        source_rank = int(transfer.source_rank)
        target_rank = int(transfer.target_rank)
        _validate_layout_rank(rank=source_rank, rank_count=rank_count)
        _validate_layout_rank(rank=target_rank, rank_count=rank_count)
        if target_rank == rank:
            if send_ids_by_peer[source_rank]:
                raise RuntimeError(
                    "DSV4 compression halo has duplicate source->target transfer: "
                    f"{source_rank}->{target_rank}"
                )
            send_ids_by_peer[source_rank] = transfer.token_ids
        if source_rank == rank:
            if recv_ids_by_peer[target_rank]:
                raise RuntimeError(
                    "DSV4 compression halo has duplicate source->target transfer: "
                    f"{source_rank}->{target_rank}"
                )
            recv_ids_by_peer[target_rank] = transfer.token_ids
    return Dsv4TensorExchangePlan(
        send_ids_by_peer=tuple(send_ids_by_peer),
        recv_ids_by_peer=tuple(recv_ids_by_peer),
    )


def _incoming_halo_gradient_transfers(
    *,
    layout: Dsv4CompressedLayout,
    rank: int,
) -> tuple[Dsv4HaloTransfer, ...]:
    rank = int(rank)
    rank_count = _layout_rank_count(layout)
    _validate_layout_rank(rank=rank, rank_count=rank_count)
    by_source: dict[int, Dsv4HaloTransfer] = {}
    for transfer in layout.halo_transfers:
        if int(transfer.source_rank) != rank:
            continue
        source_rank = int(transfer.target_rank)
        if source_rank in by_source:
            raise RuntimeError(
                "DSV4 compression halo has duplicate source->target transfer: "
                f"{rank}->{source_rank}"
            )
        by_source[source_rank] = transfer
    return tuple(
        by_source[source_rank]
        for source_rank in range(rank_count)
        if source_rank in by_source
    )


def _layout_rank_count(layout: Dsv4CompressedLayout) -> int:
    return len(layout.entry_ids_by_owner_rank)


def _validate_layout_rank(*, rank: int, rank_count: int) -> None:
    if int(rank) < 0 or int(rank) >= int(rank_count):
        raise RuntimeError(
            f"DSV4 compression halo rank {rank} is outside rank count {rank_count}"
        )


def _validate_layout_rank_value(*, rank: int, rank_count: int) -> int:
    _validate_layout_rank(rank=int(rank), rank_count=int(rank_count))
    return int(rank)


def _validate_halo_payload_tensors(payload: Dsv4CompressionHaloPayload) -> None:
    _validate_projected_pair(
        projected_kv=payload.projected_kv,
        projected_gate=payload.projected_gate,
    )
    _normalize_token_ids(
        token_ids=payload.token_ids,
        tensor_token_count=int(payload.projected_kv.shape[-2]),
        name="halo payload token_ids",
    )
    _row_by_token_id(payload.entry_ids, name="halo payload entry_ids")


def _validate_halo_gradient_payload_tensors(
    payload: Dsv4CompressionHaloGradientPayload,
) -> None:
    _validate_projected_pair(
        projected_kv=payload.dprojected_kv,
        projected_gate=payload.dprojected_gate,
    )
    _normalize_token_ids(
        token_ids=payload.token_ids,
        tensor_token_count=int(payload.dprojected_kv.shape[-2]),
        name="halo gradient payload token_ids",
    )
    _row_by_token_id(payload.entry_ids, name="halo gradient payload entry_ids")


def _validate_payload_matches_plan(
    *,
    layout: Dsv4CompressedLayout,
    source_rank: int,
    target_rank: int,
    token_ids: Sequence[int],
    entry_ids: Sequence[int],
) -> None:
    transfer = next(
        (
            transfer
            for transfer in layout.halo_transfers
            if int(transfer.source_rank) == int(source_rank)
            and int(transfer.target_rank) == int(target_rank)
        ),
        None,
    )
    if transfer is None:
        raise RuntimeError(
            "DSV4 compression halo payload has no planned transfer: "
            f"source={source_rank}, target={target_rank}"
        )
    if tuple(int(token_id) for token_id in token_ids) != transfer.token_ids:
        raise RuntimeError(
            "DSV4 compression halo payload token ids do not match planned transfer"
        )
    if tuple(int(entry_id) for entry_id in entry_ids) != transfer.entry_ids:
        raise RuntimeError(
            "DSV4 compression halo payload entry ids do not match planned transfer"
        )


def _validate_payload_compatible(
    *,
    reference: torch.Tensor,
    payload_tensor: torch.Tensor,
    name: str,
) -> None:
    if payload_tensor.ndim != reference.ndim:
        raise RuntimeError(
            f"DSV4 {name} rank mismatch: {payload_tensor.ndim} vs {reference.ndim}"
        )
    if payload_tensor.device != reference.device:
        raise RuntimeError(
            f"DSV4 {name} device mismatch: {payload_tensor.device} vs {reference.device}"
        )
    if payload_tensor.dtype != reference.dtype:
        raise RuntimeError(
            f"DSV4 {name} dtype mismatch: {payload_tensor.dtype} vs {reference.dtype}"
        )
    token_dim = _token_dim(reference)
    reference_outer = tuple(
        size for dim, size in enumerate(reference.shape) if dim != token_dim
    )
    payload_outer = tuple(
        size for dim, size in enumerate(payload_tensor.shape) if dim != token_dim
    )
    if payload_outer != reference_outer:
        raise RuntimeError(
            f"DSV4 {name} non-token shape mismatch: {payload_outer} vs {reference_outer}"
        )


def _row_by_token_id(ids: Sequence[int], name: str) -> dict[int, int]:
    row_by_id: dict[int, int] = {}
    for row, token_id in enumerate(ids):
        token = int(token_id)
        if token in row_by_id:
            raise RuntimeError(f"DSV4 {name} contains duplicate id {token}")
        row_by_id[token] = row
    return row_by_id


def _positions_for_token_ids(
    *,
    available_token_ids: Sequence[int],
    requested_token_ids: Sequence[int],
    name: str,
    precomputed_index: dict[int, int] | None = None,
) -> tuple[int, ...]:
    _row_by_token_id(requested_token_ids, name=f"requested {name}")
    token_index = (
        _row_by_token_id(available_token_ids, name=name)
        if precomputed_index is None
        else precomputed_index
    )
    missing = tuple(
        int(token_id)
        for token_id in requested_token_ids
        if int(token_id) not in token_index
    )
    if missing:
        raise RuntimeError(f"DSV4 requested tokens missing from {name}: {missing}")
    return tuple(token_index[int(token_id)] for token_id in requested_token_ids)


def _token_dim(tensor: torch.Tensor) -> int:
    return 0 if tensor.ndim == 2 else 1


def _narrow_token_dim(
    tensor: torch.Tensor,
    start: int,
    length: int,
) -> torch.Tensor:
    return tensor.narrow(_token_dim(tensor), int(start), int(length))


def _index_select_token_dim(
    tensor: torch.Tensor,
    positions: Sequence[int],
) -> torch.Tensor:
    indices = torch.tensor(
        tuple(int(position) for position in positions),
        device=tensor.device,
        dtype=torch.long,
    )
    return tensor.index_select(_token_dim(tensor), indices)


def _index_add_token_dim(
    target: torch.Tensor,
    positions: Sequence[int],
    source: torch.Tensor,
) -> None:
    indices = torch.tensor(
        tuple(int(position) for position in positions),
        device=target.device,
        dtype=torch.long,
    )
    target.index_add_(_token_dim(target), indices, source)


def _empty_token_rows_like(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.new_empty((0, int(tensor.shape[-1])))
    return tensor.new_empty((int(tensor.shape[0]), 0, int(tensor.shape[-1])))


def _validate_metadata(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(group_ids.shape) != tuple(parent_ids.shape):
        raise RuntimeError(
            "DSV4 group_ids and parent_ids must share shape, got "
            f"{tuple(group_ids.shape)} vs {tuple(parent_ids.shape)}"
        )
    if group_ids.ndim != 2:
        raise RuntimeError(
            f"DSV4 shared-prefix metadata must be rank-2, got {group_ids.ndim}"
        )
    if int(group_ids.shape[0]) != 1:
        raise RuntimeError(
            "DSV4 CP currently supports one packed row, got "
            f"batch={int(group_ids.shape[0])}"
        )
    valid_count = _valid_token_count(group_ids[0])
    return (
        group_ids[0, :valid_count].contiguous(),
        parent_ids[0, :valid_count].contiguous(),
    )


def _valid_token_count(group_row: torch.Tensor) -> int:
    valid = group_row != _PADDING_GROUP_ID
    count = int(valid.sum().item())
    if count == 0:
        return 0
    if not bool(valid[:count].all().item()):
        raise RuntimeError("DSV4 shared-prefix padding must be a contiguous tail")
    return count


def _build_streams(
    *,
    group_row: torch.Tensor,
    parent_row: torch.Tensor,
) -> tuple[Dsv4StreamSpec, ...]:
    streams: list[Dsv4StreamSpec] = []
    for start, end, group_id, parent_id in _scan_runs(group_row, parent_row):
        if group_id == _PADDING_GROUP_ID:
            continue
        kind = (
            Dsv4StreamKind.PREFIX
            if _is_prefix_run(start=start, group_id=group_id, parent_id=parent_id)
            else Dsv4StreamKind.COMPLETION
        )
        streams.append(
            Dsv4StreamSpec(
                stream_id=group_id,
                kind=kind,
                parent_stream_id=None if kind is Dsv4StreamKind.PREFIX else parent_id,
                start=start,
                end=end,
            )
        )
    _validate_stream_graph(streams)
    return tuple(streams)


def _scan_runs(
    group_row: torch.Tensor,
    parent_row: torch.Tensor,
) -> list[tuple[int, int, int, int]]:
    if int(group_row.numel()) == 0:
        return []
    groups = tuple(int(group_id) for group_id in group_row.tolist())
    parents = tuple(int(parent_id) for parent_id in parent_row.tolist())
    runs: list[tuple[int, int, int, int]] = []
    start = 0
    prev_group = groups[0]
    prev_parent = parents[0]
    for idx in range(1, len(groups)):
        group_id = groups[idx]
        parent_id = parents[idx]
        if group_id == prev_group and parent_id != prev_parent:
            raise RuntimeError(
                "DSV4 found a contiguous group run with inconsistent parent ids: "
                f"group_id={group_id}, index={idx}"
            )
        if group_id == prev_group:
            continue
        runs.append((start, idx, prev_group, prev_parent))
        start = idx
        prev_group = group_id
        prev_parent = parent_id
    runs.append((start, len(groups), prev_group, prev_parent))
    return runs


def _is_prefix_run(*, start: int, group_id: int, parent_id: int) -> bool:
    return int(group_id) == int(parent_id) or (
        int(start) == 0 and int(parent_id) == _PADDING_GROUP_ID
    )


def _validate_stream_graph(streams: list[Dsv4StreamSpec]) -> None:
    seen: set[int] = set()
    prefixes: set[int] = set()
    for stream in streams:
        if stream.stream_id in seen:
            raise RuntimeError(f"DSV4 stream {stream.stream_id} appears more than once")
        seen.add(stream.stream_id)
        if stream.kind is Dsv4StreamKind.PREFIX:
            prefixes.add(stream.stream_id)
    for stream in streams:
        if stream.kind is Dsv4StreamKind.PREFIX:
            continue
        if stream.parent_stream_id not in prefixes:
            raise RuntimeError(
                "DSV4 completion stream points to missing prefix: "
                f"stream_id={stream.stream_id}, parent={stream.parent_stream_id}"
            )


def _build_branch_views(
    streams: tuple[Dsv4StreamSpec, ...],
) -> tuple[Dsv4BranchView, ...]:
    by_id = {stream.stream_id: stream for stream in streams}
    branch_views: list[Dsv4BranchView] = []
    for stream in streams:
        if stream.kind is Dsv4StreamKind.PREFIX:
            branch_views.append(_make_branch_view(prefix=stream, suffix=None))
    for stream in streams:
        if stream.kind is Dsv4StreamKind.PREFIX:
            continue
        parent = by_id.get(stream.parent_stream_id)
        if parent is None:
            raise RuntimeError(
                f"DSV4 completion {stream.stream_id} missing parent {stream.parent_stream_id}"
            )
        branch_views.append(_make_branch_view(prefix=parent, suffix=stream))
    return tuple(branch_views)


def _make_branch_view(
    *,
    prefix: Dsv4StreamSpec,
    suffix: Dsv4StreamSpec | None,
) -> Dsv4BranchView:
    return Dsv4BranchView(
        branch_stream_id=prefix.stream_id if suffix is None else suffix.stream_id,
        prefix_stream_id=prefix.stream_id,
        suffix_stream_id=None if suffix is None else suffix.stream_id,
        prefix_start=prefix.start,
        prefix_end=prefix.end,
        suffix_start=None if suffix is None else suffix.start,
        suffix_end=None if suffix is None else suffix.end,
        prefix_token_count=prefix.size(),
    )


def _build_token_ownership(
    token_layout_index: TokenLayoutIndexLike,
) -> dict[int, tuple[int, int]]:
    owners: dict[int, tuple[int, int]] = {}
    for rank, ranges in enumerate(token_layout_index.ownership_ranges_by_rank):
        for start, end, local_start in ranges:
            for token_id in range(int(start), int(end)):
                owners[token_id] = (rank, int(local_start) + token_id - int(start))
    return owners


def _select_entries(
    *,
    layout: Dsv4CompressedLayout,
    entry_ids: Sequence[int] | None,
) -> tuple[Dsv4CompressedEntry, ...]:
    if entry_ids is None:
        return layout.entries
    return tuple(layout.entries[int(entry_id)] for entry_id in entry_ids)


def _compressed_head_dim(
    *,
    layout: Dsv4CompressedLayout,
    projected_dim: int,
    positional_bias: torch.Tensor,
) -> int:
    if int(positional_bias.shape[-1]) != int(projected_dim):
        raise RuntimeError(
            "DSV4 projected tensor and positional-bias dims must match, got "
            f"{projected_dim} vs {int(positional_bias.shape[-1])}"
        )
    if layout.spec.kind == Dsv4CompressionKind.HCA:
        return projected_dim
    if layout.spec.kind == Dsv4CompressionKind.CSA:
        if projected_dim % 2 != 0:
            raise RuntimeError(
                f"DSV4 CSA projected dim must be even, got {projected_dim}"
            )
        return projected_dim // 2
    raise RuntimeError(f"Unsupported DSV4 compression kind: {layout.spec.kind}")


def _empty_compressed_output(projected_kv: torch.Tensor, head_dim: int) -> torch.Tensor:
    if projected_kv.ndim == 2:
        return projected_kv.new_empty((0, head_dim))
    return projected_kv.new_empty((int(projected_kv.shape[0]), 0, head_dim))


def _build_projected_compression_gather(
    *,
    entries: tuple[Dsv4CompressedEntry, ...],
    layout: Dsv4CompressedLayout,
    token_ids: Sequence[int] | None,
    tensor_token_count: int,
) -> dict[str, torch.Tensor]:
    max_rows = _max_compression_rows(layout.spec)
    token_to_row = _token_to_row_map(token_ids, tensor_token_count)
    token_index = torch.zeros((len(entries), max_rows), dtype=torch.long)
    valid = torch.zeros((len(entries), max_rows), dtype=torch.bool)
    ape_row = torch.zeros((len(entries), max_rows), dtype=torch.long)
    half = torch.zeros((len(entries), max_rows), dtype=torch.long)
    ratio = int(layout.spec.ratio)

    for row, entry in enumerate(entries):
        deps = tuple(int(token_id) for token_id in entry.dependency_token_ids)
        if layout.spec.kind == Dsv4CompressionKind.HCA:
            slots = range(len(deps))
            dep_halves = (0,) * len(deps)
        elif layout.spec.kind == Dsv4CompressionKind.CSA:
            if len(deps) == ratio:
                slots = range(ratio, 2 * ratio)
                dep_halves = (1,) * len(deps)
            elif len(deps) == 2 * ratio:
                slots = range(2 * ratio)
                dep_halves = (0,) * ratio + (1,) * ratio
            else:
                raise RuntimeError(
                    "DSV4 CSA entries must contain ratio or 2*ratio dependencies, got "
                    f"{len(deps)} for entry_id={entry.entry_id}"
                )
        else:
            raise RuntimeError(f"Unsupported DSV4 compression kind: {layout.spec.kind}")

        for dep_offset, slot in enumerate(slots):
            token_index[row, slot] = _token_row(
                token_to_row=token_to_row,
                token_id=deps[dep_offset],
                tensor_token_count=tensor_token_count,
            )
            valid[row, slot] = True
            ape_row[row, slot] = dep_offset % ratio
            half[row, slot] = dep_halves[dep_offset]

    return {
        "token_index": token_index,
        "valid": valid,
        "ape_row": ape_row,
        "half": half,
    }


def _max_compression_rows(spec: Dsv4CompressionSpec) -> int:
    if spec.kind == Dsv4CompressionKind.CSA:
        return 2 * int(spec.ratio)
    if spec.kind == Dsv4CompressionKind.HCA:
        return int(spec.ratio)
    raise RuntimeError(f"Unsupported DSV4 compression kind: {spec.kind}")


def _token_to_row_map(
    token_ids: Sequence[int] | None,
    tensor_token_count: int,
) -> dict[int, int] | None:
    if token_ids is None:
        return None
    if len(token_ids) != tensor_token_count:
        raise RuntimeError(
            "DSV4 token_ids length must match projected tensor token count, got "
            f"{len(token_ids)} vs {tensor_token_count}"
        )
    token_to_row: dict[int, int] = {}
    for row, token_id in enumerate(token_ids):
        token = int(token_id)
        if token in token_to_row:
            raise RuntimeError(f"DSV4 token id {token} appears more than once")
        token_to_row[token] = row
    return token_to_row


def _token_row(
    *,
    token_to_row: dict[int, int] | None,
    token_id: int,
    tensor_token_count: int,
) -> int:
    if token_to_row is not None:
        row = token_to_row.get(int(token_id))
        if row is None:
            raise RuntimeError(
                f"DSV4 projected token buffer is missing packed token {token_id}"
            )
        return int(row)
    if int(token_id) < 0 or int(token_id) >= int(tensor_token_count):
        raise RuntimeError(
            f"DSV4 packed token {token_id} is outside projected tensor length "
            f"{tensor_token_count}"
        )
    return int(token_id)


def _gather_tokens(projected: torch.Tensor, token_index: torch.Tensor) -> torch.Tensor:
    if projected.ndim == 2:
        return projected.index_select(0, token_index.reshape(-1)).reshape(
            *token_index.shape,
            int(projected.shape[-1]),
        )
    return projected[:, token_index.reshape(-1), :].reshape(
        int(projected.shape[0]),
        *token_index.shape,
        int(projected.shape[-1]),
    )


def _select_csa_halves(
    tensor: torch.Tensor,
    half: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    split = tensor.reshape(*tensor.shape[:-1], 2, head_dim)
    half_index = half.to(tensor.device)
    if tensor.ndim == 3:
        gather_index = (
            half_index.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                *half_index.shape,
                1,
                head_dim,
            )
        )
    elif tensor.ndim == 4:
        gather_index = (
            half_index.unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                int(tensor.shape[0]),
                *half_index.shape,
                1,
                head_dim,
            )
        )
    else:
        raise RuntimeError(f"Unexpected DSV4 CSA tensor rank: {tensor.ndim}")
    return split.gather(-2, gather_index).squeeze(-2)


def _select_csa_positional_bias(
    positional_bias: torch.Tensor,
    ape_row: torch.Tensor,
    half: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    split = positional_bias.reshape(int(positional_bias.shape[0]), 2, head_dim)
    flat = split[ape_row.reshape(-1), half.reshape(-1)]
    return flat.reshape(*ape_row.shape, head_dim)


def _build_entries(
    *,
    branch_views: tuple[Dsv4BranchView, ...],
    token_owner: dict[int, tuple[int, int]],
    spec: Dsv4CompressionSpec,
) -> tuple[tuple[Dsv4CompressedEntry, ...], dict[int, tuple[int, ...]]]:
    entries: list[Dsv4CompressedEntry] = []
    entry_ids_by_branch: dict[int, list[int]] = defaultdict(list)
    dedupe: dict[tuple[Dsv4CompressionKind, int, tuple[int, ...], int], int] = {}
    for branch_view in branch_views:
        if branch_view.suffix_stream_id is not None:
            entry_ids_by_branch[int(branch_view.branch_stream_id)].extend(
                entry_ids_by_branch[int(branch_view.prefix_stream_id)]
            )
        branch_entry_index = len(entry_ids_by_branch[int(branch_view.branch_stream_id)])
        for dependency_positions in _dependency_positions(
            branch_len=branch_view.size(),
            spec=spec,
        ):
            closure_view_pos = dependency_positions[-1]
            if branch_view.suffix_stream_id is not None and int(closure_view_pos) < int(
                branch_view.prefix_token_count
            ):
                continue
            dependency_tokens = tuple(
                int(branch_view.token_id_at(position))
                for position in dependency_positions
            )
            closure_token_id = int(branch_view.token_id_at(closure_view_pos))
            shared_prefix_entry = int(closure_view_pos) < int(
                branch_view.prefix_token_count
            )
            dedupe_key = (
                spec.kind,
                int(spec.ratio),
                dependency_tokens,
                closure_token_id,
            )
            if shared_prefix_entry and dedupe_key in dedupe:
                entry_ids_by_branch[int(branch_view.branch_stream_id)].append(
                    dedupe[dedupe_key]
                )
                branch_entry_index += 1
                continue
            owner_rank, owner_local_offset = _owner_for_token(
                token_owner=token_owner,
                token_id=closure_token_id,
            )
            remote_dependency_token_ids = tuple(
                token_id
                for token_id in dependency_tokens
                if _owner_for_token(token_owner=token_owner, token_id=token_id)[0]
                != owner_rank
            )
            entry = Dsv4CompressedEntry(
                entry_id=len(entries),
                kind=spec.kind,
                ratio=spec.ratio,
                branch_stream_id=branch_view.branch_stream_id,
                prefix_stream_id=branch_view.prefix_stream_id,
                closure_token_id=closure_token_id,
                closure_view_pos=closure_view_pos,
                owner_rank=owner_rank,
                owner_local_offset=owner_local_offset,
                dependency_token_ids=dependency_tokens,
                remote_dependency_token_ids=remote_dependency_token_ids,
                shared_prefix_entry=shared_prefix_entry,
                branch_entry_index=branch_entry_index,
            )
            if shared_prefix_entry:
                dedupe[dedupe_key] = int(entry.entry_id)
            entries.append(entry)
            entry_ids_by_branch[int(branch_view.branch_stream_id)].append(
                int(entry.entry_id)
            )
            branch_entry_index += 1
    return tuple(entries), {
        branch_id: tuple(entry_ids)
        for branch_id, entry_ids in sorted(entry_ids_by_branch.items())
    }


def _dependency_positions(
    *,
    branch_len: int,
    spec: Dsv4CompressionSpec,
) -> Iterator[tuple[int, ...]]:
    if spec.kind == Dsv4CompressionKind.HCA:
        for start in range(0, branch_len - spec.ratio + 1, spec.ratio):
            yield tuple(range(start, start + spec.ratio))
        return
    if spec.kind != Dsv4CompressionKind.CSA:
        raise RuntimeError(f"Unsupported DSV4 compression kind: {spec.kind}")
    for start in range(0, branch_len - spec.ratio + 1, spec.ratio):
        current = tuple(range(start, start + spec.ratio))
        previous = tuple(range(start - spec.ratio, start)) if start > 0 else tuple()
        yield previous + current


def _owner_for_token(
    *,
    token_owner: dict[int, tuple[int, int]],
    token_id: int,
) -> tuple[int, int]:
    owner = token_owner.get(int(token_id))
    if owner is None:
        raise RuntimeError(f"DSV4 token {token_id} has no CP owner")
    return owner


def _entry_ids_by_closure_token(
    entries: tuple[Dsv4CompressedEntry, ...],
) -> dict[int, tuple[int, ...]]:
    by_token: dict[int, list[int]] = defaultdict(list)
    for entry in entries:
        by_token[int(entry.closure_token_id)].append(int(entry.entry_id))
    return {token_id: tuple(entry_ids) for token_id, entry_ids in by_token.items()}


def _closure_token_ids(entries: tuple[Dsv4CompressedEntry, ...]) -> tuple[int, ...]:
    return tuple(sorted({int(entry.closure_token_id) for entry in entries}))


def _build_halo_transfers(
    entries: tuple[Dsv4CompressedEntry, ...],
    token_owner: dict[int, tuple[int, int]],
) -> tuple[Dsv4HaloTransfer, ...]:
    by_peer: dict[tuple[int, int], dict[int, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for entry in entries:
        for token_id in entry.remote_dependency_token_ids:
            source_rank, _ = _owner_for_token(
                token_owner=token_owner,
                token_id=token_id,
            )
            by_peer[(source_rank, entry.owner_rank)][token_id].add(entry.entry_id)

    transfers: list[Dsv4HaloTransfer] = []
    for (source_rank, target_rank), token_to_entries in sorted(by_peer.items()):
        token_ids = tuple(sorted(token_to_entries))
        entry_ids = tuple(
            sorted(
                {
                    entry_id
                    for entries_ in token_to_entries.values()
                    for entry_id in entries_
                }
            )
        )
        transfers.append(
            Dsv4HaloTransfer(
                source_rank=source_rank,
                target_rank=target_rank,
                token_ids=token_ids,
                entry_ids=entry_ids,
            )
        )
    return tuple(transfers)


def _entry_ids_by_owner(
    entries: tuple[Dsv4CompressedEntry, ...],
    token_layout_index: TokenLayoutIndexLike,
) -> tuple[tuple[int, ...], ...]:
    by_rank: list[list[int]] = [
        [] for _ in range(len(token_layout_index.ownership_ranges_by_rank))
    ]
    for entry in entries:
        by_rank[int(entry.owner_rank)].append(int(entry.entry_id))
    return tuple(tuple(entry_ids) for entry_ids in by_rank)


def _raw_token_owner_ranks(
    token_owner: dict[int, tuple[int, int]],
) -> tuple[int, ...]:
    if not token_owner:
        return ()
    ranks = [-1] * (max(token_owner) + 1)
    for token_id, owner in token_owner.items():
        ranks[int(token_id)] = int(owner[0])
    return tuple(ranks)
