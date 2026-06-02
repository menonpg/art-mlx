from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
import pytest
import torch

from art.megatron.dsv4 import (
    Dsv4CompressedLayout,
    Dsv4CompressionHaloPayload,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    accumulate_dsv4_compression_halo_gradient_payloads,
    build_dsv4_compressed_layout,
    compress_owned_projected_kv,
    compress_projected_kv,
    materialize_dsv4_compression_token_buffer,
    pack_dsv4_compression_halo_gradient_payloads,
    pack_dsv4_compression_halo_payloads,
)


class _LayoutIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    ownership_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    token_counts_by_rank: tuple[int, ...]


def test_csa_projected_compression_matches_slow_reference_and_grads() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    torch.manual_seed(1)
    projected_kv = torch.randn(18, 10, requires_grad=True)
    projected_gate = torch.randn(18, 10, requires_grad=True)
    positional_bias = torch.randn(4, 10, requires_grad=True)
    ref_kv = projected_kv.detach().clone().requires_grad_()
    ref_gate = projected_gate.detach().clone().requires_grad_()
    ref_bias = positional_bias.detach().clone().requires_grad_()

    actual = compress_projected_kv(
        layout=layout,
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        positional_bias=positional_bias,
    )
    expected = _slow_compress_projected(
        layout=layout,
        projected_kv=ref_kv,
        projected_gate=ref_gate,
        positional_bias=ref_bias,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    upstream = torch.randn_like(actual)
    actual.mul(upstream).sum().backward()
    expected.mul(upstream).sum().backward()

    torch.testing.assert_close(projected_kv.grad, ref_kv.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        projected_gate.grad,
        ref_gate.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        positional_bias.grad,
        ref_bias.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_grad_nonzero(projected_kv)
    _assert_grad_nonzero(projected_gate)
    _assert_grad_nonzero(positional_bias)


def test_hca_projected_compression_matches_slow_reference() -> None:
    layout = _layout(Dsv4CompressionKind.HCA)
    torch.manual_seed(2)
    projected_kv = torch.randn(18, 6)
    projected_gate = torch.randn(18, 6)
    positional_bias = torch.randn(4, 6)

    actual = compress_projected_kv(
        layout=layout,
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        positional_bias=positional_bias,
    )
    expected = _slow_compress_projected(
        layout=layout,
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        positional_bias=positional_bias,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_projected_compression_supports_compact_token_buffer_and_entry_subset() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    torch.manual_seed(3)
    projected_kv = torch.randn(18, 10)
    projected_gate = torch.randn(18, 10)
    positional_bias = torch.randn(4, 10)
    entry_ids = (2, 3)
    token_ids = tuple(
        sorted(
            {
                token_id
                for entry_id in entry_ids
                for token_id in layout.entries[entry_id].dependency_token_ids
            }
        )
    )

    expected = compress_projected_kv(
        layout=layout,
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        positional_bias=positional_bias,
        entry_ids=entry_ids,
    )
    actual = compress_projected_kv(
        layout=layout,
        projected_kv=projected_kv.index_select(0, torch.tensor(token_ids)),
        projected_gate=projected_gate.index_select(0, torch.tensor(token_ids)),
        positional_bias=positional_bias,
        entry_ids=entry_ids,
        token_ids=token_ids,
    )
    owned = compress_owned_projected_kv(
        layout=layout,
        owner_rank=1,
        projected_kv=projected_kv.index_select(0, torch.tensor(token_ids)),
        projected_gate=projected_gate.index_select(0, torch.tensor(token_ids)),
        positional_bias=positional_bias,
        token_ids=token_ids,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(owned, expected, rtol=1e-6, atol=1e-6)


def test_compression_halo_payloads_match_global_compression_and_return_grads() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    torch.manual_seed(4)
    ref_kv = torch.randn(18, 10, requires_grad=True)
    ref_gate = torch.randn(18, 10, requires_grad=True)
    ref_bias = torch.randn(4, 10, requires_grad=True)
    source_kv = ref_kv.detach()[:8].clone().requires_grad_()
    source_gate = ref_gate.detach()[:8].clone().requires_grad_()
    target_kv = ref_kv.detach()[8:].clone().requires_grad_()
    target_gate = ref_gate.detach()[8:].clone().requires_grad_()
    actual_bias = ref_bias.detach().clone().requires_grad_()

    sent_payloads = pack_dsv4_compression_halo_payloads(
        layout=layout,
        source_rank=0,
        projected_kv=source_kv,
        projected_gate=source_gate,
        token_ids=tuple(range(8)),
    )
    assert len(sent_payloads) == 1
    assert sent_payloads[0].token_ids == (4, 5, 6, 7)
    assert sent_payloads[0].entry_ids == (2, 3)

    received_payloads = tuple(
        _detach_halo_payload(payload) for payload in sent_payloads
    )
    buffer = materialize_dsv4_compression_token_buffer(
        layout=layout,
        owner_rank=1,
        projected_kv=target_kv,
        projected_gate=target_gate,
        token_ids=tuple(range(8, 18)),
        halo_payloads=received_payloads,
    )
    assert buffer.token_ids == (4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16)

    actual = compress_owned_projected_kv(
        layout=layout,
        owner_rank=1,
        projected_kv=buffer.projected_kv,
        projected_gate=buffer.projected_gate,
        positional_bias=actual_bias,
        token_ids=buffer.token_ids,
    )
    expected = compress_owned_projected_kv(
        layout=layout,
        owner_rank=1,
        projected_kv=ref_kv,
        projected_gate=ref_gate,
        positional_bias=ref_bias,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    actual_loss = actual.square().sum()
    d_buffer_kv, d_buffer_gate = torch.autograd.grad(
        actual_loss,
        (buffer.projected_kv, buffer.projected_gate),
        retain_graph=True,
    )
    actual_loss.backward()
    expected.square().sum().backward()
    assert ref_kv.grad is not None
    assert ref_gate.grad is not None
    assert ref_bias.grad is not None
    assert actual_bias.grad is not None

    grad_payloads = pack_dsv4_compression_halo_gradient_payloads(
        layout=layout,
        owner_rank=1,
        token_ids=buffer.token_ids,
        dprojected_kv=d_buffer_kv,
        dprojected_gate=d_buffer_gate,
    )
    returned = accumulate_dsv4_compression_halo_gradient_payloads(
        target_rank=0,
        token_ids=tuple(range(8)),
        dprojected_kv=torch.zeros_like(source_kv),
        dprojected_gate=torch.zeros_like(source_gate),
        halo_gradient_payloads=grad_payloads,
    )

    torch.testing.assert_close(
        returned.projected_kv[:4], torch.zeros_like(source_kv[:4])
    )
    torch.testing.assert_close(
        returned.projected_kv[4:],
        ref_kv.grad[:8][4:],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        returned.projected_gate[:4], torch.zeros_like(source_gate[:4])
    )
    torch.testing.assert_close(
        returned.projected_gate[4:],
        ref_gate.grad[:8][4:],
        rtol=1e-6,
        atol=1e-6,
    )
    assert target_kv.grad is not None
    assert target_gate.grad is not None
    torch.testing.assert_close(target_kv.grad, ref_kv.grad[8:], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        target_gate.grad,
        ref_gate.grad[8:],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(actual_bias.grad, ref_bias.grad, rtol=1e-6, atol=1e-6)


def test_compression_halo_payloads_reject_missing_or_duplicate_tokens() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    projected_kv = torch.randn(18, 10)
    projected_gate = torch.randn(18, 10)
    payloads = pack_dsv4_compression_halo_payloads(
        layout=layout,
        source_rank=0,
        projected_kv=projected_kv[:8],
        projected_gate=projected_gate[:8],
        token_ids=tuple(range(8)),
    )

    with pytest.raises(RuntimeError, match="missing from local\\+halo token_ids"):
        materialize_dsv4_compression_token_buffer(
            layout=layout,
            owner_rank=1,
            projected_kv=projected_kv[8:],
            projected_gate=projected_gate[8:],
            token_ids=tuple(range(8, 18)),
        )
    with pytest.raises(RuntimeError, match="local\\+halo token_ids contains duplicate"):
        materialize_dsv4_compression_token_buffer(
            layout=layout,
            owner_rank=1,
            projected_kv=projected_kv[8:],
            projected_gate=projected_gate[8:],
            token_ids=tuple(range(8, 18)),
            halo_payloads=payloads + payloads,
        )
    bad_source_payload = Dsv4CompressionHaloPayload(
        source_rank=2,
        target_rank=1,
        token_ids=payloads[0].token_ids,
        entry_ids=payloads[0].entry_ids,
        projected_kv=payloads[0].projected_kv,
        projected_gate=payloads[0].projected_gate,
    )
    with pytest.raises(RuntimeError, match="no planned transfer"):
        materialize_dsv4_compression_token_buffer(
            layout=layout,
            owner_rank=1,
            projected_kv=projected_kv[8:],
            projected_gate=projected_gate[8:],
            token_ids=tuple(range(8, 18)),
            halo_payloads=(bad_source_payload,),
        )
    with pytest.raises(RuntimeError, match="token_ids contains duplicate"):
        pack_dsv4_compression_halo_payloads(
            layout=layout,
            source_rank=0,
            projected_kv=projected_kv[:8],
            projected_gate=projected_gate[:8],
            token_ids=(0, 1, 2, 3, 4, 5, 6, 6),
        )


def test_batched_projected_compression_matches_per_sequence_reference() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    torch.manual_seed(5)
    projected_kv = torch.randn(2, 18, 10)
    projected_gate = torch.randn(2, 18, 10)
    positional_bias = torch.randn(4, 10)

    actual = compress_projected_kv(
        layout=layout,
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        positional_bias=positional_bias,
    )
    expected = _slow_compress_projected(
        layout=layout,
        projected_kv=projected_kv,
        projected_gate=projected_gate,
        positional_bias=positional_bias,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def _slow_compress_projected(
    *,
    layout: Dsv4CompressedLayout,
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    positional_bias: torch.Tensor,
    entry_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    entries = (
        layout.entries
        if entry_ids is None
        else tuple(layout.entries[int(entry_id)] for entry_id in entry_ids)
    )
    if projected_kv.ndim == 2:
        return torch.stack(
            [
                _slow_compress_entry(
                    layout=layout,
                    entry_id=int(entry.entry_id),
                    projected_kv=projected_kv,
                    projected_gate=projected_gate,
                    positional_bias=positional_bias,
                )
                for entry in entries
            ]
        )
    return torch.stack(
        [
            _slow_compress_projected(
                layout=layout,
                projected_kv=projected_kv[batch],
                projected_gate=projected_gate[batch],
                positional_bias=positional_bias,
                entry_ids=entry_ids,
            )
            for batch in range(int(projected_kv.shape[0]))
        ]
    )


def _assert_grad_nonzero(tensor: torch.Tensor) -> None:
    assert tensor.grad is not None
    assert not bool(tensor.grad.abs().sum().eq(0).item())


def _detach_halo_payload(
    payload: Dsv4CompressionHaloPayload,
) -> Dsv4CompressionHaloPayload:
    return Dsv4CompressionHaloPayload(
        source_rank=payload.source_rank,
        target_rank=payload.target_rank,
        token_ids=payload.token_ids,
        entry_ids=payload.entry_ids,
        projected_kv=payload.projected_kv.detach().clone().requires_grad_(),
        projected_gate=payload.projected_gate.detach().clone().requires_grad_(),
    )


def _slow_compress_entry(
    *,
    layout: Dsv4CompressedLayout,
    entry_id: int,
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    positional_bias: torch.Tensor,
) -> torch.Tensor:
    entry = layout.entries[entry_id]
    deps = torch.tensor(entry.dependency_token_ids, dtype=torch.long)
    ratio = int(layout.spec.ratio)
    if layout.spec.kind == Dsv4CompressionKind.HCA:
        rows_kv = projected_kv.index_select(0, deps)
        rows_score = (
            projected_gate.index_select(0, deps)
            + positional_bias[: len(entry.dependency_token_ids)]
        )
    elif layout.spec.kind == Dsv4CompressionKind.CSA:
        head_dim = int(projected_kv.shape[-1]) // 2
        if len(entry.dependency_token_ids) == ratio:
            rows_kv = projected_kv.index_select(0, deps)[..., head_dim:]
            rows_score = (
                projected_gate.index_select(0, deps)[..., head_dim:]
                + positional_bias[..., head_dim:]
            )
        else:
            previous = deps[:ratio]
            current = deps[ratio:]
            rows_kv = torch.cat(
                [
                    projected_kv.index_select(0, previous)[..., :head_dim],
                    projected_kv.index_select(0, current)[..., head_dim:],
                ],
                dim=0,
            )
            rows_score = torch.cat(
                [
                    projected_gate.index_select(0, previous)[..., :head_dim]
                    + positional_bias[..., :head_dim],
                    projected_gate.index_select(0, current)[..., head_dim:]
                    + positional_bias[..., head_dim:],
                ],
                dim=0,
            )
    else:
        raise RuntimeError(f"Unsupported compression kind {layout.spec.kind}")
    weights = torch.softmax(rows_score.float(), dim=0)
    return rows_kv.mul(weights).sum(dim=0)


def _layout(kind: Dsv4CompressionKind) -> Dsv4CompressedLayout:
    return build_dsv4_compressed_layout(
        group_ids=torch.tensor([[0] * 8 + [1] * 5 + [2] * 5 + [-1] * 2]),
        parent_ids=torch.tensor([[0] * 8 + [0] * 5 + [0] * 5 + [-1] * 2]),
        token_layout_index=_LayoutIndex(
            ownership_ranges_by_rank=(
                ((0, 8, 0),),
                ((8, 18, 0),),
            ),
            token_counts_by_rank=(8, 10),
        ),
        spec=Dsv4CompressionSpec(kind=kind, ratio=4),
    )
