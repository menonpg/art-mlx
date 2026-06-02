from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
import torch

from art.megatron.dsv4 import (
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    build_dsv4_compressed_layout,
    compress_owned_projected_kv,
    compress_projected_kv,
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


def test_batched_projected_compression_matches_per_sequence_reference() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    torch.manual_seed(4)
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
