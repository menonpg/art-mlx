from __future__ import annotations

import math

import pytest
import torch

from art.megatron.dsv4 import (
    dsv4_disabled_attn_sink,
    dsv4_sparse_bwd,
    dsv4_sparse_fwd,
)
import art.megatron.dsv4.sparse_kernel as sparse_kernel


def test_sparse_fwd_converts_miles_log2_lse_to_natural_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, torch.Tensor | float | None] = {}

    def fake_fwd(
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        sm_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls["q"] = q
        calls["kv"] = kv
        calls["attn_sink"] = attn_sink
        calls["topk_idxs"] = topk_idxs
        calls["sm_scale"] = sm_scale
        assert topk_idxs.dtype == torch.int32
        assert attn_sink.dtype == torch.float32
        return torch.full_like(q, 2.0), torch.full(q.shape[:-1], 4.0)

    monkeypatch.setattr(
        sparse_kernel, "_load_miles_sparse_mla", lambda: (fake_fwd, None)
    )

    q = torch.randn(2, 3, 4, 8)
    kv = torch.randn(2, 5, 8)
    sink = torch.randn(4, dtype=torch.float64)
    topk = torch.tensor([[[0, 1], [2, -1], [4, 3]]] * 2, dtype=torch.int64)

    result = dsv4_sparse_fwd(q=q, kv=kv, attn_sink=sink, topk=topk, scale=0.125)

    torch.testing.assert_close(result.out, torch.full_like(q, 2.0))
    torch.testing.assert_close(
        result.lse, torch.full(q.shape[:-1], 4.0 * math.log(2.0))
    )
    assert calls["sm_scale"] == 0.125


def test_sparse_bwd_converts_global_lse_to_miles_log2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, torch.Tensor | float | None] = {}

    def fake_bwd(
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        o: torch.Tensor,
        do: torch.Tensor,
        topk_idxs: torch.Tensor,
        lse: torch.Tensor,
        sm_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        calls["global_out"] = o
        calls["grad_out"] = do
        calls["topk_idxs"] = topk_idxs
        calls["lse"] = lse
        calls["sm_scale"] = sm_scale
        assert topk_idxs.dtype == torch.int32
        assert attn_sink.dtype == torch.float32
        return (
            torch.ones_like(q),
            torch.full_like(kv, 2.0),
            torch.full_like(attn_sink, 3.0),
        )

    monkeypatch.setattr(
        sparse_kernel, "_load_miles_sparse_mla", lambda: (None, fake_bwd)
    )

    q = torch.randn(2, 3, 4, 8)
    kv = torch.randn(2, 5, 8)
    sink = torch.full((4,), float("-inf"))
    topk = torch.tensor([[[0, 1], [2, -1], [4, 3]]] * 2, dtype=torch.int64)
    global_out = torch.randn_like(q)
    grad_out = torch.randn_like(q)
    global_lse = torch.full(q.shape[:-1], math.log(8.0))

    result = dsv4_sparse_bwd(
        q=q,
        kv=kv,
        attn_sink=sink,
        topk=topk,
        global_out=global_out,
        grad_out=grad_out,
        global_lse=global_lse,
        scale=0.25,
    )

    torch.testing.assert_close(result.dq, torch.ones_like(q))
    torch.testing.assert_close(result.dkv, torch.full_like(kv, 2.0))
    torch.testing.assert_close(result.d_attn_sink, torch.full_like(sink, 3.0))
    torch.testing.assert_close(calls["lse"], torch.full(q.shape[:-1], 3.0))
    torch.testing.assert_close(calls["global_out"], global_out)
    torch.testing.assert_close(calls["grad_out"], grad_out)
    assert calls["sm_scale"] == 0.25


def test_disabled_attn_sink_is_negative_infinity() -> None:
    sink = torch.randn(4)
    disabled = dsv4_disabled_attn_sink(sink)

    assert disabled.shape == sink.shape
    assert disabled.dtype == sink.dtype
    assert torch.isneginf(disabled).all()


def test_sparse_kernel_rejects_mismatched_topk_shape() -> None:
    with pytest.raises(RuntimeError, match="topk batch/query"):
        dsv4_sparse_fwd(
            q=torch.randn(1, 3, 2, 4),
            kv=torch.randn(1, 5, 4),
            attn_sink=torch.randn(2),
            topk=torch.zeros(1, 2, 3, dtype=torch.long),
        )
