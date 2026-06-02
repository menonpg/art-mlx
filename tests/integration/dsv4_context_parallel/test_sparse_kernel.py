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


def test_sparse_bwd_casts_replay_tensors_to_q_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, torch.dtype] = {}

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
        del attn_sink, topk_idxs, lse, sm_scale
        calls["q"] = q.dtype
        calls["kv"] = kv.dtype
        calls["global_out"] = o.dtype
        calls["grad_out"] = do.dtype
        return (torch.ones_like(q), torch.ones_like(kv), torch.ones(q.shape[2]))

    monkeypatch.setattr(
        sparse_kernel, "_load_miles_sparse_mla", lambda: (None, fake_bwd)
    )

    q = torch.randn(1, 2, 2, 4, dtype=torch.bfloat16)
    kv = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    dsv4_sparse_bwd(
        q=q,
        kv=kv,
        attn_sink=torch.full((2,), float("-inf")),
        topk=torch.tensor([[[0, 1], [2, -1]]]),
        global_out=torch.randn(1, 2, 2, 4),
        grad_out=torch.randn(1, 2, 2, 4),
        global_lse=torch.randn(1, 2, 2),
    )

    assert calls == {
        "q": torch.bfloat16,
        "kv": torch.bfloat16,
        "global_out": torch.bfloat16,
        "grad_out": torch.bfloat16,
    }


def test_sparse_bwd_chunks_long_topk_and_computes_sink_grad_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[torch.Tensor] = []

    def fake_preprocess(
        batch: int,
        query_count: int,
        head_count: int,
        head_dim: int,
    ):
        del batch, query_count, head_count, head_dim

        def kernel(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
            return (o.float() * do.float()).sum(dim=-1)

        return kernel

    def fake_bwd(
        batch: int,
        query_count: int,
        kv_count: int,
        head_count: int,
        head_dim: int,
        topk: int,
        sm_scale: float | None = None,
        block_size: int = 32,
    ):
        del batch, query_count, kv_count, head_count, head_dim, sm_scale
        assert topk == 64
        assert block_size == 64

        def kernel(
            q: torch.Tensor,
            kv: torch.Tensor,
            do: torch.Tensor,
            attn_sink: torch.Tensor,
            indices: torch.Tensor,
            lse: torch.Tensor,
            delta: torch.Tensor,
            dkv: torch.Tensor,
            d_attn_sink: torch.Tensor,
        ) -> torch.Tensor:
            del kv, do, attn_sink, lse, delta
            calls.append(indices.detach().clone())
            dkv.add_(1.0)
            d_attn_sink.add_(999.0)
            return torch.full_like(q, float(len(calls)))

        return kernel

    def fake_postprocess(batch: int, kv_count: int, head_dim: int):
        del batch, kv_count, head_dim

        def kernel(dkv: torch.Tensor) -> torch.Tensor:
            return dkv

        return kernel

    monkeypatch.setattr(
        sparse_kernel,
        "_load_miles_sparse_mla_bwd_factories",
        lambda: (fake_preprocess, fake_bwd, fake_postprocess),
    )

    q = torch.randn(1, 1, 2, 4)
    kv = torch.randn(1, 3, 4)
    global_out = torch.ones_like(q)
    grad_out = torch.full_like(q, 2.0)
    global_lse = torch.full(q.shape[:-1], math.log(4.0))
    topk = torch.arange(96).reshape(1, 1, 96).remainder(3)

    result = dsv4_sparse_bwd(
        q=q,
        kv=kv,
        attn_sink=torch.tensor([0.0, float("-inf")]),
        topk=topk,
        global_out=global_out,
        grad_out=grad_out,
        global_lse=global_lse,
        scale=0.25,
    )

    assert len(calls) == 2
    assert calls[0].shape == (1, 1, 64)
    assert calls[1].shape == (1, 1, 64)
    assert calls[1][0, 0, 32:].tolist() == [-1] * 32
    torch.testing.assert_close(result.dq, torch.full_like(q, 3.0))
    torch.testing.assert_close(result.dkv, torch.full_like(kv, 2.0))
    torch.testing.assert_close(result.d_attn_sink, torch.tensor([-2.0, 0.0]))


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


def test_sparse_kernel_rejects_q_kv_dtype_mismatch_before_miles_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_loader() -> tuple[object, object]:
        raise AssertionError("dtype mismatch should not load Miles")

    monkeypatch.setattr(sparse_kernel, "_load_miles_sparse_mla", fail_loader)

    with pytest.raises(RuntimeError, match="dtypes must match"):
        dsv4_sparse_fwd(
            q=torch.randn(1, 3, 2, 4, dtype=torch.bfloat16),
            kv=torch.randn(1, 5, 4),
            attn_sink=torch.randn(2),
            topk=torch.zeros(1, 3, 2, dtype=torch.long),
        )


def test_sparse_fwd_patches_all_invalid_rows_without_changing_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, torch.Tensor] = {}

    def fake_fwd(
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        sm_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kv, attn_sink, sm_scale
        calls["topk"] = topk_idxs
        assert topk_idxs.dtype == torch.int32
        assert int(topk_idxs[0, 0, 0].item()) == 0
        assert topk_idxs[0, 0, 1:].tolist() == [-1, -1]
        assert topk_idxs[0, 1].tolist() == [-1, 2, -1]
        out = torch.arange(q.numel(), dtype=q.dtype).reshape_as(q)
        lse = torch.full(q.shape[:-1], 3.0)
        return out, lse

    monkeypatch.setattr(
        sparse_kernel, "_load_miles_sparse_mla", lambda: (fake_fwd, None)
    )

    q = torch.randn(1, 2, 2, 4)
    kv = torch.randn(1, 3, 4)
    topk = torch.tensor([[[-1, -1, -1], [-1, 2, -1]]], dtype=torch.long)

    result = dsv4_sparse_fwd(
        q=q,
        kv=kv,
        attn_sink=torch.full((2,), float("-inf")),
        topk=topk,
    )

    torch.testing.assert_close(result.out[:, 0], torch.zeros_like(result.out[:, 0]))
    assert torch.isneginf(result.lse[:, 0]).all()
    assert not bool(result.out[:, 1].abs().sum().eq(0).item())
    torch.testing.assert_close(result.lse[:, 1], torch.full((1, 2), 3.0 * math.log(2)))


def test_sparse_bwd_patches_all_invalid_rows_and_zeroes_replay_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, torch.Tensor] = {}

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
        del sm_scale
        calls["global_out"] = o
        calls["grad_out"] = do
        calls["topk"] = topk_idxs
        calls["lse"] = lse
        assert int(topk_idxs[0, 0, 0].item()) == 0
        assert topk_idxs[0, 0, 1:].tolist() == [-1, -1]
        assert topk_idxs[0, 1].tolist() == [1, -1, -1]
        return (
            torch.ones_like(q),
            torch.full_like(kv, 2.0),
            torch.full_like(attn_sink, 3.0),
        )

    monkeypatch.setattr(
        sparse_kernel, "_load_miles_sparse_mla", lambda: (None, fake_bwd)
    )

    q = torch.randn(1, 2, 2, 4)
    kv = torch.randn(1, 3, 4)
    topk = torch.tensor([[[-1, -1, -1], [1, -1, -1]]], dtype=torch.long)
    global_out = torch.randn_like(q)
    grad_out = torch.randn_like(q)
    global_lse = torch.full(q.shape[:-1], math.log(8.0))

    result = dsv4_sparse_bwd(
        q=q,
        kv=kv,
        attn_sink=torch.full((2,), float("-inf")),
        topk=topk,
        global_out=global_out,
        grad_out=grad_out,
        global_lse=global_lse,
    )

    torch.testing.assert_close(result.dq[:, 0], torch.zeros_like(result.dq[:, 0]))
    torch.testing.assert_close(result.dq[:, 1], torch.ones_like(result.dq[:, 1]))
    torch.testing.assert_close(calls["global_out"][:, 0], torch.zeros_like(q[:, 0]))
    torch.testing.assert_close(calls["grad_out"][:, 0], torch.zeros_like(q[:, 0]))
    torch.testing.assert_close(calls["lse"][:, 0], torch.zeros_like(global_lse[:, 0]))
    torch.testing.assert_close(calls["lse"][:, 1], torch.full((1, 2), 3.0))


def test_sparse_empty_kv_stage_skips_miles_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_loader() -> tuple[object, object]:
        raise AssertionError("empty DSV4 sparse stage should not load Miles")

    monkeypatch.setattr(sparse_kernel, "_load_miles_sparse_mla", fail_loader)

    q = torch.randn(1, 2, 2, 4)
    kv = torch.empty(1, 0, 4)
    topk = torch.empty(1, 2, 0, dtype=torch.long)
    sink = torch.tensor([0.5, -1.25])
    fwd = dsv4_sparse_fwd(q=q, kv=kv, attn_sink=sink, topk=topk)
    bwd = dsv4_sparse_bwd(
        q=q,
        kv=kv,
        attn_sink=sink,
        topk=topk,
        global_out=torch.randn_like(q),
        grad_out=torch.randn_like(q),
        global_lse=torch.randn(q.shape[:-1]),
    )

    torch.testing.assert_close(fwd.out, torch.zeros_like(q))
    torch.testing.assert_close(
        fwd.lse,
        sink.reshape(1, 1, -1).expand(q.shape[:-1]),
    )
    torch.testing.assert_close(bwd.dq, torch.zeros_like(q))
    torch.testing.assert_close(bwd.dkv, torch.zeros_like(kv))
    torch.testing.assert_close(bwd.d_attn_sink, torch.zeros_like(sink))
