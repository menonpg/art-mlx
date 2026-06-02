from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import torch

_MILES_PATH = os.environ.get("DSV4_MILES_PATH", "/mnt/ws_pvc/ws/scratch/miles_inspect")
if _MILES_PATH and Path(_MILES_PATH).exists() and _MILES_PATH not in sys.path:
    sys.path.insert(0, _MILES_PATH)

pytest.importorskip(
    "miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_fwd"
)
pytest.importorskip(
    "miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_bwd"
)

from art.megatron.dsv4 import dsv4_sparse_bwd, dsv4_sparse_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Miles TileLang sparse MLA reference test requires CUDA",
)


def test_miles_sparse_mla_matches_eager_reference_with_all_invalid_row() -> None:
    torch.manual_seed(7)
    scale = 1.0 / (512**0.5)
    q = torch.randn(1, 3, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, 6, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(64, device="cuda", dtype=torch.float32) * 0.1
    topk = torch.tensor(
        [[[0, 1, -1, -1], [2, 3, 4, -1], [-1, -1, -1, -1]]],
        device="cuda",
    )

    q_ref = q.detach().float().requires_grad_()
    kv_ref = kv.detach().float().requires_grad_()
    sink_ref = sink.detach().float().requires_grad_()
    ref = _eager_sparse_mqa(
        q=q_ref,
        kv=kv_ref,
        sink=sink_ref,
        topk=topk,
        scale=scale,
    )
    grad_out = torch.randn_like(ref).bfloat16()
    (ref * grad_out.float()).sum().backward()
    assert q_ref.grad is not None
    assert kv_ref.grad is not None
    assert sink_ref.grad is not None

    fwd = dsv4_sparse_fwd(q=q, kv=kv, attn_sink=sink, topk=topk, scale=scale)
    bwd = dsv4_sparse_bwd(
        q=q,
        kv=kv,
        attn_sink=sink,
        topk=topk,
        global_out=fwd.out,
        grad_out=grad_out,
        global_lse=fwd.lse,
        scale=scale,
    )
    torch.cuda.synchronize()

    assert _mean_abs_pct(fwd.out.float(), ref.detach()) <= 3.0
    assert _mean_abs_pct(bwd.dq.float(), q_ref.grad) <= 5.0
    assert _mean_abs_pct(bwd.dkv.float(), kv_ref.grad) <= 5.0
    assert _mean_abs_pct(bwd.d_attn_sink.float(), sink_ref.grad) <= 5.0
    assert not bool(fwd.out[:, :2].abs().sum().eq(0).item())
    assert not bool(bwd.dq[:, :2].abs().sum().eq(0).item())
    torch.testing.assert_close(fwd.out[:, 2], torch.zeros_like(fwd.out[:, 2]))
    torch.testing.assert_close(fwd.lse[:, 2], sink.reshape(1, -1))


def test_miles_sparse_mla_long_topk_backward_matches_eager_reference() -> None:
    torch.manual_seed(23)
    scale = 1.0 / (512**0.5)
    q = torch.randn(1, 2, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, 65, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(64, device="cuda", dtype=torch.float32) * 0.1
    topk = torch.arange(65, device="cuda").reshape(1, 1, 65).expand(1, 2, 65)

    q_ref = q.detach().float().requires_grad_()
    kv_ref = kv.detach().float().requires_grad_()
    sink_ref = sink.detach().float().requires_grad_()
    ref = _eager_sparse_mqa(
        q=q_ref,
        kv=kv_ref,
        sink=sink_ref,
        topk=topk,
        scale=scale,
    )
    grad_out = torch.randn_like(ref).bfloat16()
    (ref * grad_out.float()).sum().backward()
    assert q_ref.grad is not None
    assert kv_ref.grad is not None
    assert sink_ref.grad is not None

    fwd = dsv4_sparse_fwd(q=q, kv=kv, attn_sink=sink, topk=topk, scale=scale)
    bwd = dsv4_sparse_bwd(
        q=q,
        kv=kv,
        attn_sink=sink,
        topk=topk,
        global_out=fwd.out,
        grad_out=grad_out,
        global_lse=fwd.lse,
        scale=scale,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(bwd.dq).all()
    assert torch.isfinite(bwd.dkv).all()
    assert _mean_abs_pct(fwd.out.float(), ref.detach()) <= 3.0
    assert _mean_abs_pct(bwd.dq.float(), q_ref.grad) <= 5.0
    assert _mean_abs_pct(bwd.dkv.float(), kv_ref.grad) <= 5.0
    assert _mean_abs_pct(bwd.d_attn_sink.float(), sink_ref.grad) <= 5.0


def _eager_sparse_mqa(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    topk: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    outs = []
    for batch in range(int(q.shape[0])):
        batch_out = []
        for query in range(int(q.shape[1])):
            selected = [
                int(index)
                for index in topk[batch, query].detach().cpu().tolist()
                if int(index) >= 0
            ]
            if not selected:
                batch_out.append(torch.zeros_like(q[batch, query]))
                continue
            kv_rows = kv[batch, selected]
            real_logits = torch.einsum("hd,kd->hk", q[batch, query], kv_rows) * scale
            logits = torch.cat((real_logits, sink.unsqueeze(-1)), dim=-1)
            probs = torch.softmax(logits, dim=-1)[..., : len(selected)]
            batch_out.append(torch.einsum("hk,kd->hd", probs, kv_rows))
        outs.append(torch.stack(batch_out))
    return torch.stack(outs)


def _mean_abs_pct(candidate: torch.Tensor, target: torch.Tensor) -> float:
    denominator = target.abs().mean().clamp_min(1e-8)
    return float(((candidate - target).abs().mean() / denominator * 100).item())
