from __future__ import annotations

from collections.abc import Callable
import importlib
import math

import torch

from .types import Dsv4SparseBackwardResult, Dsv4SparseForwardResult

_LN2 = math.log(2.0)
_LOG2E = 1.0 / _LN2

_SparseFwd = Callable[..., tuple[torch.Tensor, torch.Tensor]]
_SparseBwd = Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def dsv4_sparse_fwd(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk: torch.Tensor,
    scale: float | None = None,
) -> Dsv4SparseForwardResult:
    """Run the Miles DSV4 list-sparse forward kernel.

    ART-facing LSE is natural log. Miles TileLang emits log2 LSE, so this
    wrapper is the single boundary where forward LSE is multiplied by ln(2).
    CP stage callers should pass `sink=-inf` and merge the real sink once after
    all real-key stages.
    """
    _validate_sparse_inputs(q=q, kv=kv, attn_sink=attn_sink, topk=topk)
    fwd, _ = _load_miles_sparse_mla()
    out, lse_log2 = fwd(
        q.contiguous(),
        kv.contiguous(),
        attn_sink.to(dtype=torch.float32).contiguous(),
        topk.to(dtype=torch.int32).contiguous(),
        sm_scale=scale,
    )
    return Dsv4SparseForwardResult(out=out, lse=lse_log2 * _LN2)


def dsv4_sparse_bwd(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk: torch.Tensor,
    global_out: torch.Tensor,
    grad_out: torch.Tensor,
    global_lse: torch.Tensor,
    scale: float | None = None,
) -> Dsv4SparseBackwardResult:
    """Replay one DSV4 sparse stage backward with global output and LSE.

    `global_lse` is natural log on the ART side and is divided by ln(2) before
    entering Miles backward. This is the CP replay path: pass the globally
    merged output/LSE, not the stage-local forward output/LSE.
    """
    _validate_sparse_inputs(q=q, kv=kv, attn_sink=attn_sink, topk=topk)
    if global_out.shape != q.shape or grad_out.shape != q.shape:
        raise RuntimeError(
            "DSV4 sparse backward global_out and grad_out must match q shape, got "
            f"q={tuple(q.shape)}, global_out={tuple(global_out.shape)}, "
            f"grad_out={tuple(grad_out.shape)}"
        )
    if global_lse.shape != q.shape[:-1]:
        raise RuntimeError(
            "DSV4 sparse backward global_lse must have shape [B,Q,H], got "
            f"{tuple(global_lse.shape)} for q={tuple(q.shape)}"
        )
    _require_same_device(
        q.device,
        kv=kv,
        attn_sink=attn_sink,
        topk=topk,
        global_out=global_out,
        grad_out=grad_out,
        global_lse=global_lse,
    )

    _, bwd = _load_miles_sparse_mla()
    dq, dkv, d_attn_sink = bwd(
        q.contiguous(),
        kv.contiguous(),
        attn_sink.to(dtype=torch.float32).contiguous(),
        global_out.contiguous(),
        grad_out.contiguous(),
        topk.to(dtype=torch.int32).contiguous(),
        (global_lse * _LOG2E).to(dtype=torch.float32).contiguous(),
        sm_scale=scale,
    )
    return Dsv4SparseBackwardResult(dq=dq, dkv=dkv, d_attn_sink=d_attn_sink)


def dsv4_disabled_attn_sink(attn_sink: torch.Tensor) -> torch.Tensor:
    return torch.full_like(attn_sink, float("-inf"))


def _validate_sparse_inputs(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk: torch.Tensor,
) -> None:
    if q.ndim != 4:
        raise RuntimeError(f"DSV4 sparse q must have shape [B,Q,H,D], got {q.shape}")
    if kv.ndim != 3:
        raise RuntimeError(f"DSV4 sparse kv must have shape [B,K,D], got {kv.shape}")
    if topk.ndim != 3:
        raise RuntimeError(
            f"DSV4 sparse topk must have shape [B,Q,L], got {topk.shape}"
        )
    if attn_sink.ndim != 1:
        raise RuntimeError(
            f"DSV4 sparse attn_sink must have shape [H], got {attn_sink.shape}"
        )
    if int(kv.shape[0]) != int(q.shape[0]) or int(kv.shape[-1]) != int(q.shape[-1]):
        raise RuntimeError(
            "DSV4 sparse q/kv batch and dim must match, got "
            f"q={tuple(q.shape)}, kv={tuple(kv.shape)}"
        )
    if tuple(topk.shape[:2]) != tuple(q.shape[:2]):
        raise RuntimeError(
            "DSV4 sparse topk batch/query shape must match q, got "
            f"topk={tuple(topk.shape)}, q={tuple(q.shape)}"
        )
    if int(attn_sink.shape[0]) != int(q.shape[2]):
        raise RuntimeError(
            "DSV4 sparse attn_sink head count must match q, got "
            f"sink={int(attn_sink.shape[0])}, q_heads={int(q.shape[2])}"
        )
    _require_same_device(q.device, kv=kv, attn_sink=attn_sink, topk=topk)


def _require_same_device(
    device: torch.device,
    **tensors: torch.Tensor,
) -> None:
    mismatched = {
        name: tensor.device
        for name, tensor in tensors.items()
        if tensor.device != device
    }
    if mismatched:
        raise RuntimeError(
            f"DSV4 sparse tensors must share device {device}, got {mismatched}"
        )


def _load_miles_sparse_mla() -> tuple[_SparseFwd, _SparseBwd]:
    fwd_module = importlib.import_module(
        "miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_fwd"
    )
    bwd_module = importlib.import_module(
        "miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_bwd"
    )
    return fwd_module.sparse_mqa_fwd_interface, bwd_module.sparse_mqa_bwd_interface
