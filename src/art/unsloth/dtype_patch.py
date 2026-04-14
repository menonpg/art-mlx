"""Patch Unsloth's fused LoRA kernels to handle mixed bf16/fp16 dtypes.

On certain GPU accelerators (e.g. H200), base model activations run in bf16
while LoRA adapter weights remain in fp16. Unsloth's ``matmul_lora`` and
``fast_linear_forward`` call ``addmm_`` / ``addmv_`` which require matching
dtypes, causing a RuntimeError. This module patches those functions to cast
tensors to a common dtype before the fused ops.

Apply once at startup via :func:`ensure_dtype_patch`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

_PATCHED = False


def _cast_if_needed(tensor: Any, dtype: Any) -> Any:
    if tensor is None:
        return None
    if getattr(tensor, "dtype", None) == dtype:
        return tensor
    try:
        return tensor.to(dtype)
    except AttributeError:
        return tensor


def ensure_dtype_patch(log: logging.Logger | None = None) -> bool:
    """Patch Unsloth LoRA helpers for mixed-precision safety. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import torch
        import unsloth.kernels.utils as utils
    except ImportError:
        if log:
            log.debug("Unsloth not available; skipping dtype patch.")
        return False

    Float8Tensor = getattr(utils, "Float8Tensor", None)
    torch_matmul: Callable[..., Any] = utils.torch_matmul
    fast_dequantize: Callable[..., Any] = utils.fast_dequantize
    fp8_linear: Callable[..., Any] | None = getattr(utils, "fp8_linear", None)
    fast_gemv: Callable[..., Any] | None = getattr(utils, "fast_gemv", None)
    torch_mm: Callable[..., Any] = utils.torch_mm
    torch_mv: Callable[..., Any] = utils.torch_mv
    get_lora_parameters_bias: Callable[..., Any] = utils.get_lora_parameters_bias

    original_fast_linear_forward = utils.fast_linear_forward
    original_matmul_lora = utils.matmul_lora

    bf16 = torch.bfloat16

    def _target_dtype(out_tensor: Any, hidden_dtype: Any) -> Any:
        if hidden_dtype == bf16:
            return bf16
        if out_tensor is not None:
            return out_tensor.dtype
        return hidden_dtype

    def patched_matmul_lora(
        X: Any,
        W: Any,
        W_quant: Any,
        A: Any,
        B: Any,
        s: Any,
        out: Any = None,
    ) -> Any:
        dtype = X.dtype
        reshape = False
        if X.dim() == 3:
            batch, seq_len, _ = X.shape
            X = X.view(-1, X.shape[-1])
            reshape = True

        if Float8Tensor is not None and isinstance(W, Float8Tensor):
            if W.ndim != 2:
                raise ValueError("Expected 2D Float8Tensor for LoRA matmul.")
            if W.block_size[0] == W.shape[0] and W.block_size[1] == 1:
                W_full = W.dequantize()
            else:
                W_full = W.contiguous()
            out = torch_matmul(X, W_full.t(), out=out)
        elif getattr(W, "dtype", None) == getattr(torch, "float8_e4m3fn", None):
            if fp8_linear is None:
                raise RuntimeError("FP8 weights detected but fp8_linear unavailable.")
            out = fp8_linear(X, W, W_quant)
        else:
            W_full = fast_dequantize(W, W_quant, use_global_buffer=True)
            out = torch_matmul(X, W_full.t(), out=out)

        if A is not None:
            td = _target_dtype(out, dtype)
            XA = torch_matmul(_cast_if_needed(X, td), _cast_if_needed(A.t(), td))
            out = _cast_if_needed(out, td)
            out = out.addmm_(XA, _cast_if_needed(B.t(), td), alpha=s)

        return out.view(batch, seq_len, -1) if reshape else out

    def patched_fast_linear_forward(
        proj: Any, X: Any, temp_lora: Any = None, out: Any = None
    ) -> Any:
        W, W_quant, lora_A, lora_B, lora_S, bias = get_lora_parameters_bias(proj)
        bsz, q_len, in_dim = X.shape

        if q_len != 1:
            return patched_matmul_lora(X, W, W_quant, lora_A, lora_B, lora_S)

        if W_quant is None:
            out = torch_matmul(X, W.t(), out=out)
        elif getattr(W, "dtype", None) == getattr(torch, "float8_e4m3fn", None):
            if fp8_linear is None:
                raise RuntimeError("FP8 weights detected but fp8_linear unavailable.")
            out = fp8_linear(X, W, W_quant, bias)
        elif fast_gemv is not None and bsz == 1 and q_len == 1:
            out = fast_gemv(X, W, W_quant, out=out)
        else:
            W_full = fast_dequantize(W.t(), W_quant, use_global_buffer=True)
            out = torch_matmul(X, W_full, out=out)

        if lora_A is not None:
            td = _target_dtype(out, X.dtype)
            if (
                not hasattr(lora_A, "_fast_lora")
                or getattr(lora_A._fast_lora, "dtype", None) != td
            ):
                lora_A._fast_lora = lora_A.to(td)
                lora_B._fast_lora = lora_B.to(td)

            X_lora = _cast_if_needed(X, td)
            out = _cast_if_needed(out, td)
            out_dim = out.shape[2]

            if bsz == 1:
                out = out.view(out_dim)
                temp_lora = torch_mv(lora_A._fast_lora, X_lora.ravel(), out=temp_lora)
                out.addmv_(lora_B._fast_lora, temp_lora, alpha=lora_S)
                out = out.view(1, 1, out_dim)
            else:
                out = out.view(bsz, out_dim)
                temp_lora = torch_mm(
                    X_lora.view(bsz, in_dim),
                    lora_A._fast_lora.t(),
                    out=temp_lora,
                )
                out.addmm_(temp_lora, lora_B._fast_lora.t(), alpha=lora_S)
                out = out.view(bsz, 1, out_dim)

        if bias is not None:
            out = out + _cast_if_needed(bias, out.dtype)

        return out

    utils.matmul_lora = patched_matmul_lora  # type: ignore[assignment]
    utils.fast_linear_forward = patched_fast_linear_forward  # type: ignore[assignment]
    utils._original_fast_linear_forward = original_fast_linear_forward  # type: ignore[attr-defined]
    utils._original_matmul_lora = original_matmul_lora  # type: ignore[attr-defined]

    _PATCHED = True
    if log:
        log.debug("Applied Unsloth LoRA dtype harmonisation patch.")
    return True
