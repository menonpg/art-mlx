from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.70710678118654752440))


@triton.jit
def _gelu_grad(x):
    cdf = 0.5 * (1.0 + tl.erf(x * 0.70710678118654752440))
    pdf = 0.39894228040143267794 * tl.exp(-0.5 * x * x)
    return cdf + x * pdf


@triton.jit
def _conv_gelu_fwd_kernel(
    qkv,
    conv_initial,
    weight,
    bias,
    lengths,
    out,
    final,
    C: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUTPUT_FINAL: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    b = tl.program_id(2)
    tail: tl.constexpr = K - 1
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c = offs_c[:, None]
    t = offs_t[None, :]
    mask = (offs_c[:, None] < C) & (offs_t[None, :] < T)
    acc = tl.zeros((BLOCK_C, BLOCK_T), dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias + offs_c, mask=offs_c < C, other=0.0)[:, None].to(
            tl.float32
        )
    for j in tl.static_range(0, K):
        ext = t + j
        from_initial = ext < tail
        init_idx = (b * C + c) * tail + ext
        qkv_idx = (b * C + c) * T + (ext - tail)
        x_init = tl.load(conv_initial + init_idx, mask=mask & from_initial, other=0.0)
        x_qkv = tl.load(qkv + qkv_idx, mask=mask & ~from_initial, other=0.0)
        w = tl.load(weight + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += (x_init + x_qkv).to(tl.float32) * w[:, None]
    tl.store(out + (b * C + c) * T + t, _gelu(acc), mask=mask)

    if OUTPUT_FINAL:
        length = tl.load(lengths + b)
        for r in tl.static_range(0, tail):
            ext = length + r
            from_initial = ext < tail
            init_idx = (b * C + offs_c) * tail + ext
            qkv_idx = (b * C + offs_c) * T + (ext - tail)
            x_init = tl.load(
                conv_initial + init_idx,
                mask=(pid_t == 0) & (offs_c < C) & from_initial,
                other=0.0,
            )
            x_qkv = tl.load(
                qkv + qkv_idx,
                mask=(pid_t == 0) & (offs_c < C) & ~from_initial,
                other=0.0,
            )
            tl.store(
                final + (b * C + offs_c) * tail + r,
                x_init + x_qkv,
                mask=(pid_t == 0) & (offs_c < C),
            )


@triton.jit
def _conv_gelu_grad_preact_kernel(
    qkv,
    conv_initial,
    weight,
    bias,
    grad_out,
    grad_preact,
    C: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    b = tl.program_id(2)
    tail: tl.constexpr = K - 1
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c = offs_c[:, None]
    t = offs_t[None, :]
    mask = (offs_c[:, None] < C) & (offs_t[None, :] < T)
    acc = tl.zeros((BLOCK_C, BLOCK_T), dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias + offs_c, mask=offs_c < C, other=0.0)[:, None].to(
            tl.float32
        )
    for j in tl.static_range(0, K):
        ext = t + j
        from_initial = ext < tail
        init_idx = (b * C + c) * tail + ext
        qkv_idx = (b * C + c) * T + (ext - tail)
        x_init = tl.load(conv_initial + init_idx, mask=mask & from_initial, other=0.0)
        x_qkv = tl.load(qkv + qkv_idx, mask=mask & ~from_initial, other=0.0)
        w = tl.load(weight + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += (x_init + x_qkv).to(tl.float32) * w[:, None]
    go = tl.load(grad_out + (b * C + c) * T + t, mask=mask, other=0.0).to(tl.float32)
    tl.store(grad_preact + (b * C + c) * T + t, go * _gelu_grad(acc), mask=mask)


@triton.jit
def _conv_gelu_bwd_input_kernel(
    grad_preact,
    weight,
    lengths,
    grad_final,
    grad_qkv,
    grad_initial,
    C: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    HAS_FINAL_GRAD: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_c = tl.program_id(1)
    b = tl.program_id(2)
    tail: tl.constexpr = K - 1
    ext_len: tl.constexpr = T + K - 1
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    c = offs_c[:, None]
    e = offs_e[None, :]
    mask = (offs_c[:, None] < C) & (offs_e[None, :] < ext_len)
    acc = tl.zeros((BLOCK_C, BLOCK_E), dtype=tl.float32)
    for j in tl.static_range(0, K):
        t = e - j
        valid = mask & (t >= 0) & (t < T)
        gz = tl.load(grad_preact + (b * C + c) * T + t, mask=valid, other=0.0)
        w = tl.load(weight + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += gz.to(tl.float32) * w[:, None]
    if HAS_FINAL_GRAD:
        length = tl.load(lengths + b)
        r = e - length
        valid_final = mask & (r >= 0) & (r < tail)
        gf = tl.load(
            grad_final + (b * C + c) * tail + r,
            mask=valid_final,
            other=0.0,
        )
        acc += gf.to(tl.float32)

    init_mask = mask & (e < tail)
    qkv_mask = mask & (e >= tail)
    tl.store(grad_initial + (b * C + c) * tail + e, acc, mask=init_mask)
    tl.store(grad_qkv + (b * C + c) * T + (e - tail), acc, mask=qkv_mask)


@triton.jit
def _conv_gelu_bwd_weight_kernel(
    qkv,
    conv_initial,
    grad_preact,
    grad_weight,
    grad_bias,
    C: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_BT: tl.constexpr,
):
    c = tl.program_id(0)
    tail: tl.constexpr = K - 1
    bt_total: tl.constexpr = B * T
    offsets = tl.arange(0, BLOCK_BT)
    bias_acc = tl.zeros((BLOCK_BT,), dtype=tl.float32)
    for j in tl.static_range(0, K):
        weight_acc = tl.zeros((BLOCK_BT,), dtype=tl.float32)
        for start in range(0, bt_total, BLOCK_BT):
            bt = start + offsets
            mask = bt < bt_total
            b = bt // T
            t = bt - b * T
            gz = tl.load(grad_preact + (b * C + c) * T + t, mask=mask, other=0.0)
            ext = t + j
            from_initial = ext < tail
            init_idx = (b * C + c) * tail + ext
            qkv_idx = (b * C + c) * T + (ext - tail)
            x_init = tl.load(
                conv_initial + init_idx, mask=mask & from_initial, other=0.0
            )
            x_qkv = tl.load(qkv + qkv_idx, mask=mask & ~from_initial, other=0.0)
            weight_acc += gz.to(tl.float32) * (x_init + x_qkv).to(tl.float32)
            if HAS_BIAS and j == 0:
                bias_acc += gz.to(tl.float32)
        tl.store(grad_weight + c * K + j, tl.sum(weight_acc, axis=0))
    if HAS_BIAS:
        tl.store(grad_bias + c, tl.sum(bias_acc, axis=0))


class _VarlenCausalConvGelu(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        qkv: Tensor,
        conv_initial: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        lengths: Tensor,
        output_final_state: bool,
    ) -> tuple[Tensor, Tensor | None]:
        _validate_inputs(qkv, conv_initial, weight, bias, lengths)
        qkv = qkv.contiguous()
        conv_initial = conv_initial.contiguous()
        weight = weight.contiguous()
        bias_tensor = (
            bias.contiguous()
            if bias is not None
            else torch.empty((0,), device=qkv.device, dtype=qkv.dtype)
        )
        lengths = lengths.contiguous()
        batch, channels, max_len = qkv.shape
        kernel_width = int(weight.shape[1])
        out = torch.empty_like(qkv)
        final = (
            torch.empty(
                (batch, channels, kernel_width - 1),
                device=qkv.device,
                dtype=qkv.dtype,
            )
            if output_final_state
            else None
        )
        block_c, block_t, num_warps = _tile_config(channels, max_len)
        grid = (triton.cdiv(max_len, block_t), triton.cdiv(channels, block_c), batch)
        _conv_gelu_fwd_kernel[grid](
            qkv,
            conv_initial,
            weight,
            bias_tensor,
            lengths,
            out,
            out if final is None else final,
            channels,
            max_len,
            kernel_width,
            HAS_BIAS=bias is not None,
            OUTPUT_FINAL=output_final_state,
            BLOCK_C=block_c,
            BLOCK_T=block_t,
            num_warps=num_warps,
        )
        ctx.save_for_backward(qkv, conv_initial, weight, bias_tensor, lengths)
        ctx.has_bias = bias is not None
        ctx.output_final_state = bool(output_final_state)
        ctx.tile = (block_c, block_t, num_warps)
        return out, final

    @staticmethod
    def backward(
        ctx: Any, grad_out: Tensor, grad_final: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, None, None]:
        qkv, conv_initial, weight, bias, lengths = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_final_tensor = (
            grad_final.contiguous()
            if grad_final is not None
            else torch.empty((0,), device=qkv.device, dtype=qkv.dtype)
        )
        batch, channels, max_len = qkv.shape
        kernel_width = int(weight.shape[1])
        grad_qkv = torch.empty_like(qkv)
        grad_initial = torch.empty_like(conv_initial)
        grad_weight = torch.empty_like(weight)
        grad_bias = torch.empty_like(bias) if bool(ctx.has_bias) else None
        grad_preact = torch.empty(qkv.shape, device=qkv.device, dtype=torch.float32)
        block_c, block_t, num_warps = ctx.tile
        grid_t = (
            triton.cdiv(max_len, block_t),
            triton.cdiv(channels, block_c),
            batch,
        )
        _conv_gelu_grad_preact_kernel[grid_t](
            qkv,
            conv_initial,
            weight,
            bias,
            grad_out,
            grad_preact,
            channels,
            max_len,
            kernel_width,
            HAS_BIAS=bool(ctx.has_bias),
            BLOCK_C=block_c,
            BLOCK_T=block_t,
            num_warps=num_warps,
        )
        ext_len = max_len + kernel_width - 1
        grid_e = (
            triton.cdiv(ext_len, block_t),
            triton.cdiv(channels, block_c),
            batch,
        )
        _conv_gelu_bwd_input_kernel[grid_e](
            grad_preact,
            weight,
            lengths,
            grad_final_tensor,
            grad_qkv,
            grad_initial,
            channels,
            max_len,
            kernel_width,
            HAS_FINAL_GRAD=grad_final is not None,
            BLOCK_C=block_c,
            BLOCK_E=block_t,
            num_warps=num_warps,
        )
        reduce_block = 256
        _conv_gelu_bwd_weight_kernel[(channels,)](
            qkv,
            conv_initial,
            grad_preact,
            grad_weight,
            grad_bias if grad_bias is not None else grad_weight,
            channels,
            batch,
            max_len,
            kernel_width,
            HAS_BIAS=bool(ctx.has_bias),
            BLOCK_BT=reduce_block,
            num_warps=8,
        )
        return grad_qkv, grad_initial, grad_weight, grad_bias, None, None


def varlen_causal_conv_gelu(
    qkv: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    lengths: Tensor,
    *,
    output_final_state: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Run ART GDN's prepared-varlen causal depthwise conv followed by GELU.

    Inputs use the existing prepared GDN layout: ``qkv`` is ``[segments, channels,
    max_len]`` with padded positions already zeroed, ``conv_initial`` is
    ``[segments, channels, kernel_width - 1]``, and ``lengths`` contains each
    segment's real token count. The dense output intentionally matches the
    current production conv path over the padded tensor; callers can keep using
    the existing real-token mask after this fused operation.
    """

    return _VarlenCausalConvGelu.apply(
        qkv, conv_initial, weight, bias, lengths, output_final_state
    )


def gdn_varlen_causal_conv_gelu(
    gdn: Any,
    qkv: Tensor,
    conv_initial: Tensor,
    lengths: Tensor,
    *,
    output_final_state: bool = True,
) -> tuple[Tensor, Tensor | None]:
    if str(getattr(gdn, "activation", "")) != "gelu":
        raise ValueError(
            "fused varlen causal conv is only defined for GDN GELU activation, "
            f"got {getattr(gdn, 'activation', None)!r}"
        )
    return varlen_causal_conv_gelu(
        qkv,
        conv_initial,
        gdn.conv1d.weight.squeeze(1),
        gdn.conv1d.bias,
        lengths,
        output_final_state=output_final_state,
    )


def _tile_config(channels: int, max_len: int) -> tuple[int, int, int]:
    del channels
    if max_len >= 512:
        return 2, 128, 4
    return 4, 64, 4


def _validate_inputs(
    qkv: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    lengths: Tensor,
) -> None:
    if not qkv.is_cuda:
        raise ValueError("qkv must be a CUDA tensor")
    if qkv.ndim != 3:
        raise ValueError(f"qkv must be [segments, channels, max_len], got {qkv.shape}")
    if conv_initial.ndim != 3:
        raise ValueError(
            "conv_initial must be [segments, channels, kernel_width - 1], "
            f"got {conv_initial.shape}"
        )
    if weight.ndim != 2:
        raise ValueError(f"weight must be [channels, kernel_width], got {weight.shape}")
    batch, channels, _ = qkv.shape
    kernel_width = int(weight.shape[1])
    if kernel_width < 1:
        raise ValueError("kernel_width must be at least 1")
    if tuple(conv_initial.shape) != (batch, channels, kernel_width - 1):
        raise ValueError(
            "conv_initial shape must match qkv and weight tail, got "
            f"qkv={tuple(qkv.shape)} conv_initial={tuple(conv_initial.shape)} "
            f"weight={tuple(weight.shape)}"
        )
    if int(weight.shape[0]) != channels:
        raise ValueError(
            f"weight channels {int(weight.shape[0])} must match qkv channels {channels}"
        )
    if bias is not None and tuple(bias.shape) != (channels,):
        raise ValueError(f"bias must be [channels], got {tuple(bias.shape)}")
    if tuple(lengths.shape) != (batch,):
        raise ValueError(f"lengths must be [segments], got {tuple(lengths.shape)}")
    if lengths.device != qkv.device:
        raise ValueError("lengths must be on the same CUDA device as qkv")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"lengths must be int32 or int64, got {lengths.dtype}")
    for name, tensor in (
        ("conv_initial", conv_initial),
        ("weight", weight),
        ("bias", bias),
    ):
        if tensor is not None and tensor.device != qkv.device:
            raise ValueError(f"{name} must be on the same CUDA device as qkv")
        if tensor is not None and tensor.dtype != qkv.dtype:
            raise ValueError(f"{name} dtype {tensor.dtype} must match qkv {qkv.dtype}")
