from __future__ import annotations

from enum import IntEnum
from typing import Any

import torch
from torch import Tensor
import triton
import triton.language as tl


class PackedConvActivation(IntEnum):
    NONE = 0
    SILU = 1
    SWISH = 1
    GELU = 2


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.70710678118654752440))


@triton.jit
def _gelu_grad(x):
    cdf = 0.5 * (1.0 + tl.erf(x * 0.70710678118654752440))
    pdf = 0.39894228040143267794 * tl.exp(-0.5 * x * x)
    return cdf + x * pdf


@triton.jit
def _apply_activation(x, ACTIVATION: tl.constexpr):
    if ACTIVATION == 0:
        return x
    if ACTIVATION == 1:
        sigmoid = tl.sigmoid(x)
        return x * sigmoid
    return _gelu(x)


@triton.jit
def _activation_grad(x, ACTIVATION: tl.constexpr):
    if ACTIVATION == 0:
        return x * 0.0 + 1.0
    if ACTIVATION == 1:
        sigmoid = tl.sigmoid(x)
        return sigmoid + x * sigmoid * (1.0 - sigmoid)
    return _gelu_grad(x)


@triton.jit(do_not_specialize=["SEGMENTS"])
def _segment_for_token(
    cu_seqlens,
    token,
    SEGMENTS,
    SEARCH_STEPS: tl.constexpr,
):
    lo = tl.zeros(token.shape, dtype=tl.int64)
    hi = lo + SEGMENTS.to(tl.int64) - 1
    for _ in tl.static_range(0, SEARCH_STEPS):
        mid = (lo + hi + 1) // 2
        mid_start = tl.load(cu_seqlens + mid)
        take_upper = mid_start <= token
        lo = tl.where(take_upper, mid, lo)
        hi = tl.where(take_upper, hi, mid - 1)
    return lo


@triton.jit(do_not_specialize=["TOTAL_TOKENS", "SEGMENTS"])
def _packed_conv_token_metadata_kernel(
    cu_seqlens,
    token_segment,
    token_local_t,
    TOTAL_TOKENS,
    SEGMENTS,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    token = offs_n.to(tl.int64)
    mask = offs_n < TOTAL_TOKENS
    segment = _segment_for_token(cu_seqlens, token, SEGMENTS, SEARCH_STEPS)
    start = tl.load(cu_seqlens + segment).to(tl.int64)
    tl.store(token_segment + token, segment, mask=mask)
    tl.store(token_local_t + token, token - start, mask=mask)


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
    b64 = b.to(tl.int64)
    c64 = c.to(tl.int64)
    t64 = t.to(tl.int64)
    offs_c64 = offs_c.to(tl.int64)
    mask = (offs_c[:, None] < C) & (offs_t[None, :] < T)
    acc = tl.zeros((BLOCK_C, BLOCK_T), dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias + offs_c, mask=offs_c < C, other=0.0)[:, None].to(
            tl.float32
        )
    for j in tl.static_range(0, K):
        ext = t + j
        ext64 = ext.to(tl.int64)
        from_initial = ext < tail
        init_idx = (b64 * C + c64) * tail + ext64
        qkv_idx = (b64 * C + c64) * T + (ext64 - tail)
        x_init = tl.load(conv_initial + init_idx, mask=mask & from_initial, other=0.0)
        x_qkv = tl.load(qkv + qkv_idx, mask=mask & ~from_initial, other=0.0)
        w = tl.load(weight + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += (x_init + x_qkv).to(tl.float32) * w[:, None]
    tl.store(out + (b64 * C + c64) * T + t64, _gelu(acc), mask=mask)

    if OUTPUT_FINAL:
        length = tl.load(lengths + b)
        for r in tl.static_range(0, tail):
            ext = length + r
            ext64 = ext.to(tl.int64)
            from_initial = ext < tail
            init_idx = (b64 * C + offs_c64) * tail + ext64
            qkv_idx = (b64 * C + offs_c64) * T + (ext64 - tail)
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
                final + (b64 * C + offs_c64) * tail + r,
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
    b64 = b.to(tl.int64)
    c64 = c.to(tl.int64)
    t64 = t.to(tl.int64)
    mask = (offs_c[:, None] < C) & (offs_t[None, :] < T)
    acc = tl.zeros((BLOCK_C, BLOCK_T), dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias + offs_c, mask=offs_c < C, other=0.0)[:, None].to(
            tl.float32
        )
    for j in tl.static_range(0, K):
        ext = t + j
        ext64 = ext.to(tl.int64)
        from_initial = ext < tail
        init_idx = (b64 * C + c64) * tail + ext64
        qkv_idx = (b64 * C + c64) * T + (ext64 - tail)
        x_init = tl.load(conv_initial + init_idx, mask=mask & from_initial, other=0.0)
        x_qkv = tl.load(qkv + qkv_idx, mask=mask & ~from_initial, other=0.0)
        w = tl.load(weight + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += (x_init + x_qkv).to(tl.float32) * w[:, None]
    out_idx = (b64 * C + c64) * T + t64
    go = tl.load(grad_out + out_idx, mask=mask, other=0.0).to(tl.float32)
    tl.store(grad_preact + out_idx, go * _gelu_grad(acc), mask=mask)


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
    b64 = b.to(tl.int64)
    c64 = c.to(tl.int64)
    e64 = e.to(tl.int64)
    mask = (offs_c[:, None] < C) & (offs_e[None, :] < ext_len)
    acc = tl.zeros((BLOCK_C, BLOCK_E), dtype=tl.float32)
    for j in tl.static_range(0, K):
        t = e - j
        t64 = t.to(tl.int64)
        valid = mask & (t >= 0) & (t < T)
        gz = tl.load(grad_preact + (b64 * C + c64) * T + t64, mask=valid, other=0.0)
        w = tl.load(weight + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += gz.to(tl.float32) * w[:, None]
    if HAS_FINAL_GRAD:
        length = tl.load(lengths + b)
        r = e - length
        r64 = r.to(tl.int64)
        valid_final = mask & (r >= 0) & (r < tail)
        gf = tl.load(
            grad_final + (b64 * C + c64) * tail + r64,
            mask=valid_final,
            other=0.0,
        )
        acc += gf.to(tl.float32)

    init_mask = mask & (e < tail)
    qkv_mask = mask & (e >= tail)
    tl.store(grad_initial + (b64 * C + c64) * tail + e64, acc, mask=init_mask)
    tl.store(grad_qkv + (b64 * C + c64) * T + (e64 - tail), acc, mask=qkv_mask)


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
            b64 = b.to(tl.int64)
            t64 = t.to(tl.int64)
            c64 = c.to(tl.int64)
            gz = tl.load(grad_preact + (b64 * C + c64) * T + t64, mask=mask, other=0.0)
            ext = t + j
            ext64 = ext.to(tl.int64)
            from_initial = ext < tail
            init_idx = (b64 * C + c64) * tail + ext64
            qkv_idx = (b64 * C + c64) * T + (ext64 - tail)
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


@triton.jit(do_not_specialize=["TOTAL_TOKENS"])
def _packed_conv_fwd_kernel(
    conv_in,
    token_segment,
    token_local_t,
    conv_initial,
    weight,
    bias,
    out,
    C: tl.constexpr,
    TOTAL_TOKENS,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    tail: tl.constexpr = K - 1
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    token = offs_n.to(tl.int64)
    segment = tl.load(token_segment + token, mask=offs_n < TOTAL_TOKENS, other=0).to(
        tl.int64
    )
    local_t = tl.load(token_local_t + token, mask=offs_n < TOTAL_TOKENS, other=0).to(
        tl.int64
    )
    n = offs_n[:, None].to(tl.int64)
    c = offs_c[None, :].to(tl.int64)
    segment_bc = segment[:, None].to(tl.int64)
    local_t_bc = local_t[:, None]
    mask = (offs_n[:, None] < TOTAL_TOKENS) & (offs_c[None, :] < C)
    acc = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias + offs_c, mask=offs_c < C, other=0.0)[None, :].to(
            tl.float32
        )
    for j in tl.static_range(0, K):
        ext = local_t_bc + j
        from_initial = ext < tail
        init_idx = (segment_bc * C + c) * tail + ext
        in_idx = (n + j - tail) * C + c
        x_init = tl.load(conv_initial + init_idx, mask=mask & from_initial, other=0.0)
        x_in = tl.load(conv_in + in_idx, mask=mask & ~from_initial, other=0.0)
        w = tl.load(weight + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += (x_init + x_in).to(tl.float32) * w[None, :]
    tl.store(out + n * C + c, _apply_activation(acc, ACTIVATION), mask=mask)


@triton.jit
def _packed_conv_final_kernel(
    conv_in,
    cu_seqlens,
    conv_initial,
    final,
    C: tl.constexpr,
    K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    segment = tl.program_id(2)
    tail: tl.constexpr = K - 1
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    start = tl.load(cu_seqlens + segment).to(tl.int64)
    end = tl.load(cu_seqlens + segment + 1).to(tl.int64)
    length = end - start
    r = offs_r[:, None].to(tl.int64)
    c = offs_c[None, :].to(tl.int64)
    ext = length + r
    from_initial = ext < tail
    mask = (offs_r[:, None] < tail) & (offs_c[None, :] < C)
    init_idx = (segment.to(tl.int64) * C + c) * tail + ext
    in_idx = (start + ext - tail) * C + c
    x_init = tl.load(conv_initial + init_idx, mask=mask & from_initial, other=0.0)
    x_in = tl.load(conv_in + in_idx, mask=mask & ~from_initial, other=0.0)
    tl.store(
        final + (segment.to(tl.int64) * C + c) * tail + r,
        x_init + x_in,
        mask=mask,
    )


@triton.jit(do_not_specialize=["TOTAL_TOKENS"])
def _packed_conv_grad_preact_weight_partial_kernel(
    conv_in,
    token_segment,
    token_local_t,
    conv_initial,
    weight,
    bias,
    grad_out,
    grad_preact,
    grad_weight_partial,
    grad_bias_partial,
    C: tl.constexpr,
    TOTAL_TOKENS,
    CHANNEL_TILES: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    tail: tl.constexpr = K - 1
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    token = offs_n.to(tl.int64)
    segment = tl.load(token_segment + token, mask=offs_n < TOTAL_TOKENS, other=0).to(
        tl.int64
    )
    local_t = tl.load(token_local_t + token, mask=offs_n < TOTAL_TOKENS, other=0).to(
        tl.int64
    )
    n = offs_n[:, None].to(tl.int64)
    c = offs_c[None, :].to(tl.int64)
    segment_bc = segment[:, None].to(tl.int64)
    local_t_bc = local_t[:, None]
    mask = (offs_n[:, None] < TOTAL_TOKENS) & (offs_c[None, :] < C)
    acc = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias + offs_c, mask=offs_c < C, other=0.0)[None, :].to(
            tl.float32
        )
    for j in tl.static_range(0, K):
        ext = local_t_bc + j
        from_initial = ext < tail
        init_idx = (segment_bc * C + c) * tail + ext
        in_idx = (n + j - tail) * C + c
        x_init = tl.load(conv_initial + init_idx, mask=mask & from_initial, other=0.0)
        x_in = tl.load(conv_in + in_idx, mask=mask & ~from_initial, other=0.0)
        w = tl.load(weight + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += (x_init + x_in).to(tl.float32) * w[None, :]
    go = tl.load(grad_out + n * C + c, mask=mask, other=0.0).to(tl.float32)
    gz = go * _activation_grad(acc, ACTIVATION)
    tl.store(
        grad_preact + n * C + c,
        gz,
        mask=mask,
    )
    partial_base = (pid_n * CHANNEL_TILES + pid_c) * K * BLOCK_C
    partial_c = tl.arange(0, BLOCK_C)
    for j in tl.static_range(0, K):
        ext = local_t_bc + j
        from_initial = ext < tail
        init_idx = (segment_bc * C + c) * tail + ext
        in_idx = (n + j - tail) * C + c
        x_init = tl.load(conv_initial + init_idx, mask=mask & from_initial, other=0.0)
        x_in = tl.load(conv_in + in_idx, mask=mask & ~from_initial, other=0.0)
        weight_partial = tl.sum(gz * (x_init + x_in).to(tl.float32), axis=0)
        tl.store(
            grad_weight_partial + partial_base + j * BLOCK_C + partial_c,
            weight_partial,
            mask=offs_c < C,
        )
    if HAS_BIAS:
        bias_partial = tl.sum(gz, axis=0)
        tl.store(
            grad_bias_partial + (pid_n * CHANNEL_TILES + pid_c) * BLOCK_C + partial_c,
            bias_partial,
            mask=offs_c < C,
        )


@triton.jit(do_not_specialize=["TOTAL_TOKENS"])
def _packed_conv_bwd_input_kernel(
    cu_seqlens,
    token_segment,
    weight,
    grad_preact,
    grad_final,
    grad_conv_in,
    C: tl.constexpr,
    TOTAL_TOKENS,
    K: tl.constexpr,
    HAS_FINAL_GRAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    tail: tl.constexpr = K - 1
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    token = offs_n.to(tl.int64)
    segment = tl.load(token_segment + token, mask=offs_n < TOTAL_TOKENS, other=0).to(
        tl.int64
    )
    end = tl.load(cu_seqlens + segment + 1).to(tl.int64)
    out_token_base = token[:, None] + tail
    c = offs_c[None, :].to(tl.int64)
    mask = (offs_n[:, None] < TOTAL_TOKENS) & (offs_c[None, :] < C)
    acc = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
    for j in tl.static_range(0, K):
        out_token = out_token_base - j
        valid = mask & (out_token < end[:, None])
        gz = tl.load(
            grad_preact + out_token * C + c,
            mask=valid,
            other=0.0,
        )
        w = tl.load(weight + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += gz.to(tl.float32) * w[None, :]
    if HAS_FINAL_GRAD:
        r = out_token_base - end[:, None]
        valid_final = mask & (r >= 0) & (r < tail)
        gf = tl.load(
            grad_final + (segment[:, None].to(tl.int64) * C + c) * tail + r,
            mask=valid_final,
            other=0.0,
        )
        acc += gf.to(tl.float32)
    tl.store(grad_conv_in + token[:, None] * C + c, acc, mask=mask)


@triton.jit
def _packed_conv_bwd_initial_kernel(
    cu_seqlens,
    weight,
    grad_preact,
    grad_final,
    grad_initial,
    C: tl.constexpr,
    K: tl.constexpr,
    HAS_FINAL_GRAD: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    segment = tl.program_id(2)
    tail: tl.constexpr = K - 1
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    start = tl.load(cu_seqlens + segment).to(tl.int64)
    end = tl.load(cu_seqlens + segment + 1).to(tl.int64)
    length = end - start
    e = offs_r[:, None].to(tl.int64)
    c = offs_c[None, :].to(tl.int64)
    mask = (offs_r[:, None] < tail) & (offs_c[None, :] < C)
    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for j in tl.static_range(0, K):
        out_t = e - j
        valid = mask & (out_t >= 0) & (out_t < length)
        gz = tl.load(grad_preact + (start + out_t) * C + c, mask=valid, other=0.0)
        w = tl.load(weight + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += gz.to(tl.float32) * w[None, :]
    if HAS_FINAL_GRAD:
        r = e - length
        valid_final = mask & (r >= 0) & (r < tail)
        gf = tl.load(
            grad_final + (segment.to(tl.int64) * C + c) * tail + r,
            mask=valid_final,
            other=0.0,
        )
        acc += gf.to(tl.float32)
    tl.store(grad_initial + (segment.to(tl.int64) * C + c) * tail + e, acc, mask=mask)


@triton.jit(do_not_specialize=["TOKEN_TILES"])
def _packed_conv_bwd_weight_reduce_kernel(
    grad_weight_partial,
    grad_weight,
    C: tl.constexpr,
    TOKEN_TILES,
    CHANNEL_TILES: tl.constexpr,
    K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    pid_c = tl.program_id(0)
    j = tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    partial_c = tl.arange(0, BLOCK_C)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    weight_acc = tl.zeros((BLOCK_TILES, BLOCK_C), dtype=tl.float32)
    start_tile = 0
    while start_tile < TOKEN_TILES:
        tile = start_tile + tile_offsets
        partial_idx = (
            (tile[:, None] * CHANNEL_TILES + pid_c) * K + j
        ) * BLOCK_C + partial_c[None, :]
        weight_acc += tl.load(
            grad_weight_partial + partial_idx,
            mask=(tile[:, None] < TOKEN_TILES) & c_mask[None, :],
            other=0.0,
        )
        start_tile += BLOCK_TILES
    tl.store(grad_weight + offs_c * K + j, tl.sum(weight_acc, axis=0), mask=c_mask)


@triton.jit(do_not_specialize=["TOKEN_TILES"])
def _packed_conv_bwd_bias_reduce_kernel(
    grad_bias_partial,
    grad_bias,
    C: tl.constexpr,
    TOKEN_TILES,
    CHANNEL_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    pid_c = tl.program_id(0)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    partial_c = tl.arange(0, BLOCK_C)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    bias_acc = tl.zeros((BLOCK_TILES, BLOCK_C), dtype=tl.float32)
    start_tile = 0
    while start_tile < TOKEN_TILES:
        tile = start_tile + tile_offsets
        partial_idx = (tile[:, None] * CHANNEL_TILES + pid_c) * BLOCK_C + partial_c[
            None, :
        ]
        bias_acc += tl.load(
            grad_bias_partial + partial_idx,
            mask=(tile[:, None] < TOKEN_TILES) & c_mask[None, :],
            other=0.0,
        )
        start_tile += BLOCK_TILES
    tl.store(grad_bias + offs_c, tl.sum(bias_acc, axis=0), mask=c_mask)


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
        reduce_block = 1024
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


class _PackedVarlenCausalConv(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        conv_in: Tensor,
        cu_seqlens: Tensor,
        conv_initial: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        output_final_state: bool,
        activation: str | PackedConvActivation,
    ) -> tuple[Tensor, Tensor | None]:
        activation_code = _activation_code(activation)
        _validate_packed_inputs(conv_in, cu_seqlens, conv_initial, weight, bias)
        conv_in = conv_in.contiguous()
        cu_seqlens = cu_seqlens.contiguous()
        conv_initial = conv_initial.contiguous()
        weight = weight.contiguous()
        bias_tensor = (
            bias.contiguous()
            if bias is not None
            else torch.empty((0,), device=conv_in.device, dtype=conv_in.dtype)
        )
        _assert_valid_cu_seqlens(cu_seqlens, int(conv_in.shape[0]))
        total_tokens, channels = conv_in.shape
        segments = int(cu_seqlens.numel()) - 1
        kernel_width = int(weight.shape[1])
        out = torch.empty_like(conv_in)
        final = (
            torch.empty(
                (segments, channels, kernel_width - 1),
                device=conv_in.device,
                dtype=conv_in.dtype,
            )
            if output_final_state
            else None
        )
        block_n, block_c, num_warps = _packed_tile_config(channels)
        search_steps = _search_steps(segments)
        metadata_dtype = (
            torch.long
            if max(total_tokens, segments) > torch.iinfo(torch.int32).max
            else torch.int32
        )
        token_segment = torch.empty(
            (total_tokens,), device=conv_in.device, dtype=metadata_dtype
        )
        token_local_t = torch.empty_like(token_segment)
        if total_tokens > 0:
            metadata_block_n = 256
            _packed_conv_token_metadata_kernel[
                (triton.cdiv(total_tokens, metadata_block_n),)
            ](
                cu_seqlens,
                token_segment,
                token_local_t,
                total_tokens,
                segments,
                search_steps,
                BLOCK_N=metadata_block_n,
                num_warps=4,
            )
            _packed_conv_fwd_kernel[
                (triton.cdiv(total_tokens, block_n), triton.cdiv(channels, block_c))
            ](
                conv_in,
                token_segment,
                token_local_t,
                conv_initial,
                weight,
                bias_tensor,
                out,
                channels,
                total_tokens,
                kernel_width,
                HAS_BIAS=bias is not None,
                ACTIVATION=activation_code,
                BLOCK_N=block_n,
                BLOCK_C=block_c,
                num_warps=num_warps,
            )
        if final is not None and kernel_width > 1 and segments > 0:
            block_r = _tail_block(kernel_width - 1)
            _packed_conv_final_kernel[
                (
                    triton.cdiv(kernel_width - 1, block_r),
                    triton.cdiv(channels, block_c),
                    segments,
                )
            ](
                conv_in,
                cu_seqlens,
                conv_initial,
                final,
                channels,
                kernel_width,
                BLOCK_C=block_c,
                BLOCK_R=block_r,
                num_warps=num_warps,
            )
        ctx.save_for_backward(
            conv_in,
            cu_seqlens,
            token_segment,
            token_local_t,
            conv_initial,
            weight,
            bias_tensor,
        )
        ctx.has_bias = bias is not None
        ctx.has_final = bool(output_final_state)
        ctx.activation = activation_code
        ctx.tile = (block_n, block_c, num_warps)
        return out, final

    @staticmethod
    def backward(
        ctx: Any, grad_out: Tensor, grad_final: Tensor | None
    ) -> tuple[Tensor, None, Tensor, Tensor, Tensor | None, None, None]:
        (
            conv_in,
            cu_seqlens,
            token_segment,
            token_local_t,
            conv_initial,
            weight,
            bias,
        ) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_final_tensor = (
            grad_final.contiguous()
            if grad_final is not None
            else torch.empty((0,), device=conv_in.device, dtype=conv_in.dtype)
        )
        total_tokens, channels = conv_in.shape
        segments = int(cu_seqlens.numel()) - 1
        kernel_width = int(weight.shape[1])
        grad_conv_in = torch.empty_like(conv_in)
        grad_initial = torch.empty_like(conv_initial)
        grad_weight = torch.empty_like(weight)
        grad_bias = torch.empty_like(bias) if bool(ctx.has_bias) else None
        block_n, block_c, num_warps = ctx.tile
        grad_preact = torch.empty(
            conv_in.shape, device=conv_in.device, dtype=torch.float32
        )
        if total_tokens > 0:
            token_tiles = triton.cdiv(total_tokens, block_n)
            channel_tiles = triton.cdiv(channels, block_c)
            grad_weight_partial = torch.empty(
                (token_tiles, channel_tiles, kernel_width, block_c),
                device=conv_in.device,
                dtype=torch.float32,
            )
            grad_bias_partial = (
                torch.empty(
                    (token_tiles, channel_tiles, block_c),
                    device=conv_in.device,
                    dtype=torch.float32,
                )
                if bool(ctx.has_bias)
                else torch.empty((0,), device=conv_in.device, dtype=torch.float32)
            )
            grid_n = (
                token_tiles,
                channel_tiles,
            )
            _packed_conv_grad_preact_weight_partial_kernel[grid_n](
                conv_in,
                token_segment,
                token_local_t,
                conv_initial,
                weight,
                bias,
                grad_out,
                grad_preact,
                grad_weight_partial,
                grad_bias_partial,
                channels,
                total_tokens,
                channel_tiles,
                kernel_width,
                HAS_BIAS=bool(ctx.has_bias),
                ACTIVATION=int(ctx.activation),
                BLOCK_N=block_n,
                BLOCK_C=block_c,
                num_warps=num_warps,
            )
            _packed_conv_bwd_input_kernel[grid_n](
                cu_seqlens,
                token_segment,
                weight,
                grad_preact,
                grad_final_tensor,
                grad_conv_in,
                channels,
                total_tokens,
                kernel_width,
                HAS_FINAL_GRAD=grad_final is not None,
                BLOCK_N=block_n,
                BLOCK_C=block_c,
                num_warps=num_warps,
            )
            _packed_conv_bwd_weight_reduce_kernel[(channel_tiles, kernel_width)](
                grad_weight_partial,
                grad_weight,
                channels,
                token_tiles,
                channel_tiles,
                kernel_width,
                BLOCK_C=block_c,
                BLOCK_TILES=64,
                num_warps=4,
            )
            if grad_bias is not None:
                _packed_conv_bwd_bias_reduce_kernel[(channel_tiles,)](
                    grad_bias_partial,
                    grad_bias,
                    channels,
                    token_tiles,
                    channel_tiles,
                    BLOCK_C=block_c,
                    BLOCK_TILES=64,
                    num_warps=4,
                )
        else:
            grad_conv_in = torch.zeros_like(conv_in)
            grad_weight = torch.zeros_like(weight)
            if grad_bias is not None:
                grad_bias = torch.zeros_like(bias)
        if kernel_width > 1 and segments > 0:
            block_r = _tail_block(kernel_width - 1)
            _packed_conv_bwd_initial_kernel[
                (
                    triton.cdiv(kernel_width - 1, block_r),
                    triton.cdiv(channels, block_c),
                    segments,
                )
            ](
                cu_seqlens,
                weight,
                grad_preact,
                grad_final_tensor,
                grad_initial,
                channels,
                kernel_width,
                HAS_FINAL_GRAD=grad_final is not None,
                BLOCK_C=block_c,
                BLOCK_R=block_r,
                num_warps=num_warps,
            )
        else:
            grad_initial = torch.zeros_like(conv_initial)
        return grad_conv_in, None, grad_initial, grad_weight, grad_bias, None, None


def packed_varlen_causal_conv(
    conv_in: Tensor,
    cu_seqlens: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    activation: str | PackedConvActivation = PackedConvActivation.GELU,
    output_final_state: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Run packed-varlen causal depthwise conv over real tokens only.

    ``conv_in`` is compact ``[total_real_tokens, channels]`` data and
    ``cu_seqlens`` is the exclusive prefix sum for segment lengths. The returned
    output has the same compact token layout. ``conv_initial`` and the optional
    final state keep the recurrent tail layout ``[segments, channels, K - 1]``.
    """

    return _PackedVarlenCausalConv.apply(
        conv_in,
        cu_seqlens,
        conv_initial,
        weight,
        bias,
        output_final_state,
        activation,
    )


def packed_varlen_causal_conv_gelu(
    conv_in: Tensor,
    cu_seqlens: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    output_final_state: bool = True,
) -> tuple[Tensor, Tensor | None]:
    return packed_varlen_causal_conv(
        conv_in,
        cu_seqlens,
        conv_initial,
        weight,
        bias,
        activation=PackedConvActivation.GELU,
        output_final_state=output_final_state,
    )


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


def _packed_tile_config(channels: int) -> tuple[int, int, int]:
    del channels
    return 128, 16, 4


def _tail_block(tail: int) -> int:
    return max(1, min(16, 1 << (tail - 1).bit_length()))


def _search_steps(segments: int) -> int:
    return max(1, (segments - 1).bit_length())


def _activation_code(activation: str | PackedConvActivation) -> int:
    if isinstance(activation, PackedConvActivation):
        return int(activation)
    activation_key = str(activation).lower()
    if activation_key == "none":
        return int(PackedConvActivation.NONE)
    if activation_key in ("silu", "swish"):
        return int(PackedConvActivation.SILU)
    if activation_key == "gelu":
        return int(PackedConvActivation.GELU)
    raise ValueError(
        "packed varlen causal conv activation must be one of "
        "'none', 'silu', 'swish', or 'gelu'; got "
        f"{activation!r}"
    )


def _assert_valid_cu_seqlens(cu_seqlens: Tensor, total_tokens: int) -> None:
    torch._assert_async(cu_seqlens[0] == 0)
    torch._assert_async(cu_seqlens[-1] == total_tokens)
    if cu_seqlens.numel() > 1:
        torch._assert_async(torch.all(cu_seqlens[1:] >= cu_seqlens[:-1]))


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


def _validate_packed_inputs(
    conv_in: Tensor,
    cu_seqlens: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
) -> None:
    if not conv_in.is_cuda:
        raise ValueError("conv_in must be a CUDA tensor")
    if conv_in.ndim != 2:
        raise ValueError(
            f"conv_in must be [total_real_tokens, channels], got {conv_in.shape}"
        )
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be [segments + 1], got {cu_seqlens.shape}")
    if cu_seqlens.numel() < 1:
        raise ValueError("cu_seqlens must contain at least the leading zero")
    if cu_seqlens.device != conv_in.device:
        raise ValueError("cu_seqlens must be on the same CUDA device as conv_in")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}")
    if conv_initial.ndim != 3:
        raise ValueError(
            "conv_initial must be [segments, channels, kernel_width - 1], "
            f"got {conv_initial.shape}"
        )
    if weight.ndim != 2:
        raise ValueError(f"weight must be [channels, kernel_width], got {weight.shape}")
    total_tokens, channels = conv_in.shape
    segments = int(cu_seqlens.numel()) - 1
    if total_tokens > 0 and segments == 0:
        raise ValueError("cu_seqlens must describe at least one segment for conv_in")
    kernel_width = int(weight.shape[1])
    if kernel_width < 1:
        raise ValueError("kernel_width must be at least 1")
    if tuple(conv_initial.shape) != (segments, channels, kernel_width - 1):
        raise ValueError(
            "conv_initial shape must match conv_in, cu_seqlens, and weight tail, got "
            f"conv_in={tuple(conv_in.shape)} "
            f"cu_seqlens={tuple(cu_seqlens.shape)} "
            f"conv_initial={tuple(conv_initial.shape)} weight={tuple(weight.shape)}"
        )
    if int(weight.shape[0]) != channels:
        raise ValueError(
            f"weight channels {int(weight.shape[0])} must match conv_in channels "
            f"{channels}"
        )
    if bias is not None and tuple(bias.shape) != (channels,):
        raise ValueError(f"bias must be [channels], got {tuple(bias.shape)}")
    for name, tensor in (
        ("conv_initial", conv_initial),
        ("weight", weight),
        ("bias", bias),
    ):
        if tensor is not None and tensor.device != conv_in.device:
            raise ValueError(f"{name} must be on the same CUDA device as conv_in")
        if tensor is not None and tensor.dtype != conv_in.dtype:
            raise ValueError(
                f"{name} dtype {tensor.dtype} must match conv_in {conv_in.dtype}"
            )
