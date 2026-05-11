from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import socket

import pytest
import torch
from torch import Tensor
from torch.distributed import destroy_process_group, init_process_group, is_initialized
import torch.nn.functional as F

from art.megatron.gdn.conv_gelu import (
    gdn_varlen_causal_conv_gelu,
    packed_varlen_causal_conv,
    varlen_causal_conv_gelu,
)
from art.megatron.gdn.gdn_shared_prefix import (
    GdnPlannerConfig,
    build_gdn_rank_execution_plan,
    parse_gdn_shared_prefix_segments,
)
from tests.integration.megatron.gdn_shared_prefix.benchmark_gdn import (
    make_qwen35_gdn_pair,
)
from tests.integration.megatron.gdn_shared_prefix.cases import (
    GdnFamilyShape,
    GdnPackedRowShape,
    GdnPhase0Case,
)
from tests.integration.megatron.gdn_shared_prefix.metrics import assert_mean_abs_pct
from tests.integration.megatron.gdn_shared_prefix.packed_layout import (
    build_phase0_packed_tensors,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_varlen_causal_conv_gelu_matches_reference_with_final_grads() -> None:
    _run_case(batch=3, channels=11, max_len=9, kernel_width=4, has_bias=True)


def test_varlen_causal_conv_gelu_matches_reference_without_bias() -> None:
    _run_case(batch=4, channels=7, max_len=6, kernel_width=3, has_bias=False)


def test_varlen_causal_conv_gelu_supports_unit_kernel() -> None:
    _run_case(batch=2, channels=5, max_len=8, kernel_width=1, has_bias=True)


def test_packed_varlen_causal_conv_gelu_matches_reference_with_final_grads() -> None:
    _run_packed_case(
        lengths=(1, 2, 4, 7),
        channels=9,
        kernel_width=4,
        has_bias=True,
        activation="gelu",
    )


def test_packed_varlen_causal_conv_gelu_matches_reference_without_bias() -> None:
    _run_packed_case(
        lengths=(1, 3, 5),
        channels=7,
        kernel_width=3,
        has_bias=False,
        activation="gelu",
    )


def test_packed_varlen_causal_conv_silu_and_swish_match_reference() -> None:
    for activation in ("silu", "swish"):
        _run_packed_case(
            lengths=(1, 4, 6),
            channels=5,
            kernel_width=5,
            has_bias=True,
            activation=activation,
        )


def test_packed_varlen_causal_conv_supports_unit_kernel() -> None:
    _run_packed_case(
        lengths=(1, 5),
        channels=4,
        kernel_width=1,
        has_bias=True,
        activation="none",
    )


def test_packed_varlen_causal_conv_rejects_unsupported_activation() -> None:
    conv_in, cu_seqlens, conv_initial, weight, bias, _, _ = _packed_inputs(
        lengths=(2,),
        channels=3,
        kernel_width=2,
        has_bias=True,
        seed=17,
    )
    with pytest.raises(ValueError, match="activation"):
        packed_varlen_causal_conv(
            conv_in,
            cu_seqlens,
            conv_initial,
            weight,
            bias,
            activation="relu",
        )


def test_gdn_varlen_causal_conv_gelu_matches_qwen_planner_bucket() -> None:
    case = GdnPhase0Case(
        name="conv_gelu_qwen_bucket",
        sequence_length=64,
        rows=(
            GdnPackedRowShape(
                families=(
                    GdnFamilyShape(prefix_length=7, suffix_lengths=(2, 6, 3)),
                    GdnFamilyShape(prefix_length=3, suffix_lengths=(8, 1)),
                )
            ),
        ),
        seed=61,
    )
    tensors = build_phase0_packed_tensors(case)
    spec = parse_gdn_shared_prefix_segments(
        tensors["group_ids"].cuda(),
        tensors["parent_ids"].cuda(),
        min_completions_per_family=1,
    )
    plan = build_gdn_rank_execution_plan(
        spec,
        device=torch.device("cuda"),
        planner_config=GdnPlannerConfig(max_padding_ratio=4.0),
    )
    bucket = plan.completion_with_prefix_tail_buckets[0]
    with _single_rank_model_parallel():
        ref_gdn, _ = make_qwen35_gdn_pair(
            params_dtype=torch.float32, linear_policy="noop"
        )
        ref_gdn.eval()
        qkv, conv_initial, _, _, out_grad, final_grad = _inputs(
            batch=int(bucket.segment_count),
            channels=int(ref_gdn.conv_dim_local_tp),
            max_len=int(bucket.length),
            kernel_width=int(ref_gdn.conv_kernel_dim),
            has_bias=True,
            seed=123,
        )
        qkv = qkv.masked_fill(~bucket.real_mask.transpose(0, 1).unsqueeze(1), 0)
        weight = ref_gdn.conv1d.weight.detach().squeeze(1).contiguous()
        bias = (
            None
            if ref_gdn.conv1d.bias is None
            else ref_gdn.conv1d.bias.detach().contiguous()
        )
        ref = _run_reference(
            qkv, conv_initial, weight, bias, bucket.lengths, out_grad, final_grad
        )
        ref_gdn.zero_grad(set_to_none=True)
        cand = _run_fused_gdn(
            ref_gdn, qkv, conv_initial, bucket.lengths, out_grad, final_grad
        )
        _assert_results_close(ref, cand)


def _run_case(
    *,
    batch: int,
    channels: int,
    max_len: int,
    kernel_width: int,
    has_bias: bool,
) -> None:
    qkv, conv_initial, weight, bias, out_grad, final_grad = _inputs(
        batch=batch,
        channels=channels,
        max_len=max_len,
        kernel_width=kernel_width,
        has_bias=has_bias,
        seed=kernel_width * 100 + channels,
    )
    lengths = torch.tensor(
        [max(1, max_len - (index * 2) % max_len) for index in range(batch)],
        device="cuda",
        dtype=torch.long,
    )
    qkv = qkv.masked_fill(
        ~(torch.arange(max_len, device="cuda")[None, :] < lengths[:, None]).unsqueeze(
            1
        ),
        0,
    )
    ref = _run_reference(qkv, conv_initial, weight, bias, lengths, out_grad, final_grad)
    cand = _run_fused(qkv, conv_initial, weight, bias, lengths, out_grad, final_grad)
    _assert_results_close(ref, cand)


def _run_packed_case(
    *,
    lengths: tuple[int, ...],
    channels: int,
    kernel_width: int,
    has_bias: bool,
    activation: str,
) -> None:
    inputs = _packed_inputs(
        lengths=lengths,
        channels=channels,
        kernel_width=kernel_width,
        has_bias=has_bias,
        seed=kernel_width * 100 + channels + len(lengths),
    )
    conv_in, cu_seqlens, conv_initial, weight, bias, out_grad, final_grad = inputs
    ref = _run_packed_reference(
        conv_in,
        cu_seqlens,
        conv_initial,
        weight,
        bias,
        out_grad,
        final_grad,
        activation=activation,
    )
    cand = _run_packed_fused(
        conv_in,
        cu_seqlens,
        conv_initial,
        weight,
        bias,
        out_grad,
        final_grad,
        activation=activation,
    )
    _assert_packed_results_close(ref, cand)


def _inputs(
    *,
    batch: int,
    channels: int,
    max_len: int,
    kernel_width: int,
    has_bias: bool,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor, Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(
        batch,
        channels,
        max_len,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    conv_initial = torch.randn(
        batch,
        channels,
        kernel_width - 1,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    weight = torch.randn(
        channels, kernel_width, device="cuda", dtype=torch.float32, generator=generator
    )
    bias = (
        torch.randn(channels, device="cuda", dtype=torch.float32, generator=generator)
        if has_bias
        else None
    )
    out_grad = torch.randn_like(qkv, generator=generator)
    final_grad = torch.randn_like(conv_initial, generator=generator)
    return qkv, conv_initial, weight, bias, out_grad, final_grad


def _packed_inputs(
    *,
    lengths: tuple[int, ...],
    channels: int,
    kernel_width: int,
    has_bias: bool,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor, Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    cu_values = [0]
    for length in lengths:
        cu_values.append(cu_values[-1] + length)
    total_tokens = cu_values[-1]
    conv_in = torch.randn(
        total_tokens,
        channels,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    conv_initial = torch.randn(
        len(lengths),
        channels,
        kernel_width - 1,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    weight = torch.randn(
        channels, kernel_width, device="cuda", dtype=torch.float32, generator=generator
    )
    bias = (
        torch.randn(channels, device="cuda", dtype=torch.float32, generator=generator)
        if has_bias
        else None
    )
    out_grad = torch.randn(
        total_tokens,
        channels,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    final_grad = torch.randn(
        len(lengths),
        channels,
        kernel_width - 1,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    cu_seqlens = torch.tensor(cu_values, device="cuda", dtype=torch.int32)
    return conv_in, cu_seqlens, conv_initial, weight, bias, out_grad, final_grad


def _run_reference(
    qkv: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    lengths: Tensor,
    out_grad: Tensor,
    final_grad: Tensor,
) -> dict[str, Tensor | None]:
    qkv = qkv.detach().clone().requires_grad_(True)
    conv_initial = conv_initial.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    bias = None if bias is None else bias.detach().clone().requires_grad_(True)
    extended = torch.cat([conv_initial, qkv], dim=-1)
    out = F.conv1d(extended, weight.unsqueeze(1), bias, groups=qkv.shape[1])
    out = F.gelu(out).to(dtype=qkv.dtype)
    final = _reference_final(qkv, conv_initial, lengths)
    ((out * out_grad).sum() + (final * final_grad).sum()).backward()
    return _result(qkv, conv_initial, weight, bias, out, final)


def _run_packed_reference(
    conv_in: Tensor,
    cu_seqlens: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    out_grad: Tensor,
    final_grad: Tensor,
    *,
    activation: str,
) -> dict[str, Tensor | None]:
    conv_in = conv_in.detach().clone().requires_grad_(True)
    conv_initial = conv_initial.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    bias = None if bias is None else bias.detach().clone().requires_grad_(True)
    pieces = []
    for segment in range(int(cu_seqlens.numel()) - 1):
        start = int(cu_seqlens[segment].item())
        end = int(cu_seqlens[segment + 1].item())
        segment_in = conv_in[start:end].transpose(0, 1).unsqueeze(0)
        extended = torch.cat([conv_initial[segment : segment + 1], segment_in], dim=-1)
        out = F.conv1d(extended, weight.unsqueeze(1), bias, groups=conv_in.shape[1])
        pieces.append(_torch_activation(out.squeeze(0).transpose(0, 1), activation))
    out = torch.cat(pieces, dim=0)
    final = _packed_reference_final(conv_in, cu_seqlens, conv_initial)
    ((out * out_grad).sum() + (final * final_grad).sum()).backward()
    return _packed_result(conv_in, conv_initial, weight, bias, out, final)


def _run_fused(
    qkv: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    lengths: Tensor,
    out_grad: Tensor,
    final_grad: Tensor,
) -> dict[str, Tensor | None]:
    qkv = qkv.detach().clone().requires_grad_(True)
    conv_initial = conv_initial.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    bias = None if bias is None else bias.detach().clone().requires_grad_(True)
    out, final = varlen_causal_conv_gelu(
        qkv, conv_initial, weight, bias, lengths, output_final_state=True
    )
    assert final is not None
    ((out * out_grad).sum() + (final * final_grad).sum()).backward()
    return _result(qkv, conv_initial, weight, bias, out, final)


def _run_packed_fused(
    conv_in: Tensor,
    cu_seqlens: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    out_grad: Tensor,
    final_grad: Tensor,
    *,
    activation: str,
) -> dict[str, Tensor | None]:
    conv_in = conv_in.detach().clone().requires_grad_(True)
    conv_initial = conv_initial.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    bias = None if bias is None else bias.detach().clone().requires_grad_(True)
    out, final = packed_varlen_causal_conv(
        conv_in,
        cu_seqlens,
        conv_initial,
        weight,
        bias,
        activation=activation,
        output_final_state=True,
    )
    assert final is not None
    ((out * out_grad).sum() + (final * final_grad).sum()).backward()
    return _packed_result(conv_in, conv_initial, weight, bias, out, final)


def _run_fused_gdn(
    gdn: torch.nn.Module,
    qkv: Tensor,
    conv_initial: Tensor,
    lengths: Tensor,
    out_grad: Tensor,
    final_grad: Tensor,
) -> dict[str, Tensor | None]:
    qkv = qkv.detach().clone().requires_grad_(True)
    conv_initial = conv_initial.detach().clone().requires_grad_(True)
    out, final = gdn_varlen_causal_conv_gelu(
        gdn, qkv, conv_initial, lengths, output_final_state=True
    )
    assert final is not None
    ((out * out_grad).sum() + (final * final_grad).sum()).backward()
    return {
        "out": out.detach(),
        "final": final.detach(),
        "qkv_grad": qkv.grad.detach(),
        "conv_initial_grad": conv_initial.grad.detach(),
        "weight_grad": gdn.conv1d.weight.grad.detach().squeeze(1),
        "bias_grad": None if gdn.conv1d.bias is None else gdn.conv1d.bias.grad.detach(),
    }


def _reference_final(qkv: Tensor, conv_initial: Tensor, lengths: Tensor) -> Tensor:
    tail_width = int(conv_initial.shape[-1])
    if tail_width == 0:
        return conv_initial
    batch_size, _, max_len = qkv.shape
    arange = torch.arange(batch_size, device=qkv.device)
    pieces = []
    for tail_offset in range(tail_width):
        source = lengths - tail_width + tail_offset
        from_qkv = source >= 0
        qkv_index = source.clamp(min=0, max=max_len - 1)
        init_index = (source + tail_width).clamp(min=0, max=tail_width - 1)
        pieces.append(
            torch.where(
                from_qkv.unsqueeze(1),
                qkv[arange, :, qkv_index],
                conv_initial[arange, :, init_index],
            )
        )
    return torch.stack(pieces, dim=-1)


def _packed_reference_final(
    conv_in: Tensor, cu_seqlens: Tensor, conv_initial: Tensor
) -> Tensor:
    tail_width = int(conv_initial.shape[-1])
    if tail_width == 0:
        return conv_initial
    pieces = []
    for segment in range(int(cu_seqlens.numel()) - 1):
        start = int(cu_seqlens[segment].item())
        end = int(cu_seqlens[segment + 1].item())
        extended = torch.cat([conv_initial[segment], conv_in[start:end].T], dim=-1)
        length = end - start
        pieces.append(extended[:, length : length + tail_width])
    return torch.stack(pieces, dim=0)


def _torch_activation(tensor: Tensor, activation: str) -> Tensor:
    if activation == "none":
        return tensor
    if activation in ("silu", "swish"):
        return F.silu(tensor)
    if activation == "gelu":
        return F.gelu(tensor)
    raise ValueError(activation)


def _result(
    qkv: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    out: Tensor,
    final: Tensor,
) -> dict[str, Tensor | None]:
    return {
        "out": out.detach(),
        "final": final.detach(),
        "qkv_grad": qkv.grad.detach(),
        "conv_initial_grad": conv_initial.grad.detach(),
        "weight_grad": weight.grad.detach(),
        "bias_grad": None if bias is None else bias.grad.detach(),
    }


def _packed_result(
    conv_in: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    out: Tensor,
    final: Tensor,
) -> dict[str, Tensor | None]:
    return {
        "out": out.detach(),
        "final": final.detach(),
        "conv_in_grad": conv_in.grad.detach(),
        "conv_initial_grad": conv_initial.grad.detach(),
        "weight_grad": weight.grad.detach(),
        "bias_grad": None if bias is None else bias.grad.detach(),
    }


def _assert_results_close(
    reference: dict[str, Tensor | None], candidate: dict[str, Tensor | None]
) -> None:
    for name in ("out", "final", "qkv_grad", "conv_initial_grad", "weight_grad"):
        ref_tensor = reference[name]
        cand_tensor = candidate[name]
        assert ref_tensor is not None and cand_tensor is not None
        if ref_tensor.numel() > 0:
            assert torch.any(ref_tensor != 0), f"{name} reference is all zero"
        assert_mean_abs_pct(ref_tensor, cand_tensor, name)
    if reference["bias_grad"] is not None:
        assert candidate["bias_grad"] is not None
        assert_mean_abs_pct(reference["bias_grad"], candidate["bias_grad"], "bias_grad")


def _assert_packed_results_close(
    reference: dict[str, Tensor | None], candidate: dict[str, Tensor | None]
) -> None:
    for name in ("out", "final", "conv_in_grad", "conv_initial_grad", "weight_grad"):
        ref_tensor = reference[name]
        cand_tensor = candidate[name]
        assert ref_tensor is not None and cand_tensor is not None
        assert ref_tensor.dtype == torch.float32
        assert cand_tensor.dtype == torch.float32
        if ref_tensor.numel() > 0:
            assert torch.any(ref_tensor != 0), f"{name} reference is all zero"
        assert_mean_abs_pct(ref_tensor, cand_tensor, name)
    if reference["bias_grad"] is not None:
        assert candidate["bias_grad"] is not None
        assert reference["bias_grad"].dtype == torch.float32
        assert candidate["bias_grad"].dtype == torch.float32
        assert_mean_abs_pct(reference["bias_grad"], candidate["bias_grad"], "bias_grad")


@contextmanager
def _single_rank_model_parallel() -> Iterator[None]:
    from megatron.core import parallel_state as ps

    if is_initialized():
        raise RuntimeError("torch.distributed is already initialized")
    init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{_free_port()}",
        rank=0,
        world_size=1,
    )
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        yield
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        if is_initialized():
            destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
