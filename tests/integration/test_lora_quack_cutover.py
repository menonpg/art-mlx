from __future__ import annotations

import pytest
import torch

pytest.importorskip("quack")

from art.megatron.kernels.cute_grouped_lora_quack import quack_grouped_lora_dual
from art.megatron.lora import LoRA


def _require_grad(grad: torch.Tensor | None, *, name: str) -> torch.Tensor:
    if grad is None:
        raise AssertionError(f"{name}.grad unexpectedly None")
    return grad


def _eager_grouped_lora(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b_t: torch.Tensor,
    counts: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    start = 0
    for expert_idx, token_count in enumerate(counts.tolist()):
        if token_count == 0:
            continue
        stop = start + int(token_count)
        outputs.append(x[start:stop] @ a_t[expert_idx] @ b_t[expert_idx])
        start = stop
    if start != x.shape[0]:
        raise RuntimeError(
            f"Grouped split mismatch: consumed {start} rows for shape {tuple(x.shape)}"
        )
    return torch.cat(outputs, dim=0) * scale


def _eager_grouped_lora_dual(
    x: torch.Tensor,
    gate_a_t: torch.Tensor,
    gate_b_t: torch.Tensor,
    up_a_t: torch.Tensor,
    up_b_t: torch.Tensor,
    counts: torch.Tensor,
    *,
    scale_gate: float,
    scale_up: float,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    start = 0
    for expert_idx, token_count in enumerate(counts.tolist()):
        if token_count == 0:
            continue
        stop = start + int(token_count)
        gate_out = x[start:stop] @ gate_a_t[expert_idx] @ gate_b_t[expert_idx]
        up_out = x[start:stop] @ up_a_t[expert_idx] @ up_b_t[expert_idx]
        outputs.append(torch.cat((gate_out * scale_gate, up_out * scale_up), dim=1))
        start = stop
    if start != x.shape[0]:
        raise RuntimeError(
            f"Grouped split mismatch: consumed {start} rows for shape {tuple(x.shape)}"
        )
    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("rank", [1, 3, 7, 16, 24])
def test_lora_grouped_forward_cutover_matches_reference(rank: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the LoRA QuACK cutover test.")

    device = torch.device("cuda:0")
    torch.manual_seed(20260323 + rank)

    lora = LoRA(
        adapter_model_prefix="test.{expert}",
        in_features=64,
        out_features=64,
        rank=rank,
        alpha=32,
        dtype=torch.bfloat16,
        device=device,
        num_local_experts=4,
    )
    with torch.no_grad():
        lora.A_T.copy_(torch.randn_like(lora.A_T) * 0.05)
        lora.B_T.copy_(torch.randn_like(lora.B_T) * 0.05)

    counts = torch.tensor([32, 0, 16, 24], dtype=torch.int64)
    total_tokens = int(counts.sum().item())
    x = torch.randn(total_tokens, 64, device=device, dtype=torch.bfloat16) * 0.05
    loss_grad = torch.randn(total_tokens, 64, device=device, dtype=torch.bfloat16)

    x_ref = x.detach().clone().requires_grad_(True)
    a_ref = lora.A_T.detach().clone().requires_grad_(True)
    b_ref = lora.B_T.detach().clone().requires_grad_(True)
    ref_out = _eager_grouped_lora(
        x_ref,
        a_ref,
        b_ref,
        counts,
        scale=lora.scale,
    )
    ref_loss = (ref_out.float() * loss_grad.float()).sum() / max(1, loss_grad.numel())
    ref_loss.backward()

    x_test = x.detach().clone().requires_grad_(True)
    lora.zero_grad(set_to_none=True)
    got_out = lora(x_test, tokens_per_expert=counts)
    got_loss = (got_out.float() * loss_grad.float()).sum() / max(1, loss_grad.numel())
    got_loss.backward()

    x_ref_grad = _require_grad(x_ref.grad, name="x_ref")
    x_test_grad = _require_grad(x_test.grad, name="x_test")
    a_ref_grad = _require_grad(a_ref.grad, name="a_ref")
    a_test_grad = _require_grad(lora.A_T.grad, name="lora.A_T")
    b_ref_grad = _require_grad(b_ref.grad, name="b_ref")
    b_test_grad = _require_grad(lora.B_T.grad, name="lora.B_T")

    assert torch.allclose(ref_out, got_out.detach(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(x_ref_grad, x_test_grad, atol=5e-2, rtol=5e-2)
    assert torch.allclose(a_ref_grad, a_test_grad, atol=5e-2, rtol=5e-2)
    assert torch.allclose(b_ref_grad, b_test_grad, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("rank", [1, 3, 7, 16, 24])
def test_lora_grouped_dual_forward_cutover_matches_reference(rank: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the LoRA QuACK cutover test.")

    device = torch.device("cuda:0")
    torch.manual_seed(20260324 + rank)

    counts = torch.tensor([32, 0, 16, 24], dtype=torch.int64)
    total_tokens = int(counts.sum().item())
    x = torch.randn(total_tokens, 64, device=device, dtype=torch.bfloat16) * 0.05
    gate_a_t = torch.randn(4, 64, rank, device=device, dtype=torch.bfloat16) * 0.05
    gate_b_t = torch.randn(4, rank, 64, device=device, dtype=torch.bfloat16) * 0.05
    up_a_t = torch.randn(4, 64, rank, device=device, dtype=torch.bfloat16) * 0.05
    up_b_t = torch.randn(4, rank, 64, device=device, dtype=torch.bfloat16) * 0.05
    loss_grad = torch.randn(total_tokens, 128, device=device, dtype=torch.bfloat16)
    scale_gate = 2.0
    scale_up = 3.0

    x_ref = x.detach().clone().requires_grad_(True)
    gate_a_ref = gate_a_t.detach().clone().requires_grad_(True)
    gate_b_ref = gate_b_t.detach().clone().requires_grad_(True)
    up_a_ref = up_a_t.detach().clone().requires_grad_(True)
    up_b_ref = up_b_t.detach().clone().requires_grad_(True)
    ref_out = _eager_grouped_lora_dual(
        x_ref,
        gate_a_ref,
        gate_b_ref,
        up_a_ref,
        up_b_ref,
        counts,
        scale_gate=scale_gate,
        scale_up=scale_up,
    )
    ref_loss = (ref_out.float() * loss_grad.float()).sum() / max(1, loss_grad.numel())
    ref_loss.backward()

    x_test = x.detach().clone().requires_grad_(True)
    gate_a_test = gate_a_t.detach().clone().requires_grad_(True)
    gate_b_test = gate_b_t.detach().clone().requires_grad_(True)
    up_a_test = up_a_t.detach().clone().requires_grad_(True)
    up_b_test = up_b_t.detach().clone().requires_grad_(True)
    got_out = quack_grouped_lora_dual(
        x_test,
        gate_a_test,
        gate_b_test,
        up_a_test,
        up_b_test,
        counts,
        scale_gate=scale_gate,
        scale_up=scale_up,
    )
    got_loss = (got_out.float() * loss_grad.float()).sum() / max(1, loss_grad.numel())
    got_loss.backward()

    assert torch.allclose(ref_out, got_out.detach(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(
        _require_grad(x_ref.grad, name="x_ref"),
        _require_grad(x_test.grad, name="x_test"),
        atol=5e-2,
        rtol=5e-2,
    )
    assert torch.allclose(
        _require_grad(gate_a_ref.grad, name="gate_a_ref"),
        _require_grad(gate_a_test.grad, name="gate_a_test"),
        atol=5e-2,
        rtol=5e-2,
    )
    assert torch.allclose(
        _require_grad(gate_b_ref.grad, name="gate_b_ref"),
        _require_grad(gate_b_test.grad, name="gate_b_test"),
        atol=5e-2,
        rtol=5e-2,
    )
    assert torch.allclose(
        _require_grad(up_a_ref.grad, name="up_a_ref"),
        _require_grad(up_a_test.grad, name="up_a_test"),
        atol=5e-2,
        rtol=5e-2,
    )
    assert torch.allclose(
        _require_grad(up_b_ref.grad, name="up_b_ref"),
        _require_grad(up_b_test.grad, name="up_b_test"),
        atol=5e-2,
        rtol=5e-2,
    )
