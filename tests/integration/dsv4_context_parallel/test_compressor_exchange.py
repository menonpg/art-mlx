from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict
import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.multiprocessing as mp

from art.megatron.dsv4 import (
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    accumulate_dsv4_compression_halo_gradient_payloads,
    build_dsv4_compressed_layout,
    compress_owned_projected_kv,
    compress_projected_kv,
    launch_dsv4_compressed_kv_backward,
    launch_dsv4_compressed_kv_forward,
    launch_dsv4_compression_halo_exchange,
    launch_dsv4_compression_halo_gradient_exchange,
    materialize_dsv4_compression_token_buffer,
)


class _LayoutIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    ownership_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    token_counts_by_rank: tuple[int, ...]


def test_compression_halo_exchange_feeds_owner_rank_compression(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_compression_halo_exchange_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _halo_exchange_worker,
        args=(2, str(init_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def test_compressed_kv_work_matches_global_oracle_and_reduces_bias_grad(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_compressed_kv_work_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _compressed_kv_work_worker,
        args=(2, str(init_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def _halo_exchange_worker(rank: int, world_size: int, init_path: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29624")
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        layout = _layout()
        full_kv = _global_projected(18, 10)
        full_gate = full_kv + 1000
        positional_bias = torch.linspace(-0.5, 0.5, steps=40).reshape(4, 10)
        local_ids_by_rank = (tuple(range(8)), tuple(range(8, 18)))
        local_ids = local_ids_by_rank[rank]
        local_index = torch.tensor(local_ids, dtype=torch.long)

        work = launch_dsv4_compression_halo_exchange(
            layout=layout,
            rank=rank,
            projected_kv=full_kv.index_select(0, local_index),
            projected_gate=full_gate.index_select(0, local_index),
            token_ids=local_ids,
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
        )
        payloads = work.wait_post_process()
        grad_token_ids = local_ids
        grad_kv = torch.zeros_like(full_kv.index_select(0, local_index))
        grad_gate = torch.zeros_like(full_gate.index_select(0, local_index))

        if rank == 0:
            assert payloads == ()
        else:
            assert len(payloads) == 1
            payload = payloads[0]
            assert payload.source_rank == 0
            assert payload.target_rank == 1
            assert payload.token_ids == (4, 5, 6, 7)
            assert payload.entry_ids == (2, 3)
            torch.testing.assert_close(
                payload.projected_kv,
                full_kv.index_select(0, torch.tensor(payload.token_ids)),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                payload.projected_gate,
                full_gate.index_select(0, torch.tensor(payload.token_ids)),
                rtol=0,
                atol=0,
            )

            buffer = materialize_dsv4_compression_token_buffer(
                layout=layout,
                owner_rank=rank,
                projected_kv=full_kv.index_select(0, local_index),
                projected_gate=full_gate.index_select(0, local_index),
                token_ids=local_ids,
                halo_payloads=payloads,
            )
            actual = compress_owned_projected_kv(
                layout=layout,
                owner_rank=rank,
                projected_kv=buffer.projected_kv,
                projected_gate=buffer.projected_gate,
                positional_bias=positional_bias,
                token_ids=buffer.token_ids,
            )
            expected = compress_owned_projected_kv(
                layout=layout,
                owner_rank=rank,
                projected_kv=full_kv,
                projected_gate=full_gate,
                positional_bias=positional_bias,
            )
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

            buffer_kv = buffer.projected_kv.detach().clone().requires_grad_()
            buffer_gate = buffer.projected_gate.detach().clone().requires_grad_()
            grad_actual = compress_owned_projected_kv(
                layout=layout,
                owner_rank=rank,
                projected_kv=buffer_kv,
                projected_gate=buffer_gate,
                positional_bias=positional_bias,
                token_ids=buffer.token_ids,
            )
            grad_actual.square().sum().backward()
            assert buffer_kv.grad is not None
            assert buffer_gate.grad is not None
            grad_token_ids = buffer.token_ids
            grad_kv = buffer_kv.grad
            grad_gate = buffer_gate.grad

        grad_work = launch_dsv4_compression_halo_gradient_exchange(
            layout=layout,
            rank=rank,
            token_ids=grad_token_ids,
            dprojected_kv=grad_kv,
            dprojected_gate=grad_gate,
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
        )
        grad_payloads = grad_work.wait_post_process()

        if rank == 1:
            assert grad_payloads == ()
            return

        assert len(grad_payloads) == 1
        grad_payload = grad_payloads[0]
        assert grad_payload.source_rank == 1
        assert grad_payload.target_rank == 0
        assert grad_payload.token_ids == (4, 5, 6, 7)
        assert grad_payload.entry_ids == (2, 3)

        ref_grad_kv, ref_grad_gate = _global_owner_one_grads(
            layout=layout,
            full_kv=full_kv,
            full_gate=full_gate,
            positional_bias=positional_bias,
        )
        returned = accumulate_dsv4_compression_halo_gradient_payloads(
            target_rank=rank,
            token_ids=local_ids,
            dprojected_kv=torch.zeros_like(full_kv.index_select(0, local_index)),
            dprojected_gate=torch.zeros_like(full_gate.index_select(0, local_index)),
            halo_gradient_payloads=grad_payloads,
        )
        torch.testing.assert_close(
            returned.projected_kv[:4],
            torch.zeros_like(returned.projected_kv[:4]),
        )
        torch.testing.assert_close(
            returned.projected_gate[:4],
            torch.zeros_like(returned.projected_gate[:4]),
        )
        torch.testing.assert_close(
            returned.projected_kv[4:],
            ref_grad_kv[:8][4:],
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            returned.projected_gate[4:],
            ref_grad_gate[:8][4:],
            rtol=1e-6,
            atol=1e-6,
        )
    finally:
        destroy_process_group()


def _compressed_kv_work_worker(rank: int, world_size: int, init_path: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29625")
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        layout = _layout()
        full_kv = _global_projected(18, 10) / 50
        full_gate = (full_kv + 0.25).sin()
        positional_bias = torch.linspace(-0.5, 0.5, steps=40).reshape(4, 10)
        local_ids_by_rank = (tuple(range(8)), tuple(range(8, 18)))
        local_ids = local_ids_by_rank[rank]
        local_index = torch.tensor(local_ids, dtype=torch.long)

        forward_work = launch_dsv4_compressed_kv_forward(
            layout=layout,
            rank=rank,
            projected_kv=full_kv.index_select(0, local_index),
            projected_gate=full_gate.index_select(0, local_index),
            positional_bias=positional_bias,
            token_ids=local_ids,
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
        )
        forward_result = forward_work.wait_post_process()
        expected_owned = compress_owned_projected_kv(
            layout=layout,
            owner_rank=rank,
            projected_kv=full_kv,
            projected_gate=full_gate,
            positional_bias=positional_bias,
        )
        assert forward_result.owner_rank == rank
        assert forward_result.local_token_ids == local_ids
        assert (
            forward_result.compressed_entry_ids == layout.entry_ids_by_owner_rank[rank]
        )
        torch.testing.assert_close(
            forward_result.compressed_kv,
            expected_owned,
            rtol=1e-6,
            atol=1e-6,
        )

        backward_work = launch_dsv4_compressed_kv_backward(
            forward_result=forward_result,
            dcompressed_kv=2 * forward_result.compressed_kv.detach(),
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
        )
        grad_result = backward_work.wait_post_process()

        ref_kv = full_kv.detach().clone().requires_grad_()
        ref_gate = full_gate.detach().clone().requires_grad_()
        ref_bias = positional_bias.detach().clone().requires_grad_()
        ref_compressed = compress_projected_kv(
            layout=layout,
            projected_kv=ref_kv,
            projected_gate=ref_gate,
            positional_bias=ref_bias,
        )
        ref_compressed.square().sum().backward()
        assert ref_kv.grad is not None
        assert ref_gate.grad is not None
        assert ref_bias.grad is not None

        assert grad_result.token_ids == local_ids
        torch.testing.assert_close(
            grad_result.dprojected_kv,
            ref_kv.grad.index_select(0, local_index),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            grad_result.dprojected_gate,
            ref_gate.grad.index_select(0, local_index),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            grad_result.dpositional_bias,
            ref_bias.grad,
            rtol=1e-6,
            atol=1e-6,
        )
        assert not bool(grad_result.dprojected_kv.abs().sum().eq(0).item())
        assert not bool(grad_result.dprojected_gate.abs().sum().eq(0).item())
        assert not bool(grad_result.dpositional_bias.abs().sum().eq(0).item())
    finally:
        destroy_process_group()


def _global_projected(token_count: int, width: int) -> torch.Tensor:
    return torch.arange(token_count * width, dtype=torch.float32).reshape(
        token_count,
        width,
    )


def _global_owner_one_grads(
    *,
    layout: Dsv4CompressedLayout,
    full_kv: torch.Tensor,
    full_gate: torch.Tensor,
    positional_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_kv = full_kv.detach().clone().requires_grad_()
    ref_gate = full_gate.detach().clone().requires_grad_()
    expected = compress_owned_projected_kv(
        layout=layout,
        owner_rank=1,
        projected_kv=ref_kv,
        projected_gate=ref_gate,
        positional_bias=positional_bias,
    )
    expected.square().sum().backward()
    assert ref_kv.grad is not None
    assert ref_gate.grad is not None
    return ref_kv.grad, ref_gate.grad


def _layout() -> Dsv4CompressedLayout:
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
        spec=Dsv4CompressionSpec(kind=Dsv4CompressionKind.CSA, ratio=4),
    )
