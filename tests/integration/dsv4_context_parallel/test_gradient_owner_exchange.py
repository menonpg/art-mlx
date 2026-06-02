from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.multiprocessing as mp

from art.megatron.dsv4 import (
    Dsv4AttentionGradientResult,
    accumulate_dsv4_gradient_owner_buckets,
    launch_dsv4_gradient_owner_bucket_exchange,
)


def test_gradient_owner_bucket_exchange_reduces_duplicate_remote_ids(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_gradient_owner_exchange_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _gradient_owner_exchange_worker,
        args=(2, str(init_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def _gradient_owner_exchange_worker(rank: int, world_size: int, init_path: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29625")
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        gradients = _rank_gradients(rank)
        work = launch_dsv4_gradient_owner_bucket_exchange(
            gradients=gradients,
            query_owner_ranks=_query_owner_ranks(rank),
            raw_owner_ranks=_raw_owner_ranks(rank),
            compressed_owner_ranks=_compressed_owner_ranks(rank),
            recv_query_token_ids_by_peer=_recv_query_ids(rank),
            recv_raw_token_ids_by_peer=_recv_raw_ids(rank),
            recv_compressed_entry_ids_by_peer=_recv_compressed_ids(rank),
            rank=rank,
            rank_count=world_size,
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
        )
        buckets = work.wait_post_process()
        reduced = accumulate_dsv4_gradient_owner_buckets(
            buckets=buckets,
            query_token_ids=_owned_query_ids(rank),
            raw_token_ids=_owned_raw_ids(rank),
            compressed_entry_ids=_owned_compressed_ids(rank),
            d_attn_sink=torch.zeros(2, dtype=torch.float64),
        )
        expected = _expected_reduced(rank)
        torch.testing.assert_close(reduced.dq, expected.dq, rtol=0, atol=0)
        torch.testing.assert_close(reduced.draw_kv, expected.draw_kv, rtol=0, atol=0)
        torch.testing.assert_close(
            reduced.dcompressed_kv,
            expected.dcompressed_kv,
            rtol=0,
            atol=0,
        )
    finally:
        destroy_process_group()


def _rank_gradients(rank: int) -> Dsv4AttentionGradientResult:
    if rank == 0:
        return Dsv4AttentionGradientResult(
            query_token_ids=(10, 11),
            raw_token_ids=(20,),
            compressed_entry_ids=(100,),
            dq=_query_rows((1.0, 2.0)),
            draw_kv=_kv_rows((3.0,)),
            dcompressed_kv=_kv_rows((4.0,)),
            d_attn_sink=torch.zeros(2, dtype=torch.float64),
        )
    return Dsv4AttentionGradientResult(
        query_token_ids=(11, 12),
        raw_token_ids=(20,),
        compressed_entry_ids=(101,),
        dq=_query_rows((5.0, 6.0)),
        draw_kv=_kv_rows((7.0,)),
        dcompressed_kv=_kv_rows((8.0,)),
        d_attn_sink=torch.zeros(2, dtype=torch.float64),
    )


def _expected_reduced(rank: int) -> Dsv4AttentionGradientResult:
    if rank == 0:
        return Dsv4AttentionGradientResult(
            query_token_ids=(10, 11),
            raw_token_ids=(20,),
            compressed_entry_ids=(),
            dq=_query_rows((1.0, 7.0)),
            draw_kv=_kv_rows((10.0,)),
            dcompressed_kv=_kv_rows(()),
            d_attn_sink=torch.zeros(2, dtype=torch.float64),
        )
    return Dsv4AttentionGradientResult(
        query_token_ids=(12,),
        raw_token_ids=(),
        compressed_entry_ids=(100, 101),
        dq=_query_rows((6.0,)),
        draw_kv=_kv_rows(()),
        dcompressed_kv=_kv_rows((4.0, 8.0)),
        d_attn_sink=torch.zeros(2, dtype=torch.float64),
    )


def _query_owner_ranks(rank: int) -> tuple[int, ...]:
    return (0, 0) if rank == 0 else (0, 1)


def _raw_owner_ranks(rank: int) -> tuple[int, ...]:
    return (0,) if rank in (0, 1) else ()


def _compressed_owner_ranks(rank: int) -> tuple[int, ...]:
    return (1,) if rank in (0, 1) else ()


def _recv_query_ids(rank: int) -> tuple[tuple[int, ...], ...]:
    return ((10, 11), (11,)) if rank == 0 else ((), (12,))


def _recv_raw_ids(rank: int) -> tuple[tuple[int, ...], ...]:
    return ((20,), (20,)) if rank == 0 else ((), ())


def _recv_compressed_ids(rank: int) -> tuple[tuple[int, ...], ...]:
    return ((), ()) if rank == 0 else ((100,), (101,))


def _owned_query_ids(rank: int) -> tuple[int, ...]:
    return (10, 11) if rank == 0 else (12,)


def _owned_raw_ids(rank: int) -> tuple[int, ...]:
    return (20,) if rank == 0 else ()


def _owned_compressed_ids(rank: int) -> tuple[int, ...]:
    return () if rank == 0 else (100, 101)


def _query_rows(values: tuple[float, ...]) -> torch.Tensor:
    return (
        torch.tensor(values, dtype=torch.float64)
        .view(1, len(values), 1, 1)
        .expand(
            -1,
            -1,
            2,
            3,
        )
    )


def _kv_rows(values: tuple[float, ...]) -> torch.Tensor:
    return (
        torch.tensor(values, dtype=torch.float64)
        .view(1, len(values), 1)
        .expand(
            -1,
            -1,
            3,
        )
    )
