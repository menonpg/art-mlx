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
    build_dsv4_compressed_layout,
    compress_owned_projected_kv,
    launch_dsv4_compression_halo_exchange,
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

        if rank == 0:
            assert payloads == ()
            return

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
    finally:
        destroy_process_group()


def _global_projected(token_count: int, width: int) -> torch.Tensor:
    return torch.arange(token_count * width, dtype=torch.float32).reshape(
        token_count,
        width,
    )


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
