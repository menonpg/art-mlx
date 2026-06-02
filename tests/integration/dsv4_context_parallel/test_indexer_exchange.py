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
    build_dsv4_indexer_kv_exchange_peer_plans,
    compute_indexer_stage_topk,
    launch_dsv4_indexer_kv_exchange,
)


class _LayoutIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    ownership_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    token_counts_by_rank: tuple[int, ...]


def test_indexer_kv_exchange_matches_full_stage_topk(tmp_path: Path) -> None:
    init_path = tmp_path / "dsv4_indexer_kv_exchange_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _indexer_kv_exchange_worker,
        args=(2, str(init_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def _indexer_kv_exchange_worker(rank: int, world_size: int, init_path: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29627")
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        layout = _layout()
        local_entry_ids = (1,) if rank == 0 else (2,)
        query_token_ids = (7,) if rank == 0 else (11,)
        indexer_q = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
        indexer_weights = torch.ones(1, 1, dtype=torch.float32)
        peer_plan = build_dsv4_indexer_kv_exchange_peer_plans(
            layout=layout,
            candidate_entry_ids_by_rank=((1, 2), (1, 2)),
        )[rank]
        assert peer_plan.recv_entry_ids_by_peer == ((1,), (2,))
        assert peer_plan.send_entry_ids_by_peer == (local_entry_ids, local_entry_ids)
        work = launch_dsv4_indexer_kv_exchange(
            layout=layout,
            query_token_ids=query_token_ids,
            candidate_entry_ids=(1, 2),
            indexer_q=indexer_q,
            indexer_weights=indexer_weights,
            indexer_kv=_indexer_kv(local_entry_ids),
            indexer_kv_entry_ids=local_entry_ids,
            send_entry_ids_by_peer=peer_plan.send_entry_ids_by_peer,
            recv_entry_ids_by_peer=peer_plan.recv_entry_ids_by_peer,
            topk=2,
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
        )
        actual = work.wait_post_process()
        expected = compute_indexer_stage_topk(
            layout=layout,
            query_token_ids=query_token_ids,
            candidate_entry_ids=(1, 2),
            indexer_q=indexer_q,
            indexer_weights=indexer_weights,
            indexer_kv=_indexer_kv((1, 2)),
            indexer_kv_entry_ids=(1, 2),
            topk=2,
        )

        torch.testing.assert_close(actual.scores, expected.scores, rtol=0, atol=0)
        torch.testing.assert_close(actual.indices, expected.indices, rtol=0, atol=0)
        if rank == 0:
            assert actual.indices[0, 0].tolist() == [1, -1]
        else:
            assert actual.indices[0, 0].tolist() == [2, 1]
    finally:
        destroy_process_group()


def _indexer_kv(entry_ids: tuple[int, ...]) -> torch.Tensor:
    values = {
        1: (2.0, 0.0),
        2: (3.0, 0.0),
    }
    return torch.tensor([values[int(entry_id)] for entry_id in entry_ids])


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
