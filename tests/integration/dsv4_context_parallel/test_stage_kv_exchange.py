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
    build_dsv4_stage_inputs,
    build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans,
    build_dsv4_stage_plan_slots,
    launch_dsv4_stage_kv_exchange_from_stage_plan_slot,
)


class _LayoutIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    ownership_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    token_counts_by_rank: tuple[int, ...]


class _Range(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int
    end: int


class _StagePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage_index: int
    global_q_ranges: tuple[_Range, ...]
    global_k_ranges: tuple[_Range, ...]


def test_stage_kv_exchange_materializes_fused_raw_and_compressed_stage(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_stage_kv_exchange_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _stage_kv_exchange_worker,
        args=(2, str(init_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def _stage_kv_exchange_worker(rank: int, world_size: int, init_path: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29626")
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        layout = _layout()
        slot = build_dsv4_stage_plan_slots(
            stage_plans_by_rank=(
                (_stage_plan(stage_index=9, q_ranges=((7, 8),), k_ranges=((4, 13),)),),
                (
                    _stage_plan(
                        stage_index=9,
                        q_ranges=((11, 12),),
                        k_ranges=((4, 13),),
                    ),
                ),
            ),
        )[0]
        stages = tuple(
            build_dsv4_stage_inputs(
                layout=layout,
                compression_kind=layout.spec.kind,
                stage_index=9,
                query_token_ids=query_ids,
                global_k_ranges=(_Range(start=4, end=13),),
                global_topk=torch.tensor([[1, 2]], dtype=torch.long),
                window_size=4,
            )
            for query_ids in ((7,), (11,))
        )
        stage = stages[rank]
        query_ids = stage.query_token_ids
        peer_plans = build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans(
            layout=layout,
            stage_plans_by_rank=slot.stage_plans_by_rank,
        )
        peer_plan = peer_plans[rank]
        assert peer_plan.recv_raw_token_ids_by_peer == (
            (4, 5, 6, 7),
            (8, 9, 10, 11, 12),
        )
        assert peer_plan.recv_compressed_entry_ids_by_peer == ((1,), (2,))
        local_raw_ids = (4, 5, 6, 7) if rank == 0 else (8, 9, 10, 11, 12)
        local_compressed_ids = (1,) if rank == 0 else (2,)
        assert peer_plan.send_raw_token_ids_by_peer == (
            local_raw_ids,
            local_raw_ids,
        )
        assert peer_plan.send_compressed_entry_ids_by_peer == (
            local_compressed_ids,
            local_compressed_ids,
        )
        work = launch_dsv4_stage_kv_exchange_from_stage_plan_slot(
            layout=layout,
            rank=rank,
            stage_plan_slot=slot,
            local_stage_inputs=stage,
            query=_query_rows(query_ids),
            query_token_ids=query_ids,
            raw_kv=_kv_rows(local_raw_ids),
            raw_token_ids=local_raw_ids,
            compressed_kv=_compressed_rows(local_compressed_ids),
            compressed_entry_ids=local_compressed_ids,
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
            peer_plans=peer_plans,
        )
        materialized = work.wait_post_process()

        assert materialized.query_token_ids == query_ids
        assert materialized.raw_count == 9
        assert materialized.compressed_count == 2
        assert materialized.key_global_ids == (
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            1,
            2,
        )
        assert materialized.q_stage[0, 0, 0, 0].item() == float(query_ids[0])
        assert materialized.kv_stage[0, :, 0].tolist() == [
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            201.0,
            202.0,
        ]
        if rank == 0:
            assert materialized.topk_stage_local[0, 0].tolist() == [
                0,
                1,
                2,
                3,
                9,
                -1,
            ]
        else:
            assert materialized.topk_stage_local[0, 0].tolist() == [
                4,
                5,
                6,
                7,
                9,
                10,
            ]
    finally:
        destroy_process_group()


def _query_rows(token_ids: tuple[int, ...]) -> torch.Tensor:
    return torch.stack(
        [torch.full((2, 4), float(token_id)) for token_id in token_ids],
        dim=0,
    )


def _kv_rows(token_ids: tuple[int, ...]) -> torch.Tensor:
    return torch.stack(
        [torch.full((4,), float(token_id)) for token_id in token_ids],
        dim=0,
    )


def _compressed_rows(entry_ids: tuple[int, ...]) -> torch.Tensor:
    return torch.stack(
        [torch.full((4,), float(entry_id + 200)) for entry_id in entry_ids],
        dim=0,
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


def _stage_plan(
    *,
    stage_index: int,
    q_ranges: tuple[tuple[int, int], ...],
    k_ranges: tuple[tuple[int, int], ...],
) -> _StagePlan:
    return _StagePlan(
        stage_index=stage_index,
        global_q_ranges=tuple(_Range(start=start, end=end) for start, end in q_ranges),
        global_k_ranges=tuple(_Range(start=start, end=end) for start, end in k_ranges),
    )
