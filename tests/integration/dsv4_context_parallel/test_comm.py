from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.multiprocessing as mp

from art.megatron.dsv4 import (
    Dsv4TensorExchangePlan,
    launch_dsv4_tensor_exchange,
)


def test_dsv4_tensor_exchange_all_to_all_roundtrips_explicit_ids(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "dsv4_tensor_exchange_gloo"
    if init_path.exists():
        init_path.unlink()
    mp.start_processes(
        _exchange_worker,
        args=(2, str(init_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    if init_path.exists():
        init_path.unlink()


def test_dsv4_tensor_exchange_rejects_bad_id_spaces() -> None:
    tensor = _rows((0, 1))
    plan = Dsv4TensorExchangePlan(
        send_ids_by_peer=((2,),),
        recv_ids_by_peer=((0,),),
    )
    with pytest.raises(RuntimeError, match="missing ids"):
        launch_dsv4_tensor_exchange(
            tensor=tensor,
            tensor_ids=(0, 1),
            plan=plan,
            group=None,
            async_op=False,
        )

    duplicate_recv = Dsv4TensorExchangePlan(
        send_ids_by_peer=((0,),),
        recv_ids_by_peer=((0, 0),),
    )
    with pytest.raises(RuntimeError, match="duplicate id"):
        launch_dsv4_tensor_exchange(
            tensor=tensor,
            tensor_ids=(0, 1),
            plan=duplicate_recv,
            group=None,
            async_op=False,
        )

    with pytest.raises(RuntimeError, match="tensor_ids contains duplicate"):
        launch_dsv4_tensor_exchange(
            tensor=tensor,
            tensor_ids=(0, 0),
            plan=Dsv4TensorExchangePlan(
                send_ids_by_peer=((0,),),
                recv_ids_by_peer=((0,),),
            ),
            group=None,
            async_op=False,
        )


def _exchange_worker(rank: int, world_size: int, init_path: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29623")
    init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        local_ids_by_rank = ((0, 1, 2), (3, 4))
        local_ids = local_ids_by_rank[rank]
        local = _rows(local_ids)
        plans = (
            Dsv4TensorExchangePlan(
                send_ids_by_peer=((1,), (0, 2)),
                recv_ids_by_peer=((1,), (3,)),
            ),
            Dsv4TensorExchangePlan(
                send_ids_by_peer=((3,), (4,)),
                recv_ids_by_peer=((0, 2), (4,)),
            ),
        )
        expected_ids_by_rank = ((1, 3), (0, 2, 4))

        work = launch_dsv4_tensor_exchange(
            tensor=local,
            tensor_ids=local_ids,
            plan=plans[rank],
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=True,
        )
        result = work.wait_post_process()
        expected_ids = expected_ids_by_rank[rank]
        assert result.ids == expected_ids
        torch.testing.assert_close(result.tensor, _rows(expected_ids), rtol=0, atol=0)

        batched = torch.stack((local, local + 100), dim=0)
        batched_work = launch_dsv4_tensor_exchange(
            tensor=batched,
            tensor_ids=local_ids,
            plan=plans[rank],
            group=cast(Any, torch.distributed).group.WORLD,
            async_op=False,
        )
        batched_result = batched_work.wait_post_process()
        expected_batched = torch.stack(
            (_rows(expected_ids), _rows(expected_ids) + 100),
            dim=0,
        )
        assert batched_result.ids == expected_ids
        torch.testing.assert_close(
            batched_result.tensor,
            expected_batched,
            rtol=0,
            atol=0,
        )
    finally:
        destroy_process_group()


def _rows(ids: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(
        [[float(id_), float(id_ + 10)] for id_ in ids],
        dtype=torch.float32,
    )
