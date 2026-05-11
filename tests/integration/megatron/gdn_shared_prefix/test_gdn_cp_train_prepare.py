from __future__ import annotations

from pathlib import Path
import socket
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")

from megatron.core import parallel_state as ps  # noqa: E402
from torch.distributed import destroy_process_group, init_process_group  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from art.megatron import train as megatron_train  # noqa: E402
from art.megatron.context_parallel.runtime import prepare_cp_micro  # noqa: E402
from art.megatron.context_parallel.types import (  # noqa: E402
    ArtContextParallelState,
    ParallelTopology,
)
from art.preprocessing.pack import PackedTensors  # noqa: E402

from .cases import default_phase0_cases  # noqa: E402
from .packed_layout import build_phase0_packed_tensors  # noqa: E402


class _Handler:
    build_gdn_execution_spec = True


def test_gdn_cp_training_batch_carries_prebuilt_rank_plan(tmp_path: Path) -> None:
    cp_size = 2
    if not torch.cuda.is_available() or torch.cuda.device_count() < cp_size:
        pytest.skip(f"requires {cp_size} CUDA devices")
    port = _find_free_port()
    mp.spawn(
        _worker,
        args=(cp_size, port, str(tmp_path)),
        nprocs=cp_size,
        join=True,
    )
    for rank in range(cp_size):
        assert (tmp_path / f"rank_{rank}.ok").read_text() == "ok\n"


def _worker(rank: int, cp_size: int, port: int, output_dir: str) -> None:
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=cp_size,
    )
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
            expert_model_parallel_size=1,
        )
        micro = cast(
            PackedTensors,
            {
                key: value.cuda() if isinstance(value, torch.Tensor) else value
                for key, value in build_phase0_packed_tensors(
                    default_phase0_cases()[1]
                ).items()
            },
        )
        prepared = prepare_cp_micro(
            micro=micro,
            topology=ParallelTopology(cp=cp_size),
            config=megatron_train.ContextParallelConfig(),
            cp_group=ps.get_context_parallel_group(check_initialized=False),
            cp_rank=ps.get_context_parallel_rank(),
            build_gdn_execution_spec=True,
        )
        state = prepared.attention_state
        assert isinstance(state, ArtContextParallelState)
        plan = state.gdn_execution_plan
        assert plan is not None
        assert plan.cp_rank == rank
        assert plan.cp_size == cp_size
        assert state.gdn_execution_spec is not None
        assert prepared.tensors.tokens.shape == (1, int(plan.attention_token_count))
        assert prepared.tensors.labels.shape == prepared.tensors.tokens.shape
        assert prepared.tensors.input_pos.shape == prepared.tensors.tokens.shape
        assert prepared.tensors.valid_lengths == (int(plan.attention_token_count),)
        Path(output_dir, f"rank_{rank}.ok").write_text("ok\n")
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_cp_training_guard_allows_attention_and_gdn_handlers() -> None:
    for handler in (object(), _Handler()):
        megatron_train._validate_context_parallel_training_supported(
            model_chunks=cast(Any, []),
            model_support_handler=handler,
            experimental_config={},
            topology=ParallelTopology(cp=2),
        )


@pytest.mark.parametrize(
    "experimental_config",
    (
        {"importance_sampling_level": "sequence"},
        {"truncated_importance_sampling": 2.0},
    ),
)
def test_cp_training_guard_rejects_unsupported_loss_knobs(
    experimental_config: dict[str, object],
) -> None:
    with pytest.raises(NotImplementedError):
        megatron_train._validate_context_parallel_training_supported(
            model_chunks=cast(Any, []),
            model_support_handler=_Handler(),
            experimental_config=cast(Any, experimental_config),
            topology=ParallelTopology(cp=2),
        )


def test_sft_cp_guard_allows_gdn_handler() -> None:
    megatron_train._validate_context_parallel_training_supported(
        model_chunks=cast(Any, []),
        model_support_handler=_Handler(),
        experimental_config={},
        topology=ParallelTopology(cp=2),
    )
