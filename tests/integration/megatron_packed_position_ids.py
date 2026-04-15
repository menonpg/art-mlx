from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import socket
from typing import Any, Iterator, cast

from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from pydantic import BaseModel, Field
import torch
from torch.distributed import destroy_process_group, init_process_group, is_initialized

from art.megatron import train as megatron_train
from art.megatron.provider import get_provider_bundle

from .megatron_oracle_harness import (
    ORACLE_TOPOLOGY,
    OracleCaseConfig,
    PackedTensorConfig,
    _build_packed_tensors,
)
from .megatron_oracle_worker import _configure_provider


def _slugify(value: str) -> str:
    return value.lower().replace("/", "_").replace(".", "_").replace("-", "_")


def _artifact_dir(base_model: str) -> Path:
    root = Path(__file__).resolve().parents[2] / ".local" / "model_support_validation"
    path = root / _slugify(base_model) / "packed_position_ids"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _single_rank_model_parallel() -> Iterator[None]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for packed position id validation")
    if is_initialized():
        raise RuntimeError("torch.distributed is already initialized")

    torch.cuda.set_device(0)
    init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{_find_free_port()}",
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
        model_parallel_cuda_manual_seed(1234)
        yield
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        if is_initialized():
            destroy_process_group()


def _locate_gpt_module(model_chunks: list[Any]) -> GPTModel:
    for chunk in model_chunks:
        module: Any = chunk
        while hasattr(module, "module"):
            module = module.module
        if isinstance(module, GPTModel):
            return module
        language_model = getattr(module, "language_model", None)
        if isinstance(language_model, GPTModel):
            return language_model
    raise RuntimeError("Failed to locate GPTModel for packed position id validation")


class PackedPositionIdScenario(BaseModel):
    name: str
    num_sequences: int
    sequence_length: int
    checked_token_count: int
    prompt_family_count: int
    matched: bool


class PackedPositionIdsReport(BaseModel):
    base_model: str
    output_dir: str
    num_layers: int
    scenarios: list[PackedPositionIdScenario] = Field(default_factory=list)


def _prompt_family_count(group_ids: torch.Tensor, parent_ids: torch.Tensor) -> int:
    families = 0
    for row_index in range(int(group_ids.shape[0])):
        valid_tokens = int((group_ids[row_index] != -1).sum().item())
        cursor = 0
        while cursor < valid_tokens:
            group_id = int(group_ids[row_index, cursor].item())
            parent_id = int(parent_ids[row_index, cursor].item())
            if group_id == parent_id:
                families += 1
            while (
                cursor < valid_tokens
                and int(group_ids[row_index, cursor].item()) == group_id
            ):
                cursor += 1
    return families


def run_packed_position_ids(
    *,
    base_model: str,
    num_layers: int,
) -> PackedPositionIdsReport:
    output_dir = _artifact_dir(base_model)
    scenarios = [
        (
            "stop_early",
            PackedTensorConfig(
                num_sequences=4,
                sequence_length=95,
                prefill_tokens=13,
                completion_branches_per_prefix=2,
                decode_tokens=11,
                decode_tokens_jitter=3,
                packing_mode="stop_early",
            ),
        ),
        (
            "truncate",
            PackedTensorConfig(
                num_sequences=4,
                sequence_length=61,
                prefill_tokens=17,
                completion_branches_per_prefix=2,
                decode_tokens=15,
                decode_tokens_jitter=0,
                packing_mode="truncate",
            ),
        ),
    ]
    report = PackedPositionIdsReport(
        base_model=base_model,
        output_dir=str(output_dir),
        num_layers=num_layers,
    )

    with _single_rank_model_parallel():
        case_config = OracleCaseConfig(
            base_model=base_model,
            precision="fp32",
            num_layers=num_layers,
        )
        provider_bundle = get_provider_bundle(
            base_model,
            torch_dtype=torch.float32,
            runtime_profile="single_gpu_parity",
        )
        provider = provider_bundle.provider
        _configure_provider(provider, ORACLE_TOPOLOGY, case_config)
        model_chunks = cast(
            list[Any],
            provider.provide_distributed_model(
                ddp_config=DistributedDataParallelConfig(
                    grad_reduce_in_fp32=True,
                    average_in_collective=False,
                ),
                data_parallel_random_init=False,
                mixed_precision_wrapper=None,
            ),
        )
        gpt_module = _locate_gpt_module(model_chunks)

        def _fake_preprocess(
            *args: Any, **kwargs: Any
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del args
            position_ids = cast(torch.Tensor, kwargs["position_ids"])
            batch_size, sequence_length = position_ids.shape
            embedding_dim = 4
            hidden = torch.zeros(
                (sequence_length, batch_size, embedding_dim),
                device=position_ids.device,
                dtype=torch.float32,
            )
            max_position = int(position_ids.max().item()) + 1
            table = torch.arange(
                max_position * embedding_dim,
                device=position_ids.device,
                dtype=torch.float32,
            ).view(max_position, 1, 1, embedding_dim)
            return hidden, table

        gpt_module._preprocess = _fake_preprocess  # type: ignore[attr-defined]
        megatron_train._install_gpt_preprocess_hook(model_chunks)

        for scenario_name, packed_config in scenarios:
            packed_tensors = _build_packed_tensors(packed_config, case_config.seed)
            position_ids = cast(torch.Tensor, packed_tensors["input_pos"]).cuda()
            input_ids = torch.zeros_like(position_ids)
            group_ids = cast(torch.Tensor, packed_tensors["group_ids"])
            parent_ids = cast(torch.Tensor, packed_tensors["parent_ids"])
            _hidden, rotary = gpt_module._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
            )
            embedding_dim = int(rotary.shape[-1])
            max_position = int(position_ids.max().item()) + 1
            expected_table = torch.arange(
                max_position * embedding_dim,
                device=position_ids.device,
                dtype=torch.float32,
            ).view(max_position, embedding_dim)
            expected = (
                expected_table.index_select(0, position_ids.reshape(-1))
                .view(position_ids.shape[0], position_ids.shape[1], embedding_dim)
                .permute(1, 0, 2)
                .contiguous()
                .unsqueeze(2)
            )
            report.scenarios.append(
                PackedPositionIdScenario(
                    name=scenario_name,
                    num_sequences=int(position_ids.shape[0]),
                    sequence_length=int(position_ids.shape[1]),
                    checked_token_count=int((group_ids != -1).sum().item()),
                    prompt_family_count=_prompt_family_count(group_ids, parent_ids),
                    matched=torch.equal(rotary, expected),
                )
            )
        del model_chunks, provider_bundle
        torch.cuda.empty_cache()

    (output_dir / "report.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return report
