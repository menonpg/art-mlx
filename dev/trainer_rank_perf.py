from __future__ import annotations

from collections.abc import Callable, Sequence
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import typer

from art.megatron.trainer_rank import (
    ForwardInput,
    TopK,
    TrainerRank,
    _batch_seq_logits,
    _language_model,
    _pack_forward_items,
)


def main(
    model: str = "Qwen/Qwen3-0.6B",
    layers: int = 1,
    seq_len: int = 2048,
    prefix_families: int = 0,
    prefix_len: int = 5000,
    mid_prefixes_per_family: int = 1,
    mid_prefix_len: int = 0,
    branches_per_prefix: int = 16,
    completion_len: int = 100,
    warmup: int = 2,
    repeat: int = 5,
    head_chunk_tokens: int = 512,
    shared_prefix_max_depth: int = 1,
    benchmark: str = "target_builtin_fwd",
    target_count: int = 4,
    top_k: int = 5,
    top_k_values: str = "1,2,5,10,20,50",
    max_unpacked_output_gb: float = 0.5,
    mask_prefix_targets: bool = True,
    workload: str = "regular",
    tree_depth: int = 3,
    tree_seed: int = 1,
    tree_duplicate_factor: int = 1,
    adapter_slots: int = 0,
    adapter_slot_mode: str = "family",
    adapter_slot_rank: int = 1,
    output_jsonl: str = "",
) -> None:
    os.environ.setdefault("ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_CONTEXT_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE", "1")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    try:
        from megatron.core import parallel_state as ps

        from art.megatron import train as megatron_train

        provider_configure = (
            (lambda provider: setattr(provider, "num_layers", layers))
            if layers > 0
            else None
        )
        runtime = megatron_train.build_training_runtime(
            model_identifier=model,
            provider_configure=provider_configure,
            print_env=dist.get_rank() == 0,
        )
        for chunk in runtime.model:
            chunk.eval()
        rank = TrainerRank(
            runtime,
            head_chunk_tokens=head_chunk_tokens,
            shared_prefix_max_depth=shared_prefix_max_depth,
        )
        if adapter_slots < 0:
            raise ValueError("adapter_slots must be >= 0")
        if adapter_slot_rank < 1:
            raise ValueError("adapter_slot_rank must be >= 1")
        if adapter_slots:
            loaded_sites = _load_adapter_slots(
                rank,
                count=adapter_slots,
                slot_rank=adapter_slot_rank,
            )
        else:
            loaded_sites = 0
        hidden_size, vocab_size, dtype_size = _runtime_output_shape(runtime)
        model_config = getattr(_language_model(runtime.model[0]), "config", None)

        benchmarks = {
            name.strip().replace("-", "_")
            for name in benchmark.split(",")
            if name.strip()
        }
        if "all" in benchmarks:
            benchmarks = {
                "target_builtin_fwd",
                "target_trainer_fwd",
                "target_hidden_fwd",
                "logits_builtin_fwd",
                "logits_hidden_fwd",
                "target_builtin_fwd_bwd",
                "target_builtin_masked_fwd_bwd",
                "target_trainer_fwd_bwd",
                "target_hidden_fwd_bwd",
                "trainer_multi_target_fwd_bwd",
                "trainer_target",
                "trainer_multi_target",
                "trainer_topk",
                "trainer_topk_head",
                "trainer_topk_fwd_bwd",
                "trainer_topk_sweep",
                "trainer_target_topk",
                "trainer_hidden",
                "trainer_all_no_logits",
                "trainer_logits",
            }
        if "trainer_all" in benchmarks:
            benchmarks.update(
                {
                    "trainer_target",
                    "trainer_multi_target",
                    "trainer_multi_target_fwd_bwd",
                    "trainer_topk",
                    "trainer_topk_head",
                    "trainer_topk_fwd_bwd",
                    "trainer_topk_sweep",
                    "trainer_target_topk",
                    "trainer_hidden",
                    "trainer_all_no_logits",
                    "trainer_logits",
                }
            )

        if target_count < 1:
            raise ValueError("target_count must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        requests, multi_target_requests, request_metadata = _requests(
            seq_len=seq_len,
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=mid_prefixes_per_family,
            mid_prefix_len=mid_prefix_len,
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
            target_count=target_count,
            mask_prefix_targets=mask_prefix_targets,
            workload=workload,
            tree_depth=tree_depth,
            tree_seed=tree_seed,
            tree_duplicate_factor=tree_duplicate_factor,
        )
        requests = _route_adapter_slots(
            requests,
            adapter_slots=adapter_slots,
            mode=adapter_slot_mode,
        )
        multi_target_requests = _route_adapter_slots(
            multi_target_requests,
            adapter_slots=adapter_slots,
            mode=adapter_slot_mode,
        )
        stats_items = [rank._forward_item(request) for request in requests]
        stats_batch = _pack_forward_items(
            stats_items,
            max_depth=rank.shared_prefix_max_depth,
        )
        stats_prepared = rank._prepare_packed_forward(stats_batch)
        request_stats = _packed_request_stats(
            requests,
            stats_items,
            stats_batch,
            request_metadata=request_metadata,
        )
        planner_metadata = _gather_planner_metadata(stats_prepared)
        target_items = None
        target_prepared = None
        if any(name.startswith("target_") for name in benchmarks):
            target_items = stats_items
            target_prepared = stats_prepared
        logits_items = None
        logits_prepared = None
        if any(name.startswith("logits_") for name in benchmarks):
            logits_items = [
                rank._forward_item(
                    ForwardInput(input_tokens=request.input_tokens, logits=True)
                )
                for request in requests
            ]
            logits_prepared = rank._prepare_packed_forward(
                _pack_forward_items(
                    logits_items,
                    max_depth=rank.shared_prefix_max_depth,
                )
            )
        results: dict[str, float] = {}
        metadata: dict[str, object] = {}
        rate_units: dict[str, dict[str, int]] = {}

        def register_case(
            name: str,
            case_requests: Sequence[
                ForwardInput[
                    torch.Tensor | None,
                    TopK | None,
                    torch.Tensor | None,
                    torch.Tensor | None,
                ]
            ],
            case_stats: dict[str, int | str],
        ) -> None:
            units = _rate_units(
                case_requests,
                case_stats,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                dtype_size=dtype_size,
            )
            rate_units[name] = units
            for key, value in units.items():
                metadata[f"{name}_{key}"] = value

        for name in (
            "target_builtin_fwd",
            "target_hidden_fwd",
            "target_trainer_fwd",
            "target_builtin_fwd_bwd",
            "target_builtin_masked_fwd_bwd",
            "target_trainer_fwd_bwd",
            "target_hidden_fwd_bwd",
        ):
            register_case(name, requests, request_stats)

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            if "target_builtin_fwd" in benchmarks:
                assert target_items is not None and target_prepared is not None
                results["target_builtin_fwd_ms"] = _bench(
                    lambda: _builtin(
                        rank,
                        target_prepared,
                        _packed_labels(target_items, target_prepared),
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            if "target_hidden_fwd" in benchmarks:
                assert target_items is not None and target_prepared is not None
                results["target_hidden_fwd_ms"] = _bench(
                    lambda: rank._project_head(
                        target_items,
                        target_prepared,
                        rank._gather_sequence_parallel_hidden(
                            rank._decoder_hidden(target_prepared)
                        ),
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            if "target_trainer_fwd" in benchmarks:
                assert target_items is not None and target_prepared is not None
                results["target_trainer_fwd_ms"] = _bench(
                    lambda: rank._forward_packed(target_items, target_prepared),
                    warmup=warmup,
                    repeat=repeat,
                )
            if "logits_builtin_fwd" in benchmarks:
                assert logits_prepared is not None
                register_case(
                    "logits_builtin_fwd", _logits_requests(requests), request_stats
                )
                results["logits_builtin_fwd_ms"] = _bench(
                    lambda: _full_logits(rank, logits_prepared),
                    warmup=warmup,
                    repeat=repeat,
                )
            if "logits_hidden_fwd" in benchmarks:
                assert logits_items is not None and logits_prepared is not None
                register_case(
                    "logits_hidden_fwd", _logits_requests(requests), request_stats
                )
                results["logits_hidden_fwd_ms"] = _bench(
                    lambda: rank._project_head(
                        logits_items,
                        logits_prepared,
                        rank._gather_sequence_parallel_hidden(
                            rank._decoder_hidden(logits_prepared)
                        ),
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            trainer_cases = {
                "trainer_target": requests,
                "trainer_multi_target": multi_target_requests,
                "trainer_topk": [
                    ForwardInput(input_tokens=request.input_tokens, top_k=top_k)
                    for request in requests
                ],
                "trainer_target_topk": [
                    ForwardInput(
                        input_tokens=request.input_tokens,
                        target_tokens=request.target_tokens,
                        top_k=top_k,
                    )
                    for request in requests
                ],
                "trainer_hidden": [
                    ForwardInput(input_tokens=request.input_tokens, hidden_states=True)
                    for request in requests
                ],
                "trainer_all_no_logits": [
                    ForwardInput(
                        input_tokens=request.input_tokens,
                        target_tokens=multi_request.target_tokens,
                        top_k=top_k,
                        hidden_states=True,
                    )
                    for request, multi_request in zip(
                        requests, multi_target_requests, strict=True
                    )
                ],
                "trainer_logits": [
                    ForwardInput(input_tokens=request.input_tokens, logits=True)
                    for request in requests
                ],
            }
            if "trainer_topk_sweep" in benchmarks:
                for k in _int_values(top_k_values):
                    trainer_cases[f"trainer_topk_{k}"] = [
                        ForwardInput(input_tokens=request.input_tokens, top_k=k)
                        for request in requests
                    ]
            for name, case_requests in trainer_cases.items():
                if name not in benchmarks and not (
                    "trainer_topk_sweep" in benchmarks
                    and name.startswith("trainer_topk_")
                ):
                    continue
                output_gb = _request_output_gb(
                    case_requests,
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    dtype_size=dtype_size,
                )
                metadata[f"{name}_output_gb"] = round(output_gb, 3)
                if max_unpacked_output_gb > 0 and output_gb > max_unpacked_output_gb:
                    metadata[f"{name}_skipped"] = "unpacked_output_cap"
                    continue
                items = [rank._forward_item(request) for request in case_requests]
                batch = _pack_forward_items(
                    items,
                    max_depth=rank.shared_prefix_max_depth,
                )
                register_case(
                    name,
                    case_requests,
                    _packed_request_stats(
                        case_requests, items, batch, request_metadata={}
                    ),
                )
                prepared = rank._prepare_packed_forward(batch)
                results[f"{name}_ms"] = _bench(
                    lambda items=items, prepared=prepared: rank._forward_packed(
                        items,
                        prepared,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            if "trainer_topk_head" in benchmarks:
                case_requests = [
                    ForwardInput(input_tokens=request.input_tokens, top_k=top_k)
                    for request in requests
                ]
                output_gb = _request_output_gb(
                    case_requests,
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    dtype_size=dtype_size,
                )
                metadata["trainer_topk_head_output_gb"] = round(output_gb, 3)
                items = [rank._forward_item(request) for request in case_requests]
                batch = _pack_forward_items(
                    items,
                    max_depth=rank.shared_prefix_max_depth,
                )
                register_case(
                    "trainer_topk_head",
                    case_requests,
                    _packed_request_stats(
                        case_requests, items, batch, request_metadata={}
                    ),
                )
                prepared = rank._prepare_packed_forward(batch)
                hidden = rank._gather_sequence_parallel_hidden(
                    rank._decoder_hidden(prepared)
                )
                results["trainer_topk_head_ms"] = _bench(
                    lambda: rank._project_head(items, prepared, hidden),
                    warmup=warmup,
                    repeat=repeat,
                )

        if "target_builtin_fwd_bwd" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            assert target_items is not None and target_prepared is not None
            results["target_builtin_fwd_bwd_ms"] = _bench(
                lambda: _target_builtin_loss(
                    rank,
                    target_items,
                    target_prepared,
                ).backward(),
                warmup=warmup,
                repeat=repeat,
                after=rank.zero_grad,
            )
        if "target_builtin_masked_fwd_bwd" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            assert target_items is not None and target_prepared is not None
            results["target_builtin_masked_fwd_bwd_ms"] = _bench(
                lambda: _target_builtin_masked_loss(
                    rank,
                    target_items,
                    target_prepared,
                ).backward(),
                warmup=warmup,
                repeat=repeat,
                after=rank.zero_grad,
            )
        if "target_trainer_fwd_bwd" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            assert target_items is not None and target_prepared is not None
            results["target_trainer_fwd_bwd_ms"] = _bench(
                lambda: _target_trainer_loss(
                    rank,
                    target_items,
                    target_prepared,
                ).backward(),
                warmup=warmup,
                repeat=repeat,
                after=rank.zero_grad,
            )
        if "target_hidden_fwd_bwd" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            assert target_items is not None and target_prepared is not None
            results["target_hidden_fwd_bwd_ms"] = _bench(
                lambda: _target_hidden_loss(
                    rank,
                    target_items,
                    target_prepared,
                ).backward(),
                warmup=warmup,
                repeat=repeat,
                after=rank.zero_grad,
            )
        if "trainer_multi_target_fwd_bwd" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            items = [rank._forward_item(request) for request in multi_target_requests]
            batch = _pack_forward_items(
                items,
                max_depth=rank.shared_prefix_max_depth,
            )
            register_case(
                "trainer_multi_target_fwd_bwd",
                multi_target_requests,
                _packed_request_stats(
                    multi_target_requests,
                    items,
                    batch,
                    request_metadata={},
                ),
            )
            prepared = rank._prepare_packed_forward(batch)
            results["trainer_multi_target_fwd_bwd_ms"] = _bench(
                lambda: _target_trainer_loss(rank, items, prepared).backward(),
                warmup=warmup,
                repeat=repeat,
                after=rank.zero_grad,
            )
        if "trainer_topk_fwd_bwd" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            topk_requests = [
                ForwardInput(input_tokens=request.input_tokens, top_k=top_k)
                for request in requests
            ]
            items = [rank._forward_item(request) for request in topk_requests]
            batch = _pack_forward_items(
                items,
                max_depth=rank.shared_prefix_max_depth,
            )
            register_case(
                "trainer_topk_fwd_bwd",
                topk_requests,
                _packed_request_stats(topk_requests, items, batch, request_metadata={}),
            )
            prepared = rank._prepare_packed_forward(batch)
            results["trainer_topk_fwd_bwd_ms"] = _bench(
                lambda: _trainer_topk_loss(rank, items, prepared).backward(),
                warmup=warmup,
                repeat=repeat,
                after=rank.zero_grad,
            )

        if dist.get_rank() == 0:
            token_rates = _rate_metrics(results, rate_units)
            payload = {
                "world": dist.get_world_size(),
                "tp": int(ps.get_tensor_model_parallel_world_size()),
                "cp": int(ps.get_context_parallel_world_size()),
                "seq_len": seq_len,
                "prefix_families": prefix_families,
                "prefix_len": prefix_len,
                "mid_prefixes_per_family": mid_prefixes_per_family,
                "mid_prefix_len": mid_prefix_len,
                "branches_per_prefix": branches_per_prefix,
                "completion_len": completion_len,
                "head_chunk_tokens": head_chunk_tokens,
                "shared_prefix_max_depth": shared_prefix_max_depth,
                "warmup": warmup,
                "repeat": repeat,
                "target_count": target_count,
                "top_k": top_k,
                "top_k_values": top_k_values,
                "max_unpacked_output_gb": max_unpacked_output_gb,
                "mask_prefix_targets": mask_prefix_targets,
                "workload": workload,
                "tree_depth": tree_depth,
                "tree_seed": tree_seed,
                "tree_duplicate_factor": tree_duplicate_factor,
                "adapter_slots": adapter_slots,
                "adapter_slot_mode": adapter_slot_mode,
                "adapter_slot_rank": adapter_slot_rank,
                "adapter_loaded_sites": loaded_sites,
                "mtp_num_layers": getattr(model_config, "mtp_num_layers", None),
                "cross_entropy_loss_fusion": getattr(
                    model_config, "cross_entropy_loss_fusion", None
                ),
                "cross_entropy_fusion_impl": getattr(
                    model_config, "cross_entropy_fusion_impl", None
                ),
                **request_stats,
                "peak_memory_gb": round(
                    torch.cuda.max_memory_allocated() / 1024**3,
                    3,
                ),
                **results,
                **token_rates,
                **metadata,
                **planner_metadata,
            }
            line = json.dumps(payload, sort_keys=True)
            print(line, flush=True)
            if output_jsonl:
                output_path = Path(output_jsonl)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("a", encoding="utf-8") as output_file:
                    output_file.write(line + "\n")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _requests(
    *,
    seq_len: int,
    prefix_families: int,
    prefix_len: int,
    mid_prefixes_per_family: int,
    mid_prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
    target_count: int,
    mask_prefix_targets: bool,
    workload: str,
    tree_depth: int,
    tree_seed: int,
    tree_duplicate_factor: int,
) -> tuple[
    list[ForwardInput[torch.Tensor, None, None, None]],
    list[ForwardInput[torch.Tensor, None, None, None]],
    dict[str, int | str],
]:
    if workload == "regular" and prefix_families <= 0:
        tokens = torch.arange(seq_len, dtype=torch.long) % 32_000 + 100
        labels = _labels(tokens, target_count=1)
        return (
            [ForwardInput(input_tokens=tokens, target_tokens=labels)],
            [
                ForwardInput(
                    input_tokens=tokens,
                    target_tokens=_labels(tokens, target_count=target_count),
                )
            ],
            {
                "request_count": 1,
                "workload_shape": "single",
            },
        )

    if prefix_len < 1 or branches_per_prefix < 1 or completion_len < 1:
        raise ValueError(
            "prefix_len, branches_per_prefix, and completion_len must be >= 1"
        )
    if mid_prefixes_per_family < 1 or mid_prefix_len < 0:
        raise ValueError("mid_prefixes_per_family must be >= 1 and mid_prefix_len >= 0")

    sequences, prefix_lengths, workload_shape = _workload_sequences(
        workload=workload,
        seq_len=seq_len,
        prefix_families=max(prefix_families, 1),
        prefix_len=prefix_len,
        mid_prefixes_per_family=mid_prefixes_per_family,
        mid_prefix_len=mid_prefix_len,
        branches_per_prefix=branches_per_prefix,
        completion_len=completion_len,
        tree_depth=tree_depth,
        tree_seed=tree_seed,
        tree_duplicate_factor=tree_duplicate_factor,
    )
    requests = []
    multi_requests = []
    for tokens, shared_length in zip(sequences, prefix_lengths, strict=True):
        labels = _labels(tokens, target_count=1)
        multi_labels = _labels(tokens, target_count=target_count)
        if mask_prefix_targets and shared_length:
            labels[:shared_length] = -100
            multi_labels[:shared_length] = -100
        requests.append(ForwardInput(input_tokens=tokens, target_tokens=labels))
        multi_requests.append(
            ForwardInput(input_tokens=tokens, target_tokens=multi_labels)
        )

    return (
        requests,
        multi_requests,
        {
            "request_count": len(requests),
            "workload_shape": workload_shape,
        },
    )


def _load_adapter_slots(
    rank: TrainerRank,
    *,
    count: int,
    slot_rank: int,
) -> int:
    loaded_sites = 0
    for slot_index in range(count):
        loaded_sites += rank.load_checkpoint_slot(
            f"S{slot_index}",
            _synthetic_adapter(
                rank.runtime.model, slot_rank=slot_rank, seed=slot_index
            ),
        )
    return loaded_sites


def _synthetic_adapter(
    model: Sequence[torch.nn.Module],
    *,
    slot_rank: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    from art.megatron.lora import LoRA

    adapter: dict[str, torch.Tensor] = {}
    generator = torch.Generator(device="cuda").manual_seed(10_000 + seed)
    for chunk in model:
        for module in chunk.modules():
            if not isinstance(module, LoRA):
                continue
            a_keys = module._expected_weight_keys("lora_A")
            b_keys = module._expected_weight_keys("lora_B")
            for a_key, b_key in zip(a_keys, b_keys, strict=True):
                adapter[a_key] = (
                    torch.randn(
                        slot_rank,
                        module.in_features,
                        dtype=module.A_T.dtype,
                        device=module.A_T.device,
                        generator=generator,
                    )
                    * 0.01
                )
                adapter[b_key] = (
                    torch.randn(
                        module.out_features,
                        slot_rank,
                        dtype=module.B_T.dtype,
                        device=module.B_T.device,
                        generator=generator,
                    )
                    * 0.01
                )
    if not adapter:
        raise RuntimeError("adapter slot stress requested, but model has no LoRA sites")
    return adapter


def _route_adapter_slots(
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    *,
    adapter_slots: int,
    mode: str,
) -> list[
    ForwardInput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ]
]:
    if adapter_slots == 0:
        return list(requests)
    if mode not in {"family", "round_robin", "single"}:
        raise ValueError(
            "adapter_slot_mode must be one of: family, round_robin, single"
        )
    return [
        ForwardInput(
            input_tokens=request.input_tokens,
            target_tokens=request.target_tokens,
            top_k=request.top_k,
            logits=request.logits,
            hidden_states=request.hidden_states,
            checkpoint=f"S{_adapter_slot_index(index, request, adapter_slots, mode)}",
        )
        for index, request in enumerate(requests)
    ]


def _adapter_slot_index(
    index: int,
    request: ForwardInput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ],
    adapter_slots: int,
    mode: str,
) -> int:
    if mode == "single":
        return 0
    if mode == "round_robin":
        return index % adapter_slots
    first_token = (
        int(request.input_tokens[0].item()) if request.input_tokens.numel() else 0
    )
    return (first_token // 10_000_019) % adapter_slots


def _workload_sequences(
    *,
    workload: str,
    seq_len: int,
    prefix_families: int,
    prefix_len: int,
    mid_prefixes_per_family: int,
    mid_prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
    tree_depth: int,
    tree_seed: int,
    tree_duplicate_factor: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], str]:
    if workload in {"austin_198k", "austin_5k_16x100"}:
        return _regular_tree_sequences(
            prefix_families=30,
            prefix_len=5000,
            mid_prefixes_per_family=1,
            mid_prefix_len=0,
            branches_per_prefix=16,
            completion_len=100,
        )
    if workload == "regular":
        return _regular_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=mid_prefixes_per_family,
            mid_prefix_len=mid_prefix_len,
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
        )
    if workload == "single":
        tokens = torch.arange(seq_len, dtype=torch.long) % 32_000 + 100
        return (tokens,), (0,), "single"
    if workload == "long_root":
        return _regular_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=1,
            mid_prefix_len=0,
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
        )
    if workload == "long_mid":
        return _regular_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=max(2, mid_prefixes_per_family),
            mid_prefix_len=max(1, mid_prefix_len),
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
        )
    if workload == "many_tiny_leaves":
        return _regular_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=max(1, mid_prefixes_per_family),
            mid_prefix_len=max(0, mid_prefix_len),
            branches_per_prefix=branches_per_prefix,
            completion_len=max(1, completion_len),
        )
    if workload == "uneven":
        return _uneven_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=max(2, mid_prefixes_per_family),
            mid_prefix_len=max(1, mid_prefix_len),
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
        )
    if workload == "duplicates":
        sequences, shared, shape = _regular_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=max(2, mid_prefixes_per_family),
            mid_prefix_len=max(1, mid_prefix_len),
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
        )
        factor = max(1, tree_duplicate_factor)
        return (
            tuple(sequence for sequence in sequences for _ in range(factor)),
            tuple(length for length in shared for _ in range(factor)),
            f"{shape}:duplicates={factor}",
        )
    if workload == "random":
        return _random_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            branches_per_prefix=max(2, min(branches_per_prefix, 4)),
            completion_len=completion_len,
            tree_depth=max(1, tree_depth),
            seed=tree_seed,
        )
    raise ValueError(
        "workload must be one of: regular, single, long_root, long_mid, "
        "many_tiny_leaves, uneven, duplicates, random, austin_198k"
    )


def _regular_tree_sequences(
    *,
    prefix_families: int,
    prefix_len: int,
    mid_prefixes_per_family: int,
    mid_prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], str]:
    nested = mid_prefixes_per_family > 1 and mid_prefix_len > 0
    sequences: list[torch.Tensor] = []
    shared_lengths: list[int] = []
    for family in range(prefix_families):
        family_base = family * 10_000_019
        root = _tokens(family_base, prefix_len)
        mid_count = mid_prefixes_per_family if nested else 1
        for mid in range(mid_count):
            mid_prefix = (
                _tokens(family_base + 1_000_003 + mid * 100_003, mid_prefix_len)
                if nested
                else torch.empty(0, dtype=torch.long)
            )
            shared = torch.cat((root, mid_prefix))
            for branch in range(branches_per_prefix):
                sequences.append(
                    torch.cat(
                        (
                            shared,
                            _tokens(
                                family_base + mid * 100_003 + branch * 1009 + 17,
                                completion_len,
                            ),
                        )
                    )
                )
                shared_lengths.append(int(shared.numel()))
    shape = (
        f"families={prefix_families}:mid={mid_prefixes_per_family}:"
        f"branches={branches_per_prefix}:nested={int(nested)}"
    )
    return tuple(sequences), tuple(shared_lengths), shape


def _uneven_tree_sequences(
    *,
    prefix_families: int,
    prefix_len: int,
    mid_prefixes_per_family: int,
    mid_prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], str]:
    sequences: list[torch.Tensor] = []
    shared_lengths: list[int] = []
    for family in range(prefix_families):
        family_base = family * 10_000_019
        root_len = max(1, prefix_len // (family + 1))
        root = _tokens(family_base, root_len)
        for mid in range(mid_prefixes_per_family):
            mid_len = max(1, mid_prefix_len // (mid + 1))
            mid_prefix = _tokens(family_base + 1_000_003 + mid * 100_003, mid_len)
            branch_count = max(1, branches_per_prefix - mid)
            for branch in range(branch_count):
                leaf_len = max(1, completion_len * (branch + 1) // branch_count)
                shared = torch.cat((root, mid_prefix))
                sequences.append(
                    torch.cat(
                        (
                            shared,
                            _tokens(
                                family_base + mid * 100_003 + branch * 1009 + 17,
                                leaf_len,
                            ),
                        )
                    )
                )
                shared_lengths.append(int(shared.numel()))
    return tuple(sequences), tuple(shared_lengths), "uneven"


def _random_tree_sequences(
    *,
    prefix_families: int,
    prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
    tree_depth: int,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], str]:
    generator = torch.Generator().manual_seed(seed)
    next_offset = 1
    sequences: list[torch.Tensor] = []
    shared_lengths: list[int] = []

    def randint(low: int, high: int) -> int:
        return int(torch.randint(low, high + 1, (), generator=generator).item())

    def segment(length: int) -> torch.Tensor:
        nonlocal next_offset
        out = _tokens(next_offset, max(1, length))
        next_offset += max(1, length) + 10_000
        return out

    def length_for_depth(depth: int) -> int:
        if depth == 0:
            return max(1, prefix_len)
        choices = (1, 8, 64, max(1, completion_len), max(1, prefix_len // 2))
        return choices[randint(0, len(choices) - 1)]

    def walk(prefix: torch.Tensor, depth: int) -> None:
        shared = torch.cat((prefix, segment(length_for_depth(depth))))
        if depth + 1 >= tree_depth:
            leaf_count = randint(2, branches_per_prefix)
            for _ in range(leaf_count):
                leaf = segment(randint(1, max(1, completion_len)))
                sequences.append(torch.cat((shared, leaf)))
                shared_lengths.append(int(shared.numel()))
            return
        for _ in range(randint(2, branches_per_prefix)):
            walk(shared, depth + 1)

    for _ in range(prefix_families):
        walk(torch.empty(0, dtype=torch.long), 0)
    return tuple(sequences), tuple(shared_lengths), f"random:depth={tree_depth}"


def _packed_request_stats(
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    items: Sequence[object],
    batch: object,
    *,
    request_metadata: dict[str, int | str],
) -> dict[str, int | str]:
    from art.megatron.shared_prefix_tree import max_shared_prefix_tree_depth

    trainable_mask = torch.zeros(int(batch.tokens.numel()), dtype=torch.bool)
    trainable_tokens = 0
    for item, positions in zip(items, batch.positions_by_item, strict=True):
        labels = getattr(item, "labels", None)
        if labels is None:
            continue
        mask = labels != -100
        row_mask = mask.reshape(int(mask.shape[0]), -1).any(dim=1)
        trainable_tokens += int(mask.sum().item())
        trainable_mask[positions.reshape(-1).cpu()] |= row_mask.cpu()
    group_ids = batch.group_ids
    parent_ids = batch.parent_ids
    return {
        **request_metadata,
        "request_count": len(requests),
        "packed_tokens": int(batch.tokens.numel()),
        "logical_tokens": sum(
            int(request.input_tokens.numel()) for request in requests
        ),
        "trainable_tokens": trainable_tokens,
        "packed_trainable_tokens": int(trainable_mask.sum().item()),
        "packed_group_count": int(group_ids.max().item())
        if int(group_ids.numel())
        else 0,
        "nested_prefix_depth": max_shared_prefix_tree_depth(
            group_ids=group_ids,
            parent_ids=parent_ids,
        ),
    }


def _gather_planner_metadata(prepared: object) -> dict[str, object]:
    local = _local_planner_metadata(prepared)
    gathered: list[dict[str, object] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if dist.get_rank() != 0:
        return {}
    ranks = [metrics or {} for metrics in gathered]
    gdn_tokens = [int(metrics.get("gdn_tokens", 0)) for metrics in ranks]
    attention_tokens = [int(metrics.get("attention_tokens", 0)) for metrics in ranks]
    keys = (
        "tree_local_bucket_count",
        "tree_chain_bucket_count",
        "tree_local_segment_count",
        "tree_chain_segment_count",
        "tree_local_real_tokens",
        "tree_chain_real_tokens",
        "tree_state_transfer_count",
        "tree_state_transfer_rows",
        "tree_max_padding_ratio",
    )
    merged: dict[str, object] = {
        "planner_rank_gdn_tokens": gdn_tokens,
        "planner_rank_attention_tokens": attention_tokens,
        "planner_gdn_token_imbalance": max(gdn_tokens, default=0)
        - min(gdn_tokens, default=0),
    }
    for key in keys:
        values = [metrics[key] for metrics in ranks if key in metrics]
        if not values:
            continue
        if key.endswith("_ratio"):
            merged[f"planner_{key}_max"] = round(
                max(float(value) for value in values), 3
            )
        else:
            merged[f"planner_{key}_sum"] = int(sum(int(value) for value in values))
            merged[f"planner_{key}_max"] = int(max(int(value) for value in values))
    rank0 = ranks[0] if ranks else {}
    for key in ("tree_depth_count", "tree_family_count", "tree_completion_count"):
        if key in rank0:
            merged[f"planner_{key}"] = rank0[key]
    return merged


def _local_planner_metadata(prepared: object) -> dict[str, object]:
    plan = getattr(
        getattr(prepared, "attention_state", None), "gdn_execution_plan", None
    )
    if plan is None:
        return {}
    local_buckets = tuple(
        bucket
        for depth in getattr(plan, "tree_segment_buckets_by_depth", ())
        for bucket in depth
    )
    chain_buckets = tuple(
        bucket
        for depth in getattr(plan, "tree_chain_buckets_by_depth", ())
        for bucket in depth
    )
    all_buckets = (*local_buckets, *chain_buckets)
    padding_ratios = [
        bucket.length * bucket.segment_count / max(1, bucket.real_token_count)
        for bucket in all_buckets
    ]
    transfers_by_depth = getattr(plan, "tree_state_transfers_by_depth", ())
    return {
        "attention_tokens": int(getattr(plan, "attention_token_count", 0)),
        "gdn_tokens": int(getattr(plan, "gdn_token_count", 0)),
        "tree_depth_count": len(getattr(plan, "tree_segment_buckets_by_depth", ())),
        "tree_family_count": int(getattr(plan, "family_count", 0)),
        "tree_completion_count": int(getattr(plan, "completion_count", 0)),
        "tree_local_bucket_count": len(local_buckets),
        "tree_chain_bucket_count": len(chain_buckets),
        "tree_local_segment_count": sum(
            bucket.segment_count for bucket in local_buckets
        ),
        "tree_chain_segment_count": sum(
            bucket.segment_count for bucket in chain_buckets
        ),
        "tree_local_real_tokens": sum(
            bucket.real_token_count for bucket in local_buckets
        ),
        "tree_chain_real_tokens": sum(
            bucket.real_token_count for bucket in chain_buckets
        ),
        "tree_state_transfer_count": sum(
            len(transfers) for transfers in transfers_by_depth
        ),
        "tree_state_transfer_rows": sum(
            len(transfer.family_indices)
            for transfers in transfers_by_depth
            for transfer in transfers
        ),
        "tree_max_padding_ratio": max(padding_ratios, default=1.0),
    }


def _tokens(offset: int, length: int) -> torch.Tensor:
    return (torch.arange(length, dtype=torch.long) + offset) % 32_000 + 100


def _int_values(value: str) -> list[int]:
    values = [int(part) for part in value.split(",") if part.strip()]
    if not values or any(item < 1 for item in values):
        raise ValueError("top_k_values must contain positive integers")
    return values


def _labels(tokens: torch.Tensor, *, target_count: int) -> torch.Tensor:
    labels = torch.stack(
        [((tokens * 7 + 3 + index) % 32_000) for index in range(target_count)],
        dim=1,
    )
    if target_count > 1:
        labels[::17, -1] = -100
        return labels
    return labels[:, 0]


def _bench(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
    after: Callable[[], object] | None = None,
) -> float:
    for _ in range(warmup):
        fn()
        if after is not None:
            after()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
        if after is not None:
            after()
    stop.record()
    torch.cuda.synchronize()
    elapsed = torch.tensor(start.elapsed_time(stop) / repeat, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return round(float(elapsed.item()), 3)


def _builtin(
    rank: TrainerRank,
    prepared: object,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    from art.megatron.train import _placeholder_attention_mask

    return rank.runtime.model[0](
        input_ids=prepared.tokens,
        position_ids=prepared.position_ids,
        attention_mask=_placeholder_attention_mask(rank.device),
        labels=labels,
        packed_seq_params=prepared.packed_seq_params,
        **rank._handler().get_forward_kwargs(
            rank.runtime.model[0],
            attention_bias=prepared.attention_state,
        ),
    )


def _full_logits(rank: TrainerRank, prepared: object) -> torch.Tensor:
    logits = rank._gather_tensor_parallel_logits(_builtin(rank, prepared, None))
    return _batch_seq_logits(logits, seq_len=int(prepared.tokens.shape[1]))


def _target_builtin_loss(
    rank: TrainerRank,
    items: object,
    prepared: object,
) -> torch.Tensor:
    return _builtin(rank, prepared, _packed_labels(items, prepared)).float().sum()


def _target_builtin_masked_loss(
    rank: TrainerRank,
    items: object,
    prepared: object,
) -> torch.Tensor:
    labels = _packed_labels(items, prepared)
    per_token_loss = _builtin(rank, prepared, labels).float().reshape(-1)
    valid = labels.reshape(-1) != -100
    return per_token_loss[valid].sum() + per_token_loss.sum() * 0.0


def _target_hidden_loss(
    rank: TrainerRank,
    items: object,
    prepared: object,
) -> torch.Tensor:
    hidden = rank._gather_sequence_parallel_hidden(rank._decoder_hidden(prepared))
    outputs = rank._project_head(items, prepared, hidden)
    losses = [
        -target_logprobs.sum()
        for target_logprobs in outputs.target_logprobs
        if target_logprobs is not None
    ]
    if not losses:
        raise RuntimeError("target logprobs were not produced")
    return torch.stack(losses).sum()


def _target_trainer_loss(
    rank: TrainerRank,
    items: object,
    prepared: object,
) -> torch.Tensor:
    outputs = rank._forward_packed(items, prepared)
    losses = [
        -output.target_logprobs.sum()
        for output in outputs
        if output.target_logprobs is not None
    ]
    if not losses:
        raise RuntimeError("target logprobs were not produced")
    return torch.stack(losses).sum()


def _trainer_topk_loss(
    rank: TrainerRank,
    items: object,
    prepared: object,
) -> torch.Tensor:
    outputs = rank._forward_packed(items, prepared)
    losses = [
        -output.top_k.logprobs.sum() for output in outputs if output.top_k is not None
    ]
    if not losses:
        raise RuntimeError("top_k logprobs were not produced")
    return torch.stack(losses).sum()


def _runtime_output_shape(runtime: object) -> tuple[int, int, int]:
    provider = getattr(runtime, "provider")
    model = _language_model(getattr(runtime, "model")[0])
    hidden_size = int(
        getattr(provider, "hidden_size", None)
        or getattr(getattr(model, "config", None), "hidden_size", 0)
    )
    vocab_size = int(
        getattr(getattr(model, "config", None), "padded_vocab_size", None)
        or getattr(model, "vocab_size", 0)
    )
    dtype_size = next(getattr(runtime, "model")[0].parameters()).element_size()
    if hidden_size <= 0 or vocab_size <= 0:
        raise RuntimeError(
            f"could not infer output shape: hidden_size={hidden_size}, "
            f"vocab_size={vocab_size}"
        )
    return hidden_size, vocab_size, dtype_size


def _request_output_gb(
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    *,
    hidden_size: int,
    vocab_size: int,
    dtype_size: int,
) -> float:
    return (
        sum(
            _request_output_bytes(
                request,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                dtype_size=dtype_size,
            )
            for request in requests
        )
        / 1024**3
    )


def _request_output_bytes(
    request: ForwardInput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ],
    *,
    hidden_size: int,
    vocab_size: int,
    dtype_size: int,
) -> int:
    seq_len = int(request.input_tokens.numel())
    bytes_total = 0
    if request.target_tokens is not None:
        bytes_total += int(request.target_tokens.numel()) * 4
    if request.top_k is not None:
        bytes_total += seq_len * int(request.top_k) * (4 + 8)
    if request.logits:
        bytes_total += seq_len * vocab_size * dtype_size
    if request.hidden_states:
        bytes_total += seq_len * hidden_size * dtype_size
    return bytes_total


def _logits_requests(
    requests: Sequence[ForwardInput[torch.Tensor, None, None, None]],
) -> list[ForwardInput[None, None, torch.Tensor, None]]:
    return [
        ForwardInput(input_tokens=request.input_tokens, logits=True)
        for request in requests
    ]


def _rate_units(
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    stats: dict[str, int | str],
    *,
    hidden_size: int,
    vocab_size: int,
    dtype_size: int,
) -> dict[str, int]:
    return {
        "packed_tokens": int(stats.get("packed_tokens", 0)),
        "logical_tokens": int(stats.get("logical_tokens", 0)),
        "target_values": _target_value_count(requests),
        "output_bytes": sum(
            _request_output_bytes(
                request,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                dtype_size=dtype_size,
            )
            for request in requests
        ),
    }


def _target_value_count(
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
) -> int:
    count = 0
    for request in requests:
        if request.target_tokens is not None:
            count += int((request.target_tokens != -100).sum().item())
    return count


def _rate_metrics(
    results: dict[str, float],
    units_by_name: dict[str, dict[str, int]],
) -> dict[str, float]:
    suffixes = {
        "packed_tokens": "packed_tok_s",
        "logical_tokens": "logical_tok_s",
        "target_values": "target_logprob_s",
        "output_bytes": "output_gb_s",
    }
    metrics: dict[str, float] = {}
    for key, ms in results.items():
        if ms <= 0:
            continue
        name = key.removesuffix("_ms")
        units = units_by_name.get(name, {})
        for unit_key, suffix in suffixes.items():
            value = int(units.get(unit_key, 0))
            if value <= 0:
                continue
            scale = 1024**3 if unit_key == "output_bytes" else 1
            metrics[f"{name}_{suffix}"] = round(value * 1000.0 / ms / scale, 3)
    return metrics


def _packed_labels(items: object, prepared: object) -> torch.Tensor:
    labels = torch.full_like(prepared.tokens, -100)
    for item, positions, source_positions in zip(
        items,
        prepared.positions_by_item,
        prepared.source_positions_by_item,
        strict=True,
    ):
        if item.labels is None:
            continue
        labels.reshape(-1)[positions.to(device=labels.device)] = item.labels.to(
            device=labels.device
        ).index_select(0, source_positions.to(device=labels.device))
    return labels


if __name__ == "__main__":
    typer.run(main)
