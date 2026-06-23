from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
import time

import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import create_block_mask as torch_block_mask
import typer

from art.megatron.context_parallel.block_mask import (
    build_block_mask_from_context,
    prepare_block_mask_context,
)
from art.megatron.context_parallel.builder import build_shared_prefix_attention_spec
from art.megatron.context_parallel.runtime import (
    _RUNTIME_PLAN_CACHE,
    get_or_build_runtime_plan,
    make_runtime_key,
)
from art.megatron.context_parallel.types import (
    ContextParallelConfig,
    FlexMaskSpec,
    ParallelTopology,
)
from art.megatron.flex_attn.attention import FlexAttentionWrapper
from art.megatron.shared_prefix_packing import SharedPrefixPack, pack_shared_prefixes
from art.megatron.shared_prefix_state import create_shared_prefix_state


def main(
    workload: str = "austin_198k",
    max_depth: int = 1,
    cp_size: int = 4,
    block_size: int = 128,
    prefix_families: int = 4,
    prefix_len: int = 1024,
    mid_prefixes_per_family: int = 1,
    mid_prefix_len: int = 0,
    branches_per_prefix: int = 8,
    completion_len: int = 128,
    warmup: int = 3,
    repeat: int = 10,
    shape_variants: int = 4,
    validate_torch: bool = True,
    run_flex: bool = True,
    flex_token_cap: int = 8192,
    flex_heads: int = 2,
    flex_head_dim: int = 64,
    flex_mask_variants: str = "current,flat_pair,token_group,local_or_flat_pair",
    output_jsonl: Path = Path(".local/trainer_rank_review/block_mask_flex.jsonl"),
) -> None:
    if warmup < 0 or repeat < 1:
        raise ValueError("warmup must be >= 0 and repeat must be >= 1")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    pack = _pack_workload(
        workload=workload,
        max_depth=max_depth,
        prefix_families=prefix_families,
        prefix_len=prefix_len,
        mid_prefixes_per_family=mid_prefixes_per_family,
        mid_prefix_len=mid_prefix_len,
        branches_per_prefix=branches_per_prefix,
        completion_len=completion_len,
    )
    spec = build_shared_prefix_attention_spec(
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
    )
    config = ContextParallelConfig(block_size=block_size)
    topology = ParallelTopology(cp=cp_size)
    base = {
        "workload": workload,
        "max_depth": max_depth,
        "cp_size": cp_size,
        "block_size": block_size,
        "packed_tokens": int(pack.tokens.numel()),
        "logical_tokens": _logical_tokens(pack),
        "warmup": warmup,
        "repeat": repeat,
    }

    plan, plan_ms = _bench_cpu(
        lambda: _build_cp_plan(pack, spec, topology, config),
        warmup=warmup,
        repeat=repeat,
        before_each=_RUNTIME_PLAN_CACHE.clear,
    )
    _write(
        output_jsonl,
        {
            **base,
            "case": "cp_planning_cold",
            "ms": plan_ms,
            **_plan_stats(plan),
        },
    )

    cached_plan, cached_plan_ms = _bench_cpu(
        lambda: _build_cp_plan(pack, spec, topology, config),
        warmup=warmup,
        repeat=repeat,
    )
    _write(
        output_jsonl,
        {
            **base,
            "case": "cp_planning_cached",
            "ms": cached_plan_ms,
            **_plan_stats(cached_plan),
        },
    )

    stage_masks, mask_ms = _bench_cpu(
        lambda: _build_stage_masks(pack, plan, config),
        warmup=warmup,
        repeat=repeat,
    )
    masks = tuple(mask for mask, _ in stage_masks)
    if validate_torch:
        for mask, slices in stage_masks:
            _assert_matches_torch_block_mask(mask, slices=slices)
    _write(
        output_jsonl,
        {
            **base,
            "case": "block_mask_build",
            "ms": mask_ms,
            **_mask_stats(masks),
        },
    )

    if run_flex:
        for record in _flex_records(
            pack,
            warmup=warmup,
            repeat=repeat,
            token_cap=flex_token_cap,
            heads=flex_heads,
            head_dim=flex_head_dim,
            variants=_csv_values(flex_mask_variants),
        ):
            _write(output_jsonl, {**base, **record})

    for variant in range(shape_variants):
        variant_pack = _pack_workload(
            workload="regular",
            max_depth=max_depth,
            prefix_families=prefix_families,
            prefix_len=prefix_len + variant * 17,
            mid_prefixes_per_family=mid_prefixes_per_family,
            mid_prefix_len=mid_prefix_len + variant * 3,
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len + variant * 11,
        )
        variant_spec = build_shared_prefix_attention_spec(
            group_ids=variant_pack.group_ids,
            parent_ids=variant_pack.parent_ids,
        )
        variant_plan, variant_plan_ms = _bench_cpu(
            lambda pack=variant_pack, spec=variant_spec: _build_cp_plan(
                pack,
                spec,
                topology,
                config,
            ),
            warmup=0,
            repeat=1,
            before_each=_RUNTIME_PLAN_CACHE.clear,
        )
        variant_stage_masks, variant_mask_ms = _bench_cpu(
            lambda pack=variant_pack, plan=variant_plan: _build_stage_masks(
                pack,
                plan,
                config,
            ),
            warmup=0,
            repeat=1,
        )
        variant_masks = tuple(mask for mask, _ in variant_stage_masks)
        if validate_torch:
            for mask, slices in variant_stage_masks:
                _assert_matches_torch_block_mask(mask, slices=slices)
        _write(
            output_jsonl,
            {
                **base,
                "case": "shape_variant",
                "variant": variant,
                "variant_packed_tokens": int(variant_pack.tokens.numel()),
                "variant_logical_tokens": _logical_tokens(variant_pack),
                "cp_planning_ms": variant_plan_ms,
                "block_mask_build_ms": variant_mask_ms,
                **_plan_stats(variant_plan),
                **_mask_stats(variant_masks),
            },
        )

    print(f"wrote review perf records to {output_jsonl}", flush=True)


def _pack_workload(
    *,
    workload: str,
    max_depth: int,
    prefix_families: int,
    prefix_len: int,
    mid_prefixes_per_family: int,
    mid_prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
) -> SharedPrefixPack:
    sequences = (
        _austin_sequences()
        if workload == "austin_198k"
        else _austin_varied_sequences()
        if workload == "austin_varied"
        else _regular_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=mid_prefixes_per_family,
            mid_prefix_len=mid_prefix_len,
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
        )
    )
    return pack_shared_prefixes(sequences, max_depth=max_depth)


def _austin_sequences() -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.cat(
            (
                _tokens(family * 10_000_019, 5000),
                _tokens(family * 10_000_019 + branch * 1009 + 17, 100),
            )
        )
        for family in range(30)
        for branch in range(16)
    )


def _austin_varied_sequences() -> tuple[torch.Tensor, ...]:
    sequences: list[torch.Tensor] = []
    for family in range(30):
        family_base = family * 10_000_019
        prefix_len = 4500 + ((family * 137) % 1001)
        root = _tokens(family_base, prefix_len)
        branch_count = 10 + ((family * 7) % 13)
        for branch in range(branch_count):
            completion_len = 32 + ((family * 19 + branch * 23) % 145)
            sequences.append(
                torch.cat(
                    (
                        root,
                        _tokens(
                            family_base + branch * 1009 + 17,
                            completion_len,
                        ),
                    )
                )
            )
    return tuple(sequences)


def _regular_sequences(
    *,
    prefix_families: int,
    prefix_len: int,
    mid_prefixes_per_family: int,
    mid_prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
) -> tuple[torch.Tensor, ...]:
    sequences = []
    for family in range(max(1, prefix_families)):
        family_base = family * 10_000_019
        root = _tokens(family_base, max(1, prefix_len))
        for mid in range(max(1, mid_prefixes_per_family)):
            mid_prefix = _tokens(
                family_base + 1_000_003 + mid * 100_003,
                max(0, mid_prefix_len),
            )
            prefix = torch.cat((root, mid_prefix))
            for branch in range(max(1, branches_per_prefix)):
                sequences.append(
                    torch.cat(
                        (
                            prefix,
                            _tokens(
                                family_base + mid * 100_003 + branch * 1009 + 17,
                                max(1, completion_len),
                            ),
                        )
                    )
                )
    return tuple(sequences)


def _tokens(offset: int, length: int) -> torch.Tensor:
    return (torch.arange(length, dtype=torch.long) + offset) % 32_000 + 100


def _build_cp_plan(
    pack: SharedPrefixPack,
    spec: object,
    topology: ParallelTopology,
    config: ContextParallelConfig,
) -> object:
    return get_or_build_runtime_plan(
        spec,
        topology=topology,
        config=config,
        runtime_key=make_runtime_key(spec, topology=topology, config=config),
        original_seq_len=int(pack.tokens.numel()),
    )


def _build_stage_masks(
    pack: SharedPrefixPack,
    plan: object,
    config: ContextParallelConfig,
) -> tuple[tuple[BlockMask, tuple[object, ...]], ...]:
    masks = []
    context = prepare_block_mask_context(
        group_ids=pack.group_ids[0],
        parent_ids=pack.parent_ids[0],
    )
    for rank_plan in plan.rank_plans:
        for stage in rank_plan.stage_plans:
            if stage.mask_metadata is None:
                continue
            mask = build_block_mask_from_context(
                FlexMaskSpec(
                    q_len=stage.q_len,
                    k_len=stage.k_len,
                    block_size=config.block_size,
                    slices=stage.slices,
                    exact_mask=stage.mask_metadata,
                ),
                context=context,
                device=torch.device("cpu"),
                validate=False,
            )
            if mask is not None:
                masks.append((mask, tuple(stage.slices)))
    return tuple(masks)


def _flex_records(
    pack: SharedPrefixPack,
    *,
    warmup: int,
    repeat: int,
    token_cap: int,
    heads: int,
    head_dim: int,
    variants: Sequence[str],
) -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        return [{"case": "flex_attention_fwd_bwd", "skipped": "cuda_unavailable"}]
    if int(pack.tokens.numel()) > int(token_cap):
        return [
            {
                "case": "flex_attention_fwd_bwd",
                "skipped": "packed_tokens_exceed_flex_token_cap",
                "flex_token_cap": int(token_cap),
            }
        ]
    device = torch.device("cuda")
    group_ids = pack.group_ids.to(device)
    parent_ids = pack.parent_ids.to(device)
    attention_state = create_shared_prefix_state(
        group_ids,
        parent_ids,
        target_device=device,
    )
    shape = (1, int(heads), int(pack.tokens.numel()), int(head_dim))
    records: list[dict[str, object]] = []
    block_masks = _flex_mask_variants(
        attention_state.block_mask,
        pack,
        variants=variants,
        device=device,
    )
    for variant, block_mask in block_masks:
        q = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        wrapper = FlexAttentionWrapper()

        def step() -> None:
            q.grad = None
            k.grad = None
            v.grad = None
            out = wrapper(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=float(head_dim) ** -0.5,
                enable_gqa=False,
            )
            out.float().sum().backward()

        try:
            torch.cuda.synchronize()
            first_started = time.perf_counter()
            step()
            torch.cuda.synchronize()
            first_call_ms = round((time.perf_counter() - first_started) * 1000.0, 3)
            ms = _bench_cuda(step, warmup=warmup, repeat=repeat)
        except Exception as exc:
            torch.cuda.empty_cache()
            records.append(
                {
                    "case": "flex_attention_fwd_bwd",
                    "flex_mask_variant": variant,
                    "compile_error": type(exc).__name__,
                    "compile_error_message": str(exc).splitlines()[0][:500],
                    "flex_heads": heads,
                    "flex_head_dim": head_dim,
                }
            )
            continue
        records.append(
            {
                "case": "flex_attention_fwd_bwd",
                "flex_mask_variant": variant,
                "first_call_ms": first_call_ms,
                "ms": ms,
                "packed_tok_s": round(int(pack.tokens.numel()) * 1000.0 / ms, 3),
                "flex_heads": heads,
                "flex_head_dim": head_dim,
                "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            }
        )
    return records


def _flex_mask_variants(
    block_mask: BlockMask,
    pack: SharedPrefixPack,
    *,
    variants: Sequence[str],
    device: torch.device,
) -> tuple[tuple[str, BlockMask], ...]:
    group_ids = pack.group_ids[0].to(device=device, dtype=torch.long)
    can_attend = _group_can_attend(pack).to(device=device)
    token_group_can_attend = can_attend.index_select(0, group_ids)
    stride = int(can_attend.shape[1])
    can_attend_flat = can_attend.reshape(-1)
    out = []
    for variant in variants:
        if variant == "current":
            out.append((variant, block_mask))
            continue
        if variant == "flat_pair":

            def mask_mod(batch_idx, head_idx, query_idx, kv_idx):
                del batch_idx, head_idx
                q_group = group_ids[query_idx]
                k_group = group_ids[kv_idx]
                return (query_idx >= kv_idx) & can_attend_flat[
                    q_group * stride + k_group
                ]

        elif variant == "token_group":

            def mask_mod(batch_idx, head_idx, query_idx, kv_idx):
                del batch_idx, head_idx
                k_group = group_ids[kv_idx]
                return (query_idx >= kv_idx) & token_group_can_attend[
                    query_idx, k_group
                ]

        elif variant == "local_or_flat_pair":

            def mask_mod(batch_idx, head_idx, query_idx, kv_idx):
                del batch_idx, head_idx
                q_group = group_ids[query_idx]
                k_group = group_ids[kv_idx]
                allowed = (q_group == k_group) | can_attend_flat[
                    q_group * stride + k_group
                ]
                return (query_idx >= kv_idx) & allowed

        else:
            raise ValueError(f"unknown flex_mask_variant {variant!r}")
        out.append((variant, _replace_block_mask_mod(block_mask, mask_mod)))
    return tuple(out)


def _group_can_attend(pack: SharedPrefixPack) -> torch.Tensor:
    group_ids = pack.group_ids[0].to(dtype=torch.long).cpu()
    parent_ids = pack.parent_ids[0].to(dtype=torch.long).cpu()
    max_group = int(group_ids.max().item()) if int(group_ids.numel()) else 0
    parents = [0 for _ in range(max_group + 1)]
    for group, parent in zip(group_ids.tolist(), parent_ids.tolist(), strict=True):
        if int(group) >= 0:
            parents[int(group)] = max(0, int(parent))
    can_attend = torch.zeros((max_group + 1, max_group + 1), dtype=torch.bool)
    for group in range(1, max_group + 1):
        current = group
        seen: set[int] = set()
        while current > 0 and current not in seen:
            seen.add(current)
            can_attend[group, current] = True
            parent = parents[current]
            if parent == current:
                break
            current = parent
    return can_attend


def _replace_block_mask_mod(block_mask: BlockMask, mask_mod: object) -> BlockMask:
    return BlockMask(
        seq_lengths=block_mask.seq_lengths,
        kv_num_blocks=block_mask.kv_num_blocks,
        kv_indices=block_mask.kv_indices,
        full_kv_num_blocks=block_mask.full_kv_num_blocks,
        full_kv_indices=block_mask.full_kv_indices,
        q_num_blocks=block_mask.q_num_blocks,
        q_indices=block_mask.q_indices,
        full_q_num_blocks=block_mask.full_q_num_blocks,
        full_q_indices=block_mask.full_q_indices,
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
        mask_mod=mask_mod,
    )


def _bench_cpu(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
    before_each: Callable[[], object] | None = None,
) -> tuple[object, float]:
    result = None
    for _ in range(warmup):
        if before_each is not None:
            before_each()
        result = fn()
    elapsed = []
    for _ in range(repeat):
        if before_each is not None:
            before_each()
        start = time.perf_counter()
        result = fn()
        elapsed.append((time.perf_counter() - start) * 1000.0)
    assert result is not None
    return result, round(sum(elapsed) / len(elapsed), 3)


def _bench_cuda(fn: Callable[[], object], *, warmup: int, repeat: int) -> float:
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return round(float(start.elapsed_time(stop)) / repeat, 3)


def _plan_stats(plan: object) -> dict[str, int]:
    stage_count = 0
    remote_stage_count = 0
    mask_stage_count = 0
    for rank_plan in plan.rank_plans:
        for stage in rank_plan.stage_plans:
            stage_count += 1
            remote_stage_count += int(not stage.is_local_stage)
            mask_stage_count += int(stage.mask_metadata is not None)
    return {
        "rank_count": len(plan.rank_plans),
        "stage_count": stage_count,
        "remote_stage_count": remote_stage_count,
        "mask_stage_count": mask_stage_count,
    }


def _mask_stats(masks: Sequence[BlockMask]) -> dict[str, int]:
    return {
        "mask_count": len(masks),
        "partial_kv_blocks": sum(_block_count(mask, "kv_num_blocks") for mask in masks),
        "full_kv_blocks": sum(
            _block_count(mask, "full_kv_num_blocks") for mask in masks
        ),
        "partial_q_blocks": sum(_block_count(mask, "q_num_blocks") for mask in masks),
        "full_q_blocks": sum(_block_count(mask, "full_q_num_blocks") for mask in masks),
    }


def _block_count(block_mask: BlockMask, name: str) -> int:
    counts = getattr(block_mask, name)
    return 0 if counts is None else int(counts.sum().item())


def _assert_matches_torch_block_mask(
    block_mask: BlockMask,
    *,
    slices: Sequence[object] = (),
) -> None:
    q_len, k_len = block_mask.seq_lengths
    reference = torch_block_mask(
        _slice_mask_mod(block_mask.mask_mod, slices),
        B=int(block_mask.kv_num_blocks.shape[0]),
        H=1,
        Q_LEN=q_len,
        KV_LEN=k_len,
        device="cpu",
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
    )
    for counts_name, indices_name in (
        ("kv_num_blocks", "kv_indices"),
        ("full_kv_num_blocks", "full_kv_indices"),
        ("q_num_blocks", "q_indices"),
        ("full_q_num_blocks", "full_q_indices"),
    ):
        actual = _block_entries(block_mask, counts_name, indices_name)
        expected = _block_entries(reference, counts_name, indices_name)
        if actual != expected:
            raise AssertionError(f"{counts_name}/{indices_name} mismatch")


def _slice_mask_mod(mask_mod: object, slices: Sequence[object]) -> object:
    if not slices:
        return mask_mod

    def sliced_mask_mod(
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        query_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        in_slice = (query_idx < 0) & (kv_idx < 0)
        for slice_ in slices:
            in_slice |= (
                (query_idx >= int(slice_.q_range.start))
                & (query_idx < int(slice_.q_range.end))
                & (kv_idx >= int(slice_.k_range.start))
                & (kv_idx < int(slice_.k_range.end))
            )
        return in_slice & mask_mod(batch_idx, head_idx, query_idx, kv_idx)

    return sliced_mask_mod


def _block_entries(
    block_mask: BlockMask,
    counts_name: str,
    indices_name: str,
) -> set[tuple[int, int, int, int]]:
    counts = getattr(block_mask, counts_name)
    indices = getattr(block_mask, indices_name)
    if counts is None or indices is None:
        return set()
    entries = set()
    for batch_index in range(int(counts.shape[0])):
        for head_index in range(int(counts.shape[1])):
            for block_index in range(int(counts.shape[2])):
                block_count = int(counts[batch_index, head_index, block_index])
                for other_block in indices[
                    batch_index,
                    head_index,
                    block_index,
                    :block_count,
                ].tolist():
                    entries.add(
                        (
                            batch_index,
                            head_index,
                            block_index,
                            int(other_block),
                        )
                    )
    return entries


def _logical_tokens(pack: SharedPrefixPack) -> int:
    return sum(int(positions.numel()) for positions in pack.positions_by_sequence)


def _csv_values(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("CSV option must contain at least one value")
    return values


def _write(path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload, sort_keys=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    typer.run(main)
