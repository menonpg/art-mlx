from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

import torch

from .compressor import build_dsv4_compressed_layouts_from_cp_state
from .cp_attention import build_dsv4_attention_backward_plans_from_stage_plan_slots
from .cp_stage import (
    build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans_for_layouts,
    build_dsv4_stage_plan_slots,
)
from .indexer import (
    build_dsv4_indexer_kv_exchange_peer_plans,
    build_dsv4_indexer_stage_plan_from_stage_plans,
)
from .types import (
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    Dsv4ContextParallelState,
    Dsv4PreparedPlan,
)

if TYPE_CHECKING:
    from art.megatron.context_parallel.types import ArtContextParallelState
else:
    ArtContextParallelState = Any


@torch.compiler.disable
def prepare_dsv4_context_parallel_state(
    *,
    cp_state: ArtContextParallelState,
    csa_ratio: int = 4,
    hca_ratio: int = 128,
    include_csa: bool = True,
    include_hca: bool = True,
    extra: dict[str, Any] | None = None,
) -> Dsv4ContextParallelState:
    """Wrap ART CP state with DSV4 host-only planning metadata; must not synchronize CUDA."""
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        return _prepare_dsv4_context_parallel_state_impl(
            cp_state=cp_state,
            csa_ratio=csa_ratio,
            hca_ratio=hca_ratio,
            include_csa=include_csa,
            include_hca=include_hca,
            extra=extra,
        )
    finally:
        if gc_was_enabled:
            gc.enable()


def _prepare_dsv4_context_parallel_state_impl(
    *,
    cp_state: ArtContextParallelState,
    csa_ratio: int,
    hca_ratio: int,
    include_csa: bool,
    include_hca: bool,
    extra: dict[str, Any] | None,
) -> Dsv4ContextParallelState:
    """Build reusable DSV4 host-ahead metadata without reading activation tensors or CUDA state."""
    if not include_csa and not include_hca:
        raise RuntimeError("DSV4 planning requires CSA, HCA, or both")
    if (
        cp_state.group_ids.device.type != "cpu"
        or cp_state.parent_ids.device.type != "cpu"
    ):
        raise RuntimeError(
            "DSV4 CP planning expects CPU shared-prefix metadata. Passing CUDA "
            "group_ids or parent_ids would synchronize and break host-ahead planning."
        )
    if cp_state.group_ids.ndim != 1 or cp_state.parent_ids.ndim != 1:
        raise RuntimeError(
            "DSV4 CP planning expects rank-1 ART CP metadata, got "
            f"{tuple(cp_state.group_ids.shape)} and {tuple(cp_state.parent_ids.shape)}"
        )

    rank_int = int(cp_state.rank_plan.rank)
    runtime_plan = _runtime_plan_from_cp_state(cp_state)
    stage_slots = build_dsv4_stage_plan_slots(
        stage_plans_by_rank=tuple(
            rank_plan.stage_plans for rank_plan in runtime_plan.rank_plans
        ),
    )
    layout_names: list[str] = []
    layout_specs: list[Dsv4CompressionSpec] = []
    if include_csa:
        layout_names.append("csa")
        layout_specs.append(
            Dsv4CompressionSpec(
                kind=Dsv4CompressionKind.CSA,
                ratio=int(csa_ratio),
            )
        )
    if include_hca:
        layout_names.append("hca")
        layout_specs.append(
            Dsv4CompressionSpec(
                kind=Dsv4CompressionKind.HCA,
                ratio=int(hca_ratio),
            )
        )
    built_layouts = dict(
        zip(
            layout_names,
            build_dsv4_compressed_layouts_from_cp_state(
                state=cp_state,
                specs=tuple(layout_specs),
            ),
            strict=True,
        )
    )
    csa_layout = built_layouts.get("csa")
    hca_layout = built_layouts.get("hca")
    csa_indexer_stage_plans = (
        tuple(
            build_dsv4_indexer_stage_plan_from_stage_plans(
                layout=csa_layout,
                stage_plans_by_rank=slot.stage_plans_by_rank,
                local_rank=rank_int,
            )
            for slot in stage_slots
        )
        if csa_layout is not None
        else ()
    )
    csa_indexer_kv_peer_plans_by_stage = (
        tuple(
            build_dsv4_indexer_kv_exchange_peer_plans(
                layout=csa_layout,
                candidate_entry_ids_by_rank=stage_plan.candidate_entry_ids_by_rank,
                local_rank=rank_int,
            )
            for stage_plan in csa_indexer_stage_plans
        )
        if csa_layout is not None
        else ()
    )
    stage_kv_layout_names: list[str] = []
    stage_kv_layouts: list[Any] = []
    if csa_layout is not None:
        stage_kv_layout_names.append("csa")
        stage_kv_layouts.append(csa_layout)
    if hca_layout is not None:
        stage_kv_layout_names.append("hca")
        stage_kv_layouts.append(hca_layout)
    stage_kv_plans_by_name: dict[str, list[Any]] = {
        name: [] for name in stage_kv_layout_names
    }
    for slot_index, slot in enumerate(stage_slots):
        compressed_peer_plans_by_layout = tuple(
            csa_indexer_kv_peer_plans_by_stage[slot_index] if name == "csa" else None
            for name in stage_kv_layout_names
        )
        plans_for_slot = (
            build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans_for_layouts(
                layouts=tuple(stage_kv_layouts),
                stage_plans_by_rank=slot.stage_plans_by_rank,
                compressed_peer_plans_by_layout=compressed_peer_plans_by_layout,
            )
        )
        for name, plans in zip(stage_kv_layout_names, plans_for_slot, strict=True):
            stage_kv_plans_by_name[name].append(plans)
    csa_stage_kv_peer_plans_by_slot = tuple(stage_kv_plans_by_name.get("csa", ()))
    hca_stage_kv_peer_plans_by_slot = tuple(stage_kv_plans_by_name.get("hca", ()))
    backward_layout_names: list[str] = []
    backward_layouts: list[Any] = []
    if csa_layout is not None:
        backward_layout_names.append("csa")
        backward_layouts.append(csa_layout)
    if hca_layout is not None:
        backward_layout_names.append("hca")
        backward_layouts.append(hca_layout)
    backward_plans = dict(
        zip(
            backward_layout_names,
            build_dsv4_attention_backward_plans_from_stage_plan_slots(
                layouts=tuple(backward_layouts),
                stage_plan_slots=stage_slots,
                stage_kv_peer_plans_by_layout=tuple(
                    stage_kv_plans_by_name[name] for name in backward_layout_names
                ),
                local_rank=rank_int,
            ),
            strict=True,
        )
    )
    csa_attention_backward_plan = backward_plans.get("csa")
    hca_attention_backward_plan = backward_plans.get("hca")
    return Dsv4ContextParallelState.model_construct(
        cp_state=cp_state,
        dsv4_plan=Dsv4PreparedPlan.model_construct(
            csa_layout=csa_layout,
            hca_layout=hca_layout,
            stage_plan_slots=stage_slots,
            csa_indexer_stage_plans=csa_indexer_stage_plans,
            csa_indexer_kv_peer_plans_by_stage=csa_indexer_kv_peer_plans_by_stage,
            csa_stage_kv_peer_plans_by_slot=csa_stage_kv_peer_plans_by_slot,
            hca_stage_kv_peer_plans_by_slot=hca_stage_kv_peer_plans_by_slot,
            csa_attention_backward_plan=csa_attention_backward_plan,
            hca_attention_backward_plan=hca_attention_backward_plan,
        ),
        extra=dict(extra or {}),
    )


def _runtime_plan_from_cp_state(cp_state: ArtContextParallelState):
    from art.megatron.context_parallel import runtime as cp_runtime

    original_seq_len = int(cp_state.rank_plan.original_seq_len)
    runtime_plan = cp_runtime._RUNTIME_PLAN_CACHE.get(
        (cp_state.runtime_key, original_seq_len)
    )
    if runtime_plan is None:
        raise RuntimeError(
            "DSV4 CP planning requires the normal CP runtime plan prepared in the same host-ahead planning pass"
        )
    local_rank = int(cp_state.rank_plan.rank)
    if local_rank >= len(runtime_plan.rank_plans):
        raise RuntimeError(
            "DSV4 CP planning local rank is outside runtime plan: "
            f"{local_rank} >= {len(runtime_plan.rank_plans)}"
        )
    if tuple(
        int(stage.stage_index) for stage in cp_state.rank_plan.stage_plans
    ) != tuple(
        int(stage.stage_index)
        for stage in runtime_plan.rank_plans[local_rank].stage_plans
    ):
        raise RuntimeError(
            "DSV4 CP planning reconstructed a runtime plan inconsistent with the provided rank plan"
        )
    return runtime_plan
