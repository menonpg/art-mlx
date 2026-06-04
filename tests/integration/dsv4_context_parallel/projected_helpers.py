from __future__ import annotations

from typing import Any

from art.megatron.dsv4 import (
    launch_dsv4_csa_projected_attention_forward_from_compression_work,
    launch_dsv4_csa_projected_compression_forward,
    launch_dsv4_hca_projected_attention_forward_from_compression_work,
    launch_dsv4_hca_projected_compression_forward,
)


def launch_dsv4_csa_projected_attention_forward_from_stage_plan_slots(
    **kw: Any,
) -> Any:
    kw.pop("compression_kind", None)
    compression = launch_dsv4_csa_projected_compression_forward(
        layout=kw["layout"],
        rank=kw["rank"],
        main_projected_kv=kw["main_projected_kv"],
        main_projected_gate=kw["main_projected_gate"],
        main_positional_bias=kw["main_positional_bias"],
        main_token_ids=kw["main_token_ids"],
        indexer_projected_kv=kw["indexer_projected_kv"],
        indexer_projected_gate=kw["indexer_projected_gate"],
        indexer_positional_bias=kw["indexer_positional_bias"],
        indexer_token_ids=kw["indexer_token_ids"],
        group=kw["group"],
        async_op=kw["async_op"],
    )
    return launch_dsv4_csa_projected_attention_forward_from_compression_work(
        compression_work=compression,
        stage_plan_slots=kw["stage_plan_slots"],
        query=kw["query"],
        query_token_ids=kw["query_token_ids"],
        raw_kv=kw["raw_kv"],
        raw_token_ids=kw["raw_token_ids"],
        indexer_q=kw["indexer_q"],
        indexer_weights=kw["indexer_weights"],
        indexer_topk=kw["indexer_topk"],
        attn_sink=kw["attn_sink"],
        group=kw["group"],
        async_op=kw["async_op"],
        indexer_stage_plans=kw.get("indexer_stage_plans"),
        indexer_kv_peer_plans_by_stage=kw.get("indexer_kv_peer_plans_by_stage"),
        stage_kv_peer_plans_by_slot=kw.get("stage_kv_peer_plans_by_slot"),
        indexer_score_scale=kw.get("indexer_score_scale", 1.0),
        scale=kw.get("scale"),
        window_size=kw.get("window_size", 128),
        raw_list_size=kw.get("raw_list_size"),
        compressed_list_size=kw.get("compressed_list_size"),
    )


def launch_dsv4_hca_projected_attention_forward_from_stage_plan_slots(
    **kw: Any,
) -> Any:
    kw.pop("compression_kind", None)
    compression = launch_dsv4_hca_projected_compression_forward(
        layout=kw["layout"],
        rank=kw["rank"],
        projected_kv=kw["projected_kv"],
        projected_gate=kw["projected_gate"],
        positional_bias=kw["positional_bias"],
        token_ids=kw["token_ids"],
        group=kw["group"],
        async_op=kw["async_op"],
    )
    return launch_dsv4_hca_projected_attention_forward_from_compression_work(
        compression_work=compression,
        stage_plan_slots=kw["stage_plan_slots"],
        query=kw["query"],
        query_token_ids=kw["query_token_ids"],
        raw_kv=kw["raw_kv"],
        raw_token_ids=kw["raw_token_ids"],
        attn_sink=kw["attn_sink"],
        group=kw["group"],
        async_op=kw["async_op"],
        stage_kv_peer_plans_by_slot=kw.get("stage_kv_peer_plans_by_slot"),
        scale=kw.get("scale"),
        window_size=kw.get("window_size", 128),
        raw_list_size=kw.get("raw_list_size"),
        compressed_list_size=kw.get("compressed_list_size"),
    )
