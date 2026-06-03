from __future__ import annotations

from pydantic import BaseModel, ConfigDict
import pytest
import torch

from art.megatron.dsv4 import (
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    Dsv4StageKeyKind,
    build_dsv4_compressed_layout,
    build_dsv4_stage_inputs_from_stage_plan,
    build_dsv4_stage_kv_exchange_peer_plans,
    build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans,
    build_dsv4_stage_plan_slots,
    build_stage_local_topk_for_csa,
    build_stage_local_topk_for_hca,
    launch_dsv4_stage_kv_exchange_deferred_from_stage_plan_slot,
    launch_dsv4_stage_kv_exchange_from_stage_plan_slot,
    materialize_dsv4_stage_tensors,
    raw_swa_token_ids_for_query,
)
import art.megatron.dsv4.cp_stage as cp_stage


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


def test_raw_swa_visibility_uses_branch_views_not_physical_siblings() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    candidates = tuple(range(18))

    assert raw_swa_token_ids_for_query(
        layout=layout,
        query_token_id=7,
        candidate_token_ids=candidates,
        window_size=4,
    ) == (4, 5, 6, 7)
    assert raw_swa_token_ids_for_query(
        layout=layout,
        query_token_id=8,
        candidate_token_ids=candidates,
        window_size=4,
    ) == (5, 6, 7, 8)
    assert raw_swa_token_ids_for_query(
        layout=layout,
        query_token_id=14,
        candidate_token_ids=candidates,
        window_size=4,
    ) == (6, 7, 13, 14)


def test_csa_stage_inputs_remap_raw_and_selected_compressed_keys() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    stage = build_stage_local_topk_for_csa(
        layout=layout,
        stage_index=3,
        query_token_ids=(8, 11, 16),
        global_k_ranges=(_Range(start=8, end=13),),
        global_topk=torch.tensor(
            [
                [2, 0, -1],
                [2, 1, -1],
                [2, 3, -1],
            ],
            dtype=torch.long,
        ),
        window_size=4,
    )

    assert stage.stage_index == 3
    assert stage.query_token_ids == (8, 11, 16)
    assert stage.raw_token_ids == (8, 9, 10, 11, 12)
    assert stage.compressed_entry_ids == (2,)
    assert stage.key_kinds == (Dsv4StageKeyKind.RAW,) * 5 + (
        Dsv4StageKeyKind.COMPRESSED,
    )
    assert stage.key_global_ids == (8, 9, 10, 11, 12, 2)
    assert stage.raw_token_ids_by_query == ((8,), (8, 9, 10, 11), ())
    assert stage.compressed_entry_ids_by_query == ((), (2,), ())
    assert stage.topk_stage_local.shape == (1, 3, 7)
    assert stage.topk_stage_local[0, 0].tolist() == [0, -1, -1, -1, -1, -1, -1]
    assert stage.topk_stage_local[0, 1].tolist() == [0, 1, 2, 3, 5, -1, -1]
    assert stage.topk_stage_local[0, 2].tolist() == [-1, -1, -1, -1, -1, -1, -1]


def test_csa_stage_inputs_preserve_batch_specific_topk() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    stage = build_stage_local_topk_for_csa(
        layout=layout,
        stage_index=4,
        query_token_ids=(11,),
        global_k_ranges=(_Range(start=8, end=13), _Range(start=13, end=18)),
        global_topk=torch.tensor(
            [
                [[2, -1]],
                [[3, 2]],
            ],
            dtype=torch.long,
        ),
        window_size=4,
    )

    assert stage.raw_token_ids == tuple(range(8, 18))
    assert stage.compressed_entry_ids == (2, 3)
    assert stage.compressed_entry_ids_by_query == ((2,),)
    assert stage.topk_stage_local.shape == (2, 1, 6)
    assert stage.topk_stage_local[0, 0].tolist() == [0, 1, 2, 3, 10, -1]
    assert stage.topk_stage_local[1, 0].tolist() == [0, 1, 2, 3, 10, -1]


def test_csa_stage_inputs_tensorized_remap_filters_duplicates_without_metadata() -> (
    None
):
    layout = _layout(Dsv4CompressionKind.CSA)
    stage = build_stage_local_topk_for_csa(
        layout=layout,
        stage_index=4,
        query_token_ids=(11,),
        global_k_ranges=(_Range(start=8, end=13), _Range(start=13, end=18)),
        global_topk=torch.tensor([[[2, 2, 3, 2, -1]]], dtype=torch.long),
        window_size=4,
        materialize_compressed_metadata=False,
    )

    assert stage.compressed_entry_ids == (2, 3)
    assert stage.compressed_entry_ids_by_query == ((),)
    assert stage.topk_stage_local.shape == (1, 1, 9)
    assert stage.topk_stage_local[0, 0].tolist() == [
        0,
        1,
        2,
        3,
        10,
        -1,
        -1,
        -1,
        -1,
    ]


def test_hca_stage_inputs_include_all_visible_compressed_entries() -> None:
    layout = _layout(Dsv4CompressionKind.HCA)
    stage = build_stage_local_topk_for_hca(
        layout=layout,
        stage_index=5,
        query_token_ids=(8, 11, 16),
        global_k_ranges=(_Range(start=8, end=13),),
        window_size=4,
    )

    assert stage.raw_token_ids == (8, 9, 10, 11, 12)
    assert stage.compressed_entry_ids == (2,)
    assert stage.raw_token_ids_by_query == ((8,), (8, 9, 10, 11), ())
    assert stage.compressed_entry_ids_by_query == ((), (2,), ())
    assert stage.topk_stage_local.shape == (1, 3, 5)
    assert stage.topk_stage_local[0, 0].tolist() == [0, -1, -1, -1, -1]
    assert stage.topk_stage_local[0, 1].tolist() == [0, 1, 2, 3, 5]
    assert stage.topk_stage_local[0, 2].tolist() == [-1, -1, -1, -1, -1]


def test_stage_inputs_filter_padding_tokens_from_cp_k_ranges() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    stage = build_stage_local_topk_for_csa(
        layout=layout,
        stage_index=6,
        query_token_ids=(16,),
        global_k_ranges=(_Range(start=16, end=20),),
        global_topk=torch.tensor([[3, -1]], dtype=torch.long),
        window_size=4,
    )

    assert stage.raw_token_ids == (16, 17)
    assert stage.compressed_entry_ids == (3,)
    assert stage.key_global_ids == (16, 17, 3)
    assert stage.topk_stage_local[0, 0].tolist() == [0, -1, -1, -1, 2, -1]


def test_stage_kv_exchange_peer_plan_uses_layout_ownership() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    stages = (
        build_stage_local_topk_for_csa(
            layout=layout,
            stage_index=0,
            query_token_ids=(7,),
            global_k_ranges=(_Range(start=4, end=13),),
            global_topk=torch.tensor([[1, 2]], dtype=torch.long),
            window_size=4,
        ),
        build_stage_local_topk_for_csa(
            layout=layout,
            stage_index=0,
            query_token_ids=(16,),
            global_k_ranges=(_Range(start=8, end=20),),
            global_topk=torch.tensor([[3, 2]], dtype=torch.long),
            window_size=4,
        ),
    )

    plans = build_dsv4_stage_kv_exchange_peer_plans(
        layout=layout,
        stage_inputs_by_rank=stages,
    )

    assert plans[0].recv_raw_token_ids_by_peer == (
        (4, 5, 6, 7),
        (8, 9, 10, 11, 12),
    )
    assert plans[0].recv_compressed_entry_ids_by_peer == ((1,), (2,))
    assert plans[1].recv_raw_token_ids_by_peer == ((), tuple(range(8, 18)))
    assert plans[1].recv_compressed_entry_ids_by_peer == ((), (2, 3))
    assert plans[0].send_raw_token_ids_by_peer == ((4, 5, 6, 7), ())
    assert plans[0].send_compressed_entry_ids_by_peer == ((1,), ())
    assert plans[1].send_raw_token_ids_by_peer == (
        (8, 9, 10, 11, 12),
        tuple(range(8, 18)),
    )
    assert plans[1].send_compressed_entry_ids_by_peer == ((2,), (2, 3))


def test_stage_inputs_derive_from_art_stage_plans() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    rank0_plan = _stage_plan(
        stage_index=5,
        q_ranges=((7, 8),),
        k_ranges=((4, 13),),
    )
    rank1_plan = _stage_plan(
        stage_index=5,
        q_ranges=((11, 12), (16, 17)),
        k_ranges=((8, 18),),
    )

    rank0 = build_dsv4_stage_inputs_from_stage_plan(
        layout=layout,
        stage_plan=rank0_plan,
        compression_kind=Dsv4CompressionKind.CSA,
        global_topk=torch.tensor([[[0, 1], [1, 2], [2, 3]]], dtype=torch.long),
        topk_query_token_ids=(3, 7, 8),
        window_size=4,
    )
    rank1 = build_dsv4_stage_inputs_from_stage_plan(
        layout=layout,
        stage_plan=rank1_plan,
        compression_kind=Dsv4CompressionKind.CSA,
        global_topk=torch.tensor([[[2, 3], [3, 2], [0, 1]]], dtype=torch.long),
        topk_query_token_ids=(11, 16, 17),
        window_size=4,
    )

    assert rank0.stage_index == 5
    assert rank1.stage_index == 5
    assert rank0.query_token_ids == (7,)
    assert rank0.raw_token_ids == tuple(range(4, 13))
    assert rank0.compressed_entry_ids == (1, 2)
    assert rank0.topk_stage_local[0, 0].tolist() == [0, 1, 2, 3, 9, -1]
    assert rank1.query_token_ids == (11, 16)
    assert rank1.raw_token_ids == tuple(range(8, 18))
    assert rank1.compressed_entry_ids == (2, 3)
    assert rank1.topk_stage_local[0, 0].tolist() == [0, 1, 2, 3, 10, -1]
    assert rank1.topk_stage_local[0, 1].tolist() == [5, 6, 7, 8, 11, -1]


def test_stage_plan_slots_group_stage_indices_in_rank0_order() -> None:
    slots = build_dsv4_stage_plan_slots(
        stage_plans_by_rank=(
            (
                _stage_plan(stage_index=9, q_ranges=((0, 1),), k_ranges=((0, 4),)),
                _stage_plan(stage_index=3, q_ranges=((1, 2),), k_ranges=((0, 8),)),
            ),
            (
                _stage_plan(stage_index=3, q_ranges=((8, 9),), k_ranges=((8, 12),)),
                _stage_plan(stage_index=9, q_ranges=((9, 10),), k_ranges=((4, 12),)),
            ),
        ),
    )

    assert tuple(slot.stage_index for slot in slots) == (9, 3)
    assert tuple(slot.stage_plans_by_rank[1].stage_index for slot in slots) == (9, 3)


def test_stage_kv_exchange_plan_from_stage_plans_does_not_need_topk() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    stage_plans = (
        _stage_plan(stage_index=5, q_ranges=((7, 8),), k_ranges=((4, 13),)),
        _stage_plan(
            stage_index=5,
            q_ranges=((11, 12), (16, 17)),
            k_ranges=((8, 18),),
        ),
    )
    topk_dependent_stages = (
        build_dsv4_stage_inputs_from_stage_plan(
            layout=layout,
            stage_plan=stage_plans[0],
            compression_kind=Dsv4CompressionKind.CSA,
            global_topk=torch.tensor([[[1, 2]]], dtype=torch.long),
            topk_query_token_ids=(7,),
            window_size=4,
        ),
        build_dsv4_stage_inputs_from_stage_plan(
            layout=layout,
            stage_plan=stage_plans[1],
            compression_kind=Dsv4CompressionKind.CSA,
            global_topk=torch.tensor([[[3, 2], [0, 1]]], dtype=torch.long),
            topk_query_token_ids=(11, 16),
            window_size=4,
        ),
    )

    from_stage_inputs = build_dsv4_stage_kv_exchange_peer_plans(
        layout=layout,
        stage_inputs_by_rank=topk_dependent_stages,
    )
    from_stage_plans = build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans(
        layout=layout,
        stage_plans_by_rank=stage_plans,
    )

    assert from_stage_plans == from_stage_inputs


def test_stage_slot_exchange_uses_prepared_peer_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    slot = build_dsv4_stage_plan_slots(
        stage_plans_by_rank=(
            (_stage_plan(stage_index=5, q_ranges=((7, 8),), k_ranges=((4, 13),)),),
            (
                _stage_plan(
                    stage_index=5,
                    q_ranges=((11, 12),),
                    k_ranges=((8, 18),),
                ),
            ),
        )
    )[0]
    local_stage_inputs = build_stage_local_topk_for_csa(
        layout=layout,
        stage_index=5,
        query_token_ids=(7,),
        global_k_ranges=(_Range(start=4, end=13),),
        global_topk=torch.tensor([[[1, 2]]], dtype=torch.long),
        window_size=4,
    )
    prepared = build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans(
        layout=layout,
        stage_plans_by_rank=slot.stage_plans_by_rank,
    )
    captured: dict[str, object] = {}

    def fail_rebuild(**_kwargs: object) -> object:
        raise AssertionError("stage peer plan should be prepared")

    def fake_launch(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        cp_stage,
        "build_dsv4_stage_kv_exchange_peer_plans_from_stage_plans",
        fail_rebuild,
    )
    monkeypatch.setattr(cp_stage, "launch_dsv4_stage_kv_exchange", fake_launch)

    result = launch_dsv4_stage_kv_exchange_from_stage_plan_slot(
        layout=layout,
        rank=1,
        stage_plan_slot=slot,
        local_stage_inputs=local_stage_inputs,
        query=torch.empty(1, 0, 1, 4),
        query_token_ids=(),
        raw_kv=torch.empty(0, 4),
        raw_token_ids=(),
        compressed_kv=torch.empty(0, 4),
        compressed_entry_ids=(),
        group=None,
        async_op=False,
        peer_plans=prepared,
    )

    assert result is not None
    assert (
        captured["send_raw_token_ids_by_peer"] == prepared[1].send_raw_token_ids_by_peer
    )
    assert (
        captured["recv_raw_token_ids_by_peer"] == prepared[1].recv_raw_token_ids_by_peer
    )
    assert (
        captured["send_compressed_entry_ids_by_peer"]
        == prepared[1].send_compressed_entry_ids_by_peer
    )


def test_deferred_stage_slot_exchange_binds_stage_inputs_later() -> None:
    layout = _single_rank_layout(Dsv4CompressionKind.CSA)
    slot = build_dsv4_stage_plan_slots(
        stage_plans_by_rank=(
            (_stage_plan(stage_index=5, q_ranges=((7, 8),), k_ranges=((4, 13),)),),
        )
    )[0]
    stage_inputs = build_stage_local_topk_for_csa(
        layout=layout,
        stage_index=5,
        query_token_ids=(7,),
        global_k_ranges=(_Range(start=4, end=13),),
        global_topk=torch.tensor([[[1, 2]]], dtype=torch.long),
        window_size=4,
    )
    work = launch_dsv4_stage_kv_exchange_deferred_from_stage_plan_slot(
        layout=layout,
        rank=0,
        stage_plan_slot=slot,
        query=_query_rows((7,)),
        query_token_ids=(7,),
        raw_kv=_kv_rows(tuple(range(18))),
        raw_token_ids=tuple(range(18)),
        compressed_kv=_kv_rows((201, 202)),
        compressed_entry_ids=(1, 2),
        group=None,
        async_op=False,
    )

    with pytest.raises(RuntimeError, match="bound to stage inputs"):
        work.wait_post_process()

    materialized = work.bind_stage_inputs(stage_inputs).wait_post_process()

    assert materialized.stage_index == 5
    assert materialized.query_token_ids == (7,)
    assert materialized.raw_count == 9
    assert materialized.compressed_count == 2
    torch.testing.assert_close(
        materialized.topk_stage_local,
        stage_inputs.topk_stage_local,
    )
    with pytest.raises(RuntimeError, match="already has stage inputs"):
        work.bind_stage_inputs(stage_inputs)


def test_materialize_stage_tensors_uses_explicit_id_maps() -> None:
    layout = _layout(Dsv4CompressionKind.CSA)
    stage = build_stage_local_topk_for_csa(
        layout=layout,
        stage_index=6,
        query_token_ids=(8, 11, 16),
        global_k_ranges=(_Range(start=8, end=13),),
        global_topk=torch.tensor([[2, 0, -1], [2, 1, -1], [2, 3, -1]]),
        window_size=4,
    )

    query_ids = (16, 8, 11)
    query = _query_rows(query_ids)
    raw_ids = (12, 8, 11, 10, 9)
    raw_kv = _kv_rows(raw_ids)
    compressed_ids = (2,)
    compressed_kv = _kv_rows((202,))

    materialized = materialize_dsv4_stage_tensors(
        stage_inputs=stage,
        query=query,
        query_token_ids=query_ids,
        raw_kv=raw_kv,
        raw_token_ids=raw_ids,
        compressed_kv=compressed_kv,
        compressed_entry_ids=compressed_ids,
    )

    assert materialized.raw_count == 5
    assert materialized.compressed_count == 1
    assert materialized.key_global_ids == (8, 9, 10, 11, 12, 2)
    assert materialized.q_stage.shape == (1, 3, 2, 4)
    assert materialized.kv_stage.shape == (1, 6, 4)
    assert materialized.q_stage[0, :, 0, 0].tolist() == [8.0, 11.0, 16.0]
    assert materialized.kv_stage[0, :, 0].tolist() == [
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        202.0,
    ]
    torch.testing.assert_close(materialized.topk_stage_local, stage.topk_stage_local)


def test_materialize_stage_tensors_preserves_batch_topk_and_expands_singletons() -> (
    None
):
    layout = _layout(Dsv4CompressionKind.CSA)
    stage = build_stage_local_topk_for_csa(
        layout=layout,
        stage_index=7,
        query_token_ids=(11,),
        global_k_ranges=(_Range(start=8, end=13), _Range(start=13, end=18)),
        global_topk=torch.tensor([[[2, -1]], [[3, 2]]], dtype=torch.long),
        window_size=4,
    )
    query = torch.stack((_query_rows((11,)), _query_rows((111,))), dim=0)

    materialized = materialize_dsv4_stage_tensors(
        stage_inputs=stage,
        query=query,
        query_token_ids=(11,),
        raw_kv=_kv_rows(tuple(reversed(range(8, 18)))),
        raw_token_ids=tuple(reversed(range(8, 18))),
        compressed_kv=_kv_rows((202, 203)),
        compressed_entry_ids=(2, 3),
    )

    assert materialized.q_stage.shape == (2, 1, 2, 4)
    assert materialized.kv_stage.shape == (2, 12, 4)
    assert materialized.q_stage[:, 0, 0, 0].tolist() == [11.0, 111.0]
    assert materialized.kv_stage[0, :, 0].tolist() == [
        *[float(token_id) for token_id in range(8, 18)],
        202.0,
        203.0,
    ]
    torch.testing.assert_close(materialized.kv_stage[0], materialized.kv_stage[1])
    torch.testing.assert_close(materialized.topk_stage_local, stage.topk_stage_local)


def test_materialize_stage_tensors_rejects_missing_ids() -> None:
    layout = _layout(Dsv4CompressionKind.HCA)
    stage = build_stage_local_topk_for_hca(
        layout=layout,
        stage_index=8,
        query_token_ids=(11,),
        global_k_ranges=(_Range(start=8, end=13),),
        window_size=4,
    )

    with pytest.raises(RuntimeError, match="raw_kv tensor is missing ids"):
        materialize_dsv4_stage_tensors(
            stage_inputs=stage,
            query=_query_rows((11,)),
            query_token_ids=(11,),
            raw_kv=_kv_rows((8, 9, 10)),
            raw_token_ids=(8, 9, 10),
            compressed_kv=_kv_rows((202,)),
            compressed_entry_ids=(2,),
        )


def _layout(kind: Dsv4CompressionKind) -> Dsv4CompressedLayout:
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
        spec=Dsv4CompressionSpec(kind=kind, ratio=4),
    )


def _single_rank_layout(kind: Dsv4CompressionKind) -> Dsv4CompressedLayout:
    return build_dsv4_compressed_layout(
        group_ids=torch.tensor([[0] * 8 + [1] * 5 + [2] * 5 + [-1] * 2]),
        parent_ids=torch.tensor([[0] * 8 + [0] * 5 + [0] * 5 + [-1] * 2]),
        token_layout_index=_LayoutIndex(
            ownership_ranges_by_rank=(((0, 18, 0),),),
            token_counts_by_rank=(18,),
        ),
        spec=Dsv4CompressionSpec(kind=kind, ratio=4),
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
