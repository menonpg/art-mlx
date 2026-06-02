from __future__ import annotations

from pydantic import BaseModel, ConfigDict
import torch

from art.megatron.dsv4 import (
    Dsv4CompressedLayout,
    Dsv4CompressionKind,
    Dsv4CompressionSpec,
    Dsv4StageKeyKind,
    build_dsv4_compressed_layout,
    build_stage_local_topk_for_csa,
    build_stage_local_topk_for_hca,
    raw_swa_token_ids_for_query,
)


class _LayoutIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    ownership_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    token_counts_by_rank: tuple[int, ...]


class _Range(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int
    end: int


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
