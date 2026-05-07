import importlib
from pathlib import Path
import sys

import pytest
import torch

TESTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS_ROOT))

megatron_oracle_harness = importlib.import_module("integration.megatron.model_support.oracle_harness")
PackedTensorConfig = megatron_oracle_harness.PackedTensorConfig
_build_packed_tensors = megatron_oracle_harness._build_packed_tensors


def _row_runs(
    group_row: torch.Tensor,
    parent_row: torch.Tensor,
) -> list[tuple[int, int, int, int]]:
    valid_tokens = int((group_row != -1).sum().item())
    runs: list[tuple[int, int, int, int]] = []
    cursor = 0
    while cursor < valid_tokens:
        group_id = int(group_row[cursor].item())
        parent_id = int(parent_row[cursor].item())
        end = cursor + 1
        while end < valid_tokens and int(group_row[end].item()) == group_id:
            assert int(parent_row[end].item()) == parent_id
            end += 1
        runs.append((cursor, end, group_id, parent_id))
        cursor = end
    return runs


@pytest.mark.parametrize(
    ("seed", "config"),
    [
        (
            7,
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
    ],
)
def test_oracle_harness_stop_early_keeps_whole_prompt_families(
    seed: int,
    config: PackedTensorConfig,
) -> None:
    packed_tensors = _build_packed_tensors(config, seed)

    for row_index in range(config.num_sequences):
        runs = _row_runs(
            packed_tensors["group_ids"][row_index],
            packed_tensors["parent_ids"][row_index],
        )
        cursor = 0
        prompt_count = 0
        while cursor < len(runs):
            start, end, prompt_group_id, prompt_parent_id = runs[cursor]
            assert prompt_group_id == prompt_parent_id
            assert end - start == config.prefill_tokens
            assert not bool(
                packed_tensors["assistant_mask"][row_index, start:end].any().item()
            )
            assert torch.isnan(packed_tensors["logprobs"][row_index, start:end]).all()
            assert packed_tensors["input_pos"][row_index, start:end].tolist() == list(
                range(config.prefill_tokens)
            )
            cursor += 1
            completion_count = 0
            while cursor < len(runs) and runs[cursor][3] == prompt_group_id:
                completion_start, completion_end, _group_id, _parent_id = runs[cursor]
                completion_length = completion_end - completion_start
                assert bool(
                    packed_tensors["assistant_mask"][
                        row_index, completion_start:completion_end
                    ]
                    .all()
                    .item()
                )
                assert not torch.isnan(
                    packed_tensors["logprobs"][
                        row_index, completion_start:completion_end
                    ]
                ).any()
                assert packed_tensors["input_pos"][
                    row_index, completion_start:completion_end
                ].tolist() == list(
                    range(
                        config.prefill_tokens,
                        config.prefill_tokens + completion_length,
                    )
                )
                completion_count += 1
                cursor += 1
            assert 1 <= completion_count <= config.completion_branches_per_prefix
            prompt_count += 1
        assert prompt_count >= 2


def test_oracle_harness_truncate_mode_fills_the_row_for_ablation() -> None:
    stop_early_config = PackedTensorConfig(
        num_sequences=4,
        sequence_length=61,
        prefill_tokens=17,
        completion_branches_per_prefix=2,
        decode_tokens=15,
        decode_tokens_jitter=0,
        packing_mode="stop_early",
    )
    truncate_config = stop_early_config.model_copy(update={"packing_mode": "truncate"})

    stop_early = _build_packed_tensors(stop_early_config, seed=41)
    truncated = _build_packed_tensors(truncate_config, seed=41)

    assert any(
        int((stop_early["group_ids"][row_index] == -1).sum().item()) > 0
        for row_index in range(stop_early_config.num_sequences)
    )
    assert bool((truncated["group_ids"] != -1).all().item())
