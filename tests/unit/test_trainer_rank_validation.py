from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from art.megatron.trainer_rank import (
    ForwardInput,
    TrainerRank,
    _validate_top_k,
)


class _Model:
    vocab_size = 8


def test_forward_input_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        ForwardInput(input_tokens=torch.tensor([1]), top_k=0)


def test_validate_top_k_rejects_values_above_vocab_size() -> None:
    with pytest.raises(ValueError, match="top_k=9 exceeds vocabulary size 8"):
        _validate_top_k(9, _Model())  # type: ignore[arg-type]


def test_trainer_rank_accepts_nested_shared_prefix_for_gdn_runtime() -> None:
    runtime = SimpleNamespace(
        model=[torch.nn.Linear(1, 1)],
        optimizer=None,
        model_support_handler=SimpleNamespace(build_gdn_execution_spec=True),
    )

    trainer = TrainerRank(runtime, shared_prefix_max_depth=2)  # type: ignore[arg-type]

    assert trainer.shared_prefix_max_depth == 2


def test_trainer_rank_accepts_zero_depth_shared_prefix_for_gdn_runtime() -> None:
    runtime = SimpleNamespace(
        model=[torch.nn.Linear(1, 1)],
        optimizer=None,
        model_support_handler=SimpleNamespace(build_gdn_execution_spec=True),
    )

    trainer = TrainerRank(runtime, shared_prefix_max_depth=0)  # type: ignore[arg-type]

    assert trainer.shared_prefix_max_depth == 0
