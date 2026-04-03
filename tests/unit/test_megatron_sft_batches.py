from pathlib import Path

import torch

from art.megatron.sft_batches import load_sft_batch_from_disk, materialize_sft_batches
from art.preprocessing.tokenize import SFTBatch


def test_materialize_and_load_sft_batches_round_trip(tmp_path: Path) -> None:
    batches = [
        SFTBatch(
            trajectory_tensors=[
                {
                    "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.int64),
                    "labels": torch.tensor([[-100, 2, 3]], dtype=torch.int64),
                },
                {
                    "input_ids": torch.tensor([[4, 5]], dtype=torch.int64),
                    "attention_mask": torch.tensor([[1, 1]], dtype=torch.int64),
                    "labels": torch.tensor([[-100, 5]], dtype=torch.int64),
                },
            ],
            learning_rate=1e-4,
            num_trajectories=2,
            num_trainable_tokens=3,
        )
    ]

    serialized = materialize_sft_batches(
        batches,
        sft_data_dir=str(tmp_path / "megatron-sft"),
    )

    assert serialized.num_batches == 1
    assert serialized.learning_rates == [1e-4]

    metadata, trajectories = load_sft_batch_from_disk(
        str(Path(serialized.sft_data_dir) / "batch_000000")
    )

    assert metadata == {
        "learning_rate": 1e-4,
        "num_trajectories": 2,
        "num_trainable_tokens": 3,
        "num_trajectory_tensors": 2,
    }
    assert len(trajectories) == 2
    assert torch.equal(trajectories[0]["input_ids"], torch.tensor([1, 2, 3]))
    assert torch.equal(trajectories[0]["labels"], torch.tensor([-100, 2, 3]))
    assert torch.equal(trajectories[1]["attention_mask"], torch.tensor([1, 1]))
