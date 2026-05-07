from __future__ import annotations

import torch

from art.preprocessing.pack import PackedTensors


@torch.no_grad()
def build_sft_trajectory_tensors_from_packed_tensors(
    packed_tensors: PackedTensors,
) -> list[dict[str, torch.Tensor]]:
    tokens = packed_tensors["tokens"]
    assistant_mask = packed_tensors["assistant_mask"]
    labels = torch.where(assistant_mask, tokens, torch.full_like(tokens, -100))
    attention_mask = torch.ones_like(tokens, dtype=torch.long)
    return [
        {
            "input_ids": tokens[index].detach().clone(),
            "attention_mask": attention_mask[index].detach().clone(),
            "labels": labels[index].detach().clone(),
        }
        for index in range(int(tokens.shape[0]))
    ]
