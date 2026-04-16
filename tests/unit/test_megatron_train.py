import os

import torch

from art.megatron.train import (
    _compile_enabled_for_handler,
    _maybe_rewrite_packed_rotary_pos_emb,
)


def test_rewrite_packed_rotary_pos_emb_gathers_rank2_positions() -> None:
    rotary_pos_emb = torch.arange(6 * 4, dtype=torch.float32).view(6, 1, 1, 4)
    position_ids = torch.tensor([[5, 1, 3], [0, 2, 4]])

    rewritten = _maybe_rewrite_packed_rotary_pos_emb(
        rotary_pos_emb,
        position_ids=position_ids,
        position_embedding_type="rope",
    )

    assert rewritten is not None
    assert rewritten.shape == (3, 2, 1, 4)
    assert torch.equal(rewritten[:, 0, 0, :], rotary_pos_emb[position_ids[0], 0, 0, :])
    assert torch.equal(rewritten[:, 1, 0, :], rotary_pos_emb[position_ids[1], 0, 0, :])


def test_rewrite_packed_rotary_pos_emb_skips_mrope_positions() -> None:
    rotary_pos_emb = torch.arange(5 * 2 * 1 * 4, dtype=torch.float32).view(5, 2, 1, 4)
    position_ids = torch.arange(3 * 2 * 5, dtype=torch.long).view(3, 2, 5)

    rewritten = _maybe_rewrite_packed_rotary_pos_emb(
        rotary_pos_emb,
        position_ids=position_ids,
        position_embedding_type="mrope",
    )

    assert rewritten is rotary_pos_emb


def test_compile_enabled_for_handler_disables_qwen35(monkeypatch) -> None:
    monkeypatch.delenv("ART_DISABLE_MEGATRON_COMPILE", raising=False)

    assert _compile_enabled_for_handler("default_dense") is True
    assert _compile_enabled_for_handler("qwen3_5_moe") is False


def test_compile_enabled_for_handler_respects_env_disable(monkeypatch) -> None:
    monkeypatch.setenv("ART_DISABLE_MEGATRON_COMPILE", "1")

    assert _compile_enabled_for_handler("default_dense") is False
