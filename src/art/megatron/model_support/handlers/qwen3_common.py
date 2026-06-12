from __future__ import annotations

from typing import Any, Sequence, cast


def _context_parallel_world_size(config: Any) -> int:
    from megatron.core import parallel_state as ps
    from torch.distributed import is_initialized

    if is_initialized() and ps.model_parallel_is_initialized():
        return int(ps.get_context_parallel_world_size())
    return int(getattr(config, "context_parallel_size", 1) or 1)


def install_qwen3_text_preprocess_patch(model_chunks: Sequence[Any]) -> None:
    from megatron.core.models.gpt.gpt_model import GPTModel
    import torch

    for chunk in list(model_chunks):
        module: Any = chunk
        while hasattr(module, "module"):
            module = module.module
        gpt_module = (
            module
            if isinstance(module, GPTModel)
            else cast(GPTModel, getattr(module, "language_model"))
        )
        preprocess = gpt_module._preprocess

        def preprocess_hook(*args, _preprocess=preprocess, **kwargs):
            position_ids = kwargs.get("position_ids")
            rotary_pos_emb = getattr(gpt_module, "rotary_pos_emb", None)
            rotary_cp_group = getattr(rotary_pos_emb, "cp_group", None)
            config = getattr(gpt_module, "config", None)
            cp_world_size = _context_parallel_world_size(config)
            uses_dispatched_local_cp_positions = (
                isinstance(position_ids, torch.Tensor)
                and position_ids.ndim == 2
                and cp_world_size > 1
                and rotary_cp_group is not None
            )
            if uses_dispatched_local_cp_positions:
                setattr(rotary_pos_emb, "cp_group", None)
            try:
                preproc_output = list(_preprocess(*args, **kwargs))
            finally:
                if uses_dispatched_local_cp_positions:
                    setattr(rotary_pos_emb, "cp_group", rotary_cp_group)
            decoder_input = cast(torch.Tensor, preproc_output[0])
            if not decoder_input.requires_grad and decoder_input.is_leaf:
                decoder_input.requires_grad_(True)
            position_ids = cast(torch.Tensor, position_ids)
            table = cast(torch.Tensor, preproc_output[1])
            if table is None:
                return tuple(preproc_output)
            embedding_dim = int(table.shape[-1])
            batch_size, sequence_length = position_ids.shape
            gathered = table.view(table.shape[0], embedding_dim).index_select(
                0,
                position_ids.reshape(-1),
            )
            preproc_output[1] = (
                gathered.view(batch_size, sequence_length, embedding_dim)
                .permute(1, 0, 2)
                .contiguous()
                .unsqueeze(2)
            )
            return tuple(preproc_output)

        gpt_module._preprocess = preprocess_hook  # type: ignore[attr-defined]
