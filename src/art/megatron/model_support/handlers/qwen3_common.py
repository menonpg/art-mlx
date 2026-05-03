from typing import Any, Sequence, cast

from megatron.core.models.gpt.gpt_model import GPTModel
import torch

from art.megatron.model_chunks import ModelChunks


def install_qwen3_text_preprocess_patch(model_chunks: Sequence[Any]) -> None:
    for chunk in cast(ModelChunks, list(model_chunks)):
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
            preproc_output = list(_preprocess(*args, **kwargs))
            decoder_input = cast(torch.Tensor, preproc_output[0])
            if not decoder_input.requires_grad and decoder_input.is_leaf:
                decoder_input.requires_grad_(True)
            position_ids = cast(torch.Tensor, kwargs["position_ids"])
            table = cast(torch.Tensor, preproc_output[1])
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
