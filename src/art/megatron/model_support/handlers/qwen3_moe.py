from typing import Any, Sequence, cast

from art.megatron.model_chunks import ModelChunks
from art.megatron.model_support.handlers.default_dense import DefaultDenseHandler


class Qwen3MoeHandler(DefaultDenseHandler):
    key = "qwen3_moe"

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        from art.megatron.train import _install_gpt_preprocess_hook

        _install_gpt_preprocess_hook(cast(ModelChunks, list(model_chunks)))


QWEN3_MOE_HANDLER = Qwen3MoeHandler()
