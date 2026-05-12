from typing import Any, Sequence

from art.megatron.model_support.handlers.default_dense import (
    DefaultMoeHandler,
    _compile_workaround_flags_for_provider,
)
from art.megatron.model_support.handlers.qwen3_common import (
    install_qwen3_text_preprocess_patch,
)
from art.megatron.model_support.spec import CompileWorkaroundConfig

_QWEN3_MOE_COMPILE_WORKAROUND_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
    "deepep_dispatch_combine",
    "deepep_permute_restore",
    "te_triton_permute_with_mask_map",
)


class Qwen3MoeHandler(DefaultMoeHandler):
    key = "qwen3_moe"
    native_vllm_lora_status = "validated"

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        install_qwen3_text_preprocess_patch(model_chunks)

    def compile_workaround_config(
        self,
        provider: Any,
    ) -> CompileWorkaroundConfig:
        return CompileWorkaroundConfig(
            flags=_compile_workaround_flags_for_provider(
                provider,
                _QWEN3_MOE_COMPILE_WORKAROUND_FLAGS,
            )
        )


QWEN3_MOE_HANDLER = Qwen3MoeHandler()
