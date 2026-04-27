from art.megatron.model_support.handlers.qwen3_moe import QWEN3_MOE_HANDLER


def test_qwen3_moe_compile_workarounds_cover_deepep_permute_restore() -> None:
    config = QWEN3_MOE_HANDLER.compile_workaround_config(object())
    assert config.flags == (
        "alltoall_dtoh",
        "alltoall_dispatch_preprocess",
        "deepep_permute_restore",
    )
