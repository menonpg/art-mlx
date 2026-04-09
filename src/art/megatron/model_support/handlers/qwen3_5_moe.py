from art.megatron.model_support.handlers.default_dense import DefaultDenseHandler


class Qwen35MoeHandler(DefaultDenseHandler):
    key = "qwen3_5_moe"


QWEN3_5_MOE_HANDLER = Qwen35MoeHandler()
