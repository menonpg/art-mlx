from art.megatron.model_support.handlers import (
    DEFAULT_DENSE_HANDLER,
    QWEN3_5_MOE_HANDLER,
)


def test_default_dense_handler_returns_standard_attention_kwargs() -> None:
    assert DEFAULT_DENSE_HANDLER.get_forward_kwargs(
        object(),
        attention_bias="bias",
    ) == {"extra_block_kwargs": {"attention_bias": "bias"}}


def test_qwen_handler_wraps_qwen3vl_forward_kwargs() -> None:
    qwen_model = type("Qwen3VLModel", (), {})()

    assert QWEN3_5_MOE_HANDLER.get_forward_kwargs(
        qwen_model,
        attention_bias="bias",
    ) == {"extra_block_kwargs": {"extra_block_kwargs": {"attention_bias": "bias"}}}


def test_qwen_handler_unwraps_model_wrappers() -> None:
    qwen_model = type("Qwen3VLModel", (), {})()
    wrapper = type("Wrapper", (), {"module": qwen_model})()

    assert QWEN3_5_MOE_HANDLER.get_forward_kwargs(
        wrapper,
        attention_bias="bias",
    ) == {"extra_block_kwargs": {"extra_block_kwargs": {"attention_bias": "bias"}}}
