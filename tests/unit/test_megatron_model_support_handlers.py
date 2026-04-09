from art.megatron.model_support.handlers import (
    DEFAULT_DENSE_HANDLER,
    QWEN3_5_MOE_HANDLER,
)
from art.megatron.model_support.spec import LayerFamilyInstance


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


def test_default_dense_handler_collects_dense_layer_families() -> None:
    provider = type("Provider", (), {"num_moe_experts": 0})()

    assert DEFAULT_DENSE_HANDLER.collect_layer_families(provider) == [
        LayerFamilyInstance(key="standard_attention", layer_index=0),
        LayerFamilyInstance(key="dense_mlp", layer_index=0),
    ]


def test_default_dense_handler_collects_moe_layer_families() -> None:
    provider = type(
        "Provider",
        (),
        {
            "num_moe_experts": 8,
            "moe_shared_expert_intermediate_size": 4096,
        },
    )()

    assert DEFAULT_DENSE_HANDLER.collect_layer_families(provider) == [
        LayerFamilyInstance(key="standard_attention", layer_index=0),
        LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
        LayerFamilyInstance(key="shared_experts_mlp", layer_index=0),
    ]


def test_qwen_handler_collects_expected_layer_families() -> None:
    provider = type("Provider", (), {"linear_attention_freq": 4, "num_layers": 8})()

    assert QWEN3_5_MOE_HANDLER.collect_layer_families(provider) == [
        LayerFamilyInstance(key="standard_attention", layer_index=3),
        LayerFamilyInstance(key="gated_delta_net_attention", layer_index=0),
        LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
        LayerFamilyInstance(key="shared_experts_mlp", layer_index=0),
    ]
