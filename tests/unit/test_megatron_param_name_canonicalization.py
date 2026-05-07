from art.megatron.weights.param_name_canonicalization import (
    canonical_art_param_name,
    is_art_adapter_param_name,
)


def test_canonical_art_param_name_strips_art_wrapper_segments() -> None:
    assert (
        canonical_art_param_name(
            "module.language_model.decoder.layers.0.self_attention.out_proj.linear_proj.weight"
        )
        == "language_model.decoder.layers.0.self_attention.out_proj.weight"
    )
    assert (
        canonical_art_param_name(
            "module.language_model.decoder.layers.0.mlp.linear_fc2.row_parallel_lora.linear_proj.weight"
        )
        == "language_model.decoder.layers.0.mlp.linear_fc2.weight"
    )
    assert (
        canonical_art_param_name(
            "module.language_model.decoder.layers.0.self_attention.linear_qkv.linear_qkv.weight"
        )
        == "language_model.decoder.layers.0.self_attention.linear_qkv.weight"
    )


def test_is_art_adapter_param_name_recognizes_wrapped_lora_params() -> None:
    assert is_art_adapter_param_name(
        "language_model.decoder.layers.0.self_attention.linear_qkv.q_proj_lora.A_T"
    )
    assert is_art_adapter_param_name(
        "language_model.decoder.layers.0.mlp.experts.linear_fc1.gate_lora.B_T"
    )
    assert not is_art_adapter_param_name(
        "language_model.decoder.layers.0.self_attention.linear_qkv.weight"
    )
