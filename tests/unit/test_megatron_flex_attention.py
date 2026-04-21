from art.megatron import flex_attention


def test_flex_attention_resolves_eager_path_when_compile_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ART_DISABLE_MEGATRON_COMPILE", "1")
    monkeypatch.setattr(
        flex_attention.FlexAttentionWrapper,
        "_compiled_flex_attention",
        None,
    )
    monkeypatch.setattr(flex_attention, "_compiled_create_block_mask", None)

    assert (
        flex_attention.FlexAttentionWrapper._resolve_impl()
        is flex_attention.flex_attention
    )
    assert (
        flex_attention._resolve_create_block_mask()
        is flex_attention.create_block_mask
    )


def test_flex_attention_compiles_lazily_once_when_enabled(
    monkeypatch,
) -> None:
    compiled_calls: list[tuple[object, object]] = []

    def _fake_compile(fn, options=None):
        compiled_calls.append((fn, options))
        return lambda *args, **kwargs: (fn, args, kwargs)

    monkeypatch.delenv("ART_DISABLE_MEGATRON_COMPILE", raising=False)
    monkeypatch.setattr(flex_attention.torch, "compile", _fake_compile)
    monkeypatch.setattr(
        flex_attention.FlexAttentionWrapper,
        "_compiled_flex_attention",
        None,
    )
    monkeypatch.setattr(flex_attention, "_compiled_create_block_mask", None)

    compiled_attention = flex_attention.FlexAttentionWrapper._resolve_impl()
    compiled_attention_again = flex_attention.FlexAttentionWrapper._resolve_impl()
    compiled_mask = flex_attention._resolve_create_block_mask()
    compiled_mask_again = flex_attention._resolve_create_block_mask()

    assert compiled_attention is compiled_attention_again
    assert compiled_mask is compiled_mask_again
    assert len(compiled_calls) == 2
    assert compiled_calls[0][0] is flex_attention.flex_attention
    assert compiled_calls[1][0] is flex_attention.create_block_mask
