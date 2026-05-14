import pytest

from art.local.backend import (
    _apply_configured_chat_template,
    _apply_configured_chat_template_server_args,
    _tokenizer_cache_key,
)


class _Tokenizer:
    chat_template = "base"


def test_apply_configured_chat_template_reads_path(tmp_path) -> None:
    path = tmp_path / "chat_template.jinja"
    path.write_text("custom template", encoding="utf-8")
    tokenizer = _Tokenizer()

    _apply_configured_chat_template(
        tokenizer,  # type: ignore[arg-type]
        {"chat_template_path": str(path)},
    )

    assert tokenizer.chat_template == "custom template"


def test_apply_configured_chat_template_server_args_preserves_explicit_override() -> (
    None
):
    config_dict = {"server_args": {"chat_template": "explicit"}}

    _apply_configured_chat_template_server_args(
        config_dict,
        {
            "chat_template": "default",
            "chat_template_content_format": "string",
        },
    )

    assert config_dict["server_args"] == {
        "chat_template": "explicit",
        "chat_template_content_format": "string",
    }


def test_configured_chat_template_rejects_ambiguous_config(tmp_path) -> None:
    path = tmp_path / "chat_template.jinja"
    path.write_text("custom template", encoding="utf-8")

    with pytest.raises(ValueError, match="Set only one"):
        _apply_configured_chat_template(
            _Tokenizer(),  # type: ignore[arg-type]
            {
                "chat_template": "raw",
                "chat_template_path": str(path),
            },
        )


def test_tokenizer_cache_key_includes_chat_template_content(tmp_path) -> None:
    path = tmp_path / "chat_template.jinja"
    path.write_text("first template", encoding="utf-8")
    first_key = _tokenizer_cache_key("base-model", {"chat_template_path": str(path)})

    path.write_text("second template", encoding="utf-8")
    second_key = _tokenizer_cache_key("base-model", {"chat_template_path": str(path)})

    assert _tokenizer_cache_key("base-model", {}) == ("base-model", None)
    assert first_key != second_key
