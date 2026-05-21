from typing import Any

THINKING_CHAT_TEMPLATE_KWARGS: dict[str, object] = {
    "enable_thinking": False,
    "preserve_thinking": True,
}


def default_chat_template_kwargs_for_template(
    chat_template: object,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if not isinstance(chat_template, str):
        return kwargs
    if "enable_thinking" in chat_template:
        kwargs["enable_thinking"] = False
    if "preserve_thinking" in chat_template:
        kwargs["preserve_thinking"] = True
    return kwargs


def default_chat_template_kwargs_for_tokenizer(tokenizer: object) -> dict[str, object]:
    return default_chat_template_kwargs_for_template(
        getattr(tokenizer, "chat_template", None)
    )


def merge_chat_template_kwargs(
    defaults: dict[str, object] | None,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    return {**(defaults or {}), **(overrides or {})}
