from typing import Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def token_ids_for_template_part(
    tokenizer: PreTrainedTokenizerBase,
    template_part: str,
) -> list[int]:
    return list(tokenizer(template_part, add_special_tokens=False).input_ids)


def _find_subsequence(
    values: list[int],
    pattern: list[int],
    *,
    start: int = 0,
) -> int | None:
    if not pattern:
        return None
    last_start = len(values) - len(pattern)
    for index in range(start, last_start + 1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return None


def response_only_labels(
    input_ids: list[int],
    *,
    instruction_ids: list[int],
    response_ids: list[int],
) -> list[int]:
    labels = [-100] * len(input_ids)
    index = 0
    while index < len(input_ids):
        response_start = _find_subsequence(input_ids, response_ids, start=index)
        if response_start is None:
            break

        trainable_start = response_start + len(response_ids)
        next_instruction_start = _find_subsequence(
            input_ids,
            instruction_ids,
            start=trainable_start,
        )
        trainable_end = (
            len(input_ids) if next_instruction_start is None else next_instruction_start
        )
        labels[trainable_start:trainable_end] = input_ids[trainable_start:trainable_end]
        index = trainable_end
    return labels


def last_assistant_sample(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    *,
    tools: Any = None,
    template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Token ids and labels supervising only the final assistant message.

    The prompt is rendered exactly as inference renders it (generation prompt
    included) and encoded as one unit; the final assistant message is encoded
    separately so BPE cannot merge tokens across the prompt/completion boundary.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(
            "train_on='last_assistant' requires the final message to be an "
            "assistant message"
        )
    template_kwargs = template_kwargs or {}
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    assert isinstance(prompt_text, str) and isinstance(full_text, str)
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "Cannot compile a last-assistant loss mask: the chat template does "
            "not render the conversation prompt as a prefix of the full "
            "conversation."
        )
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False).input_ids)
    emitted_ids = list(
        tokenizer(full_text[len(prompt_text) :], add_special_tokens=False).input_ids
    )
    return prompt_ids + emitted_ids, [-100] * len(prompt_ids) + emitted_ids
