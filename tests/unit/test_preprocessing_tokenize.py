from typing import Any, cast

import pytest
from transformers.tokenization_utils_base import BatchEncoding

from art.preprocessing.tokenize import tokenize_sft_batch
from art.trajectories import Trajectory
from art.types import MessagesAndChoices

pytest.importorskip("torch")
pytest.importorskip("transformers")


class _FakeTokenizer:
    chat_template = ""
    eos_token = "\x00"
    eos_token_id = 0

    def __init__(self) -> None:
        self.apply_chat_template_kwargs: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=None,
        **kwargs,
    ):
        del tools
        self.apply_chat_template_kwargs.append(dict(kwargs))
        rendered_parts = []
        for message in messages:
            tool_calls = "".join(
                f"<tool>{tool_call['function']['name']}:{tool_call['function']['arguments']}"
                for tool_call in message.get("tool_calls", [])
            )
            rendered_parts.append(
                f"<{message['role']}>{tool_calls}{message.get('content', '')}"
            )
        rendered = "".join(rendered_parts)
        if not tokenize:
            return rendered
        token_ids = self.encode(rendered, add_special_tokens=False)
        if return_dict is False:
            return token_ids
        return BatchEncoding(
            {
                "input_ids": token_ids,
                "attention_mask": [1] * len(token_ids),
            }
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def __call__(self, text: str, add_special_tokens: bool = False):
        return type(
            "TokenizedText",
            (),
            {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)},
        )()

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            return chr(token_ids)
        return "".join(chr(token_id) for token_id in token_ids)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self.convert_tokens_to_ids(token) for token in tokens]
        if isinstance(tokens, str) and len(tokens) == 1:
            return ord(tokens)
        return self.eos_token_id


def test_tokenize_sft_batch_masks_response_tokens_without_unsloth_import() -> None:
    tokenizer = _FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "OK"},
        ],
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[Trajectory(messages_and_choices=messages, reward=1.0)],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
    )

    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert tokenizer.decode(trainable_token_ids) == "OK"
    assert batch.num_trainable_tokens == 2


def test_tokenize_sft_batch_passes_chat_template_kwargs() -> None:
    tokenizer = _FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "OK"},
        ],
    )

    tokenize_sft_batch(
        trajectory_batch=[Trajectory(messages_and_choices=messages, reward=1.0)],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
        chat_template_kwargs={
            "enable_thinking": False,
            "preserve_thinking": True,
        },
    )

    assert tokenizer.apply_chat_template_kwargs[-1]["enable_thinking"] is False
    assert tokenizer.apply_chat_template_kwargs[-1]["preserve_thinking"] is True


class _GenPromptTokenizer(_FakeTokenizer):
    """Fake tokenizer whose template supports add_generation_prompt."""

    generation_prompt = "<assistant>"

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=None,
        **kwargs,
    ):
        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        rendered = super().apply_chat_template(
            messages, tools=tools, tokenize=False, **kwargs
        )
        if add_generation_prompt:
            rendered += self.generation_prompt
        if not tokenize:
            return rendered
        return self.encode(rendered, add_special_tokens=False)


class _MergingTokenizer(_GenPromptTokenizer):
    """Fake BPE-style tokenizer that merges adjacent '!!' into one token."""

    generation_prompt = "<assistant>!"
    merged_token_id = 9999

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=None,
        **kwargs,
    ):
        messages = [
            {**message, "content": f"!{message['content']}"}
            if message["role"] == "assistant"
            else message
            for message in messages
        ]
        return super().apply_chat_template(
            messages,
            tools=tools,
            tokenize=tokenize,
            return_dict=return_dict,
            **kwargs,
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        token_ids = []
        index = 0
        while index < len(text):
            if text[index : index + 2] == "!!":
                token_ids.append(self.merged_token_id)
                index += 2
            else:
                token_ids.append(ord(text[index]))
                index += 1
        return token_ids


def test_tokenize_sft_batch_last_assistant_masks_only_final_response() -> None:
    tokenizer = _GenPromptTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "B"},
            {"role": "assistant", "content": "second"},
        ],
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[Trajectory(messages_and_choices=messages)],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="",
        response_part="",
        train_on="last_assistant",
    )

    input_ids = batch.trajectory_tensors[0]["input_ids"][0].tolist()
    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert (
        tokenizer.decode(input_ids) == "<user>A<assistant>first<user>B<assistant>second"
    )
    assert tokenizer.decode(trainable_token_ids) == "second"
    assert batch.num_trainable_tokens == len("second")


def test_tokenize_sft_batch_last_assistant_avoids_boundary_token_merge() -> None:
    tokenizer = _MergingTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "!x"},
        ],
    )
    # One-shot tokenization merges the prompt's trailing "!" with the
    # response's leading "!" into a single token that can never occur at
    # inference time, where the prompt is already fixed in context.
    one_shot = tokenizer.apply_chat_template(messages, tokenize=True)
    assert tokenizer.merged_token_id in one_shot

    batch = tokenize_sft_batch(
        trajectory_batch=[Trajectory(messages_and_choices=messages)],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="",
        response_part="",
        train_on="last_assistant",
    )

    input_ids = batch.trajectory_tensors[0]["input_ids"][0].tolist()
    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert tokenizer.merged_token_id not in input_ids
    assert tokenizer.decode(trainable_token_ids) == "!x"


def test_tokenize_sft_batch_last_assistant_requires_final_assistant() -> None:
    tokenizer = _GenPromptTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "B"},
        ],
    )

    with pytest.raises(ValueError, match="final message"):
        tokenize_sft_batch(
            trajectory_batch=[Trajectory(messages_and_choices=messages)],
            learning_rate=1e-5,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            instruction_part="",
            response_part="",
            train_on="last_assistant",
        )


def test_tokenize_sft_batch_last_assistant_rejects_non_prefix_template() -> None:
    class _NonPrefixTokenizer(_GenPromptTokenizer):
        def apply_chat_template(
            self,
            messages,
            tools=None,
            tokenize=True,
            return_dict=None,
            **kwargs,
        ):
            rendered = super().apply_chat_template(
                messages, tools=tools, tokenize=False, **kwargs
            )
            rendered = f"<count={len(messages)}>{rendered}"
            if tokenize:
                return self.encode(rendered, add_special_tokens=False)
            return rendered

    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "first"},
        ],
    )

    with pytest.raises(ValueError, match="prefix"):
        tokenize_sft_batch(
            trajectory_batch=[Trajectory(messages_and_choices=messages)],
            learning_rate=1e-5,
            tokenizer=_NonPrefixTokenizer(),  # type: ignore[arg-type]
            instruction_part="",
            response_part="",
            train_on="last_assistant",
        )
