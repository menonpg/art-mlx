import sys
import types
from typing import Any, cast

from openai.types.chat.chat_completion import Choice
import pytest

import art
from art.preprocessing.tokenize import (
    tokenize_sft_batch,
    tokenize_trajectory,
)
from art.trajectories import History, Trajectory
from art.types import MessagesAndChoices

pytest.importorskip("torch")
pytest.importorskip("transformers")


class _FakeTokenizer:
    chat_template = ""
    vocab_size = 256
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
        rendered = "".join(
            f"<{message['role']}>{message.get('content', '')}" for message in messages
        )
        if not tokenize:
            return rendered
        token_ids = self.encode(rendered, add_special_tokens=False)
        assert return_dict is False
        return token_ids

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

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


class _Qwen3_5FakeTokenizer(_FakeTokenizer):
    chat_template = (
        "{% for args_name, args_value in tool_call.arguments|items %}{% endfor %}"
    )

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=None,
        **kwargs,
    ):
        for message in messages:
            tool_calls = message.get("tool_calls")
            if tool_calls is None:
                continue
            assert isinstance(tool_calls, list)
            for tool_call in tool_calls:
                assert isinstance(tool_call, dict)
                function = tool_call["function"]
                assert isinstance(function, dict)
                assert isinstance(function["arguments"], dict)
        return super().apply_chat_template(
            messages,
            tools=tools,
            tokenize=tokenize,
            return_dict=return_dict,
            **kwargs,
        )


class _ContinueFinalMessageRejectingTokenizer(_FakeTokenizer):
    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=None,
        **kwargs,
    ):
        if kwargs.get("continue_final_message") is True and messages[-1].get(
            "content", ""
        ).startswith("<think>"):
            raise ValueError(
                "continue_final_message is set but the final message does not appear "
                "in the chat after applying the chat template!"
            )
        return super().apply_chat_template(
            messages,
            tools=tools,
            tokenize=tokenize,
            return_dict=return_dict,
            **kwargs,
        )


def test_tokenize_trajectory_requests_list_chat_template_output() -> None:
    tokenizer = _FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "OK"},
        ],
    )
    history = History(messages_and_choices=messages)
    trajectory = Trajectory(messages_and_choices=messages, reward=1.0)

    result = tokenize_trajectory(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        image_processor=None,
        history=history,
        advantage=1.0,
        allow_training_without_logprobs=True,
        trajectory=trajectory,
    )

    assert result is not None
    assistant_ids = [
        token_id
        for token_id, mask in zip(result.token_ids, result.assistant_mask)
        if mask
    ]
    assert assistant_ids == tokenizer.encode("OK", add_special_tokens=False)


def test_tokenize_trajectory_passes_chat_template_kwargs() -> None:
    tokenizer = _FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "OK"},
        ],
    )
    history = History(messages_and_choices=messages)
    trajectory = Trajectory(messages_and_choices=messages, reward=1.0)

    result = tokenize_trajectory(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        image_processor=None,
        history=history,
        advantage=1.0,
        allow_training_without_logprobs=True,
        trajectory=trajectory,
        chat_template_kwargs={
            "enable_thinking": False,
            "preserve_thinking": True,
        },
    )

    assert result is not None
    assert tokenizer.apply_chat_template_kwargs
    assert all(
        call.get("enable_thinking") is False and call.get("preserve_thinking") is True
        for call in tokenizer.apply_chat_template_kwargs
    )


def test_tokenize_trajectory_does_not_continue_real_completion_with_thinking() -> None:
    tokenizer = _ContinueFinalMessageRejectingTokenizer()
    choice = Choice.model_validate(
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": {
                "content": [
                    {
                        "token": "token_id:79",
                        "bytes": [79],
                        "logprob": -0.1,
                        "top_logprobs": [],
                    },
                    {
                        "token": "token_id:75",
                        "bytes": [75],
                        "logprob": -0.2,
                        "top_logprobs": [],
                    },
                ],
                "refusal": None,
            },
            "message": {
                "content": "<think>\n reasoning \n</think>\n\nOK",
                "refusal": None,
                "role": "assistant",
                "annotations": None,
                "audio": None,
                "function_call": None,
                "tool_calls": None,
            },
        }
    )
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Hi"},
            choice,
        ],
    )
    history = History(messages_and_choices=messages)
    trajectory = Trajectory(messages_and_choices=messages, reward=1.0)

    result = tokenize_trajectory(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        image_processor=None,
        history=history,
        advantage=1.0,
        allow_training_without_logprobs=False,
        trajectory=trajectory,
        chat_template_kwargs={
            "enable_thinking": False,
            "preserve_thinking": True,
        },
    )

    assert result is not None
    assistant_ids = [
        token_id
        for token_id, mask in zip(result.token_ids, result.assistant_mask)
        if mask
    ]
    assert assistant_ids == [79, 75]
    continue_values = [
        call.get("continue_final_message")
        for call in tokenizer.apply_chat_template_kwargs
    ]
    assert continue_values[:2] == [False, False]
    assert continue_values[-1] is True


def test_tokenize_sft_batch_requests_list_chat_template_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizer()

    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth_zoo = types.ModuleType("unsloth_zoo")
    fake_dataset_utils = types.ModuleType("unsloth_zoo.dataset_utils")

    def _train_on_responses_only(**kwargs):
        del kwargs

        def _labels_fn(batch):
            return {"labels": [list(batch["input_ids"][0])]}

        return _labels_fn

    fake_dataset_utils.train_on_responses_only = _train_on_responses_only  # type: ignore[attr-defined]
    fake_unsloth_zoo.dataset_utils = fake_dataset_utils  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    monkeypatch.setitem(sys.modules, "unsloth_zoo", fake_unsloth_zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.dataset_utils", fake_dataset_utils)

    trajectory = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[trajectory],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
    )

    expected_ids = tokenizer.encode(
        tokenizer.apply_chat_template(
            trajectory.messages_and_choices,
            tokenize=False,
            add_generation_prompt=False,
        ),
        add_special_tokens=False,
    )

    assert batch.trajectory_tensors[0]["input_ids"].tolist() == [expected_ids]
    assert batch.trajectory_tensors[0]["attention_mask"].tolist() == [
        [1] * len(expected_ids)
    ]
    assert batch.num_dropped_trajectories == 0
    assert batch.num_tokens == len(expected_ids)
    assert batch.num_trainable_tokens == len(expected_ids)

    dropped_batch = tokenize_sft_batch(
        trajectory_batch=[trajectory],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
        max_seq_length=len(expected_ids) - 1,
    )
    assert dropped_batch.trajectory_tensors == []
    assert dropped_batch.num_trajectories == 0
    assert dropped_batch.num_tokens == 0
    assert dropped_batch.num_trainable_tokens == 0
    assert dropped_batch.num_dropped_trajectories == 1


def test_tokenize_trajectory_normalizes_mapping_tool_arguments_for_chat_template() -> (
    None
):
    tokenizer = _Qwen3_5FakeTokenizer()
    choice = Choice.model_validate(
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": {
                "content": [
                    {
                        "token": "token_id:65",
                        "bytes": [65],
                        "logprob": -0.1,
                        "top_logprobs": [],
                    }
                ],
                "refusal": None,
            },
            "message": {
                "content": "",
                "refusal": None,
                "role": "assistant",
                "annotations": None,
                "audio": None,
                "function_call": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "arguments": '{"city": "San Francisco", "days": 3}',
                            "name": "lookup_weather",
                        },
                        "type": "function",
                    }
                ],
            },
        }
    )
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Weather?"},
            choice,
        ],
    )
    history = History(messages_and_choices=messages)
    trajectory = Trajectory(messages_and_choices=messages, reward=1.0)

    result = tokenize_trajectory(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        image_processor=None,
        history=history,
        advantage=1.0,
        allow_training_without_logprobs=False,
        trajectory=trajectory,
    )

    assert result is not None
