import math
from typing import Any, cast

from openai.types.chat.chat_completion import Choice
import pytest
from transformers.tokenization_utils_base import BatchEncoding

from art.preprocessing.tokenize import (
    tokenize_sft_batch,
    tokenize_trajectory,
    tokenize_vllm_trajectory_histories,
)
from art.preprocessing.vllm_tokens import attach_vllm_token_metadata_to_choice
from art.trajectories import History, Trajectory
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


def _choice(
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    *,
    with_logprobs: bool = True,
    message: dict[str, Any] | None = None,
) -> Choice:
    raw_choice: dict[str, Any] = {
        "finish_reason": "stop",
        "index": 0,
        "message": message or {"role": "assistant", "content": "x"},
        "token_ids": completion_token_ids,
    }
    if with_logprobs:
        raw_choice["logprobs"] = {
            "content": [
                {
                    "token": f"token_id:{token_id}",
                    "bytes": [token_id % 256],
                    "logprob": -0.1 * (i + 1),
                    "top_logprobs": [],
                }
                for i, token_id in enumerate(completion_token_ids)
            ],
            "refusal": None,
        }
    response_payload = {
        "prompt_token_ids": prompt_token_ids,
        "choices": [raw_choice],
    }
    choice = Choice.model_validate(raw_choice)
    attach_vllm_token_metadata_to_choice(
        choice=choice,
        response_payload=response_payload,
        choice_index=0,
    )
    return choice


def test_tokenize_trajectory_uses_vllm_prompt_and_completion_tokens() -> None:
    tokenizer = _FakeTokenizer()
    choice = _choice([10, 11], [20, 21])
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Hi"},
            choice,
        ],
    )
    trajectory = Trajectory(messages_and_choices=messages, reward=1.0)

    result = tokenize_trajectory(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        image_processor=None,
        history=History(messages_and_choices=messages),
        advantage=1.0,
        allow_training_without_logprobs=False,
        trajectory=trajectory,
    )

    assert result is not None
    assert result.token_ids == [10, 11, 20, 21]
    assert result.assistant_mask == [0, 0, 1, 1]
    assert result.choice_offsets == [2]
    assert all(math.isnan(logprob) for logprob in result.logprobs[:2])
    assert result.logprobs[2:] == [-0.1, -0.2]
    assert tokenizer.apply_chat_template_kwargs == []


def test_tokenize_trajectory_requires_vllm_token_metadata() -> None:
    raw_choice = {
        "finish_reason": "stop",
        "index": 0,
        "logprobs": {
            "content": [
                {
                    "token": "token_id:20",
                    "bytes": [20],
                    "logprob": -0.1,
                    "top_logprobs": [],
                }
            ],
            "refusal": None,
        },
        "message": {"role": "assistant", "content": "x"},
    }
    choice = Choice.model_validate(raw_choice)
    messages = cast(MessagesAndChoices, [{"role": "user", "content": "Hi"}, choice])

    with pytest.raises(RuntimeError, match="missing ART vLLM token metadata"):
        tokenize_trajectory(
            tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
            image_processor=None,
            history=History(messages_and_choices=messages),
            advantage=1.0,
            allow_training_without_logprobs=False,
            trajectory=Trajectory(messages_and_choices=messages, reward=1.0),
        )


def test_tokenize_vllm_trajectory_histories_collapses_append_only_turns() -> None:
    tokenizer = _FakeTokenizer()
    first_choice = _choice([1, 2], [3])
    second_choice = _choice([1, 2, 3, 4], [5])
    trajectory = Trajectory(
        messages_and_choices=cast(
            MessagesAndChoices,
            [{"role": "user", "content": "first"}, first_choice],
        ),
        additional_histories=[
            History(
                messages_and_choices=cast(
                    MessagesAndChoices,
                    [{"role": "user", "content": "second"}, second_choice],
                )
            )
        ],
        reward=1.0,
    )

    results = tokenize_vllm_trajectory_histories(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        histories=[
            History(messages_and_choices=trajectory.messages_and_choices),
            *trajectory.additional_histories,
        ],
        advantage=1.0,
        allow_training_without_logprobs=False,
        trajectory=trajectory,
    )

    assert len(results) == 1
    assert results[0].token_ids == [1, 2, 3, 4, 5]
    assert results[0].assistant_mask == [0, 0, 1, 0, 1]
    assert results[0].choice_offsets == [2, 4]


def test_tokenize_vllm_trajectory_histories_splits_non_append_turns() -> None:
    first_choice = _choice([1, 2], [3])
    second_choice = _choice([9], [10])
    trajectory = Trajectory(messages_and_choices=[], reward=1.0)

    results = tokenize_vllm_trajectory_histories(
        tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
        histories=[
            History(messages_and_choices=cast(MessagesAndChoices, [first_choice])),
            History(messages_and_choices=cast(MessagesAndChoices, [second_choice])),
        ],
        advantage=1.0,
        allow_training_without_logprobs=False,
        trajectory=trajectory,
    )

    assert [result.token_ids for result in results] == [[1, 2, 3], [9, 10]]
    assert [result.choice_offsets for result in results] == [[2], [1]]


def test_tokenize_trajectory_allows_missing_logprobs_when_requested() -> None:
    choice = _choice([10], [20, 21], with_logprobs=False)
    messages = cast(MessagesAndChoices, [{"role": "user", "content": "Hi"}, choice])

    result = tokenize_trajectory(
        tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
        image_processor=None,
        history=History(messages_and_choices=messages),
        advantage=1.0,
        allow_training_without_logprobs=True,
        trajectory=Trajectory(messages_and_choices=messages, reward=1.0),
    )

    assert result is not None
    assert result.token_ids == [10, 20, 21]
    assert result.assistant_mask == [0, 1, 1]
    assert all(math.isnan(logprob) for logprob in result.logprobs)


def test_tokenize_trajectory_uses_exact_tokens_for_tool_call_choice() -> None:
    choice = _choice(
        [10],
        [65],
        message={
            "content": "prefix",
            "refusal": None,
            "role": "assistant",
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "arguments": '{"offer_id": None}',
                        "name": "create_booking",
                    },
                    "type": "function",
                }
            ],
        },
    )
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Book it."},
            choice,
        ],
    )

    result = tokenize_trajectory(
        tokenizer=_Qwen3_5FakeTokenizer(),  # type: ignore[arg-type]
        image_processor=None,
        history=History(messages_and_choices=messages),
        advantage=1.0,
        allow_training_without_logprobs=False,
        trajectory=Trajectory(messages_and_choices=messages, reward=1.0),
    )

    assert result is not None
    assistant_ids = [
        token_id
        for token_id, mask in zip(result.token_ids, result.assistant_mask)
        if mask
    ]
    assert assistant_ids == [65]


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

    assert tokenizer.apply_chat_template_kwargs
    assert all(
        call.get("enable_thinking") is False and call.get("preserve_thinking") is True
        for call in tokenizer.apply_chat_template_kwargs
    )


def test_tokenize_sft_batch_normalizes_mapping_tool_arguments_for_chat_template() -> (
    None
):
    tokenizer = _Qwen3_5FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
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
        ],
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[Trajectory(messages_and_choices=messages, reward=1.0)],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
    )

    assert batch.num_trajectories == 1
