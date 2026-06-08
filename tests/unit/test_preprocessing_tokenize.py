import importlib
import sys
from typing import Any, cast

from openai.types.chat.chat_completion import Choice
import pytest
from transformers.tokenization_utils_base import BatchEncoding

from art.preprocessing.tokenize import tokenize_sft_batch, tokenize_trajectory
from art.trajectories import History, Trajectory
from art.types import MessagesAndChoices

if "tests" not in sys.path:
    sys.path.insert(0, "tests")

build_chat_template_conformance_inputs = importlib.import_module(
    "support.chat_template_conformance_cases"
).build_chat_template_conformance_inputs

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


class _NonRoundTripDecodeTokenizer(_FakeTokenizer):
    def decode(self, token_ids):
        if isinstance(token_ids, int) and token_ids > self.vocab_size // 2:
            return "not-a-single-token"
        return super().decode(token_ids)


def test_tokenize_trajectory_accepts_batchencoding_chat_template_output() -> None:
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


def test_tokenize_trajectory_does_not_require_unused_vocab_token_roundtrip() -> None:
    tokenizer = _NonRoundTripDecodeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "OK"},
        ],
    )
    result = tokenize_trajectory(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        image_processor=None,
        history=History(messages_and_choices=messages),
        advantage=1.0,
        allow_training_without_logprobs=True,
        trajectory=Trajectory(messages_and_choices=messages, reward=1.0),
    )

    assert result is not None
    assert [
        token_id
        for token_id, mask in zip(result.token_ids, result.assistant_mask)
        if mask
    ] == tokenizer.encode("OK", add_special_tokens=False)


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
                "content": "<think>\nreasoning\n</think>\n\nOK",
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


def test_tokenize_trajectory_uses_exact_tokens_for_malformed_final_tool_call() -> None:
    tokenizer = _Qwen3_5FakeTokenizer()
    choice = Choice.model_validate(
        {
            "finish_reason": "tool_calls",
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
        }
    )
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Book it."},
            choice,
        ],
    )
    result = tokenize_trajectory(
        tokenizer=tokenizer,  # type: ignore[arg-type]
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


def test_tokenize_trajectory_non_final_tool_call_mutation_changes_prefill_tokens() -> (
    None
):
    tokenizer = _Qwen3_5FakeTokenizer()
    inputs = build_chat_template_conformance_inputs(tokenizer)  # type: ignore[arg-type]

    base = tokenize_trajectory(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        image_processor=None,
        history=History(
            messages_and_choices=inputs.non_final_tool_call_base.messages_and_choices,
            tools=inputs.non_final_tool_call_base.tools,
        ),
        advantage=1.0,
        allow_training_without_logprobs=False,
        trajectory=inputs.non_final_tool_call_base,
    )
    mutated = tokenize_trajectory(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        image_processor=None,
        history=History(
            messages_and_choices=inputs.non_final_tool_call_mutated.messages_and_choices,
            tools=inputs.non_final_tool_call_mutated.tools,
        ),
        advantage=1.0,
        allow_training_without_logprobs=False,
        trajectory=inputs.non_final_tool_call_mutated,
    )

    assert base is not None
    assert mutated is not None
    assert len(base.choice_offsets) >= 2
    assert len(mutated.choice_offsets) >= 2
    assert (
        base.token_ids[: base.choice_offsets[-1]]
        != mutated.token_ids[: mutated.choice_offsets[-1]]
    )


def test_tokenize_trajectory_rejects_assistant_tool_calls_without_logprobs() -> None:
    tokenizer = _Qwen3_5FakeTokenizer()
    inputs = build_chat_template_conformance_inputs(tokenizer)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Assistant message has tool_calls"):
        tokenize_trajectory(
            tokenizer=tokenizer,  # type: ignore[arg-type]
            image_processor=None,
            history=History(
                messages_and_choices=inputs.unsupported_assistant_tool_calls.messages_and_choices,
                tools=inputs.unsupported_assistant_tool_calls.tools,
            ),
            advantage=1.0,
            allow_training_without_logprobs=True,
            trajectory=inputs.unsupported_assistant_tool_calls,
        )
