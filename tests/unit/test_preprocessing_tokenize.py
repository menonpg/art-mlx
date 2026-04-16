import importlib
import sys
from typing import cast

from openai.types.chat.chat_completion import Choice
import pytest
from transformers.tokenization_utils_base import BatchEncoding

from art.preprocessing.tokenize import tokenize_trajectory
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

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=None,
        **kwargs,
    ):
        del tools, kwargs
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
        del kwargs
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
        )


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
