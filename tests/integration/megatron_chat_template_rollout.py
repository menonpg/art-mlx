from __future__ import annotations

import json
from pathlib import Path

from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

import art
from art.local import LocalBackend
from art.preprocessing.tokenize import _normalize_tool_call_arguments_for_chat_template


def _slugify(value: str) -> str:
    return value.lower().replace("/", "_").replace(".", "_").replace("-", "_")


def _artifact_dir(base_model: str) -> Path:
    root = Path(__file__).resolve().parents[2] / ".local" / "model_support_validation"
    path = root / _slugify(base_model) / "chat_template_rollout"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _choice_for_text(text: str, token_ids: list[int]) -> Choice:
    return Choice.model_validate(
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": {
                "content": [
                    {
                        "token": f"token_id:{token_id}",
                        "bytes": list(str(token_id).encode("utf-8")),
                        "logprob": -0.1,
                        "top_logprobs": [],
                    }
                    for token_id in token_ids
                ],
                "refusal": None,
            },
            "message": {
                "content": text,
                "refusal": None,
                "role": "assistant",
                "annotations": None,
                "audio": None,
                "function_call": None,
                "tool_calls": [],
            },
        }
    )


class ChatTemplateRolloutReport(BaseModel):
    base_model: str
    output_dir: str
    packed_num_sequences: int
    packed_sequence_length: int
    assistant_token_count: int
    requires_mapping_tool_arguments: bool
    normalized_mapping_tool_arguments: bool


def run_chat_template_rollout(base_model: str) -> ChatTemplateRolloutReport:
    output_dir = _artifact_dir(base_model)
    backend = LocalBackend(path=str(output_dir))
    model = art.TrainableModel(
        name="model-support-chat-template",
        project="model-support-validation",
        base_model=base_model,
        _internal_config={"init_args": {"max_seq_length": 2048}},
    )
    tokenizer = backend._tokenizers.get(base_model)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        backend._tokenizers[base_model] = tokenizer

    maybe_ids = tokenizer.encode("maybe", add_special_tokens=False)
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    trajectory_group = art.TrajectoryGroup(
        [
            art.Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": "Respond with one word."},
                    _choice_for_text("maybe", maybe_ids),
                ],
                reward=1.0,
            ),
            art.Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": "Respond with one word."},
                    _choice_for_text("yes", yes_ids),
                ],
                reward=0.0,
            ),
        ]
    )
    packed_tensors = backend._get_packed_tensors(
        model,
        [trajectory_group],
        advantage_balance=0.0,
        allow_training_without_logprobs=False,
        scale_rewards=True,
        plot_tensors=False,
        packed_sequence_length=512,
        logprob_calculation_chunk_size=256,
    )
    if packed_tensors is None:
        raise RuntimeError("chat template rollout packing produced no packed tensors")

    requires_mapping_tool_arguments = "tool_call.arguments|items" in str(
        getattr(tokenizer, "chat_template", "")
    )
    normalized_mapping_tool_arguments = False
    if requires_mapping_tool_arguments:
        normalized = _normalize_tool_call_arguments_for_chat_template(
            tokenizer,
            [
                {"role": "user", "content": "Use the weather tool."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": json.dumps(
                                    {"city": "San Francisco", "days": 3}
                                ),
                            },
                        }
                    ],
                },
            ],
        )
        normalized_mapping_tool_arguments = isinstance(
            normalized[1]["tool_calls"][0]["function"]["arguments"],
            dict,
        )

    report = ChatTemplateRolloutReport(
        base_model=base_model,
        output_dir=str(output_dir),
        packed_num_sequences=int(packed_tensors["tokens"].shape[0]),
        packed_sequence_length=int(packed_tensors["tokens"].shape[1]),
        assistant_token_count=int(packed_tensors["assistant_mask"].sum().item()),
        requires_mapping_tool_arguments=requires_mapping_tool_arguments,
        normalized_mapping_tool_arguments=normalized_mapping_tool_arguments,
    )
    (output_dir / "report.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return report
