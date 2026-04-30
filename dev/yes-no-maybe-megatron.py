"""Yes-no-maybe training demo for the Megatron backend.

By default this runs Qwen 3.6 in dedicated merged mode, which needs two GPUs:
GPU 0 runs Megatron training and GPU 1 runs the dedicated vLLM inference server.
After each train step, Megatron keeps the LoRA adapter checkpoint and pushes merged
weights into vLLM for rollouts, because direct vLLM LoRA serving does not yet
reflect these target-parameter MoE adapters reliably. Override TRAINER_GPU_IDS,
INFERENCE_GPU_IDS, or ROLLOUT_WEIGHTS_MODE if you need a different layout.
"""

import asyncio
from itertools import permutations
import os
import random
import uuid

from dotenv import load_dotenv
import openai

import art
from art.megatron import MegatronBackend


async def rollout(
    client: openai.AsyncOpenAI, model: art.TrainableModel, prompt: str
) -> art.Trajectory:
    messages: art.Messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    chat_completion = await client.chat.completions.create(
        messages=messages, model=model.get_inference_name(), max_tokens=100, timeout=100
    )
    choice = chat_completion.choices[0]
    content = choice.message.content
    assert isinstance(content, str)
    if content == "yes":
        reward = 0.5
    elif content == "no":
        reward = 0.75
    elif content == "maybe":
        reward = 1.0
    else:
        reward = 0.0
    return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)


def with_quotes(w: str) -> str:
    return f"'{w}'"


async def main():
    load_dotenv()

    backend = MegatronBackend()
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3.6-35B-A3B")
    model = art.TrainableModel(
        name=os.environ.get(
            "MODEL_NAME", f"yes-no-maybe-megatron-{uuid.uuid4().hex[:8]}"
        ),
        project="yes-no-maybe-megatron",
        base_model=base_model,
        _internal_config=art.dev.InternalModelConfig(
            engine_args=art.dev.EngineArgs(
                gpu_memory_utilization=float(
                    os.environ.get("GPU_MEMORY_UTILIZATION", "0.8")
                ),
                max_model_len=int(os.environ.get("MAX_MODEL_LEN", "4096")),
                max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "8")),
                tensor_parallel_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
            ),
            trainer_gpu_ids=[
                int(gpu_id)
                for gpu_id in os.environ.get("TRAINER_GPU_IDS", "0").split(",")
            ],
            inference_gpu_ids=[
                int(gpu_id)
                for gpu_id in os.environ.get("INFERENCE_GPU_IDS", "1").split(",")
            ],
            rollout_weights_mode=os.environ.get("ROLLOUT_WEIGHTS_MODE", "merged"),
            chat_template_kwargs={
                "enable_thinking": False,
                "preserve_thinking": True,
            },
        ),
    )

    try:
        await model.register(backend)

        prompts = [
            f"{prefix} with {', '.join([with_quotes(w) if use_quotes else w for w in words]) if len(words) == 3 else f'{words[0]}' + (f' or {words[1]}' if len(words) > 1 else '')}"
            for prefix in ["respond", "just respond"]
            for use_quotes in [True, False]
            for words in (
                list(p) for n in [3, 2] for p in permutations(["yes", "no", "maybe"], n)
            )
        ]
        prompts = prompts[: int(os.environ.get("PROMPTS_LIMIT", str(len(prompts))))]

        openai_client = model.openai_client()
        max_steps = int(os.environ.get("NUM_STEPS", "20"))
        groups_per_step = int(os.environ.get("GROUPS_PER_STEP", str(len(prompts))))
        rollouts_per_group = int(
            os.environ.get(
                "ROLLOUTS_PER_GROUP",
                os.environ.get("ROLLOUTS_PER_PROMPT", "32"),
            )
        )
        packed_sequence_length = int(
            os.environ.get(
                "PACKED_SEQUENCE_LENGTH",
                os.environ.get("MAX_SEQ_LENGTH", "4096"),
            )
        )
        start_step = await model.get_step()
        for _ in range(start_step, start_step + max_steps):
            step_prompts = random.sample(
                prompts,
                k=min(groups_per_step, len(prompts)),
            )
            train_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        rollout(openai_client, model, prompt)
                        for _ in range(rollouts_per_group)
                    )
                    for prompt in step_prompts
                )
            )
            result = await backend.train(
                model,
                train_groups,
                learning_rate=1e-4,
                packed_sequence_length=packed_sequence_length,
            )
            await model.log(
                train_groups,
                metrics=result.metrics,
                step=result.step,
                split="train",
            )
    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
