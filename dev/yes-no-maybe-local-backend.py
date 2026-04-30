import asyncio
from itertools import permutations
import os
import random
import uuid

from dotenv import load_dotenv
import openai

try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import art
from art.local import LocalBackend


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

    backend = LocalBackend()
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-4B")
    model = art.TrainableModel(
        name=os.environ.get("MODEL_NAME", f"yes-no-maybe-local-{uuid.uuid4().hex[:8]}"),
        project="yes-no-maybe",
        base_model=base_model,
        _internal_config=art.dev.InternalModelConfig(
            engine_args=art.dev.EngineArgs(enforce_eager=True),
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
