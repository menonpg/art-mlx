"""Launch the dedicated LocalBackend artifact-resume yes-no-maybe run on SkyPilot."""

from __future__ import annotations

import argparse
import os
import textwrap

from dotenv import load_dotenv
import sky
from sky import ClusterStatus

load_dotenv()

DEFAULT_IMAGE_ID = "docker:nvidia/cuda:12.8.1-devel-ubuntu22.04"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _format_env_bool(value: bool) -> str:
    return "true" if value else "false"


parser = argparse.ArgumentParser(
    description="Launch a dedicated LocalBackend PipelineTrainer artifact-resume run."
)
parser.add_argument("--fast", action="store_true")
parser.add_argument("--cluster-name", type=str, default="art-llama31-localbackend-artifact-ynm")
parser.add_argument("--project", type=str, default="llama31-localbackend-artifact-ynm")
parser.add_argument(
    "--model-name-prefix",
    type=str,
    default="llama31-localbackend-artifact-ynm",
)
parser.add_argument("--artifact", type=str, default=os.environ.get("WANDB_ARTIFACT"))
parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
parser.add_argument("--accelerator", type=str, default="H200:2")
parser.add_argument("--image-id", type=str, default=DEFAULT_IMAGE_ID)
parser.add_argument("--num-steps", type=int, default=20)
parser.add_argument("--rollouts-per-prompt", type=int, default=8)
parser.add_argument("--num-rollout-workers", type=int, default=4)
parser.add_argument("--min-batch-size", type=int, default=4)
parser.add_argument("--max-batch-size", type=int, default=4)
parser.add_argument("--eval-prompts", type=int, default=12)
parser.add_argument("--eval-every-n-steps", type=int, default=2)
parser.add_argument("--learning-rate", type=float, default=5e-5)
parser.add_argument("--max-tokens", type=int, default=5)
parser.add_argument("--eval-max-tokens", type=int, default=5)
parser.add_argument("--timeout", type=float, default=120.0)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
parser.add_argument("--max-model-len", type=int, default=2048)
parser.add_argument("--max-seq-length", type=int, default=2048)
parser.add_argument("--max-num-seqs", type=int, default=16)
parser.add_argument(
    "--load-in-4bit", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument(
    "--load-in-16bit", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--force-download-artifact",
    action=argparse.BooleanOptionalAction,
    default=False,
)
args = parser.parse_args()

if not args.artifact:
    parser.error("--artifact is required (or set WANDB_ARTIFACT in the environment).")

cluster_name = args.cluster_name
cluster_prefix = os.environ.get("CLUSTER_PREFIX")
if cluster_prefix:
    cluster_name = f"{cluster_prefix}-{cluster_name}"

setup_script = textwrap.dedent("""\
    echo 'Setting up environment...'
    apt-get update
    apt-get install -y python3 python3-pip python-is-python3 git curl ninja-build
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
""")

env = [
    f"PROJECT={args.project}",
    f"MODEL_NAME={args.model_name_prefix}-$(date +%Y%m%d-%H%M%S)",
    f"WANDB_ARTIFACT={args.artifact}",
    f"BASE_MODEL={args.base_model}",
    "TRAINER_GPU_IDS=0",
    "INFERENCE_GPU_IDS=1",
    "ACCELERATE_MIXED_PRECISION=bf16",
    f"FORCE_DOWNLOAD_ARTIFACT={_format_env_bool(args.force_download_artifact)}",
    f"NUM_STEPS={args.num_steps}",
    f"ROLLOUTS_PER_PROMPT={args.rollouts_per_prompt}",
    f"NUM_ROLLOUT_WORKERS={args.num_rollout_workers}",
    f"MIN_BATCH_SIZE={args.min_batch_size}",
    f"MAX_BATCH_SIZE={args.max_batch_size}",
    f"EVAL_PROMPTS={args.eval_prompts}",
    f"EVAL_EVERY_N_STEPS={args.eval_every_n_steps}",
    f"LEARNING_RATE={args.learning_rate}",
    f"MAX_TOKENS={args.max_tokens}",
    f"EVAL_MAX_TOKENS={args.eval_max_tokens}",
    f"TIMEOUT={args.timeout}",
    f"GPU_MEMORY_UTILIZATION={args.gpu_memory_utilization}",
    f"MAX_MODEL_LEN={args.max_model_len}",
    f"MAX_SEQ_LENGTH={args.max_seq_length}",
    f"MAX_NUM_SEQS={args.max_num_seqs}",
    f"LOAD_IN_4BIT={_format_env_bool(args.load_in_4bit)}",
    f"LOAD_IN_16BIT={_format_env_bool(args.load_in_16bit)}",
]
env_block = " \\\n    ".join(env)

run_script = textwrap.dedent(
    f"""\
    source $HOME/.local/bin/env
    cd ~/sky_workdir
    ~/.local/bin/uv sync --extra backend
    {env_block} \
    ~/.local/bin/uv run --python 3.11 --extra backend dev/pipeline_localbackend_artifact_yes_no_maybe.py
"""
)

task = sky.Task(
    name="llama31-localbackend-artifact-ynm",
    setup=setup_script,
    run=run_script,
    workdir=".",
)
task.set_resources(
    sky.Resources(
        accelerators=args.accelerator,
        cloud=sky.clouds.Kubernetes(),
        image_id=args.image_id,
    )
)
task.set_file_mounts({"~/sky_workdir/.env": ".env"})

print(f"Launching on cluster: {cluster_name}")
print(f"  project: {args.project}")
print(f"  artifact: {args.artifact}")
print(f"  base_model: {args.base_model}")
print(f"  accelerator: {args.accelerator}")
print(f"  num_steps: {args.num_steps}")
print(f"  rollouts_per_prompt: {args.rollouts_per_prompt}")
print(f"  num_rollout_workers: {args.num_rollout_workers}")
print(f"  min_batch_size: {args.min_batch_size}")
print(f"  max_batch_size: {args.max_batch_size}")
print(f"  eval_prompts: {args.eval_prompts}")
print(f"  eval_every_n_steps: {args.eval_every_n_steps}")
print(f"  learning_rate: {args.learning_rate}")
print(f"  gpu_memory_utilization: {args.gpu_memory_utilization}")
print(f"  max_model_len: {args.max_model_len}")
print(f"  max_seq_length: {args.max_seq_length}")
print(f"  max_num_seqs: {args.max_num_seqs}")
print(f"  load_in_4bit: {args.load_in_4bit}")
print(f"  load_in_16bit: {args.load_in_16bit}")

cluster_status = sky.stream_and_get(sky.status(cluster_names=[cluster_name]))
if cluster_status and cluster_status[0]["status"] == ClusterStatus.UP:
    print(f"Cluster {cluster_name} is UP. Canceling any active jobs...")
    sky.stream_and_get(sky.cancel(cluster_name, all=True))

job_id, _ = sky.stream_and_get(
    sky.launch(
        task,
        cluster_name=cluster_name,
        retry_until_up=True,
        idle_minutes_to_autostop=60,
        down=True,
        fast=args.fast,
    )
)

print(f"Job submitted (ID: {job_id}). Streaming logs...")
exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
print(f"Job {job_id} finished with exit code {exit_code}.")
