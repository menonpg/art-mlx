import asyncio
from dataclasses import dataclass
import datetime
from functools import cached_property
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any, AsyncIterator

from peft.tuners.lora.config import LoraConfig
import torch
from vllm import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.async_llm import AsyncLLM

from .. import dev, types
from ..dev.get_model_config import default_target_modules
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.pack import DiskPackedTensors
from ..preprocessing.tokenize import SFTBatch
from ..unsloth.service import do_sleep, do_wake_up, gc_and_empty_cuda_cache
from ..utils.convert_moe_lora import convert_checkpoint_if_needed
from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir
from ..vllm import get_llm, openai_server_task, run_on_workers
from .jobs import (
    DEFAULT_JOBS_DIR,
    DEFAULT_VLLM_WAKE_LOCK_PATH,
    MegatronTrainingJob,
)
from .train import merge_lora_adapter

safetensors = importlib.import_module("safetensors")
safe_open = safetensors.safe_open


@dataclass
class MegatronService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    _is_sleeping: bool = False
    _latest_step: int = 0
    _lora_id_counter: int = 1
    _megatron_process: asyncio.subprocess.Process | None = None
    _optimizer_state_path: str | None = None

    def _next_lora_id(self) -> int:
        self._lora_id_counter += 1
        return self._lora_id_counter

    def _get_optimizer_state_path(self) -> str:
        if self._optimizer_state_path is not None:
            return self._optimizer_state_path
        self._optimizer_state_path = os.path.join(self.output_dir, "optimizer_states")
        os.makedirs(self._optimizer_state_path, exist_ok=True)
        return self._optimizer_state_path

    def _default_lora_adapter_config(self) -> LoraConfig:
        # Keep in sync with LoRA settings in megatron/train.py.
        return LoraConfig(
            base_model_name_or_path=self.base_model,
            r=1,
            lora_alpha=32,
            target_modules=default_target_modules(self.base_model),
            bias="none",
        )

    def _adapter_has_weights(self, lora_path: str) -> bool:
        adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            return False
        try:
            with safe_open(adapter_path, framework="pt") as adapter_file:
                for key in adapter_file.keys():
                    tensor = adapter_file.get_tensor(key)
                    if torch.any(tensor != 0):
                        return True
        except Exception:
            return False
        return False

    def _create_identity_lora(self, lora_path: str) -> None:
        from unittest.mock import patch

        from accelerate import init_empty_weights
        from peft import get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM

        base_config = AutoConfig.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                base_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        model.name_or_path = self.base_model
        lora_config = self._default_lora_adapter_config()
        lora_config.target_modules = []
        lora_config.target_parameters = [
            name
            for name, _ in model.named_parameters()
            if name.endswith(
                (
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "mlp.experts.gate_up_proj",
                    "mlp.experts.down_proj",
                )
            )
        ]

        meta = torch.device("meta")
        orig_to = torch.nn.Module.to

        def _skip_meta_to(module: torch.nn.Module, *args: Any, **kwargs: Any):
            device = kwargs.get("device") or (args[0] if args else None)
            if device == meta or str(device) == "meta":
                return module
            return orig_to(module, *args, **kwargs)

        with patch.object(torch.nn.Module, "to", _skip_meta_to):
            peft_model = get_peft_model(model, lora_config)

        os.makedirs(lora_path, exist_ok=True)
        peft_model.save_pretrained(lora_path)
        convert_checkpoint_if_needed(lora_path)
        self._default_lora_adapter_config().save_pretrained(lora_path)
        del peft_model, model
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _ensure_identity_lora(self, lora_path: str) -> None:
        if self._adapter_has_weights(lora_path):
            return
        self._create_identity_lora(lora_path)

    def _ensure_lora_adapter_config(
        self, lora_path: str, *, source_path: str | None = None
    ) -> None:
        config_path = os.path.join(lora_path, "adapter_config.json")
        if os.path.exists(config_path):
            return
        os.makedirs(lora_path, exist_ok=True)
        if source_path is not None:
            source_config = os.path.join(source_path, "adapter_config.json")
            if os.path.exists(source_config):
                shutil.copy(source_config, config_path)
                return
        self._default_lora_adapter_config().save_pretrained(lora_path)

    async def _add_lora_aliases(
        self, llm: AsyncLLM, step: int, checkpoint_dir: str
    ) -> None:
        added = await llm.add_lora(
            LoRARequest(
                lora_name=f"{self.model_name}@{step}",
                lora_int_id=self._next_lora_id(),
                lora_path=checkpoint_dir,
            )
        )
        if not added:
            raise RuntimeError(f"Failed to add LoRA adapter for step {step}")
        self._latest_step = step

    async def register_lora_for_step(self, step: int, checkpoint_dir: str) -> None:
        llm = await self.llm
        await llm.pause_generation()
        await self._add_lora_aliases(llm, step, checkpoint_dir)
        await llm.resume_generation()

    async def _ensure_megatron_running(self) -> None:
        """Lazily start Megatron training process if not running."""
        if self._megatron_process is not None:
            if self._megatron_process.returncode is None:
                return
            self._megatron_process = None

        try:
            import megatron.bridge  # type: ignore

            setup_cmd = ""
        except ImportError:
            setup_script = Path(__file__).parent / "setup.sh"
            setup_cmd = f"bash {setup_script} && "

        subprocess.run(["pkill", "-9", "megatron-service"], check=False)
        train_script = Path(__file__).parent / "train.py"
        project_root = Path(__file__).resolve().parents[3]
        num_gpus = torch.cuda.device_count()
        os.environ["MODEL_IDENTIFIER"] = self.base_model

        command = (
            f"{setup_cmd}uv run --project {shlex.quote(str(project_root))} "
            f"torchrun --nproc_per_node {num_gpus} {shlex.quote(str(train_script))}"
        )
        self._megatron_process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(project_root),
        )

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            self._latest_step = 0
        else:
            self._latest_step = get_step_from_dir(self.output_dir)
        self._ensure_identity_lora(lora_path)
        self._ensure_lora_adapter_config(lora_path)

        lora_path_for_server = (
            lora_path if self._adapter_has_weights(lora_path) else None
        )
        server_config = dev.get_openai_server_config(
            model_name=self.model_name,
            base_model=self.base_model,
            log_file=f"{self.output_dir}/logs/vllm.log",
            lora_path=lora_path_for_server,
            config=config,
        )
        await openai_server_task(engine=await self.llm, config=server_config)
        return (
            server_config.get("server_args", {}).get("host") or "0.0.0.0",
            server_config.get("server_args", {}).get("port", 8000),
        )

    async def vllm_engine_is_sleeping(self) -> bool:
        return self._is_sleeping

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        llm = await self.llm
        await llm.pause_generation()
        await llm.reset_prefix_cache()
        await run_on_workers(llm, do_sleep, level=2)
        self._is_sleeping = True
        gc_and_empty_cuda_cache()

        # Start Megatron after vLLM has freed GPU memory.
        await self._ensure_megatron_running()

        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            self._latest_step = 0
        self._ensure_identity_lora(lora_path)
        self._ensure_lora_adapter_config(lora_path)

        self._optimizer_state_path = self._get_optimizer_state_path()

        os.makedirs(DEFAULT_JOBS_DIR, exist_ok=True)
        for job_name in os.listdir(DEFAULT_JOBS_DIR):
            if job_name.endswith(".json"):
                os.remove(os.path.join(DEFAULT_JOBS_DIR, job_name))
        if _config.get("moe_routing_replay_bundle") is not None:
            raise RuntimeError(
                "moe_routing_replay_bundle is only supported for in-process/runtime APIs; "
                "MegatronService subprocess jobs must use moe_routing_replay_path."
            )
        log_dir = "/tmp/megatron_training_logs"
        os.makedirs(log_dir, exist_ok=True)
        job = MegatronTrainingJob(
            lora_path=lora_path,
            optimizer_state_path=self._optimizer_state_path,
            disk_packed_tensors=disk_packed_tensors,
            config=config,
            experimental_config=_config,
            moe_routing_replay_path=_config.get("moe_routing_replay_path"),
            moe_routing_replay_strict=_config.get("moe_routing_replay_strict", True),
            log_path=os.path.join(
                log_dir, f"{datetime.datetime.now().isoformat()}.jsonl"
            ),
        )
        job_path = os.path.join(
            DEFAULT_JOBS_DIR,
            f"{datetime.datetime.now().isoformat()}.json",
        )
        with open(job_path, "w") as f:
            f.write(job.model_dump_json())

        num_lines = 0
        while True:
            await asyncio.sleep(0.1)
            try:
                with open(job.log_path, "a+") as log_file:
                    log_file.seek(0)
                    lines = log_file.readlines()[num_lines:]
                    for line in lines:
                        if line := line.strip():
                            if line == "all done":
                                merge_lora_adapter(lora_path)
                                os.remove(job.log_path)
                                break
                            num_lines += 1
                            yield json.loads(line)
                    else:
                        continue
                    break
            except FileNotFoundError:
                continue

        next_step = self._latest_step + 1
        new_checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
        os.makedirs(new_checkpoint_dir, exist_ok=True)
        shutil.copy(
            f"{lora_path}/adapter_model.safetensors",
            f"{new_checkpoint_dir}/adapter_model.safetensors",
        )
        self._ensure_lora_adapter_config(new_checkpoint_dir, source_path=lora_path)

        wake_lock_path = DEFAULT_VLLM_WAKE_LOCK_PATH
        try:
            with open(wake_lock_path, "w") as lock_file:
                lock_file.write("waking vllm\n")
            await run_on_workers(llm, do_wake_up)
            self._is_sleeping = False
        finally:
            if os.path.exists(wake_lock_path):
                os.remove(wake_lock_path)

        await self._add_lora_aliases(llm, next_step, new_checkpoint_dir)
        await llm.resume_generation()

    # SFT not supported for MegatronService
    async def train_sft(
        self,
        batches: list[Any],
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        raise NotImplementedError("SFT training is not supported for MegatronService")
        yield {}  # Make this a generator

    @cached_property
    def llm(self) -> asyncio.Task[AsyncLLM]:
        engine_args = {
            **self.config.get("engine_args", {}),
            "enable_lora": True,
            "max_loras": self.config.get("engine_args", {}).get("max_loras", 2),
        }
        for key in ["enable_log_requests", "disable_log_requests"]:
            engine_args.pop(key, None)
        return asyncio.create_task(get_llm(AsyncEngineArgs(**engine_args)))  # type: ignore
