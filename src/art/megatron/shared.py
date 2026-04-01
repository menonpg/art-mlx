from .jobs import MegatronSFTTrainingJob, MegatronTrainingJob
from .train import (
    DEFAULT_MODEL_IDENTIFIER,
    TrainingRuntime,
    build_training_runtime,
    finalize_megatron_job,
    merge_lora_adapter,
    run_megatron_rl_job,
    run_megatron_sft_job,
    run_megatron_worker_loop,
)

MegatronTrainContext = TrainingRuntime
MegatronJob = MegatronTrainingJob | MegatronSFTTrainingJob


def create_megatron_train_context(
    model_identifier: str = DEFAULT_MODEL_IDENTIFIER,
) -> MegatronTrainContext:
    return build_training_runtime(model_identifier=model_identifier)
