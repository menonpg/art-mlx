from .yes_no_trainability import run_megatron_dedicated_yes_no_trainability


def run_native_vllm_lora(base_model: str):
    return run_megatron_dedicated_yes_no_trainability(
        base_model,
        rollout_weights_mode="lora",
    )
