import argparse
from pathlib import Path

from art.megatron.model_support.spec import ArchitectureReport
from art.megatron.model_support.workflow import (
    run_chat_template_rollout_stage,
    run_correctness_sensitivity_stage,
    run_hf_parity_stage,
    run_lora_coverage_stage,
    run_merged_vllm_serving_stage,
    run_yes_no_trainability_stage,
)

_STAGE_RUNNERS = {
    "hf_parity": run_hf_parity_stage,
    "lora_coverage": run_lora_coverage_stage,
    "merged_vllm_serving": run_merged_vllm_serving_stage,
    "correctness_sensitivity": run_correctness_sensitivity_stage,
    "chat_template_rollout": run_chat_template_rollout_stage,
    "yes_no_trainability": run_yes_no_trainability_stage,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--architecture-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    architecture = ArchitectureReport.model_validate_json(
        Path(args.architecture_json).read_text(encoding="utf-8")
    )
    stage_runner = _STAGE_RUNNERS[args.stage]
    result = stage_runner(
        base_model=args.base_model,
        architecture=architecture,
    )
    Path(args.output_json).write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
