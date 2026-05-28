import subprocess
import sys
import textwrap


def test_registry_metadata_queries_do_not_import_handlers() -> None:
    code = textwrap.dedent(
        """
        import sys

        from art.megatron.model_support import (
            default_target_modules_for_model,
            model_uses_expert_parallel,
            native_vllm_lora_status_for_model,
        )

        assert default_target_modules_for_model("Qwen/Qwen3.5-397B-A17B") == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "experts",
        ]
        assert model_uses_expert_parallel("Qwen/Qwen3.5-397B-A17B") is True
        assert native_vllm_lora_status_for_model("Qwen/Qwen3.5-397B-A17B") == "validated"
        forbidden = [
            "art.megatron.model_support.handlers",
            "art.megatron.model_support.handlers.qwen3_5",
            "megatron.bridge",
        ]
        loaded = [name for name in forbidden if name in sys.modules]
        assert loaded == [], loaded
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
