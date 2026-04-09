"""Compatibility wrapper around the ART-owned vLLM runtime entrypoint."""

from art_vllm_runtime.dedicated_server import _append_cli_arg, main, parse_args

__all__ = ["_append_cli_arg", "main", "parse_args"]


if __name__ == "__main__":
    main()
