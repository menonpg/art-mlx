"""Shared compile-state helpers for ART's Megatron backend."""

import os


def megatron_compile_enabled() -> bool:
    value = os.environ.get("ART_DISABLE_MEGATRON_COMPILE", "0")
    return value.strip().lower() not in {"1", "true", "yes", "on"}
