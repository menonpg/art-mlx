from typing import Any

__all__ = ["MegatronBackend"]


def __getattr__(name: str) -> Any:
    if name == "MegatronBackend":
        from .runtime.backend import MegatronBackend

        return MegatronBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
