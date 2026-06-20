from typing import Any

_TRAINER_RANK_EXPORTS = (
    "AdamParams",
    "ForwardInput",
    "ForwardOutput",
    "MicroBatch",
    "TopK",
    "TrainerRank",
)

__all__ = ["MegatronBackend", *_TRAINER_RANK_EXPORTS]


def __getattr__(name: str) -> Any:
    if name == "MegatronBackend":
        from .backend import MegatronBackend

        return MegatronBackend
    if name in _TRAINER_RANK_EXPORTS:
        from . import trainer_rank

        return getattr(trainer_rank, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
