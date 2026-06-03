from importlib import import_module
from typing import Any

_MODULES = (
    "types",
    "comm",
    "sparse_kernel",
    "compressor",
    "cp_stage",
    "indexer",
    "cp_attention",
    "planning",
)


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    for module_name in _MODULES:
        module = import_module(f"{__name__}.{module_name}")
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
