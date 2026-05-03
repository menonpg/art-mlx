from .backend import TinkerBackend
from .renderers import get_renderer_name

__all__ = ["TinkerBackend", "get_renderer_name", "OpenAICompatibleTinkerServer"]


def __getattr__(name: str):
    if name != "OpenAICompatibleTinkerServer":
        raise AttributeError(name)
    from .server import OpenAICompatibleTinkerServer

    return OpenAICompatibleTinkerServer
