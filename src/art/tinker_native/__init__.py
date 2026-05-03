__all__ = ["TinkerNativeBackend"]


def __getattr__(name: str):
    if name != "TinkerNativeBackend":
        raise AttributeError(name)
    from .backend import TinkerNativeBackend

    return TinkerNativeBackend
