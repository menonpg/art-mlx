from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys

_MAMBA_PREFIX = "mamba_ssm"
_MAMBA_BLOCKER_SENTINEL = "_art_mamba_ssm_blocker"
_BROKEN_MAMBA_DISABLED = False


def _is_mamba_name(module_name: str) -> bool:
    return module_name == _MAMBA_PREFIX or module_name.startswith(_MAMBA_PREFIX + ".")


def _is_broken_mamba_error(error: BaseException) -> bool:
    checked: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in checked:
        checked.add(id(current))
        message = str(current).lower()
        if (
            "mamba_ssm" in message
            and "ssd_chunk_scan" in message
            and "_chunk_scan_fwd" in message
        ):
            return True
        current = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )
    return False


class _MambaImportBlockerLoader(importlib.abc.Loader):
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        return None

    def exec_module(self, module) -> None:  # type: ignore[no-untyped-def]
        raise ModuleNotFoundError(f"No module named '{self.module_name}'")


class _MambaImportBlockerFinder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        setattr(self, _MAMBA_BLOCKER_SENTINEL, True)

    def find_spec(self, fullname, path=None, target=None):  # type: ignore[no-untyped-def]
        if not _BROKEN_MAMBA_DISABLED or not _is_mamba_name(fullname):
            return None
        return importlib.machinery.ModuleSpec(
            name=fullname,
            loader=_MambaImportBlockerLoader(fullname),
            is_package=fullname == _MAMBA_PREFIX,
        )


def _patch_find_spec_for_mamba() -> None:
    current_find_spec = importlib.util.find_spec
    if getattr(current_find_spec, "_art_mamba_find_spec_patch", False):
        return

    def _blocked_find_spec(name, package=None):  # type: ignore[no-untyped-def]
        if (
            _BROKEN_MAMBA_DISABLED
            and isinstance(name, str)
            and _is_mamba_name(
                importlib.util.resolve_name(name, package)
                if name.startswith(".") and package
                else name
            )
        ):
            return None
        return current_find_spec(name, package)

    _blocked_find_spec._art_mamba_find_spec_patch = True  # type: ignore[attr-defined]
    importlib.util.find_spec = _blocked_find_spec


def _install_mamba_blocker() -> None:
    _patch_find_spec_for_mamba()
    for finder in sys.meta_path:
        if getattr(finder, _MAMBA_BLOCKER_SENTINEL, False):
            return
    sys.meta_path.insert(0, _MambaImportBlockerFinder())


def _clear_mamba_modules() -> None:
    for module_name in list(sys.modules):
        if _is_mamba_name(module_name):
            sys.modules.pop(module_name, None)


def disable_broken_mamba_ssm() -> bool:
    global _BROKEN_MAMBA_DISABLED
    if _BROKEN_MAMBA_DISABLED:
        _install_mamba_blocker()
        return True

    try:
        if importlib.util.find_spec(_MAMBA_PREFIX) is None:
            return False
    except Exception:
        return False

    try:
        importlib.import_module(_MAMBA_PREFIX)
        return False
    except Exception as error:
        if not _is_broken_mamba_error(error):
            return False

    _BROKEN_MAMBA_DISABLED = True
    _clear_mamba_modules()
    _install_mamba_blocker()
    return True
