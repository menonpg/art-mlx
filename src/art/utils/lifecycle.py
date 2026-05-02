from __future__ import annotations

import atexit
from collections.abc import Callable, Sequence
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


def managed_process_cmd(
    command: Sequence[str], *, parent_pid: int | None = None
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve().with_name("managed_process.py")),
        "--parent-pid",
        str(parent_pid or os.getpid()),
        "--",
        *command,
    ]


def kill_process_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except ProcessLookupError:
        pass


def terminate_popen_process_group(
    process: subprocess.Popen[Any],
    *,
    timeout: float = 5.0,
) -> None:
    if process.poll() is None:
        kill_process_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_group(process.pid, signal.SIGKILL)
        process.wait()


def terminate_asyncio_process_group(process: Any, *, timeout: float = 5.0) -> None:
    if process.returncode is None:
        kill_process_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            finished_pid, _ = os.waitpid(process.pid, os.WNOHANG)
        except ChildProcessError:
            return
        if finished_pid:
            return
        time.sleep(0.05)
    kill_process_group(process.pid, signal.SIGKILL)
    try:
        os.waitpid(process.pid, 0)
    except ChildProcessError:
        pass


class ServiceLifecycle:
    def __init__(self) -> None:
        self.closing = False
        self._close_callback: Callable[[], None] | None = None
        self._previous_signal_handlers: dict[int, Any] = {}

    def begin_close(self) -> bool:
        if self.closing:
            return False
        self.closing = True
        return True

    def install_parent_cleanup(self, close: Callable[[], None]) -> None:
        if self._close_callback is not None:
            return
        self._close_callback = close
        atexit.register(close)

        def _default_signal_exit(signum: int) -> None:
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.getsignal(signum)
            self._previous_signal_handlers[signum] = previous

            def _handler(received_signum, frame, *, _previous=previous):
                close()
                if callable(_previous):
                    _previous(received_signum, frame)
                    return
                if _previous == signal.SIG_IGN:
                    return
                _default_signal_exit(received_signum)

            signal.signal(signum, _handler)

    def restore_parent_cleanup(self) -> None:
        if self._close_callback is not None:
            try:
                atexit.unregister(self._close_callback)
            except ValueError:
                pass
            self._close_callback = None
        for signum, previous in self._previous_signal_handlers.items():
            signal.signal(signum, previous)
        self._previous_signal_handlers.clear()
