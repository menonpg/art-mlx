from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ART-owned child process")
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command")
    return args


def main() -> None:
    args = parse_args()
    if hasattr(os, "setsid") and os.getpgrp() != os.getpid():
        os.setsid()

    process: subprocess.Popen[bytes] | None = None
    child_pgid: int | None = None
    shutting_down = False

    def signal_child_group(sig: signal.Signals) -> None:
        if child_pgid is None:
            return
        try:
            os.killpg(child_pgid, sig)
        except ProcessLookupError:
            pass

    def sweep_child_group() -> None:
        signal_child_group(signal.SIGTERM)
        time.sleep(float(os.environ.get("ART_MANAGED_PROCESS_SWEEP_GRACE", 0.5)))
        signal_child_group(signal.SIGKILL)

    def shutdown(sig: signal.Signals, exit_code: int) -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        signal_child_group(sig)
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_child_group(signal.SIGKILL)
                process.wait()
        sweep_child_group()
        os._exit(exit_code)

    def handle_signal(signum: int, _frame: object | None) -> None:
        shutdown(signal.Signals(signum), 128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    process = subprocess.Popen(args.command, start_new_session=True)
    child_pgid = process.pid

    def monitor_parent() -> None:
        while process is not None and process.poll() is None:
            if os.getppid() != args.parent_pid:
                shutdown(signal.SIGTERM, 1)
            time.sleep(0.5)

    threading.Thread(target=monitor_parent, daemon=True).start()
    return_code = process.wait()
    sweep_child_group()
    sys.exit(return_code)


if __name__ == "__main__":
    main()
