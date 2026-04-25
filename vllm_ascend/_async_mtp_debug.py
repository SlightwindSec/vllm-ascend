# SPDX-License-Identifier: Apache-2.0
"""Standalone file-based tracer for async-scheduling + MTP debugging.

Lines are written to /home/caoyi/x/logs/mtp.log so the trace stays
separate from serve.log. Callers gate on rank-0 themselves; this module
only handles formatting and file I/O. Always enabled for this debug
branch.
"""
import os
import time

_LOG_PATH = "/home/caoyi/x/logs/mtp.log"
_fh = None


def _open() -> object | None:
    global _fh
    if _fh is not None:
        return _fh
    try:
        os.makedirs(os.path.dirname(_LOG_PATH) or ".", exist_ok=True)
        _fh = open(_LOG_PATH, "a", buffering=1)  # noqa: SIM115
        _fh.write(f"---- amtp tracer pid={os.getpid()} t={time.time():.3f} ----\n")
    except OSError as e:
        _fh = None
        print(f"[AMTP] failed to open {_LOG_PATH}: {e}", flush=True)
    return _fh


def amtp_log(fmt: str, *args) -> None:
    fh = _open()
    if fh is None:
        return
    try:
        msg = fmt % args if args else fmt
    except Exception as e:
        msg = f"<format-error fmt={fmt!r} err={e}>"
    fh.write(f"{time.time():.3f} pid={os.getpid()} {msg}\n")
