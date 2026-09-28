"""Nonblocking local advisory locks released by the operating system on exit.

Lock files persist and contain no ownership state. Callers must not unlink them:
keeping one stable file identity prevents two writers from locking different
inodes at the same path. Old content left by an interrupted writer is harmless.
"""

from __future__ import annotations

import errno
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


class LockUnavailableError(ValueError):
    """Another local process currently owns the requested advisory lock."""


def _acquire(handle: BinaryIO) -> None:
    handle.seek(0)
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            raise LockUnavailableError("prospective writer is locked by another process") from exc
        raise


def _release(handle: BinaryIO) -> None:
    handle.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def advisory_lock(path: Path) -> Iterator[None]:
    """Own a stable file's nonblocking OS lock for the context lifetime.

    Windows locks its first byte; POSIX locks the open file. Closing a process's
    handles releases ownership even after forced termination. An existing empty
    file or a former PID sentinel does not represent an active owner.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        _acquire(handle)
        try:
            yield
        finally:
            _release(handle)
