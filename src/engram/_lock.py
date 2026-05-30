"""Cross-platform advisory file lock — zero external dependencies.

Used to serialise writes to a tier's ``.pst`` / ``.meta.json`` so two
processes — e.g. the warm ``engram-mcp`` server and a one-shot CLI/skill
invocation, or two agent sessions sharing the global tier — can't corrupt
the store by flushing at the same time.

Mechanism (advisory, exclusive, blocking-with-timeout):

* **POSIX**   — ``fcntl.flock`` on a dedicated ``<pst>.lock`` fd.
* **Windows** — ``msvcrt.locking`` on the same.

The lock lives on the fd, so the OS releases it automatically when the
process exits — there are no stale lock files to clean up.

Scope / limits
--------------
This prevents *physical* corruption from concurrent flushes (two writers
interleaving bytes into the same file).  It does **not** prevent a *lost
update* when two long-lived processes each hold an in-memory copy of the
tier and flush in turn — the second flush overwrites the first.  The
``engram-mcp`` server is intended to be the primary writer; heavy
concurrent writers should coordinate at a higher level.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional


class FileLock:
    """Exclusive advisory lock on ``lock_path``.

    Use as a context manager::

        with FileLock(str(pst) + ".lock"):
            ...  # critical section: write the files

    ``acquire`` blocks (polling) until the lock is free or ``timeout``
    seconds elapse, in which case it raises :class:`TimeoutError` rather
    than wedging the caller forever behind a hung holder.
    """

    def __init__(self, lock_path: str, timeout: float = 10.0, poll: float = 0.05):
        self._lock_path = str(lock_path)
        self._timeout = float(timeout)
        self._poll = float(poll)
        self._fd: Optional[int] = None

    def acquire(self) -> "FileLock":
        if self._fd is not None:
            return self  # already held by this instance
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self._timeout
        try:
            while True:
                if self._try_lock(fd):
                    self._fd = fd
                    return self
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"could not acquire lock {self._lock_path!r} within "
                        f"{self._timeout}s (another process is writing?)"
                    )
                time.sleep(self._poll)
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _try_lock(fd: int) -> bool:
        """Non-blocking exclusive lock attempt; ``True`` on success."""
        if sys.platform == "win32":
            import msvcrt
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError:
                return False

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            if sys.platform == "win32":
                import msvcrt
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            os.close(fd)

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


__all__ = ["FileLock"]
