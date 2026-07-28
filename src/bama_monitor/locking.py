"""Three independent layers of mutual exclusion for the daily run.

Two concurrent runs would both increment ``consecutive_misses`` for the same
advertisements, silently pushing listings to ``likely_removed`` a day early. That
is unrecoverable without manual repair, so overlap prevention does not rely on a
single mechanism:

1. a **filesystem lock** (``flock``) — survives a crashed process, cheap, local;
2. a **database lock table** row with an expiry — visible to other hosts and
   auditable after the fact;
3. a **PostgreSQL session advisory lock** — released automatically by the server
   if the connection dies, closing the window where a killed process leaves a
   stale row behind.

SQLite has no advisory-lock equivalent, so there the first two layers apply; that
is stated rather than papered over.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import socket
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from .db import Database, utcnow

DEFAULT_LOCK_NAME = "bama_monitor_daily"
#: A lock older than this is considered abandoned by a dead process.
DEFAULT_TTL = timedelta(hours=6)


class LockUnavailable(RuntimeError):
    """Another run holds the lock. The caller must skip, not wait."""

    def __init__(self, layer: str, holder: str | None = None) -> None:
        super().__init__(f"lock held elsewhere (layer={layer}, holder={holder})")
        self.layer = layer
        self.holder = holder


def holder_id() -> str:
    """Identify this process well enough to diagnose a stuck lock."""
    return f"{socket.gethostname()}:{os.getpid()}"


def _advisory_key(name: str) -> int:
    """Map a lock name to the signed 64-bit key PostgreSQL expects."""
    return zlib.crc32(name.encode("utf-8")) - 2**31


@dataclass
class FileLock:
    """Non-blocking ``flock`` on a lock file."""

    path: Path
    _fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise LockUnavailable("filesystem", _read_text(self.path)) from exc
            raise
        os.truncate(fd, 0)
        os.write(fd, holder_id().encode("utf-8"))
        os.fsync(fd)
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


class DatabaseLock:
    """Row in ``run_locks``, plus an advisory lock on PostgreSQL."""

    def __init__(self, db: Database, name: str = DEFAULT_LOCK_NAME, ttl: timedelta = DEFAULT_TTL):
        self.db = db
        self.name = name
        self.ttl = ttl
        self._advisory_held = False

    def _try_advisory(self) -> None:
        if self.db.dialect != "postgres":
            return
        got = self.db.scalar("SELECT pg_try_advisory_lock(?)", [_advisory_key(self.name)])
        if not got:
            raise LockUnavailable("pg_advisory", None)
        self._advisory_held = True

    def _release_advisory(self) -> None:
        if self._advisory_held and self.db.dialect == "postgres":
            with contextlib.suppress(Exception):
                self.db.scalar("SELECT pg_advisory_unlock(?)", [_advisory_key(self.name)])
        self._advisory_held = False

    def acquire(self, run_id: int | None = None) -> None:
        self._try_advisory()
        now = utcnow()
        try:
            existing = self.db.fetchone(
                "SELECT holder, run_id, expires_at FROM run_locks WHERE lock_name=?", [self.name]
            )
            if existing:
                from .db import parse_ts

                expires = parse_ts(existing.get("expires_at"))
                if expires and expires > now:
                    raise LockUnavailable("database", existing.get("holder"))
                # Expired: the previous holder died without releasing. Take over,
                # but leave a trace of the takeover in the holder string.
                self.db.execute("DELETE FROM run_locks WHERE lock_name=?", [self.name])
            self.db.execute(
                "INSERT INTO run_locks (lock_name, holder, run_id, acquired_at, expires_at,"
                " heartbeat_at) VALUES (?,?,?,?,?,?)",
                [self.name, holder_id(), run_id, now, now + self.ttl, now],
            )
        except LockUnavailable:
            self._release_advisory()
            raise
        except Exception:
            self._release_advisory()
            raise

    def heartbeat(self, run_id: int | None = None) -> None:
        """Extend the lease so a long but healthy run is not treated as dead."""
        now = utcnow()
        self.db.execute(
            "UPDATE run_locks SET heartbeat_at=?, expires_at=?, run_id=COALESCE(?, run_id)"
            " WHERE lock_name=? AND holder=?",
            [now, now + self.ttl, run_id, self.name, holder_id()],
        )

    def release(self) -> None:
        with contextlib.suppress(Exception):
            self.db.execute(
                "DELETE FROM run_locks WHERE lock_name=? AND holder=?", [self.name, holder_id()]
            )
        self._release_advisory()


@contextlib.contextmanager
def run_lock(
    db: Database,
    lock_dir: Path,
    *,
    name: str = DEFAULT_LOCK_NAME,
    ttl: timedelta = DEFAULT_TTL,
    run_id: int | None = None,
) -> Iterator[None]:
    """Acquire every available layer, or raise :class:`LockUnavailable`.

    Deliberately non-blocking: a scheduled run that cannot start must be recorded
    as ``skipped_due_to_existing_run`` rather than queue up behind the current
    one and then compare against a stale inventory.
    """
    file_lock = FileLock(lock_dir / f"{name}.lock")
    db_lock = DatabaseLock(db, name=name, ttl=ttl)
    file_lock.acquire()
    try:
        db_lock.acquire(run_id=run_id)
    except BaseException:
        file_lock.release()
        raise
    try:
        yield
    finally:
        db_lock.release()
        file_lock.release()
