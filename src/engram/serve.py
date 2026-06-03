"""Embedding-only warm daemon for Engram (``engram serve``).

Why this exists
---------------
Every ``UserPromptSubmit`` hook turn is a brand-new subprocess. With the
default local e5 backend each turn pays ~12 s of cold start (~9 s of which
is just ``import torch`` + ``sentence_transformers`` — unavoidable per
process). This daemon keeps **one** warm embedder alive between turns: the
first turn pays the ~12 s once, every later turn is a ~millisecond loopback
round-trip. This is what makes "always-on recall" viable under the default
e5 backend.

Design contract (see ``DAEMON_DESIGN.md``)
------------------------------------------
* **Embedding-only.** The daemon turns *text into vectors* and nothing
  else. It **never** touches ``.pst`` / ``.pcc``. The hook reads the store
  itself each turn (~0.2 s) and does KNN + gating locally. This sidesteps
  the whole cross-process lock / lost-update problem (``_lock.py``): the
  daemon is not a store writer, so there is no write to serialize.
* **Self-managed lifecycle (model A).** Lazily spawned by the hook when
  absent, self-exits after an idle window, and is re-spawned by the next
  hook if it died. No OS service, no watchdog, no boot autostart.
* **Default off.** Nothing here runs unless the user turned on
  「记忆长留」 via ``engram autorecall on`` (which installs the context-mode
  hook). No hook ⇒ no daemon ⇒ zero background process.

Transport / protocol
--------------------
Loopback TCP (``127.0.0.1`` + random high port) carrying **JSON Lines**
(one UTF-8 JSON object per ``\\n``-terminated line). Connection info is
published to ``serve.json`` (see :class:`ServeInfo`). Ops: ``embed`` /
``ping`` / ``shutdown`` (MVP) and ``stats`` (phase 2). All validated with a
random per-daemon token. Verified feasible on Windows by the spike
recorded in ``DAEMON_DESIGN.md`` §9b.

This module owns the daemon process and the wire protocol. The *client*
that the rest of Engram uses — the ``daemon:`` embedder backend — lives in
``embedder.py`` and imports the protocol helpers from here.

Entry points
------------
* ``engram serve`` → :func:`run_foreground` (debug; logs to stderr)
* ``engram serve --detach`` → :func:`spawn_detached`
* ``engram serve --stop`` → :func:`stop`
* ``engram serve --status`` → :func:`status`
The ``daemon:`` backend calls :func:`spawn_detached` on a cold miss.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._lock import FileLock
from .store import global_dir


# ── Protocol constants ──────────────────────────────────────────────────────

#: Wire-protocol version. Bump on any incompatible request/response change.
#: The client checks this against ``serve.json`` and forces a daemon
#: restart on mismatch (handles "pip upgraded engram under a running
#: daemon" — see DAEMON_DESIGN §7b.6).
PROTO_VERSION = 1

#: Idle self-exit window (seconds). Reset on every served request.
DEFAULT_IDLE_SECONDS = 1800  # 30 min (decision §9)

#: Loopback only — never bind a routable address.
HOST = "127.0.0.1"

#: Per-request safety caps: a single request may not blow up the daemon.
MAX_TEXTS_PER_REQUEST = 64
MAX_BYTES_PER_REQUEST = 256 * 1024

#: How long the accept loop blocks before re-checking the stop flag.
_ACCEPT_POLL_SECONDS = 1.0


# ── serve.json: discovery + identity ────────────────────────────────────────

def serve_json_path() -> Path:
    """Path to the connection-info file (``~/.claude/engram/serve.json``).

    Lives in :func:`store.global_dir` so it is ``ENGRAM_HOME``-aware and
    shared by every project (one daemon per machine, see DAEMON_DESIGN
    §7b.4)."""
    return global_dir() / "serve.json"


def serve_lock_path() -> Path:
    """Path to the single-instance lock (``serve.lock``).

    Held via :class:`engram._lock.FileLock` while a daemon is starting so
    two hooks racing to spawn don't create two daemons. Not the store
    lock — unrelated to ``.pst`` serialization."""
    return global_dir() / "serve.lock"


@dataclass
class ServeInfo:
    """Contents of ``serve.json`` — how a client finds and trusts a daemon.

    ``token`` gates write-ish ops (embed/stats/shutdown). ``proto`` +
    ``embedder`` let a client detect a stale/incompatible daemon and force
    a restart instead of talking to it.
    """
    pid: int
    port: int
    token: str
    embedder: str          # the RESOLVED underlying spec (never "daemon:...")
    dim: int
    started_at: int
    proto: int = PROTO_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "port": self.port,
            "token": self.token,
            "embedder": self.embedder,
            "dim": self.dim,
            "started_at": self.started_at,
            "proto": self.proto,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ServeInfo":
        return cls(
            pid=int(d["pid"]),
            port=int(d["port"]),
            token=str(d["token"]),
            embedder=str(d.get("embedder", "")),
            dim=int(d["dim"]),
            started_at=int(d.get("started_at", 0)),
            proto=int(d.get("proto", 0)),
        )

    def write_atomic(self, path: Optional[Path] = None) -> None:
        """Publish atomically: write ``<path>.tmp`` then ``os.replace``.

        (Spike-proven: avoids a reader seeing a half-written file.) Also
        chmod ``0600`` so only the owner can read the token (§7); on
        Windows chmod is a near-no-op but harmless."""
        path = path or serve_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        data = json.dumps(self.to_dict(), ensure_ascii=False)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)

    @classmethod
    def read(cls, path: Optional[Path] = None) -> Optional["ServeInfo"]:
        """Load ``serve.json`` or ``None`` if absent/corrupt. Never raises."""
        path = path or serve_json_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except (OSError, ValueError, KeyError):
            return None


# ── The daemon ──────────────────────────────────────────────────────────────

class EmbedDaemon:
    """A warm embedder behind a loopback TCP socket.

    Loads the underlying embedder **once** (the expensive step) and then
    answers ``embed`` requests until idle-timeout or ``shutdown``.

    Critical subtlety — ``embedder_spec`` must be the **resolved underlying**
    spec (e.g. ``"local"`` / ``"local?device=cuda"``), **never**
    ``"daemon:..."``. The daemon builds a real embedder via
    :func:`embedder.build_embedder`; passing ``daemon:`` would recurse into
    a client that tries to reach a daemon — itself. The launcher
    (:func:`spawn_detached`) is responsible for resolving ``daemon:auto`` /
    ``daemon:e5`` down to the real spec before starting the process.

    The daemon uses a **bare** embedder (no :class:`CachedEmbedder`): the
    on-disk cache lives next to ``.pst``, which the daemon must not touch,
    and per-turn queries are near-unique anyway.
    """

    def __init__(
        self,
        embedder_spec: str,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        host: str = HOST,
    ):
        self._spec = embedder_spec
        self._idle = float(idle_seconds)
        self._host = host
        self._embedder: Any = None          # built lazily in serve()
        self._dim: int = 0
        self._token: str = ""
        self._started_at: int = 0
        self._served: int = 0
        self._last_req: float = 0.0
        self._port: int = 0
        self._stop = threading.Event()
        # phase-2 state (query-mu / calibration), see _op_stats:
        self._mu: Optional[np.ndarray] = None
        self._mu_n: int = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def serve(self) -> None:
        """Run the daemon to completion (blocking). See class docstring.

        Order matters: build the embedder (the ~12 s cold start) and bind
        the socket *before* publishing ``serve.json`` so a client only ever
        discovers a daemon that can actually answer.
        """
        # The store dir may not exist yet (fresh ENGRAM_HOME): create it
        # before the lock, since FileLock's os.open(O_CREAT) can't make
        # parent dirs and would otherwise raise FileNotFoundError here.
        global_dir().mkdir(parents=True, exist_ok=True)

        # 1) single-instance: hold the lock across the check-and-publish
        #    window so two racing spawns can't both publish.
        try:
            lock = FileLock(str(serve_lock_path()), timeout=2.0).acquire()
        except TimeoutError:
            # Someone else is starting a daemon right now — let them win.
            return
        srv: Optional[socket.socket] = None
        try:
            # Another live daemon already published? Then yield.
            existing = ServeInfo.read()
            if existing is not None and _pid_alive(existing.pid):
                return

            # 2) cold start (expensive) — before we advertise ourselves.
            self._build_embedder()

            # 3) bind a random loopback port, then publish.
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self._host, 0))
            srv.listen(16)
            srv.settimeout(_ACCEPT_POLL_SECONDS)
            self._port = srv.getsockname()[1]
            self._publish(self._port)
        finally:
            # Startup window done: drop the lock so --stop/restart aren't
            # blocked for the daemon's whole life. serve.json + pid is the
            # liveness source of truth from here on.
            lock.release()

        if srv is None:                       # we yielded above
            return

        # 4) idle watchdog + accept loop.
        self._last_req = time.monotonic()
        wd = threading.Thread(target=self._idle_watchdog, daemon=True)
        wd.start()
        try:
            self._accept_loop(srv)
        finally:
            self._cleanup(srv)

    def _build_embedder(self) -> None:
        """Construct the bare underlying embedder from ``self._spec``.

        Never a ``daemon:`` spec (see class docstring) and never wrapped in
        :class:`CachedEmbedder` (the cache lives next to ``.pst``, which we
        must not touch)."""
        import secrets

        from .embedder import build_embedder  # local import: defer torch

        self._embedder = build_embedder(self._spec)
        self._dim = int(self._embedder.dim)
        self._token = secrets.token_hex(16)
        self._started_at = int(time.time())

    def _publish(self, port: int) -> None:
        """Write :class:`ServeInfo` to ``serve.json`` atomically (§4.2)."""
        ServeInfo(
            pid=os.getpid(),
            port=port,
            token=self._token,
            embedder=self._spec,
            dim=self._dim,
            started_at=self._started_at,
        ).write_atomic()

    def _accept_loop(self, srv: socket.socket) -> None:
        """Accept connections until the stop flag is set.

        ``srv`` has a short timeout so the loop periodically notices the
        stop flag (set by the idle watchdog or ``shutdown``). One request
        per connection."""
        while not self._stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    self._handle(conn)
                except Exception:
                    # A bad connection must never take the daemon down.
                    pass

    def _idle_watchdog(self) -> None:
        """If idle longer than ``self._idle``, set the stop flag and nudge
        the accept loop awake with a throwaway local connection (so it wakes
        from ``accept()`` immediately rather than after the poll), then
        return."""
        while not self._stop.is_set():
            time.sleep(min(self._idle / 4.0, 5.0) or 1.0)
            if time.monotonic() - self._last_req > self._idle:
                self._stop.set()
                try:
                    socket.create_connection((self._host, self._port), 0.2).close()
                except OSError:
                    pass
                return

    def _cleanup(self, srv: socket.socket) -> None:
        """Remove ``serve.json`` (only if it's *ours*, by pid) and close the
        socket. Idempotent; safe from ``finally``."""
        try:
            info = ServeInfo.read()
            if info is not None and info.pid == os.getpid():
                try:
                    serve_json_path().unlink()
                except OSError:
                    pass
        finally:
            try:
                srv.close()
            except OSError:
                pass

    # ── request dispatch ───────────────────────────────────────────────────

    def _handle(self, conn: socket.socket) -> None:
        """Read one ``\\n``-terminated JSON request, dispatch, write one JSON
        response line. Resets the idle timer. Enforces
        :data:`MAX_BYTES_PER_REQUEST`. A malformed request gets an error
        reply but never crashes the daemon."""
        self._last_req = time.monotonic()
        conn.settimeout(2.0)
        buf = b""
        try:
            while not buf.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > MAX_BYTES_PER_REQUEST:
                    self._send(conn, {"ok": False, "code": "TOO_LARGE",
                                      "error": "request exceeds size cap"})
                    return
        except socket.timeout:
            return
        if not buf:
            return
        try:
            req = json.loads(buf.decode("utf-8"))
            if not isinstance(req, dict):
                raise ValueError("request must be a JSON object")
        except (ValueError, UnicodeDecodeError) as e:
            self._send(conn, {"ok": False, "code": "BAD_JSON", "error": str(e)})
            return
        self._send(conn, self._dispatch(req))

    @staticmethod
    def _send(conn: socket.socket, resp: Dict[str, Any]) -> None:
        conn.sendall(json.dumps(resp, ensure_ascii=False).encode("utf-8") + b"\n")

    def _dispatch(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Route ``req["op"]`` to the matching ``_op_*``. Unknown op →
        ``BAD_OP``. ``ping`` is the only unauthenticated op."""
        op = req.get("op")
        if op == "ping":
            return self._op_ping(req)
        if op == "embed":
            return self._op_embed(req)
        if op == "stats":
            return self._op_stats(req)
        if op == "shutdown":
            return self._op_shutdown(req)
        return {"ok": False, "code": "BAD_OP", "error": f"unknown op {op!r}"}

    def _check_token(self, req: Dict[str, Any]) -> bool:
        return bool(self._token) and req.get("token") == self._token

    def _op_embed(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """**MVP.** ``{"op":"embed","token","kind","texts":[...]}`` →
        ``{"ok":True,"dim","vecs":[[...]]}``.

        Phase 2 will also fold each query vector into a rolling ``self._mu``
        and return ``mu``/``mu_n`` for the client's center-split gate — the
        response schema is forward compatible (extra keys), so adding it
        later needs no proto bump."""
        if not self._check_token(req):
            return {"ok": False, "code": "BAD_TOKEN", "error": "bad or missing token"}
        texts = req.get("texts")
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return {"ok": False, "code": "BAD_ARGS", "error": "texts must be a list[str]"}
        if len(texts) > MAX_TEXTS_PER_REQUEST:
            return {"ok": False, "code": "TOO_MANY",
                    "error": f"texts exceeds {MAX_TEXTS_PER_REQUEST}"}
        kind = req.get("kind", "query")
        if kind not in ("query", "passage"):
            return {"ok": False, "code": "BAD_ARGS", "error": "kind must be query|passage"}
        if not texts:
            return {"ok": True, "dim": self._dim, "vecs": []}
        vecs = self._embedder.embed_batch(texts, kind=kind)
        arr = np.ascontiguousarray(vecs, dtype=np.float32)
        self._served += len(texts)
        return {"ok": True, "dim": self._dim, "vecs": arr.tolist()}

    def _op_ping(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """**MVP.** Unauthenticated liveness probe. Used by ``--status``,
        doctor, and the client's freshness check."""
        return {
            "ok": True,
            "pid": os.getpid(),
            "dim": self._dim,
            "embedder": self._spec,
            "proto": PROTO_VERSION,
            "uptime_s": round(time.time() - self._started_at, 2) if self._started_at else 0.0,
            "served": self._served,
        }

    def _op_stats(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """**Phase 2.** Token-gated calibration stats. Stub for MVP: query-mu
        / threshold self-calibration land in plan §10 step 4."""
        if not self._check_token(req):
            return {"ok": False, "code": "BAD_TOKEN", "error": "bad or missing token"}
        return {"ok": False, "code": "NOT_READY",
                "error": "stats (query-mu / calibration) lands in phase 2"}

    def _op_shutdown(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """**MVP.** Token-gated graceful stop. Used by ``engram serve --stop``."""
        if not self._check_token(req):
            return {"ok": False, "code": "BAD_TOKEN", "error": "bad or missing token"}
        self._stop.set()
        return {"ok": True}


# ── Launch / control (module API used by CLI + the daemon: backend) ─────────

def resolve_underlying_spec(daemon_spec: Optional[str] = None) -> str:
    """Map a ``daemon:...`` spec (or ``None``/env) to the **real** embedder
    spec the daemon should load.

    This is the guard that prevents the daemon from recursing into itself
    (it must never load a ``daemon:`` spec — see :class:`EmbedDaemon`).

    Mapping by target:

    * ``daemon`` / ``daemon:auto`` / ``daemon:e5`` / ``daemon:local`` →
      ``"local"`` (the default warm e5 backend).
    * ``daemon:hash`` → ``"hash"`` (tests — exercise the plumbing without
      torch).
    * ``daemon:<other>`` → ``"<other>"`` (pin a different warm backend).
    * a spec that isn't a ``daemon`` spec at all (e.g. ``"hash"``,
      ``"local"``) is returned unchanged — convenient for callers/tests
      that already hold the resolved spec.
    """
    spec = (daemon_spec if daemon_spec is not None
            else os.environ.get("ENGRAM_EMBEDDER", "")).strip()
    if not spec:
        return "local"
    low = spec.lower()
    is_daemon = low == "daemon" or low.startswith("daemon:") or low.startswith("daemon?")
    if not is_daemon:
        return spec
    # Extract the target between "daemon:" and any "?params".
    target = ""
    if low.startswith("daemon:"):
        target = spec[len("daemon:"):].split("?", 1)[0].strip().lower()
    if target in ("", "auto", "e5", "local"):
        return "local"
    return target


def spawn_detached(
    embedder_spec: Optional[str] = None,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
) -> int:
    """Start a daemon that **outlives the caller** and return its pid.

    Re-execs ``python -m engram.serve --run --spec <resolved> --idle N``
    with the spike-proven platform flags:

    * Windows: ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
      CREATE_NO_WINDOW``, preferring ``pythonw.exe`` (no console window);
    * POSIX: ``start_new_session=True``;

    with **all** of stdin/stdout/stderr redirected to DEVNULL (a hook must
    never let the daemon inherit its stdout). Does **not** wait for
    warm-up: the caller degrades for this turn; the daemon warms in the
    background (§5.1 / §7b.4)."""
    spec = resolve_underlying_spec(embedder_spec)
    exe = _daemon_python()
    args = [exe, "-m", "engram.serve", "--run",
            "--spec", spec, "--idle", str(idle_seconds)]
    kwargs: Dict[str, Any] = dict(
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = (
            DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(args, **kwargs)
    return proc.pid


def _daemon_python() -> str:
    """The interpreter to launch the daemon with.

    On Windows prefer a sibling ``pythonw.exe`` (windowless) next to the
    current interpreter; fall back to ``sys.executable`` (the
    ``CREATE_NO_WINDOW`` flag still suppresses the console). On POSIX just
    use ``sys.executable``."""
    if sys.platform == "win32":
        cand = Path(sys.executable).with_name("pythonw.exe")
        if cand.is_file():
            return str(cand)
    return sys.executable


def run_foreground(
    embedder_spec: Optional[str] = None,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
) -> int:
    """Build and :meth:`EmbedDaemon.serve` in the current process (no
    detach). For ``engram serve`` and the ``--run`` re-exec target of
    :func:`spawn_detached`. Returns a process exit code."""
    spec = resolve_underlying_spec(embedder_spec)
    EmbedDaemon(spec, idle_seconds=idle_seconds).serve()
    return 0


def stop(timeout: float = 3.0) -> bool:
    """Stop a running daemon: read ``serve.json``, send token-authenticated
    ``shutdown``; if it doesn't die, terminate by pid; then clean up
    ``serve.json``. Returns ``True`` if a daemon was stopped."""
    info = ServeInfo.read()
    if info is None:
        return False
    stopped = False
    try:
        resp = send_request(info, {"op": "shutdown", "token": info.token}, timeout=timeout)
        stopped = bool(resp.get("ok"))
    except OSError:
        pass
    # Wait briefly for graceful exit; else hard-terminate.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(info.pid):
            stopped = True
            break
        time.sleep(0.1)
    if _pid_alive(info.pid):
        stopped = _terminate_pid(info.pid) or stopped
    # Clean up serve.json if it still points at the (now dead) daemon.
    cur = ServeInfo.read()
    if cur is not None and cur.pid == info.pid:
        try:
            serve_json_path().unlink()
        except OSError:
            pass
    return stopped


def status() -> Dict[str, Any]:
    """Probe the daemon for human/`doctor` output. Reads ``serve.json``,
    sends ``ping``, returns ``{"running","pid","port","embedder","dim",
    "uptime_s","served","stale"}``. ``stale`` is True when ``serve.json``
    exists but the pid is dead or ping fails. Never raises."""
    info = ServeInfo.read()
    if info is None:
        return {"running": False, "stale": False}
    if not _pid_alive(info.pid):
        return {"running": False, "stale": True, "pid": info.pid,
                "port": info.port, "embedder": info.embedder, "dim": info.dim}
    try:
        pong = send_request(info, {"op": "ping"}, timeout=2.0)
    except OSError:
        return {"running": False, "stale": True, "pid": info.pid,
                "port": info.port, "embedder": info.embedder, "dim": info.dim}
    return {
        "running": bool(pong.get("ok")),
        "stale": False,
        "pid": info.pid,
        "port": info.port,
        "embedder": info.embedder,
        "dim": info.dim,
        "uptime_s": pong.get("uptime_s"),
        "served": pong.get("served"),
        "proto": pong.get("proto"),
    }


# ── Wire helpers (shared with the daemon: backend in embedder.py) ───────────

def send_request(
    info: ServeInfo,
    obj: Dict[str, Any],
    timeout: float = 3.0,
) -> Dict[str, Any]:
    """Open a loopback connection to ``info``, send one JSON-Lines request,
    read one JSON-Lines reply, close. Raises :class:`OSError` on connection
    failure (the client decides whether that means "spawn + degrade"). Kept
    here so the protocol lives in one place; ``embedder.DaemonEmbedder``
    imports it."""
    conn = socket.create_connection((HOST, info.port), timeout)
    try:
        conn.sendall(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")
        conn.settimeout(timeout)
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8"))
    finally:
        conn.close()


def _pid_alive(pid: int) -> bool:
    """True if ``pid`` is a live process. ``tasklist`` on Windows, signal-0
    on POSIX (spike-proven)."""
    if pid is None or pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True,
            )
            return str(pid) in out.stdout
        except OSError:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True   # exists but not ours to signal
        except OSError:
            return False


def _terminate_pid(pid: int) -> bool:
    """Hard-terminate ``pid`` (best effort). Returns True if it appears dead
    afterwards."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True)
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    time.sleep(0.2)
    return not _pid_alive(pid)


# ── ``python -m engram.serve`` entry (re-exec target of spawn_detached) ─────

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse ``--run`` / ``--spec`` / ``--idle`` and call
    :func:`run_foreground`. This is the module the detached process
    re-execs into; the user-facing surface is ``engram serve`` /
    ``engram autorecall`` in ``cli.py``, which delegates here."""
    import argparse

    ap = argparse.ArgumentParser(prog="engram.serve", description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true",
                    help="run the daemon in this process (re-exec target)")
    ap.add_argument("--spec", default=None,
                    help="underlying embedder spec (resolved; never daemon:)")
    ap.add_argument("--idle", type=float, default=DEFAULT_IDLE_SECONDS,
                    help="idle self-exit window in seconds")
    args = ap.parse_args(argv)
    if not args.run:
        ap.error("nothing to do; use --run (or the `engram serve` CLI)")
    return run_foreground(args.spec, idle_seconds=args.idle)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
