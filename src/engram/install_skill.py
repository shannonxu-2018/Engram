"""Back-compat shim for v0.3's :mod:`engram.install_skill`.

v0.4 split installation into an agent-aware module (:mod:`engram.install`)
because Engram now supports Claude Code, OpenCode and Codex.  Callers
that imported ``install / remove / check / GLOBAL_SNIPPET`` from here
still work — they get the Claude Code path by default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .agents import get_profile
from .install import (
    DEFAULT_AGENT,
    GLOBAL_SNIPPET,
    check as _check,
    install as _install,
    remove as _remove,
)


def _default_target() -> Path:
    return get_profile(DEFAULT_AGENT).skill_dir  # type: ignore[return-value]


DEFAULT_TARGET = _default_target()


def install(
    target: Optional[str] = None,
    dev: bool = False,
    force: bool = False,
) -> int:
    """v0.3 API: install only the file-based Claude Code skill."""
    return _install(
        agent=DEFAULT_AGENT,
        target=target,
        dev=dev,
        force=force,
        with_mcp=False,
        with_skill=True,
    )


def remove(target: Optional[str] = None) -> int:
    return _remove(agent=DEFAULT_AGENT, target=target)


def check(target: Optional[str] = None) -> int:
    return _check(agent=DEFAULT_AGENT, target=target)


__all__ = [
    "DEFAULT_TARGET",
    "GLOBAL_SNIPPET",
    "install",
    "remove",
    "check",
]
