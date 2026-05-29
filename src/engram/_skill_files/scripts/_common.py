"""Shared CLI helpers for the Engram skill scripts.

`bootstrap()` makes the ``engram`` package importable in three deployment
scenarios:

1. **Installed** (``pip install -e <repo>`` or a built wheel).  ``import
   engram`` already works — bootstrap is a no-op.
2. **In-repo dev mode** (scripts live at ``<repo>/.claude/skills/engram/
   scripts/`` and the package lives at ``<repo>/src/engram/``).  We walk up
   from this file until we find a directory that contains ``engram/
   __init__.py``, then prepend it to ``sys.path``.
3. **Override** via ``ENGRAM_REPO`` env var pointing at the repo root
   (or to ``<repo>/src``).  First place we look — useful when the skill
   has been copied to ``~/.claude/skills/`` but engram isn't installed
   (e.g. side-by-side with the repo on a non-pip machine).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional


def _has_engram(d: Path) -> bool:
    return (d / "engram" / "__init__.py").is_file()


def _find_repo_upwards(start: Path, max_levels: int = 8) -> Optional[Path]:
    """Walk up from ``start`` looking for a dir with ``engram/__init__.py``.

    With the v0.3 src/-layout this resolves to ``<repo>/src`` (which
    contains ``engram/__init__.py``); pre-v0.3 layouts that kept the
    package at the repo root also work.
    """
    cur = start.resolve()
    for _ in range(max_levels):
        if _has_engram(cur):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def bootstrap() -> Optional[Path]:
    """Ensure ``engram`` is importable. Returns the directory added to
    ``sys.path`` if one was located, or ``None`` if engram was already
    importable (installed)."""
    # 1) Already importable? Nothing to do.
    if importlib.util.find_spec("engram") is not None:
        return None

    # 2) Explicit override.
    env_repo = os.environ.get("ENGRAM_REPO")
    if env_repo:
        repo = Path(env_repo).expanduser().resolve()
        # Accept either the repo root (with src/engram inside) or the
        # src/ dir directly.
        candidates = [repo, repo / "src"]
        for c in candidates:
            if _has_engram(c):
                sys.path.insert(0, str(c))
                return c
        raise RuntimeError(
            f"ENGRAM_REPO={env_repo!r} does not contain engram/__init__.py "
            f"(checked {candidates})"
        )

    # 3) Walk up from this script.
    repo = _find_repo_upwards(Path(__file__).parent)
    if repo is not None:
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        return repo

    raise RuntimeError(
        "Cannot locate the engram package.\n"
        "  Tried: `import engram`, $ENGRAM_REPO, and walking up from "
        f"{Path(__file__).parent}.\n"
        "  Fix one of:\n"
        "    (a) pip install -e <path-to-memskill-repo>\n"
        "    (b) set ENGRAM_REPO=<path-to-memskill-repo>\n"
        "    (c) place the skill back inside the repo at "
        "<repo>/.claude/skills/engram/scripts/"
    )


def emit_text(hits, with_content: bool = False) -> None:
    """Print compact text — one line per hit, easy to feed back to the LLM."""
    if not hits:
        print("(no hits)")
        return
    for h in hits:
        # description SHOULD be single-line per save-time policy, but a
        # migrated MD memory or hand-edit could smuggle in newlines that
        # would break the column layout — strip them defensively here.
        desc = h.description.replace("\n", " ").replace("\r", " ")
        line = f"{h.id:>5} | {h.tier[:1]} | {h.type:<9} | {h.name:<28} | d={h.distance:.3f} | {desc}"
        print(line)
        if with_content and h.content:
            for ln in h.content.splitlines():
                print(f"      | {ln}")


def emit_json(payload) -> None:
    import json
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
