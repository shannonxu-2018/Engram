"""``engram uninstall`` — one-click teardown, by scope.

Engram leaves traces in two very different places, and a user who wants
"just remove it" almost never means *both*:

* **project** scope — the reverse of ``engram init``: strip the Engram
  instructions block from the project's ``CLAUDE.md`` / ``AGENTS.md``,
  drop the ``.claude/engram/`` line from ``.gitignore``, and (optionally)
  delete the project-local vector store under ``<project>/.claude/engram/``.
* **global** scope — the reverse of ``engram install`` (+ the setup
  wizard's global-enable step): remove the file-based skill, unregister
  the ``engram`` MCP server, strip the Engram block from each agent's
  *user-level* instructions file, and (optionally) delete the global
  vector store under ``~/.claude/engram/``.

The interactive :func:`run` asks which scope to tear down and whether to
also delete stored memories (off by default — memory deletion is the one
step that can't be undone).  Every individual action is idempotent and
reuses the same helpers as ``install`` / ``init`` so a half-applied
uninstall can simply be re-run.

All file rewrites go through :func:`engram.install._atomic_write_text`,
so a Ctrl+C mid-write won't corrupt an unrelated ``CLAUDE.md``.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .agents import get_profile, iter_profiles, list_agents
from .install import _atomic_write_text, remove as _remove_install
from .init_project import (
    _GITIGNORE_LINE,
    _MCP_MARKER,
    _SNIPPET_MARKER,
    _gitignore_contains,
)
from .setup_wizard import _ask_yes_no, _interactive

_GITIGNORE_COMMENT = "# Engram local vector store — not source code"


# ── Instructions-file teardown ───────────────────────────────────────────────

def _is_level1_heading(line: str) -> bool:
    """True for an ATX level-1 heading (``# title``) — not ``##``/``###``.

    ``"## foo".startswith("# ")`` is already ``False`` (it starts with
    ``"##"``), so this also correctly leaves the Engram block's own
    ``## Quick reference`` / ``## When to use`` subheadings *inside* the
    block instead of treating them as the block's end.
    """
    return line.startswith("# ")


def _strip_engram_block(text: str) -> Tuple[str, bool]:
    """Return ``text`` with the Engram instructions block removed.

    The block is the markdown section whose level-1 heading carries one of
    the snippet markers (``use Engram, not MD files`` / ``use Engram via
    MCP``).  It runs from that heading up to — but not including — the next
    level-1 heading, or end-of-file.  Blank-line padding around the removed
    region is collapsed so we don't leave a double gap.

    Returns ``(new_text, found)``.  When no block is present the original
    text is returned unchanged with ``found=False``.
    """
    lines = text.splitlines()
    start: Optional[int] = None
    for i, line in enumerate(lines):
        if _is_level1_heading(line) and (_SNIPPET_MARKER in line or _MCP_MARKER in line):
            start = i
            break
    if start is None:
        return text, False

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _is_level1_heading(lines[j]):
            end = j
            break

    before = lines[:start]
    after = lines[end:]
    while before and before[-1].strip() == "":
        before.pop()
    while after and after[0].strip() == "":
        after.pop(0)

    if before and after:
        new_lines = before + [""] + after
    else:
        new_lines = before + after

    result = "\n".join(new_lines)
    if result:
        result += "\n"
    return result, True


def _unwire_instructions_file(path: Path) -> int:
    """Remove the Engram block from ``path`` (project or user-level).

    A file that ends up empty after the strip is left as a zero-byte file
    rather than deleted — we can't know the user didn't create it solely
    for Engram, and an empty ``CLAUDE.md`` is harmless and unsurprising.
    """
    if not path.is_file():
        print(f"  instructions: {path} not found (skipped).")
        return 0
    existing = path.read_text(encoding="utf-8")
    new_text, found = _strip_engram_block(existing)
    if not found:
        print(f"  instructions: no Engram block in {path} (skipped).")
        return 0
    try:
        _atomic_write_text(path, new_text)
    except OSError as e:
        print(f"  instructions: error writing {path}: {e}", file=sys.stderr)
        return 1
    print(f"  instructions: removed Engram block from {path}")
    return 0


def _unwire_gitignore(root: Path) -> int:
    """Drop the ``.claude/engram/`` rule (and our comment) from ``.gitignore``."""
    path = root / ".gitignore"
    if not path.is_file():
        print(f"  .gitignore : {path} not found (skipped).")
        return 0
    existing = path.read_text(encoding="utf-8")
    if not _gitignore_contains(existing, _GITIGNORE_LINE):
        print(f"  .gitignore : no {_GITIGNORE_LINE} rule (skipped).")
        return 0

    normalized = _GITIGNORE_LINE.strip().rstrip("/").lstrip("/")
    out: List[str] = []
    for raw in existing.splitlines():
        code = raw.split("#", 1)[0].strip().rstrip("/").lstrip("/")
        if code == normalized:
            continue
        if raw.strip() == _GITIGNORE_COMMENT:
            continue
        out.append(raw)
    while out and out[-1].strip() == "":
        out.pop()
    new_text = "\n".join(out)
    if new_text:
        new_text += "\n"
    try:
        _atomic_write_text(path, new_text)
    except OSError as e:
        print(f"  .gitignore : error writing {path}: {e}", file=sys.stderr)
        return 1
    print(f"  .gitignore : removed {_GITIGNORE_LINE} from {path}")
    return 0


# ── Data purge ────────────────────────────────────────────────────────────────

def _purge_dir(target: Path, label: str) -> int:
    """Delete ``target`` recursively if it exists.  Irreversible."""
    if not target.exists():
        print(f"  {label}: {target} not found (skipped).")
        return 0
    try:
        shutil.rmtree(target)
    except OSError as e:
        print(f"  {label}: error deleting {target}: {e}", file=sys.stderr)
        return 1
    print(f"  {label}: deleted {target}")
    return 0


# ── Scope: project ────────────────────────────────────────────────────────────

def uninstall_project(
    *,
    agent: str = "claude-code",
    project_root: Optional[Path] = None,
    purge_data: bool = False,
) -> int:
    """Reverse ``engram init`` for ``project_root`` (or CWD).

    With ``agent="all"`` we strip every agent's instructions file (some
    share ``AGENTS.md`` — the second strip is a harmless no-op).  The
    ``.gitignore`` rule and the optional data purge run once regardless.
    """
    root = (Path(project_root) if project_root else Path.cwd()).resolve()
    if not root.is_dir():
        print(f"error: project root {root} is not a directory", file=sys.stderr)
        return 1

    print(f"uninstalling Engram from project {root}")
    agents = list_agents() if agent == "all" else [agent]

    rc = 0
    seen_files: set = set()
    for name in agents:
        profile = get_profile(name)
        path = root / profile.instructions_file
        if path in seen_files:
            continue
        seen_files.add(path)
        rc |= _unwire_instructions_file(path)

    rc |= _unwire_gitignore(root)

    if purge_data:
        from .store import local_dir
        rc |= _purge_dir(local_dir(root), "local store")
    else:
        from .store import local_dir
        print(
            f"  local store: kept {local_dir(root)} "
            "(pass --purge to delete stored memories)."
        )
    return rc


# ── Scope: global ─────────────────────────────────────────────────────────────

def uninstall_global(
    *,
    agent: str = "all",
    purge_data: bool = False,
) -> int:
    """Reverse ``engram install`` (skill + MCP) and the global-enable step.

    For each selected agent: remove the skill dir, unregister the MCP
    server (both via :func:`engram.install.remove`), and strip the Engram
    block from the agent's *user-level* instructions file.  Optionally
    delete the shared global vector store.
    """
    print("uninstalling Engram globally")
    agents = list_agents() if agent == "all" else [agent]

    rc = 0
    for name in agents:
        profile = get_profile(name)
        print(f"── {profile.display} ({name}) ──")
        rc |= _remove_install(agent=name)
        rc |= _unwire_instructions_file(profile.instructions_path)
        print()

    if purge_data:
        from .store import global_dir
        rc |= _purge_dir(global_dir(), "global store")
    else:
        from .store import global_dir
        print(
            f"global store: kept {global_dir()} "
            "(pass --purge to delete stored memories)."
        )
    return rc


# ── Interactive driver ────────────────────────────────────────────────────────

def _ask_scope(assume_yes: bool) -> Optional[str]:
    """Ask which scope to tear down.  Returns 'project' | 'global' | 'all'."""
    default = "project"
    if assume_yes or not _interactive():
        print(f"Which Engram install should I remove? > {default}")
        return default

    print("Which Engram install should I remove?")
    print("  1. project  — this directory only (reverse of `engram init`)")
    print("  2. global   — the skill + MCP + per-agent wiring (reverse of `engram install`)")
    print("  3. both")
    print(f"(press Enter for the default: {default})")
    mapping = {"1": "project", "2": "global", "3": "all",
               "project": "project", "global": "global", "both": "all", "all": "all"}
    while True:
        try:
            raw = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw == "":
            return default
        if raw in mapping:
            return mapping[raw]
        print("  please answer 1, 2, or 3.")


def run(
    *,
    scope: Optional[str] = None,
    agent: Optional[str] = None,
    project_root: Optional[Path] = None,
    purge: Optional[bool] = None,
    assume_yes: bool = False,
) -> int:
    """One-click uninstall.

    ``scope`` — ``"project"`` / ``"global"`` / ``"all"``; ``None`` asks.
    ``agent`` — narrow to one agent; ``None`` means *all* agents (the
    thorough default for a clean teardown).
    ``purge`` — delete stored memories too; ``None`` asks (default **no**,
    because memory deletion is the only irreversible step).
    """
    print("=" * 60)
    print(" Engram uninstall")
    print("=" * 60)

    if scope is None:
        if not assume_yes and not _interactive():
            print(
                "stdin is not a terminal, so I can't ask which scope to remove.\n"
                "Re-run with --scope {project,global,all} (and --yes for non-interactive):\n"
                "    engram uninstall --scope project --yes",
                file=sys.stderr,
            )
            return 2
        scope = _ask_scope(assume_yes)
        if scope is None:
            print("aborted.")
            return 1

    # Resolve the data-purge decision once, up front, so the irreversible
    # question is asked before we start changing anything.
    if purge is None:
        purge = _ask_yes_no(
            "Also DELETE stored memories? This cannot be undone",
            default=False,
            assume_yes=assume_yes,
        )

    agent_sel = agent or "all"

    rc = 0
    if scope in ("project", "all"):
        print()
        rc |= uninstall_project(
            agent=agent_sel, project_root=project_root, purge_data=purge
        )
    if scope in ("global", "all"):
        print()
        rc |= uninstall_global(agent=agent_sel, purge_data=purge)

    print()
    if rc == 0:
        print("Engram uninstalled.")
        if not purge:
            print("(Stored memories were kept — re-run with --purge to delete them.)")
    else:
        print("Uninstall finished with some warnings above.")
    return rc


__all__ = [
    "run",
    "uninstall_project",
    "uninstall_global",
]
