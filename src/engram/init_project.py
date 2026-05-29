"""``engram init`` — wire an existing project to use Engram.

The "should the agent use Engram in this project?" answer depends on two
signals (see README §"Per-project control"):

1. The skill / MCP server has to be **available** to the agent — that's
   what ``engram install`` sets up at the user level.
2. The project's instructions file (``CLAUDE.md`` / ``AGENTS.md``) has to
   **tell** the agent to use it.

Most users miss step 2.  The skill is installed, sitting at
``~/.claude/skills/engram/``, but Claude Code never voluntarily reads it
because nothing in the project mentions Engram.  ``engram init`` closes
that loop in a single command:

* Append the per-agent instructions snippet to ``<project>/CLAUDE.md``
  (or ``AGENTS.md`` for other agents) — but **only** if no Engram block
  is already present.  Idempotent.
* Add ``.claude/engram/`` to ``<project>/.gitignore`` — the local tier
  vector store should not be committed.  Idempotent.
* Create the empty ``<project>/.claude/engram/`` directory so the first
  save doesn't have to.

All file writes go through ``_atomic_write_text`` so a Ctrl+C mid-write
won't corrupt the user's existing CLAUDE.md.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .agents import AgentProfile, get_profile, list_agents
from .install import _atomic_write_text


# A small, stable marker string that lives inside both snippet templates
# (``_SKILL_SNIPPET`` and ``_MCP_SNIPPET``).  Used as a "has the user
# already wired this?" sentinel so re-running ``init`` is safe.
_SNIPPET_MARKER = "use Engram, not MD files"
_MCP_MARKER     = "use Engram via MCP"

_GITIGNORE_LINE = ".claude/engram/"


# ── Public entry point ──────────────────────────────────────────────────────

def init(
    *,
    agent: str = "claude-code",
    project_root: Optional[Path] = None,
    with_snippet: bool = True,
    with_gitignore: bool = True,
    force: bool = False,
) -> int:
    """Wire ``project_root`` (or CWD) to use Engram under ``agent``.

    Returns an exit code: 0 = OK, 1 = a step failed.  Prints a per-step
    summary to stdout (errors to stderr).  Idempotent — running ``init``
    twice does not duplicate the instructions snippet or .gitignore line.

    Parameters
    ----------
    agent
        Target agent (one of :func:`engram.agents.list_agents` or
        ``"all"``).  Defaults to ``"claude-code"``.
    project_root
        Project root to wire.  Defaults to the current working directory.
    with_snippet
        If ``False``, skip appending the instructions block.
    with_gitignore
        If ``False``, skip touching ``.gitignore``.
    force
        Re-append the snippet even if a marker is found.  Useful when
        the user has manually deleted the block and wants it back.
    """
    root = (Path(project_root) if project_root else Path.cwd()).resolve()
    if not root.exists():
        print(f"error: project root {root} does not exist", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"error: project root {root} is not a directory", file=sys.stderr)
        return 1

    if agent == "all":
        rc = 0
        for name in list_agents():
            print(f"── {name} ──")
            rc |= init(
                agent=name,
                project_root=root,
                with_snippet=with_snippet,
                with_gitignore=with_gitignore,
                force=force,
            )
            print()
        return rc

    profile = get_profile(agent)
    print(f"initialising {root} for {profile.display} ({profile.name})")

    rc = 0
    if with_snippet:
        rc |= _wire_instructions_file(root, profile, force=force)
    if with_gitignore:
        rc |= _wire_gitignore(root)
    rc |= _ensure_local_dir(root)
    if rc == 0:
        print("OK — project is now wired for Engram.")
    return rc


# ── Per-step helpers ────────────────────────────────────────────────────────

def _wire_instructions_file(
    root: Path,
    profile: AgentProfile,
    *,
    force: bool,
) -> int:
    """Append the agent's instructions snippet to the project file.

    Project files are named the same as the per-user one (e.g.
    ``CLAUDE.md`` / ``AGENTS.md``) — that's how Claude Code / OpenCode /
    Codex discover project-local instructions.
    """
    path = root / profile.instructions_file
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""

    if _has_engram_block(existing) and not force:
        print(
            f"  instructions: {path.name} already contains an Engram block "
            f"(skipped — pass --force to re-append)."
        )
        return 0

    snippet = profile.render_instructions_snippet().rstrip() + "\n"
    # Separate from existing prose with a single blank line; never add a
    # leading separator to an empty file.
    if existing and not existing.endswith("\n"):
        existing += "\n"
    sep = "\n" if existing else ""
    new_content = existing + sep + snippet
    try:
        _atomic_write_text(path, new_content)
    except OSError as e:
        print(f"  instructions: error writing {path}: {e}", file=sys.stderr)
        return 1
    action = "appended to" if existing else "created"
    print(f"  instructions: {action} {path}")
    return 0


def _wire_gitignore(root: Path) -> int:
    """Add ``.claude/engram/`` to ``.gitignore`` if not already there."""
    path = root / ".gitignore"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""

    if _gitignore_contains(existing, _GITIGNORE_LINE):
        print(f"  .gitignore : already ignores {_GITIGNORE_LINE} (skipped).")
        return 0

    # Preserve the operator's trailing-newline convention if any.
    if existing and not existing.endswith("\n"):
        existing += "\n"
    leading = "\n" if existing and not existing.endswith("\n\n") else ""
    addendum = leading + "# Engram local vector store — not source code\n" + _GITIGNORE_LINE + "\n"
    new_content = existing + addendum
    try:
        _atomic_write_text(path, new_content)
    except OSError as e:
        print(f"  .gitignore : error writing {path}: {e}", file=sys.stderr)
        return 1
    action = "appended" if existing else "created"
    print(f"  .gitignore : {action} {path} (+ {_GITIGNORE_LINE})")
    return 0


def _ensure_local_dir(root: Path) -> int:
    """Create ``<project>/.claude/engram/`` so the first save doesn't have to."""
    target = root / ".claude" / "engram"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"  local tier : error creating {target}: {e}", file=sys.stderr)
        return 1
    if any(target.iterdir()):
        print(f"  local tier : {target} already populated (skipped).")
    else:
        print(f"  local tier : created {target} (empty)")
    return 0


# ── Detection helpers ──────────────────────────────────────────────────────

def _has_engram_block(text: str) -> bool:
    """True if ``text`` already contains an Engram instructions block.

    We look for the markers that ship inside both the skill snippet
    (`use Engram, not MD files`) and the MCP snippet (`use Engram via
    MCP`).  Either one is enough — re-init will not double-append.
    """
    return _SNIPPET_MARKER in text or _MCP_MARKER in text


def _gitignore_contains(text: str, pattern: str) -> bool:
    """True if ``pattern`` is present as an active rule (non-comment, non-blank)
    in ``text``.

    Tolerates the operator using a leading slash (``/.claude/engram/``) or
    omitting the trailing slash — both ignore the same paths in git.
    """
    normalized = pattern.strip().rstrip("/").lstrip("/")
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip().rstrip("/").lstrip("/")
        if line == normalized:
            return True
    return False


__all__ = ["init"]
