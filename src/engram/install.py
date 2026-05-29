"""Multi-agent installer for Engram.

This replaces the v0.3 :mod:`engram.install_skill` (which only knew about
Claude Code).  ``install_skill`` is kept as a thin alias for back-compat.

What "install" means per agent
------------------------------

Each agent has 0, 1, or 2 install steps depending on its
:class:`engram.agents.AgentProfile.install_kinds`:

* ``"skill"``  — copy / symlink ``_skill_files/`` into the agent's skill
  directory (Claude Code only).
* ``"mcp"``    — register the ``engram-mcp`` command in the agent's MCP
  config file (Claude Code, OpenCode, Codex).

In addition, ``--print-instructions-snippet`` emits the markdown block
the user can paste into their per-agent instructions file
(``CLAUDE.md`` / ``AGENTS.md``).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .agents import (
    AgentProfile,
    BUILTIN_PROFILES,
    get_profile,
    list_agents,
    render_mcp_servers_block,
)


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_AGENT = "claude-code"
MCP_SERVER_NAME = "engram"
MCP_COMMAND = "engram-mcp"
MCP_ARGS: List[str] = []


# ── Snippet (kept for back-compat with v0.3's --print-global-snippet) ────────

GLOBAL_SNIPPET = """\
# Memory: use Engram, not MD files

I have Engram (vector-DB memory traces) installed as a Claude Code skill at
`~/.claude/skills/engram/`. It replaces the legacy `MEMORY.md` /
per-file markdown memory under `~/.claude/projects/.../memory/`.

* **Never** write new files under `~/.claude/projects/.../memory/`. The
  vector store is the source of truth.
* **Never** read `MEMORY.md` for context. Use the skill scripts.

The full protocol (when to recall, what to save, body structure, the four
memory types) is at `~/.claude/skills/engram/SKILL.md`.

## Quick reference

* **Recall**: `python ~/.claude/skills/engram/scripts/recall.py "<query>" --k 5`
* **Save**:   `python ~/.claude/skills/engram/scripts/save.py <type> <name> "<desc>" "<content>"`
* **Expand**: `python ~/.claude/skills/engram/scripts/expand.py <id>`
* **List**:   `python ~/.claude/skills/engram/scripts/list.py`
* **Forget**: `python ~/.claude/skills/engram/scripts/forget.py --name <slug>`

Types: `user`, `feedback`, `project`, `reference`. If `save` exits with
`MERGE_SUGGESTION`, decide whether to `--update` or `--force`.
"""


# ── Source-tree discovery ────────────────────────────────────────────────────

def _source_dir() -> Path:
    src = Path(__file__).resolve().parent / "_skill_files"
    if not src.is_dir():
        raise RuntimeError(
            f"Bundled skill files not found at {src}. This usually means "
            "engram was installed without package_data — try reinstalling: "
            "pip install --force-reinstall engram"
        )
    return src


def _remove_existing(dst: Path) -> None:
    if dst.is_symlink():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    elif dst.exists():
        dst.unlink()


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically.

    The naive ``path.write_text`` / ``open(path, "w")`` pattern truncates
    the target first — if the process dies (Ctrl+C, OOM, power loss)
    between the truncate and the final byte, the user is left with a
    half-written ``.claude.json`` / ``config.toml``.  Those files often
    contain unrelated user settings (OAuth tokens, permissions, model
    pins) so a half-write is *worse* than not running at all.

    Pattern: write the new bytes to a sibling tempfile, fsync, then
    ``os.replace()`` it over the target — that last step is atomic on
    both POSIX and Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Some filesystems / streams don't support fsync; the rename
            # is still atomic, just not durable across power loss.  Don't
            # let that fail the install.
            pass
    os.replace(tmp, path)


def _strip_codex_section(text: str, server_name: str) -> str:
    """Return ``text`` with any ``[mcp_servers.<server_name>]`` block removed.

    The block starts at the matching header line and runs until the next
    blank line or the next ``[…]`` table header.  Matching is whitespace-
    tolerant: ``[mcp_servers.engram]``, ``[ mcp_servers.engram ]`` and a
    trailing-comment variant all count.
    """
    header_re = f"[mcp_servers.{server_name}]"
    out_lines: List[str] = []
    dropping = False
    for line in text.splitlines():
        stripped = line.strip()
        if dropping:
            if not stripped:
                dropping = False
                continue
            if stripped.startswith("["):
                dropping = False
                # fall through: keep this line — it's the next section
            else:
                continue
        # Whitespace-tolerant match: strip the header itself before compare.
        if stripped.replace(" ", "") == header_re.replace(" ", ""):
            dropping = True
            continue
        out_lines.append(line)
    return "\n".join(out_lines).rstrip()


# ── Per-step: skill files ────────────────────────────────────────────────────

def _install_skill_files(
    profile: AgentProfile,
    target_override: Optional[str],
    dev: bool,
    force: bool,
) -> int:
    if "skill" not in profile.install_kinds:
        print(f"  skill: skipped (agent {profile.name!r} has no file-based skill)")
        return 0

    if target_override is not None:
        if not target_override.strip():
            # ``Path("").expanduser()`` quietly resolves to the CWD — the
            # user almost certainly didn't mean "install the skill into
            # whatever directory I happen to be in".  Reject loud.
            print(
                "  skill: error: --target was empty; pass a real directory path.",
                file=sys.stderr,
            )
            return 1
        dst: Optional[Path] = Path(target_override).expanduser().resolve()
    else:
        dst = profile.skill_dir
    if dst is None:
        print(f"  skill: profile has no skill_dir; nothing to install", file=sys.stderr)
        return 1
    src = _source_dir()

    if (dst.exists() or dst.is_symlink()) and not force:
        print(
            f"  skill: error: {dst} already exists.\n"
            "         Re-run with --force to overwrite, or --remove to delete first.",
            file=sys.stderr,
        )
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)

    mode = "symlink" if dev else "copy"
    print(f"  skill: {mode} {src} -> {dst}")

    if dev:
        # Probe symlink support *before* removing the existing install.
        # Earlier versions removed dst first, then failed the symlink call
        # on Windows without Developer Mode — leaving the user with no
        # skill at all and an error message.  Probe with a sibling.
        probe = dst.parent / (dst.name + ".symlink-probe")
        try:
            if probe.exists() or probe.is_symlink():
                probe.unlink()
            probe.symlink_to(src, target_is_directory=True)
        except OSError as e:
            print(
                f"  skill: error: could not create symlink at {dst}\n"
                f"         {e}\n"
                "         On Windows: enable Developer Mode "
                "(Settings -> Privacy & security -> For developers),\n"
                "         run as Admin, or omit --dev to use copy mode.\n"
                "         (Existing install at {dst} was left intact.)",
                file=sys.stderr,
            )
            return 1
        # Probe succeeded — safe to swap.  Remove the probe, remove the
        # old install, then symlink-to-final-location.
        try:
            probe.unlink()
        except OSError:
            pass
        _remove_existing(dst)
        try:
            dst.symlink_to(src, target_is_directory=True)
        except OSError as e:
            # Extraordinarily unlikely given the probe succeeded, but the
            # filesystem could have changed between probe and final write
            # (e.g. perms revoked).  Surface honestly.
            print(
                f"  skill: error: symlink failed after probe succeeded: {e}",
                file=sys.stderr,
            )
            return 1
    else:
        # Copy mode: stage into a sibling tempdir, then atomically swap.
        # If copytree dies halfway, the old install is still intact.
        staging = dst.parent / (dst.name + ".staging")
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(src, staging)
        _remove_existing(dst)
        os.replace(staging, dst)
    return 0


# ── Per-step: MCP registration ───────────────────────────────────────────────

def _merge_claude_servers(path: Path, entry: Dict[str, object]) -> None:
    """Merge an ``{name: {...}}`` entry under the top-level ``mcpServers`` key.

    Used for Claude Code's ``~/.claude.json`` and OpenCode's
    ``opencode.json`` — both follow the standard MCP-config shape.

    Atomic — see :func:`_atomic_write_text`.  Crash-safe even on power loss.
    """
    data: Any = {}
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"  mcp: refusing to overwrite malformed JSON at {path}. "
                f"Inspect/repair it manually, then re-run."
            ) from e
    # The file's root *should* be a JSON object — but a hand-edited or
    # corrupted file might have ``null``, a list, or a number.  Bail
    # rather than wipe their data; the operator can decide what to do.
    if not isinstance(data, dict):
        raise RuntimeError(
            f"  mcp: {path} root is {type(data).__name__}, expected object; "
            f"refusing to overwrite."
        )
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise RuntimeError(
            f"  mcp: {path} has a non-object 'mcpServers' field; aborting."
        )
    servers[MCP_SERVER_NAME] = entry  # idempotent: overwrites our prior entry
    _atomic_write_text(path, json.dumps(data, indent=2) + "\n")


def _merge_codex_toml(path: Path, server_name: str, command: str, args: List[str]) -> None:
    """Insert/replace ``[mcp_servers.<server_name>]`` in Codex's TOML config.

    We do a minimal text-level merge (no toml dependency) — if the section
    exists we strip it and re-append; if the file's absent we create it.
    Codex's ``config.toml`` is small (a handful of sections) so this is
    safe.  Atomic — see :func:`_atomic_write_text`.
    """
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    cleaned = _strip_codex_section(existing, server_name)

    args_repr = ", ".join(json.dumps(a) for a in args)
    new_section = (
        f"[mcp_servers.{server_name}]\n"
        f"command = {json.dumps(command)}\n"
        f"args = [{args_repr}]\n"
    )
    sep = "\n\n" if cleaned else ""
    _atomic_write_text(path, cleaned + sep + new_section)


def _install_mcp_registration(profile: AgentProfile) -> int:
    if "mcp" not in profile.install_kinds:
        print(f"  mcp:   skipped (agent {profile.name!r} has no MCP integration)")
        return 0
    path = profile.mcp_config_path
    if path is None:
        print(f"  mcp:   profile has no mcp_config_file; nothing to register",
              file=sys.stderr)
        return 1

    if profile.mcp_config_kind == "claude_servers":
        entry = profile.render_mcp_entry(MCP_COMMAND, MCP_ARGS)
        _merge_claude_servers(path, entry)
        print(f"  mcp:   registered '{MCP_SERVER_NAME}' in {path}")
        return 0
    if profile.mcp_config_kind == "codex_toml":
        _merge_codex_toml(path, MCP_SERVER_NAME, MCP_COMMAND, MCP_ARGS)
        print(f"  mcp:   registered '{MCP_SERVER_NAME}' in {path}")
        return 0
    print(f"  mcp:   unknown config kind {profile.mcp_config_kind!r}", file=sys.stderr)
    return 1


# ── Public entry points ──────────────────────────────────────────────────────

def install(
    agent: str = DEFAULT_AGENT,
    target: Optional[str] = None,
    dev: bool = False,
    force: bool = False,
    with_mcp: Optional[bool] = None,
    with_skill: Optional[bool] = None,
) -> int:
    """Install Engram for ``agent``.

    ``with_skill`` / ``with_mcp`` default to "whatever the profile
    supports".  Pass an explicit ``False`` to suppress one half.
    """
    profile = get_profile(agent)
    do_skill = "skill" in profile.install_kinds if with_skill is None else with_skill
    do_mcp   = "mcp"   in profile.install_kinds if with_mcp   is None else with_mcp

    print(f"installing engram for {profile.display} ({profile.name})")
    print(f"  home  : {profile.home_dir}")

    rc = 0
    if do_skill:
        rc |= _install_skill_files(profile, target, dev, force)
    if do_mcp:
        rc |= _install_mcp_registration(profile)

    if rc == 0:
        print("OK")
        if do_skill and profile.skill_dir:
            print(f"  verify skill: python \"{profile.skill_dir / 'scripts' / 'list.py'}\"")
        if do_mcp:
            print(f"  verify mcp  : engram-mcp --help")
        print(
            f"  next: append the snippet to {profile.instructions_path} "
            f"with `engram install --agent {profile.name} --print-instructions-snippet`"
        )
    return rc


def install_all(
    dev: bool = False,
    force: bool = False,
) -> int:
    """Install Engram for every built-in agent in turn.

    Each agent is best-effort: failures (e.g. an agent's home dir is on
    a filesystem we can't write to) propagate into the return code via
    bitwise OR, but don't stop the loop — so a single broken target
    won't block the rest.
    """
    rc = 0
    for name in list_agents():
        rc |= install(agent=name, dev=dev, force=force)
        print()
    return rc


def remove(agent: str = DEFAULT_AGENT, target: Optional[str] = None) -> int:
    profile = get_profile(agent)
    rc = 0
    if "skill" in profile.install_kinds:
        dst = Path(target).expanduser().resolve() if target else profile.skill_dir
        if dst and (dst.exists() or dst.is_symlink()):
            if dst.is_symlink():
                dst.unlink()
                print(f"removed symlink {dst}")
            else:
                shutil.rmtree(dst)
                print(f"removed directory {dst}")
        elif dst:
            print(f"nothing to remove at {dst}")

    if "mcp" in profile.install_kinds and profile.mcp_config_path is not None:
        path = profile.mcp_config_path
        if path.is_file():
            if profile.mcp_config_kind == "claude_servers":
                try:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    print(f"warning: could not parse {path}; skipped", file=sys.stderr)
                    rc = 1
                    data = None
                if isinstance(data, dict):
                    servers = data.get("mcpServers", {})
                    if isinstance(servers, dict) and MCP_SERVER_NAME in servers:
                        servers.pop(MCP_SERVER_NAME, None)
                        _atomic_write_text(
                            path, json.dumps(data, indent=2) + "\n"
                        )
                        print(f"unregistered '{MCP_SERVER_NAME}' from {path}")
            elif profile.mcp_config_kind == "codex_toml":
                # Single pass: read → strip → write atomically.  The old
                # code round-tripped through ``_merge_codex_toml`` first
                # (which *re*-wrote the engram section) and then stripped
                # it — two disk writes for a delete, with a window where
                # a crash would leave the server section re-installed.
                text = path.read_text(encoding="utf-8")
                stripped = _strip_codex_section(text, MCP_SERVER_NAME)
                # If nothing changed, don't touch the file (preserves mtime).
                if stripped.rstrip() != text.rstrip():
                    suffix = "\n" if stripped else ""
                    _atomic_write_text(path, stripped + suffix)
                    print(f"unregistered '{MCP_SERVER_NAME}' from {path}")
    return rc


def check(agent: str = DEFAULT_AGENT, target: Optional[str] = None) -> int:
    profile = get_profile(agent)
    src = _source_dir()
    print(f"agent          : {profile.display} ({profile.name})")
    print(f"home           : {profile.home_dir}")
    print(f"instructions   : {profile.instructions_path}")
    print(f"install kinds  : {', '.join(profile.install_kinds)}")
    print(f"source         : {src}")

    rc = 0
    if "skill" in profile.install_kinds:
        dst = Path(target).expanduser().resolve() if target else profile.skill_dir
        print(f"skill target   : {dst}")
        if dst is None or (not dst.exists() and not dst.is_symlink()):
            print("skill status   : NOT INSTALLED")
            rc = 1
        elif dst.is_symlink():
            real = dst.resolve()
            print(f"skill status   : SYMLINK -> {real}")
        else:
            print("skill status   : COPY")

    if "mcp" in profile.install_kinds:
        path = profile.mcp_config_path
        registered = False
        if path is not None and path.is_file():
            if profile.mcp_config_kind == "claude_servers":
                try:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict):
                    servers = data.get("mcpServers")
                    if isinstance(servers, dict):
                        registered = MCP_SERVER_NAME in servers
            elif profile.mcp_config_kind == "codex_toml":
                registered = f"[mcp_servers.{MCP_SERVER_NAME}]" in path.read_text(
                    encoding="utf-8", errors="replace"
                )
        print(f"mcp config     : {path}")
        print(f"mcp status     : {'REGISTERED' if registered else 'NOT REGISTERED'}")
        if not registered:
            rc = 1
    return rc


def print_instructions_snippet(
    agent: str = DEFAULT_AGENT,
    kind: Optional[str] = None,
) -> int:
    """Print the instructions block for ``agent``.

    ``kind`` selects ``"skill"`` or ``"mcp"`` framing.  Defaults to the
    profile's primary integration kind.
    """
    profile = get_profile(agent)
    print(profile.render_instructions_snippet(kind=kind), end="")
    return 0


__all__ = [
    "DEFAULT_AGENT",
    "GLOBAL_SNIPPET",
    "MCP_SERVER_NAME",
    "MCP_COMMAND",
    "MCP_ARGS",
    "install",
    "install_all",
    "remove",
    "check",
    "print_instructions_snippet",
]
