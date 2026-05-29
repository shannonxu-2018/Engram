"""Agent profiles — Engram's multi-agent abstraction.

Engram started life as a Claude Code skill (file-based ``SKILL.md`` +
scripts under ``~/.claude/skills/engram/``).  Two other coding agents
have since converged on a different convention:

* **OpenCode** (sst/opencode) — reads ``AGENTS.md`` for instructions and
  talks to tools through **MCP**.  No file-based skill protocol.
* **Codex** (openai/codex) — same story: ``AGENTS.md`` + MCP.

So the integration matrix collapses to two axes:

1. *Does the agent support a file-based skill?*  Only Claude Code does.
2. *What's its agent-instructions filename and home dir?*

This module captures both axes as :class:`AgentProfile` records.  The
rest of Engram (``store.py``, ``install.py``, ``mcp_server.py``,
``cli.py``) is profile-driven so adding a new agent later is one entry
in :data:`BUILTIN_PROFILES`.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Profile ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentProfile:
    """One target agent's integration knobs.

    Attributes:
        name:               Stable id used on the CLI (``--agent NAME``).
        display:            Human-readable name shown in CLI output.
        home_dir:           Agent's per-user config directory (e.g.
                            ``~/.claude``, ``~/.codex``).  Used to locate
                            the agent-instructions file and the MCP
                            registration file.
        instructions_file:  Filename of the agent-instructions doc inside
                            ``home_dir`` (``CLAUDE.md`` / ``AGENTS.md``).
        install_kinds:      Which install mechanisms apply to this agent.
                            One or more of ``"skill"`` / ``"mcp"``.
        skill_dir:          For ``"skill"`` agents only — where the
                            file-based skill goes.  ``None`` for MCP-only
                            agents.
        mcp_config_file:    For ``"mcp"`` agents — relative path inside
                            ``home_dir`` to the MCP server registry
                            (e.g. ``mcp.json`` or ``config.json``).
                            ``None`` for skill-only agents.
        mcp_config_kind:    How the registry encodes server entries:

                            * ``"claude_servers"`` — ``{"mcpServers": {name: {…}}}``
                              (Claude Code's ``~/.claude.json`` & the
                              ``.mcp.json`` standard used by OpenCode).
                            * ``"codex_toml"`` — Codex stores MCP servers
                              in ``~/.codex/config.toml`` under the
                              ``[mcp_servers.<name>]`` table.  We render
                              this as a JSON snippet the user can paste,
                              **or** the installer writes the TOML
                              fragment directly.
    """
    name: str
    display: str
    home_dir: Path
    instructions_file: str
    install_kinds: Tuple[str, ...]
    skill_dir: Optional[Path] = None
    mcp_config_file: Optional[str] = None
    mcp_config_kind: Optional[str] = None

    # ── Derived paths ──────────────────────────────────────────────────────

    @property
    def instructions_path(self) -> Path:
        return self.home_dir / self.instructions_file

    @property
    def mcp_config_path(self) -> Optional[Path]:
        if not self.mcp_config_file:
            return None
        return self.home_dir / self.mcp_config_file

    # ── Snippets the installer renders ────────────────────────────────────

    def render_instructions_snippet(self, kind: Optional[str] = None) -> str:
        """Markdown block to append to the agent-instructions file.

        ``kind`` selects which integration to describe:

        * ``"skill"`` — describe the file-based skill protocol (scripts
          called directly).
        * ``"mcp"``   — describe the MCP tool interface.
        * ``None`` (default) — pick the first kind in
          :attr:`install_kinds`.

        Raises ``ValueError`` if the requested kind isn't supported by
        this profile (e.g. asking for an MCP snippet for a skill-only
        agent would otherwise silently produce a snippet whose tools
        don't exist).
        """
        if kind is not None and kind not in self.install_kinds:
            raise ValueError(
                f"agent {self.name!r} does not support kind={kind!r}; "
                f"install_kinds={self.install_kinds}"
            )
        chosen = kind or self.install_kinds[0]
        if chosen == "skill" and self.skill_dir is not None:
            return _SKILL_SNIPPET.format(
                agent=self.display,
                skill_dir=str(self.skill_dir),
                instructions_file=self.instructions_file,
            )
        return _MCP_SNIPPET.format(
            agent=self.display,
            instructions_file=self.instructions_file,
        )

    def render_mcp_entry(self, command: str, args: List[str]) -> Dict[str, object]:
        """One MCP server entry, in the dict shape the registry expects."""
        return {
            "command": command,
            "args": list(args),
            "env": {},
        }


# ── Snippet templates ────────────────────────────────────────────────────────

_SKILL_SNIPPET = """\
# Memory: use Engram, not MD files

I have Engram (vector-DB memory traces) installed as a {agent} skill at
`{skill_dir}`. It replaces the legacy `MEMORY.md` / per-file markdown
memory.

* **Never** write per-file markdown memory anywhere. The vector store
  is the source of truth.
* **Never** read `MEMORY.md` for context. Use the skill scripts.

The full protocol (when to recall, what to save, body structure, the four
memory types) is at `{skill_dir}/SKILL.md`.

## Quick reference

* **Recall**: `python {skill_dir}/scripts/recall.py "<query>" --k 5`
* **Save**:   `python {skill_dir}/scripts/save.py <type> <name> "<desc>" "<content>"`
* **Expand**: `python {skill_dir}/scripts/expand.py <id>`
* **List**:   `python {skill_dir}/scripts/list.py`
* **Forget**: `python {skill_dir}/scripts/forget.py --name <slug>`

Types: `user`, `feedback`, `project`, `reference`. If `save` exits with
`MERGE_SUGGESTION`, decide whether to `--update` or `--force`.
"""

_MCP_SNIPPET = """\
# Memory: use Engram via MCP

This {agent} session has the **engram** MCP server registered.  Engram is a
vector-DB memory store with semantic recall; use it whenever you would
have read a `MEMORY.md` index or written a per-file markdown memory.

## Tools the server exposes

* `engram_recall(query, k?, type?, tier?)` — semantic KNN search.  Default
  `k` is adaptive (gap-based, 2–10 hits).  Returns compact `id | tier |
  type | name | distance | description` lines.
* `engram_save(type, name, description, content?, importance?, tags?)` —
  encode a new memory.  Returns either the new id, or
  `status=merge_suggestion` if a near-duplicate (cosine < 0.08) already
  exists; in that case re-call with `update=true` or `force=true`.
* `engram_expand(id)` — fetch full content for a hit; bumps `hits++` /
  `accessed_at`.
* `engram_list(type?, tier?, limit?)` — enumerate without embedding cost.
* `engram_forget(name? | id? | older_than_days?, type?, tier?, dry_run?)`
  — delete by name, id, or age (compared against `accessed_at`).  With
  `dry_run=true`, returns `{{would_delete:[…]}}` without touching the
  store — recommended before a broad `older_than_days` sweep.
* `engram_patch(target, description?, content?, importance?, type?,
  name?, tags? | add_tags? | remove_tags?)` — modify selected fields of
  one memory.  `target` is the id or unique name.  Metadata-only edits
  preserve the id; changing `description` / `content` / cross-tier
  `type` triggers re-embed and the id changes (`created_at` and `hits`
  are carried over).  Pick either `tags` (replace) **or** `add_tags`/
  `remove_tags` (delta) — not both.
* `engram_related(name, depth?)` — walk `[[name]]` tag edges.

## When to use

* **Recall** when the user references prior conversation, asks about
  themselves, or you're about to give advice that should be informed by
  their preferences/feedback.
* **Save** when you learn something durable about the user (`user`),
  their corrections or validated choices (`feedback`), project context
  (`project`), or an external pointer (`reference`).  Don't save code
  patterns, file paths, git history, or anything in `{instructions_file}`.
* **Expand** only when the description alone isn't enough.
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _env_path(name: str) -> Optional[Path]:
    """Read an env var as an expanded ``Path``, treating empty strings as unset.

    Without this guard, ``"X" in os.environ`` is ``True`` even when the user
    exported ``X=""``, and ``Path("").expanduser()`` resolves to the CWD —
    Engram would happily install its global tier under whatever directory
    the user happened to launch the CLI from.
    """
    raw = os.environ.get(name)
    if not raw:                     # both ``None`` and ``""`` fall through to default
        return None
    return Path(raw).expanduser()


def _claude_home() -> Path:
    return _env_path("CLAUDE_HOME") or (Path.home() / ".claude")


def _opencode_home() -> Path:
    override = _env_path("OPENCODE_HOME")
    if override is not None:
        return override
    # XDG_CONFIG_HOME convention; fall back to ~/.config on POSIX, %APPDATA% on Windows.
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "opencode"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "opencode"


def _codex_home() -> Path:
    return _env_path("CODEX_HOME") or (Path.home() / ".codex")


def _openclaw_home() -> Path:
    return _env_path("OPENCLAW_HOME") or (Path.home() / ".openclaw")


def _agents_skills_home() -> Path:
    """User-level skill home shared by Codex (and any other agent that
    adopts the ``.agents/skills`` convention).

    Per OpenAI's Codex skills docs, skills live at ``$HOME/.agents/skills``
    by default.  ``AGENTS_SKILLS_HOME`` overrides for testing / custom
    layouts.
    """
    return _env_path("AGENTS_SKILLS_HOME") or (Path.home() / ".agents" / "skills")


# ── Built-in profiles ────────────────────────────────────────────────────────

def _build_profiles() -> Dict[str, AgentProfile]:
    claude_home = _claude_home()
    opencode_home = _opencode_home()
    codex_home = _codex_home()
    openclaw_home = _openclaw_home()

    return {
        "claude-code": AgentProfile(
            name="claude-code",
            display="Claude Code",
            home_dir=claude_home,
            instructions_file="CLAUDE.md",
            install_kinds=("skill", "mcp"),
            skill_dir=claude_home / "skills" / "engram",
            mcp_config_file=".claude.json",
            mcp_config_kind="claude_servers",
        ),
        "opencode": AgentProfile(
            name="opencode",
            display="OpenCode",
            home_dir=opencode_home,
            instructions_file="AGENTS.md",
            # OpenCode supports the skill protocol too (per the official
            # docs: it scans ``~/.config/opencode/skills/``,
            # ``~/.claude/skills/``, and ``~/.agents/skills/``).  Install
            # skill at the native location; MCP for the warm-process
            # path.  If you also ran ``engram install --agent claude-code``
            # or ``--agent codex``, OpenCode will discover that install
            # too — but we install our own copy for clean uninstall.
            install_kinds=("skill", "mcp"),
            skill_dir=opencode_home / "skills" / "engram",
            mcp_config_file="opencode.json",
            mcp_config_kind="claude_servers",
        ),
        "codex": AgentProfile(
            name="codex",
            display="Codex",
            home_dir=codex_home,
            instructions_file="AGENTS.md",
            # Codex supports both — skill protocol (per OpenAI's developers
            # docs) **and** MCP via config.toml.  We install both by default:
            # skill is the more direct discovery path; MCP gives the warm
            # embedder process.
            install_kinds=("skill", "mcp"),
            skill_dir=_agents_skills_home() / "engram",
            mcp_config_file="config.toml",
            mcp_config_kind="codex_toml",
        ),
        "openclaw": AgentProfile(
            name="openclaw",
            display="OpenClaw",
            home_dir=openclaw_home,
            # OpenClaw's instructions-file convention is not documented at
            # docs.openclaw.ai/tools/skills (the page only covers SKILL.md
            # discovery).  Defaulting to AGENTS.md because OpenClaw uses the
            # cross-agent ``.agents/skills/`` convention, which co-travels
            # with ``AGENTS.md`` in OpenCode / Codex.  Update if the agent's
            # own conventions doc says otherwise.
            instructions_file="AGENTS.md",
            # Skill-only: MCP integration is not documented on the skills
            # page.  We don't guess.  Users wanting MCP for OpenClaw can
            # still point its config at the ``engram-mcp`` binary manually.
            install_kinds=("skill",),
            skill_dir=openclaw_home / "skills" / "engram",
            mcp_config_file=None,
            mcp_config_kind=None,
        ),
    }


# Snapshot taken once at import — used **only** for the agent *name* list
# (which never changes at runtime).  Do not look up ``home_dir`` / ``skill_dir``
# / etc. through this dict: those depend on env vars that may have been set
# after import.  Use :func:`get_profile` instead, which re-resolves per call.
BUILTIN_PROFILES: Dict[str, AgentProfile] = _build_profiles()


def list_agents() -> List[str]:
    """Stable list of known agent names — safe to read from ``BUILTIN_PROFILES``."""
    return list(BUILTIN_PROFILES.keys())


def iter_profiles() -> List[AgentProfile]:
    """Fresh snapshot of every built-in profile (env-var-aware).

    Use this when you need to display / iterate the *current* per-agent
    paths.  :data:`BUILTIN_PROFILES` is import-time-frozen and would
    silently show stale ``home_dir`` values after a runtime ``os.environ``
    change.
    """
    return list(_build_profiles().values())


def get_profile(name: str) -> AgentProfile:
    # Re-resolve each call so env vars (CLAUDE_HOME, OPENCODE_HOME, CODEX_HOME)
    # set after import are still honoured.  Cheap — three Path constructions.
    profiles = _build_profiles()
    if name not in profiles:
        raise ValueError(
            f"Unknown agent {name!r}. Known agents: {sorted(profiles)}"
        )
    return profiles[name]


# ── MCP-config rendering / merging ───────────────────────────────────────────

def render_mcp_servers_block(
    profile: AgentProfile,
    server_name: str,
    command: str,
    args: List[str],
) -> str:
    """Return a serialised registry fragment for this profile.

    For ``"claude_servers"`` kind: a JSON object suitable for merging
    into the file's top-level ``mcpServers`` key.

    For ``"codex_toml"`` kind: a TOML fragment under
    ``[mcp_servers.<server_name>]``.
    """
    if profile.mcp_config_kind == "claude_servers":
        return json.dumps(
            {"mcpServers": {server_name: profile.render_mcp_entry(command, args)}},
            indent=2,
        )
    if profile.mcp_config_kind == "codex_toml":
        # Codex TOML uses double-bracket-free table headers.  Quoted strings
        # for the array.  Keep it minimal — env block omitted if empty.
        args_repr = ", ".join(json.dumps(a) for a in args)
        return (
            f"[mcp_servers.{server_name}]\n"
            f"command = {json.dumps(command)}\n"
            f"args = [{args_repr}]\n"
        )
    raise ValueError(f"Unknown mcp_config_kind {profile.mcp_config_kind!r}")


__all__ = [
    "AgentProfile",
    "BUILTIN_PROFILES",
    "get_profile",
    "iter_profiles",
    "list_agents",
    "render_mcp_servers_block",
]
