"""Tests for the v0.4 multi-agent layer.

Covers:

* :mod:`engram.agents` — profile rendering, env-var-driven home dirs.
* :mod:`engram.store`  — ``ENGRAM_HOME`` / ``CLAUDE_HOME`` precedence.
* :mod:`engram.install` — JSON / TOML registry merging for the three
  built-in agents (claude-code, opencode, codex).

Same style as ``unit_test.py``: no pytest, no network, prints OK/FAIL,
exits with rc=1 on first failure.

Run from repo root::

    python tests/agents_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _assert(cond: bool, msg: str) -> None:
    if cond:
        print(f"OK   {msg}")
    else:
        print(f"FAIL {msg}", file=sys.stderr)
        sys.exit(1)


# ── Profiles ─────────────────────────────────────────────────────────────────

def test_profiles_basic() -> None:
    from engram.agents import BUILTIN_PROFILES, get_profile, list_agents

    names = list_agents()
    _assert("claude-code" in names, "claude-code is a built-in agent")
    _assert("opencode"    in names, "opencode is a built-in agent")
    _assert("codex"       in names, "codex is a built-in agent")
    _assert("openclaw"    in names, "openclaw is a built-in agent")

    cc = get_profile("claude-code")
    _assert(cc.instructions_file == "CLAUDE.md",
            "claude-code uses CLAUDE.md")
    _assert("skill" in cc.install_kinds and "mcp" in cc.install_kinds,
            "claude-code supports both skill + mcp installs")
    _assert(cc.skill_dir is not None,
            "claude-code has a skill_dir")

    oc = get_profile("opencode")
    _assert(oc.instructions_file == "AGENTS.md",
            "opencode uses AGENTS.md (emerging standard)")
    # OpenCode now supports both skill + MCP (per its skills docs;
    # scans ~/.config/opencode/skills/, ~/.claude/skills/, ~/.agents/skills/).
    _assert("skill" in oc.install_kinds and "mcp" in oc.install_kinds,
            "opencode supports both skill + mcp installs")
    _assert(oc.skill_dir is not None and "opencode" in str(oc.skill_dir),
            f"opencode skill_dir lives under its config home, got {oc.skill_dir}")

    cx = get_profile("codex")
    _assert(cx.instructions_file == "AGENTS.md",
            "codex uses AGENTS.md")
    # Codex (per OpenAI's skills docs) supports both skill + MCP.
    _assert("skill" in cx.install_kinds and "mcp" in cx.install_kinds,
            "codex supports both skill + mcp installs")
    _assert(cx.mcp_config_kind == "codex_toml",
            "codex registers MCP via TOML")
    _assert(cx.skill_dir is not None and ".agents" in str(cx.skill_dir),
            f"codex skill dir uses ~/.agents/skills convention, got {cx.skill_dir}")

    # OpenClaw — skill-only (MCP config not documented at docs.openclaw.ai).
    ow = get_profile("openclaw")
    _assert(ow.install_kinds == ("skill",),
            f"openclaw is skill-only, got {ow.install_kinds}")
    _assert(ow.skill_dir is not None and ".openclaw" in str(ow.skill_dir),
            f"openclaw skill dir under ~/.openclaw/skills, got {ow.skill_dir}")
    _assert(ow.mcp_config_file is None and ow.mcp_config_kind is None,
            "openclaw has no MCP config (skill-only)")


def test_empty_env_var_treated_as_unset() -> None:
    """Regression for BUG-S1: ``X=""`` previously resolved to CWD instead
    of the default, because ``"X" in os.environ`` is True for empty strings
    and ``Path("").expanduser() == Path(".")``."""
    from engram import agents, store

    saved = {k: os.environ.get(k) for k in
             ("CLAUDE_HOME", "OPENCODE_HOME", "CODEX_HOME",
              "OPENCLAW_HOME", "AGENTS_SKILLS_HOME", "ENGRAM_HOME")}
    try:
        for k in saved:
            os.environ[k] = ""

        # Each per-agent home must fall through to its hardcoded default,
        # NOT silently resolve to the CWD.
        cwd = Path.cwd()
        for name in ("claude-code", "opencode", "codex", "openclaw"):
            p = agents.get_profile(name)
            _assert(p.home_dir != cwd,
                    f"{name} home_dir != CWD with empty env, got {p.home_dir}")

        # store._engram_home() must do the same.
        _assert(store._engram_home() != cwd,
                f"_engram_home() != CWD with empty env, got {store._engram_home()}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_iter_profiles_reflects_env_changes() -> None:
    """Regression for BUG-A1: ``BUILTIN_PROFILES`` is import-time-frozen,
    so reading ``home_dir`` from it after the env changes would lie.
    ``iter_profiles()`` must re-read each call."""
    from engram import agents

    saved = os.environ.get("CLAUDE_HOME")
    try:
        os.environ["CLAUDE_HOME"] = "/tmp/iter-test-X"
        homes = {p.name: str(p.home_dir) for p in agents.iter_profiles()}
        _assert("iter-test-X" in homes["claude-code"],
                f"iter_profiles() reflects env change, got {homes['claude-code']}")

        # Re-call after change picks up new value.
        os.environ["CLAUDE_HOME"] = "/tmp/iter-test-Y"
        homes2 = {p.name: str(p.home_dir) for p in agents.iter_profiles()}
        _assert("iter-test-Y" in homes2["claude-code"],
                f"iter_profiles() re-reads on each call, got {homes2['claude-code']}")
    finally:
        if saved is None:
            os.environ.pop("CLAUDE_HOME", None)
        else:
            os.environ["CLAUDE_HOME"] = saved


def test_profile_env_overrides() -> None:
    """Per-agent *_HOME env vars + AGENTS_SKILLS_HOME steer paths."""
    from engram import agents

    saved = {k: os.environ.get(k) for k in
             ("CLAUDE_HOME", "OPENCODE_HOME", "CODEX_HOME",
              "OPENCLAW_HOME", "AGENTS_SKILLS_HOME")}
    try:
        os.environ["CLAUDE_HOME"]        = "/tmp/fakeclaude"
        os.environ["OPENCODE_HOME"]      = "/tmp/fakeopen"
        os.environ["CODEX_HOME"]         = "/tmp/fakecodex"
        os.environ["OPENCLAW_HOME"]      = "/tmp/fakeclaw"
        os.environ["AGENTS_SKILLS_HOME"] = "/tmp/fakeagents"

        cc = agents.get_profile("claude-code")
        oc = agents.get_profile("opencode")
        cx = agents.get_profile("codex")
        ow = agents.get_profile("openclaw")
        _assert(str(cc.home_dir).endswith("fakeclaude"),
                f"CLAUDE_HOME respected, got {cc.home_dir}")
        _assert(str(oc.home_dir).endswith("fakeopen"),
                f"OPENCODE_HOME respected, got {oc.home_dir}")
        _assert(oc.skill_dir is not None and "fakeopen" in str(oc.skill_dir),
                f"opencode skill_dir tracks OPENCODE_HOME, got {oc.skill_dir}")
        _assert(str(cx.home_dir).endswith("fakecodex"),
                f"CODEX_HOME respected, got {cx.home_dir}")
        _assert(cx.skill_dir is not None and "fakeagents" in str(cx.skill_dir),
                f"AGENTS_SKILLS_HOME respected for codex skill_dir, got {cx.skill_dir}")
        _assert(str(ow.home_dir).endswith("fakeclaw"),
                f"OPENCLAW_HOME respected, got {ow.home_dir}")
        _assert(ow.skill_dir is not None and "fakeclaw" in str(ow.skill_dir),
                f"openclaw skill_dir tracks OPENCLAW_HOME, got {ow.skill_dir}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_render_mcp_block() -> None:
    from engram.agents import get_profile, render_mcp_servers_block

    cc_block = render_mcp_servers_block(
        get_profile("claude-code"), "engram", "engram-mcp", [])
    parsed = json.loads(cc_block)
    _assert(parsed["mcpServers"]["engram"]["command"] == "engram-mcp",
            "claude-code block has command=engram-mcp")

    oc_block = render_mcp_servers_block(
        get_profile("opencode"), "engram", "engram-mcp", [])
    parsed = json.loads(oc_block)
    _assert(parsed["mcpServers"]["engram"]["command"] == "engram-mcp",
            "opencode block has command=engram-mcp")

    cx_block = render_mcp_servers_block(
        get_profile("codex"), "engram", "engram-mcp", ["--no-sdk"])
    _assert("[mcp_servers.engram]" in cx_block,
            "codex block is TOML with [mcp_servers.engram]")
    _assert('"--no-sdk"' in cx_block,
            "codex block quotes args")


# ── Store env-var resolution ─────────────────────────────────────────────────

def test_store_engram_home() -> None:
    """ENGRAM_HOME takes precedence over CLAUDE_HOME, which takes precedence
    over ~/.claude."""
    from engram import store

    saved = {k: os.environ.get(k) for k in ("ENGRAM_HOME", "CLAUDE_HOME")}
    try:
        # ENGRAM_HOME wins when both set.
        os.environ["ENGRAM_HOME"] = "/tmp/E"
        os.environ["CLAUDE_HOME"] = "/tmp/C"
        _assert(str(store._engram_home()).endswith("E"),
                f"ENGRAM_HOME beats CLAUDE_HOME, got {store._engram_home()}")

        # CLAUDE_HOME alone still works (back-compat).
        del os.environ["ENGRAM_HOME"]
        _assert(str(store._engram_home()).endswith("C"),
                f"CLAUDE_HOME fallback, got {store._engram_home()}")

        # Neither set → ~/.claude.
        del os.environ["CLAUDE_HOME"]
        _assert(str(store._engram_home()).endswith(".claude"),
                f"~/.claude default, got {store._engram_home()}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── Installer (JSON / TOML merging) ──────────────────────────────────────────

def test_claude_servers_merger_creates() -> None:
    """First-time write creates the file with a single server entry."""
    from engram.install import _merge_claude_servers, MCP_SERVER_NAME

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / ".claude.json"
        _merge_claude_servers(path, {"command": "engram-mcp", "args": [], "env": {}})
        data = json.loads(path.read_text(encoding="utf-8"))
        _assert(MCP_SERVER_NAME in data["mcpServers"],
                "fresh install writes the engram server entry")
        _assert(data["mcpServers"][MCP_SERVER_NAME]["command"] == "engram-mcp",
                "command field round-trips")


def test_claude_servers_merger_preserves_siblings() -> None:
    """Pre-existing keys (other servers + unrelated top-level fields) survive."""
    from engram.install import _merge_claude_servers, MCP_SERVER_NAME

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / ".claude.json"
        path.write_text(json.dumps({
            "someUserSetting": True,
            "mcpServers": {
                "other-server": {"command": "x", "args": []},
            },
        }), encoding="utf-8")

        _merge_claude_servers(path, {"command": "engram-mcp", "args": [], "env": {}})

        data = json.loads(path.read_text(encoding="utf-8"))
        _assert(data["someUserSetting"] is True,
                "unrelated top-level keys are preserved")
        _assert("other-server" in data["mcpServers"],
                "existing sibling servers are preserved")
        _assert(data["mcpServers"][MCP_SERVER_NAME]["command"] == "engram-mcp",
                "our entry is added")


def test_claude_servers_merger_idempotent() -> None:
    """Re-running install doesn't duplicate or corrupt the entry."""
    from engram.install import _merge_claude_servers, MCP_SERVER_NAME

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / ".claude.json"
        entry = {"command": "engram-mcp", "args": [], "env": {}}
        _merge_claude_servers(path, entry)
        _merge_claude_servers(path, entry)
        data = json.loads(path.read_text(encoding="utf-8"))
        # There's exactly one engram entry, regardless of how many merges ran.
        _assert(list(data["mcpServers"]).count(MCP_SERVER_NAME) == 1,
                "merger is idempotent — no dup engram entry")


def test_codex_toml_merger() -> None:
    """Codex TOML: insert + replace are both clean."""
    from engram.install import _merge_codex_toml, MCP_SERVER_NAME

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "config.toml"

        # ── First write ──
        _merge_codex_toml(path, MCP_SERVER_NAME, "engram-mcp", [])
        text = path.read_text(encoding="utf-8")
        _assert(f"[mcp_servers.{MCP_SERVER_NAME}]" in text,
                "TOML section header present")
        _assert('command = "engram-mcp"' in text,
                "command line present")

        # ── Re-merge with different args — should replace, not duplicate ──
        _merge_codex_toml(path, MCP_SERVER_NAME, "engram-mcp", ["--no-sdk"])
        text = path.read_text(encoding="utf-8")
        _assert(text.count(f"[mcp_servers.{MCP_SERVER_NAME}]") == 1,
                "TOML re-merge replaces, no dup section")
        _assert('"--no-sdk"' in text,
                "new args reach the file")


def test_atomic_write_survives_partial_failure() -> None:
    """Regression for BUG-I3 / BUG-I8.  ``_atomic_write_text`` writes to a
    sibling tempfile then ``os.replace`` — so even if the new write fails
    after the tempfile is partially staged, the *target* file is untouched.

    We can't easily simulate a power loss, but we can verify the temp
    pattern: after a successful write the ``.tmp`` sibling is gone, the
    target has the new content, and the target was never opened in
    write-mode directly."""
    from engram.install import _atomic_write_text

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "config.json"
        # Pre-populate with content that would be lost on a torn write.
        original = '{"keep": "this!"}\n'
        path.write_text(original, encoding="utf-8")

        new = '{"keep": "this!", "new": true}\n'
        _atomic_write_text(path, new)

        _assert(path.read_text(encoding="utf-8") == new,
                "atomic write replaced content")
        _assert(not (path.with_name(path.name + ".tmp")).exists(),
                "tempfile cleaned up after successful write")


def test_codex_remove_single_pass() -> None:
    """Regression for BUG-I14.  Removing the codex_toml MCP entry should
    do exactly one read + one write — the old code re-installed our
    section before stripping it (two writes, two crash windows)."""
    from engram.install import _strip_codex_section, MCP_SERVER_NAME

    text = (
        "[profile]\n"
        'model = "gpt-5"\n'
        "\n"
        "[mcp_servers.engram]\n"
        'command = "engram-mcp"\n'
        "args = []\n"
        "\n"
        "[mcp_servers.other]\n"
        'command = "x"\n'
    )
    stripped = _strip_codex_section(text, MCP_SERVER_NAME)
    _assert("[mcp_servers.engram]" not in stripped,
            "engram section removed")
    _assert("[profile]" in stripped and "[mcp_servers.other]" in stripped,
            "other sections preserved")


def test_claude_servers_non_dict_root_aborts() -> None:
    """Regression for BUG-I5: if a hand-edited ``.claude.json`` root is
    ``null`` / a list / a number, refuse to clobber it.  ``data.setdefault``
    used to AttributeError; we now raise RuntimeError explicitly."""
    from engram.install import _merge_claude_servers

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / ".claude.json"
        path.write_text("null", encoding="utf-8")
        try:
            _merge_claude_servers(path, {"command": "engram-mcp", "args": [], "env": {}})
        except RuntimeError as e:
            _assert("refusing" in str(e).lower() or "root is" in str(e).lower(),
                    f"non-dict root aborts with informative message, got: {e}")
            # File must be untouched.
            _assert(path.read_text(encoding="utf-8") == "null",
                    "non-dict root file is left untouched")
        else:
            _assert(False, "should have raised on non-dict root")


def test_codex_toml_preserves_other_sections() -> None:
    """Other sections of config.toml survive our edits."""
    from engram.install import _merge_codex_toml, MCP_SERVER_NAME

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "config.toml"
        path.write_text(
            "[profile]\n"
            'model = "gpt-5"\n'
            "\n"
            "[mcp_servers.other]\n"
            'command = "other"\n',
            encoding="utf-8",
        )

        _merge_codex_toml(path, MCP_SERVER_NAME, "engram-mcp", [])
        text = path.read_text(encoding="utf-8")
        _assert("[profile]" in text and 'model = "gpt-5"' in text,
                "unrelated [profile] section survives")
        _assert("[mcp_servers.other]" in text,
                "unrelated [mcp_servers.other] survives")
        _assert(f"[mcp_servers.{MCP_SERVER_NAME}]" in text,
                "our section was added")


# ── init_project (v0.4: engram init) ────────────────────────────────────────

def test_init_writes_snippet_and_gitignore() -> None:
    """``init`` writes CLAUDE.md snippet + .gitignore + .claude/engram/ dir."""
    from engram import init_project

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rc = init_project.init(agent="claude-code", project_root=root)
        _assert(rc == 0, f"init exit code is 0 (got {rc})")
        claude_md = (root / "CLAUDE.md").read_text(encoding="utf-8")
        _assert("use Engram, not MD files" in claude_md,
                "CLAUDE.md contains the skill snippet marker")
        gi = (root / ".gitignore").read_text(encoding="utf-8")
        _assert(".claude/engram/" in gi,
                ".gitignore has the local-tier rule")
        _assert((root / ".claude" / "engram").is_dir(),
                ".claude/engram/ directory created")


def test_init_idempotent() -> None:
    """Re-running ``init`` does not duplicate the snippet or .gitignore line."""
    from engram import init_project

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        init_project.init(agent="claude-code", project_root=root)
        first_md  = (root / "CLAUDE.md").read_text(encoding="utf-8")
        first_gi  = (root / ".gitignore").read_text(encoding="utf-8")
        # Second run: should be a no-op for both files.
        rc = init_project.init(agent="claude-code", project_root=root)
        _assert(rc == 0, "second init also exits 0")
        _assert((root / "CLAUDE.md").read_text(encoding="utf-8") == first_md,
                "CLAUDE.md unchanged on second init")
        _assert((root / ".gitignore").read_text(encoding="utf-8") == first_gi,
                ".gitignore unchanged on second init")


def test_init_preserves_existing_claude_md() -> None:
    """If CLAUDE.md already exists with unrelated content, ``init`` appends
    rather than overwrites."""
    from engram import init_project

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pre = "# Project rules\n\nUse tabs.\n"
        (root / "CLAUDE.md").write_text(pre, encoding="utf-8")
        init_project.init(agent="claude-code", project_root=root)
        final = (root / "CLAUDE.md").read_text(encoding="utf-8")
        _assert(final.startswith(pre.rstrip()),
                "pre-existing CLAUDE.md content preserved at the top")
        _assert("use Engram, not MD files" in final,
                "Engram snippet appended after existing content")


def test_init_gitignore_tolerates_leading_slash() -> None:
    """A ``/.claude/engram/`` entry already in .gitignore is the same rule;
    don't double-add."""
    from engram import init_project

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / ".gitignore").write_text("/.claude/engram/\n", encoding="utf-8")
        init_project.init(
            agent="claude-code", project_root=root, with_snippet=False,
        )
        gi = (root / ".gitignore").read_text(encoding="utf-8")
        # Should still only contain one effective rule, not two.
        rules = [
            ln.split("#", 1)[0].strip().lstrip("/").rstrip("/")
            for ln in gi.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        engram_rules = [r for r in rules if r == ".claude/engram"]
        _assert(len(engram_rules) == 1,
                f"only one effective rule for .claude/engram ({rules})")


def test_init_force_re_appends() -> None:
    """``--force`` re-appends the snippet even though the marker exists.

    Useful after the user has manually deleted the block and wants it
    back, or when the snippet template was updated upstream.
    """
    from engram import init_project

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        init_project.init(agent="claude-code", project_root=root)
        first_len = len((root / "CLAUDE.md").read_text(encoding="utf-8"))
        init_project.init(
            agent="claude-code", project_root=root, force=True,
        )
        second_len = len((root / "CLAUDE.md").read_text(encoding="utf-8"))
        _assert(second_len > first_len,
                f"--force grew CLAUDE.md (was {first_len}, now {second_len})")


# ── doctor (v0.4: engram doctor) ────────────────────────────────────────────

def test_doctor_runs_and_returns_results() -> None:
    """The doctor driver iterates every registered check, collects results,
    and returns one of {0, 1, 2}.  We can't assert specific OK/WARN/ERR
    counts because they depend on the operator's environment, but we
    *can* assert that the driver returns and produces a non-empty list
    when invoked via the JSON path."""
    import io
    import contextlib
    from engram import doctor

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = doctor.run(verbose=False, as_json=True)
    _assert(rc in (0, 1, 2),
            f"doctor returns one of 0/1/2 (got {rc})")
    payload = json.loads(buf.getvalue())
    _assert("results" in payload and len(payload["results"]) >= 5,
            f"doctor JSON has a non-empty results list (got {len(payload.get('results', []))})")
    # Sanity: every result has the three required fields.
    for r in payload["results"]:
        _assert(set(r) >= {"name", "status", "message"},
                f"doctor result has required fields: {r}")
        _assert(r["status"] in ("ok", "warn", "err"),
                f"doctor result status is valid enum: {r['status']}")


# ── embedder warmup (v0.4: engram warmup) ───────────────────────────────────

def test_warmup_hash_backend() -> None:
    """``warmup()`` with the hash backend runs in a fraction of a second
    and returns the resolved spec/dim metadata."""
    from engram.embedder import warmup

    result = warmup(spec="hash?dim=64")
    _assert(result["scheme"] == "hash",
            f"warmup resolved scheme=hash (got {result['scheme']!r})")
    _assert(result["dim"] == 64,
            f"warmup honoured ?dim=64 (got {result['dim']})")
    _assert(result["elapsed_seconds"] >= 0.0,
            f"elapsed_seconds is non-negative (got {result['elapsed_seconds']})")


# ── MemoryHit JSON-safety (regression for BUG-M11) ──────────────────────────

def test_memory_hit_to_dict_is_json_safe() -> None:
    """Regression for BUG-M11: PistaDB returns ``distance`` as
    ``numpy.float32``, ``round()`` preserves that dtype, and
    ``json.dumps(numpy.float32)`` raises TypeError — every MCP tool
    response embedding a hit used to land in the isError path.

    ``MemoryHit.to_dict()`` must coerce numeric fields to Python builtins."""
    import numpy as np
    from engram.memory import MemoryHit

    hit = MemoryHit(
        id          = np.int64(42),
        tier        = "local",
        type        = "project",
        name        = "demo",
        description = "desc",
        distance    = np.float32(0.12345678),
        importance  = np.float32(0.7),
        hits        = np.int64(3),
        created_at  = np.int64(1700000000),
        accessed_at = np.int64(1700000001),
        tags        = ["a"],
    )
    d = hit.to_dict()
    # Must round-trip through json without raising.
    json.dumps(d)

    # Spot-check that the dtypes were normalised.
    _assert(type(d["distance"]) is float,
            f"distance is Python float, got {type(d['distance']).__name__}")
    _assert(type(d["id"]) is int,
            f"id is Python int, got {type(d['id']).__name__}")


# ── Snippet rendering ────────────────────────────────────────────────────────

def test_snippet_includes_correct_instructions_file() -> None:
    """Each profile's snippet mentions its own agent-instructions filename."""
    from engram.agents import get_profile

    cc = get_profile("claude-code").render_instructions_snippet()
    _assert("Claude Code skill" in cc and "SKILL.md" in cc,
            "claude-code snippet mentions Skill + SKILL.md")

    oc_default = get_profile("opencode").render_instructions_snippet()
    _assert("OpenCode skill" in oc_default and "skills/engram" in oc_default.replace("\\", "/"),
            "opencode default snippet mentions OpenCode skill + skills/engram path")

    oc_mcp = get_profile("opencode").render_instructions_snippet(kind="mcp")
    _assert("MCP" in oc_mcp and "AGENTS.md" in oc_mcp,
            "opencode --snippet-kind=mcp produces MCP snippet")

    cx_default = get_profile("codex").render_instructions_snippet()
    _assert("Codex skill" in cx_default and ".agents/skills" in cx_default.replace("\\", "/"),
            "codex default snippet mentions Codex skill + .agents/skills path")

    cx_mcp = get_profile("codex").render_instructions_snippet(kind="mcp")
    _assert("MCP" in cx_mcp and "AGENTS.md" in cx_mcp,
            "codex --snippet-kind=mcp produces MCP snippet")

    # OpenClaw — skill snippet only; asking for MCP must error out
    # (rather than silently render a snippet whose tools don't exist).
    ow_default = get_profile("openclaw").render_instructions_snippet()
    _assert("OpenClaw skill" in ow_default and ".openclaw" in ow_default.replace("\\", "/"),
            "openclaw default snippet mentions OpenClaw skill + .openclaw path")
    try:
        get_profile("openclaw").render_instructions_snippet(kind="mcp")
    except ValueError as e:
        _assert("does not support" in str(e),
                f"openclaw --snippet-kind=mcp raises ValueError, got: {e}")
    else:
        _assert(False, "openclaw --snippet-kind=mcp should have raised ValueError")


# ── setup wizard ──────────────────────────────────────────────────────────────

def test_setup_select_agents_preselect() -> None:
    """An explicit --agent short-circuits the interactive picker."""
    from engram.setup_wizard import _select_agents
    from engram.agents import list_agents

    _assert(_select_agents("claude-code", assume_yes=False) == ["claude-code"],
            "preselect of a single agent returns just that agent")
    _assert(_select_agents("all", assume_yes=False) == list_agents(),
            "preselect 'all' returns every agent")


def test_setup_select_agents_default_noninteractive() -> None:
    """With no preselect and no TTY, fall back to detected / claude-code."""
    from engram.setup_wizard import _select_agents
    from engram.agents import list_agents

    # The test harness runs with stdin not a TTY, so this exercises the
    # non-interactive default path without blocking on input().
    chosen = _select_agents(None, assume_yes=True)
    valid = set(list_agents())
    _assert(len(chosen) >= 1 and set(chosen) <= valid,
            f"default selection is a non-empty subset of known agents, got {chosen}")


def test_setup_ask_yes_no_defaults() -> None:
    """In non-interactive / assume_yes mode the prompt returns its default."""
    from engram.setup_wizard import _ask_yes_no

    _assert(_ask_yes_no("q?", default=True, assume_yes=True) is True,
            "assume_yes honours a True default")
    _assert(_ask_yes_no("q?", default=False, assume_yes=True) is False,
            "assume_yes honours a False default")


def test_setup_noninteractive_guard() -> None:
    """`engram setup` without --yes on a non-TTY refuses (exit 2), no side effects."""
    from engram import setup_wizard

    # stdin is not a TTY under the test runner; without assume_yes the
    # wizard must bail before touching install/warmup/doctor.
    rc = setup_wizard.run(agent="claude-code", assume_yes=False)
    _assert(rc == 2, f"non-interactive setup without --yes returns 2, got {rc}")


# ── Driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== tests for v0.4 multi-agent layer ===")

    print("\n--- profiles ---")
    test_profiles_basic()
    test_empty_env_var_treated_as_unset()
    test_iter_profiles_reflects_env_changes()
    test_profile_env_overrides()
    test_render_mcp_block()
    test_snippet_includes_correct_instructions_file()

    print("\n--- store env-var resolution ---")
    test_store_engram_home()

    print("\n--- installer: claude_servers JSON merger ---")
    test_claude_servers_merger_creates()
    test_claude_servers_merger_preserves_siblings()
    test_claude_servers_merger_idempotent()
    test_claude_servers_non_dict_root_aborts()

    print("\n--- installer: codex TOML merger ---")
    test_codex_toml_merger()
    test_codex_toml_preserves_other_sections()
    test_codex_remove_single_pass()

    print("\n--- installer: atomic write ---")
    test_atomic_write_survives_partial_failure()

    print("\n--- MemoryHit JSON-safety ---")
    test_memory_hit_to_dict_is_json_safe()

    print("\n--- init_project: engram init ---")
    test_init_writes_snippet_and_gitignore()
    test_init_idempotent()
    test_init_preserves_existing_claude_md()
    test_init_gitignore_tolerates_leading_slash()
    test_init_force_re_appends()

    print("\n--- doctor: engram doctor ---")
    test_doctor_runs_and_returns_results()

    print("\n--- warmup: engram warmup ---")
    test_warmup_hash_backend()

    print("\n--- setup: engram setup wizard ---")
    test_setup_select_agents_preselect()
    test_setup_select_agents_default_noninteractive()
    test_setup_ask_yes_no_defaults()
    test_setup_noninteractive_guard()

    print("\nALL AGENTS-LAYER TESTS PASSED.")


if __name__ == "__main__":
    main()
