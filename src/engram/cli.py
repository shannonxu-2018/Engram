"""Console entry point for the ``engram`` CLI.

Wired up via ``pyproject.toml``:

    [project.scripts]
    engram     = "engram.cli:main"
    engram-mcp = "engram.mcp_server:main"

Subcommands:
    engram install         --agent {claude-code,opencode,codex,all}
                            [--target DIR] [--dev] [--force] [--remove]
                            [--check] [--print-instructions-snippet]
                            [--no-skill] [--no-mcp]
    engram install-skill   (alias of `install --no-mcp`, kept for back-compat)
    engram setup           interactive wizard: install + enable + warmup + doctor
    engram uninstall       one-click teardown by scope (project / global / all)
    engram agents          list known agents and their integration mode
    engram version
    engram --version
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("engram")
    except Exception:
        return "0.0.0+unknown"


def _add_install_args(s: argparse.ArgumentParser, *, legacy_skill_only: bool) -> None:
    """Shared flags between ``install`` and ``install-skill``."""
    if not legacy_skill_only:
        # ``choices`` lets argparse reject typos at parse time with a clean
        # message, instead of letting the bad name reach ``get_profile()``
        # and surface as an unfiltered ValueError traceback.
        from .agents import list_agents as _list_agents
        s.add_argument(
            "--agent",
            default="claude-code",
            choices=_list_agents() + ["all"],
            help="Target agent (default: claude-code).",
        )
    s.add_argument(
        "--target",
        help="Override skill target dir (default = profile.skill_dir).",
    )
    s.add_argument(
        "--dev",
        action="store_true",
        help="Symlink skill files instead of copying (edits in repo "
             "propagate live; needs Windows Developer Mode or admin).",
    )
    s.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing install.",
    )
    s.add_argument(
        "--remove",
        action="store_true",
        help="Remove the install for the chosen agent and exit.",
    )
    s.add_argument(
        "--check",
        action="store_true",
        help="Report current status without writing.",
    )
    if not legacy_skill_only:
        s.add_argument(
            "--no-skill",
            action="store_true",
            help="Skip the file-based skill install (MCP-only).",
        )
        s.add_argument(
            "--no-mcp",
            action="store_true",
            help="Skip MCP registration (file-skill-only — v0.3 behaviour).",
        )
        s.add_argument(
            "--hook",
            action="store_true",
            help="Also register the per-turn UserPromptSubmit hook that "
                 "injects standing directives (Claude Code & Codex). OFF by "
                 "default; toggle later with `engram hook`.",
        )
        s.add_argument(
            "--print-instructions-snippet",
            dest="print_snippet",
            action="store_true",
            help="Print the markdown block to append to the agent's "
                 "instructions file (CLAUDE.md / AGENTS.md) and exit.",
        )
        s.add_argument(
            "--snippet-kind",
            choices=["skill", "mcp"],
            default=None,
            help="Snippet framing: 'skill' (script-call protocol) or "
                 "'mcp' (MCP tool protocol).  Default = the agent's "
                 "primary integration.",
        )
    s.add_argument(
        "--print-global-snippet",
        dest="print_global_snippet",
        action="store_true",
        help="Deprecated alias of --print-instructions-snippet.",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="engram",
        description="Engram — vector memory traces for AI agents.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"engram {_version()}",
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # ── install ─────────────────────────────────────────────────────────
    inst = sub.add_parser(
        "install",
        help="Install / manage Engram for one or all supported agents.",
    )
    _add_install_args(inst, legacy_skill_only=False)
    inst.set_defaults(func=_cmd_install)

    # ── install-skill (alias / back-compat) ─────────────────────────────
    sk = sub.add_parser(
        "install-skill",
        help="Alias of `install --agent claude-code --no-mcp` (kept for v0.3 users).",
    )
    _add_install_args(sk, legacy_skill_only=True)
    sk.set_defaults(func=_cmd_install_skill)

    # ── agents ──────────────────────────────────────────────────────────
    ag = sub.add_parser("agents", help="List the supported agents.")
    ag.set_defaults(func=_cmd_agents)

    # ── setup ───────────────────────────────────────────────────────────
    from .agents import list_agents as _la_for_setup
    st = sub.add_parser(
        "setup",
        help="Interactive wizard: install + enable + warmup + doctor in one go.",
    )
    st.add_argument(
        "--agent",
        default=None,
        choices=_la_for_setup() + ["all"],
        help="Skip the agent question and target this agent (or 'all').",
    )
    st.add_argument(
        "--yes", "-y", dest="assume_yes", action="store_true",
        help="Non-interactive: accept all recommended defaults (for scripts).",
    )
    st.add_argument(
        "--dev", action="store_true",
        help="Symlink skill files instead of copying (needs Windows Developer Mode).",
    )
    st.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing skill install / re-append snippets.",
    )
    st.set_defaults(func=_cmd_setup)

    # ── init ────────────────────────────────────────────────────────────
    from .agents import list_agents as _la_for_init
    ini = sub.add_parser(
        "init",
        help="Wire the current project (CLAUDE.md / .gitignore / .claude/engram/).",
    )
    ini.add_argument(
        "--agent",
        default="claude-code",
        choices=_la_for_init() + ["all"],
        help="Target agent (default: claude-code).",
    )
    ini.add_argument(
        "--in", dest="project_root", default=None,
        help="Project root to wire (default: current working directory).",
    )
    ini.add_argument(
        "--no-snippet", action="store_true",
        help="Skip appending the instructions block to CLAUDE.md/AGENTS.md.",
    )
    ini.add_argument(
        "--no-gitignore", action="store_true",
        help="Skip touching .gitignore.",
    )
    ini.add_argument(
        "--force", action="store_true",
        help="Re-append the instructions block even if a marker is found.",
    )
    ini.set_defaults(func=_cmd_init)

    # ── uninstall ───────────────────────────────────────────────────────
    from .agents import list_agents as _la_for_uninstall
    un = sub.add_parser(
        "uninstall",
        help="One-click teardown: remove the project-level wiring, the "
             "global skill+MCP install, or both.",
    )
    un.add_argument(
        "--scope",
        choices=["project", "global", "all"],
        default=None,
        help="What to remove: 'project' (reverse of `engram init`), "
             "'global' (reverse of `engram install`), or 'all'. "
             "Omit to be asked interactively.",
    )
    un.add_argument(
        "--agent",
        default=None,
        choices=_la_for_uninstall() + ["all"],
        help="Narrow to one agent (default: all agents).",
    )
    un.add_argument(
        "--in", dest="project_root", default=None,
        help="Project root to un-wire (default: current working directory).",
    )
    un.add_argument(
        "--purge", action="store_true",
        help="Also DELETE the stored vector memories (irreversible).",
    )
    un.add_argument(
        "--keep-data", action="store_true",
        help="Keep stored memories without being asked (the safe default).",
    )
    un.add_argument(
        "--yes", "-y", dest="assume_yes", action="store_true",
        help="Non-interactive: accept defaults (scope=project, keep data).",
    )
    un.set_defaults(func=_cmd_uninstall)

    # ── doctor ──────────────────────────────────────────────────────────
    doc = sub.add_parser(
        "doctor",
        help="Run a health check across package / agents / tiers / embedder.",
    )
    doc.add_argument(
        "--verbose", action="store_true",
        help="Print per-check details under each line.",
    )
    doc.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Emit machine-readable JSON instead of the human report.",
    )
    doc.set_defaults(func=_cmd_doctor)

    # ── warmup ──────────────────────────────────────────────────────────
    wu = sub.add_parser(
        "warmup",
        help="Pre-load the embedder so the first recall doesn't pay cold-start.",
    )
    wu.add_argument(
        "--spec", default=None,
        help="Override ENGRAM_EMBEDDER for this run (e.g. 'openai:text-embedding-3-small').",
    )
    wu.set_defaults(func=_cmd_warmup)

    # ── remember (the simple path) ──────────────────────────────────────
    rem = sub.add_parser(
        "remember",
        help="Quickly save a memory — the simple path (auto type/name/importance).",
    )
    rem.add_argument("text", help="What to remember (a one-line fact or preference).")
    rem.add_argument(
        "--project", action="store_true",
        help="Scope to THIS project (default: personal — follows you everywhere).",
    )
    rem.add_argument(
        "--name", default=None,
        help="Slug for later forget/links (default: auto-generated from the text).",
    )
    rem.add_argument(
        "--pin", action="store_true",
        help="Make it an always-on standing directive (injected every turn).",
    )
    rem.add_argument(
        "--force", action="store_true",
        help="Save even if a near-duplicate already exists.",
    )
    rem.set_defaults(func=_cmd_remember)

    # ── recall / list / forget (core memory ops, now first-class on the CLI) ─
    rcl = sub.add_parser("recall", help="Semantic search over your memories.")
    rcl.add_argument("query")
    rcl.add_argument("--k", type=int, default=None, help="Top-K (default: adaptive 2-10).")
    rcl.add_argument("--type", default=None,
                     choices=["user", "feedback", "project", "reference"])
    rcl.add_argument("--tier", default=None, choices=["global", "local"])
    rcl.add_argument("--with-content", dest="with_content", action="store_true")
    rcl.add_argument("--json", dest="as_json", action="store_true")
    rcl.set_defaults(func=_cmd_recall)

    lst = sub.add_parser("list", help="Enumerate memories (no embedding cost).")
    lst.add_argument("--type", default=None,
                     choices=["user", "feedback", "project", "reference"])
    lst.add_argument("--tier", default=None, choices=["global", "local"])
    lst.add_argument("--limit", type=int, default=100)
    lst.add_argument("--json", dest="as_json", action="store_true")
    lst.set_defaults(func=_cmd_list)

    fgt = sub.add_parser("forget", help="Delete memories by name / id / age.")
    fgt.add_argument("--name", default=None)
    fgt.add_argument("--id", type=int, default=None)
    fgt.add_argument("--older-than", dest="older_than", type=float, default=None,
                     help="Delete memories not accessed in this many days.")
    fgt.add_argument("--type", default=None,
                     choices=["user", "feedback", "project", "reference"])
    fgt.add_argument("--tier", default=None, choices=["global", "local"])
    fgt.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="Preview what would be deleted without deleting.")
    fgt.set_defaults(func=_cmd_forget)

    # ── directives ──────────────────────────────────────────────────────
    di = sub.add_parser(
        "directives",
        help="Print standing directives (pinned, always-on constraints). "
             "Used as the per-turn UserPromptSubmit hook command.",
    )
    di.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Emit JSON instead of the injectable text block.",
    )
    di.set_defaults(func=_cmd_directives)

    # ── hook ────────────────────────────────────────────────────────────
    from .agents import list_agents as _la_for_hook
    hk = sub.add_parser(
        "hook",
        help="Enable (default) or disable the per-turn UserPromptSubmit "
             "directives hook for Claude Code / Codex.  Off by default.",
    )
    hk.add_argument(
        "--agent", default="claude-code",
        choices=_la_for_hook() + ["all"],
        help="Target agent (default: claude-code).",
    )
    hk.add_argument(
        "--disable", action="store_true",
        help="Remove the hook instead of installing it.",
    )
    hk.set_defaults(func=_cmd_hook)

    # ── serve (warm embedding daemon) ────────────────────────────────────
    srv = sub.add_parser(
        "serve",
        help="Warm-embedder daemon (keeps e5 loaded between hook turns). "
             "Normally self-managed; these flags are for manual control.",
    )
    srv.add_argument(
        "--detach", action="store_true",
        help="Start the daemon in the background and return immediately.",
    )
    srv.add_argument(
        "--stop", action="store_true",
        help="Stop a running daemon.",
    )
    srv.add_argument(
        "--status", action="store_true",
        help="Report whether a daemon is running (and for how long).",
    )
    srv.add_argument(
        "--idle", type=float, default=None,
        help="Idle self-exit window in seconds (default: 1800).",
    )
    srv.set_defaults(func=_cmd_serve)

    # ── version ─────────────────────────────────────────────────────────
    v = sub.add_parser("version", help="Print the engram version and exit.")
    v.set_defaults(func=_cmd_version)

    return p


def _cmd_install(args: argparse.Namespace) -> int:
    from . import install as _inst

    agent = args.agent
    if getattr(args, "print_snippet", False) or getattr(args, "print_global_snippet", False):
        if agent == "all":
            print("--print-instructions-snippet needs a single --agent (not 'all').",
                  file=sys.stderr)
            return 2
        return _inst.print_instructions_snippet(
            agent=agent,
            kind=getattr(args, "snippet_kind", None),
        )
    if args.check:
        if agent == "all":
            rc = 0
            for a in _inst_list_agents():
                print(f"── {a} ──")
                rc |= _inst.check(agent=a, target=args.target)
                print()
            return rc
        return _inst.check(agent=agent, target=args.target)
    if args.remove:
        if agent == "all":
            rc = 0
            for a in _inst_list_agents():
                rc |= _inst.remove(agent=a, target=args.target)
            return rc
        return _inst.remove(agent=agent, target=args.target)

    if args.no_skill and args.no_mcp:
        # Don't silently no-op — a user passing both flags almost certainly
        # mis-typed and would otherwise see "OK" with nothing actually done.
        print(
            "--no-skill and --no-mcp together would skip both halves of "
            "the install — refusing to no-op.",
            file=sys.stderr,
        )
        return 2
    with_skill = None if not args.no_skill else False
    with_mcp   = None if not args.no_mcp   else False
    # Hook is opt-in: only when --hook is passed.  None ⇒ install() leaves it off.
    with_hook  = True if getattr(args, "hook", False) else None
    if agent == "all":
        return _inst.install_all(dev=args.dev, force=args.force, with_hook=with_hook)
    return _inst.install(
        agent=agent,
        target=args.target,
        dev=args.dev,
        force=args.force,
        with_skill=with_skill,
        with_mcp=with_mcp,
        with_hook=with_hook,
    )


def _cmd_install_skill(args: argparse.Namespace) -> int:
    """v0.3 path — Claude Code file-skill only, no MCP."""
    from . import install as _inst

    if getattr(args, "print_global_snippet", False):
        return _inst.print_instructions_snippet(agent="claude-code")
    if args.check:
        return _inst.check(agent="claude-code", target=args.target)
    if args.remove:
        return _inst.remove(agent="claude-code", target=args.target)
    return _inst.install(
        agent="claude-code",
        target=args.target,
        dev=args.dev,
        force=args.force,
        with_skill=True,
        with_mcp=False,
    )


def _cmd_agents(_args: argparse.Namespace) -> int:
    # iter_profiles() re-reads env vars on each call so per-agent home_dir
    # values reflect the live environment (and so $CLAUDE_HOME etc. set in
    # this shell session take effect).  Don't iterate BUILTIN_PROFILES here —
    # that snapshot is frozen at import time.
    from .agents import iter_profiles
    for p in iter_profiles():
        kinds = "+".join(p.install_kinds)
        print(f"  {p.name:<14} {p.display:<14} home={p.home_dir}  kinds={kinds}")
    return 0


def _cmd_setup(args: argparse.Namespace) -> int:
    from . import setup_wizard

    return setup_wizard.run(
        agent=args.agent,
        assume_yes=args.assume_yes,
        dev=args.dev,
        force=args.force,
    )


def _cmd_init(args: argparse.Namespace) -> int:
    from . import init_project

    return init_project.init(
        agent=args.agent,
        project_root=args.project_root,
        with_snippet=not args.no_snippet,
        with_gitignore=not args.no_gitignore,
        force=args.force,
    )


def _cmd_uninstall(args: argparse.Namespace) -> int:
    from . import uninstall_engram

    if args.purge and args.keep_data:
        print(
            "--purge and --keep-data are contradictory — pick one.",
            file=sys.stderr,
        )
        return 2
    # purge tri-state: True (--purge) / False (--keep-data) / None (ask).
    if args.purge:
        purge: Optional[bool] = True
    elif args.keep_data:
        purge = False
    else:
        purge = None
    return uninstall_engram.run(
        scope=args.scope,
        agent=args.agent,
        project_root=args.project_root,
        purge=purge,
        assume_yes=args.assume_yes,
    )


def _cmd_doctor(args: argparse.Namespace) -> int:
    from . import doctor

    return doctor.run(verbose=args.verbose, as_json=args.as_json)


def _cmd_warmup(args: argparse.Namespace) -> int:
    from .embedder import warmup

    try:
        result = warmup(spec=args.spec)
    except Exception as e:
        print(f"warmup failed: {e}", file=sys.stderr)
        return 1
    print(
        f"warmup OK — scheme={result['scheme']!r} "
        f"dim={result['dim']} elapsed={result['elapsed_seconds']}s"
    )
    return 0


def _slugify(text: str, maxlen: int = 40) -> str:
    """Make a kebab-case slug from free text; fall back to a short content
    hash when the text has no ASCII words (e.g. pure Chinese)."""
    import hashlib
    import re
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    slug = "-".join(words)[:maxlen].strip("-")
    if not slug:
        slug = "note-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return slug


def _cmd_remember(args: argparse.Namespace) -> int:
    """The simple path: one positional arg, everything else defaulted.

    `engram remember "<text>"` → a personal (user-tier) memory, importance
    0.5, name auto-generated.  `--project` scopes it to the cwd's project;
    `--pin` makes it an always-on standing directive.
    """
    from . import MemoryManager

    text = args.text.strip()
    if not text:
        print("remember: nothing to save (empty text).", file=sys.stderr)
        return 2
    mem_type = "project" if args.project else "user"
    name = args.name or _slugify(text)
    tags = ["pin"] if args.pin else []
    try:
        with MemoryManager() as mgr:
            res = mgr.save(
                type=mem_type, name=name, description=text,
                tags=tags, importance=0.5, force=args.force,
            )
    except Exception as e:
        print(f"remember failed: {e}", file=sys.stderr)
        return 1
    if res.status == "merge_suggestion" and res.duplicate_of is not None:
        dup = res.duplicate_of
        print(f"a near-duplicate already exists: [{dup.id}] {dup.name} — {dup.description}")
        print("re-run with --force to save anyway, or rephrase.")
        return 2
    scope = "this project" if args.project else "all projects"
    pinned = ", pinned (always-on)" if args.pin else ""
    print(f"remembered: [{res.id}] {name}  ({mem_type}, {scope}{pinned})")
    return 0


def _emit_hits(hits, *, as_json: bool = False, with_content: bool = False) -> None:
    """Compact one-line-per-hit output shared by recall / list / forget."""
    if as_json:
        import json
        print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
        return
    if not hits:
        print("(no hits)")
        return
    for h in hits:
        desc = h.description.replace("\n", " ").replace("\r", " ")
        print(f"{h.id:>5} | {h.tier[:1]} | {h.type:<9} | {h.name:<24} | "
              f"d={h.distance:.3f} | {desc}")
        if with_content and h.content:
            for ln in h.content.splitlines():
                print(f"      | {ln}")


def _cmd_recall(args: argparse.Namespace) -> int:
    from . import MemoryManager
    try:
        with MemoryManager() as mgr:
            hits = mgr.recall(
                args.query, k=args.k,
                types=[args.type] if args.type else None,
                tiers=[args.tier] if args.tier else None,
                with_content=args.with_content,
            )
    except Exception as e:
        print(f"recall failed: {e}", file=sys.stderr)
        return 1
    _emit_hits(hits, as_json=args.as_json, with_content=args.with_content)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from . import MemoryManager
    try:
        with MemoryManager() as mgr:
            hits = mgr.list(type=args.type, tier=args.tier, limit=args.limit)
    except Exception as e:
        print(f"list failed: {e}", file=sys.stderr)
        return 1
    _emit_hits(hits, as_json=args.as_json)
    return 0


def _cmd_forget(args: argparse.Namespace) -> int:
    from . import MemoryManager
    try:
        with MemoryManager() as mgr:
            if args.dry_run:
                targets = mgr.find_for_forget(
                    name=args.name, id=args.id, type=args.type,
                    tier=args.tier, older_than_days=args.older_than,
                )
                print(f"would delete {len(targets)}:")
                _emit_hits(targets)
                return 0
            n = mgr.forget(
                name=args.name, id=args.id, type=args.type,
                tier=args.tier, older_than_days=args.older_than,
            )
    except ValueError as e:  # e.g. no selector given
        print(f"forget: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"forget failed: {e}", file=sys.stderr)
        return 1
    print(f"forgot {n} memor{'y' if n == 1 else 'ies'}")
    return 0


def _cmd_directives(args: argparse.Namespace) -> int:
    """Print standing directives for injection.

    This is the command wired into each agent's UserPromptSubmit hook, so it
    runs on *every* turn.  It must be robust and quiet: on any error (bad
    embedder config, missing store, …) it prints a short note to stderr and
    exits 0 with no stdout — never block or pollute the turn.  Empty result =
    no output, so nothing is injected when there are no directives.
    (UTF-8 stdout is already forced by ``main()`` for every subcommand.)
    """
    try:
        from . import MemoryManager

        with MemoryManager() as mgr:
            rows = mgr.directives()
    except Exception as e:  # never let the per-turn hook fail the turn
        print(f"engram directives: {e}", file=sys.stderr)
        return 0

    if getattr(args, "as_json", False):
        import json
        print(json.dumps([h.to_dict() for h in rows], ensure_ascii=False, indent=2))
        return 0

    if not rows:
        return 0  # nothing to inject
    print("# Standing directives — always apply, regardless of the current task:")
    for h in rows:
        desc = h.description.replace("\n", " ").replace("\r", " ")
        print(f"  - {desc}")
    return 0


def _cmd_hook(args: argparse.Namespace) -> int:
    from . import install as _inst

    enable = not args.disable
    if args.agent == "all":
        return _inst.hook_all(enable=enable)
    return _inst.hook(agent=args.agent, enable=enable)


def _cmd_serve(args: argparse.Namespace) -> int:
    """Manual control of the warm embedding daemon (``engram serve``).

    The daemon is normally **self-managed** — the ``daemon:`` embedder
    backend spawns it lazily and it idle-exits on its own (see
    ``DAEMON_DESIGN.md``).  These flags exist for debugging / explicit
    lifecycle control:

    * ``--status``  : print whether one is running.
    * ``--stop``    : stop a running one.
    * ``--detach``  : start one in the background.
    * (no flag)     : run one in the **foreground** (Ctrl-C to quit) — handy
                      for watching it work while developing.
    """
    from . import serve as _serve

    if args.status:
        st = _serve.status()
        if not st.get("running"):
            extra = " (stale serve.json present)" if st.get("stale") else ""
            print(f"daemon: not running{extra}")
            return 0
        print(
            f"daemon: running — pid={st.get('pid')} port={st.get('port')} "
            f"embedder={st.get('embedder')!r} dim={st.get('dim')} "
            f"uptime={st.get('uptime_s')}s served={st.get('served')}"
        )
        return 0

    if args.stop:
        stopped = _serve.stop()
        print("daemon: stopped" if stopped else "daemon: nothing to stop")
        return 0

    idle = args.idle if args.idle is not None else _serve.DEFAULT_IDLE_SECONDS
    if args.detach:
        pid = _serve.spawn_detached(idle_seconds=idle)
        print(f"daemon: spawned in background (pid={pid}); warming up…")
        return 0

    # Foreground (debug). resolve_underlying_spec maps ENGRAM_EMBEDDER's
    # daemon:* (or anything) to a real backend so we never recurse.
    print("daemon: running in foreground (Ctrl-C to stop)…", file=sys.stderr)
    try:
        return _serve.run_foreground(idle_seconds=idle)
    except KeyboardInterrupt:
        return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    print(_version())
    return 0


def _inst_list_agents() -> list[str]:
    from .agents import list_agents
    return list_agents()


def _force_utf8_io() -> None:
    """Force UTF-8 stdout/stderr so non-ASCII output survives Windows
    consoles / hook pipes that default to cp936/GBK.  Best-effort."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _force_utf8_io()
    parser = _build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
