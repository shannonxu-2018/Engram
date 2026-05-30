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


def _cmd_directives(args: argparse.Namespace) -> int:
    """Print standing directives for injection.

    This is the command wired into each agent's UserPromptSubmit hook, so it
    runs on *every* turn.  It must be robust and quiet: on any error (bad
    embedder config, missing store, …) it prints a short note to stderr and
    exits 0 with no stdout — never block or pollute the turn.  Empty result =
    no output, so nothing is injected when there are no directives.
    """
    # Hook pipes default to the OS code page on Windows (cp936/GBK), which
    # mangles non-ASCII directive text.  Force UTF-8 so injected Chinese etc.
    # round-trips intact.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
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


def _cmd_version(_args: argparse.Namespace) -> int:
    print(_version())
    return 0


def _inst_list_agents() -> list[str]:
    from .agents import list_agents
    return list_agents()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
