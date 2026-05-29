"""``engram setup`` — one interactive command that wires Engram end-to-end.

The four-step recipe in the README (``install`` → enable-globally →
``init`` → ``warmup`` → ``doctor``) collapses the most common new-user
surprises, but it still assumes the user *knows* to run all of them in
order.  A user who "only uses an AI-agent CLI" and has no programming
background won't.  ``setup`` walks them through it with plain-language
yes/no prompts and sensible, detected defaults — every individual step is
just a call into the same library functions the standalone subcommands
use, so there is no new behaviour here, only orchestration.

Design notes:

* **Thin.** This module owns *no* install logic.  It calls
  :func:`engram.install.install` / :func:`install_all`,
  :func:`engram.init_project.init`, :func:`engram.embedder.warmup` and
  :func:`engram.doctor.run`.  The global-enable step reuses
  :func:`engram.init_project._wire_instructions_file` (pointed at the
  agent's *home* dir) so snippet-marker idempotency is shared too.
* **Safe in a pipe.** If stdin isn't a TTY we don't block on ``input()``
  forever — the caller must pass ``--yes`` to accept all defaults, else
  we explain and bail.
* **Idempotent.** Re-running ``setup`` is harmless: install is
  force-free (skips existing), snippet appends are marker-guarded, and
  warmup just re-probes a warm model.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .agents import get_profile, iter_profiles, list_agents


# ── Prompt primitives ────────────────────────────────────────────────────────

def _interactive() -> bool:
    """True only when we can actually read an answer from the user.

    ``input()`` against a closed / piped stdin raises ``EOFError`` (or
    blocks under some shells) — guarding on ``isatty`` lets us fall back
    to ``--yes`` semantics instead of hanging a non-interactive caller.
    """
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def _ask_yes_no(question: str, *, default: bool, assume_yes: bool) -> bool:
    """Prompt ``question`` and return the boolean answer.

    ``default`` is what an empty line (just Enter) means and what we use
    in ``assume_yes`` / non-interactive mode.
    """
    suffix = "[Y/n]" if default else "[y/N]"
    if assume_yes or not _interactive():
        print(f"{question} {suffix} {'y' if default else 'n'}")
        return default
    while True:
        try:
            raw = input(f"{question} {suffix} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  please answer y or n.")


def _detected(profile) -> bool:
    """Heuristic: has this agent ever run on this machine?

    We treat the agent's home dir existing as the signal.  ``~/.claude``
    exists once Claude Code has launched, ``~/.codex`` once Codex has,
    etc.  Used only to pre-select sensible defaults — never to *block* a
    choice (the user can still install for an agent we didn't detect).
    """
    try:
        return profile.home_dir.exists()
    except Exception:
        return False


def _select_agents(preselect: Optional[str], assume_yes: bool) -> List[str]:
    """Decide which agents to install for.

    Priority: an explicit ``--agent`` wins; otherwise offer a numbered
    list with detected agents pre-ticked.  ``all`` selects everything.
    Empty input / non-interactive accepts the detected default.
    """
    if preselect:
        return list_agents() if preselect == "all" else [preselect]

    profiles = iter_profiles()
    detected = [p.name for p in profiles if _detected(p)]
    # Fall back to Claude Code if we couldn't detect anything — it's the
    # most common host and the one the rest of the docs assume.
    default = detected or ["claude-code"]

    print("Which agent(s) should use Engram?")
    for i, p in enumerate(profiles, 1):
        mark = " (detected)" if p.name in detected else ""
        kinds = "+".join(p.install_kinds)
        print(f"  {i}. {p.display:<12} [{kinds}]{mark}")
    print("  a. all of them")
    print(f"(press Enter for the default: {', '.join(default)})")

    if assume_yes or not _interactive():
        print(f"> {', '.join(default)}")
        return default

    names = [p.name for p in profiles]
    while True:
        try:
            raw = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if raw == "":
            return default
        if raw in ("a", "all"):
            return list(names)
        chosen: List[str] = []
        ok = True
        for tok in raw.replace(",", " ").split():
            if tok.isdigit() and 1 <= int(tok) <= len(names):
                chosen.append(names[int(tok) - 1])
            elif tok in names:
                chosen.append(tok)
            else:
                print(f"  '{tok}' isn't a valid choice — pick numbers, names, or 'a'.")
                ok = False
                break
        if ok and chosen:
            # De-dup while preserving order.
            seen: set = set()
            return [n for n in chosen if not (n in seen or seen.add(n))]


# ── Steps ─────────────────────────────────────────────────────────────────────

def _step_install(agents: Sequence[str], *, dev: bool, force: bool) -> int:
    from . import install as _inst

    print()
    print(f"── Step 1/4: install the skill + MCP for {', '.join(agents)} ──")
    rc = 0
    for name in agents:
        rc |= _inst.install(agent=name, dev=dev, force=force)
        print()
    return rc


def _step_enable_global(agents: Sequence[str], *, force: bool) -> int:
    """Append the per-agent snippet to each agent's *global* instructions file.

    Reuses :func:`init_project._wire_instructions_file` pointed at the
    agent's home dir, so the "already wired?" marker check and the atomic
    write are shared with ``engram init`` — no duplicate logic, and a
    second run won't double-append.
    """
    from .init_project import _wire_instructions_file

    print("── Step 2/4: enable Engram for every project (global) ──")
    rc = 0
    for name in agents:
        profile = get_profile(name)
        # root = home_dir → writes home_dir/CLAUDE.md (or AGENTS.md), i.e.
        # the agent's *global* instructions file.
        rc |= _wire_instructions_file(profile.home_dir, profile, force=force)
    return rc


def _step_init_project(agents: Sequence[str]) -> int:
    from . import init_project

    print("── Step 3/4: wire the current project ──")
    root = Path.cwd()
    print(f"  project: {root}")
    rc = 0
    for name in agents:
        rc |= init_project.init(
            agent=name,
            project_root=root,
            with_snippet=True,
            with_gitignore=True,
        )
    return rc


def _step_warmup() -> int:
    from .embedder import warmup

    print("── Step 4/4: pre-load the embedding model ──")
    print("  (local backend downloads ~471 MB on first run — this can take a minute)")
    try:
        result = warmup()
    except Exception as e:
        print(f"  warmup failed: {e}", file=sys.stderr)
        print("  you can retry later with `engram warmup`.", file=sys.stderr)
        return 1
    print(
        f"  warmup OK — scheme={result['scheme']!r} dim={result['dim']} "
        f"elapsed={result['elapsed_seconds']}s"
    )
    return 0


# ── Driver ──────────────────────────────────────────────────────────────────

def run(
    *,
    agent: Optional[str] = None,
    assume_yes: bool = False,
    dev: bool = False,
    force: bool = False,
) -> int:
    """Run the interactive setup wizard.

    Returns ``0`` when setup succeeded (``doctor`` found no hard
    *errors*) and ``2`` when something is genuinely broken.  Note that a
    single-agent setup leaves ``doctor`` *warning* about the agents the
    user didn't pick — that's expected and benign, so we deliberately do
    **not** fail the wizard on warnings.  A failed install/warmup step is
    surfaced but doesn't abort the wizard — we still run ``doctor`` at the
    end so the user gets one consolidated picture with copy-pasteable
    fixes.
    """
    print("=" * 60)
    print(" Engram setup — let's get memory working for your agent.")
    print("=" * 60)

    if not assume_yes and not _interactive():
        print(
            "stdin is not a terminal, so I can't ask you anything.\n"
            "Re-run with --yes to accept all recommended defaults:\n"
            "    engram setup --yes",
            file=sys.stderr,
        )
        return 2

    agents = _select_agents(agent, assume_yes)

    # Step 1 — always (it's the whole point).
    rc = _step_install(agents, dev=dev, force=force)

    # Step 2 — global enable (recommended default yes: it's what makes the
    # agent actually reach for Engram without per-project wiring).
    if _ask_yes_no(
        "Enable Engram for EVERY project (recommended)?",
        default=True,
        assume_yes=assume_yes,
    ):
        rc |= _step_enable_global(agents, force=force)
    print()

    # Step 3 — wire the current project (default no: only meaningful if the
    # user launched setup from inside a project they care about).
    if _ask_yes_no(
        "Also wire the CURRENT directory as a project?",
        default=False,
        assume_yes=assume_yes,
    ):
        rc |= _step_init_project(agents)
    print()

    # Step 4 — warmup (default yes: avoids the silent 30 s first-recall stall).
    if _ask_yes_no(
        "Download the embedding model now so the first recall is fast?",
        default=True,
        assume_yes=assume_yes,
    ):
        _step_warmup()  # advisory — don't fail setup if the download flakes
    print()

    # Closing health check — the single source of truth on whether it worked.
    print("── Health check ──")
    from . import doctor
    doctor_rc = doctor.run(verbose=False, as_json=False)

    print()
    # doctor_rc: 0 = all OK, 1 = warnings only, 2 = hard errors.  We only
    # treat hard errors as a failed setup — warnings here are almost
    # always "you didn't install for agent X", which is fine.
    if doctor_rc < 2:
        print(f"All set for {', '.join(agents)}.")
        print("Your agent will use Engram for memory from now on.")
        print("Try it: say \"remember that I prefer …\" in a session.")
        if doctor_rc == 1:
            print(
                "(The warnings above are for agents you didn't set up — "
                "ignore them unless you want those too.)"
            )
        return 0
    print(
        "Setup hit a real error above (see the 'Quick fixes' block). "
        "Fix it, then re-run `engram doctor` to confirm."
    )
    return 2


__all__ = ["run"]
