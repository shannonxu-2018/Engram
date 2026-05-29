"""``engram doctor`` — one-shot environment health check.

Engram has half a dozen moving parts (Python package, PistaDB native
library, embedder backend, per-agent skill files, per-agent MCP
registrations, two-tier .pst stores).  When recall silently doesn't
work, the user has to root-cause across all of them.  ``doctor``
inspects each in turn and produces a single coloured-ish report plus
copy-pasteable fix commands.

Each check is an independent function returning a :class:`CheckResult`.
The driver doesn't know what any specific check does — it just iterates,
prints, and collects exit-code signal.  Add a new check by adding a
function to :data:`_CHECKS`.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ── Result shape ────────────────────────────────────────────────────────────

OK   = "ok"
WARN = "warn"
ERR  = "err"


@dataclass
class CheckResult:
    """One health-check outcome.

    Attributes
    ----------
    name
        Short label printed at the start of the line (≤ 24 chars).
    status
        One of :data:`OK` / :data:`WARN` / :data:`ERR`.
    message
        One-line human-readable explanation.
    fix_hint
        Optional copy-pasteable command that addresses the issue.  Doctor
        prints these at the bottom of the report as a "quick fixes"
        block so the user can act without re-reading the docs.
    details
        Optional extra dict surfaced only in ``--verbose`` / ``--json``.
    """
    name: str
    status: str
    message: str
    fix_hint: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Individual checks ──────────────────────────────────────────────────────

def _check_package_version() -> CheckResult:
    try:
        from importlib.metadata import version
        v = version("engram")
        return CheckResult(
            name="package",
            status=OK,
            message=f"engram {v} installed and importable",
            details={"version": v},
        )
    except Exception as e:
        return CheckResult(
            name="package",
            status=ERR,
            message=f"could not read engram version: {e}",
            fix_hint="pip install --force-reinstall engram",
        )


def _check_pistadb_lib() -> CheckResult:
    """Verify the PistaDB native library loads.

    On Win x64 / Linux x64 / macOS arm64 the .dll/.so/.dylib ships in the
    wheel; on other platforms the user must set ``PISTADB_LIB_PATH``.

    The ctypes load happens lazily inside PistaDB — touch ``_find_lib()``
    explicitly so a broken / missing binary surfaces here instead of at
    the next ``MemoryManager`` open.
    """
    try:
        from pistadb import _find_lib
        handle = _find_lib()
        return CheckResult(
            name="pistadb",
            status=OK,
            message=f"native lib loaded ({getattr(handle, '_name', '?')})",
        )
    except Exception as e:
        return CheckResult(
            name="pistadb",
            status=ERR,
            message=f"PistaDB native lib failed to load: {e}",
            fix_hint=(
                "Set PISTADB_LIB_PATH=/abs/path/to/lib pointing at a "
                "matching binary, or rebuild from "
                "https://github.com/shannonxu-2018/PistaDB."
            ),
        )


def _check_embedder() -> CheckResult:
    """Verify the current ``ENGRAM_EMBEDDER`` spec parses + constructs.

    Doesn't actually call ``embed()`` — that would download e5 / hit the
    API — but does run the factory, which is where most config typos
    surface.
    """
    spec = os.environ.get("ENGRAM_EMBEDDER", "")
    from .embedder import build_embedder, parse_spec
    try:
        parsed = parse_spec(spec)
        emb = build_embedder(spec)
        return CheckResult(
            name="embedder",
            status=OK,
            message=(
                f"spec={spec!r}  → scheme={parsed.scheme!r} dim={emb.dim}"
            ),
            details={"scheme": parsed.scheme, "dim": emb.dim,
                     "params": dict(parsed.params)},
        )
    except Exception as e:
        return CheckResult(
            name="embedder",
            status=ERR,
            message=f"could not build embedder from ENGRAM_EMBEDDER={spec!r}: {e}",
            fix_hint=(
                "Pick a working spec, e.g. ENGRAM_EMBEDDER='' (local e5) "
                "or 'ollama:nomic-embed-text'."
            ),
        )


def _check_e5_model_cache() -> CheckResult:
    """For the local backend: is the e5 model already in HF cache?

    Cold-starts can take 30 s while sentence-transformers downloads
    471 MB.  Flag it as a warning so the user can run ``engram warmup``
    before hitting a real call.
    """
    spec = os.environ.get("ENGRAM_EMBEDDER", "").strip().lower()
    # Only meaningful for the local backend.
    if spec and spec != "local" and not spec.startswith("local?"):
        return CheckResult(
            name="e5 model",
            status=OK,
            message=f"skipped (ENGRAM_EMBEDDER={spec!r}, not using local e5)",
        )

    hf_home = (
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or str(Path.home() / ".cache" / "huggingface")
    )
    snapshots_root = Path(hf_home) / "hub" / "models--intfloat--multilingual-e5-small"
    custom_path = os.environ.get("ENGRAM_E5_MODEL_PATH")
    if custom_path:
        p = Path(custom_path).expanduser()
        if p.is_dir():
            return CheckResult(
                name="e5 model",
                status=OK,
                message=f"using custom snapshot at {p}",
            )
        return CheckResult(
            name="e5 model",
            status=ERR,
            message=f"ENGRAM_E5_MODEL_PATH={custom_path!r} is not a directory",
            fix_hint=(
                "Either unset ENGRAM_E5_MODEL_PATH to use the HuggingFace "
                "cache, or point it at a valid e5 snapshot directory."
            ),
        )

    if snapshots_root.is_dir():
        return CheckResult(
            name="e5 model",
            status=OK,
            message=f"cached at {snapshots_root}",
        )
    return CheckResult(
        name="e5 model",
        status=WARN,
        message=(
            "multilingual-e5-small not in HF cache — first recall will "
            "download ~471 MB"
        ),
        fix_hint="engram warmup",
    )


def _check_mcp_command() -> CheckResult:
    """Is the ``engram-mcp`` console script on PATH?"""
    path = shutil.which("engram-mcp")
    if path:
        return CheckResult(
            name="engram-mcp",
            status=OK,
            message=f"on PATH at {path}",
            details={"path": path},
        )
    return CheckResult(
        name="engram-mcp",
        status=WARN,
        message=(
            "engram-mcp not on PATH — MCP-based agents (OpenCode/Codex/"
            "Claude Code MCP) won't be able to launch the server"
        ),
        fix_hint=(
            "Make sure the venv that has engram installed is active "
            "before launching the host, or pass the absolute path of "
            "engram-mcp to the host's MCP config."
        ),
    )


def _check_agents() -> List[CheckResult]:
    """Per-agent skill + MCP registration status."""
    from .agents import iter_profiles
    from .install import MCP_SERVER_NAME

    out: List[CheckResult] = []
    for profile in iter_profiles():
        if "skill" in profile.install_kinds:
            dst = profile.skill_dir
            if dst is None:
                continue
            if dst.is_symlink():
                out.append(CheckResult(
                    name=f"{profile.name} skill",
                    status=OK,
                    message=f"SYMLINK at {dst} → {dst.resolve()}",
                ))
            elif dst.is_dir():
                out.append(CheckResult(
                    name=f"{profile.name} skill",
                    status=OK,
                    message=f"COPY at {dst}",
                ))
            else:
                out.append(CheckResult(
                    name=f"{profile.name} skill",
                    status=WARN,
                    message=f"not installed at {dst}",
                    fix_hint=f"engram install --agent {profile.name}",
                ))

        if "mcp" in profile.install_kinds:
            cfg = profile.mcp_config_path
            registered = False
            if cfg is not None and cfg.is_file():
                if profile.mcp_config_kind == "claude_servers":
                    try:
                        with cfg.open("r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            servers = data.get("mcpServers")
                            if isinstance(servers, dict):
                                registered = MCP_SERVER_NAME in servers
                    except json.JSONDecodeError:
                        pass
                elif profile.mcp_config_kind == "codex_toml":
                    text = cfg.read_text(encoding="utf-8", errors="replace")
                    registered = f"[mcp_servers.{MCP_SERVER_NAME}]" in text
            if registered:
                out.append(CheckResult(
                    name=f"{profile.name} mcp",
                    status=OK,
                    message=f"registered in {cfg}",
                ))
            else:
                out.append(CheckResult(
                    name=f"{profile.name} mcp",
                    status=WARN,
                    message=f"not registered (would live in {cfg})",
                    fix_hint=f"engram install --agent {profile.name}",
                ))
    return out


def _check_tier_files() -> List[CheckResult]:
    """Sanity-check that any existing .pst has its meta.json + cache sibling.

    A .pst without its meta.json (or vice versa) is corrupt — engram refuses
    to start with one such tier, which can puzzle users who see "memory
    suddenly stopped working" after a partial sync.
    """
    from .store import resolve_tiers

    out: List[CheckResult] = []
    tiers = resolve_tiers()
    for tname, paths in tiers.items():
        pst = paths.pst
        meta = Path(str(pst) + ".meta.json")
        cache = paths.cache
        if not pst.exists() and not meta.exists():
            out.append(CheckResult(
                name=f"tier:{tname}",
                status=OK,
                message=f"fresh — no .pst yet at {paths.pst}",
            ))
            continue
        if pst.exists() and meta.exists():
            out.append(CheckResult(
                name=f"tier:{tname}",
                status=OK,
                message=(
                    f"present ({_human_size(pst.stat().st_size)} pst + "
                    f"{_human_size(meta.stat().st_size)} meta + "
                    f"{_human_size(cache.stat().st_size) if cache.exists() else '0 B'} cache)"
                ),
            ))
            continue
        # Exactly one of the two present → corrupt half-state.
        missing = "meta.json sidecar" if pst.exists() else ".pst"
        out.append(CheckResult(
            name=f"tier:{tname}",
            status=ERR,
            message=f"inconsistent — {missing} missing at {paths.pst}",
            fix_hint=(
                "Restore from backup, or delete both files to start over:\n"
                f"        rm {pst} {meta} {cache}"
            ),
        ))
    return out


# ── Driver ────────────────────────────────────────────────────────────────

# Each entry is a callable returning either a CheckResult or a list of
# CheckResults.  Ordering matters: cheapest / most-fundamental first so the
# user spots a broken install before doctor wastes time on per-agent probes.
_CHECKS: List[Callable[[], Any]] = [
    _check_package_version,
    _check_pistadb_lib,
    _check_embedder,
    _check_e5_model_cache,
    _check_mcp_command,
    _check_agents,
    _check_tier_files,
]


def run(*, verbose: bool = False, as_json: bool = False) -> int:
    results: List[CheckResult] = []
    for check in _CHECKS:
        try:
            r = check()
        except Exception as e:
            r = CheckResult(
                name=check.__name__,
                status=ERR,
                message=f"check raised {type(e).__name__}: {e}",
            )
        if isinstance(r, list):
            results.extend(r)
        else:
            results.append(r)

    if as_json:
        json.dump(
            {"results": [r.to_dict() for r in results]},
            sys.stdout, indent=2, ensure_ascii=False,
        )
        sys.stdout.write("\n")
    else:
        _print_human(results, verbose=verbose)

    rc = 0
    if any(r.status == ERR for r in results):
        rc = 2
    elif any(r.status == WARN for r in results):
        rc = 1
    return rc


def _print_human(results: List[CheckResult], *, verbose: bool) -> None:
    glyph = {OK: "OK  ", WARN: "WARN", ERR: "FAIL"}
    counts = {OK: 0, WARN: 0, ERR: 0}
    for r in results:
        counts[r.status] += 1
        print(f"  [{glyph[r.status]}] {r.name:<18} {r.message}")
        if verbose and r.details:
            for k, v in r.details.items():
                print(f"                       {k} = {v}")

    print()
    print(
        f"{counts[OK]} OK, {counts[WARN]} warning"
        + ("" if counts[WARN] == 1 else "s")
        + f", {counts[ERR]} error"
        + ("" if counts[ERR] == 1 else "s")
        + "."
    )

    fixes = [r for r in results if r.fix_hint and r.status != OK]
    if fixes:
        print()
        print("Quick fixes:")
        for r in fixes:
            print(f"  ({r.name:<18}) {r.fix_hint}")


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


__all__ = ["CheckResult", "run", "OK", "WARN", "ERR"]
