# engram — project guide for Claude

This repo *is* Engram: a vector-DB memory system that replaces the
legacy `MEMORY.md` / per-file markdown memory. When you work in this
repo, treat Engram as the source of truth for everything the old
`auto memory` system used to do — do not write new memory files under
`~/.claude/projects/.../memory/`.

## Memory protocol — use Engram, not MD files

After `pip install -e ".[all]"` + `engram install-skill --dev` (run
`./bootstrap.ps1` on Windows or `./bootstrap.sh` on Linux/macOS), the
skill lives at `~/.claude/skills/engram/`. Call its scripts via the
Python on PATH (or the user's chosen interpreter):

* **Directives (do this FIRST, every turn)** — Engram now carries the
  CLAUDE.md role for *always-on global constraints* (e.g. "reply in
  Simplified Chinese in this repo"). At the start of each turn run:

  ```
  python ~/.claude/skills/engram/scripts/directives.py
  ```

  and obey every line it returns. These are memories tagged `pin`;
  semantic `recall` will **not** surface them, so this per-turn step is the
  only thing that makes them fire. It's cheap (no embedding). Pin a new
  constraint with `save.py <type> <name> "<rule>" --tag pin --importance 0.9`.

* **Recall** — when the user references prior conversation, asks about
  themselves, or you're about to give advice that should be informed by
  preference/feedback:

  ```
  python ~/.claude/skills/engram/scripts/recall.py "<query>" --k 5
  ```

  Inject the returned lines as context. Only call `expand.py <id>` if a
  description alone isn't enough.

* **Save** — same triggers as the legacy rules, but use:

  ```
  python ~/.claude/skills/engram/scripts/save.py <type> <name> "<description>" "<content>"
  ```

  If it exits with `MERGE_SUGGESTION`, decide whether to `--update` the
  existing memory or skip.

* **Forget / list**: `forget.py --name <slug>` and `list.py`.

The skill scripts use `import engram` to find the library, so
`pip install -e .` (which `bootstrap.{ps1,sh}` does for you) is the only
prerequisite.

## What changes from the old MD protocol

* Do **not** read `MEMORY.md` — the vector store is authoritative. Use
  `list.py` for an overview.
* Do **not** create `.md` files under `memory/`. The save script writes
  to `.pst` files atomically.
* The four memory types (`user`, `feedback`, `project`, `reference`),
  body structure (`Why:` / `How to apply:`) and the "what NOT to save"
  rules are **unchanged** — see `src/engram/_skill_files/SKILL.md` §5.

## Storage layout

* `~/.claude/engram/global.{pst,pcc}` — `user` + `feedback` memories
  (shared across all projects).
* `<this repo>/.claude/engram/local.{pst,pcc}` — `project` +
  `reference` memories (project-scoped, `.gitignore`d).

## One-time migration (optional)

Importing existing MD memories from the legacy location:

```
python ~/.claude/skills/engram/scripts/migrate_md.py \
  ~/.claude/projects/<project-slug>/memory/
```

Idempotent (uses `--update` semantics).

## Dev notes for this codebase

* **Source layout** (after the v0.3 refactor): `src/engram/` (package)
  and `src/pistadb/` (vendored vector DB). Tests use `sys.path` shims
  pointing at `src/`; everything else relies on `pip install -e .`.
* **Skill source of truth**: `src/engram/_skill_files/`. The pip
  install ships these as package data; `engram install-skill --dev`
  symlinks them into `~/.claude/skills/engram/` so edits in the repo
  propagate live.
* **Keep CLI scripts thin** — they should `import engram` and call into
  `MemoryManager`, never duplicate logic. The library lives in
  `src/engram/{embedder,store,memory,decay,consolidate}.py`.
* **Smoke test**: `ENGRAM_EMBEDDER=hash python tests/smoke_test.py`
  for a no-dep end-to-end check. The `hash` backend is testing-only;
  for real semantic recall use the default local e5 (downloads from HF
  on first use) or set `ENGRAM_EMBEDDER=openai` with an API key.
* **Native libs**: `src/pistadb/{pistadb.dll, libpistadb.so, libpistadb.dylib}`
  cover Win x64 / Linux x64 / macOS arm64. Intel Mac is unsupported.

## v0.4 multi-agent layer (Claude Code / OpenCode / Codex)

Engram now ships with three integration profiles defined in
`src/engram/agents.py`:

| Agent | Integration kind(s) | Instructions file | Skill dir | MCP config |
|-------|---------------------|-------------------|-----------|------------|
| `claude-code` | `skill` + `mcp` | `~/.claude/CLAUDE.md` | `~/.claude/skills/engram/` | `~/.claude/.claude.json` |
| `opencode`    | `skill` + `mcp` | `~/.config/opencode/AGENTS.md` (XDG) or `%APPDATA%\opencode\AGENTS.md` on Win | `~/.config/opencode/skills/engram/` *(also scans `~/.claude/skills/` & `~/.agents/skills/`)* | `opencode.json` |
| `codex`       | `skill` + `mcp` | `~/.codex/AGENTS.md` | `~/.agents/skills/engram/` *(per OpenAI's `.agents/skills` convention)* | `~/.codex/config.toml` |
| `openclaw`    | `skill` only    | `~/.openclaw/AGENTS.md` *(default; not specified in upstream docs)* | `~/.openclaw/skills/engram/` *(also scans `~/.claude/skills/` & `~/.agents/skills/`)* | — *(MCP config not documented at docs.openclaw.ai)* |

* **MCP server** lives in `src/engram/mcp_server.py` and is exposed as
  the `engram-mcp` console script. It binds *one* `MemoryManager` for
  the lifetime of the process — so the e5 embedder loads once, not per
  call. Uses the official `mcp` SDK when installed (`pip install
  "engram[mcp]"`); otherwise a tiny built-in stdio JSON-RPC loop covers
  `initialize` / `tools/list` / `tools/call` so the dep is optional.
* **Installer split**: agent-aware logic in `src/engram/install.py`;
  `src/engram/install_skill.py` is a back-compat shim that delegates
  to it.
* **CLI** has seven top-level subcommands:
  * `engram install --agent {claude-code,opencode,codex,openclaw,all}` —
    user-level skill + MCP install (the v0.3 `install-skill` is a
    Claude-Code-skill-only alias and still works).
  * `engram init [--agent NAME] [--in PATH]` — *project-level* wiring:
    appends the instructions snippet to `CLAUDE.md` / `AGENTS.md`, adds
    `.claude/engram/` to `.gitignore`, creates the local-tier directory.
    Idempotent.  Implementation: `src/engram/init_project.py`.
  * `engram uninstall [--scope {project,global,all}] [--agent NAME]
    [--in PATH] [--purge | --keep-data] [--yes]` — one-click teardown.
    `project` reverses `init` (strips the `CLAUDE.md`/`AGENTS.md` Engram
    block, drops the `.gitignore` line); `global` reverses `install`
    (skill + MCP + the per-agent user-level snippet).  `--purge` also
    deletes the vector store (`global.{pst,pcc}` / local `.claude/engram/`);
    memories are kept by default.  Scope/purge are asked interactively
    when omitted.  Implementation: `src/engram/uninstall_engram.py`.
  * `engram warmup [--spec SPEC]` — preloads the embedder + runs three
    dummy embeds so the first real recall doesn't pay 30 s of cold
    start.  For the local backend, also pre-downloads the e5 model.
    Implementation: `engram.embedder.warmup()`.
  * `engram doctor [--verbose] [--json]` — seven health checks
    (package / pistadb native lib / embedder spec / e5 model cache /
    `engram-mcp` on PATH / each agent's skill+MCP status / `.pst` tier
    integrity).  Returns 0 / 1 / 2 based on the worst result.  Each
    failing check carries a copy-pasteable fix.  Implementation:
    `src/engram/doctor.py`.
  * `engram agents` — list known agents + their current home_dir
    (env-var-aware via `iter_profiles()`).
  * `engram version`.
* **Home-dir resolution in `store.py`**: now checks `ENGRAM_HOME` →
  `CLAUDE_HOME` → `~/.claude`. Keep `~/.claude/engram/` semantics
  unchanged — the global store still lives there by default, even when
  the active agent is OpenCode or Codex.
* **Tests**: `python tests/agents_test.py` covers profile resolution,
  env-var precedence, JSON merger (Claude Code & OpenCode shape) and
  TOML merger (Codex shape).
