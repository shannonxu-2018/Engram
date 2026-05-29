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
* **CLI**: `engram install --agent {claude-code,opencode,codex,all}`;
  the old `engram install-skill` is preserved as a Claude-Code-skill-only
  alias.
* **Home-dir resolution in `store.py`**: now checks `ENGRAM_HOME` →
  `CLAUDE_HOME` → `~/.claude`. Keep `~/.claude/engram/` semantics
  unchanged — the global store still lives there by default, even when
  the active agent is OpenCode or Codex.
* **Tests**: `python tests/agents_test.py` covers profile resolution,
  env-var precedence, JSON merger (Claude Code & OpenCode shape) and
  TOML merger (Codex shape).
