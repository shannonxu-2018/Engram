<div align="center">

# Installation guide

Engram on **Claude Code**, **OpenCode**, and **Codex** — installation, configuration, troubleshooting.

[English](INSTALL.md) · [中文](INSTALL_CN.md) · [Project overview](README.md)

</div>

---

## TL;DR

```bash
pip install git+https://github.com/shannonxu-2018/Engram.git
engram install --agent claude-code   # or:  opencode | codex | openclaw | all
engram init                          # wire CURRENT project (CLAUDE.md / .gitignore / .claude/engram/)
engram warmup                        # pre-load the embedder so the first recall is fast
engram doctor                        # 1-shot health check across package / agents / tiers / embedder
```

`init` / `warmup` / `doctor` are optional but **strongly recommended** —
they collapse the most common new-user surprises (snippet not added,
30 s model download mid-recall, "why is nothing happening?") into
explicit, idempotent steps.  The `--agent` flag picks the integration;
everything below is reference.

The legacy v0.3 form `engram install-skill` still works — it's an alias for `engram install --agent claude-code --no-mcp` (file-skill only, no MCP registration).

## Contents

1. [Requirements](#1-requirements)
2. [Install the package](#2-install-the-package)
3. [Install for one or more agents](#3-install-for-one-or-more-agents)
4. [(Optional) Enable globally](#4-optional-enable-globally)
5. [Verify](#5-verify)
6. [Data locations](#6-data-locations)
7. [Configuration (env vars)](#7-configuration-env-vars)
8. [Uninstall](#8-uninstall)
9. [Developing](#9-developing-working-in-this-repo)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Requirements

- **Python** ≥ 3.10
- **Network**: needed on first `recall` / `save` to download
  `intfloat/multilingual-e5-small` (~471 MB) from HuggingFace. Fully
  offline afterwards.
- **OS / arch**:

  | Platform | Native lib shipped | Supported |
  |----------|--------------------|-----------|
  | Windows x64              | `pistadb.dll`        | ✅ |
  | Linux x64                | `libpistadb.so`      | ✅ |
  | macOS Apple Silicon (arm64) | `libpistadb.dylib` | ✅ |
  | macOS Intel (x86_64)     | —                    | ❌ unsupported |
  | Linux aarch64 / other    | —                    | ❌ unsupported |

  Other platforms: build your own PistaDB binary from
  [shannonxu-2018/PistaDB](https://github.com/shannonxu-2018/PistaDB) and
  point `PISTADB_LIB_PATH=/abs/path/to/lib` at it.

---

## 2. Install the package

Private git repo — pick ssh or https:

```bash
# HTTPS (works without an SSH key)
pip install git+https://github.com/shannonxu-2018/Engram.git

# SSH (use this if you have a GitHub SSH key configured)
pip install git+ssh://git@github.com/shannonxu-2018/Engram.git

# Pinned version / branch / tag
pip install "engram @ git+https://github.com/shannonxu-2018/Engram.git@v0.4.0"
```

### Optional embedder backends

```bash
pip install "engram[local]"   # sentence-transformers, local e5 (default, recommended)
pip install "engram[openai]"  # OpenAI text-embedding-3-small/large
pip install "engram[eval]"    # tiktoken — for running the benchmark
pip install "engram[mcp]"     # official mcp SDK — for spec-perfect engram-mcp framing
pip install "engram[all]"     # all of the above
```

> The `mcp` extra is **optional**. `engram-mcp` ships with a built-in
> stdio JSON-RPC loop that implements just enough of MCP for Claude
> Code, OpenCode, and Codex to use it. Install the extra only if you
> want the official SDK's spec-perfect framing (e.g. for non-Claude
> hosts that require strict transport conformance). OpenClaw goes
> through the skill path and doesn't use MCP, so this extra is a no-op
> for that agent.

Without extras, the package still imports — but only the built-in `hash`
embedder works (smoke-test only, no semantic understanding).

---

## 3. Install for one or more agents

```bash
# Claude Code: file-based skill + MCP server registration
engram install --agent claude-code

# OpenCode: file-based skill + MCP server registration
engram install --agent opencode

# Codex: file-based skill + MCP server registration
engram install --agent codex

# OpenClaw: file-based skill only (MCP not documented upstream)
engram install --agent openclaw

# Everything in one shot
engram install --agent all
```

`engram install --agent claude-code` does two things:

1. Creates `~/.claude/skills/engram/` and copies `SKILL.md` +
   `scripts/*.py` from the pip package (file-skill path — same as
   v0.3's `install-skill`).
2. Registers the `engram-mcp` command under `mcpServers.engram` in
   `~/.claude.json` (MCP path — new in v0.4).

`engram install --agent codex` does the same two things using Codex's
own conventions:

1. Installs the skill at `~/.agents/skills/engram/` (per OpenAI's Codex
   skills docs — Codex scans `$HOME/.agents/skills/` for user-level
   skills with a `SKILL.md` manifest).
2. Registers `engram-mcp` under `[mcp_servers.engram]` in
   `~/.codex/config.toml`.

`engram install --agent opencode` is symmetric:

1. Installs the skill at `~/.config/opencode/skills/engram/` (or
   `%APPDATA%\opencode\skills\engram\` on Windows). OpenCode also scans
   `~/.claude/skills/` and `~/.agents/skills/`, so if you already
   installed for Claude Code or Codex, OpenCode finds *those* copies
   automatically — the dedicated `--agent opencode` install just gives
   you a clean per-agent footprint for uninstall.
2. Registers `engram-mcp` under `mcpServers.engram` in
   `~/.config/opencode/opencode.json`.

`engram install --agent openclaw` is **skill-only** (per
[`docs.openclaw.ai/tools/skills`](https://docs.openclaw.ai/tools/skills)):

1. Installs the skill at `~/.openclaw/skills/engram/`. Like OpenCode,
   OpenClaw also scans `~/.claude/skills/` and `~/.agents/skills/`, so
   prior installs are picked up cross-ecosystem.
2. *No MCP step.* The OpenClaw skills doc doesn't specify MCP server
   config; we don't guess. If you wire MCP up manually, point it at the
   `engram-mcp` binary that ships with the package.

Either half can be skipped per-agent with `--no-skill` / `--no-mcp`
(no-op for skill-only agents like OpenClaw).

Flags:

| Flag | Effect |
|------|--------|
| `--agent NAME` | Target agent: `claude-code` (default), `opencode`, `codex`, `openclaw`, or `all`. |
| `--dev` | Symlink the skill files instead of copying (needs Windows Developer Mode). Only meaningful for agents with a file-based skill. |
| `--target DIR` | Override skill target dir. |
| `--force` | Overwrite an existing install. |
| `--remove` | Uninstall for the chosen agent (file-skill + MCP registration). |
| `--check` | Report status without writing. |
| `--no-skill` | MCP-only install (skip the file-skill step). |
| `--no-mcp` | File-skill only (skip MCP registration — v0.3 behaviour). |
| `--print-instructions-snippet` | Print the markdown block to append to the agent's `CLAUDE.md` / `AGENTS.md`. |
| `--snippet-kind {skill,mcp}` | Override the snippet's framing. Default = the agent's primary integration (e.g. `skill` for Claude Code & Codex, `mcp` for OpenCode). |

The v0.3 form is preserved:

```bash
engram install-skill          # alias of `install --agent claude-code --no-mcp`
engram install-skill --check  # same as `install --agent claude-code --check`
```

List the supported agents and where their config lives:

```bash
engram agents
```

---

## 3b. Per-project wiring & health (`init`, `warmup`, `doctor`)

After installing the skill/MCP, three follow-up commands handle the
parts most new users miss:

```bash
engram init                                     # current project
engram init --agent opencode                    # per-agent variant
engram init --agent all --in <other-project>    # all four agents into a different repo
```

`init` is idempotent — re-running it after a manual edit of `CLAUDE.md`
will not duplicate the snippet (it scans for the marker first).  Pass
`--force` to re-append after you deleted the block manually.

```bash
engram warmup                                                # uses ENGRAM_EMBEDDER
engram warmup --spec 'openai:text-embedding-3-large'         # one-off probe
```

`warmup` loads the embedder, runs three dummy embeds, and reports
elapsed time.  For the local backend, the first call also downloads
the ~471 MB `multilingual-e5-small` model from HuggingFace.

```bash
engram doctor                # human-readable report
engram doctor --verbose      # +per-check details
engram doctor --json         # machine-readable
```

`doctor` returns exit code `0` when everything is OK, `1` if there are
warnings (e.g. some agents not installed — usually fine), and `2` if
something is actually broken.  Every non-OK line carries a
copy-pasteable `fix command` printed at the bottom of the report.

---

## 4. (Optional) Enable globally

By default the agent uses Engram only when a project's instructions
file (e.g. `CLAUDE.md` / `AGENTS.md`) mentions it. To make **every**
new session prefer Engram, append the per-agent snippet to the
global instructions file:

```bash
# Claude Code
engram install --agent claude-code --print-instructions-snippet \
    >> ~/.claude/CLAUDE.md

# OpenCode
engram install --agent opencode --print-instructions-snippet \
    >> ~/.config/opencode/AGENTS.md

# Codex
engram install --agent codex --print-instructions-snippet \
    >> ~/.codex/AGENTS.md

# OpenClaw
engram install --agent openclaw --print-instructions-snippet \
    >> ~/.openclaw/AGENTS.md
```

`--print-global-snippet` is kept as a deprecated alias of
`--print-instructions-snippet`.

The snippets are agent-aware: the Claude Code snippet points at the
skill scripts; the OpenCode / Codex snippets describe the MCP tools
(`engram_recall`, `engram_save`, …).

---

## 5. Verify

```bash
# package importable?
python -c "import engram; print(engram.__version__ if hasattr(engram, '__version__') else 'ok')"

# CLIs on PATH?
engram version
engram-mcp --help

# install status (per agent)
engram install --agent claude-code --check
engram install --agent opencode    --check
engram install --agent codex       --check
engram install --agent openclaw    --check

# run a real list (likely empty)
python ~/.claude/skills/engram/scripts/list.py
```

If the above succeed, the install is complete.

---

## 6. Data locations

| Tier | Path | Holds |
|------|------|-------|
| **global** | `~/.claude/engram/global.{pst,pcc}` | `user`, `feedback` (follows the user across projects) |
| **local**  | `<project>/.claude/engram/local.{pst,pcc}` | `project`, `reference` (per-project isolation) |

Created on first save. **Strongly recommended**: add `.claude/engram/`
to each project's `.gitignore` to keep the vector store out of git.

---

## 7. Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `ENGRAM_EMBEDDER` | `""` *(= `local`)* | **URI spec** picking the backend. See the "Embedder spec" section below. |
| `ENGRAM_E5_MODEL_PATH` | — | Absolute path to a local e5 snapshot (offline override) |
| `OPENAI_API_KEY` | — | Required for `openai:…` |
| `COHERE_API_KEY` | — | Required for `cohere:…` |
| `VOYAGE_API_KEY` | — | Required for `voyage:…` |
| `ENGRAM_OPENAI_MODEL` | — | v0.3 back-compat: model name for the bare `openai` spec |
| `ENGRAM_HTTP_URL` / `ENGRAM_HTTP_DIM` | — | v0.3 back-compat: endpoint + dim for the bare `http` spec |
| `ENGRAM_DECAY_TAU_DAYS` | `30` | Ebbinghaus decay time constant (days) |
| `ENGRAM_RANK_BETA` | `0.10` | Rerank weight for `importance_eff` |
| `ENGRAM_RANK_GAMMA` | `0.02` | Rerank weight for `log1p(hits)` |
| `ENGRAM_HOME` | `~/.claude` | Root for the global tier (v0.4; takes precedence over `CLAUDE_HOME`) |
| `CLAUDE_HOME` | `~/.claude` | v0.3 back-compat alias of `ENGRAM_HOME` |
| `OPENCODE_HOME` | XDG / `%APPDATA%\opencode` | OpenCode config dir (where `AGENTS.md` + `opencode.json` live) |
| `CODEX_HOME` | `~/.codex` | Codex config dir (where `AGENTS.md` + `config.toml` live) |
| `OPENCLAW_HOME` | `~/.openclaw` | OpenClaw config & skill dir |
| `AGENTS_SKILLS_HOME` | `~/.agents/skills` | User-level skill dir for the `.agents/skills` convention (Codex). |
| `PISTADB_LIB_PATH` | — | Absolute path override for the PistaDB native lib |

---

## 7b. Embedder spec (`ENGRAM_EMBEDDER`)

One env var picks the backend.  Syntax: `scheme:target?key=val&key=val`,
or paste a full URL for the generic HTTP backend.

| Spec example | Backend | Required env |
|--------------|---------|--------------|
| *(unset)* / `local` | local `multilingual-e5-small` (default) | — |
| `local?device=cuda&model_path=/abs/path` | local with overrides | — |
| `openai:text-embedding-3-small` | OpenAI / OpenAI-compatible | `OPENAI_API_KEY` |
| `openai:text-embedding-3-large?dim=2048` | OpenAI with explicit dim | `OPENAI_API_KEY` |
| `ollama:nomic-embed-text` | local Ollama @ `localhost:11434` | — |
| `ollama:bge-m3?host=http://my-box:11434&dim=1024` | remote Ollama | — |
| `cohere:embed-multilingual-v3.0` | Cohere | `COHERE_API_KEY` |
| `voyage:voyage-3` | Voyage AI | `VOYAGE_API_KEY` |
| `http://localhost:8080/embeddings?dim=1024` | any OpenAI-compat endpoint | — |
| `hash?dim=384` | deterministic hash (tests only) | — |

**Pick once per `.pst` store** — the vector dim is locked at first
creation.  Switching backends with a mismatched dim fails fast at
startup with a clear three-fix message (switch back, delete the tier
files, or re-embed from scratch).

**Third-party backends** plug in cleanly:

```python
from engram.embedder import register_embedder, ParsedSpec

def _my_factory(spec: ParsedSpec):
    return MyEmbedder(model=spec.target, **spec.params)

register_embedder("my-backend", _my_factory)
# ENGRAM_EMBEDDER='my-backend:foo?param=bar'  now works.
```

> v0.3 env vars (`ENGRAM_OPENAI_MODEL`, `ENGRAM_HTTP_URL`,
> `ENGRAM_HTTP_DIM`) are still honoured — they auto-rewrite into the
> new URI spec at load time.

---

## 8. Uninstall

```bash
# Uninstall for a single agent (removes file-skill + MCP registration):
engram install --agent claude-code --remove
engram install --agent opencode    --remove
engram install --agent codex       --remove
engram install --agent openclaw    --remove

# Or all at once:
engram install --agent all --remove

pip uninstall engram

# Optional: wipe the data too
rm -rf ~/.claude/engram/             # global vector store
# Per-project local stores must be deleted individually:
rm -rf <project>/.claude/engram/
```

The v0.3 `engram install-skill --remove` still works (Claude Code skill only).

---

## 9. Developing (working in this repo)

```bash
git clone https://github.com/shannonxu-2018/Engram.git
cd Engram
pip install -e ".[all]"                # editable + all extras

# dev mode: symlink the skill so source edits propagate live
engram install-skill --dev

# smoke test (no external deps)
ENGRAM_EMBEDDER=hash python tests/smoke_test.py

# benchmark vs the old MD memory
python -m eval.runner
```

Or just use the bundled bootstrap:

```powershell
./bootstrap.ps1                        # Windows
```

```bash
./bootstrap.sh                         # Linux / macOS
```

Both do `pip install -e ".[all]"` + `engram install-skill --dev` by default.

---

## 10. Troubleshooting

### `OSError: PistaDB shared library not found`

Your OS/arch isn't in the bundled set. Either:

1. Build / fetch the matching PistaDB binary from upstream
   ([shannonxu-2018/PistaDB](https://github.com/shannonxu-2018/PistaDB))
   and set `PISTADB_LIB_PATH=/abs/path/to/lib`
2. Drop the binary into the installed `pistadb/` package dir so auto-discovery finds it

### `ImportError: sentence_transformers`

The default embedder needs it. Fix:

```bash
pip install "engram[local]"
```

Or switch backends:

```bash
export ENGRAM_EMBEDDER=openai
export OPENAI_API_KEY=sk-...
```

### First run is very slow with no visible progress

On the first instantiation of `LocalE5Embedder`, sentence-transformers
fetches e5-small (~471 MB) from HuggingFace; progress goes to stderr.
Pre-warm:

```bash
python -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('intfloat/multilingual-e5-small')"
```

Subsequent calls hit `~/.cache/huggingface/` and run offline.

### Claude can't find the skill

```bash
ls ~/.claude/skills/engram/SKILL.md  # exists?
```

If not, re-run `engram install --agent claude-code`. If yes but Claude
still doesn't use it, check that your project `CLAUDE.md` (or
`~/.claude/CLAUDE.md`) references Engram — see step 4.

### OpenClaw doesn't see the engram skill

```bash
ls ~/.openclaw/skills/engram/SKILL.md   # exists?
```

If not, re-run `engram install --agent openclaw`. If yes but OpenClaw
still doesn't surface it, confirm the skill is enabled — per the
OpenClaw skills docs the agent snapshots eligible skills at session
start, so you may need to restart the session.

### OpenCode / Codex doesn't see the `engram` MCP server

Check the registry file the agent reads:

```bash
# OpenCode
cat ~/.config/opencode/opencode.json   # or %APPDATA%\opencode\opencode.json on Windows

# Codex
cat ~/.codex/config.toml
```

You should see `engram` under `mcpServers` (OpenCode) or
`[mcp_servers.engram]` (Codex). If not, re-run
`engram install --agent <name>`. If yes but the agent still doesn't
call Engram, append the per-agent instructions snippet (step 4) — the
host needs both the registration *and* the prompt to actually use it.

### `engram-mcp: command not found`

The `engram-mcp` console script ships with the package — `pip install`
should have put it on `PATH`. Verify:

```bash
which engram-mcp                       # POSIX
where engram-mcp                       # Windows
python -m engram.mcp_server --help     # last-resort fallback
```

If only the `python -m …` form works, your shell's `PATH` is missing
the venv's scripts dir. Either reinstall the agent with that absolute
path, or activate the venv before launching the agent.

### No HuggingFace access (corporate network)

Two options:

1. Pre-warm on a connected machine (see above), then copy
   `~/.cache/huggingface/hub/` over.
2. Use a non-local backend: set `ENGRAM_EMBEDDER=openai` (or `http`
   pointing at an internal embedding service).

---

## See also

| File | Audience | Purpose |
|------|----------|---------|
| [`README.md`](README.md) / [`README_CN.md`](README_CN.md) | End users | What Engram is, design, benchmarks |
| `INSTALL.md` (this file) / [`INSTALL_CN.md`](INSTALL_CN.md) | Operators | Install, configure, troubleshoot, uninstall |
| [`CLAUDE.md`](CLAUDE.md) | Contributors | Dev conventions for this repo |
| `SKILL.md` (inside the pip package) | The agent | Universal skill protocol — used by Claude in any project |
