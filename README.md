<div align="center">

# Engram

**Vector memory for AI agents — semantic, cheap, hippocampus-inspired.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-v0.4%20beta-orange)](#roadmap)
[![Platforms](https://img.shields.io/badge/platforms-Win%20x64%20%7C%20Linux%20x64%20%7C%20macOS%20arm64-lightgrey)](#installation)

[English](README.md) · [中文](README_CN.md) · [Install guide](INSTALL.md) · [Full benchmarks](eval/RESULTS.md)

</div>

---

Engram is a **vector-database memory backend** for AI coding agents. One ``.pst`` store, four hosts: **Claude Code**, **OpenCode**, **Codex**, **OpenClaw**. It replaces the legacy `MEMORY.md` / per-file markdown memory system with semantic KNN search over a local HNSW index — and gets pattern separation, sleep replay, and Ebbinghaus decay for free.

```bash
pip install git+https://github.com/shannonxu-2018/Engram.git
engram setup            # one interactive wizard: install + enable + warm up + verify
```

---

## Why Engram

- **Smarter recall, lighter context.** **100 % Recall@3** with adaptive top-K, while the legacy MD index manages 92 % at **26× the token cost**. ([benchmarks](#benchmarks))
- **One store, four agents.** Drop-in to **Claude Code**, **OpenCode**, **Codex** (skill **+** MCP each), or **OpenClaw** (skill only — its docs don't cover MCP). Personal preferences follow you everywhere; project facts stay project-scoped.
- **Hippocampus-inspired.** Pattern separation rejects near-duplicate writes, Ebbinghaus decay fades unused memories, importance × recency reranking surfaces what matters *now*, and sleep replay periodically consolidates the store.
- **Persistent and warm.** The `engram-mcp` server keeps the embedder loaded — recall latency drops from ~3 s (cold script) to ~10 ms (warm process).
- **Local-first.** Default `multilingual-e5-small` (384-d, multilingual) runs entirely offline after a one-time ~471 MB download. Swap to OpenAI / self-hosted via one env var.
- **Single wheel, no daemon.** `pip install engram` + one CLI. PistaDB native libs ship in the wheel for Windows x64, Linux x64, and macOS Apple Silicon.

## Benchmarks

40 synthetic memories, 25 queries (10 easy / 10 medium / 5 hard), real `multilingual-e5-small` embedder, `tiktoken cl100k_base` token counter, amortised over a 50-turn session.

| Metric                       | MD baseline (top-3) | Engram (k=3) | Δ |
|------------------------------|---------------------|--------------|---|
| **Recall@3**                 | 92.0 %              | **100.0 %**  | +8.0 pp |
| **MRR**                      | 0.880               | **0.973**    | +0.093 |
| **Tokens / turn amortised**  | 1430.8              | **54.7**     | **26.1× cheaper** |
| Baseline (always-loaded)     | 1331 tokens         | **0 tokens** | — |
| Avg query tokens             | 199.6               | 109.4        | — |

> **Two-sided win:** more accurate **and** an order of magnitude cheaper in tokens. There is no always-loaded `MEMORY.md` index; recall returns one compact line per hit. Full bench: [`eval/RESULTS.md`](eval/RESULTS.md).

Reproduce:

```bash
pip install "engram[eval]"
python -m eval.runner
```

## Supported agents

| Agent | Integration | Instructions file | Discovery point |
|-------|-------------|-------------------|-----------------|
| **Claude Code** | file-skill **+** MCP | `~/.claude/CLAUDE.md` | `~/.claude/skills/engram/` & `~/.claude.json` |
| **OpenCode**    | file-skill **+** MCP | `~/.config/opencode/AGENTS.md` *(XDG)* | `~/.config/opencode/skills/engram/` & `opencode.json` |
| **Codex**       | file-skill **+** MCP | `~/.codex/AGENTS.md` | `~/.agents/skills/engram/` & `~/.codex/config.toml` |
| **OpenClaw**    | file-skill (MCP not documented) | `~/.openclaw/AGENTS.md` | `~/.openclaw/skills/engram/` |

> **Cross-ecosystem discovery.** OpenCode and OpenClaw both scan multiple skill roots (`~/.config/opencode/skills/` or `~/.openclaw/skills/`, plus `~/.claude/skills/` and `~/.agents/skills/`) — so if you already ran `engram install --agent claude-code` or `--agent codex`, those agents find the skill automatically. The dedicated `--agent <name>` install just gives you a clean per-agent footprint for uninstall.

Install for one, several, or all:

```bash
engram install --agent claude-code   # skill + MCP
engram install --agent opencode      # skill + MCP
engram install --agent codex         # skill + MCP
engram install --agent openclaw      # skill only
engram install --agent all           # everything
```

> **MCP transport:** the `engram-mcp` console script ships with the package. Without the official `mcp` Python SDK installed it falls back to a built-in stdio JSON-RPC loop covering `initialize` / `tools/list` / `tools/call` — every supported host accepts it. Install `pip install "engram[mcp]"` for the spec-perfect framing.

## Installation

For users of any supported agent, the easy path is one interactive command:

```bash
pip install git+https://github.com/shannonxu-2018/Engram.git
engram setup            # interactive wizard — recommended
```

`setup` detects your installed agents, then walks you through install →
enable-for-every-project → model warm-up → a closing health check, with a
sensible default for every yes/no prompt (just press Enter). Add `--yes`
to accept all defaults non-interactively, or `--agent NAME` to skip the
agent question.

<details>
<summary><b>Manual path</b> — the steps <code>setup</code> runs for you</summary>

```bash
pip install git+https://github.com/shannonxu-2018/Engram.git
engram install --agent claude-code   # install skill + register MCP
engram init                          # wire CURRENT project (CLAUDE.md / .gitignore)
engram warmup                        # pre-download the embedder (avoids 30 s cold-start)
engram doctor                        # confirm everything is wired correctly
```

This four-command sequence covers the three failure modes new users hit
the most: *skill not installed* (`install`), *agent doesn't know it
should use the skill* (`init`), and *first recall stalls while a 471 MB
model downloads in silence* (`warmup`).  `doctor` ties them together
with a single green/yellow/red report and copy-pasteable fix commands.

</details>

For developers — clone and bootstrap:

```bash
git clone https://github.com/shannonxu-2018/Engram.git && cd Engram
./bootstrap.ps1     # Windows
./bootstrap.sh      # Linux / macOS
```

> Full install paths, the supported-OS matrix, env vars, troubleshooting, and uninstall live in [`INSTALL.md`](INSTALL.md).

## The `engram` CLI

Eight top-level subcommands.  Run `engram <cmd> --help` for the full
flag list; the table below covers the everyday invocation.

| Command | What it does | Most common form |
|---------|--------------|------------------|
| **`setup`** | **Interactive wizard (recommended).** Detects your agents, then runs install → enable-globally → warmup → doctor, with a default for every prompt.  `--yes` for an unattended run, `--agent NAME` to preselect. | `engram setup` |
| **`install`** | Sets up Engram for one agent (or `--agent all`).  Copies/symlinks the file-based skill into the agent's skill dir **and** registers the `engram-mcp` server in the agent's MCP config. Idempotent. | `engram install --agent claude-code` |
| **`init`** | Wires the **current project** to actually *use* Engram: appends the instructions snippet to `CLAUDE.md` / `AGENTS.md`, adds `.claude/engram/` to `.gitignore`, creates the local-tier directory.  Idempotent — re-runs are no-ops unless `--force`. | `engram init` |
| **`uninstall`** | One-click teardown, by scope. `--scope project` reverses `init` (strips the Engram block from `CLAUDE.md`/`AGENTS.md`, drops the `.gitignore` line); `--scope global` reverses `install` (removes the skill, unregisters MCP, strips the per-agent user-level snippet); `--scope all` does both. Omit `--scope` to be asked. Stored memories are **kept** unless you pass `--purge`. | `engram uninstall --scope project` |
| **`warmup`** | Pre-loads the embedder so the first `recall` doesn't pay 30 s of cold-start.  For the local backend, also pre-downloads the ~471 MB `multilingual-e5-small` model from HuggingFace. | `engram warmup` |
| **`doctor`** | One-shot health check across package install, PistaDB native lib, embedder spec, e5 model cache, every agent's skill+MCP state, the `.pst` tier files, and `engram-mcp` on PATH.  Prints copy-pasteable fix commands for anything not OK.  Exit codes: `0` OK / `1` warnings / `2` errors. | `engram doctor` |
| **`agents`** | List the four built-in agents and their integration mode. | `engram agents` |
| **`version`** | Print the installed package version. | `engram version` |

> **First-time recipe**: just run `engram setup` — it does `install` →
> enable → `warmup` → `doctor` for you, with a default for every prompt.
> The standalone subcommands stay available for scripted or fine-grained
> setups, and each is idempotent.  Together they collapse the most
> common new-user surprises (skill not installed / agent doesn't know
> to use it / "why is the first recall hanging?" / silently mis-wired
> across multiple agents) into explicit, verifiable actions.
>
> The legacy `engram install-skill` and v0.3-style env vars still
> work — back-compat is preserved across all of these.

## How it works

### The four memory types

Types are not free-form — the model classifies each save against this fixed list. Same four-tuple as the legacy MD policy, so saving heuristics carry over.

| Type | What it captures | Example trigger |
|------|------------------|-----------------|
| `user`      | Identity / background / preferences         | *"I'm a data scientist focused on logging"* |
| `feedback`  | Corrections **and** validated non-obvious choices | *"Don't mock the DB"* / *"Yes, the bundled PR was right"* |
| `project`   | Why-decisions, deadlines, motivations       | *"Auth rewrite is compliance-driven, not tech debt"* |
| `reference` | Pointers to external systems / dashboards   | *"Pipeline bugs live in Linear project INGEST"* |

### Two-tier storage

Routing is automatic by type. Personal preferences travel with the user; project facts stay with the project.

| Tier | Path | Holds | Scope |
|------|------|-------|-------|
| **global** | `~/.claude/engram/global.{pst,pcc}` | `user`, `feedback` | Follows the user across all projects |
| **local**  | `<project>/.claude/engram/local.{pst,pcc}` | `project`, `reference` | Per-project, isolated |

> The `local` tier should be in your project's `.gitignore` — vector stores are not source code.

### Memory policy (when to save what)

Engram is **passive storage**, not an auto-ingester. Nothing in the code base listens to your conversation, dumps transcripts, or compresses turns into memories at the end of a session. A row in `.pst` exists only because **the model actively called `save`**, guided by the policy below.

| Signal | Type | Example |
|--------|------|---------|
| User reveals role / background / preference | `user` | *"I'm a data scientist focused on logging"* |
| User corrects an approach, **or** confirms a non-obvious one | `feedback` | *"Don't mock the DB"* / *"Yes, the bundled PR was right"* |
| Project context, motivation, deadline, the "why" behind a decision | `project` | *"Auth rewrite is compliance-driven"* |
| Pointer to an external system | `reference` | *"Pipeline bugs are in Linear INGEST"* |

**What never gets saved** (even if explicitly asked): code patterns, file paths, git history, debug recipes, ephemeral task state, anything already in `CLAUDE.md` / `AGENTS.md`. Cheaper to re-derive from the repo than to store.

**Importance does not gate writing — it gates ranking.** `save` accepts `--importance 0.7` (default `0.5`). At recall time the composite is `distance − β·importance_eff − γ·log1p(hits)`, so important or frequently-accessed memories outrank stale equidistant ones. Every `expand` does `hits++` and applies a boost. With 30-day Ebbinghaus decay, attention self-concentrates on what's actually useful — but nothing is auto-deleted.

**Write-time duplicate suppression.** Before inserting, `save` runs a KNN check; if a near-duplicate exists (cosine distance < 0.08) it returns `MERGE_SUGGESTION` instead of writing. Either `--update` the existing row, `--force` a new one, or skip.

User overrides:

- *"Remember that …"* → forces an immediate save
- *"Forget …"* → `forget --name <slug>`

### Hippocampus-inspired mechanics

| Brain mechanism | Engram implementation |
|------------------|-----------------------|
| **Pattern separation** | `save()` KNN-checks; if `dist < 0.08` it refuses to write a near-duplicate and offers a merge instead |
| **Pattern completion** | `recall()` KNN against the query — finds the right memory from a partial cue |
| **Trace strengthening** | every recall/expand bumps `hits++`, updates `accessed_at`, applies `boost_on_access` to importance |
| **Decay** | Ebbinghaus: `eff = base · exp(-Δt/τ)`, τ default 30 days |
| **Sleep replay** | `consolidate` periodically merges near-duplicates and re-applies decay |
| **Reactivation reranking** | recall composite = `distance − β·importance_eff − γ·log1p(hits)` — busy, important memories outrank stale equidistant ones |

## Per-project control

Whether the agent actually *uses* Engram in a given project depends on **two signals**, not one:

1. **Availability** — is the skill / MCP server registered for this agent?
2. **Instruction** — does the project's instructions file (or its global equivalent) tell the agent to use it?

<details>
<summary><b>Enable Engram in a project</b></summary>

**(a) Per-project (most common).** Add to the project's instructions file:

```markdown
# Memory

Use the Engram skill at `~/.claude/skills/engram/` for all memory
operations in this project. Follow the protocol in its SKILL.md.

The local tier writes to `.claude/engram/local.pst` — keep that in
.gitignore.
```

Then add `.claude/engram/` to `.gitignore`.

**(b) Globally — every future project.** Append the snippet to your global instructions file (once):

```bash
engram install --agent claude-code --print-instructions-snippet \
    >> ~/.claude/CLAUDE.md
```

**(c) Bundled into the project (team-shared).** Install the skill *inside* the project for everyone who clones it:

```bash
cd <project>
engram install --agent claude-code --target ./.claude/skills/engram
```

Teammates still need `pip install engram` to satisfy the Python deps.

</details>

<details>
<summary><b>Disable Engram in a project</b></summary>

Three levels, from soft to hard:

**(a) Do nothing.** Without the global snippet, the agent only uses Engram on explicit user triggers (*"remember…"*). That's already a soft opt-out.

**(b) Tell the agent not to use it.** Per-project instructions file:

```markdown
# Memory

This project does NOT use the engram skill. Ignore any global
instruction to use it. Fall back to default memory behavior.
```

Project files override global ones.

**(c) Block the scripts entirely.** Project `.claude/settings.local.json`:

```json
{
  "permissions": {
    "deny": [
      "Bash(*engram*)",
      "Bash(*~/.claude/skills/engram*)"
    ]
  }
}
```

The harness blocks calls before the agent can even attempt them.

</details>

<details>
<summary><b>User-global vs project-local skill precedence</b></summary>

If both `<project>/.claude/skills/engram/` and `~/.claude/skills/engram/` exist, **the project-level wins** (Claude Code reads project skills first). So you can always override the user-global default for one specific project.

| Goal | Where to install |
|------|------------------|
| "I want Engram available everywhere" | user-global (default of `engram install`) |
| "Team should share a pinned version via git" | project-local (`--target ./.claude/skills/engram`, **no** `--dev`) |
| "This one project needs a customised SKILL.md" | project-local override |

</details>

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│   Claude Code  /  OpenCode  /  Codex  /  OpenClaw                 │
│   (instructions file tells the agent to use Engram)               │
└──────────────────────────────────────────────────────────────────┘
        │                                            ▲
        │ skill script / MCP tool call                │ compact text
        ▼                                            │
┌───────────────────────────────────────────────────────┴──────────┐
│   ~/.claude/skills/engram/scripts/*.py    (skill-based agents)   │
│   engram-mcp  (long-lived stdio MCP server, MCP-capable agents)  │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│   engram (Python package, in site-packages)                       │
│   embedder.py   ─►  7 backends (local e5, OpenAI, Ollama, Cohere, │
│                     Voyage, HTTP, Hash (Hash = testing only)      │
│   store.py      ─►  Schema + two-tier paths                       │
│   memory.py     ─►  MemoryManager.save / recall / expand / forget │
│   decay.py      ─►  importance_eff, boost_on_access, rerank       │
│   consolidate.py ►  sleep replay (merge_pass + decay_pass)        │
│   agents.py     ─►  AgentProfile registry (v0.4)                  │
│   install.py    ─►  multi-agent installer (v0.4)                  │
│   mcp_server.py ─►  engram-mcp entrypoint (v0.4)                  │
└──────────────────────────────────────────────────────────────────┘
        │                                  │
        ▼                                  ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│ pistadb (vendored)          │   │ HuggingFace cache           │
│   ctypes wrapper + bundled  │   │ ~/.cache/huggingface/       │
│   .dll/.so/.dylib           │   │ multilingual-e5-small (384) │
│   HNSW + COSINE             │   │ (auto-download on first use)│
└─────────────────────────────┘   └─────────────────────────────┘
```

<details>
<summary><b>Schema</b> (Milvus-style, declared in <code>engram/store.py</code>)</summary>

```python
[
  FieldSchema("mem_id",      INT64,        is_primary=True, auto_id=True),
  FieldSchema("type",        VARCHAR, max_length=20),
  FieldSchema("name",        VARCHAR, max_length=128),
  FieldSchema("description", VARCHAR, max_length=512),    # gist (recall returns this)
  FieldSchema("content",     VARCHAR, max_length=8192),   # full body (lazy via expand)
  FieldSchema("tags",        JSON),                       # [[link]] / keywords
  FieldSchema("importance",  FLOAT),
  FieldSchema("hits",        INT64),
  FieldSchema("created_at",  INT64),
  FieldSchema("accessed_at", INT64),
  FieldSchema("vector",      FLOAT_VECTOR, dim=384),
]
# Metric.COSINE + Index.HNSW
```

</details>

<details>
<summary><b>Embedder backends (7 built-in + extensible)</b></summary>

`Embedder` is a `Protocol` — any provider can plug in.  Pick one via the
`ENGRAM_EMBEDDER` URI-spec env var:

```bash
ENGRAM_EMBEDDER='openai:text-embedding-3-large?dim=2048'
ENGRAM_EMBEDDER='ollama:nomic-embed-text'                       # local, no API key
ENGRAM_EMBEDDER='cohere:embed-multilingual-v3.0'
ENGRAM_EMBEDDER='voyage:voyage-3'
ENGRAM_EMBEDDER='http://localhost:8080/embeddings?dim=1024'     # any OpenAI-compatible endpoint
ENGRAM_EMBEDDER='hash?dim=384'                                  # tests only
```

URI shape: `scheme:target?key=val&key=val` — or just paste a full URL
for the generic HTTP endpoint.  Empty / unset = local default.

| Scheme | Backend class | Required env / params | Default dim |
|--------|---------------|------------------------|-------------|
| `local` *(default)* | `LocalE5Embedder` | (none — `?device=cuda` / `?model_path=…` optional) | 384 |
| `openai`            | `OpenAIEmbedder`  | `OPENAI_API_KEY`; `?dim=…&base_url=…` optional         | 1536 / 3072 |
| `ollama`            | `OllamaEmbedder`  | Local Ollama; `?host=http://...&dim=…` optional        | 768 |
| `cohere`            | `CohereEmbedder`  | `COHERE_API_KEY`                                         | 1024 |
| `voyage`            | `VoyageEmbedder`  | `VOYAGE_API_KEY`                                         | 1024 |
| `http`              | `HTTPEmbedder`    | URL + `?dim=N`                                           | (you set) |
| `hash`              | `HashEmbedder`    | (none — `?dim=…` optional)                              | 384 |

The vector dim is locked to the `.pst` at first creation — pick one
backend per store, or delete the tier files and start fresh.  A dim
mismatch on startup gives a clear error with the three fix paths.

**Third-party backends** plug in cleanly:

```python
from engram.embedder import register_embedder, ParsedSpec

def _bedrock_factory(spec: ParsedSpec):
    return MyBedrockEmbedder(
        model=spec.target,
        region=spec.params.get("region", "us-east-1"),
    )

register_embedder("bedrock", _bedrock_factory)
# Then anywhere:  ENGRAM_EMBEDDER='bedrock:cohere.embed-multilingual-v3?region=us-west-2'
```

> v0.3 env vars still work — `ENGRAM_EMBEDDER=openai` + `ENGRAM_OPENAI_MODEL=…`
> and `ENGRAM_EMBEDDER=http` + `ENGRAM_HTTP_URL=…` + `ENGRAM_HTTP_DIM=…`
> are auto-rewritten into the new URI form at load time.

</details>

## CLI reference

All scripts live under `~/.claude/skills/engram/scripts/` after install. The same operations are exposed via MCP tools (`engram_recall` / `engram_save` / …) for OpenCode and Codex.

| Script | Purpose |
|--------|---------|
| `recall.py <query> [--k N] [--type T] [--tier T] [--no-rerank] [--json]` | Semantic search. `--k` defaults to **adaptive** (gap-based, 2-10 hits). |
| `save.py <type> <name> <desc> [<content>] [--tag X] [--importance F] [--force\|--update]` | Encode a memory; dedup-checked. |
| `expand.py <id> [--json]` | Fetch full content (bumps `hits++`). |
| `list.py [--type T] [--tier T] [--limit N] [--json]` | Enumerate without embedding cost. |
| `forget.py (--name <slug> \| --id <n> \| --older-than DAYS) [--type T] [--tier T] [--dry-run]` | Delete by name, id, or age. `--older-than` compares `accessed_at`. `--dry-run` previews. |
| `patch.py <id-or-name> [--description X] [--content X] [--importance F] [--type T] [--rename X] [--add-tag X] [--remove-tag X] [--set-tag X]` | Edit selected fields of one memory. Metadata-only ⇒ id preserved; desc/content/tier ⇒ id changes. |
| `consolidate.py [--dry-run] [--no-merge\|--no-decay] [--threshold F]` | Sleep replay (merge + decay). |
| `related.py <name-or-id> [--depth N] [--with-content]` | BFS over `[[name]]` tag links. |
| `migrate_md.py <source-dir> [--dry-run] [--force]` | One-time import from legacy MD memory. |

**When to call each** (the agent's heuristics):

- **Recall** — when the user references prior conversation, asks about themselves, or you're about to give advice that depends on their preferences / feedback.
- **Save** — when you learn something durable about the user, their feedback, the project, or an external reference.
- **Expand** — only when the description alone isn't enough.
- **Consolidate** — weekly or after a heavy save burst.

<details>
<summary><b>Worked examples</b></summary>

#### `recall` — semantic search

```bash
$ recall.py "user preferences"
# adaptive k=2
    9 | l | reference | disable-pattern  | d=0.082 | three levels of opt-out for engram in a project ...
    1 | g | user      | lang-chinese     | d=0.103 | user prefers Simplified Chinese for chat ...
```

Columns: `id | tier(g/l) | type | name | distance | description`. Lower distance = better match.

**Adaptive k** (default): the script fetches a generous candidate pool, sorts by distance, and cuts at the first big gap (bounded between K_MIN=2 and K_MAX=10). Force a fixed K with `--k 5`.

#### `save` — encode a memory

```bash
$ save.py feedback no-mock-db \
    "integration tests must hit a real database, not mocks" \
    "Why: prior incident where mock/prod divergence masked a broken migration. \
     How to apply: never mock the DB layer under tests/integration/**."

inserted: id=11 tier=global
```

If a near-duplicate (cosine < 0.08) exists, the script refuses and prints `MERGE_SUGGESTION` — re-run with `--update` to overwrite, `--force` to insert anyway, or skip.

#### `expand` — full content + trace strengthening

```bash
$ expand.py 9
id   : 9
tier : local
type : reference
name : disable-pattern
hits : 3
---
Why: user-level skill installs reach every project, hence the need for opt-out.
How to apply: see README §"Per-project control".
```

Side effects: `hits++`, `accessed_at = now`. Use sparingly.

#### `related` — graph traversal via `[[name]]` tags

If a memory carries `--tag "[[auth-rewrite-driver]]"`:

```bash
$ related.py auth-rewrite-driver --depth 1
   12 | l | project | compliance-deadline | d=0.000 | auth rewrite must ship before 2026-04-01 ...
```

Pure metadata walk — zero embedding cost.

</details>

## Python API

```python
from engram import MemoryManager

with MemoryManager() as mgr:
    mgr.save(
        type="feedback",
        name="terse-replies",
        description="user wants terse responses with no trailing summaries",
        content="Why: aesthetic. How to apply: skip the recap at end of turn.",
        tags=["style", "[[role-data-sci]]"],
    )

    hits = mgr.recall("how should I phrase the wrap-up?", k=5)
    for h in hits:
        print(h.to_compact())

    neighbours = mgr.recall_related("terse-replies", depth=1)

# Maintenance
from engram import consolidate
with MemoryManager() as mgr:
    report = consolidate(mgr, dry_run=True)
    print(report.to_dict())
```

## Configuration

All knobs are environment variables, picked up fresh each call:

| Var | Default | Meaning |
|-----|---------|---------|
| `ENGRAM_EMBEDDER` | `""` *(= `local`)* | **URI spec** picking the embedder backend. Examples: `openai:text-embedding-3-small`, `ollama:nomic-embed-text`, `cohere:embed-multilingual-v3.0`, `voyage:voyage-3`, `http://host:port/embed?dim=N`, `hash?dim=384`. See the Embedder-backends section above. |
| `ENGRAM_E5_MODEL_PATH` | — | Absolute path to a local e5 snapshot (offline override) |
| `OPENAI_API_KEY` | — | Required for `openai:…` spec |
| `COHERE_API_KEY` | — | Required for `cohere:…` spec |
| `VOYAGE_API_KEY` | — | Required for `voyage:…` spec |
| `ENGRAM_OPENAI_MODEL` | — | v0.3 back-compat: when `ENGRAM_EMBEDDER=openai` (bare), this picks the model. Equivalent to `openai:<model>` in the new form. |
| `ENGRAM_HTTP_URL` / `ENGRAM_HTTP_DIM` | — | v0.3 back-compat for `ENGRAM_EMBEDDER=http`. Equivalent to `http://URL?dim=DIM`. |
| `ENGRAM_DECAY_TAU_DAYS` | `30` | Ebbinghaus decay τ (days) |
| `ENGRAM_RANK_BETA` | `0.10` | Weight of `importance_eff` in rerank composite |
| `ENGRAM_RANK_GAMMA` | `0.02` | Weight of `log1p(hits)` in rerank composite |
| `ENGRAM_HOME` | `~/.claude` | Root for the global tier *(v0.4; takes precedence over `CLAUDE_HOME`)* |
| `CLAUDE_HOME` | `~/.claude` | v0.3 back-compat alias of `ENGRAM_HOME` |
| `OPENCODE_HOME` | XDG / `%APPDATA%\opencode` | OpenCode config dir |
| `CODEX_HOME` | `~/.codex` | Codex config dir (`AGENTS.md`, `config.toml`) |
| `OPENCLAW_HOME` | `~/.openclaw` | OpenClaw config & skill dir |
| `AGENTS_SKILLS_HOME` | `~/.agents/skills` | User-level skill dir for the agent-agnostic `.agents/skills` convention (Codex) |
| `PISTADB_LIB_PATH` | — | Absolute path override for the PistaDB native lib |

## Project layout

```
engram/                                  (repo root)
├── README.md / README_CN.md             project intro
├── INSTALL.md / INSTALL_CN.md           deploy / configure / troubleshoot
├── CLAUDE.md                            dev conventions for this repo
├── pyproject.toml                       pip-installable, src layout
├── bootstrap.ps1 / bootstrap.sh         one-shot dev install
├── src/
│   ├── engram/                          the package
│   │   ├── embedder.py / store.py / memory.py
│   │   ├── decay.py / consolidate.py    (v2 mechanics)
│   │   ├── agents.py / install.py       (v0.4 multi-agent)
│   │   ├── init_project.py / doctor.py  (v0.4 init + health check)
│   │   ├── mcp_server.py                (v0.4 `engram-mcp`)
│   │   ├── cli.py / setup_wizard.py     (`engram` CLI + setup wizard)
│   │   ├── install_skill.py             (v0.3 install-skill alias)
│   │   └── _skill_files/                shipped as package data
│   │       ├── SKILL.md
│   │       └── scripts/{recall,save,expand,list,forget,patch,
│   │                    related,consolidate,migrate_md,_common}.py
│   └── pistadb/                         vendored vector DB
│       └── pistadb.dll / libpistadb.so / libpistadb.dylib
├── tests/
│   ├── smoke_test.py                    end-to-end with HashEmbedder
│   ├── unit_test.py                     v0.3 review fixes
│   └── agents_test.py                   v0.4 multi-agent layer
└── eval/                                head-to-head benchmark vs MD baseline
    └── RESULTS.md                       (generated)
```

## Built on PistaDB

Engram's vector store is **[PistaDB](https://github.com/shannonxu-2018/PistaDB)** — *"the embedded vector database for LLM-native applications."* It provides the HNSW + cosine index, the single-`.pst` file format (with optional `.wal` files for crash recovery), and a zero-external-dependency C99 core. Engram vendors PistaDB at `src/pistadb/` together with prebuilt native libraries for Windows x64, Linux x64, and macOS Apple Silicon, so `pip install engram` is all you need — there is no external service to run and no extra library to fetch.

Why PistaDB:

- **Single-file, embedded.** One `.pst` per tier means the global store is just a file under `~/.claude/engram/` and the local store is just a file under the project. Backup, sync, and `.gitignore` are trivial.
- **Zero external dependencies.** No daemon, no Docker, no managed service. The C99 core compiles to a small `.dll`/`.so`/`.dylib` that Engram loads via `ctypes`.
- **Index choice.** HNSW (recommended for RAG) plus seven other indexes and five distance metrics — Engram picks `HNSW + COSINE`, but the schema can switch if your workload changes.
- **Polyglot.** PistaDB has wrappers in Python, Go, C++, Java, Swift, Rust, and C# — so the same `.pst` file Engram writes is readable from any of those if you ever need it.

If you want to read or contribute to the vector DB itself, see the upstream repo: <https://github.com/shannonxu-2018/PistaDB>.

## Roadmap

| Version | Status | Highlights |
|---------|--------|------------|
| **v0.3** | shipped | Single-agent (Claude Code skill), four memory types, two-tier storage, hippocampus mechanics (pattern separation, decay, sleep replay, importance rerank). |
| **v0.4** | shipped | **Multi-agent**: same store reachable from Claude Code, OpenCode, Codex (each with both file-skill and MCP), and OpenClaw (file-skill only — MCP not documented). OpenCode/OpenClaw also discover cross-ecosystem (`~/.claude/skills/`, `~/.agents/skills/`). New `engram-mcp` server. `ENGRAM_HOME` / `AGENTS_SKILLS_HOME` / `OPENCLAW_HOME` env vars. Back-compat for all v0.3 CLI. |
| **v0.5** | planned | Cross-embedder migration (re-embed when switching backends). Active `replay.py` — LLM summarises hot clusters and re-saves the summary. Per-memory ACLs (private vs shared). |
| **v0.6** | planned | Cross-project knowledge graph (cross-tier `[[link]]` resolution). Web UI for browsing the store. Integration with Anthropic's memory API once stable. |

<details>
<summary><b>Known limitations</b> (current)</summary>

- **`save()` dedup is best-effort under concurrent writers.** PistaDB is single-writer per `.pst`; the dedup check + insert in `MemoryManager.save()` aren't atomic. Fine for the one-session-per-store pattern that AI agents use today; revisit when a server mode lands.
- **Several admin operations are O(n) over the whole tier.** `list.py`, `_bump_access` on recall, and `recall_related`'s graph walk all rewrite or rescan the entire sidecar JSON. Comfortable up to ~1 000 memories.
- **`--dev` on Windows requires Developer Mode** (or admin) for symlinks. The installer refuses to silently fall back to copy.
- **macOS Intel (x86_64) is not supported.** Only Apple Silicon `libpistadb.dylib` ships in the wheel.

</details>

## License

MIT — see [`LICENSE`](LICENSE).
