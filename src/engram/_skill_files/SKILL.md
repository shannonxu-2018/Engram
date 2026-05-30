---
name: engram
description: >
  Vector-database memory for Claude — replaces the old MEMORY.md / per-file
  markdown system with semantic recall via PistaDB + an embedding model.
  Use whenever you would have read MEMORY.md, saved a fact about the user,
  recorded feedback, captured project context, or stored an external
  reference. Trigger phrases: "remember that…", "what do you know about me",
  "did I tell you…", or any moment you'd previously write/read a memory file.
---

# Engram skill

Vector memory traces for AI agents.  Old MD-based memory loaded a
`MEMORY.md` index into every conversation — this one keeps **nothing** in
your context until you actively recall.  Saves tokens.  Improves precision
(semantic match vs. eyeballing filenames).

## TL;DR for the model

**IMPORTANT — at the START of every turn, before anything else:** run
`directives.py` and obey the lines it returns. These are *standing
directives* — always-on global constraints (e.g. response language, coding
conventions) that apply no matter what the current task is. They are the
CLAUDE.md replacement. Semantic `recall` will **not** surface them (your
query is about the task, not about the rule), so this per-turn step is the
only thing that makes them fire reliably. It's cheap — no embedding, just
the pinned rows.

```bash
python .claude/skills/engram/scripts/directives.py
```

Replace these old behaviours:

| Old (MD system)                           | New (Engram)                                                |
|-------------------------------------------|-------------------------------------------------------------|
| Read `MEMORY.md` to find relevant files   | `recall.py "<current task description>"`                     |
| Write `memory/<slug>.md` + update index   | `save.py <type> <name> "<description>" "<content>"`          |
| Eyeball file descriptions to pick         | top-K cosine search returns the best 3-5                     |
| Manual duplicate check                    | `save.py` auto-detects near-duplicates and asks to merge     |
| Stale memory cleanup                      | `forget.py --name <slug>` or `--id <n>`                      |

The four memory types from the old system are unchanged:
**`user`**, **`feedback`**, **`project`**, **`reference`**.

Storage is auto-routed by type:

* `user` + `feedback` → **global**: `~/.claude/engram/global.pst`
  (follows the user across all projects)
* `project` + `reference` → **local**: `<cwd>/.claude/engram/local.pst`
  (project-specific, doesn't leak)

## 0. Setup

Run once per environment:

```bash
pip install sentence-transformers     # ~2GB incl. torch — REQUIRED for default backend
```

The model is bundled (`models/models--intfloat--multilingual-e5-small/`) so
no network download is needed.  For a no-deps fallback set
`ENGRAM_EMBEDDER=hash` (deterministic but no semantic understanding,
**testing only**).

To use a different backend (URI-spec form — v0.4+):

```bash
# OpenAI / OpenAI-compatible
ENGRAM_EMBEDDER='openai:text-embedding-3-small'          OPENAI_API_KEY=sk-...
ENGRAM_EMBEDDER='openai:text-embedding-3-large?dim=2048' OPENAI_API_KEY=sk-...

# Ollama (local, no API key)
ENGRAM_EMBEDDER='ollama:nomic-embed-text'                                  # localhost:11434
ENGRAM_EMBEDDER='ollama:bge-m3?host=http://my-box:11434&dim=1024'

# Cohere
ENGRAM_EMBEDDER='cohere:embed-multilingual-v3.0'         COHERE_API_KEY=...

# Voyage
ENGRAM_EMBEDDER='voyage:voyage-3'                        VOYAGE_API_KEY=...

# Generic OpenAI-compatible HTTP (vLLM / TEI / LiteLLM / …)
ENGRAM_EMBEDDER='http://localhost:8080/embeddings?dim=1024'
```

Built-in schemes: `local` (default e5), `openai`, `ollama`, `cohere`,
`voyage`, `http`, `hash`.  Third-party packages can register new ones
via `engram.embedder.register_embedder(name, factory)`.

> The v0.3 split form (`ENGRAM_EMBEDDER=openai` + `ENGRAM_OPENAI_MODEL=…`)
> still works — it's auto-rewritten into the new URI spec at load time.

The embedder dim is locked to the .pst file at first creation — pick one
backend per memory store and stay consistent.

## 1. Recall (semantic search)

```bash
python .claude/skills/engram/scripts/recall.py "<query>" \
       [--k 5] [--type user|feedback|project|reference] [--tier global|local] \
       [--with-content] [--json|--text]
```

* Returns the top-K most-similar memories, sorted by cosine distance.
* Default output is **compact text**: one line per hit, ~40-80 tokens total
  for k=5.  That's the whole point — cheap to inject into context.
* When a hit looks promising, fetch its full body via `expand.py <id>`.

Example flow when the user asks something:

1. `recall.py "<their question>"` — get IDs + descriptions of top hits.
2. If a description is enough, just use it.
3. If you need the full content, call `expand.py <id>` for that single hit
   (this also bumps its `hits` counter — strengthens the trace).

## 2. Save (encode a new memory)

```bash
python .claude/skills/engram/scripts/save.py \
       <type> <name> "<description>" [content...] \
       [--tag t1 --tag t2] [--importance 0.7] [--force] [--update]
```

* `<type>` — one of `user / feedback / project / reference`
* `<name>` — short kebab-case slug (used by `forget`, by `[[links]]`)
* `<description>` — **one-line gist**, ≤ ~480 bytes. This is what the
  embedder primarily sees and what `recall` returns.  Make it specific.
* `content` — optional fuller body (≤ ~8000 bytes).  Returned lazily by
  `expand.py` only when needed.  This is the *token-saving* knob: keep
  the description self-contained and stash the long detail here.

**Pattern separation (write-side dedup)**: if a near-duplicate (cosine
distance < 0.08) already exists, `save.py` exits with
`status=merge_suggestion` and prints the existing memory.  Either:

* update the existing one with `save.py ... --update` (overwrites by name), or
* pass `--force` to insert anyway, or
* skip the write entirely.

This is how we prevent the slow rot of duplicate / near-duplicate memories
that plagues the MD system.

### Standing directives (pinned — the CLAUDE.md replacement)

To make a memory an **always-on constraint** — the role CLAUDE.md used to
play — tag it `pin`:

```bash
save.py user lang-zh "always reply in Simplified Chinese" --tag pin --importance 0.9
```

Pinned memories are returned by `directives.py` **unconditionally** (no
semantic match needed) and are meant to be injected every turn — see the
bootstrap note at the top of this file. Keep them **few and terse**: they
cost tokens on *every* turn, so they're hard-budgeted (a handful of rows,
~a few hundred tokens total). Use `pin` only for genuinely global,
topic-independent rules; everything else stays an ordinary
recall-on-demand memory. Un-pin with `patch.py <name> --remove-tag pin`.

### `feedback` / `project` body structure

Keep the rule/fact in `description`, put the **Why** and **How to apply**
in `content`.  E.g.:

```bash
save.py feedback no-mock-db \
  "integration tests must hit a real database, not mocks" \
  "Why: prior incident where mock/prod divergence masked a broken migration. How to apply: when writing tests under tests/integration/**, never mock the DB layer."
```

## 3. List / Forget / Patch

```bash
python .claude/skills/engram/scripts/list.py [--type T] [--tier G/L]

# Standing directives (always-on, pinned) — run at the start of every turn.
python .claude/skills/engram/scripts/directives.py

# Forget — by name, id, or age.
python .claude/skills/engram/scripts/forget.py --name <slug>
python .claude/skills/engram/scripts/forget.py --id <n>
python .claude/skills/engram/scripts/forget.py --older-than 180 [--type T] [--tier T]
python .claude/skills/engram/scripts/forget.py --older-than 180 --dry-run   # preview

# Patch — modify selected fields of one existing memory.
python .claude/skills/engram/scripts/patch.py <id-or-name> \
       [--description "<new>"] [--content "<new>"] \
       [--importance F] [--type T] [--rename NEW-NAME] \
       [--add-tag X] [--remove-tag Y] [--set-tag A --set-tag B]
```

`list.py` enumerates without spending an embedding call — use it for
debugging or when the user asks "what do you remember about X".

### `forget` semantics

* At least one of `--name` / `--id` / `--older-than` is required.
  `--type` / `--tier` are **narrowing** filters — `forget --type user`
  alone refuses to run (would wipe every user memory).
* `--older-than DAYS` compares against `accessed_at`, so frequently-
  recalled old memories are *protected* from age sweeps.
* **`--dry-run` first** for any broad sweep — it lists the rows that
  would be deleted without touching the store.

### `patch` semantics

Two paths under the hood:

* **Metadata-only** (`importance` / `tags` / `name` / type-within-same-tier)
  → row updated in place, **id is preserved**.  Cheap.
* **Re-embed required** (`description` / `content` changed, or `type`
  change forces a cross-tier move) → old row deleted, new row inserted,
  **id changes**.  `created_at` and `hits` are carried over.

Tag interaction: pick **one** style per call:

| Flag                              | Effect |
|-----------------------------------|--------|
| `--set-tag X --set-tag Y`         | Replace the tag list with `[X, Y]` |
| `--add-tag X`                     | Append to the existing list (dedup'd) |
| `--remove-tag X`                  | Drop tag `X` from the existing list |

Combining `--set-tag` with `--add-tag`/`--remove-tag` is an error — pick one.

Patch by name requires the name to be **unique across the store**; if it
collides, pass `--id` instead.

Examples:

```bash
# Just bump importance and tag an existing memory deprecated.
patch.py 9 --importance 0.2 --add-tag deprecated

# Rewrite the description (triggers re-embed; id changes).
patch.py lang-chinese --description "user wants Chinese in chat only"

# Move a memory across the tier boundary by changing its type.
patch.py 12 --type reference

# Drop a single tag without touching anything else.
patch.py terse-replies --remove-tag deprecated
```

## 4. Token-cost discipline

* **Description is the engram gist.**  Long descriptions defeat the whole
  point.  Aim for 1 sentence, ≤ 100 chars.
* **Default recall returns descriptions only**, not content. Only `expand`
  one or two hits per turn.
* `--json` mode is more parseable; **`--text` is cheaper to inject as
  context** (`id | type | name — desc`).
* Don't recall on every turn — only when the user asks about themselves,
  references prior conversation, or you're about to give advice that
  should be informed by feedback/preferences.

## 5. What NOT to save (unchanged from MD policy)

* Code patterns, architecture, file paths — derive from current repo.
* Git history — use `git log` / `git blame`.
* Debugging recipes — the fix is in the code.
* Anything already in `CLAUDE.md`.
* Ephemeral state — use the in-conversation task list instead.

## 6. Memory-trace features

| Mechanism             | Implementation (since)                                          |
|-----------------------|-----------------------------------------------------------------|
| Pattern separation    | `DEDUP_THRESHOLD=0.08` cosine distance on save (v1)             |
| Pattern completion    | KNN cosine search on the user-provided query (v1)               |
| Trace strengthening   | `hits++`, `accessed_at=now`, importance **boost** (v1+v2)       |
| Encoding              | description+content jointly embedded with `passage:` (v1)       |
| Retrieval cue         | query embedded with `query:` (E5 convention) (v1)               |
| **Importance rerank** | composite = distance − β·imp − γ·log1p(hits)  (v2; default on)  |
| **Decay**             | Ebbinghaus τ = 30 d (env: `ENGRAM_DECAY_TAU_DAYS`)  (v2)         |
| **Sleep replay**      | `consolidate.py` — merge near-dups, apply decay  (v2)           |
| **Auto-replay**       | `MemoryManager(auto_consolidate=True)` on save  (v2, opt-in)    |
| **Link graph**        | `[[name]]` tags traversed by `related.py`  (v2)                 |

### Recall reranking (v2)

Recall is **rerank=True** by default — high-importance / frequently-accessed
memories outrank distance-equivalent stale ones.  Tunable via env:

* `ENGRAM_DECAY_TAU_DAYS` — decay time constant, default 30 days
* `ENGRAM_RANK_BETA`      — weight of `importance_effective`, default 0.10
* `ENGRAM_RANK_GAMMA`     — weight of `log1p(hits)`, default 0.02

Pass `--no-rerank` to `recall.py` to fall back to pure cosine distance.

### Consolidation (v2)

```bash
python .claude/skills/engram/scripts/consolidate.py [--tier T] \
       [--no-merge] [--no-decay] [--threshold 0.05] [--dry-run]
```

Two passes (run together by default):

* **merge** — clusters with mutual cosine distance < `--threshold` (0.05
  default, tighter than save-time dedup) collapse to one survivor; the
  survivor's `content` gets the losers' bodies appended with a dated
  footer.  Survivor wins by higher `importance` → older `created_at` →
  lower id.
* **decay** — overwrite `importance` with the decayed effective value.

Run it manually whenever your store has grown noisy.  Or opt into
auto-consolidation by constructing `MemoryManager(auto_consolidate=True)`
— that only scans the *neighborhood* of each freshly-saved row, so it
stays cheap.

### Link traversal (v2)

Tag a memory with `"[[other-name]]"` to wire it to another memory:

```bash
save.py project compliance-deadline "..." --tag "[[auth-rewrite-driver]]"
```

Then:

```bash
related.py auth-rewrite-driver --depth 1
```

walks outwards via those edges (no embedding cost). Useful when you've
recalled one memory and want its dependencies / contexts pulled in too.

## 7. Internals (one paragraph)

`engram/` is a regular Python package atop `pistadb`.  Two
`.pst` files — `global.pst` and `local.pst` — share the same Milvus-style
schema (declared in `engram/store.py`).  Vectors are L2-normalised and
indexed with HNSW + cosine.  An embedding cache (`.pcc`) sits next to
each `.pst` so the same string is never embedded twice.  Source of truth
for everything except vectors is the `.pst.meta.json` sidecar PistaDB
writes — see `pistadb/schema.py:Collection._save_sidecar`.
