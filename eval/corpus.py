"""Synthetic but realistic memory corpus + ground-truth query set.

40 memories spread across the four types; 25 queries each annotated with
the memory ``name(s)`` that should be retrieved.  Difficulty levels:

* **easy**  — query reuses keywords from the description
* **medium** — query paraphrases the description
* **hard**  — query is only thematically related; requires semantic match

Both eval systems are scored against the same ground-truth labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Memory:
    name: str
    type: str
    description: str
    content: str


@dataclass
class Query:
    text: str
    targets: List[str]      # ground-truth memory name(s)
    difficulty: str         # "easy" | "medium" | "hard"


# ── 40 memories ──────────────────────────────────────────────────────────────

MEMORIES: List[Memory] = [
    # ── user (8) ─────────────────────────────────────────────────────────────
    Memory("u-role-backend-go", "user",
           "user is a senior backend engineer with 10 years of Go experience",
           "Background: led the payments team at a fintech. Currently exploring frontend work in React. Prefers Go-flavored explanations for systems concepts."),
    Memory("u-lang-chinese", "user",
           "user prefers Simplified Chinese for chat and documentation, English for code identifiers",
           "Why: native language efficiency. Technical API names should stay English (e.g. fetch, await, HashMap)."),
    Memory("u-platform-windows", "user",
           "user works on Windows 11 with Python from miniconda3",
           "The system PATH python.exe is a Microsoft Store stub that blocks; use /c/Users/pgdni/miniconda3/python.exe directly in shell commands."),
    Memory("u-tz-shanghai", "user",
           "user is in Asia/Shanghai timezone, UTC+8",
           "Convert all 'tomorrow' / 'next Friday' to absolute dates relative to UTC+8 when saving project memories with deadlines."),
    Memory("u-name-tony", "user",
           "user goes by Tony in chat, but signs commits as anthony.li",
           "Use 'Tony' in conversational replies. Commit metadata should preserve 'anthony.li@example.com' if asked to author commits."),
    Memory("u-vim-keys", "user",
           "user uses vim-style keybindings everywhere including the browser (Vimium)",
           "Esc / hjkl / yy. Avoid suggesting GUI-only workflows; prefer keyboard shortcuts when offering instructions."),
    Memory("u-coffee-no-meeting-am", "user",
           "user does deep work in the morning and dislikes meetings before 11am local",
           "When proposing schedules, push meetings to afternoons when possible."),
    Memory("u-blind-spot-css", "user",
           "user has stated that CSS layout is the weakest area of their frontend knowledge",
           "When explaining frontend changes that touch layout, add brief CSS notes (display/flex/grid) instead of assuming."),

    # ── feedback (12) ────────────────────────────────────────────────────────
    Memory("f-no-mock-db", "feedback",
           "integration tests must hit a real database, not mocks",
           "Why: a prior incident where mock/prod divergence masked a broken migration. How to apply: under tests/integration/**, never mock the DB layer; spin up a postgres container in setup."),
    Memory("f-terse-replies", "feedback",
           "user wants terse responses with no trailing summaries",
           "Why: aesthetic. How to apply: skip the 'here is what I did' recap at end of turn; the diff already shows that."),
    Memory("f-bundled-pr", "feedback",
           "for refactors in this area, prefer one bundled PR over many small ones",
           "Why: small-PR ceremony costs more than it saves on tightly-coupled refactors. How to apply: when changes are inter-dependent, ship them in one PR with a structured commit history rather than chaining draft PRs."),
    Memory("f-no-emoji", "feedback",
           "user dislikes emojis in code or chat output unless explicitly asked",
           "Why: aesthetic preference. How to apply: skip emojis in all assistant output; if a library convention forces one, mention and leave it."),
    Memory("f-conservative-deps", "feedback",
           "user is conservative about adding new third-party dependencies",
           "Why: prior pain with abandoned packages and CVE chains. How to apply: prefer stdlib; if a dep is needed, justify it in the PR description."),
    Memory("f-fix-root-cause", "feedback",
           "always investigate root cause; do not paper over symptoms",
           "Why: prior incident where a retry-on-error masked a real auth bug for weeks. How to apply: if a fix feels like 'just retry / try-except / sleep', flag it and dig further."),
    Memory("f-name-snake-case", "feedback",
           "user prefers snake_case Python variable names, camelCase only at JS boundaries",
           "Why: codebase convention. How to apply: when generating Python, use snake_case even for ported JS code; only keep camelCase at the FFI/HTTP boundary."),
    Memory("f-explain-tradeoffs", "feedback",
           "when proposing a design, explicitly state the main tradeoff",
           "Why: user prefers calibrated recommendations. How to apply: every design suggestion should end with one line on what is given up by this choice."),
    Memory("f-no-trailing-questions", "feedback",
           "do not end every assistant turn with 'is there anything else?'",
           "Why: user finds it noisy. How to apply: end on the last useful sentence; let the user steer."),
    Memory("f-prefer-explicit-types", "feedback",
           "user wants explicit type annotations in Python (PEP 484/695), no implicit Any",
           "Why: large refactors safer with mypy strict. How to apply: every new function gets parameter and return annotations; rewrite Any to a concrete or generic type."),
    Memory("f-test-naming-given-when", "feedback",
           "test names should follow the given/when/then or arrange/act/assert pattern",
           "Why: readable test reports. How to apply: write test_<unit>_when_<condition>_<expected>; do not write test_<unit>_works."),
    Memory("f-no-force-push-shared", "feedback",
           "never force-push to a branch that another contributor has been on",
           "Why: prior incident lost a teammate's commits. How to apply: only force-push to personal feature branches you own end-to-end; use --force-with-lease at minimum."),

    # ── project (12) ─────────────────────────────────────────────────────────
    Memory("p-repo-purpose-memskill", "project",
           "memskill is a vector-DB-backed memory system that replaces the legacy MEMORY.md Claude memory",
           "Target replacement: ~/.claude/projects/*/memory/*.md. Motivation: cut context-window cost and improve retrieval precision."),
    Memory("p-stack-pistadb-e5", "project",
           "this repo uses PistaDB (embedded vector DB) plus a bundled multilingual-e5-small embedding model",
           "PistaDB lives in pistadb/; the model is at models/models--intfloat--multilingual-e5-small/. 384-dim vectors, COSINE metric, HNSW index."),
    Memory("p-tier-storage", "project",
           "memory storage is split into a global tier and a project-local tier",
           "Global: ~/.claude/engram/global.pst holds user+feedback. Local: <repo>/.claude/engram/local.pst holds project+reference. Why: personal prefs follow the user; project facts stay local."),
    Memory("p-mvp-scope-v1", "project",
           "v1 scope is the MVP closed loop: save / recall / expand / forget / list, plus write-time dedup",
           "Completed 2026-05-26. Source: engram/memory.py. Consolidation deferred to v2 by design."),
    Memory("p-v2-scope", "project",
           "v2 added importance-weighted recall rerank, Ebbinghaus decay, consolidate (merge+decay), and link traversal",
           "Implementation: engram/decay.py and engram/consolidate.py. Recall defaults to rerank=True since v2."),
    Memory("p-freeze-2026-03-05", "project",
           "merge freeze begins 2026-03-05 due to the mobile team cutting a release branch",
           "Non-critical PRs scheduled after that date should be flagged. Only emergency fixes can land between freeze and the release tag."),
    Memory("p-auth-rewrite-compliance", "project",
           "the auth middleware rewrite is driven by legal compliance, not tech-debt cleanup",
           "Why: prior session-token storage flagged by legal as non-compliant. Scope decisions should favor compliance over ergonomics."),
    Memory("p-ingest-pipeline-owner", "project",
           "the ingest pipeline is owned by the data infra team; ping #data-infra in Slack for changes",
           "Do not modify ingest/* without an approving review from a #data-infra member; their oncall rotation page is on PagerDuty."),
    Memory("p-py-version-3-11", "project",
           "this repo targets Python 3.11 (frozen by the CI base image)",
           "Do not use 3.12+ syntax (PEP 695 type statement, PEP 696 default type vars) until CI image bumps."),
    Memory("p-release-tag-pattern", "project",
           "release tags follow vMAJOR.MINOR.PATCH-rcN and are cut from main",
           "RC tags trigger the canary deploy; the final v* tag triggers prod. Hot-fix branches use vX.Y.Z+hotfix.N."),
    Memory("p-feature-flag-launch-default-off", "project",
           "new features ship behind a LaunchDarkly flag, defaulting to OFF in production",
           "Why: blast radius control. How to apply: every feature PR must include the flag key in the description; ramp via LaunchDarkly UI, not code defaults."),
    Memory("p-data-retention-90d", "project",
           "raw event data has a 90-day retention policy in the warehouse",
           "Aggregations beyond 90 days must derive from rolled-up tables, not raw events. The retention job runs nightly at 03:30 UTC."),

    # ── reference (8) ────────────────────────────────────────────────────────
    Memory("r-pistadb-skill", "reference",
           "PistaDB usage notes (API surface, index tradeoffs, params) are in .claude/skills/pistadb/SKILL.md",
           "When writing code that uses PistaDB directly, read that skill first; do not invent APIs beyond §1."),
    Memory("r-grafana-api-latency", "reference",
           "grafana.internal/d/api-latency is the oncall API latency dashboard",
           "Watch when editing request-path code. p95 threshold for paging is 800ms over 5 minutes."),
    Memory("r-linear-ingest-project", "reference",
           "ingest pipeline bug tickets live in Linear project INGEST",
           "URL stub: linear.app/<org>/team/INGEST. Tag urgent issues with 'incident' so oncall picks them up."),
    Memory("r-runbook-db-failover", "reference",
           "the DB failover runbook is at runbooks.internal/db/failover.md",
           "Procedure to promote the replica when primary is down. Step 4 (DNS swap) needs SRE sign-off."),
    Memory("r-slack-data-infra", "reference",
           "data infra team's primary Slack channel is #data-infra",
           "For ingest/* questions and incident coordination. Out-of-hours: page via PagerDuty rotation 'data-infra-oncall'."),
    Memory("r-docs-style-guide", "reference",
           "the company tech writing style guide is at docs.internal/style/v3",
           "Prefer present tense, sentence case headings, oxford comma off. PR descriptions follow the same."),
    Memory("r-launchdarkly-dashboard", "reference",
           "LaunchDarkly feature flag dashboard is at app.launchdarkly.com/<org>/production",
           "Ramp percentages should match the rollout plan in the PR description. Audit log is the source of truth for flag changes."),
    Memory("r-pagerduty-on-call", "reference",
           "current oncall rotation is visible at pagerduty.com/<org>/schedules",
           "Bookmark: 'platform-oncall' is the primary rotation; 'data-infra-oncall' covers ingest."),
]


# ── 25 queries ────────────────────────────────────────────────────────────────

QUERIES: List[Query] = [
    # ── easy (10) — keyword overlap with description ─────────────────────────
    Query("what timezone is the user in",
          ["u-tz-shanghai"], "easy"),
    Query("does the user prefer Chinese or English",
          ["u-lang-chinese"], "easy"),
    Query("how should I write integration tests",
          ["f-no-mock-db"], "easy"),
    Query("should I add emojis to my reply",
          ["f-no-emoji"], "easy"),
    Query("when does the merge freeze begin",
          ["p-freeze-2026-03-05"], "easy"),
    Query("which Python version does the repo target",
          ["p-py-version-3-11"], "easy"),
    Query("where is the api latency dashboard",
          ["r-grafana-api-latency"], "easy"),
    Query("what is the runbook for db failover",
          ["r-runbook-db-failover"], "easy"),
    Query("how should I name unit tests",
          ["f-test-naming-given-when"], "easy"),
    Query("can I force push to shared branches",
          ["f-no-force-push-shared"], "easy"),

    # ── medium (10) — paraphrase / partial keyword ───────────────────────────
    Query("the user uses Windows right? which python should I call",
          ["u-platform-windows"], "medium"),
    Query("is the user fluent in Go",
          ["u-role-backend-go"], "medium"),
    Query("do not just patch the symptom; what does the user want",
          ["f-fix-root-cause"], "medium"),
    Query("should we cap how many external libraries we pull in",
          ["f-conservative-deps"], "medium"),
    Query("how do I gate a new feature behind a toggle",
          ["p-feature-flag-launch-default-off"], "medium"),
    Query("how long do raw events stick around in the warehouse",
          ["p-data-retention-90d"], "medium"),
    Query("which team handles changes to the ingest pipeline",
          ["p-ingest-pipeline-owner", "r-slack-data-infra"], "medium"),
    Query("where should new INGEST bugs be filed",
          ["r-linear-ingest-project"], "medium"),
    Query("are type hints required on new Python functions",
          ["f-prefer-explicit-types"], "medium"),
    Query("does the user want a meeting at 9am",
          ["u-coffee-no-meeting-am"], "medium"),

    # ── hard (5) — only thematic; needs semantic understanding ───────────────
    Query("the user told me to stop wrapping up with 'anything else?' lines",
          ["f-no-trailing-questions"], "hard"),
    Query("what's the protocol around shipping a tightly coupled refactor",
          ["f-bundled-pr"], "hard"),
    Query("the auth project — is it tech debt or something else driving it",
          ["p-auth-rewrite-compliance"], "hard"),
    Query("how do release tags map to deploys",
          ["p-release-tag-pattern"], "hard"),
    Query("what's the convention for naming variables in this codebase",
          ["f-name-snake-case"], "hard"),
]


def by_name(memories: List[Memory]) -> dict:
    return {m.name: m for m in memories}
