"""Engram evaluation harness.

A head-to-head benchmark comparing:

* **MD baseline** — the legacy ``MEMORY.md`` + per-file ``.md`` retrieval
  Claude Code uses by default.  Modeled as: ``MEMORY.md`` is always in
  context; per-query, Claude reads the top-N most plausible ``.md``
  files (keyword overlap on the index line).
* **Engram** — the system in this repo.  No always-loaded index;
  per-query, ``recall.py`` returns compact ``(id | type | name | desc)``
  lines, with optional ``expand.py`` lookup for the body.

Both systems answer the same query set against the same memory corpus,
and we report accuracy@k, MRR, and token cost.
"""
