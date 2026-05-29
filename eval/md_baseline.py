"""Simulate the legacy MEMORY.md + per-file MD memory retrieval.

The legacy system, as described in Claude Code's built-in ``auto memory``
prompt, works like:

1. ``MEMORY.md`` is **always loaded** into Claude's context.  It is an
   index — one line per memory in the form
   ``- [name](file.md) — one-line description``.
2. When Claude judges a query relevant, it ``Read``s one or more
   ``<slug>.md`` files.  Each file has YAML-ish frontmatter (name,
   description, type) plus a body.

We model retrieval as a keyword scoring step: pick the top-N candidate
files by Jaccard overlap of the query tokens with each index line's
description.  This is a faithful approximation of how Claude actually
picks files to read in practice.

Both files (the index and the read MD files) are counted toward the
system's token cost.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from .corpus import Memory


# ── Build the artifacts the legacy system would have on disk ─────────────────

def build_memory_md(memories: Iterable[Memory]) -> str:
    """The always-loaded ``MEMORY.md`` index."""
    lines = ["# Memory index"]
    for m in memories:
        lines.append(f"- [{m.name}]({m.name}.md) — {m.description}")
    return "\n".join(lines) + "\n"


def build_md_file(m: Memory) -> str:
    """One memory's standalone ``.md`` file (what Claude would Read)."""
    return (
        "---\n"
        f"name: {m.name}\n"
        f"description: {m.description}\n"
        "metadata:\n"
        f"  type: {m.type}\n"
        "---\n\n"
        f"{m.content}\n"
    )


# ── Retrieval: keyword scoring on the index lines ────────────────────────────

_CJK_RE = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


def _tokenize(s: str) -> List[str]:
    """Tokenize for keyword matching: word-level English + char-level CJK.

    Lowercased English words ≥ 2 chars (avoid noise from 'a', 'i').
    Each CJK glyph is its own token.  Matches how a model scanning the
    index would notice keywords.
    """
    cjk = _CJK_RE.findall(s)
    words = [w.lower() for w in _WORD_RE.findall(s) if len(w) >= 2]
    # Drop common English stopwords that don't help with matching.
    return [w for w in (words + cjk) if w not in _STOPWORDS]


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "of", "to", "for", "on", "in", "at",
    "by", "with", "as", "it", "this", "that", "these", "those",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "do", "does", "did", "i", "you", "they", "we", "user", "should",
    "will", "would", "can", "could", "may", "might", "have", "has",
    "had", "my", "your", "their", "our",
}


def _jaccard(q_tokens: List[str], doc_tokens: List[str]) -> float:
    if not q_tokens or not doc_tokens:
        return 0.0
    q = set(q_tokens)
    d = set(doc_tokens)
    return len(q & d) / len(q | d)


def md_retrieve(
    query: str, memories: List[Memory], top_n: int
) -> List[Tuple[Memory, float]]:
    """Return the top-N memories Claude would read for ``query``.

    Sort key: Jaccard overlap of query tokens with the index line's
    description tokens (descending).  Ties broken by name (stable).
    """
    q_tokens = _tokenize(query)
    scored: List[Tuple[float, Memory]] = []
    for m in memories:
        d_tokens = _tokenize(m.description)
        s = _jaccard(q_tokens, d_tokens)
        scored.append((s, m))
    # Stable-sort: higher score first, then alphabetical.
    scored.sort(key=lambda t: (-t[0], t[1].name))
    return [(m, s) for s, m in scored[:top_n]]
