"""Engram — vector memory traces for AI agents.

Public API:

    from engram import MemoryManager, MemoryType
    from engram.embedder import LocalE5Embedder, OpenAIEmbedder, HashEmbedder
    from engram.consolidate import consolidate, merge_pass, decay_pass

    mgr = MemoryManager()                  # auto-detects tiers + embedder
    mgr.save("user", "role-data-sci",
             description="user is a data scientist focused on logging",
             content="...")
    hits = mgr.recall("how should I explain logs?", k=5)   # rerank=True by default
    related = mgr.recall_related("role-data-sci", depth=1)
    consolidate(mgr, dry_run=True)         # report what would change
"""
from __future__ import annotations

from .consolidate import (
    ConsolidateReport,
    MERGE_THRESHOLD,
    MergeAction,
    consolidate,
    decay_pass,
    merge_pass,
)
from .decay import (
    boost_on_access,
    composite_score,
    importance_effective,
    rerank as rerank_hits,
)
from .memory import (
    DEDUP_THRESHOLD,
    MemoryHit,
    MemoryManager,
    MemoryType,
    SaveResult,
)

__all__ = [
    "MemoryManager",
    "MemoryHit",
    "MemoryType",
    "SaveResult",
    "DEDUP_THRESHOLD",
    # v2
    "MERGE_THRESHOLD",
    "ConsolidateReport",
    "MergeAction",
    "consolidate",
    "merge_pass",
    "decay_pass",
    "rerank_hits",
    "importance_effective",
    "composite_score",
    "boost_on_access",
]
