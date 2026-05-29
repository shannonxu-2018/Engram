"""Storage layer for Engram.

Wraps PistaDB :class:`Collection` and resolves the two-tier paths:

* **global** tier (``~/.claude/engram/global.pst``) — types ``user`` and
  ``feedback``.  Shared across all projects so personal preferences and
  collaboration style follow the user.
* **local** tier (``<cwd>/.claude/engram/local.pst``) — types ``project``
  and ``reference``.  Project-specific facts stay with the project.

Both tiers share the same schema and embedder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from pistadb import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Index,
    Metric,
    create_collection,
    load_collection,
)


# ── Schema ────────────────────────────────────────────────────────────────────

VECTOR_DIM_DEFAULT = 384  # matches multilingual-e5-small


def make_schema(dim: int = VECTOR_DIM_DEFAULT) -> CollectionSchema:
    """Schema shared by both tiers.  Vector dim must match the embedder."""
    return CollectionSchema(
        fields=[
            FieldSchema("mem_id",      DataType.INT64,   is_primary=True, auto_id=True),
            FieldSchema("type",        DataType.VARCHAR, max_length=20),
            FieldSchema("name",        DataType.VARCHAR, max_length=128),
            FieldSchema("description", DataType.VARCHAR, max_length=512),
            FieldSchema("content",     DataType.VARCHAR, max_length=8192),
            FieldSchema("tags",        DataType.JSON),
            FieldSchema("importance",  DataType.FLOAT),
            FieldSchema("hits",        DataType.INT64),
            FieldSchema("created_at",  DataType.INT64),
            FieldSchema("accessed_at", DataType.INT64),
            FieldSchema("vector",      DataType.FLOAT_VECTOR, dim=dim),
        ],
        description="Engram v1 collection",
    )


# ── Tier paths ────────────────────────────────────────────────────────────────

GLOBAL_TYPES = frozenset({"user", "feedback"})
LOCAL_TYPES  = frozenset({"project", "reference"})


def _engram_home() -> Path:
    """Return the directory that hosts the **global** tier.

    Resolution order (first hit wins):

    1. ``ENGRAM_HOME`` — explicit override.  Useful when running Engram
       under a non-Claude agent (OpenCode / Codex / future) and you want
       the global store somewhere unrelated to ``~/.claude``.
    2. ``CLAUDE_HOME`` — back-compat with v0.3 and earlier.
    3. ``~/.claude`` — historical default.

    The global tier itself always sits at ``<home>/engram/`` regardless
    of which of these resolves.  Both ``ENGRAM_HOME`` and ``CLAUDE_HOME``
    accept ``~`` and are expanded.

    Empty-string env vars are treated as **unset** — without that guard,
    ``X=""`` would resolve to the CWD via ``Path("").expanduser()`` and
    silently install the global tier under wherever the CLI was launched.
    """
    for var in ("ENGRAM_HOME", "CLAUDE_HOME"):
        raw = os.environ.get(var)
        if raw:                    # excludes both ``None`` and ``""``
            return Path(raw).expanduser()
    return Path.home() / ".claude"


# Backwards-compatible alias — older internal callers reach for this name.
_claude_home = _engram_home


def global_dir() -> Path:
    return _engram_home() / "engram"


def local_dir(project_root: Optional[Path] = None) -> Path:
    """``<project>/.claude/engram/`` — derived from ``project_root`` or CWD."""
    root = Path(project_root) if project_root else Path.cwd()
    return root / ".claude" / "engram"


@dataclass(frozen=True)
class TierPaths:
    name: str            # "global" | "local"
    pst:  Path           # vector store
    cache: Path          # embedding cache (.pcc)


def tier_for(mem_type: str) -> str:
    if mem_type in GLOBAL_TYPES:
        return "global"
    if mem_type in LOCAL_TYPES:
        return "local"
    raise ValueError(
        f"Unknown memory type {mem_type!r}; "
        f"expected one of {sorted(GLOBAL_TYPES | LOCAL_TYPES)}"
    )


def resolve_tiers(project_root: Optional[Path] = None) -> Dict[str, TierPaths]:
    """Return both tiers' paths.  Directories are created if missing."""
    g_dir = global_dir()
    l_dir = local_dir(project_root)
    g_dir.mkdir(parents=True, exist_ok=True)
    l_dir.mkdir(parents=True, exist_ok=True)
    return {
        "global": TierPaths(
            name="global",
            pst=g_dir / "global.pst",
            cache=g_dir / "global.pcc",
        ),
        "local": TierPaths(
            name="local",
            pst=l_dir / "local.pst",
            cache=l_dir / "local.pcc",
        ),
    }


# ── Tier handle ───────────────────────────────────────────────────────────────

class Tier:
    """One PistaDB :class:`Collection` representing a single tier."""

    def __init__(self, paths: TierPaths, dim: int = VECTOR_DIM_DEFAULT):
        self.paths = paths
        self._dim = dim
        meta = str(paths.pst) + ".meta.json"
        pst_exists  = Path(paths.pst).exists()
        meta_exists = Path(meta).exists()
        # PistaDB stores vectors in the .pst and schema/row metadata in a
        # .meta.json sidecar.  They are written atomically together; if
        # exactly one is present the store is corrupt and we must not
        # silently create a fresh collection over the surviving file.
        if pst_exists != meta_exists:
            raise RuntimeError(
                f"Inconsistent tier state for {paths.name}: "
                f"{paths.pst}={'present' if pst_exists else 'missing'}, "
                f"{meta}={'present' if meta_exists else 'missing'}. "
                f"Restore the missing file from backup, or delete both to "
                f"start fresh."
            )
        if meta_exists:
            self._coll = load_collection(path=str(paths.pst))
        else:
            self._coll = create_collection(
                name=paths.name,
                fields=make_schema(dim),
                description=f"Engram {paths.name} tier",
                metric=Metric.COSINE,
                index=Index.HNSW,
                path=str(paths.pst),
            )

    # ── Forwarding ─────────────────────────────────────────────────────────

    @property
    def collection(self) -> Collection:
        return self._coll

    @property
    def count(self) -> int:
        return self._coll.num_entities

    def flush(self) -> None:
        self._coll.flush()

    def close(self) -> None:
        self._coll.close()

    def __enter__(self) -> "Tier":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = [
    "VECTOR_DIM_DEFAULT",
    "GLOBAL_TYPES",
    "LOCAL_TYPES",
    "TierPaths",
    "Tier",
    "tier_for",
    "resolve_tiers",
    "make_schema",
    "global_dir",
    "local_dir",
]
