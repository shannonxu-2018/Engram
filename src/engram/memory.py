"""MemoryManager — the public façade.

Hippocampus analogy:

* ``save()``    — *encoding*: write a new engram (description + content +
  embedding).  Performs **pattern separation**: if the closest existing
  memory is within :data:`DEDUP_THRESHOLD` cosine distance, the call
  returns a ``MERGE_SUGGESTION`` instead of inserting, unless ``force=True``.
* ``recall()``  — *pattern completion*: KNN search against the query
  embedding.  Returns compact :class:`MemoryHit` objects (no full content
  by default — call :meth:`expand` to fetch the body).
* ``expand()``  — pull the full ``content`` for a hit; bumps ``hits`` /
  ``accessed_at`` (analogous to memory reactivation strengthening a trace).
* ``forget()``  — delete by name or id.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .decay import boost_on_access, rerank as _rerank_hits
from .embedder import CachedEmbedder, Embedder, build_default_embedder
from .store import (
    GLOBAL_TYPES,
    LOCAL_TYPES,
    Tier,
    TierPaths,
    VECTOR_DIM_DEFAULT,
    resolve_tiers,
    tier_for,
)


# ── Types & constants ────────────────────────────────────────────────────────

VALID_TYPES = GLOBAL_TYPES | LOCAL_TYPES

# Cosine **distance** = 1 − cosine similarity.  PistaDB returns distance.
# Two memories are "near-duplicates" if they're within this distance.
# 0.08 ≈ cosine sim > 0.92, empirically a tight match for e5.
DEDUP_THRESHOLD = 0.08

# ── Standing directives ("always-on" constraints) ────────────────────────────
# A memory tagged PIN_TAG is a *standing directive*: a CLAUDE.md-style global
# constraint that must apply on every turn regardless of the current topic
# (e.g. "always reply in Simplified Chinese").  Unlike ordinary memories it is
# NOT retrieved by semantic recall — `directives()` pulls every pinned memory
# unconditionally.  Because that cost is paid on *every* turn, it is hard-
# budgeted (see SKILL.md §4 token discipline): keep pinned memories few and
# terse, or this degrades into another bloated MEMORY.md.
PIN_TAG               = "pin"
DIRECTIVES_MAX        = 8       # never inject more than this many directives
DIRECTIVES_BYTE_BUDGET = 1200   # total description bytes ≈ a few hundred tokens

# Cap returned content snippet so descriptions stay token-cheap.
DESCRIPTION_MAX_BYTES = 480     # leaves headroom under VARCHAR 512
CONTENT_MAX_BYTES     = 8000    # leaves headroom under VARCHAR 8192

# ── Adaptive-k recall (gap-based knee detection) ─────────────────────────────
# When ``recall(k=None)`` (the default), we don't truncate at a fixed K — we
# fetch a generous candidate pool, sort by raw distance, locate the
# **largest** gap among the first K_MAX hits, and cut there if that gap is
# (a) absolutely large enough to be meaningful and (b) significantly bigger
# than the typical gap in the rest of the pool.  Otherwise we fall back to
# K_MIN (when nothing stands out — don't dump noise into context).
K_MIN              = 2      # always return at least this many
K_MAX              = 10     # never return more than this
GAP_FACTOR         = 2.0    # max gap must be > GAP_FACTOR × median of other gaps
MIN_SIGNIFICANT_GAP = 0.01  # in cosine distance units; below this everything is "same cluster"
ADAPTIVE_FETCH_BUDGET = 20  # candidates fetched per tier when adaptive


def _adaptive_k(
    sorted_distances: Sequence[float],
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    gap_factor: float = GAP_FACTOR,
    min_significant_gap: float = MIN_SIGNIFICANT_GAP,
) -> int:
    """Pick the cut point on a distance-sorted candidate list.

    Algorithm:

    1. If we have ``<= k_min`` candidates, return all of them.
    2. Look at the first ``k_max`` candidates.  Compute the gaps between
       consecutive distances.
    3. Find the **largest** gap.  If it's smaller than
       ``min_significant_gap`` (absolute), the candidates are all in the
       same cluster — nothing stands out — so return ``k_min``.
    4. Check that the largest gap is at least ``gap_factor`` times the
       median of the other gaps.  If yes, cut right before it.  If no
       (distances spread evenly, no sharp cliff), **also return ``k_min``**
       — better to inject fewer high-confidence hits than to dump the
       whole pool of marginally-relevant noise.
    5. Result is always clamped to ``[k_min, k_max]``.
    """
    n = len(sorted_distances)
    if n == 0:
        return 0
    if n <= k_min:
        return n

    horizon = min(n, k_max)
    gaps = [
        sorted_distances[i + 1] - sorted_distances[i]
        for i in range(horizon - 1)
    ]
    if not gaps:
        return horizon

    max_gap = max(gaps)
    # No meaningful separation anywhere — everything is in the same noise
    # band.  Give the caller the floor.
    if max_gap < min_significant_gap:
        return k_min

    max_gap_idx = gaps.index(max_gap)

    # Is the max gap actually stark vs the rest?
    other_gaps = gaps[:max_gap_idx] + gaps[max_gap_idx + 1:]
    if other_gaps:
        median_other = sorted(other_gaps)[len(other_gaps) // 2]
        if median_other > 0 and max_gap < gap_factor * median_other:
            # No clear cliff.  Conservative default: return the floor
            # rather than dump the whole pool of marginal hits.
            return k_min

    cut = max_gap_idx + 1  # keep positions 0 .. max_gap_idx inclusive
    return max(k_min, min(cut, k_max))


class MemoryType:
    """String constants — use plain strings everywhere else."""
    USER      = "user"
    FEEDBACK  = "feedback"
    PROJECT   = "project"
    REFERENCE = "reference"


# ── Public records ────────────────────────────────────────────────────────────

@dataclass
class MemoryHit:
    """One returned memory.  Compact by default; ``content`` is ``None``
    unless ``with_content=True`` was passed to :meth:`MemoryManager.recall`.
    """
    id:          int
    tier:        str       # "global" | "local"
    type:        str
    name:        str
    description: str
    distance:    float     # cosine distance, smaller = closer
    importance:  float
    hits:        int
    created_at:  int
    accessed_at: int
    tags:        List[str] = field(default_factory=list)
    content:     Optional[str] = None

    def to_compact(self) -> Dict[str, Any]:
        """Token-cheap dict — use for returning to the LLM."""
        return {
            "id":   self.id,
            "type": self.type,
            "name": self.name,
            "desc": self.description,
            "dist": round(self.distance, 4),
        }

    def to_dict(self) -> Dict[str, Any]:
        # NB: distance / importance often come in as ``numpy.float32`` from
        # the PistaDB search path.  ``round()`` preserves that dtype, but
        # ``json.dumps`` chokes on numpy scalars — every MCP tool call
        # serialising a hit would land in the isError branch.  Force a
        # Python ``float`` here so the dict is JSON-safe end-to-end.
        d = {
            "id":          int(self.id),
            "tier":        self.tier,
            "type":        self.type,
            "name":        self.name,
            "description": self.description,
            "distance":    round(float(self.distance), 6),
            "importance":  round(float(self.importance), 4),
            "hits":        int(self.hits),
            "created_at":  int(self.created_at),
            "accessed_at": int(self.accessed_at),
            "tags":        list(self.tags),
        }
        if self.content is not None:
            d["content"] = self.content
        return d


@dataclass
class SaveResult:
    """Outcome of :meth:`MemoryManager.save`."""
    status:      str                  # "inserted" | "merge_suggestion" | "overwritten"
    id:          Optional[int]        # id if inserted/overwritten
    tier:        Optional[str]
    duplicate_of: Optional[MemoryHit] = None  # set when status == "merge_suggestion"

    def to_dict(self) -> Dict[str, Any]:
        d = {"status": self.status, "id": self.id, "tier": self.tier}
        if self.duplicate_of is not None:
            d["duplicate_of"] = self.duplicate_of.to_compact()
        return d


# ── Manager ───────────────────────────────────────────────────────────────────

class MemoryManager:
    """Top-level entry point.  One instance per process.

    Owns the embedder + both tier handles, and exposes save / recall /
    expand / forget.  Always pair with a ``close()`` or use ``with``.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        project_root: Optional[Path] = None,
        dim: Optional[int] = None,
        auto_consolidate: bool = False,
    ):
        tiers = resolve_tiers(project_root)
        if embedder is None:
            embedder = build_default_embedder(
                cache_path=str(tiers["global"].cache)
            )
        elif not isinstance(embedder, CachedEmbedder):
            # Always wrap with a cache; use the global tier's .pcc.
            embedder = CachedEmbedder(
                embedder, cache_path=str(tiers["global"].cache)
            )
        self._embedder = embedder

        chosen_dim = dim if dim is not None else embedder.dim
        if chosen_dim != embedder.dim:
            raise ValueError(
                f"dim={chosen_dim} does not match embedder dim={embedder.dim}"
            )

        self._tiers: Dict[str, Tier] = {
            name: Tier(paths, dim=chosen_dim)
            for name, paths in tiers.items()
        }
        # Verify on-disk dim matches embedder.  This catches the most
        # common surprise after a backend switch — the old .pst is still
        # there from a previous embedder and the vectors don't fit.
        for t in self._tiers.values():
            stored_dim = t.collection.schema.vector_field.dim
            if stored_dim != chosen_dim:
                current = os.environ.get("ENGRAM_EMBEDDER") or "(unset → local e5)"
                raise ValueError(
                    f"Embedder dim mismatch for {t.paths.pst}:\n"
                    f"  on-disk dim   : {stored_dim}\n"
                    f"  current dim   : {chosen_dim}\n"
                    f"  ENGRAM_EMBEDDER={current!r}\n"
                    f"\n"
                    f"That .pst was created by a different embedder. Pick "
                    f"one of these fixes:\n"
                    f"  (a) Switch back to the original embedder so dims "
                    f"match (whichever backend produced dim={stored_dim}).\n"
                    f"  (b) Delete this tier's files to start fresh "
                    f"(keeps the other tier intact):\n"
                    f"        rm {t.paths.pst} {t.paths.pst}.meta.json "
                    f"{t.paths.cache}\n"
                    f"  (c) Re-embed every memory by exporting → "
                    f"deleting → re-saving under the new backend. There "
                    f"is no built-in re-embed tool yet (v0.5 plan)."
                )

        self.auto_consolidate = bool(auto_consolidate)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def __enter__(self) -> "MemoryManager":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def flush(self) -> None:
        for t in self._tiers.values():
            t.flush()
        if hasattr(self._embedder, "flush"):
            self._embedder.flush()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            for t in self._tiers.values():
                try:
                    t.close()
                except Exception:
                    pass
            try:
                if hasattr(self._embedder, "close"):
                    self._embedder.close()
            except Exception:
                pass

    # ── Public API: save ──────────────────────────────────────────────────

    def save(
        self,
        type: str,
        name: str,
        description: str,
        content: str = "",
        tags: Optional[Sequence[str]] = None,
        importance: float = 0.5,
        force: bool = False,
        overwrite_by_name: bool = False,
    ) -> SaveResult:
        """Encode a new memory.

        Parameters
        ----------
        type
            One of ``"user" / "feedback" / "project" / "reference"``.
        name
            Short kebab-case slug used for ``forget`` / linking.
        description
            One-line gist (≤ ~480 bytes).  This is what the embedder sees
            primarily; it's also what :meth:`recall` returns.  Make it
            specific — it acts as the memory's *engram gist*.
        content
            Optional fuller body (≤ ~8000 bytes).  Lazily returned via
            :meth:`expand` only when the LLM asks for it.
        tags
            Optional list of strings (e.g. ``[[related-name]]`` links).
        importance
            0.0 – 1.0 prior on how important this memory is.  Future
            consolidation will use it for retention.
        force
            If ``True``, skip the dedup check and always insert.
        overwrite_by_name
            If ``True`` and a memory with the same ``name`` exists, delete
            it first.  Use this to *update* an existing memory.
        """
        if type not in VALID_TYPES:
            raise ValueError(
                f"unknown type {type!r}; expected one of {sorted(VALID_TYPES)}"
            )
        description = _truncate_bytes(description, DESCRIPTION_MAX_BYTES)
        content     = _truncate_bytes(content,     CONTENT_MAX_BYTES)
        tags_list   = list(tags) if tags else []

        tier_name = tier_for(type)
        tier = self._tiers[tier_name]
        coll = tier.collection

        # ── Embed first (use the description + content for richer signal) ──
        # Do this *before* any store mutation: if the embedder raises (e.g.
        # an OpenAI/HTTP backend network error), we must not have already
        # deleted the existing same-name row — otherwise an overwrite would
        # silently lose the memory it was trying to update.
        text_for_embed = description if not content else f"{description}\n{content}"
        vec = self._embedder.embed(text_for_embed, kind="passage")

        # ── Optional overwrite-by-name ────────────────────────────────────
        # Runs after the embed succeeded, but still before the dedup check so
        # the row we're replacing can't self-match in _closest().
        actually_overwrote = False
        if overwrite_by_name:
            for existing in self._find_by_name(tier_name, type, name):
                coll.delete(existing["mem_id"])
                actually_overwrote = True

        # ── Pattern separation: dedup check ───────────────────────────────
        # Skip when overwrite_by_name is set: there the user treats `name` as
        # the identity ("update this memory"), so a vector near-duplicate
        # check is wrong — and worse, returning merge_suggestion *after* we
        # already deleted the same-name row above would lose that memory.
        if not force and not overwrite_by_name:
            dup = self._closest(vec, tier_name=tier_name)
            if dup is not None and dup.distance < DEDUP_THRESHOLD:
                return SaveResult(
                    status="merge_suggestion",
                    id=None,
                    tier=tier_name,
                    duplicate_of=dup,
                )

        # ── Insert ────────────────────────────────────────────────────────
        now = int(time.time())
        row = {
            "type":        type,
            "name":        name,
            "description": description,
            "content":     content,
            "tags":        tags_list,
            "importance":  float(importance),
            "hits":        0,
            "created_at":  now,
            "accessed_at": now,
            "vector":      vec,
        }
        ids = coll.insert([row])
        coll.flush()
        new_id = int(ids[0])

        # Optional cheap neighborhood consolidation — only scans the row
        # we just inserted and its KNN.  Off by default.
        if self.auto_consolidate:
            from .consolidate import merge_pass
            merge_pass(self, tier=tier_name, only_ids=[new_id])

        return SaveResult(
            status="overwritten" if actually_overwrote else "inserted",
            id=new_id,
            tier=tier_name,
        )

    # ── Public API: recall ────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        k: Optional[int] = None,
        types: Optional[Iterable[str]] = None,
        tiers: Optional[Iterable[str]] = None,
        with_content: bool = False,
        bump_access: bool = True,
        rerank: bool = True,
    ) -> List[MemoryHit]:
        """KNN search across one or both tiers.

        Parameters
        ----------
        query
            The user's current question / context window snippet.
        k
            Top-K hits to return.  **Default ``None`` = adaptive**: we fetch
            a generous candidate pool (:data:`ADAPTIVE_FETCH_BUDGET` per
            tier) and apply :func:`_adaptive_k` gap-based knee detection
            on the distance-sorted candidates, bounded by :data:`K_MIN` and
            :data:`K_MAX`.  Pass an explicit ``int`` for fixed top-K.
        types
            Optional set of memory types to keep.  Filters post-search.
        tiers
            Which tiers to consult.  Default = both.
        with_content
            If ``True``, populate :attr:`MemoryHit.content`.  Default keeps
            results compact (description only) to save tokens.
        bump_access
            If ``True`` (default), update ``hits`` / ``accessed_at`` on
            returned rows.  Set to ``False`` for pure read-only browsing
            (e.g. ``list``).
        rerank
            If ``True`` (default; v2), re-sort the candidate pool by
            :func:`engram.decay.composite_score` so high-importance /
            frequently-accessed memories outrank distance-equivalent
            stale ones.  Set ``False`` to fall back to pure vector
            distance.
        """
        adaptive = k is None
        fetch_per_tier = ADAPTIVE_FETCH_BUDGET if adaptive else k

        qvec = self._embedder.embed(query, kind="query")

        tiers_to_search: List[str]
        if tiers is None:
            tiers_to_search = list(self._tiers.keys())
        else:
            tiers_to_search = [t for t in tiers if t in self._tiers]

        type_filter = set(types) if types else None

        hits: List[MemoryHit] = []
        for tname in tiers_to_search:
            tier = self._tiers[tname]
            raw = tier.collection.search(
                qvec,
                limit=fetch_per_tier,
                output_fields=[
                    "type", "name", "description", "tags",
                    "importance", "hits", "created_at", "accessed_at",
                ] + (["content"] if with_content else []),
            )[0]
            for r in raw:
                if type_filter is not None and r["type"] not in type_filter:
                    continue
                # Use ``is None`` checks rather than ``or`` truthiness so a
                # memory saved with importance=0.0 / hits=0 round-trips as
                # itself, not as "missing → default".
                _imp      = r.get("importance")
                _hcount   = r.get("hits")
                _created  = r.get("created_at")
                _accessed = r.get("accessed_at")
                _tags     = r.get("tags")
                hits.append(
                    MemoryHit(
                        id          = r.id,
                        tier        = tname,
                        type        = r["type"],
                        name        = r["name"],
                        description = r["description"],
                        distance    = r.distance,
                        importance  = 0.0 if _imp      is None else float(_imp),
                        hits        = 0   if _hcount   is None else int(_hcount),
                        created_at  = 0   if _created  is None else int(_created),
                        accessed_at = 0   if _accessed is None else int(_accessed),
                        tags        = []  if _tags     is None else list(_tags),
                        content     = r.get("content") if with_content else None,
                    )
                )

        # Always sort by raw distance first so the adaptive cut sees a
        # monotonic distance curve (gap analysis needs that) and so the
        # static-``k`` path keeps the K closest.  Rerank is then applied
        # to the survivors as a display-order pass — it never changes the
        # *set* of returned hits, only their order.
        hits.sort(key=lambda h: h.distance)

        if adaptive:
            cut = _adaptive_k([h.distance for h in hits])
        else:
            cut = k
        hits = hits[:cut]

        if rerank:
            hits = _rerank_hits(hits)

        if bump_access and hits:
            self._bump_access([(h.tier, h.id) for h in hits])

        return hits

    # ── Public API: expand ────────────────────────────────────────────────

    def expand(self, hit_or_id: Union[int, MemoryHit], tier: Optional[str] = None) -> MemoryHit:
        """Fetch the full row (including ``content``) for an id.

        Bumps ``hits`` and ``accessed_at`` (this is the "reactivation
        strengthens trace" hook).
        """
        if isinstance(hit_or_id, MemoryHit):
            mem_id = hit_or_id.id
            tier_name = hit_or_id.tier
        else:
            mem_id = int(hit_or_id)
            tier_name = tier or self._find_tier_of(mem_id)
            if tier_name is None:
                raise KeyError(f"mem_id={mem_id} not found in either tier")

        coll = self._tiers[tier_name].collection
        row = coll.get(mem_id)
        # ``is None`` coercion — same rationale as recall(): preserve
        # explicit zeros across save → expand round-trip.
        _imp      = row.get("importance")
        _hcount   = row.get("hits")
        _created  = row.get("created_at")
        _accessed = row.get("accessed_at")
        _tags     = row.get("tags")
        _content  = row.get("content")
        hit = MemoryHit(
            id          = mem_id,
            tier        = tier_name,
            type        = row["type"],
            name        = row["name"],
            description = row["description"],
            distance    = 0.0,
            importance  = 0.0 if _imp      is None else float(_imp),
            hits        = 0   if _hcount   is None else int(_hcount),
            created_at  = 0   if _created  is None else int(_created),
            accessed_at = 0   if _accessed is None else int(_accessed),
            tags        = []  if _tags     is None else list(_tags),
            content     = ""  if _content  is None else _content,
        )
        self._bump_access([(tier_name, mem_id)])
        return hit

    # ── Public API: recall_related ────────────────────────────────────────

    def recall_related(
        self,
        seed: Union[int, str, "MemoryHit"],
        depth: int = 1,
        with_content: bool = False,
    ) -> List[MemoryHit]:
        """Follow ``[[name]]`` links from ``seed`` outwards.

        Tags written as the literal token ``"[[other-name]]"`` are
        treated as edges to other memories of the same ``name``.  This
        returns the BFS frontier up to ``depth`` hops, **excluding** the
        seed itself, with closest hops listed first.

        Cheap: this never re-embeds anything; it's a pure metadata walk.
        """
        seed_id, seed_tier = self._resolve_seed(seed)
        if seed_id is None:
            return []

        # Index: name → (tier, mem_id)
        by_name: Dict[str, Tuple[str, int]] = {}
        for tname, t in self._tiers.items():
            for mid, row in t.collection._rows.items():  # type: ignore[attr-defined]
                n = row.get("name")
                if n:
                    by_name[n] = (tname, mid)

        visited: set[Tuple[str, int]] = {(seed_tier, seed_id)}
        frontier: List[Tuple[str, int]] = [(seed_tier, seed_id)]
        out: List[MemoryHit] = []

        for hop in range(1, depth + 1):
            next_frontier: List[Tuple[str, int]] = []
            for tname, mid in frontier:
                row = self._tiers[tname].collection._rows.get(mid)  # type: ignore[attr-defined]
                if row is None:
                    continue
                for tag in (row.get("tags") or []):
                    target = _parse_link(tag)
                    if target is None:
                        continue
                    edge = by_name.get(target)
                    if edge is None or edge in visited:
                        continue
                    visited.add(edge)
                    next_frontier.append(edge)
                    n_tname, n_mid = edge
                    n_row = self._tiers[n_tname].collection._rows[n_mid]  # type: ignore[attr-defined]
                    out.append(MemoryHit(
                        id          = n_mid,
                        tier        = n_tname,
                        type        = n_row.get("type") or "",
                        name        = n_row.get("name") or "",
                        description = n_row.get("description") or "",
                        # distance encodes BFS hop count — lower is closer.
                        distance    = float(hop),
                        importance  = float(n_row.get("importance") or 0.0),
                        hits        = int(n_row.get("hits") or 0),
                        created_at  = int(n_row.get("created_at") or 0),
                        accessed_at = int(n_row.get("accessed_at") or 0),
                        tags        = list(n_row.get("tags") or []),
                        content     = (n_row.get("content") or "") if with_content else None,
                    ))
            frontier = next_frontier
            if not frontier:
                break
        return out

    def _resolve_seed(
        self, seed: Union[int, str, "MemoryHit"]
    ) -> Tuple[Optional[int], Optional[str]]:
        if isinstance(seed, MemoryHit):
            return seed.id, seed.tier
        if isinstance(seed, int):
            t = self._find_tier_of(seed)
            return seed, t
        # string → name lookup
        for tname, tier in self._tiers.items():
            for mid, row in tier.collection._rows.items():  # type: ignore[attr-defined]
                if row.get("name") == seed:
                    return mid, tname
        return None, None

    # ── Public API: forget ────────────────────────────────────────────────

    def find_for_forget(
        self,
        *,
        name: Optional[str] = None,
        id: Optional[int] = None,
        type: Optional[str] = None,
        tier: Optional[str] = None,
        older_than_days: Optional[float] = None,
    ) -> List[MemoryHit]:
        """Return the rows :meth:`forget` would delete under the same filters.

        Powers the ``--dry-run`` UX in ``forget.py`` (and a future MCP
        ``engram_forget(dry_run=True)``).  Does **not** touch the store —
        no `delete`, no `hits++`, no `accessed_at` bump.

        At least one of ``name``, ``id`` or ``older_than_days`` must be
        given — ``type`` / ``tier`` are *narrowing* filters, never the
        primary selector (otherwise ``forget(type='user')`` would wipe
        every user memory by accident).

        ``older_than_days`` is computed against ``accessed_at`` (so the
        UX is "delete what I haven't used in N days", not "delete what
        I saved N days ago" — recall bumps accessed_at, so frequently-
        reused old memories are protected).
        """
        if name is None and id is None and older_than_days is None:
            raise ValueError(
                "find_for_forget(): pass at least one of "
                "name=, id= or older_than_days= so we know what to match"
            )
        cutoff_ts: Optional[int] = None
        if older_than_days is not None:
            cutoff_ts = int(time.time() - float(older_than_days) * 86400)

        # A bare id is ambiguous when it exists in both tiers (independent
        # auto-id counters).  Refuse to match across tiers — otherwise
        # forget(id=N) would silently delete TWO unrelated memories.
        if id is not None and tier is None:
            holding = self._tiers_with_id(id)
            if len(holding) > 1:
                raise ValueError(
                    f"id={id} exists in multiple tiers ({', '.join(holding)}); "
                    f"pass tier= to disambiguate (refusing to delete across tiers)."
                )

        matches: List[MemoryHit] = []
        for tname, t in self._tiers.items():
            if tier is not None and tname != tier:
                continue
            for mem_id, row in t.collection._rows.items():  # type: ignore[attr-defined]
                if id is not None and mem_id != id:
                    continue
                if name is not None and row.get("name") != name:
                    continue
                if type is not None and row.get("type") != type:
                    continue
                if cutoff_ts is not None:
                    accessed = int(row.get("accessed_at") or 0)
                    # Strictly greater means "more recent than cutoff" → keep.
                    if accessed > cutoff_ts:
                        continue
                matches.append(self._row_to_hit(tname, mem_id, row))
        return matches

    def forget(
        self,
        name: Optional[str] = None,
        id: Optional[int] = None,
        type: Optional[str] = None,
        *,
        tier: Optional[str] = None,
        older_than_days: Optional[float] = None,
    ) -> int:
        """Delete memories by name / id / age.  Returns the count deleted.

        At least one of ``name``, ``id`` or ``older_than_days`` must be
        given — see :meth:`find_for_forget` for the rationale.  ``type``
        and ``tier`` are narrowing filters.
        """
        targets = self.find_for_forget(
            name=name, id=id, type=type, tier=tier,
            older_than_days=older_than_days,
        )
        if not targets:
            return 0
        by_tier: Dict[str, List[int]] = {}
        for hit in targets:
            by_tier.setdefault(hit.tier, []).append(hit.id)
        for tname, ids in by_tier.items():
            coll = self._tiers[tname].collection
            for mid in ids:
                coll.delete(mid)
            coll.flush()
        return len(targets)

    # ── Public API: patch ─────────────────────────────────────────────────

    def patch(
        self,
        target: Union[int, str, MemoryHit],
        *,
        description: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        add_tags: Optional[Sequence[str]] = None,
        remove_tags: Optional[Sequence[str]] = None,
        importance: Optional[float] = None,
        type: Optional[str] = None,
        name: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> MemoryHit:
        """Modify selected fields of one existing memory.

        Two paths:

        * **Metadata-only** (``importance`` / ``tags`` / ``name``) — the
          row is updated in place and **the id is preserved**.  This is
          the cheap path: no embedder call, no index rewrite.
        * **Re-embed required** (``description`` / ``content`` changed,
          or ``type`` change forces a cross-tier move) — the old row is
          deleted, the new content is re-embedded, and a fresh row is
          inserted; **the id will change** because PistaDB allocates
          ``mem_id`` on insert.  ``created_at`` and ``hits`` are
          carried over from the original.

        ``tags``, ``add_tags``, ``remove_tags`` interact as you'd expect:

        * ``tags=[...]`` *replaces* the list wholesale.
        * ``add_tags=[...]`` adds (de-duped) to the existing list.
        * ``remove_tags=[...]`` removes from the existing list.

        ``target`` may be an ``int`` (id), ``str`` (name — must be
        unique across the store, else ``ValueError``), or a
        :class:`MemoryHit`.

        Returns the resulting :class:`MemoryHit` (with the *new* id if
        re-embed was required).
        """
        # ── 1. Locate the row ────────────────────────────────────────────
        if isinstance(target, int) and tier is not None:
            # Explicit tier disambiguates a bare id present in both tiers.
            mem_id, tier_name = target, tier
            if mem_id not in self._tiers[tier_name].collection._rows:  # type: ignore[attr-defined]
                raise ValueError(f"patch: id={mem_id} not found in tier {tier_name!r}")
        else:
            mem_id, tier_name = self._resolve_seed(target)
        if mem_id is None or tier_name is None:
            raise ValueError(f"patch: target {target!r} not found")
        # Disambiguation for ``target=str``: ``_resolve_seed`` already
        # returns the first match, but if multiple rows share that name
        # the user almost certainly didn't mean to silently pick one.
        if isinstance(target, str):
            matches = []
            for tname, t in self._tiers.items():
                for mid, row in t.collection._rows.items():  # type: ignore[attr-defined]
                    if row.get("name") == target:
                        matches.append((tname, mid))
            if len(matches) > 1:
                raise ValueError(
                    f"patch: name {target!r} matches {len(matches)} rows "
                    f"({matches}); pass id= or a more specific target."
                )

        coll = self._tiers[tier_name].collection
        row = coll._rows.get(mem_id)  # type: ignore[attr-defined]
        if row is None:
            raise ValueError(f"patch: row id={mem_id} disappeared")

        # ── 2. Validate inputs ───────────────────────────────────────────
        if type is not None and type not in VALID_TYPES:
            raise ValueError(
                f"patch: unknown type {type!r}; "
                f"expected one of {sorted(VALID_TYPES)}"
            )
        if (tags is not None) and (add_tags is not None or remove_tags is not None):
            raise ValueError(
                "patch: pass either tags= (replace) OR "
                "add_tags=/remove_tags= (delta), not both."
            )

        # ── 3. Compute new field values ──────────────────────────────────
        new_desc       = description if description is not None else (row.get("description") or "")
        new_content    = content     if content     is not None else (row.get("content") or "")
        new_type       = type        if type        is not None else (row.get("type") or "")
        new_name       = name        if name        is not None else (row.get("name") or "")
        new_importance = (
            float(importance) if importance is not None
            else float(row.get("importance") or 0.0)
        )

        cur_tags = list(row.get("tags") or [])
        if tags is not None:
            new_tags = list(tags)
        elif add_tags is None and remove_tags is None:
            new_tags = cur_tags
        else:
            removed = set(remove_tags or ())
            new_tags = [t for t in cur_tags if t not in removed]
            for t in (add_tags or ()):
                if t not in new_tags:
                    new_tags.append(t)

        new_desc    = _truncate_bytes(new_desc,    DESCRIPTION_MAX_BYTES)
        new_content = _truncate_bytes(new_content, CONTENT_MAX_BYTES)

        # ── 4. Decide path: metadata-only vs re-embed ────────────────────
        needs_reembed = (description is not None) or (content is not None)
        new_tier_name = tier_for(new_type)
        needs_tier_move = (new_tier_name != tier_name)

        now = int(time.time())

        if not needs_reembed and not needs_tier_move:
            # Metadata-only fast path: edit in place, id is preserved.
            row["name"]        = new_name
            row["tags"]        = new_tags
            row["importance"] = new_importance
            row["type"]        = new_type  # safe: same tier
            row["accessed_at"] = now
            coll.flush()
            return self._row_to_hit(tier_name, mem_id, row)

        # ── 5. Re-embed path: insert the new row, *then* delete the old ──
        # Insert-before-delete (rather than the reverse) means a crash or
        # exception mid-operation leaves at worst a recoverable duplicate,
        # never a net loss.  The embed runs first so a backend failure
        # mutates nothing.  ``created_at`` / ``hits`` are read off the still-
        # live ``row`` dict before the delete pops it.
        text_for_embed = new_desc if not new_content else f"{new_desc}\n{new_content}"
        vec = self._embedder.embed(text_for_embed, kind="passage")

        target_coll = self._tiers[new_tier_name].collection
        new_row = {
            "type":        new_type,
            "name":        new_name,
            "description": new_desc,
            "content":     new_content,
            "tags":        new_tags,
            "importance":  new_importance,
            "hits":        int(row.get("hits") or 0),
            "created_at":  int(row.get("created_at") or now),
            "accessed_at": now,
            "vector":      vec,
        }
        ids = target_coll.insert([new_row])
        new_id = int(ids[0])  # auto_id ⇒ new_id != mem_id, no PK collision

        coll.delete(mem_id)

        # Flush each affected collection once.  Same-tier patch touches a
        # single collection, so a lone flush persists both the insert and
        # the delete atomically (one sidecar write).
        target_coll.flush()
        if coll is not target_coll:
            coll.flush()

        final_row = target_coll._rows[new_id]  # type: ignore[attr-defined]
        return self._row_to_hit(new_tier_name, new_id, final_row)

    # ── Row → hit ─────────────────────────────────────────────────────────

    def _row_to_hit(
        self, tier_name: str, mem_id: int, row: Dict[str, Any]
    ) -> MemoryHit:
        """Lift a raw sidecar row into the public :class:`MemoryHit` shape.

        Distance is 0.0 — these are direct lookups, not KNN results.
        Used by ``find_for_forget`` and ``patch`` so neither has to
        reinvent the field mapping.
        """
        return MemoryHit(
            id          = int(mem_id),
            tier        = tier_name,
            type        = row.get("type") or "",
            name        = row.get("name") or "",
            description = row.get("description") or "",
            distance    = 0.0,
            importance  = float(row.get("importance") or 0.0),
            hits        = int(row.get("hits") or 0),
            created_at  = int(row.get("created_at") or 0),
            accessed_at = int(row.get("accessed_at") or 0),
            tags        = list(row.get("tags") or []),
            content     = row.get("content"),
        )

    # ── Public API: list ──────────────────────────────────────────────────

    def list(
        self,
        type: Optional[str] = None,
        tier: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemoryHit]:
        """Enumerate memories (no embedding cost).  Sorted by ``created_at``."""
        out: List[MemoryHit] = []
        for tname, t in self._tiers.items():
            if tier is not None and tname != tier:
                continue
            for mem_id, row in t.collection._rows.items():  # type: ignore[attr-defined]
                if type is not None and row.get("type") != type:
                    continue
                out.append(
                    MemoryHit(
                        id          = mem_id,
                        tier        = tname,
                        type        = row.get("type") or "",
                        name        = row.get("name") or "",
                        description = row.get("description") or "",
                        distance    = 0.0,
                        importance  = row.get("importance") or 0.0,
                        hits        = row.get("hits") or 0,
                        created_at  = row.get("created_at") or 0,
                        accessed_at = row.get("accessed_at") or 0,
                        tags        = row.get("tags") or [],
                    )
                )
        out.sort(key=lambda h: h.created_at, reverse=True)
        return out[:limit]

    # ── Public API: directives (standing, always-on constraints) ───────────

    def directives(
        self,
        max_items: int = DIRECTIVES_MAX,
        byte_budget: int = DIRECTIVES_BYTE_BUDGET,
    ) -> List[MemoryHit]:
        """Return every *standing directive* — memories tagged :data:`PIN_TAG`.

        This is Engram's equivalent of a CLAUDE.md "global, always-on
        constraint".  Unlike :meth:`recall` it does **no** vector search and
        **no** similarity filtering — it pulls every pinned memory directly
        from the in-memory ``_rows`` (O(n), zero embedding cost), so a
        per-turn bootstrap can inject it unconditionally.

        Two deliberate properties:

        * **Read-only** — does not bump ``hits`` / ``accessed_at``.  These are
          read on every turn; bumping would pin importance to 1.0 and pollute
          the access stats.
        * **Hard-budgeted** — capped by both ``max_items`` and a total
          ``byte_budget`` over descriptions, sorted by ``importance`` (desc).
          Standing directives cost tokens on every turn, so the budget keeps
          them from silently re-growing into a bloated MEMORY.md.  At least
          one directive is always returned even if it alone exceeds the
          budget.
        """
        collected: List[MemoryHit] = []
        for tname, t in self._tiers.items():
            for mem_id, row in t.collection._rows.items():  # type: ignore[attr-defined]
                if PIN_TAG in (row.get("tags") or []):
                    collected.append(self._row_to_hit(tname, mem_id, row))

        # Highest importance first; ties broken by oldest-then-lowest-id so the
        # order is stable across turns.
        collected.sort(key=lambda h: (-h.importance, h.created_at, h.id))

        out: List[MemoryHit] = []
        used = 0
        for h in collected:
            if len(out) >= max_items:
                break
            cost = len(h.description.encode("utf-8"))
            if out and used + cost > byte_budget:
                break
            out.append(h)
            used += cost
        return out

    # ── Internal helpers ──────────────────────────────────────────────────

    def _closest(
        self, vec: np.ndarray, tier_name: str
    ) -> Optional[MemoryHit]:
        """Return the single nearest neighbour in a tier (or None if empty)."""
        tier = self._tiers[tier_name]
        if tier.count == 0:
            return None
        raw = tier.collection.search(
            vec,
            limit=1,
            output_fields=["type", "name", "description"],
        )[0]
        if not raw:
            return None
        r = raw[0]
        return MemoryHit(
            id          = r.id,
            tier        = tier_name,
            type        = r["type"],
            name        = r["name"],
            description = r["description"],
            distance    = r.distance,
            importance  = 0.0,
            hits        = 0,
            created_at  = 0,
            accessed_at = 0,
        )

    def _tiers_with_id(self, mem_id: int) -> List[str]:
        """Tier names whose collection currently holds ``mem_id``.

        The two tiers are independent PistaDB collections with their own
        auto-id counters, so the same id can exist in BOTH (e.g. global #2
        and local #2).  Any API that takes a bare id must disambiguate.
        """
        return [
            tname for tname, t in self._tiers.items()
            if mem_id in t.collection._rows  # type: ignore[attr-defined]
        ]

    def _find_tier_of(self, mem_id: int) -> Optional[str]:
        """Return the single tier holding ``mem_id``.

        ``None`` if absent.  Raises ``ValueError`` if the id exists in
        *both* tiers — a bare id is then ambiguous and the caller must pass
        an explicit tier, otherwise we'd silently pick one and operate on
        the wrong memory.
        """
        found = self._tiers_with_id(mem_id)
        if not found:
            return None
        if len(found) > 1:
            raise ValueError(
                f"id={mem_id} exists in multiple tiers ({', '.join(found)}); "
                f"pass tier= to disambiguate."
            )
        return found[0]

    def _find_by_name(
        self, tier_name: str, type: str, name: str
    ) -> List[Dict[str, Any]]:
        coll = self._tiers[tier_name].collection
        out = []
        for mem_id, row in coll._rows.items():  # type: ignore[attr-defined]
            if row.get("type") == type and row.get("name") == name:
                out.append({"mem_id": mem_id, **row})
        return out

    def _bump_access(self, tier_ids: Sequence[Tuple[str, int]]) -> None:
        """Increment ``hits`` and update ``accessed_at`` for given rows."""
        now = int(time.time())
        touched: Dict[str, bool] = {}
        for tname, mem_id in tier_ids:
            coll = self._tiers[tname].collection
            row = coll._rows.get(mem_id)  # type: ignore[attr-defined]
            if row is None:
                continue
            new_hits = int(row.get("hits") or 0) + 1
            row["hits"] = new_hits
            row["accessed_at"] = now
            row["importance"] = boost_on_access(
                float(row.get("importance") or 0.0), new_hits
            )
            touched[tname] = True
        # Sidecar JSON is what holds these counters — flush each touched tier.
        for tname in touched:
            self._tiers[tname].flush()


# ── helpers ──────────────────────────────────────────────────────────────────

_LINK_RE = None  # populated lazily

def _parse_link(tag: str) -> Optional[str]:
    """Return the target name if ``tag`` is a ``"[[name]]"`` link, else None."""
    global _LINK_RE
    if _LINK_RE is None:
        import re
        _LINK_RE = re.compile(r"^\s*\[\[\s*([^\[\]]+?)\s*\]\]\s*$")
    if not isinstance(tag, str):
        return None
    m = _LINK_RE.match(tag)
    return m.group(1) if m else None


def _truncate_bytes(s: str, limit: int) -> str:
    """Truncate ``s`` so its UTF-8 byte length ≤ ``limit``."""
    b = s.encode("utf-8")
    if len(b) <= limit:
        return s
    cut = b[:limit]
    # avoid cutting in the middle of a UTF-8 char
    while cut and (cut[-1] & 0xC0) == 0x80:
        cut = cut[:-1]
    return cut.decode("utf-8", errors="ignore") + "…"


__all__ = [
    "MemoryManager",
    "MemoryHit",
    "MemoryType",
    "SaveResult",
    "DEDUP_THRESHOLD",
    "CONTENT_MAX_BYTES",
    "DESCRIPTION_MAX_BYTES",
    "VALID_TYPES",
    "PIN_TAG",
    "DIRECTIVES_MAX",
    "DIRECTIVES_BYTE_BUDGET",
]
