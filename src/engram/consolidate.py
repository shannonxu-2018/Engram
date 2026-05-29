"""Consolidation passes — the "sleep replay" of Engram.

Two independent passes that can run together or alone:

* :func:`merge_pass`  — find near-duplicate clusters inside one tier and
  collapse each cluster down to a single survivor.  The survivor keeps
  its own ``description`` (the engram gist) and gets the merged-in
  bodies appended as a footer; the others are deleted.

* :func:`decay_pass` — recompute and overwrite ``importance`` for every
  surviving row, applying the Ebbinghaus decay defined in
  :mod:`engram.decay`.  This is what makes long-unused memories sink
  to the bottom of future recall results.

Both passes are **idempotent** — running them twice in a row is a no-op
the second time (well, decay re-multiplies, but it converges).  Both
support ``dry_run=True`` which reports what *would* happen without
mutating anything.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .decay import importance_effective

if TYPE_CHECKING:
    from .memory import MemoryManager
    from .store import Tier


# ── Tunables ─────────────────────────────────────────────────────────────────

#: Cosine distance below which two memories are deemed redundant and one
#: gets merged into the other.  Tighter than DEDUP_THRESHOLD because by
#: the time consolidation runs we want only true duplicates.
MERGE_THRESHOLD = 0.05

#: Cap on how many neighbours to consider for each row when scanning.
#: HNSW returns enough that this seldom matters in practice.
MERGE_KNN_K = 8

#: Footer separator used when appending a merged memory's body into the
#: survivor's ``content``.
MERGE_FOOTER = "\n\n— merged from {name} ({date}) —\n"


# ── Reports ─────────────────────────────────────────────────────────────────

@dataclass
class MergeAction:
    """One survivor-absorbs-loser action."""
    tier:       str
    keep_id:    int
    drop_id:    int
    keep_name:  str
    drop_name:  str
    distance:   float

    def to_dict(self) -> dict:
        return {
            "tier":      self.tier,
            "keep_id":   self.keep_id,
            "drop_id":   self.drop_id,
            "keep_name": self.keep_name,
            "drop_name": self.drop_name,
            "distance":  round(self.distance, 6),
        }


@dataclass
class ConsolidateReport:
    merges:   List[MergeAction] = field(default_factory=list)
    decayed:  int = 0
    scanned:  int = 0
    dry_run:  bool = False

    def to_dict(self) -> dict:
        return {
            "dry_run": self.dry_run,
            "scanned": self.scanned,
            "merges":  [m.to_dict() for m in self.merges],
            "decayed": self.decayed,
        }


# ── Merge pass ──────────────────────────────────────────────────────────────

def merge_pass(
    mgr: "MemoryManager",
    *,
    tier: Optional[str] = None,
    threshold: float = MERGE_THRESHOLD,
    dry_run: bool = False,
    only_ids: Optional[Iterable[int]] = None,
) -> ConsolidateReport:
    """Collapse near-duplicate clusters within a tier.

    Parameters
    ----------
    mgr
        The :class:`MemoryManager` to operate on.
    tier
        ``"global"`` / ``"local"`` to limit the pass to one tier, or
        ``None`` (default) to run both.
    threshold
        Cosine distance below which two memories are merged.
    dry_run
        Report what would change without writing.
    only_ids
        Limit candidate "seed" rows to this id set.  Used by the
        auto-consolidate hook to scan only the row just inserted.

    Survivor-selection rule (deterministic):

    1. **higher** ``importance`` wins
    2. tie → **older** ``created_at`` (lower number) wins
    3. tie → **lower** ``id`` wins
    """
    report = ConsolidateReport(dry_run=dry_run)
    tiers = [tier] if tier else list(mgr._tiers.keys())  # type: ignore[attr-defined]
    only = set(only_ids) if only_ids is not None else None
    visited: set[int] = set()

    for tname in tiers:
        t: "Tier" = mgr._tiers[tname]                    # type: ignore[attr-defined]
        coll = t.collection
        rows: Dict[int, dict] = dict(coll._rows)         # type: ignore[attr-defined]
        if not rows:
            continue

        seed_ids = list(rows.keys()) if only is None else [
            i for i in rows.keys() if i in only
        ]

        for seed_id in seed_ids:
            if seed_id in visited:
                continue
            if seed_id not in rows:           # already dropped this pass
                continue
            report.scanned += 1

            # Fetch the seed's own vector to query KNN against itself.
            try:
                seed_vec, _ = coll.db.get(seed_id)
            except Exception:
                continue

            hits = coll.search(
                seed_vec,
                limit=MERGE_KNN_K,
                output_fields=["name", "description"],
            )[0]

            # Build the cluster: every neighbour within threshold (skip self).
            cluster: List[Tuple[int, float]] = []
            for h in hits:
                if h.id == seed_id:
                    continue
                if h.id not in rows:          # neighbour already merged away
                    continue
                if h.distance < threshold:
                    cluster.append((h.id, h.distance))

            if not cluster:
                visited.add(seed_id)
                continue

            cluster_ids = [seed_id] + [cid for cid, _ in cluster]
            dist_by_id  = {cid: d for cid, d in cluster}
            dist_by_id[seed_id] = 0.0

            # Pick the survivor.
            def _key(i: int) -> Tuple[float, int, int]:
                r = rows[i]
                imp = float(r.get("importance") or 0.0)
                created = int(r.get("created_at") or 0)
                return (-imp, created, i)

            cluster_ids.sort(key=_key)
            keep_id = cluster_ids[0]
            keep_row = rows[keep_id]

            for drop_id in cluster_ids[1:]:
                drop_row = rows[drop_id]
                report.merges.append(MergeAction(
                    tier      = tname,
                    keep_id   = keep_id,
                    drop_id   = drop_id,
                    keep_name = str(keep_row.get("name") or ""),
                    drop_name = str(drop_row.get("name") or ""),
                    distance  = float(dist_by_id.get(drop_id, 0.0)),
                ))
                if not dry_run:
                    _absorb(keep_row, drop_row)
                    coll.delete(drop_id)
                    rows.pop(drop_id, None)
                visited.add(drop_id)

            visited.add(keep_id)

        if not dry_run and report.merges:
            t.flush()

    return report


def _absorb(survivor: dict, victim: dict) -> None:
    """Mutate ``survivor`` in-place to absorb ``victim``'s metadata.

    * Merges tags (union, preserving order).
    * Appends victim's content under a dated footer, capped to the VARCHAR
      max so the row still fits.
    * Adds the victim's ``hits`` count.
    * Takes the higher ``importance`` (already true by survivor-selection,
      but make it explicit so consolidation isn't lossy if the rule changes).
    """
    from .memory import CONTENT_MAX_BYTES  # local: avoid cyclic import

    # Order-preserving dedup of (survivor ∪ victim) tags.  O(n) via a
    # running set; the previous nested-loop version was O(n²) in tag count.
    seen: set = set()
    survivor["tags"] = [
        t
        for t in (list(survivor.get("tags") or []) + list(victim.get("tags") or []))
        if not (t in seen or seen.add(t))
    ]

    survivor["hits"] = int(survivor.get("hits") or 0) + int(victim.get("hits") or 0)
    survivor["importance"] = max(
        float(survivor.get("importance") or 0.0),
        float(victim.get("importance") or 0.0),
    )

    v_content = (victim.get("content") or "").strip()
    if v_content:
        date = time.strftime(
            "%Y-%m-%d", time.gmtime(int(victim.get("created_at") or time.time()))
        )
        footer = MERGE_FOOTER.format(
            name=victim.get("name") or "?", date=date
        ) + v_content
        new_content = (survivor.get("content") or "") + footer
        # Truncate from the *front* of the footers (oldest first) if too long.
        if len(new_content.encode("utf-8")) > CONTENT_MAX_BYTES:
            new_content = _tail_to_bytes(new_content, CONTENT_MAX_BYTES)
        survivor["content"] = new_content


def _tail_to_bytes(s: str, limit: int) -> str:
    """Keep the last ``limit`` UTF-8 bytes of ``s``, ellipsised at the front."""
    b = s.encode("utf-8")
    if len(b) <= limit:
        return s
    cut = b[-limit:]
    # avoid starting in the middle of a UTF-8 char
    while cut and (cut[0] & 0xC0) == 0x80:
        cut = cut[1:]
    return "…" + cut.decode("utf-8", errors="ignore")


# ── Decay pass ──────────────────────────────────────────────────────────────

def decay_pass(
    mgr: "MemoryManager",
    *,
    tier: Optional[str] = None,
    dry_run: bool = False,
    now: Optional[int] = None,
) -> ConsolidateReport:
    """Overwrite stored ``importance`` with its decayed effective value.

    A row with ``accessed_at == 0`` (never touched since import) is left
    alone — its importance is treated as fresh.
    """
    report = ConsolidateReport(dry_run=dry_run)
    tiers = [tier] if tier else list(mgr._tiers.keys())  # type: ignore[attr-defined]
    t_now = int(time.time()) if now is None else int(now)

    for tname in tiers:
        t = mgr._tiers[tname]                            # type: ignore[attr-defined]
        coll = t.collection
        rows: Dict[int, dict] = coll._rows               # type: ignore[attr-defined]
        any_change = False
        for mem_id, row in rows.items():
            report.scanned += 1
            base = float(row.get("importance") or 0.0)
            if base <= 0:
                continue
            accessed = int(row.get("accessed_at") or 0)
            if accessed <= 0:
                continue
            eff = importance_effective(base, accessed, now=t_now)
            if abs(eff - base) < 1e-6:
                continue
            if not dry_run:
                row["importance"] = eff
                any_change = True
            report.decayed += 1
        if any_change:
            t.flush()
    return report


# ── Combined entry point ────────────────────────────────────────────────────

def consolidate(
    mgr: "MemoryManager",
    *,
    tier: Optional[str] = None,
    do_merge: bool = True,
    do_decay: bool = True,
    threshold: float = MERGE_THRESHOLD,
    dry_run: bool = False,
) -> ConsolidateReport:
    """Run both passes and return a combined report."""
    merged = (
        merge_pass(mgr, tier=tier, threshold=threshold, dry_run=dry_run)
        if do_merge else ConsolidateReport(dry_run=dry_run)
    )
    decayed = (
        decay_pass(mgr, tier=tier, dry_run=dry_run)
        if do_decay else ConsolidateReport(dry_run=dry_run)
    )
    return ConsolidateReport(
        merges  = merged.merges,
        decayed = decayed.decayed,
        scanned = merged.scanned + decayed.scanned,
        dry_run = dry_run,
    )


__all__ = [
    "consolidate",
    "merge_pass",
    "decay_pass",
    "MergeAction",
    "ConsolidateReport",
    "MERGE_THRESHOLD",
]
