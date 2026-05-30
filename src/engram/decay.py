"""Importance decay + recall reranking.

Hippocampus analogy:

* Every time a memory is *reactivated* (recall / expand), its
  ``accessed_at`` is bumped and ``hits`` increments.  This mirrors the
  way a re-played trace strengthens (System-2 → consolidation).
* Between reactivations, importance *decays* with time-since-last-access,
  Ebbinghaus-style — a memory you haven't touched in months gets pushed
  down the ranking, but is **not** deleted (soft decay).

Knobs (all environment-overridable):

* ``ENGRAM_DECAY_TAU_DAYS``   — time constant of the exponential
  decay.  Default 30 days.  A memory not accessed for τ days retains
  ``1/e ≈ 37%`` of its base importance.
* ``ENGRAM_RANK_BETA``        — weight of *importance* in the rerank
  composite.  Default ``0.10``.
* ``ENGRAM_RANK_GAMMA``       — weight of ``log1p(hits)`` in the
  rerank composite.  Default ``0.02``.

Reranking (lower = better, matches PistaDB cosine distance):

    composite = distance  -  β · importance_effective(now)
                          -  γ · log1p(hits)

This composite orders hits **only within a "protect band" of distance**
(``ENGRAM_RANK_PROTECT_BAND``, default 0.03): a frequently-accessed
important memory wins ties against a *near-equidistant* stale one, but a
clearly closer vector always wins — importance/recency can never overturn
a real distance gap.  (Earlier versions sorted globally by the composite,
which in a narrow-distance band let a high-``hits`` old memory outrank the
closest match.)
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Tuple

if TYPE_CHECKING:
    from .memory import MemoryHit  # noqa: F401


# ── Tunables ─────────────────────────────────────────────────────────────────

_SECONDS_PER_DAY = 86_400


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def decay_tau_seconds() -> float:
    """Time constant τ in seconds.  Read fresh each call so tests can
    monkey-patch the env var."""
    return _env_float("ENGRAM_DECAY_TAU_DAYS", 30.0) * _SECONDS_PER_DAY


def rank_beta() -> float:
    return _env_float("ENGRAM_RANK_BETA", 0.10)


def rank_gamma() -> float:
    return _env_float("ENGRAM_RANK_GAMMA", 0.02)


def rank_protect_band() -> float:
    """Distance band within which rerank may reorder — a *tie-break* only.

    Importance/recency can reorder hits whose cosine distance is within
    this band of each other, but can never pull a clearly-closer hit below
    a clearly-farther one.  Default ``0.03`` — wider than same-cluster
    noise (~0.01) yet narrower than a genuine relevance gap, so the closest
    vector is never demoted by a high ``hits`` / ``importance``.
    """
    return _env_float("ENGRAM_RANK_PROTECT_BAND", 0.03)


# ── Effective importance (current decayed value) ─────────────────────────────

def importance_effective(
    base_importance: float,
    accessed_at: int,
    *,
    now: int | None = None,
    tau_seconds: float | None = None,
) -> float:
    """Return the *current* importance after exponential decay.

    Parameters
    ----------
    base_importance
        Stored ``importance`` value, in ``[0, 1]``.
    accessed_at
        Unix timestamp of last access.  ``0`` means "never touched" — in
        that case we treat it as the current time (no decay applied yet).
    now, tau_seconds
        Inject for testing.  Default to ``time.time()`` and the env-driven τ.
    """
    if base_importance <= 0:
        return 0.0
    if accessed_at <= 0:
        return float(base_importance)
    t_now = time.time() if now is None else float(now)
    tau = decay_tau_seconds() if tau_seconds is None else float(tau_seconds)
    if tau <= 0:
        return float(base_importance)
    dt = max(0.0, t_now - float(accessed_at))
    return float(base_importance) * math.exp(-dt / tau)


# ── Access boost ─────────────────────────────────────────────────────────────

def boost_on_access(base_importance: float, hits_after: int) -> float:
    """Update stored ``importance`` after an access.

    Strategy: take the **max** of (a) current base importance and
    (b) a re-strengthened value that recovers most of what decay would
    take away.  This is simple, monotone, and self-limiting (caps at 1.0).

    The boost shrinks with each subsequent hit so that very-popular
    memories don't asymptotically pin to 1.0 instantly; instead the
    function ``1 - 0.5 / (hits_after + 1)`` converges to 1.0 slowly.
    """
    target = 1.0 - 0.5 / max(1, hits_after + 1)
    return max(float(base_importance), target)


# ── Composite rerank ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RerankWeights:
    """Bundle of weights, picked up from env if not supplied."""
    beta:  float
    gamma: float
    tau_seconds: float

    @classmethod
    def from_env(cls) -> "RerankWeights":
        return cls(
            beta=rank_beta(),
            gamma=rank_gamma(),
            tau_seconds=decay_tau_seconds(),
        )


def composite_score(
    distance: float,
    importance: float,
    accessed_at: int,
    hits: int,
    *,
    now: int | None = None,
    weights: RerankWeights | None = None,
) -> float:
    """Composite for sorting — **lower is better**.

    Matches PistaDB's cosine-distance ordering so callers can sort by
    this value the same way they sort by ``distance``.
    """
    w = weights or RerankWeights.from_env()
    eff = importance_effective(
        importance, accessed_at,
        now=now, tau_seconds=w.tau_seconds,
    )
    return float(distance) - w.beta * eff - w.gamma * math.log1p(max(0, int(hits)))


def rerank(hits: Sequence["MemoryHit"]) -> list["MemoryHit"]:
    """Return ``hits`` reordered so importance/recency only *tie-breaks*
    among near-equidistant hits — never overturning a real distance gap.

    Distance is the **primary** key, quantised into protect-band buckets
    (:func:`rank_protect_band`); :func:`composite_score` orders only
    *within* a bucket.  This guarantees the closest vector is never pushed
    below a clearly-farther one by a high ``hits`` / ``importance`` — the
    failure mode the old global composite-sort had in a narrow-distance
    band.

    Does not mutate input.  ``MemoryHit.distance`` is left untouched.
    """
    now = int(time.time())
    w = RerankWeights.from_env()
    band = rank_protect_band()

    def _key(h: "MemoryHit") -> Tuple[int, float]:
        # Bucket by distance first (so a closer bucket always wins), then
        # let the composite reorder *within* the bucket.
        bucket = int(float(h.distance) / band) if band > 0 else 0
        comp = composite_score(
            h.distance, h.importance, h.accessed_at, h.hits,
            now=now, weights=w,
        )
        return (bucket, comp)

    return sorted(hits, key=_key)


__all__ = [
    "importance_effective",
    "boost_on_access",
    "composite_score",
    "rerank",
    "RerankWeights",
    "decay_tau_seconds",
    "rank_beta",
    "rank_gamma",
    "rank_protect_band",
]
