"""End-to-end smoke test for Engram.

Uses :class:`HashEmbedder` so it runs without sentence-transformers / torch.
The semantics aren't great, but it exercises every code path: save → dedup
→ recall → expand → forget → list, across both tiers, and re-opens the
.pst files in a fresh process to verify persistence.

Run from repo root::

    ENGRAM_EMBEDDER=hash CLAUDE_HOME=./tests/_tmp_home \\
        python tests/smoke_test.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Make `engram` importable when run directly (without `pip install -e .`).
# After the src/ layout migration the package lives at <repo>/src/engram/.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _setup_tmp_dirs() -> tuple[Path, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="engram_smoke_"))
    fake_home    = tmp / "home"
    fake_project = tmp / "project"
    (fake_home / ".claude" / "engram").mkdir(parents=True)
    (fake_project / ".claude" / "engram").mkdir(parents=True)
    os.environ["CLAUDE_HOME"] = str(fake_home / ".claude")
    os.environ["ENGRAM_EMBEDDER"] = "hash"
    return fake_home, fake_project


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        raise SystemExit(1)
    print(f"OK   {msg}")


def main() -> None:
    fake_home, fake_project = _setup_tmp_dirs()
    print(f"tmp home    = {fake_home}")
    print(f"tmp project = {fake_project}")

    os.chdir(fake_project)

    # ── 1. First session: save + recall + expand + forget ────────────────
    from engram import MemoryManager

    with MemoryManager() as mgr:
        # Save into both tiers
        r1 = mgr.save(
            type="user",
            name="role-data-sci",
            description="user is a data scientist focused on observability and logging",
            content="They prefer concrete log-line examples in Python when explaining.",
            tags=["role", "logging"],
        )
        _assert(r1.status == "inserted" and r1.tier == "global",
                f"user memory inserted into global ({r1.to_dict()})")

        r2 = mgr.save(
            type="feedback",
            name="terse-replies",
            description="user wants terse responses with no trailing summaries",
            tags=["style"],
        )
        _assert(r2.status == "inserted" and r2.tier == "global",
                "feedback memory inserted into global")

        r3 = mgr.save(
            type="project",
            name="auth-rewrite-driver",
            description="auth middleware rewrite is driven by legal compliance, not tech debt",
            content="Scope decisions should favor compliance over ergonomics.",
            tags=["auth", "compliance"],
        )
        _assert(r3.status == "inserted" and r3.tier == "local",
                "project memory inserted into local")

        r4 = mgr.save(
            type="reference",
            name="grafana-latency",
            description="grafana.internal/d/api-latency is the oncall latency dashboard",
            tags=["grafana", "oncall"],
        )
        _assert(r4.status == "inserted" and r4.tier == "local",
                "reference memory inserted into local")

        # ── Pattern separation: try to save a near-duplicate ─────────────
        # Same description+content → same embedding → must trigger dedup.
        r_dup = mgr.save(
            type="user",
            name="role-data-sci-2",
            description="user is a data scientist focused on observability and logging",
            content="They prefer concrete log-line examples in Python when explaining.",
        )
        _assert(
            r_dup.status == "merge_suggestion"
            and r_dup.duplicate_of is not None
            and r_dup.duplicate_of.name == "role-data-sci",
            f"near-duplicate triggered merge_suggestion → {r_dup.duplicate_of}",
        )

        # ── Force-insert overrides dedup ─────────────────────────────────
        r_force = mgr.save(
            type="user",
            name="role-data-sci-2",
            description="user is a data scientist focused on observability and logging",
            content="They prefer concrete log-line examples in Python when explaining.",
            force=True,
        )
        _assert(r_force.status == "inserted",
                "force=True bypasses dedup")

        # Clean up the forced duplicate for later listing.
        removed = mgr.forget(name="role-data-sci-2")
        _assert(removed == 1, f"forget by name removed {removed} entries")

        # ── Recall ──────────────────────────────────────────────────────
        hits = mgr.recall("how should I write log lines?", k=5)
        _assert(len(hits) > 0, f"recall returned {len(hits)} hits")
        names = [h.name for h in hits]
        _assert("role-data-sci" in names,
                f"role-data-sci surfaces for logging query (got: {names})")

        # ── Recall with type filter ─────────────────────────────────────
        only_proj = mgr.recall("compliance constraints", k=5, types=["project"])
        _assert(all(h.type == "project" for h in only_proj),
                "type filter only returns 'project' rows")

        # ── Expand ──────────────────────────────────────────────────────
        top = hits[0]
        full = mgr.expand(top)
        _assert(full.content and len(full.content) > 0,
                "expand() populates content")

        # Snapshot ids for next-session check.
        all_ids = {h.name: h.id for h in mgr.list()}

    # ── 2. Second session: reopen and verify persistence ─────────────────
    with MemoryManager() as mgr2:
        listed = mgr2.list()
        listed_names = {h.name for h in listed}
        for expected in ("role-data-sci", "terse-replies",
                         "auth-rewrite-driver", "grafana-latency"):
            _assert(expected in listed_names,
                    f"{expected} survived reopen")

        # Bumped hits should be persisted (we expanded role-data-sci above).
        rds = next(h for h in listed if h.name == "role-data-sci")
        _assert(rds.hits >= 1,
                f"hits counter persisted for role-data-sci (={rds.hits})")

        # Recall still works after reopen.
        hits2 = mgr2.recall("oncall dashboard", k=3)
        _assert(any(h.name == "grafana-latency" for h in hits2),
                "recall after reopen surfaces grafana-latency")

    # ── 3. v2 features: decay, rerank, consolidate, links ─────────────────
    print("\n=== v2 feature checks ===")
    _v2_checks(fake_project)

    # ── 4. v0.4 features: patch + forget --dry-run + --older-than ─────────
    print("\n=== v0.4 patch + dry-run forget checks ===")
    _patch_and_forget_checks(fake_project)

    print("\nALL SMOKE CHECKS PASSED.")
    # Best-effort cleanup
    try:
        shutil.rmtree(fake_home.parent)
    except OSError:
        pass


def _v2_checks(fake_project) -> None:
    """v2: decay, importance rerank, consolidate, and recall_related."""
    import time

    from engram import (
        MemoryManager,
        composite_score,
        consolidate,
        decay_pass,
        importance_effective,
    )

    # ── decay math is monotone decreasing in age ─────────────────────────
    now = 1_700_000_000
    fresh = importance_effective(0.8, accessed_at=now, now=now)
    stale = importance_effective(0.8, accessed_at=now - 60 * 86400, now=now)
    _assert(fresh > stale > 0.0,
            f"decay reduces effective importance over time ({fresh:.3f} > {stale:.3f})")

    # ── composite_score: lower distance still wins, but importance breaks ties
    s_close_stale = composite_score(0.10, importance=0.0, accessed_at=0, hits=0, now=now)
    s_far_hot     = composite_score(0.30, importance=1.0, accessed_at=now, hits=100, now=now)
    _assert(s_close_stale < s_far_hot,
            f"clearly closer still beats hot stale (composite {s_close_stale:.4f} < {s_far_hot:.4f})")
    s_a = composite_score(0.20, importance=0.0, accessed_at=0,   hits=0,  now=now)
    s_b = composite_score(0.21, importance=0.9, accessed_at=now, hits=10, now=now)
    _assert(s_b < s_a,
            f"near-tie distance: importance/hits breaks tie ({s_b:.4f} < {s_a:.4f})")

    # ── Live store: rerank pushes the higher-importance memory to the top ─
    import os, shutil
    rerank_root = fake_project.parent / "rerank_proj"
    shutil.rmtree(rerank_root, ignore_errors=True)
    (rerank_root / ".claude" / "engram").mkdir(parents=True)
    cwd_before = os.getcwd()
    os.chdir(rerank_root)
    try:
        with MemoryManager() as mgr:
            # Two project memories with identical embeddable text but
            # different importance.  Use force=True to bypass dedup.
            mgr.save("project", "low-imp", "the quick brown fox jumps over a lazy dog",
                     importance=0.0)
            mgr.save("project", "high-imp", "the quick brown fox jumps over a lazy dog",
                     importance=0.9, force=True)
            with_rr = mgr.recall("quick brown fox", k=2, rerank=True, bump_access=False)
            no_rr   = mgr.recall("quick brown fox", k=2, rerank=False, bump_access=False)
            _assert(with_rr[0].name == "high-imp",
                    f"rerank surfaces high-importance row first (got {with_rr[0].name})")
            # Pure distance can't distinguish — both are equidistant since
            # the embedded text matches exactly.  Just check both backends ran.
            _assert(len(no_rr) == 2,
                    f"no-rerank also returns 2 rows (got {len(no_rr)})")

        # ── consolidate: merge_pass collapses the duplicate ──────────────
        with MemoryManager() as mgr:
            before = len(mgr.list(tier="local"))
            report = consolidate(mgr, tier="local", do_decay=False, dry_run=True)
            _assert(len(report.merges) >= 1,
                    f"dry-run finds at least one merge ({len(report.merges)})")
            real = consolidate(mgr, tier="local", do_decay=False, dry_run=False)
            _assert(len(real.merges) >= 1,
                    f"real merge collapses at least one pair ({len(real.merges)})")
            after = len(mgr.list(tier="local"))
            _assert(after == before - len(real.merges),
                    f"row count drops by merge count ({before} → {after})")

            # ── decay pass: importance shrinks for an old accessed_at ─────
            # Reach into the local tier and backdate one row.
            local = mgr._tiers["local"]
            any_row_id = next(iter(local.collection._rows.keys()))
            row = local.collection._rows[any_row_id]
            row["importance"] = 0.8
            row["accessed_at"] = int(time.time()) - 60 * 86400
            local.flush()
            d_report = decay_pass(mgr, tier="local")
            _assert(d_report.decayed >= 1,
                    f"decay pass decayed at least one row ({d_report.decayed})")
            decayed_row = local.collection._rows[any_row_id]
            _assert(decayed_row["importance"] < 0.8,
                    f"row importance reduced after decay ({decayed_row['importance']:.3f})")
    finally:
        os.chdir(cwd_before)

    # ── recall_related: BFS over [[name]] tag links ──────────────────────
    link_root = fake_project.parent / "link_proj"
    shutil.rmtree(link_root, ignore_errors=True)
    (link_root / ".claude" / "engram").mkdir(parents=True)
    os.chdir(link_root)
    try:
        with MemoryManager() as mgr:
            mgr.save("project", "auth-rewrite", "rewrite the legacy auth layer",
                     tags=["[[compliance-deadline]]"])
            mgr.save("project", "compliance-deadline",
                     "Q3 compliance deadline forces the auth rewrite",
                     tags=["[[auth-rewrite]]"])
            mgr.save("project", "unrelated-thing",
                     "some unrelated project memory")

            hops = mgr.recall_related("auth-rewrite", depth=1)
            names = [h.name for h in hops]
            _assert("compliance-deadline" in names,
                    f"depth=1 reaches compliance-deadline ({names})")
            _assert("unrelated-thing" not in names,
                    f"unrelated memories are NOT in the neighborhood ({names})")
    finally:
        os.chdir(cwd_before)

    # ── auto_consolidate=True: save triggers neighbourhood merge ─────────
    auto_root = fake_project.parent / "auto_proj"
    shutil.rmtree(auto_root, ignore_errors=True)
    (auto_root / ".claude" / "engram").mkdir(parents=True)
    os.chdir(auto_root)
    try:
        with MemoryManager(auto_consolidate=True) as mgr:
            mgr.save("project", "first",  "the quick brown fox jumps over a lazy dog",
                     importance=0.5)
            mgr.save("project", "second", "the quick brown fox jumps over a lazy dog",
                     importance=0.9, force=True)
            remaining = mgr.list(tier="local")
            _assert(len(remaining) == 1,
                    f"auto_consolidate collapses duplicates on save (got {len(remaining)})")
            _assert(remaining[0].name == "second",
                    f"the higher-importance survivor is kept (got {remaining[0].name})")
    finally:
        os.chdir(cwd_before)


def _patch_and_forget_checks(fake_project) -> None:
    """v0.4: ``patch()`` two-path semantics + ``forget()`` dry-run & age filter.

    Spins up its own tier so the assertions don't depend on rows the
    earlier checks left behind.
    """
    import os
    import shutil
    import time

    from engram import MemoryManager

    patch_root = fake_project.parent / "patch_proj"
    shutil.rmtree(patch_root, ignore_errors=True)
    (patch_root / ".claude" / "engram").mkdir(parents=True)
    cwd_before = os.getcwd()
    os.chdir(patch_root)
    try:
        with MemoryManager() as mgr:
            # ── 1. metadata-only patch: id is preserved ─────────────────
            r = mgr.save(
                "feedback", "terse",
                description="user wants terse replies",
                content="No recap at end of turn.",
                tags=["style"],
                importance=0.5,
            )
            original_id = r.id

            hit1 = mgr.patch(
                original_id,
                importance=0.9,
                add_tags=["high-priority"],
            )
            _assert(hit1.id == original_id,
                    f"metadata-only patch preserves id ({original_id} == {hit1.id})")
            _assert(abs(hit1.importance - 0.9) < 1e-6,
                    f"patch updated importance to 0.9 (got {hit1.importance})")
            _assert("high-priority" in hit1.tags and "style" in hit1.tags,
                    f"add_tags merges with existing list (got {hit1.tags})")

            # ── 2. remove_tags drops a tag ──────────────────────────────
            hit2 = mgr.patch(original_id, remove_tags=["style"])
            _assert("style" not in hit2.tags and "high-priority" in hit2.tags,
                    f"remove_tags drops the requested tag only (got {hit2.tags})")
            _assert(hit2.id == original_id,
                    "remove_tags is still metadata-only — id stable")

            # ── 3. desc-changed patch: id changes, created_at preserved ──
            original_created = hit2.created_at
            hit3 = mgr.patch(
                original_id,
                description="user wants extremely terse replies",
            )
            _assert(hit3.id != original_id,
                    f"desc patch re-embeds → id changes ({original_id} → {hit3.id})")
            _assert(hit3.created_at == original_created,
                    f"created_at preserved across re-embed ({hit3.created_at})")
            _assert(hit3.description == "user wants extremely terse replies",
                    "new description landed in the row")

            # ── 4. cross-tier move: type change global→local ───────────
            r4 = mgr.save("user", "tier-move-test",
                          description="will get moved to a different tier",
                          importance=0.5)
            old_tier = r4.tier
            hit4 = mgr.patch(r4.id, type="reference")
            _assert(hit4.tier != old_tier,
                    f"type change crossed the tier boundary ({old_tier} → {hit4.tier})")
            _assert(hit4.type == "reference",
                    f"type field updated (got {hit4.type})")

            # ── 5. patch by name ────────────────────────────────────────
            mgr.save("project", "rename-target",
                     description="something to rename")
            hit5 = mgr.patch("rename-target", name="rename-target-v2")
            _assert(hit5.name == "rename-target-v2",
                    f"patch by name renamed (got {hit5.name})")

            # ── 6. patch: tags vs add_tags conflict raises ──────────────
            try:
                mgr.patch(hit5.id, tags=["a"], add_tags=["b"])
            except ValueError as e:
                _assert("not both" in str(e).lower() or "tags=" in str(e),
                        f"tags + add_tags raises informative error ({e})")
            else:
                _assert(False, "tags + add_tags should have raised")

            # ── 7. patch: unknown target raises ─────────────────────────
            try:
                mgr.patch(999_999_999, importance=0.5)
            except ValueError as e:
                _assert("not found" in str(e),
                        f"missing target raises ValueError ({e})")
            else:
                _assert(False, "missing target should have raised")

        # ── 8. forget --dry-run + --older-than ──────────────────────────
        with MemoryManager() as mgr:
            # Seed three project memories with different accessed_at ages.
            mgr.save("project", "fresh-row", description="fresh row",
                     importance=0.5)
            mgr.save("project", "old-row-a", description="ancient row A",
                     importance=0.5)
            mgr.save("project", "old-row-b", description="ancient row B",
                     importance=0.5)

            # Backdate the "old-*" rows directly via the sidecar dict.
            local = mgr._tiers["local"]
            for mem_id, row in local.collection._rows.items():
                if row.get("name", "").startswith("old-row-"):
                    row["accessed_at"] = int(time.time()) - 60 * 86400
            local.flush()

            # dry-run with --older-than: returns the rows without deleting.
            preview = mgr.find_for_forget(older_than_days=30, tier="local")
            preview_names = {h.name for h in preview}
            _assert(preview_names == {"old-row-a", "old-row-b"},
                    f"find_for_forget(older=30d) picks both old rows ({preview_names})")
            # Confirm nothing was deleted yet.
            still_there = {h.name for h in mgr.list(tier="local")}
            _assert("fresh-row" in still_there and "old-row-a" in still_there,
                    f"dry-run did NOT delete anything (still: {still_there})")

            # Now really forget — count matches preview, fresh row survives.
            removed = mgr.forget(older_than_days=30, tier="local")
            _assert(removed == 2,
                    f"forget(older=30d) deletes the 2 old rows (got {removed})")
            remaining = {h.name for h in mgr.list(tier="local")}
            _assert("fresh-row" in remaining and "old-row-a" not in remaining,
                    f"fresh row survives (remaining: {remaining})")

            # ── 9. forget refuses naked --type (would wipe a whole class) ─
            try:
                mgr.forget(type="project")
            except ValueError as e:
                _assert("at least one" in str(e).lower(),
                        f"forget(type=...) alone raises (got: {e})")
            else:
                _assert(False, "forget(type=...) alone should have raised")
    finally:
        os.chdir(cwd_before)


if __name__ == "__main__":
    main()
