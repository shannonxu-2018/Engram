#!/usr/bin/env python3
"""recall.py — semantic search over saved memories.

Usage:
    recall.py "<query>" [--k N] [--type T] [--tier global|local]
              [--with-content] [--json|--text]

Default: **adaptive k** — fetches a generous candidate pool and cuts at the
first big distance gap, bounded between K_MIN=2 and K_MAX=10. Pass `--k N`
to force a fixed top-K.

Output: one compact line per hit, ideal for cheap context injection.
"""
from __future__ import annotations

import argparse
import sys

from _common import bootstrap, emit_json, emit_text

bootstrap()

from engram import MemoryManager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("query", help="Free-text query.")
    ap.add_argument(
        "--k", type=int, default=None,
        help="Top-K hits. Default = adaptive (gap-based knee detection, "
             "between K_MIN=2 and K_MAX=10).",
    )
    ap.add_argument(
        "--type", action="append", default=None,
        help="Filter by memory type. Repeatable.",
    )
    ap.add_argument(
        "--tier", action="append", default=None,
        choices=["global", "local"],
        help="Restrict to one tier. Repeatable.",
    )
    ap.add_argument(
        "--with-content", action="store_true",
        help="Include full content. Costs tokens — use sparingly.",
    )
    ap.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Emit JSON instead of compact text.",
    )
    ap.add_argument(
        "--no-bump", action="store_true",
        help="Do not increment hits/accessed_at on returned rows.",
    )
    ap.add_argument(
        "--no-rerank", action="store_true",
        help="Disable importance+recency rerank; sort by pure cosine distance.",
    )
    args = ap.parse_args()

    with MemoryManager() as mgr:
        hits = mgr.recall(
            query=args.query,
            k=args.k,
            types=args.type,
            tiers=args.tier,
            with_content=args.with_content,
            bump_access=not args.no_bump,
            rerank=not args.no_rerank,
        )
        if args.as_json:
            emit_json([h.to_dict() for h in hits])
        else:
            if args.k is None:
                # Diagnostic: tell the user what adaptive picked.
                print(f"# adaptive k={len(hits)}")
            emit_text(hits, with_content=args.with_content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
