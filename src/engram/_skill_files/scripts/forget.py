#!/usr/bin/env python3
"""forget.py — delete memories by name, id, or age.

Usage:
    forget.py --name <slug> [--type T] [--tier T]
    forget.py --id <n>
    forget.py --older-than <DAYS> [--type T] [--tier T]
    forget.py ... --dry-run     # preview only — does not touch the store

``--older-than`` filters by ``accessed_at`` (so "haven't used in N days"),
**not** by ``created_at`` — recall bumps ``accessed_at``, so a frequently-
re-read old memory is protected from age-based cleanup.

At least one of ``--name`` / ``--id`` / ``--older-than`` is required.
``--type`` and ``--tier`` are *narrowing* filters, never the primary
selector — otherwise ``--type user`` alone would wipe every user memory.
"""
from __future__ import annotations

import argparse
import sys

from _common import bootstrap, emit_json

bootstrap()

from engram import MemoryManager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="Slug to delete (may match multiple).")
    g.add_argument("--id", type=int, help="Exact memory id.")
    g.add_argument("--older-than", type=float, dest="older_than",
                   metavar="DAYS",
                   help="Delete memories whose accessed_at is older than "
                        "this many days.  Combine with --type/--tier to "
                        "narrow scope; use --dry-run first.")
    ap.add_argument("--type", default=None,
                    help="Extra filter when names collide across types.")
    ap.add_argument("--tier", default=None, choices=["global", "local"],
                    help="Limit to one tier (default: both).")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="Print the rows that would be deleted, don't actually "
                         "delete anything.  Strongly recommended before a "
                         "broad --older-than sweep.")
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args()

    with MemoryManager() as mgr:
        if args.dry_run:
            targets = mgr.find_for_forget(
                name=args.name, id=args.id, type=args.type,
                tier=args.tier, older_than_days=args.older_than,
            )
            if args.as_json:
                emit_json({"would_delete": [h.to_dict() for h in targets]})
            else:
                for h in targets:
                    desc = h.description.replace("\n", " ")
                    age_days = max(0, (mgr_now() - h.accessed_at) // 86400) \
                        if h.accessed_at else None
                    age = f"age={age_days}d" if age_days is not None else ""
                    print(
                        f"  would delete {h.id:>5} | {h.tier[:1]} | "
                        f"{h.type:<9} | {h.name:<28} | {age:<10} | {desc}"
                    )
                print(f"total: {len(targets)} (dry-run; nothing changed)")
            return 0

        n = mgr.forget(
            name=args.name, id=args.id, type=args.type,
            tier=args.tier, older_than_days=args.older_than,
        )
    if args.as_json:
        emit_json({"removed": n})
    else:
        print(f"removed {n} memory record(s)")
    return 0


def mgr_now() -> int:
    """Wall-clock now() in seconds — kept here so the dry-run row dump
    doesn't have to import ``time`` at module scope just for this."""
    import time
    return int(time.time())


if __name__ == "__main__":
    sys.exit(main())
