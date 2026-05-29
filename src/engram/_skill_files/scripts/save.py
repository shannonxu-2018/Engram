#!/usr/bin/env python3
"""save.py — encode a new memory.

Usage:
    save.py <type> <name> "<description>" [content...]
            [--tag t1 --tag t2 ...] [--importance 0.7]
            [--force | --update]

Exit status:
    0  inserted (or overwritten with --update)
    2  merge_suggestion — a near-duplicate already exists; re-run with
       --update to replace, --force to insert anyway, or skip.
"""
from __future__ import annotations

import argparse
import sys

from _common import bootstrap, emit_json

bootstrap()

from engram import MemoryManager, MemoryType  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("type",
                    choices=[MemoryType.USER, MemoryType.FEEDBACK,
                             MemoryType.PROJECT, MemoryType.REFERENCE])
    ap.add_argument("name", help="Short kebab-case slug.")
    ap.add_argument("description", help="One-line gist (≤ 480 bytes).")
    ap.add_argument("content", nargs="*", default=[],
                    help="Optional fuller body (joined with spaces).")
    ap.add_argument("--tag", action="append", default=[],
                    help="Repeatable tag.")
    ap.add_argument("--importance", type=float, default=0.5)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--force", action="store_true",
                   help="Skip the dedup check; always insert.")
    g.add_argument("--update", action="store_true",
                   help="Delete any existing memory with this name first.")
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args()

    content = " ".join(args.content) if args.content else ""

    with MemoryManager() as mgr:
        res = mgr.save(
            type=args.type,
            name=args.name,
            description=args.description,
            content=content,
            tags=args.tag or None,
            importance=args.importance,
            force=args.force,
            overwrite_by_name=args.update,
        )

    out = res.to_dict()
    if args.as_json:
        emit_json(out)
    else:
        if res.status == "merge_suggestion":
            d = res.duplicate_of
            print(f"MERGE_SUGGESTION: near-duplicate exists")
            print(f"  existing id={d.id} tier={d.tier} type={d.type} name={d.name}")
            print(f"  existing desc: {d.description}")
            print(f"  distance={d.distance:.4f} (< 0.08 threshold)")
            print(f"  Re-run with --update to replace, --force to insert anyway.")
            return 2
        print(f"{res.status}: id={res.id} tier={res.tier}")
    return 0 if res.status != "merge_suggestion" else 2


if __name__ == "__main__":
    sys.exit(main())
