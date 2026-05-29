#!/usr/bin/env python3
"""expand.py — fetch the full content of a single memory by id.

Usage:
    expand.py <id> [--tier global|local] [--json]

Side effect: bumps ``hits`` and updates ``accessed_at`` (trace strengthening).
"""
from __future__ import annotations

import argparse
import sys

from _common import bootstrap, emit_json

bootstrap()

from engram import MemoryManager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("id", type=int)
    ap.add_argument("--tier", default=None, choices=["global", "local"])
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args()

    with MemoryManager() as mgr:
        try:
            hit = mgr.expand(args.id, tier=args.tier)
        except KeyError:
            scope = f"tier={args.tier}" if args.tier else "both tiers"
            print(f"error: memory id={args.id} not found ({scope})",
                  file=sys.stderr)
            return 1

    if args.as_json:
        emit_json(hit.to_dict())
    else:
        print(f"id   : {hit.id}")
        print(f"tier : {hit.tier}")
        print(f"type : {hit.type}")
        print(f"name : {hit.name}")
        print(f"desc : {hit.description}")
        if hit.tags:
            print(f"tags : {', '.join(hit.tags)}")
        print(f"hits : {hit.hits}")
        print("---")
        print(hit.content or "(empty)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
