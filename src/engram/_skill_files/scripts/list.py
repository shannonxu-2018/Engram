#!/usr/bin/env python3
"""list.py — enumerate memories without any embedding cost.

Usage:
    list.py [--type T] [--tier global|local] [--limit 100] [--json]
"""
from __future__ import annotations

import argparse
import sys

from _common import bootstrap, emit_json, emit_text

bootstrap()

from engram import MemoryManager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--type", default=None)
    ap.add_argument("--tier", default=None, choices=["global", "local"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args()

    with MemoryManager() as mgr:
        rows = mgr.list(type=args.type, tier=args.tier, limit=args.limit)

    if args.as_json:
        emit_json([h.to_dict() for h in rows])
    else:
        emit_text(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
