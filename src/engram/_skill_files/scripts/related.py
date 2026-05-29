#!/usr/bin/env python3
"""related.py — graph traversal over `[[name]]` links in tags.

Usage:
    related.py <name-or-id> [--depth 1] [--with-content] [--json|--text]

Tags written as ``"[[other-name]]"`` are treated as edges to other
memories.  This script returns BFS neighbours up to ``--depth`` hops out.

Cost: zero embeddings — pure metadata walk.
"""
from __future__ import annotations

import argparse
import sys

from _common import bootstrap, emit_json, emit_text

bootstrap()

from engram import MemoryManager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("seed", help="Memory name or integer id.")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--with-content", action="store_true")
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args()

    seed: object
    try:
        seed = int(args.seed)
    except ValueError:
        seed = args.seed

    with MemoryManager() as mgr:
        neighbours = mgr.recall_related(
            seed=seed, depth=args.depth, with_content=args.with_content
        )

    if args.as_json:
        emit_json([h.to_dict() for h in neighbours])
    else:
        if not neighbours:
            print("(no linked memories)")
            return 0
        emit_text(neighbours, with_content=args.with_content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
