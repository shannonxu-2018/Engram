#!/usr/bin/env python3
"""directives.py — print all *standing directives* (memories tagged `pin`).

These are Engram's "global, always-on constraints" — the CLAUDE.md role,
moved into the memory store.  Run this at the START of every turn, inject
the output into context, and obey it unconditionally.  Unlike `recall.py`
this does NO semantic search: it pulls every pinned memory directly, so a
constraint like "always reply in Simplified Chinese" can never be missed
just because the current query is about something else.

To create one:
    save.py user lang-zh "always reply in Simplified Chinese" --tag pin --importance 0.9

Usage:
    directives.py [--json]
"""
from __future__ import annotations

import argparse
import sys

from _common import bootstrap, emit_json

bootstrap()

from engram import MemoryManager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args()

    with MemoryManager() as mgr:
        rows = mgr.directives()

    if args.as_json:
        emit_json([h.to_dict() for h in rows])
        return 0

    if not rows:
        print("(no standing directives)")
        return 0

    print("# Standing directives — always apply, regardless of topic:")
    for h in rows:
        desc = h.description.replace("\n", " ").replace("\r", " ")
        print(f"  ★ {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
