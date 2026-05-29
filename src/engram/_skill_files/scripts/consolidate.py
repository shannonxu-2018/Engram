#!/usr/bin/env python3
"""consolidate.py — sleep replay: merge near-duplicates and apply decay.

Usage:
    consolidate.py [--tier global|local] [--no-merge] [--no-decay]
                   [--threshold 0.05] [--dry-run] [--json]

Two passes:

* **merge**  — collapse any cluster of rows within ``--threshold``
  cosine distance down to a single survivor; loser content is appended
  as a footer with the loser's name + date.
* **decay**  — overwrite stored ``importance`` with the Ebbinghaus-decayed
  value (Δt since last access, τ from ``ENGRAM_DECAY_TAU_DAYS``,
  default 30 days).

Idempotent.  Use ``--dry-run`` first to preview.
"""
from __future__ import annotations

import argparse
import sys

from _common import bootstrap, emit_json

bootstrap()

from engram import MemoryManager, consolidate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier", choices=["global", "local"], default=None)
    ap.add_argument("--no-merge", action="store_true",
                    help="Skip the merge pass.")
    ap.add_argument("--no-decay", action="store_true",
                    help="Skip the decay pass.")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="Cosine distance below which two rows are merged.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args()

    with MemoryManager() as mgr:
        report = consolidate(
            mgr,
            tier=args.tier,
            do_merge=not args.no_merge,
            do_decay=not args.no_decay,
            threshold=args.threshold,
            dry_run=args.dry_run,
        )

    if args.as_json:
        emit_json(report.to_dict())
    else:
        verb = "would" if report.dry_run else "did"
        print(f"scanned : {report.scanned}")
        print(f"merges  : {len(report.merges)} ({verb})")
        for m in report.merges:
            print(f"  {m.tier:<6} keep id={m.keep_id} ({m.keep_name})  "
                  f"<- drop id={m.drop_id} ({m.drop_name})  d={m.distance:.4f}")
        print(f"decayed : {report.decayed} rows ({verb})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
