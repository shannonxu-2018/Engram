#!/usr/bin/env python3
"""patch.py — modify selected fields of one existing memory.

Two paths under the hood (see ``MemoryManager.patch``):

  * Metadata-only edit (importance / tags / name / type-within-same-tier)
    → the row is updated in place, **the id is preserved**.
  * Re-embed required (description or content changed, or type changed
    enough to move tiers) → old row deleted, new row inserted, **the id
    changes**.  ``created_at`` and ``hits`` are carried over.

Usage:
    patch.py <id|name> [--description "<new>"] [--content "<new>"]
                       [--importance F] [--type T] [--rename NEW-NAME]
                       [--add-tag X --add-tag Y]
                       [--remove-tag X]
                       [--set-tag X --set-tag Y]    # replace the list
                       [--json]

Examples:
    # Just bump a memory's importance and tag it deprecated
    patch.py 9 --importance 0.2 --add-tag deprecated

    # Rewrite the description for memory named lang-chinese
    patch.py lang-chinese --description "user wants Chinese in chat only"

    # Move a memory between tiers by changing its type
    patch.py 12 --type reference
"""
from __future__ import annotations

import argparse
import sys

from _common import bootstrap, emit_json

bootstrap()

from engram import MemoryManager, MemoryType  # noqa: E402


_TYPES = [MemoryType.USER, MemoryType.FEEDBACK,
          MemoryType.PROJECT, MemoryType.REFERENCE]


def _parse_target(raw: str):
    """If ``raw`` is all digits we treat it as an id; otherwise as a name.

    Lets the CLI accept ``patch.py 9 …`` and ``patch.py lang-chinese …``
    interchangeably.
    """
    return int(raw) if raw.isdigit() else raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target",
                    help="id (integer) or name (slug) of the memory to patch.")
    ap.add_argument("--description", default=None,
                    help="Replace the description (triggers re-embed; id changes).")
    ap.add_argument("--content", default=None,
                    help="Replace the content (triggers re-embed; id changes).")
    ap.add_argument("--importance", type=float, default=None,
                    help="Set importance ∈ [0, 1].")
    ap.add_argument("--type", choices=_TYPES, default=None, dest="new_type",
                    help="Change the memory type; if it crosses the global/local "
                         "tier boundary the row is moved and the id changes.")
    ap.add_argument("--rename", default=None,
                    help="Change the memory's name (kebab-case slug).")
    ap.add_argument("--set-tag", action="append", default=None,
                    help="Replace the tag list (repeat).  Mutually exclusive "
                         "with --add-tag/--remove-tag.")
    ap.add_argument("--add-tag", action="append", default=None,
                    help="Append to the existing tag list (repeat).")
    ap.add_argument("--remove-tag", action="append", default=None,
                    help="Remove tags from the existing list (repeat).")
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args()

    target = _parse_target(args.target)

    # Surface the conflict ourselves so the message names *these* flags
    # rather than the engram-internal ``tags`` / ``add_tags`` parameters.
    if args.set_tag is not None and (args.add_tag or args.remove_tag):
        print("error: --set-tag is mutually exclusive with --add-tag/--remove-tag",
              file=sys.stderr)
        return 2

    with MemoryManager() as mgr:
        try:
            hit = mgr.patch(
                target,
                description=args.description,
                content=args.content,
                tags=args.set_tag,
                add_tags=args.add_tag,
                remove_tags=args.remove_tag,
                importance=args.importance,
                type=args.new_type,
                name=args.rename,
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    if args.as_json:
        emit_json(hit.to_dict())
    else:
        print(f"patched: id={hit.id} tier={hit.tier} type={hit.type} name={hit.name}")
        print(f"  desc       : {hit.description}")
        print(f"  importance : {hit.importance:.3f}")
        print(f"  tags       : {hit.tags}")
        print(f"  hits       : {hit.hits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
