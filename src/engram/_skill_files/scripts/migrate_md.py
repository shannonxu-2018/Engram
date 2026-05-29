#!/usr/bin/env python3
"""migrate_md.py — import the legacy MEMORY.md / per-file MD memory system
into Engram.

Reads every ``*.md`` (other than ``MEMORY.md``) in the source directory,
parses the YAML-ish frontmatter (``name``, ``description``, ``metadata.type``)
and the body, and saves each as a memory.  Idempotent: re-runs use
``--update`` so re-importing the same files replaces rather than duplicates.

Usage:
    migrate_md.py <source-dir>  [--dry-run] [--force]

Source directory layout (current Claude Code default)::

    ~/.claude/projects/<slug>/memory/
        MEMORY.md          (index, ignored)
        user_role.md       (one memory each)
        feedback_*.md
        project_*.md
        ...
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from _common import bootstrap

bootstrap()

from engram import MemoryManager  # noqa: E402
from engram.memory import VALID_TYPES  # noqa: E402


# YAML frontmatter is very small here — no PyYAML dependency, just a tiny parser.
_FM_RE   = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_KV_RE   = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")


def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    head, body = m.group(1), m.group(2)

    fields: Dict[str, str] = {}
    # Handle a two-level nesting (metadata.type) — flatten lazily.
    current_prefix: Optional[str] = None
    for line in head.splitlines():
        stripped = line.rstrip()
        if not stripped:
            continue
        # nested block start, e.g. "metadata:" with following indented "  type: x"
        if stripped.endswith(":") and not stripped.lstrip().startswith("-"):
            key = stripped.rstrip(":").strip()
            # nested key only if line is unindented
            if not line.startswith((" ", "\t")):
                current_prefix = key
                continue
        kv = _KV_RE.match(line)
        if not kv:
            continue
        key, value = kv.group(1), kv.group(2).strip()
        # Strip surrounding quotes for VARCHAR-style values.
        if value.startswith(("'", '"')) and value.endswith(value[0]) and len(value) >= 2:
            value = value[1:-1]
        # Indented line under a parent → nested key.
        if line.startswith((" ", "\t")) and current_prefix:
            fields[f"{current_prefix}.{key}"] = value
        else:
            current_prefix = None
            fields[key] = value
    return fields, body.strip()


def _iter_md_files(src: Path) -> Iterable[Path]:
    for p in sorted(src.glob("*.md")):
        if p.name.lower() == "memory.md":
            continue
        yield p


def _classify(fields: Dict[str, str]) -> Optional[str]:
    """Find the memory type in the frontmatter or guess from filename."""
    t = fields.get("metadata.type") or fields.get("type")
    if t and t in VALID_TYPES:
        return t
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", type=Path,
                    help="Directory containing the legacy MD memory files.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and report; do not write.")
    ap.add_argument("--force", action="store_true",
                    help="Skip the write-time dedup check (insert always).")
    args = ap.parse_args()

    src: Path = args.source
    if not src.is_dir():
        print(f"error: source dir not found: {src}", file=sys.stderr)
        return 1

    inserted = updated = skipped = errored = 0

    with MemoryManager() as mgr:
        for path in _iter_md_files(src):
            text = path.read_text(encoding="utf-8", errors="replace")
            fields, body = _parse_frontmatter(text)
            mem_type = _classify(fields)
            if mem_type is None:
                print(f"SKIP  {path.name}: no type in frontmatter")
                skipped += 1
                continue

            name = fields.get("name") or path.stem
            description = fields.get("description") or body.split("\n", 1)[0][:300]
            content = body

            if args.dry_run:
                print(f"DRY   {path.name} → type={mem_type} name={name}")
                print(f"      desc: {description[:80]}…")
                continue

            try:
                res = mgr.save(
                    type=mem_type,
                    name=name,
                    description=description,
                    content=content,
                    force=args.force,
                    overwrite_by_name=True,   # idempotent re-run
                )
                if res.status in ("inserted", "overwritten"):
                    inserted += 1 if res.status == "inserted" else 0
                    updated  += 1 if res.status == "overwritten" else 0
                    tag = "INS " if res.status == "inserted" else "UPD "
                    print(f"{tag} {path.name} → id={res.id} tier={res.tier}")
                elif res.status == "merge_suggestion":
                    skipped += 1
                    d = res.duplicate_of
                    print(f"DUP  {path.name}: matches id={d.id} name={d.name}")
            except Exception as e:
                errored += 1
                print(f"ERR  {path.name}: {e}", file=sys.stderr)

    print(
        f"\nmigration done: inserted={inserted}, updated={updated}, "
        f"skipped={skipped}, errored={errored}"
    )
    return 0 if errored == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
