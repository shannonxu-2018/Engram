"""``engram-mcp`` — Model Context Protocol server for Engram.

Why this exists
---------------

Claude Code calls Engram by executing the per-script CLIs under
``~/.claude/skills/engram/scripts/`` — one fresh Python process per
``recall``/``save``.  That's fine when Claude Code is the only host,
but two things changed:

* **OpenCode** and **Codex** don't have file-based skills.  Their
  integration point is **MCP**.
* Cold-starting Python + loading multilingual-e5-small for every recall
  costs 3-5 s.  An always-on process amortises that to ~10 ms.

So we expose ``recall`` / ``save`` / ``expand`` / ``list`` / ``forget``
/ ``related`` as MCP tools, backed by a single long-lived
:class:`engram.MemoryManager`.

Transport
---------

The script speaks **MCP over stdio**.  Hosts (Claude Code, OpenCode,
Codex) launch it as a child process with the entry point
``engram-mcp`` (declared in ``pyproject.toml``).

Dependency story
----------------

If the official ``mcp`` Python SDK is on the path we use it (preferred —
proper schema, completions, etc.).  Otherwise we fall back to a tiny
self-contained stdio JSON-RPC loop that implements just enough of MCP
to be useful (``initialize`` / ``tools/list`` / ``tools/call``).  This
keeps ``pip install engram`` free of the extra dep; users who want the
spec-perfect server add ``pip install "engram[mcp]"``.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Dict, List, Optional

from . import MemoryManager


# ── Tool implementations ─────────────────────────────────────────────────────
#
# Each ``_tool_*`` takes a JSON-serialisable args dict and returns a
# JSON-serialisable result.  These are the actual MCP tools — both the
# real-SDK path and the fallback path dispatch through this table.

def _tool_recall(mgr: MemoryManager, args: Dict[str, Any]) -> Dict[str, Any]:
    query = str(args["query"])
    k = args.get("k")
    type_ = args.get("type")
    tier = args.get("tier")
    with_content = bool(args.get("with_content", False))
    hits = mgr.recall(
        query,
        k=int(k) if k is not None else None,
        types=[type_] if type_ else None,
        tiers=[tier] if tier else None,
        with_content=with_content,
    )
    return {"hits": [h.to_dict() for h in hits]}


def _tool_save(mgr: MemoryManager, args: Dict[str, Any]) -> Dict[str, Any]:
    res = mgr.save(
        type=str(args["type"]),
        name=str(args["name"]),
        description=str(args["description"]),
        content=str(args.get("content") or ""),
        tags=list(args.get("tags") or []),
        importance=float(args.get("importance", 0.5)),
        force=bool(args.get("force", False)),
        overwrite_by_name=bool(args.get("update", False)),
    )
    return res.to_dict()


def _tool_expand(mgr: MemoryManager, args: Dict[str, Any]) -> Dict[str, Any]:
    hit = mgr.expand(int(args["id"]))
    return hit.to_dict()


def _tool_list(mgr: MemoryManager, args: Dict[str, Any]) -> Dict[str, Any]:
    hits = mgr.list(
        type=args.get("type"),
        tier=args.get("tier"),
        limit=int(args.get("limit", 100)),
    )
    return {"hits": [h.to_dict() for h in hits]}


def _tool_forget(mgr: MemoryManager, args: Dict[str, Any]) -> Dict[str, Any]:
    older = args.get("older_than_days")
    common = dict(
        name=args.get("name"),
        id=int(args["id"]) if args.get("id") is not None else None,
        type=args.get("type"),
        tier=args.get("tier"),
        older_than_days=float(older) if older is not None else None,
    )
    if bool(args.get("dry_run", False)):
        targets = mgr.find_for_forget(**common)
        return {"would_delete": [h.to_dict() for h in targets]}
    n = mgr.forget(**common)
    return {"removed": int(n)}


def _tool_patch(mgr: MemoryManager, args: Dict[str, Any]) -> Dict[str, Any]:
    raw = args.get("target")
    if raw is None:
        raise ValueError("patch: 'target' is required (id integer or name slug)")
    # The MCP wire shape sends ints as JSON numbers and names as JSON
    # strings, so we get clean discrimination here without parsing.
    target: Any
    if isinstance(raw, str) and raw.isdigit():
        target = int(raw)
    else:
        target = raw
    hit = mgr.patch(
        target,
        description=args.get("description"),
        content=args.get("content"),
        tags=args.get("tags"),
        add_tags=args.get("add_tags"),
        remove_tags=args.get("remove_tags"),
        importance=(
            float(args["importance"]) if args.get("importance") is not None else None
        ),
        type=args.get("type"),
        name=args.get("name"),
    )
    return hit.to_dict()


def _tool_related(mgr: MemoryManager, args: Dict[str, Any]) -> Dict[str, Any]:
    # ``args.get("name") or args.get("id")`` would silently treat ``id=0`` as
    # missing.  PistaDB currently allocates auto-ids from 1, but we don't
    # want to bake that in here.  Explicit ``is not None`` checks.
    name = args.get("name")
    seed: Any
    if name is not None and name != "":
        seed = name
    elif args.get("id") is not None:
        seed = args.get("id")
    else:
        raise ValueError("related: pass 'name' or 'id'")
    if isinstance(seed, str) and seed.isdigit():
        seed = int(seed)
    hits = mgr.recall_related(
        seed,
        depth=int(args.get("depth", 1)),
        with_content=bool(args.get("with_content", False)),
    )
    return {"hits": [h.to_dict() for h in hits]}


# Tool schema published to the host so the model knows what to call.
# Kept deliberately compact — the LLM sees this every session.

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "engram_recall",
        "description": (
            "Semantic KNN recall over the Engram memory store. "
            "Returns the most similar memories to <query>. "
            "k defaults to adaptive (gap-based, 2-10 hits). "
            "Filter by type (user|feedback|project|reference) or tier (global|local)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 20},
                "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"]},
                "tier": {"type": "string", "enum": ["global", "local"]},
                "with_content": {"type": "boolean"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "engram_save",
        "description": (
            "Encode a new memory.  type ∈ {user,feedback,project,reference}. "
            "Returns status=inserted|overwritten, or status=merge_suggestion "
            "if a near-duplicate exists (re-call with update=true or force=true)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"]},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "force": {"type": "boolean"},
                "update": {"type": "boolean"},
            },
            "required": ["type", "name", "description"],
        },
    },
    {
        "name": "engram_expand",
        "description": "Fetch full content for one memory id.  Bumps hits++/accessed_at.",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        },
    },
    {
        "name": "engram_list",
        "description": "Enumerate memories without embedding cost.  Optional type/tier/limit filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"]},
                "tier": {"type": "string", "enum": ["global", "local"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
        },
    },
    {
        "name": "engram_forget",
        "description": (
            "Delete memories by name / id / age. Pass at least one of "
            "name, id, or older_than_days. ``type`` and ``tier`` are "
            "narrowing filters. With dry_run=true, returns "
            "{would_delete:[...]} without touching the store — strongly "
            "recommended before a broad older_than_days sweep. "
            "older_than_days is computed against accessed_at, so "
            "frequently-recalled old memories survive."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "id": {"type": "integer"},
                "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"]},
                "tier": {"type": "string", "enum": ["global", "local"]},
                "older_than_days": {"type": "number", "minimum": 0},
                "dry_run": {"type": "boolean"},
            },
        },
    },
    {
        "name": "engram_patch",
        "description": (
            "Modify selected fields of one existing memory. "
            "Pass target as the id (integer) or unique name (string). "
            "Only the fields you include are changed. "
            "Editing description or content (or moving across the "
            "global/local tier boundary via 'type') triggers re-embed + "
            "delete-and-reinsert, so the id changes (created_at and "
            "hits are preserved). All other changes are in place; the "
            "id stays the same. Tag interaction: pass 'tags' to replace "
            "the list wholesale; or pass 'add_tags' / 'remove_tags' for "
            "delta edits (but not both)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "oneOf": [
                        {"type": "integer"},
                        {"type": "string"},
                    ],
                    "description": "Memory id (int) or unique name (string).",
                },
                "description": {"type": "string"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "add_tags": {"type": "array", "items": {"type": "string"}},
                "remove_tags": {"type": "array", "items": {"type": "string"}},
                "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"]},
                "name": {"type": "string"},
            },
            "required": ["target"],
        },
    },
    {
        "name": "engram_related",
        "description": "BFS walk over [[name]] tag edges from a seed memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "id": {"type": "integer"},
                "depth": {"type": "integer", "minimum": 1, "maximum": 5},
                "with_content": {"type": "boolean"},
            },
        },
    },
]


TOOL_DISPATCH: Dict[str, Callable[[MemoryManager, Dict[str, Any]], Dict[str, Any]]] = {
    "engram_recall":  _tool_recall,
    "engram_save":    _tool_save,
    "engram_expand":  _tool_expand,
    "engram_list":    _tool_list,
    "engram_forget":  _tool_forget,
    "engram_patch":   _tool_patch,
    "engram_related": _tool_related,
}


# ── Fallback transport (works without the official mcp SDK) ──────────────────

PROTOCOL_VERSION = "2024-11-05"


def _server_version() -> str:
    """Read the installed package version dynamically.

    Hardcoding the version drifts every time the wheel bumps — read it
    via importlib.metadata so ``serverInfo`` is always honest.
    """
    try:
        from importlib.metadata import version
        return version("engram")
    except Exception:
        return "0.0.0+unknown"


def _fallback_run(mgr: MemoryManager) -> None:
    """Tiny stdio JSON-RPC loop covering the MCP methods clients need.

    Hosts we target (Claude Code, OpenCode, Codex) all speak the same
    JSON-RPC-over-stdio frame: one JSON object per line, newline-delimited.
    """
    in_  = sys.stdin
    out  = sys.stdout

    def _send(msg: Dict[str, Any]) -> None:
        out.write(json.dumps(msg) + "\n")
        out.flush()

    def _ok(req_id: Any, result: Any) -> None:
        _send({"jsonrpc": "2.0", "id": req_id, "result": result})

    def _err(req_id: Any, code: int, message: str) -> None:
        _send({"jsonrpc": "2.0", "id": req_id,
               "error": {"code": code, "message": message}})

    for line in in_:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _err(None, -32700, f"parse error: {e}")
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}

        if method == "initialize":
            _ok(req_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "engram", "version": _server_version()},
            })
        elif method == "notifications/initialized":
            # Notification — no response required.
            continue
        elif method == "tools/list":
            _ok(req_id, {"tools": TOOLS})
        elif method == "tools/call":
            tname = params.get("name")
            targs = params.get("arguments") or {}
            fn = TOOL_DISPATCH.get(tname)
            if fn is None:
                _err(req_id, -32601, f"unknown tool {tname!r}")
                continue
            try:
                result = fn(mgr, targs)
                _ok(req_id, {
                    "content": [{"type": "text", "text": json.dumps(result)}],
                    "isError": False,
                })
            except Exception as e:
                _ok(req_id, {
                    "content": [{"type": "text", "text": f"error: {e}"}],
                    "isError": True,
                })
        elif method == "ping":
            _ok(req_id, {})
        elif method == "shutdown":
            _ok(req_id, {})
            break
        else:
            if req_id is not None:
                _err(req_id, -32601, f"method not found: {method}")


# ── Real-SDK transport (used when ``mcp`` is installed) ──────────────────────

def _sdk_run(mgr: MemoryManager) -> int:
    """Use the official ``mcp`` SDK if available.

    Returns an exit code; ``99`` signals "SDK not importable — caller
    should fall back".
    """
    try:
        import anyio  # type: ignore
        from mcp.server import Server  # type: ignore
        from mcp.server.stdio import stdio_server  # type: ignore
        from mcp import types as mcp_types  # type: ignore
    except Exception:
        return 99

    server = Server("engram")

    @server.list_tools()
    async def _list_tools() -> List[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
        fn = TOOL_DISPATCH.get(name)
        if fn is None:
            raise ValueError(f"unknown tool {name!r}")
        result = fn(mgr, arguments or {})
        return [mcp_types.TextContent(type="text", text=json.dumps(result))]

    async def _main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(_main)
    return 0


# ── Entry point ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="engram-mcp",
        description="Engram memory store, served over MCP (stdio).",
    )
    parser.add_argument(
        "--no-sdk",
        action="store_true",
        help="Force the built-in fallback transport even if mcp is installed.",
    )
    args = parser.parse_args(argv)

    # One long-lived manager; the embedder loads once, then serves all calls.
    mgr = MemoryManager()
    try:
        if not args.no_sdk:
            rc = _sdk_run(mgr)
            if rc != 99:
                return rc
        _fallback_run(mgr)
        return 0
    finally:
        try:
            mgr.close()
        except Exception:
            pass


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
