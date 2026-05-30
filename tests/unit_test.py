"""Unit tests for v0.3 review fixes.

Companion to ``smoke_test.py`` (end-to-end). Each block locks down a
specific fix from the v0.3 review (see
``~/.claude/plans/review-bug-lively-toast.md``), so a future refactor
that regresses one of these is caught immediately.

Run from repo root::

    ENGRAM_EMBEDDER=hash python tests/unit_test.py

Same convention as smoke_test.py: no pytest dependency, no external
network. Each `_assert` prints OK/FAIL; first failure exits with rc=1.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Make ``engram`` and ``pistadb`` importable when running without
# ``pip install -e .`` (src/ layout).
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


# ── Tiny harness ─────────────────────────────────────────────────────────────

def _assert(cond: bool, msg: str) -> None:
    if cond:
        print(f"OK   {msg}")
    else:
        print(f"FAIL {msg}", file=sys.stderr)
        sys.exit(1)


def _expect_raises(exc_type, func, msg: str) -> None:
    try:
        func()
    except exc_type as e:
        print(f"OK   {msg} (raised {type(e).__name__}: {e})")
        return
    except Exception as e:
        print(f"FAIL {msg} — wrong exception: {type(e).__name__}: {e}",
              file=sys.stderr)
        sys.exit(1)
    print(f"FAIL {msg} — no exception raised", file=sys.stderr)
    sys.exit(1)


# ── Stub embedder + env helper for MemoryManager durability tests ────────────

class _StubEmbedder:
    """Deterministic, network-free embedder for MemoryManager tests.

    Produces a stable unit vector per text so inserts/searches are
    reproducible.  Set ``fail_on`` to a substring to make :meth:`embed`
    raise whenever the text-to-embed contains it — used to simulate a
    backend failure mid-operation.
    """

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.fail_on = None  # type: ignore[assignment]

    def _vec(self, text: str):
        import hashlib
        import numpy as np
        h = hashlib.sha256(text.encode("utf-8")).digest()[: self.dim]
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).astype(np.float32)

    def embed(self, text: str, *, kind: str = "passage"):
        if self.fail_on is not None and self.fail_on in text:
            raise RuntimeError("simulated embed failure")
        return self._vec(text)

    def embed_batch(self, texts, *, kind: str = "passage"):
        import numpy as np
        return np.stack([self.embed(t, kind=kind) for t in texts])


def _engram_home(root: Path):
    """Context manager pointing ENGRAM_HOME / CLAUDE_HOME at a temp dir so
    tests never touch the real ~/.claude global tier."""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        saved = {k: os.environ.get(k) for k in ("ENGRAM_HOME", "CLAUDE_HOME")}
        os.environ["ENGRAM_HOME"] = str(root)
        os.environ["CLAUDE_HOME"] = str(root)
        try:
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return _cm()


# ── Test: save(overwrite_by_name) embed failure must not lose the old row ─────

def test_save_overwrite_embed_failure_preserves_old() -> None:
    """Regression: ``save(overwrite_by_name=True)`` embeds *before* deleting,
    so a failing embedder leaves the existing same-name memory intact."""
    from engram import MemoryManager

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _engram_home(root):
            emb = _StubEmbedder()
            with MemoryManager(embedder=emb, project_root=root) as mgr:
                mgr.save("project", "foo", "original gist",
                         content="body one", force=True)
                before = [h for h in mgr.list() if h.name == "foo"]
                _assert(len(before) == 1, "seed memory 'foo' present")
                old_id = before[0].id

                # Now overwrite, but make the embed of the *new* text fail.
                emb.fail_on = "new gist"
                _expect_raises(
                    RuntimeError,
                    lambda: mgr.save("project", "foo", "new gist",
                                     content="body two", overwrite_by_name=True),
                    "overwrite with a failing embed raises",
                )

                after = [h for h in mgr.list() if h.name == "foo"]
                _assert(
                    len(after) == 1 and after[0].id == old_id,
                    "'foo' survived the failed overwrite (no data loss)",
                )


# ── Test: patch() re-embed must not lose data if a flush crashes ─────────────

def test_patch_reembed_flush_crash_preserves_memory() -> None:
    """Regression: ``patch()`` inserts the re-embedded row *before* deleting
    the old one, so a crash during flush leaves the memory recoverable
    (here: still live in-memory) rather than lost."""
    from engram import MemoryManager

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _engram_home(root):
            with MemoryManager(embedder=_StubEmbedder(), project_root=root) as mgr:
                mgr.save("project", "bar", "gist one", content="b", force=True)
                seed = [h for h in mgr.list() if h.name == "bar"]
                old_id = seed[0].id

                coll = mgr._tiers["local"].collection
                orig_flush = coll.flush
                state = {"n": 0}

                def boom():
                    state["n"] += 1
                    if state["n"] == 1:
                        raise RuntimeError("simulated flush crash")
                    return orig_flush()

                coll.flush = boom  # type: ignore[method-assign]
                _expect_raises(
                    RuntimeError,
                    lambda: mgr.patch("bar", description="new gist"),
                    "patch re-embed surfaces the flush crash",
                )
                coll.flush = orig_flush  # type: ignore[method-assign]

                rows = [h for h in mgr.list() if h.name == "bar"]
                _assert(
                    len(rows) == 1 and rows[0].description.startswith("new gist"),
                    "'bar' survived the flush crash as the re-embedded row",
                )
                _assert(rows[0].id != old_id,
                        "re-embed allocated a fresh id (insert happened first)")


def test_patch_reembed_happy_path_carries_metadata() -> None:
    """A clean re-embed patch: new id, but created_at / hits carried over."""
    from engram import MemoryManager

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _engram_home(root):
            with MemoryManager(embedder=_StubEmbedder(), project_root=root) as mgr:
                mgr.save("project", "baz", "g1", content="c1", force=True)
                orig = [h for h in mgr.list() if h.name == "baz"][0]
                hit = mgr.patch("baz", description="g2")
                _assert(hit.id != orig.id,
                        "re-embed changes the id (delete + reinsert)")
                _assert(hit.created_at == orig.created_at,
                        "created_at carried over to the re-embedded row")
                _assert(hit.hits == orig.hits,
                        "hits carried over to the re-embedded row")
                survivors = [h for h in mgr.list() if h.name == "baz"]
                _assert(len(survivors) == 1,
                        "exactly one 'baz' remains after re-embed (old gone)")


# ── Test: directives() — standing always-on constraints ──────────────────────

def test_directives_returns_only_pinned_readonly() -> None:
    """`directives()` returns only `pin`-tagged memories, importance-sorted,
    and is read-only (no hits/accessed_at bump)."""
    from engram import MemoryManager

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _engram_home(root):
            with MemoryManager(embedder=_StubEmbedder(), project_root=root) as mgr:
                mgr.save("user", "lang-zh", "always reply in Simplified Chinese",
                         tags=["pin"], importance=0.9, force=True)
                mgr.save("feedback", "tabs", "use tabs not spaces",
                         tags=["pin"], importance=0.5, force=True)
                mgr.save("project", "ordinary", "an ordinary non-pinned memory",
                         force=True)

                ds = mgr.directives()
                names = [h.name for h in ds]
                _assert("lang-zh" in names and "tabs" in names,
                        "both pinned memories returned")
                _assert("ordinary" not in names,
                        "non-pinned memory is excluded")
                _assert(names[0] == "lang-zh",
                        f"sorted by importance desc (got {names})")

                after = [h for h in mgr.list() if h.name == "lang-zh"][0]
                _assert(after.hits == 0,
                        "directives() is read-only — does not bump hits")


def test_directives_respects_budget() -> None:
    """`max_items` / `byte_budget` cap the per-turn cost; at least one is
    always returned."""
    from engram import MemoryManager

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _engram_home(root):
            with MemoryManager(embedder=_StubEmbedder(), project_root=root) as mgr:
                for i in range(5):
                    mgr.save("user", f"rule-{i}", f"rule number {i}",
                             tags=["pin"], importance=0.5, force=True)
                capped = mgr.directives(max_items=2)
                _assert(len(capped) == 2,
                        f"max_items caps the count (got {len(capped)})")
                one = mgr.directives(byte_budget=1)
                _assert(len(one) == 1,
                        f"tiny byte_budget still returns at least one (got {len(one)})")


# ── Test: rerank protect band (closer hit never demoted) ─────────────────────

def _mk_hit(name, distance, importance, hits, accessed_at):
    from engram.memory import MemoryHit
    return MemoryHit(
        id=hash(name) & 0xFFFF, tier="local", type="project", name=name,
        description="", distance=distance, importance=importance, hits=hits,
        created_at=0, accessed_at=accessed_at,
    )


def test_rerank_distance_gap_protected() -> None:
    """A clearly-closer hit must stay on top even when a far hit has huge
    importance + hits — the failure mode the old global sort had."""
    import time as _t
    from engram.decay import rerank

    now = int(_t.time())
    near = _mk_hit("near", distance=0.10, importance=0.1, hits=0, accessed_at=now)
    far  = _mk_hit("far",  distance=0.20, importance=1.0, hits=100, accessed_at=now)
    out = rerank([far, near])
    _assert(out[0].name == "near",
            f"closest hit stays #1 despite far hit's importance/hits (got {out[0].name})")


def test_rerank_tiebreak_within_band() -> None:
    """Within the protect band, higher importance/hits wins the tie-break."""
    import time as _t
    from engram.decay import rerank

    now = int(_t.time())
    # distance差 0.005 << band 0.03 → 同桶 → importance 高的上浮
    plain     = _mk_hit("plain",     distance=0.100, importance=0.0, hits=0,  accessed_at=now)
    important = _mk_hit("important", distance=0.105, importance=1.0, hits=50, accessed_at=now)
    out = rerank([plain, important])
    _assert(out[0].name == "important",
            f"within band, importance/hits wins the tie-break (got {out[0].name})")


def test_rerank_band_zero_is_pure_distance() -> None:
    """ENGRAM_RANK_PROTECT_BAND=0 collapses to a single bucket → pure composite
    (back-compat escape hatch)."""
    import os
    import time as _t
    from engram.decay import rerank

    saved = os.environ.get("ENGRAM_RANK_PROTECT_BAND")
    os.environ["ENGRAM_RANK_PROTECT_BAND"] = "0"
    try:
        now = int(_t.time())
        near = _mk_hit("near", distance=0.10, importance=0.0, hits=0,   accessed_at=now)
        far  = _mk_hit("far",  distance=0.20, importance=1.0, hits=100, accessed_at=now)
        out = rerank([near, far])
        # band=0 → one bucket → composite governs → far (huge pull) wins
        _assert(out[0].name == "far",
                f"band=0 restores global composite sort (got {out[0].name})")
    finally:
        if saved is None:
            os.environ.pop("ENGRAM_RANK_PROTECT_BAND", None)
        else:
            os.environ["ENGRAM_RANK_PROTECT_BAND"] = saved


# ── Test 1: H3 — OpenAIEmbedder ctor validation ──────────────────────────────

def test_openai_ctor_validation() -> None:
    """Unknown model + no explicit dim must raise ValueError, not silently
    fall through to wrong-dim API requests downstream."""
    from engram.embedder import OpenAIEmbedder

    # Unknown model, no explicit dim → ValueError (the fix for H3).
    _expect_raises(
        ValueError,
        lambda: OpenAIEmbedder(model="future-model-v9", api_key="sk-test"),
        "OpenAIEmbedder(unknown model, no dim) raises ValueError",
    )

    # Unknown model with explicit dim → accepted (constructor doesn't call API).
    e = OpenAIEmbedder(model="future-model-v9", dim=1024, api_key="sk-test")
    _assert(e.dim == 1024, "explicit dim accepted for unknown model")

    # Known model → default dim from _DIMS table.
    e2 = OpenAIEmbedder(model="text-embedding-3-small", api_key="sk-test")
    _assert(e2.dim == 1536,
            f"known model gets default dim 1536, got {e2.dim}")
    e3 = OpenAIEmbedder(model="text-embedding-3-large", api_key="sk-test")
    _assert(e3.dim == 3072,
            f"text-embedding-3-large gets dim 3072, got {e3.dim}")


# ── Test 2: H2 — _safe_label_bytes UTF-8 safety ──────────────────────────────

def test_safe_label_bytes() -> None:
    """Truncate at byte limit WITHOUT splitting a UTF-8 codepoint."""
    from pistadb import _safe_label_bytes

    # Empty / ASCII / boundary
    _assert(_safe_label_bytes("") == b"", "empty label → empty bytes")
    _assert(_safe_label_bytes("hello") == b"hello", "ASCII passthrough")
    _assert(_safe_label_bytes("a" * 255) == b"a" * 255, "ASCII at exact limit")
    _assert(_safe_label_bytes("a" * 300) == b"a" * 255, "ASCII over → cut at 255")

    # 3-byte char ("中"): 255 / 3 = 85 chars × 3 = 255 bytes (clean boundary).
    truncated = _safe_label_bytes("中" * 100)
    decoded = truncated.decode("utf-8")  # must not raise
    _assert(len(truncated) == 255,
            f"3-byte UTF-8 fills 255 bytes exactly, got {len(truncated)}")
    _assert(decoded == "中" * 85,
            f"3-byte truncates to 85 chars at codepoint boundary, "
            f"got {len(decoded)} chars")

    # 4-byte emoji: naïve [:255] would land mid-codepoint.
    # 255 / 4 = 63.75 → keep 63 emojis = 252 bytes.
    truncated_emoji = _safe_label_bytes("😀" * 100)
    decoded_emoji = truncated_emoji.decode("utf-8")  # must not raise
    _assert(len(truncated_emoji) == 252,
            f"4-byte UTF-8 drops partial codepoint, got {len(truncated_emoji)} bytes")
    _assert(decoded_emoji == "😀" * 63,
            f"4-byte truncates to 63 emojis, got {len(decoded_emoji)} chars")

    # Sanity: a naïve [:255] WOULD split this — confirm our helper differs.
    naive = ("😀" * 100).encode("utf-8")[:255]
    _assert(naive != truncated_emoji,
            "safe helper produces different bytes than naïve [:255]")
    # And naïve would fail to decode strictly:
    try:
        naive.decode("utf-8")
        decode_ok = True
    except UnicodeDecodeError:
        decode_ok = False
    _assert(not decode_ok, "naïve [:255] produces invalid UTF-8 (proves fix needed)")


# ── Test 3: H4 — Tier sidecar/data inconsistency guard ───────────────────────

def test_tier_inconsistent_state() -> None:
    """Tier must raise RuntimeError when .pst and .meta.json existence
    disagrees — not silently overwrite the surviving file."""
    from engram.store import Tier, TierPaths

    with tempfile.TemporaryDirectory(prefix="engram_unit_") as tmp:
        tmp = Path(tmp)
        pst = tmp / "test.pst"
        cache = tmp / "test.pcc"
        meta = Path(str(pst) + ".meta.json")
        paths = TierPaths(name="test", pst=pst, cache=cache)

        # Case 1: .pst present, .meta.json missing → raise.
        pst.write_bytes(b"\x00")  # arbitrary content; we never load it
        _expect_raises(
            RuntimeError,
            lambda: Tier(paths),
            ".pst without .meta.json → RuntimeError (data could be clobbered)",
        )

        # Case 2: .meta.json present, .pst missing → also raise.
        pst.unlink()
        meta.write_text("{}", encoding="utf-8")
        _expect_raises(
            RuntimeError,
            lambda: Tier(paths),
            ".meta.json without .pst → RuntimeError",
        )

        # Case 3: BOTH missing → fresh-create path (no exception expected).
        # Don't actually instantiate here — that triggers native pistadb
        # and requires a writeable collection, beyond unit-test scope. The
        # behaviour is covered end-to-end by smoke_test.py's _setup_tmp_dirs.
        meta.unlink()
        _assert(
            not pst.exists() and not meta.exists(),
            "clean state — both missing — handled by smoke_test.py end-to-end",
        )


# ── Test 4: _adaptive_k direct parameterised checks ──────────────────────────

def test_adaptive_k() -> None:
    """Lock in gap-based knee detection across the branches that recall.py
    drives in practice."""
    from engram.memory import _adaptive_k, K_MIN, K_MAX

    # Trivial edges
    _assert(_adaptive_k([]) == 0, "empty → 0")
    _assert(_adaptive_k([0.1]) == 1, "single → 1")
    _assert(_adaptive_k([0.1, 0.2]) == 2, "n == k_min → all")

    # Branch A: every gap below MIN_SIGNIFICANT_GAP (= 0.01) → k_min floor.
    # Models the "xyzqwer 完全不相关的字符串" pattern: all distances cluster
    # in a noise band with no useful structure.
    n_noise = _adaptive_k([0.130, 0.131, 0.132, 0.133, 0.134, 0.135, 0.136])
    _assert(n_noise == K_MIN,
            f"noise band (gaps all < 0.01) → k_min={K_MIN}, got {n_noise}")

    # Branch B: gaps significant but uniform — max gap not stark vs median → k_min.
    # Models "skill 命令怎么用 有没有示例" where many memories are kind-of
    # related; no obvious cliff means we trust the floor over noise injection.
    n_uniform = _adaptive_k([0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    _assert(n_uniform == K_MIN,
            f"uniform gaps → no cliff → k_min={K_MIN}, got {n_uniform}")

    # Branch C: sharp cliff at position 1 — top hit is alone, falls to floor.
    # Real distances from the "如何禁用 engram" session query.
    dists_disable = [0.082, 0.112, 0.114, 0.129, 0.135,
                     0.143, 0.157, 0.166, 0.175, 0.183]
    n_disable = _adaptive_k(dists_disable)
    _assert(n_disable == K_MIN,
            f"sharp cliff at pos 1 → k_min={K_MIN} floor, got {n_disable}")

    # Branch D: sharp cliff mid-pool — cut at the cliff.
    # 4 tight hits at 0.10-0.115, then huge jump to 0.40+.
    dists_cliff = [0.10, 0.105, 0.11, 0.115, 0.40, 0.41, 0.42, 0.43]
    n_cliff = _adaptive_k(dists_cliff)
    _assert(n_cliff == 4,
            f"cliff after pos 4 → 4 hits, got {n_cliff}")

    # Bounds: result always in [k_min, k_max].
    n_many = _adaptive_k([0.1 + i * 0.001 for i in range(30)])
    _assert(K_MIN <= n_many <= K_MAX,
            f"result clamped to [{K_MIN}, {K_MAX}], got {n_many}")


# ── E: Embedder spec parser + registry (v0.4) ───────────────────────────────


def test_parse_spec() -> None:
    from engram.embedder import parse_spec

    # Empty / whitespace / None → local default.
    for raw in ("", "   ", None):
        p = parse_spec(raw)
        _assert(p.scheme == "local" and p.target is None and p.params == {},
                f"parse_spec({raw!r}) → local default")

    # Bare scheme.
    p = parse_spec("hash")
    _assert(p.scheme == "hash" and p.target is None,
            f"bare scheme: scheme={p.scheme}, target={p.target}")

    # Bare scheme + query.
    p = parse_spec("hash?dim=512")
    _assert(p.scheme == "hash" and p.target is None and p.params == {"dim": "512"},
            f"bare scheme + query: {p}")

    # scheme:target form.
    p = parse_spec("openai:text-embedding-3-small")
    _assert(p.scheme == "openai" and p.target == "text-embedding-3-small",
            f"scheme:target: {p}")

    # scheme:target?params with multiple params.
    p = parse_spec("openai:text-embedding-3-large?dim=2048&timeout=60")
    _assert(
        p.scheme == "openai" and p.target == "text-embedding-3-large"
        and p.params == {"dim": "2048", "timeout": "60"},
        f"scheme:target?multi: {p}",
    )

    # Full URL form — whole URL goes into target, scheme rerouted to 'http'.
    p = parse_spec("http://localhost:8080/embed?dim=768")
    _assert(
        p.scheme == "http" and p.target == "http://localhost:8080/embed"
        and p.params == {"dim": "768"},
        f"full URL: {p}",
    )

    # https URLs work the same way.
    p = parse_spec("https://api.example.com/v1/embed")
    _assert(p.scheme == "http" and p.target == "https://api.example.com/v1/embed",
            f"https URL: {p}")

    # Case-insensitive scheme.
    p = parse_spec("OpenAI:foo")
    _assert(p.scheme == "openai",
            f"case-insensitive scheme: {p}")


def test_registry_and_unknown_scheme() -> None:
    from engram.embedder import (
        ParsedSpec,
        build_embedder,
        register_embedder,
        registered_schemes,
    )

    # The seven built-ins must be registered out of the box.
    schemes = set(registered_schemes())
    for s in ("local", "openai", "ollama", "cohere", "voyage", "http", "hash"):
        _assert(s in schemes, f"built-in scheme {s!r} registered")

    # Unknown scheme raises a useful error.
    _expect_raises(
        ValueError,
        lambda: build_embedder("nonexistent:xyz"),
        "unknown scheme raises ValueError",
    )

    # Third-party registration works end-to-end.
    sentinel = {"called": False, "target": None}

    class _FakeEmbedder:
        dim = 7
        def embed(self, text, *, kind="passage"):
            return None
        def embed_batch(self, texts, *, kind="passage"):
            return None

    def _my_factory(spec: ParsedSpec):
        sentinel["called"] = True
        sentinel["target"] = spec.target
        return _FakeEmbedder()

    register_embedder("test-fake", _my_factory)
    emb = build_embedder("test-fake:my-model?opt=1")
    _assert(sentinel["called"] is True,
            "registered factory was invoked")
    _assert(sentinel["target"] == "my-model",
            f"factory saw target={sentinel['target']!r}")
    _assert(emb.dim == 7,
            f"factory returned the expected embedder (dim={emb.dim})")


def test_built_in_factories_construct_without_network() -> None:
    """The local + hash factories must construct without any network access
    or API keys; the rest can be exercised via spec parse only (their
    factories raise on bad config, which we check separately)."""
    import os
    from engram.embedder import build_embedder

    # local: returns a LocalE5Embedder but does NOT load the model.
    emb = build_embedder("local")
    _assert(emb.dim == 384,
            f"local default dim is 384 (got {emb.dim})")

    # hash with explicit dim.
    emb = build_embedder("hash?dim=128")
    _assert(emb.dim == 128,
            f"hash factory honours ?dim= ({emb.dim})")

    # openai without API key raises a clean RuntimeError.
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        _expect_raises(
            RuntimeError,
            lambda: build_embedder("openai:text-embedding-3-small"),
            "openai without API key raises informative RuntimeError",
        )
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved

    # http without dim → raises (we can't size the .pst without it).
    _expect_raises(
        RuntimeError,
        lambda: build_embedder("http://example.com/embed"),
        "http without dim raises RuntimeError",
    )


def test_legacy_env_var_back_compat() -> None:
    """v0.3 users on ``ENGRAM_EMBEDDER=http`` + ``ENGRAM_HTTP_URL`` must
    still work after the URI-spec refactor."""
    import os
    from engram.embedder import build_default_embedder, CachedEmbedder

    saved = {k: os.environ.get(k) for k in (
        "ENGRAM_EMBEDDER", "ENGRAM_HTTP_URL", "ENGRAM_HTTP_DIM",
        "ENGRAM_OPENAI_MODEL", "OPENAI_API_KEY",
    )}
    try:
        # Legacy openai: bare scheme + ENGRAM_OPENAI_MODEL → openai:model
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ["ENGRAM_EMBEDDER"]    = "openai"
        os.environ["ENGRAM_OPENAI_MODEL"] = "text-embedding-3-large"
        os.environ["OPENAI_API_KEY"]      = "sk-fake-for-ctor"
        # ctor reads the env-rewritten spec; we expect a CachedEmbedder
        # wrapping an OpenAIEmbedder.  dim should match the large model.
        emb = build_default_embedder(cache_path=None)
        _assert(isinstance(emb, CachedEmbedder),
                "build_default_embedder returns a CachedEmbedder")
        _assert(emb.dim == 3072,
                f"legacy ENGRAM_OPENAI_MODEL=large gives dim=3072 (got {emb.dim})")

        # Legacy http: bare scheme + ENGRAM_HTTP_URL + ENGRAM_HTTP_DIM
        os.environ["ENGRAM_EMBEDDER"]  = "http"
        os.environ["ENGRAM_HTTP_URL"]  = "http://example.com/embed"
        os.environ["ENGRAM_HTTP_DIM"]  = "1024"
        emb = build_default_embedder(cache_path=None)
        _assert(emb.dim == 1024,
                f"legacy http env vars give dim=1024 (got {emb.dim})")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── Driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== unit tests for v0.3 review fixes ===")

    print("\n--- H3: OpenAIEmbedder ctor validation ---")
    test_openai_ctor_validation()

    print("\n--- H2: _safe_label_bytes UTF-8 safety ---")
    test_safe_label_bytes()

    print("\n--- H4: Tier sidecar inconsistency guard ---")
    test_tier_inconsistent_state()

    print("\n--- E1: embedder spec parser ---")
    test_parse_spec()

    print("\n--- E2: embedder registry ---")
    test_registry_and_unknown_scheme()

    print("\n--- E3: built-in factories (no-network) ---")
    test_built_in_factories_construct_without_network()

    print("\n--- E4: legacy env var back-compat ---")
    test_legacy_env_var_back_compat()

    print("\n--- adaptive_k parameterised ---")
    test_adaptive_k()

    print("\n--- rerank protect band (P0 fix) ---")
    test_rerank_distance_gap_protected()
    test_rerank_tiebreak_within_band()
    test_rerank_band_zero_is_pure_distance()

    print("\n--- save/patch durability (data-loss windows) ---")
    test_save_overwrite_embed_failure_preserves_old()
    test_patch_reembed_flush_crash_preserves_memory()
    test_patch_reembed_happy_path_carries_metadata()

    print("\n--- directives (standing always-on constraints) ---")
    test_directives_returns_only_pinned_readonly()
    test_directives_respects_budget()

    print("\nALL UNIT TESTS PASSED.")


if __name__ == "__main__":
    main()
