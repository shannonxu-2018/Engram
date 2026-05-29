"""Embedding backends for Engram.

Defines the :class:`Embedder` protocol plus seven concrete implementations,
a small **URI-spec parser** and a **registry** so third parties can plug in
their own backend without forking Engram.

Backends shipped
----------------
* :class:`LocalE5Embedder`   — multilingual-e5-small via sentence-transformers
                                (default; downloads from HuggingFace on first
                                use unless ``ENGRAM_E5_MODEL_PATH`` is set)
* :class:`OpenAIEmbedder`    — text-embedding-3-small / -large via HTTP
* :class:`HTTPEmbedder`      — generic OpenAI-compatible HTTP endpoint
* :class:`OllamaEmbedder`    — local Ollama instance (``/api/embed``)
* :class:`CohereEmbedder`    — Cohere ``embed-*-v3.0`` family
* :class:`VoyageEmbedder`    — Voyage AI ``voyage-3*`` family
* :class:`HashEmbedder`      — deterministic hash → vector, for tests only

URI spec (env var ``ENGRAM_EMBEDDER``)
--------------------------------------

A single ``spec`` string picks the backend and its options:

::

    local                                       # default (alias of "")
    local?device=cuda&model_path=/abs/path
    openai:text-embedding-3-small
    openai:text-embedding-3-large?dim=2048
    ollama:nomic-embed-text                     # localhost:11434
    ollama:bge-m3?host=http://my-ollama:11434
    cohere:embed-multilingual-v3.0?input_type=search_document
    voyage:voyage-3
    http://localhost:8080/embeddings?dim=1024   # raw URL → HTTPEmbedder
    hash?dim=512                                # tests only

Use ``register_embedder()`` to plug in your own scheme; see the bottom of
this module for the six built-in factories as a template.

Caching
-------
Every backend is wrapped transparently in :class:`CachedEmbedder` so
identical strings are embedded only once per process *and* across process
restarts (the cache file lives next to the ``.pst``).
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Sequence,
    runtime_checkable,
)
from urllib.parse import parse_qsl

import numpy as np


# ── URI spec parser ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParsedSpec:
    """A parsed ``ENGRAM_EMBEDDER`` spec.

    Examples (input → ``(scheme, target, params)``):

    * ``""``                                       → ``("local", None, {})``
    * ``"local?device=cuda"``                      → ``("local", None, {"device": "cuda"})``
    * ``"openai:text-embedding-3-large?dim=2048"`` → ``("openai", "text-embedding-3-large", {"dim": "2048"})``
    * ``"http://x.y/z?dim=768"``                   → ``("http", "http://x.y/z", {"dim": "768"})``

    Notes:

    * ``scheme`` is always lower-cased so registry lookups are
      case-insensitive.
    * ``params`` values are kept as strings — each factory casts to the
      type it expects.  This keeps the parser dumb and the factories
      explicit about their contract.
    """
    scheme: str
    target: Optional[str]
    params: Dict[str, str] = field(default_factory=dict)


def parse_spec(spec: Optional[str]) -> ParsedSpec:
    """Parse an ``ENGRAM_EMBEDDER`` spec string.

    Accepts three forms:

    * **Empty / "local"** — the default local embedder.
    * **``scheme:target?key=val&...``** — scheme-prefixed.  ``target``
      is whatever the factory wants (model id for ``openai``, model name
      for ``ollama``, etc.).  The ``?...`` query string is optional.
    * **Full URL** — anything containing ``://``.  The full URL becomes
      the ``target`` of an ``http``-scheme spec, query string parsed as
      params.  Convenient for ``ENGRAM_EMBEDDER=http://localhost:8080/...``.
    """
    if spec is None or not spec.strip():
        return ParsedSpec(scheme="local", target=None)

    raw = spec.strip()

    # Full-URL form: ``http://...`` / ``https://...`` / any ``foo://...``
    # — keep the whole URL as the target, route to the http factory.
    if "://" in raw:
        url, _, qs = raw.partition("?")
        params = {k: v for k, v in parse_qsl(qs, keep_blank_values=False)}
        return ParsedSpec(scheme="http", target=url, params=params)

    # ``scheme:target?key=val`` form.
    scheme, sep, rest = raw.partition(":")
    if not sep:
        # Bare scheme like ``hash`` or ``local`` — still split off ``?``.
        scheme_only, _, qs = scheme.partition("?")
        params = {k: v for k, v in parse_qsl(qs, keep_blank_values=False)}
        return ParsedSpec(scheme=scheme_only.lower(), target=None, params=params)

    target, _, qs = rest.partition("?")
    params = {k: v for k, v in parse_qsl(qs, keep_blank_values=False)}
    return ParsedSpec(
        scheme=scheme.lower(),
        target=target or None,
        params=params,
    )


# ── Registry ─────────────────────────────────────────────────────────────────

# ``Embedder`` protocol is declared below for clarity; the registry-typing
# uses ``Any`` so we can populate it before the protocol class is defined.
_REGISTRY: Dict[str, Callable[[ParsedSpec], Any]] = {}


def register_embedder(
    scheme: str,
    factory: Callable[[ParsedSpec], "Embedder"],
) -> None:
    """Register a factory under ``scheme`` so it can be selected via
    ``ENGRAM_EMBEDDER=scheme:...``.

    Third-party packages can extend Engram without modifying it:

    .. code-block:: python

        from engram.embedder import register_embedder, ParsedSpec

        def _my_factory(spec: ParsedSpec):
            return MyEmbedder(model=spec.target,
                              endpoint=spec.params.get("endpoint"))

        register_embedder("my-backend", _my_factory)

        # ...then anywhere in the user's environment:
        #   ENGRAM_EMBEDDER='my-backend:foo?endpoint=https://...'

    The factory should:

    * Read all options it needs from ``spec.target`` / ``spec.params``.
    * Raise a clean error (ideally ``RuntimeError`` with a "how to fix
      it" hint) if a required option is missing or the underlying
      service can't be reached.
    * Return an object that implements the :class:`Embedder` protocol.
    """
    _REGISTRY[scheme.lower()] = factory


def registered_schemes() -> List[str]:
    """List the schemes the registry currently knows."""
    return sorted(_REGISTRY)


def build_embedder(spec: Optional[str] = None) -> "Embedder":
    """Construct an embedder from a spec string.

    See :func:`parse_spec` for the syntax and the module docstring for
    the list of built-in schemes.  Use :func:`build_default_embedder`
    if you want the cached + env-var-aware version.
    """
    parsed = parse_spec(spec)
    factory = _REGISTRY.get(parsed.scheme)
    if factory is None:
        raise ValueError(
            f"unknown embedder scheme {parsed.scheme!r}; "
            f"registered: {registered_schemes()}. "
            f"Third-party schemes are added via "
            f"engram.embedder.register_embedder()."
        )
    return factory(parsed)


# ── Common helpers (used by remote backends) ─────────────────────────────────

_TRUNCATE_WARN_ONCE: Dict[str, bool] = {}


def _truncate_for_remote(
    texts: Sequence[str],
    *,
    max_chars: int,
    backend: str,
) -> List[str]:
    """Truncate strings longer than ``max_chars`` before a remote call.

    Engram itself caps ``description`` and ``content`` at safe limits
    inside ``MemoryManager.save`` (≤ 480 / ≤ 8000 bytes respectively),
    but a user-supplied **query** at recall time is unbounded.  Cohere
    / Voyage / OpenAI all reject inputs above their token cap with a
    400; the embedder layer should clip + warn rather than propagate
    that as a confusing API error.

    ``max_chars`` is a coarse proxy for the underlying token limit
    (≈ 3 chars per token on English; we leave headroom).
    """
    if not texts:
        return list(texts)
    out: List[str] = []
    truncated = 0
    for t in texts:
        if len(t) > max_chars:
            out.append(t[:max_chars])
            truncated += 1
        else:
            out.append(t)
    if truncated and not _TRUNCATE_WARN_ONCE.get(backend, False):
        _TRUNCATE_WARN_ONCE[backend] = True
        print(
            f"engram: {backend} embedder truncated {truncated} string(s) "
            f"to {max_chars} chars to stay under the model's input limit. "
            f"This warning fires once per process.",
            file=sys.stderr,
        )
    return out


# ── Public protocol ───────────────────────────────────────────────────────────

@runtime_checkable
class Embedder(Protocol):
    """An embedding callable with a known output dimension.

    Implementations MUST L2-normalize their outputs so that cosine distance
    in PistaDB collapses to ``1 - dot(a, b)``.  All vectors returned have
    dtype ``float32`` and shape ``(dim,)`` (single) or ``(n, dim)`` (batch).
    """

    dim: int

    def embed(self, text: str, *, kind: str = "passage") -> np.ndarray: ...

    def embed_batch(
        self, texts: Sequence[str], *, kind: str = "passage"
    ) -> np.ndarray: ...


# ── Default model resolution ──────────────────────────────────────────────────

DEFAULT_E5_MODEL_ID = "intfloat/multilingual-e5-small"


def _default_e5_path() -> str:
    """Resolve the model path/ID for :class:`LocalE5Embedder`.

    Resolution order:

    1. ``ENGRAM_E5_MODEL_PATH`` env var — absolute path to a local snapshot
       (use this for offline / air-gapped installs after pre-downloading).
    2. The HuggingFace model ID ``intfloat/multilingual-e5-small`` —
       sentence-transformers will resolve it from the HF cache, downloading
       (~471 MB) on first use.
    """
    return os.environ.get("ENGRAM_E5_MODEL_PATH", DEFAULT_E5_MODEL_ID)


# ── 1. Local E5 (default) ─────────────────────────────────────────────────────

class LocalE5Embedder:
    """multilingual-e5-small via sentence-transformers.

    E5 convention: prefix queries with ``"query: "`` and stored documents
    with ``"passage: "``.  This class applies the prefix automatically based
    on the ``kind`` argument (``"query"`` or ``"passage"``).

    Falls back to raw ``transformers`` + manual mean-pool / L2-norm if
    ``sentence_transformers`` is unavailable but ``transformers`` is.
    """

    dim = 384

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self._model_path = model_path or _default_e5_path()
        self._device = device
        self._backend: str  # "st" | "hf"
        self._model = None
        self._tok = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        # First-load can be slow — sentence-transformers may download a
        # ~471 MB model from HuggingFace.  Print a single status line to
        # stderr so users don't think the process froze.  We do this
        # *before* touching the model so the message appears immediately;
        # the elapsed-time hint at the end lets users calibrate.
        import time
        _hf_cache_miss = self._looks_like_first_run()
        t0 = time.time()
        if _hf_cache_miss:
            print(
                f"engram: loading {self._model_path} (first run downloads "
                f"~471 MB from HuggingFace, may take 30s on a slow link) ...",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"engram: loading {self._model_path} from cache ...",
                file=sys.stderr,
                flush=True,
            )

        # Preferred: sentence-transformers handles pool + L2 norm.
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self._model_path, device=self._device)
            self._backend = "st"
            print(
                f"engram: loaded in {time.time() - t0:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            return
        except ImportError:
            pass
        except Exception as e:
            # Fall through to HF if e.g. 2_Normalize dir is missing or HF
            # download failed.  Log so the operator can tell why the
            # fast path was skipped rather than silently paying the
            # slower fallback cost forever.
            print(
                f"engram: sentence-transformers load failed ({e!r}); "
                f"falling back to transformers + manual pool.",
                file=sys.stderr,
            )

        # Fallback: transformers + manual mean pool + L2 norm.
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "LocalE5Embedder needs either sentence-transformers or "
                "transformers+torch. Install with:\n"
                "  pip install sentence-transformers\n"
                "or pick another backend (OpenAIEmbedder, HTTPEmbedder)."
            ) from e
        self._tok = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModel.from_pretrained(self._model_path).to(self._device)
        self._model.eval()
        self._torch = torch
        self._backend = "hf"
        print(
            f"engram: loaded in {time.time() - t0:.1f}s "
            f"(transformers fallback)",
            file=sys.stderr,
            flush=True,
        )

    @staticmethod
    def _looks_like_first_run() -> bool:
        """Quick heuristic: is the e5 model cache empty?

        Used only to phrase the progress message — a false positive just
        means we say "downloading" when really sentence-transformers will
        just hit the cache (harmless), and vice versa.  Errs on the side
        of warning the user about a likely slow first call.
        """
        hf_home = (
            os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or str(Path.home() / ".cache" / "huggingface")
        )
        return not (Path(hf_home) / "hub" / "models--intfloat--multilingual-e5-small").is_dir()

    def embed(self, text: str, *, kind: str = "passage") -> np.ndarray:
        return self.embed_batch([text], kind=kind)[0]

    def embed_batch(
        self, texts: Sequence[str], *, kind: str = "passage"
    ) -> np.ndarray:
        self._lazy_load()
        if kind not in ("query", "passage"):
            raise ValueError(f"kind must be 'query' or 'passage', got {kind!r}")
        prefixed = [f"{kind}: {t}" for t in texts]

        if self._backend == "st":
            vecs = self._model.encode(
                prefixed,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return np.ascontiguousarray(vecs, dtype=np.float32)

        # HF backend
        torch = self._torch
        with torch.no_grad():
            inp = self._tok(
                prefixed,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)
            out = self._model(**inp)
            last = out.last_hidden_state                    # (B, T, H)
            mask = inp["attention_mask"].unsqueeze(-1).float()
            summed = (last * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean = summed / counts
            norm = torch.nn.functional.normalize(mean, p=2, dim=1)
        return np.ascontiguousarray(norm.cpu().numpy(), dtype=np.float32)


# ── 2. OpenAI / OpenAI-compatible ─────────────────────────────────────────────

class OpenAIEmbedder:
    """OpenAI text-embedding-3-* via HTTP.

    Reads ``OPENAI_API_KEY`` from the environment (or pass ``api_key=``).
    Set ``base_url=`` to point at an OpenAI-compatible gateway (Azure,
    LiteLLM, OpenRouter, vLLM, etc.).
    """

    _DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        dim: Optional[int] = None,
        timeout: float = 30.0,
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set and no api_key= was provided."
            )
        self._base_url = base_url.rstrip("/")
        # If the model is known, use its native dim as the default; otherwise
        # require the caller to be explicit so we never silently produce
        # wrong-dim vectors (the API would return its own default and we
        # would have no way to tell ours diverged).
        if dim is not None:
            self.dim = dim
        elif model in self._DIMS:
            self.dim = self._DIMS[model]
        else:
            raise ValueError(
                f"OpenAIEmbedder: unknown model {model!r} — pass dim= "
                f"explicitly (or use one of {sorted(self._DIMS)})."
            )
        self._timeout = timeout

    def _post(self, payload: dict) -> dict:
        import json
        import urllib.request

        req = urllib.request.Request(
            f"{self._base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def embed(self, text: str, *, kind: str = "passage") -> np.ndarray:
        return self.embed_batch([text], kind=kind)[0]

    def embed_batch(
        self, texts: Sequence[str], *, kind: str = "passage"
    ) -> np.ndarray:
        # OpenAI embeddings are symmetric — kind is ignored.
        # text-embedding-3-* accept up to 8191 tokens; ~24k chars is a
        # safe cap (≈3 chars/token) that leaves headroom for non-ASCII.
        clipped = _truncate_for_remote(texts, max_chars=24_000, backend="openai")
        body = {"model": self._model, "input": clipped}
        # Send `dimensions` whenever the caller's dim differs from this
        # model's native default — including unknown models where
        # `_DIMS.get()` returns None.  (Previous logic used a fallback
        # arg that made the check vacuous for unknown models, silently
        # dropping the user's dim request.)
        if self.dim != self._DIMS.get(self._model):
            body["dimensions"] = self.dim
        data = self._post(body)
        vecs = np.asarray(
            [d["embedding"] for d in data["data"]], dtype=np.float32
        )
        # OpenAI v3 embeddings are L2-normalized already; re-normalize anyway.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        return np.ascontiguousarray(vecs / norms, dtype=np.float32)


# ── 2b. Ollama (local) ───────────────────────────────────────────────────────

class OllamaEmbedder:
    """Local embeddings via an Ollama daemon (`/api/embed`, batch-capable).

    Defaults to ``http://localhost:11434`` and the ``nomic-embed-text``
    model (768-d) — the most common Ollama embedding setup.  Override
    via ``host=`` / ``model=`` or use the spec syntax::

        ollama:nomic-embed-text                       # default
        ollama:mxbai-embed-large?dim=1024
        ollama:bge-m3?host=http://my-box:11434&dim=1024

    Ollama returns L2-normalised vectors for the standard embed models,
    but we re-normalise defensively (some community models don't).
    """

    # Known model → dim.  Override via ``dim=`` if your model isn't here.
    _DIMS = {
        "nomic-embed-text":  768,
        "mxbai-embed-large": 1024,
        "bge-m3":            1024,
        "bge-large":         1024,
        "all-minilm":        384,
    }

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        dim: Optional[int] = None,
        timeout: float = 30.0,
    ):
        self._model   = model
        self._host    = host.rstrip("/")
        self._timeout = timeout
        if dim is not None:
            self.dim = dim
        elif model in self._DIMS:
            self.dim = self._DIMS[model]
        else:
            raise ValueError(
                f"OllamaEmbedder: unknown model {model!r} — pass dim= "
                f"explicitly (or use one of {sorted(self._DIMS)})."
            )

    def _post(self, payload: dict) -> dict:
        import json
        import urllib.request

        req = urllib.request.Request(
            f"{self._host}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def embed(self, text: str, *, kind: str = "passage") -> np.ndarray:
        return self.embed_batch([text], kind=kind)[0]

    def embed_batch(
        self, texts: Sequence[str], *, kind: str = "passage"
    ) -> np.ndarray:
        # Ollama embed models don't distinguish query/passage — kind ignored.
        clipped = _truncate_for_remote(texts, max_chars=24_000, backend="ollama")
        data = self._post({"model": self._model, "input": list(clipped)})
        vecs = np.asarray(data["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        return np.ascontiguousarray(vecs / norms, dtype=np.float32)


# ── 2c. Cohere ───────────────────────────────────────────────────────────────

class CohereEmbedder:
    """Cohere embed-v3.0 family via HTTPS.

    Reads ``COHERE_API_KEY`` from the environment.  Cohere's API
    differentiates *document* vs *query* embeddings via ``input_type`` —
    we map Engram's ``kind="passage"``/``"query"`` to the right
    ``search_document``/``search_query`` value automatically.
    """

    _DIMS = {
        "embed-english-v3.0":             1024,
        "embed-english-light-v3.0":       384,
        "embed-multilingual-v3.0":        1024,
        "embed-multilingual-light-v3.0":  384,
    }

    def __init__(
        self,
        model: str = "embed-multilingual-v3.0",
        api_key: Optional[str] = None,
        base_url: str = "https://api.cohere.com/v2",
        dim: Optional[int] = None,
        timeout: float = 30.0,
    ):
        self._model    = model
        self._api_key  = api_key or os.environ.get("COHERE_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "CohereEmbedder: COHERE_API_KEY is not set and no api_key= "
                "was provided. Get a free key at https://dashboard.cohere.com/"
            )
        self._base_url = base_url.rstrip("/")
        self._timeout  = timeout
        if dim is not None:
            self.dim = dim
        elif model in self._DIMS:
            self.dim = self._DIMS[model]
        else:
            raise ValueError(
                f"CohereEmbedder: unknown model {model!r} — pass dim= "
                f"explicitly (or use one of {sorted(self._DIMS)})."
            )

    def _post(self, payload: dict) -> dict:
        import json
        import urllib.request

        req = urllib.request.Request(
            f"{self._base_url}/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type":  "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def embed(self, text: str, *, kind: str = "passage") -> np.ndarray:
        return self.embed_batch([text], kind=kind)[0]

    def embed_batch(
        self, texts: Sequence[str], *, kind: str = "passage"
    ) -> np.ndarray:
        # ``kind`` → Cohere's input_type: search_document for stored
        # memories (passage), search_query for recall queries.
        input_type = "search_query" if kind == "query" else "search_document"
        clipped = _truncate_for_remote(texts, max_chars=24_000, backend="cohere")
        body = {
            "model":            self._model,
            "texts":            list(clipped),
            "input_type":       input_type,
            "embedding_types":  ["float"],
        }
        data = self._post(body)
        # v2: {"embeddings": {"float": [[...], [...]]}, ...}
        floats = data.get("embeddings", {}).get("float")
        if floats is None:
            # v1-style fallback: {"embeddings": [[...], [...]]}
            floats = data["embeddings"]
        vecs = np.asarray(floats, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        return np.ascontiguousarray(vecs / norms, dtype=np.float32)


# ── 2d. Voyage AI ────────────────────────────────────────────────────────────

class VoyageEmbedder:
    """Voyage AI ``voyage-3*`` family via HTTPS.

    Reads ``VOYAGE_API_KEY`` from the environment.  Response shape is
    OpenAI-style (``{"data": [{"embedding": [...]}], ...}``) but the
    request uses ``input_type`` like Cohere.
    """

    _DIMS = {
        "voyage-3":        1024,
        "voyage-3-lite":   512,
        "voyage-large-2":  1536,
        "voyage-code-2":   1536,
    }

    def __init__(
        self,
        model: str = "voyage-3",
        api_key: Optional[str] = None,
        base_url: str = "https://api.voyageai.com/v1",
        dim: Optional[int] = None,
        timeout: float = 30.0,
    ):
        self._model    = model
        self._api_key  = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "VoyageEmbedder: VOYAGE_API_KEY is not set and no api_key= "
                "was provided. Sign up at https://www.voyageai.com/"
            )
        self._base_url = base_url.rstrip("/")
        self._timeout  = timeout
        if dim is not None:
            self.dim = dim
        elif model in self._DIMS:
            self.dim = self._DIMS[model]
        else:
            raise ValueError(
                f"VoyageEmbedder: unknown model {model!r} — pass dim= "
                f"explicitly (or use one of {sorted(self._DIMS)})."
            )

    def _post(self, payload: dict) -> dict:
        import json
        import urllib.request

        req = urllib.request.Request(
            f"{self._base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type":  "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def embed(self, text: str, *, kind: str = "passage") -> np.ndarray:
        return self.embed_batch([text], kind=kind)[0]

    def embed_batch(
        self, texts: Sequence[str], *, kind: str = "passage"
    ) -> np.ndarray:
        input_type = "query" if kind == "query" else "document"
        clipped = _truncate_for_remote(texts, max_chars=24_000, backend="voyage")
        body = {
            "model":      self._model,
            "input":      list(clipped),
            "input_type": input_type,
        }
        data = self._post(body)
        vecs = np.asarray(
            [d["embedding"] for d in data["data"]], dtype=np.float32
        )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        return np.ascontiguousarray(vecs / norms, dtype=np.float32)


# ── 3. Generic HTTP endpoint ──────────────────────────────────────────────────

class HTTPEmbedder:
    """Adapter for any HTTP endpoint that accepts ``{"input": [str]}`` and
    returns ``{"data": [{"embedding": [float, ...]}]}``.

    Works with vLLM, TEI (text-embeddings-inference), LiteLLM, etc.
    """

    def __init__(
        self,
        url: str,
        dim: int,
        headers: Optional[dict] = None,
        timeout: float = 30.0,
    ):
        self._url = url
        self.dim = dim
        self._headers = headers or {"Content-Type": "application/json"}
        self._timeout = timeout

    def embed(self, text: str, *, kind: str = "passage") -> np.ndarray:
        return self.embed_batch([text], kind=kind)[0]

    def embed_batch(
        self, texts: Sequence[str], *, kind: str = "passage"
    ) -> np.ndarray:
        import json
        import urllib.request

        clipped = _truncate_for_remote(texts, max_chars=24_000, backend="http")
        req = urllib.request.Request(
            self._url,
            data=json.dumps({"input": list(clipped)}).encode("utf-8"),
            headers=self._headers,
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        vecs = np.asarray(
            [d["embedding"] for d in data["data"]], dtype=np.float32
        )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        return np.ascontiguousarray(vecs / norms, dtype=np.float32)


# ── 4. Hash fallback (tests / smoke / no model) ───────────────────────────────

class HashEmbedder:
    """Deterministic token-hash → vector, for tests and smoke runs only.

    Bag-of-words SHA-1 hashing into ``dim`` buckets, L2-normalised.  No
    semantic understanding — but stable, dependency-free, and useful for
    exercising the storage / dedup / recall plumbing in CI.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> List[str]:
        # Cheap unicode-friendly tokenizer: split on whitespace, lowercase,
        # strip punctuation.  Good enough for hash buckets.
        import re
        return [t for t in re.split(r"\W+", text.lower()) if t]

    def embed(self, text: str, *, kind: str = "passage") -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            h = int.from_bytes(
                hashlib.sha1(tok.encode("utf-8")).digest()[:8], "big"
            )
            v[h % self.dim] += 1.0
            # Second hash for sign — gives some +/- variance.
            sign_h = int.from_bytes(
                hashlib.sha1((tok + "#").encode("utf-8")).digest()[:8], "big"
            )
            v[sign_h % self.dim] -= 0.5
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    def embed_batch(
        self, texts: Sequence[str], *, kind: str = "passage"
    ) -> np.ndarray:
        return np.stack([self.embed(t, kind=kind) for t in texts])


# ── Disk-backed cache wrapper ─────────────────────────────────────────────────

class CachedEmbedder:
    """Wrap any :class:`Embedder` with a persistent PistaDB
    :class:`EmbeddingCache` so each unique string is embedded only once.

    Cache key = ``f"{kind}\\x1f{text}"`` so query/passage prefixes don't
    collide.
    """

    def __init__(
        self,
        inner: Embedder,
        cache_path: Optional[str],
        max_entries: int = 100_000,
        autosave_every: int = 200,
    ):
        from pistadb import EmbeddingCache  # local import: avoid lib load if unused

        self._inner = inner
        self.dim = inner.dim
        self._cache = EmbeddingCache(
            cache_path, dim=inner.dim, max_entries=max_entries
        )
        self._autosave_every = autosave_every
        self._since_save = 0

    @staticmethod
    def _key(text: str, kind: str) -> str:
        # Length-prefixed so the (kind, text) pair always round-trips
        # uniquely — a delimiter-based scheme (e.g. ``f"{kind}\x1f{text}"``)
        # would let a maliciously-crafted text containing the delimiter
        # collide with a different (kind, text) pair.  ``\x1f`` is
        # vanishingly unlikely in real text, but the foot-gun is free
        # to remove.
        return f"{len(kind)}:{kind}:{text}"

    def embed(self, text: str, *, kind: str = "passage") -> np.ndarray:
        key = self._key(text, kind)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        v = np.ascontiguousarray(
            self._inner.embed(text, kind=kind), dtype=np.float32
        ).ravel()
        self._cache.put(key, v)
        self._bump_save()
        return v

    def embed_batch(
        self, texts: Sequence[str], *, kind: str = "passage"
    ) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        misses_idx: List[int] = []
        misses_txt: List[str] = []
        for i, t in enumerate(texts):
            hit = self._cache.get(self._key(t, kind))
            if hit is None:
                misses_idx.append(i)
                misses_txt.append(t)
            else:
                out[i] = hit
        if misses_txt:
            fresh = self._inner.embed_batch(misses_txt, kind=kind)
            for i, t, v in zip(misses_idx, misses_txt, fresh):
                out[i] = v
                self._cache.put(self._key(t, kind), v)
            # Defer the autosave check to once-per-batch instead of
            # per-miss — large batches were thrashing flush() on every
            # missed cache lookup.
            if self._autosave_every > 0:
                self._since_save += len(misses_txt)
                if self._since_save >= self._autosave_every:
                    self.flush()
                    self._since_save = 0
        return out

    def _bump_save(self) -> None:
        if self._autosave_every <= 0:
            return
        self._since_save += 1
        if self._since_save >= self._autosave_every:
            self.flush()
            self._since_save = 0

    def flush(self) -> None:
        self._cache.save()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._cache.close()


# ── Built-in factories: scheme → backend ────────────────────────────────────
#
# Each factory takes a :class:`ParsedSpec` and returns an :class:`Embedder`.
# Factories are tiny on purpose — pure plumbing from spec params to the
# backend's constructor — so adding a new backend is "implement the
# class + write a one-screen factory + register".

def _factory_local(spec: ParsedSpec) -> "Embedder":
    return LocalE5Embedder(
        model_path=spec.params.get("model_path"),
        device=spec.params.get("device", "cpu"),
    )


def _factory_openai(spec: ParsedSpec) -> "Embedder":
    model = spec.target or os.environ.get(
        "ENGRAM_OPENAI_MODEL", "text-embedding-3-small"
    )
    base_url = spec.params.get("base_url", "https://api.openai.com/v1")
    dim = int(spec.params["dim"]) if "dim" in spec.params else None
    timeout = float(spec.params.get("timeout", 30.0))
    return OpenAIEmbedder(model=model, base_url=base_url, dim=dim, timeout=timeout)


def _factory_ollama(spec: ParsedSpec) -> "Embedder":
    model = spec.target or "nomic-embed-text"
    host = spec.params.get("host", "http://localhost:11434")
    dim = int(spec.params["dim"]) if "dim" in spec.params else None
    timeout = float(spec.params.get("timeout", 30.0))
    return OllamaEmbedder(model=model, host=host, dim=dim, timeout=timeout)


def _factory_cohere(spec: ParsedSpec) -> "Embedder":
    model = spec.target or "embed-multilingual-v3.0"
    base_url = spec.params.get("base_url", "https://api.cohere.com/v2")
    dim = int(spec.params["dim"]) if "dim" in spec.params else None
    timeout = float(spec.params.get("timeout", 30.0))
    return CohereEmbedder(model=model, base_url=base_url, dim=dim, timeout=timeout)


def _factory_voyage(spec: ParsedSpec) -> "Embedder":
    model = spec.target or "voyage-3"
    base_url = spec.params.get("base_url", "https://api.voyageai.com/v1")
    dim = int(spec.params["dim"]) if "dim" in spec.params else None
    timeout = float(spec.params.get("timeout", 30.0))
    return VoyageEmbedder(model=model, base_url=base_url, dim=dim, timeout=timeout)


def _factory_http(spec: ParsedSpec) -> "Embedder":
    # Two entry points reach this factory:
    #   1) ``ENGRAM_EMBEDDER=http://host/path?dim=N`` — target is the URL.
    #   2) ``ENGRAM_EMBEDDER=http?url=http://host/path&dim=N`` — URL in params.
    # Plus back-compat with the legacy ``ENGRAM_HTTP_URL`` / ``ENGRAM_HTTP_DIM``
    # env vars: callers of ``build_default_embedder`` rewrite those into spec.
    url = spec.target or spec.params.get("url")
    if not url:
        raise RuntimeError(
            "HTTPEmbedder: no URL given. Use one of:\n"
            "  ENGRAM_EMBEDDER='http://host:port/embeddings?dim=768'\n"
            "  ENGRAM_EMBEDDER='http?url=http://host:port/embeddings&dim=768'\n"
            "  (legacy) ENGRAM_HTTP_URL=... ENGRAM_HTTP_DIM=..."
        )
    if "dim" not in spec.params:
        raise RuntimeError(
            "HTTPEmbedder: missing 'dim' — pass ?dim=N in the spec "
            "so Engram can size the .pst correctly."
        )
    dim = int(spec.params["dim"])
    timeout = float(spec.params.get("timeout", 30.0))
    return HTTPEmbedder(url=url, dim=dim, timeout=timeout)


def _factory_hash(spec: ParsedSpec) -> "Embedder":
    dim = int(
        spec.params.get("dim")
        or os.environ.get("ENGRAM_HASH_DIM", "384")
    )
    return HashEmbedder(dim=dim)


register_embedder("local",  _factory_local)
register_embedder("openai", _factory_openai)
register_embedder("ollama", _factory_ollama)
register_embedder("cohere", _factory_cohere)
register_embedder("voyage", _factory_voyage)
register_embedder("http",   _factory_http)
register_embedder("hash",   _factory_hash)


# ── Public factory: env-var-aware + cached ──────────────────────────────────

def build_default_embedder(cache_path: Optional[str] = None) -> CachedEmbedder:
    """Construct the embedder described by ``ENGRAM_EMBEDDER`` and wrap it
    in :class:`CachedEmbedder`.

    Spec resolution (in order):

    1. ``ENGRAM_EMBEDDER`` if non-empty — parsed via :func:`parse_spec`.
    2. ``OPENAI_API_KEY`` set with no explicit spec → **still** local.
       Remote backends are explicit opt-in (cost / privacy / network).
    3. Default → local e5.

    Back-compat with v0.3 env vars:

    * ``ENGRAM_EMBEDDER=openai`` + ``ENGRAM_OPENAI_MODEL=foo`` →
      spec ``openai:foo``.
    * ``ENGRAM_EMBEDDER=http`` + ``ENGRAM_HTTP_URL=U`` +
      ``ENGRAM_HTTP_DIM=N`` → spec ``U?dim=N``.

    See :func:`build_embedder` if you want a bare embedder without the
    cache wrapper.
    """
    raw = (os.environ.get("ENGRAM_EMBEDDER") or "").strip()

    # ── v0.3 env-var back-compat: rewrite bare scheme names into specs ──
    if raw.lower() == "openai" and "ENGRAM_OPENAI_MODEL" in os.environ:
        raw = f"openai:{os.environ['ENGRAM_OPENAI_MODEL']}"
    elif raw.lower() == "http" and "ENGRAM_HTTP_URL" in os.environ:
        url = os.environ["ENGRAM_HTTP_URL"]
        dim = os.environ.get("ENGRAM_HTTP_DIM")
        if not dim:
            raise RuntimeError(
                "ENGRAM_EMBEDDER=http requires ENGRAM_HTTP_DIM=N "
                "(the output dim of your HTTP endpoint). "
                "Or migrate to the URI form: "
                "ENGRAM_EMBEDDER='http://host:port/path?dim=N'"
            )
        sep = "&" if "?" in url else "?"
        raw = f"{url}{sep}dim={dim}"

    inner = build_embedder(raw)
    return CachedEmbedder(inner, cache_path=cache_path)


def warmup(spec: Optional[str] = None) -> Dict[str, Any]:
    """Pre-load the embedder so the first real call doesn't pay cold-start.

    Picks the spec from the ``spec`` arg, then ``ENGRAM_EMBEDDER``, then the
    default.  Runs three dummy embed calls so the model is in memory and
    the on-disk cache is initialised.

    Returns a small status dict (used by the CLI for human-friendly
    output) — the heavy lifting is the side effect.

    Typical use is right after ``engram install`` — see ``engram warmup``.
    """
    import time as _time
    if spec is None:
        spec = os.environ.get("ENGRAM_EMBEDDER", "")
    parsed = parse_spec(spec)

    t0 = _time.time()
    inner = build_embedder(spec)
    inner.embed_batch(
        ["engram warmup probe", "this is a second sample", "and a third"],
        kind="passage",
    )
    elapsed = _time.time() - t0
    return {
        "spec": spec,
        "scheme": parsed.scheme,
        "dim": inner.dim,
        "elapsed_seconds": round(elapsed, 2),
    }


__all__ = [
    # Protocol
    "Embedder",
    # URI / registry
    "ParsedSpec",
    "parse_spec",
    "register_embedder",
    "registered_schemes",
    "build_embedder",
    "build_default_embedder",
    "warmup",
    # Backends
    "LocalE5Embedder",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "CohereEmbedder",
    "VoyageEmbedder",
    "HTTPEmbedder",
    "HashEmbedder",
    # Cache wrapper
    "CachedEmbedder",
]
