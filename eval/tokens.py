"""Token counting — uses tiktoken if installed, else a documented heuristic.

The heuristic isn't perfect but it is:
* deterministic and dependency-free,
* roughly calibrated against GPT-4's BPE for mixed English/CJK text,
* fair — both systems are measured with the same function.

Calibration:

* English/Latin/digit/punctuation: ~4 chars per token (matches GPT-4 BPE
  on prose).
* CJK characters: ~1.5 chars per token (most CJK glyphs tokenise to one
  BPE token; some compounds to half a token).
"""
from __future__ import annotations


def _is_cjk(c: str) -> bool:
    o = ord(c)
    return (
        0x4E00 <= o <= 0x9FFF       # CJK Unified Ideographs
        or 0x3400 <= o <= 0x4DBF    # CJK Unified Ideographs Extension A
        or 0x3000 <= o <= 0x303F    # CJK Symbols & Punctuation
        or 0xFF00 <= o <= 0xFFEF    # Halfwidth/Fullwidth
    )


def _heuristic_count(text: str) -> int:
    cjk = sum(1 for c in text if _is_cjk(c))
    non_cjk_chars = len(text) - cjk
    return max(1, int(round(cjk / 1.5 + non_cjk_chars / 4))) if text else 0


def count_tokens(text: str) -> int:
    """Return an estimate of the GPT-4 BPE token count.

    Uses ``tiktoken`` if available (accurate); falls back to a heuristic
    otherwise (good enough for relative comparison).
    """
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return _heuristic_count(text)


def has_tiktoken() -> bool:
    try:
        import tiktoken  # noqa: F401
        return True
    except Exception:
        return False
