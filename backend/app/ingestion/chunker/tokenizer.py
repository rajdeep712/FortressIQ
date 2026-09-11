"""Token estimation for the standardized (non-tabular) chunker sizing.

Primary estimator is tiktoken's ``cl100k_base`` BPE. The encoder is
lazily resolved once (and cached); if tiktoken is missing or the BPE
vocabulary cannot be loaded (e.g. an offline first run) the estimator
falls back to the standard ~4 chars/token heuristic so chunking never
blocks on a network call.
"""

import threading

FALLBACK_CHARS_PER_TOKEN = 4

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency guard
    tiktoken = None

_lock = threading.Lock()
_ENCODER = None
_ENCODER_TRIED = False


def _get_encoder():
    global _ENCODER, _ENCODER_TRIED

    if _ENCODER_TRIED:
        return _ENCODER

    with _lock:
        if _ENCODER_TRIED:
            return _ENCODER
        _ENCODER_TRIED = True
        if tiktoken is None:
            return None
        try:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None

    return _ENCODER


def estimate_tokens(text: str) -> int:
    """Token count of ``text`` (>= 1 so a renderable empty window is 0 tokens
    while any real content is measured)."""
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    return max(1, len(text) // FALLBACK_CHARS_PER_TOKEN)