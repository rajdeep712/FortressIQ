"""Dense embeddings via Gemini Embedding 2 with strict free-tier rate limits.

Primary engine for document + query embeddings. The service:

1. Throttles requests so the free-tier caps (RPM / TPM / RPD) are never
   surpassed -- ``GeminiRateLimiter.acquire`` sleeps until the current
   minute/rolling-day budgets can fit a batch before the HTTP call is made.
2. Backs off exponentially on 429 (rate limit): wait 60s (or Retry-After),
   double up to ``gemini_retry_max_seconds``, give up after
   ``gemini_retry_max_attempts`` consecutive throttles.
3. Surfaces a ``GeminiUnavailableError`` so the ``EmbeddingService`` facade
   can transparently fall back to sentence-transformers.

The ``google-genai`` SDK is imported lazily so importing this module never
fails when the SDK (or the API key / network) is unavailable.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from app.core.config import settings

logger = logging.getLogger(__name__)

EMBEDDER_GEMINI = "gemini-embedding-2"

# Google's embed model rejects inputs longer than this many tokens; anything
# beyond it must be handled by the caller (or the fallback provider).
GEMINI_INPUT_TOKEN_LIMIT = 8192


class GeminiUnavailableError(Exception):
    """Raised when Gemini cannot produce embeddings for a batch of texts.

    Callers (the ``EmbeddingService`` facade) treat this as "switch to the
    fallback provider for this call".
    """


def estimate_tokens(text: str) -> int:
    """Conservative token count for rate-limit accounting.

    English is roughly 4 chars/token; using ~3 chars/token over-counts and
    leaves a safety buffer so the TPM cap is not accidentally breached.
    """
    return max(1, math.ceil(len(text) / 3))


class GeminiRateLimiter:
    """Sliding-window rate limiter enforcing free-tier RPM/TPM/RPD caps.

    Windows
    -------
    RPM:  number of requests in the trailing 60s.
    TPM:  sum of estimated tokens in the trailing 60s.
    RPD:  sum of estimated tokens in the trailing 24h, persisted to a JSON
          file so a worker restart does not reset the daily budget.

    ``acquire`` blocks until all three budgets can accommodate ``texts``,
    then records a request + token reservation. Thread-safe.
    """

    def __init__(
        self,
        *,
        rpm: int | None = None,
        tpm: int | None = None,
        rpd: int | None = None,
        state_file: str | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ):
        self.rpm = rpm if rpm is not None else settings.gemini_rate_rpm
        self.tpm = tpm if tpm is not None else settings.gemini_rate_tpm
        self.rpd = rpd if rpd is not None else settings.gemini_rate_rpd
        self.state_file = (
            state_file if state_file is not None else settings.gemini_rate_state_file
        )
        self._clock = clock or time.time
        self._sleep = sleep or time.sleep

        self._lock = threading.Lock()
        # (timestamp, tokens) per request, kept for the trailing windows.
        self._requests: list[tuple[float, int]] = []

        self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        path = self.state_file
        if not path:
            return
        p = Path(path)
        if not p.exists():
            return
        try:
            with p.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            now = self._clock()
            self._requests = [
                (float(ts), int(tokens))
                for ts, tokens in raw
                if now - float(ts) < 86400 and int(tokens) > 0
            ]
            self._requests.sort(key=lambda item: item[0])
        except Exception:  # noqa: BLE001 - corrupt/missing state must not crash
            logger.warning(
                "Could not load Gemini rate-limit state from %s",
                path,
            )

    def _save(self) -> None:
        path = self.state_file
        if not path:
            return
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(f"{path}.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self._requests, fh)
            tmp.replace(p)
        except Exception:  # noqa: BLE001 - rate-limit state is best-effort
            logger.warning(
                "Could not persist Gemini rate-limit state to %s",
                path,
            )

    # -- window helpers ------------------------------------------------------

    def _in_window(self, now: float, seconds: float) -> list[tuple[float, int]]:
        cutoff = now - seconds
        return [(ts, t) for ts, t in self._requests if ts > cutoff]

    def _release_rpm(self, now: float) -> float:
        recent = self._in_window(now, 60.0)
        if len(recent) < self.rpm:
            return 0.0
        oldest = min(ts for ts, _ in recent)
        return oldest + 60.0

    def _release_for_tokens(
        self,
        now: float,
        batch_tokens: int,
        seconds: float,
        budget: int,
    ) -> float:
        recent = self._in_window(now, seconds)
        total = sum(t for _, t in recent)
        if total + batch_tokens <= budget:
            return 0.0
        # Drop from the oldest until the batch fits; the slot frees when the
        # oldest dropped entry leaves its window.
        needed = total + batch_tokens - budget
        acc = 0
        release = now
        for ts, tokens in recent:  # recent is sorted by ts
            acc += tokens
            release = ts + seconds
            if acc >= needed:
                break
        return release

    def _release_tpm(self, now: float, batch_tokens: int) -> float:
        return self._release_for_tokens(now, batch_tokens, 60.0, self.tpm)

    def _release_rpd(self, now: float, batch_tokens: int) -> float:
        return self._release_for_tokens(now, batch_tokens, 86400.0, self.rpd)

    def _next_allowed(self, now: float, batch_tokens: int) -> float:
        return max(
            self._release_rpm(now),
            self._release_tpm(now, batch_tokens),
            self._release_rpd(now, batch_tokens),
        )

    def _record(self, now: float, tokens: int) -> None:
        self._requests.append((now, tokens))
        self._requests = [
            (ts, t) for ts, t in self._requests if now - ts < 86400
        ]
        self._save()

    # -- public API ----------------------------------------------------------

    def acquire(self, texts: Sequence[str]) -> int:
        """Block until the batch fits every cap, then reserve its tokens.

        Returns the number of tokens reserved for the batch.
        """
        tokens = sum(estimate_tokens(text) for text in texts)
        with self._lock:
            while True:
                now = self._clock()
                release = self._next_allowed(now, tokens)
                if release <= now:
                    break
                delay = release - now
                self._sleep(delay)
            self._record(now, tokens)
        return tokens


class GeminiEmbeddingService:
    """Embed texts with ``gemini-embedding-2`` under strict rate limits."""

    provider = EMBEDDER_GEMINI

    def __init__(
        self,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        output_dimensionality: int | None = None,
        batch_size: int | None = None,
        limiter: GeminiRateLimiter | None = None,
        client=None,
        sleep: Callable[[float], None] | None = None,
    ):
        self.model_name = (
            settings.gemini_embedding_model
            if model_name is None
            else model_name
        )
        self.api_key = (
            settings.gemini_api_key
            if api_key is None
            else api_key
        )
        self.output_dimensionality = (
            settings.gemini_output_dimensionality
            if output_dimensionality is None
            else output_dimensionality
        )
        self.batch_size = (
            settings.gemini_embedding_batch_size
            if batch_size is None
            else batch_size
        )
        self._limiter = limiter or GeminiRateLimiter()
        self._client = client
        self._sleep = sleep or time.sleep

        if not self.api_key:
            raise GeminiUnavailableError(
                "Gemini API key is not configured"
            )

        if self.output_dimensionality != settings.embedding_dimension:
            raise ValueError(
                "gemini_output_dimensionality "
                f"({self.output_dimensionality}) does not match "
                "EMBEDDING_DIMENSION "
                f"({settings.embedding_dimension})."
            )

        self._dimension = self.output_dimensionality

    @property
    def dimension(self) -> int:
        return self._dimension

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _is_throttle(exc: BaseException) -> bool:
        code = getattr(exc, "code", None)
        if code in (429, 403):
            return True
        message = str(exc).lower()
        return (
            "429" in message
            or "resource_exhausted" in message
            or "rate limit" in message
        )

    @staticmethod
    def _retry_after(exc: BaseException) -> float | None:
        try:
            response = getattr(exc, "response", None)
            if response is None:
                return None
            headers = getattr(response, "headers", None) or {}
            value = headers.get("retry-after") or headers.get("Retry-After")
            if value is None:
                return None
            return max(0.0, float(value))
        except Exception:  # noqa: BLE001
            return None

    def _wait(
        self,
        exc: BaseException,
        attempt: int,
        *,
        throttled: bool,
    ) -> float:
        if throttled:
            # Prefer the server-provided Retry-After, else exponential:
            # 60s, 120s, 240s, ... capped at the configured max.
            retry_after = self._retry_after(exc)
            base = settings.gemini_retry_base_seconds
            delay = (
                retry_after
                if retry_after is not None
                else base * (2 ** (attempt - 1))
            )
            delay = min(delay, settings.gemini_retry_max_seconds)
        else:
            delay = min(2 ** attempt, 16)
        self._sleep(delay)
        return delay

    def _call(self, batch: list[str]) -> list[list[float]]:
        """Single (already rate-limited) embed call with retry/backoff."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except Exception as exc:  # noqa: BLE001
                raise GeminiUnavailableError(
                    f"Could not create Gemini client: {exc}"
                ) from exc

        try:
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - env setup
            raise GeminiUnavailableError(
                "google-genai is not installed"
            ) from exc

        max_attempts = settings.gemini_retry_max_attempts
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.output_dimensionality,
                    ),
                )
                return self._parse(response)
            except Exception as exc:  # noqa: BLE001 - API layer
                throttled = self._is_throttle(exc)
                if throttled and attempt == max_attempts:
                    raise GeminiUnavailableError(
                        "Gemini rate limit exceeded after "
                        f"{max_attempts} attempts"
                    ) from exc
                if throttled or attempt < max_attempts:
                    self._wait(
                        exc,
                        attempt,
                        throttled=throttled,
                    )
                else:
                    raise GeminiUnavailableError(
                        f"Gemini embedding request failed: {exc}"
                    ) from exc

        raise GeminiUnavailableError("Gemini embedding request failed")

    def _parse(self, response) -> list[list[float]]:
        embeddings = getattr(response, "embeddings", None) or []
        return [
            [float(v) for v in embedding.values]
            for embedding in embeddings
        ]

    # -- main entry point ----------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        for text in texts:
            if estimate_tokens(text) > GEMINI_INPUT_TOKEN_LIMIT:
                raise GeminiUnavailableError(
                    "Input text exceeds gemini-embedding-2 token limit "
                    f"({GEMINI_INPUT_TOKEN_LIMIT} tokens)"
                )

        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            self._limiter.acquire(batch)
            vectors = self._call(batch)
            if len(vectors) != len(batch):
                raise GeminiUnavailableError(
                    "Gemini returned "
                    f"{len(vectors)} embeddings for {len(batch)} texts"
                )
            out.extend(vectors)
        return out