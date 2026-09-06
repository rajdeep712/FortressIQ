"""Dense embeddings via OpenRouter's OpenAI-compatible /embeddings endpoint.

Primary (and default) engine for document + query embeddings. The default
model, ``nvidia/llama-nemotron-embed-vl-1b-v2:free``, returns 2048-dim vectors
and is served free-of-charge behind Cloudflare, so requests are sent with
browser-like headers (UA, HTTP-Referer, X-Title) and a Bearer key -- the exact
flow proven in the original standalone test script.

The free tier is rate-limited: batch size defaults to 1 and a short sleep runs
between batches; 429 responses are retried respecting ``Retry-After``.

Unlike the legacy Gemini/SentenceTransformer paths this service does NOT
fall back silently -- failures raise ``OpenRouterUnavailableError`` so
ingestion fails loudly instead of persisting nothing.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

EMBEDDER_OPENROUTER = "openrouter"


class OpenRouterUnavailableError(Exception):
    """Raised when OpenRouter cannot produce embeddings (HTTP/timeout/etc.).

    Intentionally not swallowed by callers: a configured OpenRouter provider
    that fails should fail the operation loudly.
    """


class OpenRouterEmbeddingService:
    """Embed texts via POST /embeddings on the OpenRouter API."""

    provider = EMBEDDER_OPENROUTER

    def __init__(
        self,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int | None = None,
        sleep_seconds: float | None = None,
        timeout_seconds: float | None = None,
        retry_max_attempts: int | None = None,
        client: httpx.Client | None = None,
        dimension: int | None = None,
        sleep: Callable[[float], None] | None = None,
    ):
        self.model_name = (
            settings.openrouter_embedding_model
            if model_name is None
            else model_name
        )
        self.api_key = (
            settings.openrouter_api_key if api_key is None else api_key
        )
        self.base_url = (
            settings.openrouter_base_url if base_url is None else base_url
        ).rstrip("/")
        self.batch_size = (
            settings.openrouter_batch_size
            if batch_size is None
            else batch_size
        )
        self.sleep_seconds = (
            settings.openrouter_sleep_seconds
            if sleep_seconds is None
            else sleep_seconds
        )
        self.timeout_seconds = (
            settings.openrouter_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self.retry_max_attempts = (
            settings.openrouter_retry_max_attempts
            if retry_max_attempts is None
            else retry_max_attempts
        )
        self._dimension = (
            settings.embedding_dimension if dimension is None else dimension
        )
        self._client = client or httpx.Client(timeout=self.timeout_seconds)
        self._sleep = sleep or time.sleep

        if not self.api_key:
            raise OpenRouterUnavailableError(
                "OpenRouter API key is not configured (OPENROUTER_API_KEY)"
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/embeddings"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": settings.openrouter_http_referer,
            "X-Title": settings.openrouter_title,
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
        }

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        try:
            value = response.headers.get("retry-after")
            if value is None:
                return None
            return max(0.0, float(value))
        except Exception:  # noqa: BLE001 - unparseable header is fine
            return None

    def _call_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self.model_name, "input": batch}
        last_error: Exception | None = None

        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                response = self._client.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=payload,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < self.retry_max_attempts:
                    self._sleep(min(2 ** (attempt - 1), 16))
                    continue
                raise OpenRouterUnavailableError(
                    "OpenRouter embeddings request failed "
                    f"after {self.retry_max_attempts} attempts: {exc}"
                ) from exc

            retry_after = self._retry_after(response)
            if response.status_code == 200:
                vectors = self._parse(response)
                return vectors

            if response.status_code == 429 and attempt < self.retry_max_attempts:
                self._sleep(retry_after or 2 ** (attempt - 1))
                continue

            raise OpenRouterUnavailableError(
                "OpenRouter embeddings returned HTTP "
                f"{response.status_code}: {response.text[:200]}"
            )

        raise OpenRouterUnavailableError(
            f"OpenRouter embeddings failed after "
            f"{self.retry_max_attempts} attempts: {last_error}"
        )

    def _parse(self, response: httpx.Response) -> list[list[float]]:
        try:
            payload = response.json()
            items = payload.get("data") or []
            embeddings = [item.get("embedding") for item in items]
        except Exception as exc:  # noqa: BLE001 - unexpected/malformed body
            raise OpenRouterUnavailableError(
                f"Malformed OpenRouter embeddings response: {exc}"
            ) from exc

        if any(not item or len(item) != self._dimension for item in embeddings):
            lens = {
                len(item) if item is not None else 0 for item in embeddings
            }
            raise OpenRouterUnavailableError(
                "OpenRouter returned embeddings with dimension "
                f"{sorted(lens)}; expected {self._dimension}. Update "
                "EMBEDDING_DIMENSION to match the model output."
            )

        return [[float(v) for v in emb] for emb in embeddings]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts; raises ``OpenRouterUnavailableError`` on any failure.

        Despite the API accepting batched inputs, the free tier is
        rate-limited, so we send one text per request with a short sleep
        between calls (both overridable via settings / constructor args).
        """
        if not texts:
            return []

        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors = self._call_batch(batch)
            if len(vectors) != len(batch):
                raise OpenRouterUnavailableError(
                    "OpenRouter returned "
                    f"{len(vectors)} embeddings for {len(batch)} texts"
                )
            out.extend(vectors)
            if start + self.batch_size < len(texts):
                self._sleep(self.sleep_seconds)
        return out