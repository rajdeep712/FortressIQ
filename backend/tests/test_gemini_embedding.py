"""Tests for Gemini embedding service and EmbeddingService facade router."""

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.ingestion.embedding import (
    EMBEDDER_SENTENCE_TRANSFORMER,
    EmbeddingService,
)
from app.services.gemini_embedding import (
    EMBEDDER_GEMINI,
    GEMINI_INPUT_TOKEN_LIMIT,
    GeminiEmbeddingService,
    GeminiRateLimiter,
    GeminiUnavailableError,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FakeGeminiResponse:
    """Mimics the google.genai types.EmbedContentResponse."""

    def __init__(self, vectors: list[list[float]]):
        self.embeddings = [
            SimpleNamespace(values=v) for v in vectors
        ]


def _make_vectors(n: int, dim: int = 3) -> list[list[float]]:
    return [[float(i)] * dim for i in range(n)]


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:

    def test_empty_string_returns_one(self):
        assert estimate_tokens("") == 1

    def test_conservative_division_by_three(self):
        assert estimate_tokens("abc") == 1
        assert estimate_tokens("abcd") == 2  # ceil(4/3) = 2
        assert estimate_tokens("abcdefgh") == 3  # ceil(8/3) = 3


# ---------------------------------------------------------------------------
# GeminiRateLimiter – acquire
# ---------------------------------------------------------------------------

class TestGeminiRateLimiter:

    def _make_limiter(self, **kwargs):
        defaults = dict(
            rpm=3, tpm=100, rpd=10000,
            state_file="",
            clock=lambda: 0.0,
            sleep=lambda _: None,
        )
        defaults.update(kwargs)
        return GeminiRateLimiter(**defaults)

    def test_acquire_reserves_tokens(self):
        limiter = self._make_limiter()
        reserved = limiter.acquire(["hello", "world"])
        assert reserved > 0
        # Two requests recorded (one acquire call)
        assert len(limiter._requests) == 1
        assert limiter._requests[0][1] == reserved

    def test_acquire_blocks_when_rpm_exhausted(self):
        sleep_log: list[float] = []
        clock_val = [100.0]
        limiter = self._make_limiter(
            rpm=2, tpm=10000, rpd=10000,
            clock=lambda: clock_val[0],
            sleep=lambda d: (sleep_log.append(d), clock_val.__setitem__(0, clock_val[0] + d)),
        )
        limiter.acquire(["a"])
        limiter.acquire(["b"])
        assert len(limiter._requests) == 2

        limiter.acquire(["c"])
        assert len(sleep_log) == 1
        assert sleep_log[0] > 0.0

    def test_acquire_blocks_when_tpm_exhausted(self):
        sleep_log: list[float] = []
        clock_val = [0.0]
        limiter = self._make_limiter(
            rpm=100, tpm=10, rpd=10000,
            clock=lambda: clock_val[0],
            sleep=lambda d: (sleep_log.append(d), clock_val.__setitem__(0, clock_val[0] + d)),
        )
        limiter.acquire(["a" * 30])  # ~10 tokens
        assert len(sleep_log) == 0

        limiter.acquire(["b" * 30])  # should block on TPM
        assert len(sleep_log) == 1

    def test_acquire_blocks_when_rpd_exhausted(self):
        sleep_log: list[float] = []
        clock_val = [0.0]
        limiter = self._make_limiter(
            rpm=100, tpm=10000, rpd=20,
            clock=lambda: clock_val[0],
            sleep=lambda d: (sleep_log.append(d), clock_val.__setitem__(0, clock_val[0] + d)),
        )
        limiter.acquire(["x" * 60])  # ~20 tokens, fills RPD
        assert len(sleep_log) == 0

        limiter.acquire(["y" * 60])  # should block on RPD
        assert len(sleep_log) == 1

    def test_old_entries_expire(self):
        clock_val = [0.0]
        limiter = self._make_limiter(
            rpm=2, tpm=10000, rpd=10000,
            clock=lambda: clock_val[0],
        )
        limiter.acquire(["a"])
        limiter.acquire(["b"])
        # Move past 60s — old entries expire, RPM is free again
        clock_val[0] = 61.0
        sleep_log: list[float] = []
        limiter._sleep = lambda d: sleep_log.append(d)
        limiter.acquire(["c"])
        assert len(sleep_log) == 0


# ---------------------------------------------------------------------------
# GeminiRateLimiter – persistence
# ---------------------------------------------------------------------------

class TestGeminiRateLimiterPersistence:

    def test_save_and_reload(self, tmp_path: Path):
        path = str(tmp_path / "usage.json")
        limiter1 = GeminiRateLimiter(
            rpm=100, tpm=30000, rpd=1000,
            state_file=path,
            sleep=lambda _: None,
        )
        limiter1.acquire(["hello world"])
        assert Path(path).exists()

        limiter2 = GeminiRateLimiter(
            rpm=100, tpm=30000, rpd=1000,
            state_file=path,
            sleep=lambda _: None,
        )
        assert len(limiter2._requests) == 1

    def test_expired_entries_not_loaded(self, tmp_path: Path):
        path = tmp_path / "usage.json"
        path.write_text(
            json.dumps([[time.time() - 200000, 10]]),  # way past 24h
            encoding="utf-8",
        )
        limiter = GeminiRateLimiter(
            state_file=str(path),
            sleep=lambda _: None,
        )
        assert len(limiter._requests) == 0

    def test_corrupt_file_does_not_crash(self, tmp_path: Path):
        path = tmp_path / "usage.json"
        path.write_text("NOT JSON!!!", encoding="utf-8")
        limiter = GeminiRateLimiter(
            state_file=str(path),
            sleep=lambda _: None,
        )
        assert limiter._requests == []


# ---------------------------------------------------------------------------
# GeminiEmbeddingService
# ---------------------------------------------------------------------------

class FakeGeminiModels:
    def __init__(self, vectors: list[list[float]] | None = None, error=None):
        self._vectors = vectors or _make_vectors(2)
        self._error = error
        self.embed_calls: list[dict[str, Any]] = []

    def embed_content(self, *, model, contents, config=None):
        self.embed_calls.append(dict(model=model, contents=contents, config=config))
        if self._error is not None:
            raise self._error
        return FakeGeminiResponse(self._vectors[:len(contents)])


class FakeGeminiClient:
    """Simulates the google.genai Client for testing."""

    def __init__(self, vectors: list[list[float]] | None = None, error=None):
        self.models = FakeGeminiModels(vectors=vectors, error=error)


class _ThrottleError(Exception):
    """Simulates a 429 / rate-limit error from the Gemini API."""

    def __init__(self, message: str = "429"):
        super().__init__(message)
        self.code = 429
        self.response = SimpleNamespace(
            headers={"retry-after": "1"}
        )


class TestGeminiEmbeddingService:

    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        """Use small dimension (3) so tests don't need 768-dim vectors."""
        with patch("app.services.gemini_embedding.settings") as mock:
            mock.gemini_embedding_model = "gemini-embedding-2"
            mock.gemini_api_key = "test-key"
            mock.gemini_output_dimensionality = 3
            mock.gemini_embedding_batch_size = 2
            mock.gemini_rate_rpm = 100
            mock.gemini_rate_tpm = 30000
            mock.gemini_rate_rpd = 1000
            mock.gemini_rate_state_file = ""
            mock.gemini_retry_base_seconds = 60
            mock.gemini_retry_max_seconds = 600
            mock.gemini_retry_max_attempts = 4
            mock.gemini_fallback_cooldown_seconds = 300
            mock.embedding_dimension = 3
            yield mock

    def _make_service(self, **kwargs):
        defaults = dict(
            model_name="gemini-embedding-2",
            api_key="test-key",
            output_dimensionality=3,
            batch_size=2,
        )
        defaults.update(kwargs)
        return GeminiEmbeddingService(**defaults)

    def test_no_api_key_raises(self):
        with pytest.raises(GeminiUnavailableError, match="not configured"):
            GeminiEmbeddingService(api_key="", output_dimensionality=3)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            GeminiEmbeddingService(
                api_key="k", output_dimensionality=99,
            )

    def test_embed_batches_and_returns_vectors(self):
        fake_client = FakeGeminiClient(vectors=_make_vectors(4, 3))
        svc = self._make_service(batch_size=2, client=fake_client)

        result = svc.embed(["a", "b", "c", "d"])

        assert len(result) == 4
        assert all(len(v) == 3 for v in result)
        # Two batches of 2
        assert len(fake_client.models.embed_calls) == 2
        assert fake_client.models.embed_calls[0]["contents"] == ["a", "b"]
        assert fake_client.models.embed_calls[1]["contents"] == ["c", "d"]

    def test_embed_empty_returns_empty(self):
        svc = self._make_service(client=FakeGeminiClient())
        assert svc.embed([]) == []

    def test_embed_count_mismatch_raises(self):
        # Client returns wrong number of vectors
        fake_client = FakeGeminiClient(vectors=[[0.0, 0.0, 0.0]])
        svc = self._make_service(batch_size=2, client=fake_client)

        with pytest.raises(GeminiUnavailableError, match="returned 1"):
            svc.embed(["a", "b"])

    def test_429_retries_then_raises(self):
        fake_client = FakeGeminiClient(error=_ThrottleError("429"))
        svc = self._make_service(
            client=fake_client,
            sleep=lambda _: None,
        )

        with pytest.raises(GeminiUnavailableError, match="rate limit exceeded"):
            svc.embed(["a", "b"])

    def test_non_throttle_error_raises_immediately(self):
        fake_client = FakeGeminiClient(error=ValueError("bad request"))
        svc = self._make_service(client=fake_client)

        with pytest.raises(GeminiUnavailableError, match="bad request"):
            svc.embed(["a"])

    def test_provider_label(self):
        assert self._make_service().provider == EMBEDDER_GEMINI

    def test_dimension_property(self):
        svc = self._make_service()
        assert svc.dimension == 3

    def test_rate_limiter_called_with_batch(self):
        fake_client = FakeGeminiClient(vectors=_make_vectors(2, 3))
        limiter = GeminiRateLimiter(
            rpm=100, tpm=30000, rpd=1000,
            sleep=lambda _: None,
        )
        svc = self._make_service(client=fake_client, limiter=limiter)

        svc.embed(["hello", "world"])
        # limiter recorded one request
        assert len(limiter._requests) == 1


# ---------------------------------------------------------------------------
# EmbeddingService facade – Gemini → ST fallback
# ---------------------------------------------------------------------------

class TestEmbeddingServiceFacade:

    @pytest.fixture()
    def fake_st_service(self):
        """A fake SentenceTransformerEmbeddingService-like object."""
        class _FakeST:
            provider = EMBEDDER_SENTENCE_TRANSFORMER
            dimension = settings.embedding_dimension

            def embed(self, texts):
                return [[0.1] * self.dimension for _ in texts]
        return _FakeST()

    def test_no_gemini_key_uses_st(self, fake_st_service):
        with patch("app.ingestion.embedding.settings") as mock_settings:
            mock_settings.embedding_provider = "gemini"
            mock_settings.gemini_api_key = ""
            mock_settings.embedding_dimension = settings.embedding_dimension
            svc = EmbeddingService(service=fake_st_service)
            assert svc.last_provider == EMBEDDER_SENTENCE_TRANSFORMER

    def test_gemini_unavailable_at_startup_uses_st(self, fake_st_service):
        with (
            patch("app.ingestion.embedding.settings") as mock_settings,
            patch(
                "app.ingestion.embedding.GeminiEmbeddingService",
                side_effect=GeminiUnavailableError("no key"),
            ),
        ):
            mock_settings.embedding_provider = "gemini"
            mock_settings.gemini_api_key = "fake-key"
            mock_settings.embedding_dimension = settings.embedding_dimension
            svc = EmbeddingService(service=fake_st_service)
            assert svc.last_provider == EMBEDDER_SENTENCE_TRANSFORMER

    def test_injected_service_provider(self, fake_st_service):
        svc = EmbeddingService(service=fake_st_service)
        result = svc.embed(["hello"])
        assert svc.last_provider == EMBEDDER_SENTENCE_TRANSFORMER
        assert len(result) == 1

    def test_gemini_fallback_to_st_on_error(self, fake_st_service):
        call_count = [0]

        def _fail_embed(texts):
            call_count[0] += 1
            raise GeminiUnavailableError("429")

        fake_gemini = SimpleNamespace(
            embed=_fail_embed,
            provider=EMBEDDER_GEMINI,
            dimension=settings.embedding_dimension,
        )

        with patch("app.ingestion.embedding.settings") as mock_settings:
            mock_settings.gemini_fallback_cooldown_seconds = 300
            mock_settings.embedding_dimension = settings.embedding_dimension

            svc = EmbeddingService(service=fake_st_service)
            svc._gemini = fake_gemini

            result = svc.embed(["test text"])

            assert len(result) == 1
            assert svc.last_provider == EMBEDDER_SENTENCE_TRANSFORMER
            assert svc._fallback_until > time.time()

    def test_cooldown_suppresses_gemini(self, fake_st_service):
        fake_gemini = SimpleNamespace(
            embed=lambda texts: [[0.0] * settings.embedding_dimension for _ in texts],
            provider=EMBEDDER_GEMINI,
            dimension=settings.embedding_dimension,
        )

        with patch("app.ingestion.embedding.settings") as mock_settings:
            mock_settings.gemini_fallback_cooldown_seconds = 300
            mock_settings.embedding_dimension = settings.embedding_dimension

            svc = EmbeddingService(service=fake_st_service)
            svc._gemini = fake_gemini
            # Simulate active cooldown
            svc._fallback_until = time.time() + 300

            result = svc.embed(["test"])

            # Should use ST, not Gemini
            assert svc.last_provider == EMBEDDER_SENTENCE_TRANSFORMER
            assert len(result) == 1

    def test_empty_embed_returns_empty_and_clears_provider(self, fake_st_service):
        svc = EmbeddingService(service=fake_st_service)
        svc.embed(["a"])
        assert svc.last_provider == EMBEDDER_SENTENCE_TRANSFORMER

        result = svc.embed([])
        assert result == []
        assert svc.last_provider is None
