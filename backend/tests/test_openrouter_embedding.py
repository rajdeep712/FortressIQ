"""Tests for the OpenRouter embedding service and facade integration."""

import json
from types import SimpleNamespace

import httpx
import pytest

from app.core.config import settings
from app.ingestion.embedding import EmbeddingService
from app.services.openrouter_embedding import (
    EMBEDDER_OPENROUTER,
    OpenRouterEmbeddingService,
    OpenRouterUnavailableError,
)


def _make_vectors(n: int, dim: int) -> list[list[float]]:
    return [[float(i) + 0.5] * dim for i in range(n)]


def _service(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    defaults = dict(
        api_key="test-key",
        model_name="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        base_url="https://openrouter.ai/api/v1",
        batch_size=1,
        sleep_seconds=0.0,
        timeout_seconds=10,
        retry_max_attempts=2,
        dimension=2048,
        client=client,
    )
    defaults.update(kwargs)
    return OpenRouterEmbeddingService(**defaults)


class TestOpenRouterEmbeddingService:

    def test_single_batch_sends_expected_request(self):
        expected = _make_vectors(1, 2048)

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert request.url == (
                "https://openrouter.ai/api/v1/embeddings"
            )
            headers = request.headers
            assert headers["authorization"] == "Bearer test-key"
            assert headers["accept"] == "application/json"
            assert headers["content-type"] == "application/json"
            assert headers["http-referer"] == "https://openrouter.ai"
            assert headers["x-title"] == "RAG Embeddings"
            assert "Mozilla/5.0" in headers["user-agent"]

            body = json.loads(request.content)
            assert body["model"] == "nvidia/llama-nemotron-embed-vl-1b-v2:free"
            assert body["input"] == ["hello world"]

            return httpx.Response(
                200,
                json={"data": [{"embedding": expected[0]}]},
            )

        svc = _service(handler)
        assert svc.provider == EMBEDDER_OPENROUTER
        assert svc.dimension == 2048
        assert svc.model_name == "nvidia/llama-nemotron-embed-vl-1b-v2:free"

        vectors = svc.embed(["hello world"])

        assert vectors == expected

    def test_batches_and_sleeps_between_calls(self):
        calls = {"count": 0, "sleeps": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            inputs = json.loads(request.content)["input"]
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"embedding": [1.0] * 2048} for _ in inputs
                    ]
                },
            )

        def fake_sleep(_):
            calls["sleeps"] += 1

        svc = _service(handler, batch_size=2, sleep=fake_sleep)
        vectors = svc.embed(["a", "b", "c"])

        assert len(vectors) == 3
        assert calls["count"] == 2
        assert calls["sleeps"] == 1

    def test_429_retried_then_succeeds(self):
        sequence = [
            httpx.Response(429, headers={"Retry-After": "0.001"}),
            httpx.Response(200, json={"data": [{"embedding": [0.0] * 2048}]}),
        ]
        calls = {"count": 0, "sleeps": []}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return sequence.pop(0)

        def fake_sleep(delay):
            calls["sleeps"].append(delay)

        svc = _service(handler, sleep=fake_sleep)

        vectors = svc.embed(["x"])

        assert len(vectors) == 1
        assert calls["count"] == 2
        assert calls["sleeps"][0] == 0.001  # used Retry-After

    def test_http_error_raises_openrouter_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        svc = _service(handler)

        with pytest.raises(OpenRouterUnavailableError):
            svc.embed(["x"])

    def test_network_error_raises_after_attempts(self):
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            raise httpx.ConnectError("no network")

        svc = _service(handler)

        with pytest.raises(OpenRouterUnavailableError):
            svc.embed(["x"])

        assert calls["count"] == 2

    def test_dimension_mismatch_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": [{"embedding": [0.0] * 3}]},
            )

        svc = _service(handler)

        with pytest.raises(OpenRouterUnavailableError):
            svc.embed(["x"])

    def test_empty_input_returns_empty_without_request(self):
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(
                200,
                json={"data": [{"embedding": [0.0] * 2048}]},
            )

        svc = _service(handler)

        assert svc.embed([]) == []
        assert calls["count"] == 0

    def test_missing_api_key_raises_at_construction(self):
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))

        with pytest.raises(OpenRouterUnavailableError):
            OpenRouterEmbeddingService(api_key="", client=client)


class TestEmbeddingServiceOpenRouter:
    """Facade routes the default provider to OpenRouter with no fallback."""

    def test_default_provider_selects_openrouter(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "openrouter")
        monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
        monkeypatch.setattr(
            settings,
            "openrouter_embedding_model",
            "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        )
        monkeypatch.setattr(settings, "embedding_dimension", 2048)

        service = EmbeddingService()

        assert service.last_provider == EMBEDDER_OPENROUTER
        assert service.dimension == 2048
        assert service._openrouter is not None

    def test_embed_routes_to_openrouter(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "openrouter")
        monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
        monkeypatch.setattr(settings, "embedding_dimension", 2048)

        service = EmbeddingService()
        fake = SimpleNamespace(
            provider=EMBEDDER_OPENROUTER,
            model_name="nvidia/llama-nemotron-embed-vl-1b-v2:free",
            dimension=2048,
            embed=lambda texts: _make_vectors(len(texts), 2048),
        )
        service._openrouter = fake

        vectors = service.embed(["a", "b"])

        assert len(vectors) == 2
        assert all(len(v) == 2048 for v in vectors)
        assert service.last_provider == EMBEDDER_OPENROUTER
        assert service.last_model == fake.model_name

    def test_openrouter_error_propagates_without_fallback(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "openrouter")
        monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
        monkeypatch.setattr(settings, "embedding_dimension", 2048)

        service = EmbeddingService()

        def failing_embed(texts):
            raise OpenRouterUnavailableError("free tier down")

        service._openrouter = SimpleNamespace(embed=failing_embed)

        with pytest.raises(OpenRouterUnavailableError):
            service.embed(["x"])