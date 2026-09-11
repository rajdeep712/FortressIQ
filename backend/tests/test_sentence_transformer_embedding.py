import numpy as np
import pytest

from app.core.config import settings
from app.ingestion.embedding import (
    EMBEDDER_SENTENCE_TRANSFORMER,
    EmbeddingService,
    SentenceTransformerEmbeddingService,
)


class FakeModel:

    def __init__(self, dimension=None, failing=False):
        self.dimension = (
            dimension or settings.embedding_dimension
        )
        self.failing = failing
        self.encode_calls = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(
        self,
        texts,
        batch_size=None,
        normalize_embeddings=None,
        convert_to_numpy=None,
        show_progress_bar=None,
    ):
        self.encode_calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        if self.failing:
            raise RuntimeError("model exploded")
        return np.array(
            [
                [0.1] * self.dimension
                for _ in texts
            ],
            dtype=np.float32,
        )


def make_service(model=None, **kwargs):
    return SentenceTransformerEmbeddingService(
        model_name="test-model",
        model=model or FakeModel(),
        **kwargs,
    )


class TestSentenceTransformerEmbeddingService:

    def test_embed_returns_float_vectors_of_expected_dim(self):
        service = make_service()

        vectors = service.embed(["a", "b", "c"])

        assert len(vectors) == 3
        assert all(
            len(v) == settings.embedding_dimension
            for v in vectors
        )
        assert all(
            isinstance(x, float)
            for v in vectors
            for x in v
        )

    def test_encode_receives_texts_and_batch_settings(self):
        model = FakeModel()
        service = SentenceTransformerEmbeddingService(
            model_name="test-model",
            batch_size=4,
            model=model,
        )

        service.embed(["one", "two"])

        call = model.encode_calls[-1]
        assert call["texts"] == ["one", "two"]
        assert call["batch_size"] == 4
        assert call["normalize_embeddings"] is True
        assert call["convert_to_numpy"] is True

    def test_default_batch_size_from_settings(self):
        model = FakeModel()
        service = SentenceTransformerEmbeddingService(
            model_name="test-model",
            model=model,
        )

        service.embed(["x"])

        assert model.encode_calls[0]["batch_size"] == (
            settings.embedding_batch_size
        )

    def test_empty_input_returns_empty(self):
        model = FakeModel()
        service = SentenceTransformerEmbeddingService(
            model_name="test-model",
            model=model,
        )

        assert service.embed([]) == []
        assert model.encode_calls == []

    def test_provider_tag(self):
        assert make_service().provider == (
            EMBEDDER_SENTENCE_TRANSFORMER
        )

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            SentenceTransformerEmbeddingService(
                model_name="test-model",
                model=FakeModel(dimension=42),
            )


class TestEmbeddingFacade:

    def test_embed_records_provider(self):
        service = EmbeddingService(
            service=make_service()
        )

        vectors = service.embed(["a", "b"])

        assert service.last_provider == (
            EMBEDDER_SENTENCE_TRANSFORMER
        )
        assert service.last_model == "test-model"
        assert len(vectors) == 2
        assert all(
            len(v) == settings.embedding_dimension
            for v in vectors
        )

    def test_embed_records_model(self):
        service = EmbeddingService(
            service=make_service()
        )

        service.embed(["a"])

        assert service.last_model == "test-model"

    def test_injected_service_model_discovered(self):
        service = make_service()
        EmbeddingService(service=service)

        assert service.model_name == "test-model"

    def test_empty_input_leaves_provider_unset(self):
        service = EmbeddingService(
            service=make_service()
        )

        assert service.embed([]) == []
        assert service.last_provider is None
        assert service.last_model is None

    def test_provider_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            EmbeddingService(
                service=SentenceTransformerEmbeddingService(
                    model_name="test-model",
                    model=FakeModel(dimension=42),
                )
            )