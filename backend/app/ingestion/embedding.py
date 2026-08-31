import logging
import os

from app.core.config import settings

logger = logging.getLogger(__name__)

# huggingface_hub snapshots HF_* settings into module constants at import
# time, so the offline/telemetry flags must be in os.environ BEFORE
# sentence_transformers (and its huggingface_hub dependency) is imported
# below. Otherwise the worker pings the hub at every startup.
if settings.hf_hub_offline:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from sentence_transformers import SentenceTransformer  # noqa: E402

EMBEDDER_SENTENCE_TRANSFORMER = "sentence-transformer"


class SentenceTransformerEmbeddingService:

    provider = EMBEDDER_SENTENCE_TRANSFORMER

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
        model=None,
    ):
        self.model_name = (
            settings.embedding_model
            if model_name is None
            else model_name
        )
        self.batch_size = (
            settings.embedding_batch_size
            if batch_size is None
            else batch_size
        )
        self.normalize_embeddings = normalize_embeddings

        self.model = (
            model
            if model is not None
            else SentenceTransformer(self.model_name)
        )

        self.output_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        if self.output_dimension != settings.embedding_dimension:
            raise ValueError(
                f"Model {self.model_name} dimension "
                f"({self.output_dimension}) does not match "
                "EMBEDDING_DIMENSION "
                f"({settings.embedding_dimension})."
            )

    @property
    def dimension(self) -> int:
        return self.output_dimension

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return [
            list(float(value) for value in vector)
            for vector in embeddings
        ]


class EmbeddingService:

    def __init__(self, service=None):
        self._service = (
            service or SentenceTransformerEmbeddingService()
        )

        if self._service.dimension != settings.embedding_dimension:
            raise ValueError(
                f"Embedding dimension ({self._service.dimension}) "
                "does not match EMBEDDING_DIMENSION "
                f"({settings.embedding_dimension})."
            )

        self.last_provider: str | None = None

    @property
    def dimension(self) -> int:
        return self._service.dimension

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            self.last_provider = None
            return []

        vectors = self._service.embed(texts)

        self.last_provider = self._service.provider

        return vectors