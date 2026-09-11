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
            getattr(
                self.model,
                "get_embedding_dimension",
                self.model.get_sentence_embedding_dimension,
            )()
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
    """Facade over the dense embedding engine.

    Uses SentenceTransformerEmbeddingService exclusively. When no
    ``service`` is injected the constructor loads the model specified
    by ``settings.embedding_model`` (default: BAAI/bge-base-en-v1.5).

    The ``service`` injection point is preserved so tests that pass
    ``service=FakeEmbedder`` keep working.
    """

    def __init__(self, service=None):
        self._service: SentenceTransformerEmbeddingService | None = None
        self.last_provider: str | None = None
        self.last_model: str | None = None

        if service is not None:
            self._service = service
            if self._service.dimension != settings.embedding_dimension:
                raise ValueError(
                    f"Embedding dimension ({self._service.dimension}) "
                    "does not match EMBEDDING_DIMENSION "
                    f"({settings.embedding_dimension})."
                )
            self.last_provider = service.provider
            self.last_model = (
                getattr(service, "model_name", None)
                or getattr(service, "model", None)
            )
            return

        self._ensure_st()

    # --- lazy ST init (load once, cache forever) ---------------------------

    def _ensure_st(self) -> SentenceTransformerEmbeddingService:
        if self._service is not None:
            return self._service
        # Construct lazily (dimension is validated against
        # EMBEDDING_DIMENSION inside the service constructor) and cache for
        # the lifetime of this facade.
        self._service = SentenceTransformerEmbeddingService()
        self.last_provider = EMBEDDER_SENTENCE_TRANSFORMER
        self.last_model = getattr(self._service, "model_name", None)
        return self._service

    # --- public API --------------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._ensure_st().dimension

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            self.last_provider = None
            self.last_model = None
            return []

        vectors = self._ensure_st().embed(texts)
        self.last_provider = EMBEDDER_SENTENCE_TRANSFORMER
        self.last_model = getattr(self._service, "model_name", None)
        return vectors


# fastembed's SparseTextEmbedding yields objects with numpy `.indices` /
# `.values`; we normalize those into plain JSON-friendly dicts.
class SparseEmbedding:
    """A sparse vector as Qdrant expects it."""

    __slots__ = ("indices", "values")

    def __init__(self, indices, values):
        self.indices = [int(i) for i in indices]
        self.values = [float(v) for v in values]

    def to_qdrant(self) -> dict:
        return {"indices": self.indices, "values": self.values}


class SparseEmbeddingService:
    """BM25 / SPLADE sparse embeddings via fastembed's SparseTextEmbedding.

    Loads the model lazily (first use), so importing this module never pulls
    in fastembed or downloads a model. If the model cannot be loaded (missing
    fastembed, or hf_hub_offline=True and the model is not cached), the
    service reports itself unavailable and returns empty embeddings rather
    than crashing ingestion/retrieval.
    """

    def __init__(self, model_name: str | None = None, model=None, lazy: bool = True):
        self.model_name = (
            settings.sparse_model_name if model_name is None else model_name
        )
        self._model = model
        self._load_error: Exception | None = None
        if not lazy and model is None:
            self.warm()

    @property
    def model_available(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            self.warm()
        except Exception as exc:  # noqa: BLE001
            self._load_error = exc
            logger.warning(
                "Sparse embedding model unavailable (%s): %s",
                self.model_name,
                exc,
            )
            return False
        return self._model is not None

    def warm(self) -> None:
        """Load (or re-load) the underlying fastembed model; raises on failure."""
        if self._model is not None:
            return
        try:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(self.model_name)
        except Exception as exc:  # noqa: BLE001
            self._load_error = exc
            raise

    def _encode(self, texts: list[str]) -> list[SparseEmbedding]:
        if not self.model_available:
            return []
        out = []
        for sparse in self._model.embed(texts):
            out.append(
                SparseEmbedding(sparse.indices, sparse.values)
            )
        return out

    def embed(self, texts: list[str]) -> list[dict]:
        """Return sparse vectors as Qdrant-compatible dicts."""
        return [
            vec.to_qdrant()
            for vec in self._encode(texts)
        ]

    def query_embed(self, text: str) -> dict:
        """Return a single sparse Qdrant dict for a query string."""
        if not self.model_available or not text.strip():
            return {"indices": [], "values": []}
        vecs = self._encode([text])
        return vecs[0].to_qdrant() if vecs else {"indices": [], "values": []}
