from google import genai
from google.genai import types

from app.core.config import settings


class EmbeddingService:

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        output_dimension: int | None = None,
        task_type: str | None = None,
        batch_size: int | None = None,
    ):

        self.model_name = (
            model_name
            or settings.gemini_embedding_model
        )

        self.api_key = (
            api_key
            or settings.gemini_api_key
        )

        self.output_dimension = (
            output_dimension
            or settings.gemini_embedding_output_dimension
        )

        self.task_type = (
            task_type
            or settings.gemini_embedding_task_type
        )

        self.batch_size = (
            batch_size
            or settings.gemini_embedding_batch_size
        )

        if not self.api_key:

            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(
            api_key=self.api_key
        )

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:

            return []

        vectors: list[list[float]] = []

        for start in range(
            0,
            len(texts),
            self.batch_size,
        ):

            batch = texts[
                start : start + self.batch_size
            ]

            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=self.task_type,
                    output_dimensionality=self.output_dimension,
                ),
            )

            for embedding in response.embeddings:

                vectors.append(
                    list(
                        float(value)
                        for value in embedding.values
                    )
                )

        return vectors

    @property
    def dimension(self) -> int:

        return self.output_dimension