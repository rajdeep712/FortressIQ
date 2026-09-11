"""One-time model warm-up for retrieval.

Downloads the fastembed sparse (BM25) model and the FlashRank cross-encoder so
retrieval can run later with ``hf_hub_offline=True``.

Usage (from the ``backend/`` directory):

    python -m app.warm_models

Run this once with network access. After the models are cached on disk, run
the service with hf_hub_offline=True as normal -- both providers load the
cached weights without hitting the network.

NOTE ON OFFLINE FLAGS
    huggingface_hub snapshots HF_HUB_OFFLINE into a module constant the first
    time it (or sentence_transformers / fastembed) is imported. This module
    therefore sets the env var BEFORE importing anything app-side, and it must
    live OUTSIDE the ``app.retrieval`` package: importing ``app.retrieval``
    executes ``app/retrieval/__init__.py`` which imports the embedding stack
    (and freezes the offline constant) before any code here could run.

    ``python -m app.warm_models`` is safe because ``app/__init__.py`` does not
    exist (bare namespace package), so no embedding code is imported just by
    loading this module.

Requires the optional deps to be installed:

    pip install fastembed flashrank onnxruntime
"""

from __future__ import annotations

import logging
import os
import sys

# Temporarily allow downloads regardless of the configured offline flag.
# Must happen before importing app.ingestion.embedding (and thus
# sentence_transformers / fastembed / huggingface_hub).
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from app.core.config import settings  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    done = 0
    try:
        from app.ingestion.embedding import SentenceTransformerEmbeddingService

        dense = SentenceTransformerEmbeddingService()
        logger.info(
            "Warming dense embedding model: %s (%d)",
            dense.model_name,
            dense.dimension,
        )
        done += 1
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to warm dense model %s: %s",
            settings.embedding_model,
            exc,
        )

    try:
        from app.ingestion.embedding import SparseEmbeddingService

        sparse = SparseEmbeddingService(settings.sparse_model_name)
        logger.info("Warming sparse model: %s", settings.sparse_model_name)
        sparse.warm()
        logger.info("Sparse model loaded: %s", settings.sparse_model_name)
        done += 1
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to warm sparse model %s: %s",
            settings.sparse_model_name,
            exc,
        )

    try:
        from app.retrieval.reranker import FlashRankReranker

        reranker = FlashRankReranker(settings.rerank_model_name)
        logger.info("Warming rerank model: %s", settings.rerank_model_name)
        reranker.warm()
        logger.info("Rerank model loaded: %s", settings.rerank_model_name)
        done += 1
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to warm rerank model %s: %s",
            settings.rerank_model_name,
            exc,
        )

    if done == 0:
        logger.error(
            "No retrieval models loaded; check installed deps "
            "(sentence-transformers, fastembed, flashrank, onnxruntime) "
            "and try again."
        )
        return 1

    logger.info("Warmed %d/%d retrieval models.", done, 3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
