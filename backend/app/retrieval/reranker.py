"""Cross-encoder reranker for the fused document candidates.

Uses FlashRank (this dependency's ``flashrank.Ranker`` / ``RerankRequest``
API), which runs a small ONNX cross-encoder and downloads its model directly
from Hugging Face (independent of huggingface_hub's offline flag). It is
rolled with graceful fallback: if the model cannot be loaded, the reranker
reports itself unavailable and the caller falls back to the fusion order
rather than crashing.

Design keeps a single ``Reranker`` protocol so tests can inject a fake that
does not depend on FlashRank / ONNX.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence

from app.core.config import settings

logger = logging.getLogger(__name__)


class Passage(Protocol):
    """A minimal passage contract: enough to rerank a fused candidate."""

    @property
    def text(self) -> str: ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        passages: Sequence[Any],
        top_k: int,
    ) -> list[int]: ...

    @property
    def available(self) -> bool: ...


class FlashRankReranker:
    """FlashRank cross-encoder reranker with lazy model load + fallback.

    ``rerank`` returns a list of the *indices* (into ``passages``) of the top
    ``top_k`` passages, ranked best-first. If the model is unavailable, it
    returns ``[]`` so callers know to fall back to the original (fusion) order.
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
        model=None,
    ):
        self.model_name = (
            settings.rerank_model_name
            if model_name is None
            else model_name
        )
        self.cache_dir = (
            settings.rerank_cache_dir
            if cache_dir is None
            else cache_dir
        )
        self._model = model
        self._load_error: Exception | None = None

    @property
    def available(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            self.warm()
        except Exception as exc:  # noqa: BLE001
            self._load_error = exc
            logger.warning(
                "FlashRank reranker model unavailable (%s): %s",
                self.model_name,
                exc,
            )
            return False
        return self._model is not None

    def warm(self) -> None:
        """Load (or re-load) the FlashRank model; raises on failure.

        The constructor downloads the ONNX weights into ``cache_dir`` if not
        already present, so warming here both fetches and loads the model.
        """
        if self._model is not None:
            return
        try:
            from flashrank import Ranker

            self._model = Ranker(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = exc
            raise

    def rerank(
        self,
        query: str,
        passages: Sequence[Any],
        top_k: int,
    ) -> list[int]:
        if not passages or top_k <= 0:
            return []
        if not self.available:
            return []

        from flashrank import RerankRequest

        text_of = _text_of
        request = RerankRequest(
            query=query,
            passages=[
                {"id": i, "text": text}
                for i, text in enumerate(text_of(passages))
            ],
        )

        try:
            ranked = self._model.rerank(request)
        except Exception as exc:  # noqa: BLE001
            self._load_error = exc
            logger.warning(
                "FlashRank rerank failed; falling back to fusion order: %s",
                exc,
            )
            return []

        ordered: list[int] = []
        for item in ranked:
            idx = int(item.get("id", -1))
            if idx < len(passages) and idx not in ordered:
                ordered.append(idx)
        return ordered[:top_k]


def _text_of(passages: Sequence[Any]) -> list[str]:
    out = []
    for p in passages:
        if isinstance(p, str):
            out.append(p)
        else:
            out.append(getattr(p, "text", ""))
    return out
